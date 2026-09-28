"""AEL MARKET-IMPLIED WHISPER™ v2.

Goal: always produce a numerical next-quarter EPS/revenue expectation when a
reasonable public-data basis exists, while clearly separating observed data
from AEL inference. Optional licensed provider keys can improve the base
estimate (Earnings Whispers / FMP / Alpha Vantage / Finnhub).

This module never blocks the main AEL data chain. Every external request is
short-timeout, cached, and fail-open.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import math, os, threading
import requests
import pandas as pd
import yfinance as yf

_CACHE = {}
_LOCK = threading.Lock()
TTL = 900
TIMEOUT = 6


def _finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_json(url, *, headers=None, params=None, timeout=TIMEOUT):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "AEL/2.5.28"}, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)[:180]


def _env(name):
    return os.getenv(name, "").strip()


def _period_rows_from_yahoo(symbol):
    """Return normalized quarterly revenue/net-income/EPS history."""
    out = {"available": False, "rows": [], "source": "Yahoo quarterly financials", "error": None}
    try:
        t = yf.Ticker(symbol)
        df = getattr(t, "quarterly_income_stmt", None)
        if df is None or df.empty:
            df = t.quarterly_financials
        if df is None or df.empty:
            return out
        dates = list(df.columns)
        for dt in dates:
            revenue = None; net_income = None; eps = None
            for k in ("Total Revenue", "Operating Revenue"):
                if k in df.index:
                    revenue = _finite(df.loc[k, dt]);
                    if revenue is not None: break
            for k in ("Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"):
                if k in df.index:
                    net_income = _finite(df.loc[k, dt]);
                    if net_income is not None: break
            for k in ("Diluted EPS", "Basic EPS", "Diluted EPS from Continuing Operations"):
                if k in df.index:
                    eps = _finite(df.loc[k, dt]);
                    if eps is not None: break
            if revenue is not None or net_income is not None or eps is not None:
                ts = pd.Timestamp(dt)
                out["rows"].append({"date": ts, "revenue": revenue, "net_income": net_income, "eps": eps})
        out["rows"].sort(key=lambda x: x["date"])
        out["available"] = bool(out["rows"])
    except Exception as e:
        out["error"] = str(e)[:180]
    return out


def _quarterly_fallback(symbol):
    """Build a forward base from same-quarter history + recent trend."""
    hist = _period_rows_from_yahoo(symbol)
    rows = hist.get("rows") or []
    if not rows:
        return {"available": False, "error": hist.get("error"), "source": hist.get("source")}
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    past = [r for r in rows if r["date"] <= now]
    if not past:
        past = rows
    last = past[-1]

    # Target the next fiscal quarter rather than simply reusing the latest
    # quarter. For a June quarter, for example, the base should resemble the
    # historical September quarter. This materially improves seasonal names.
    target_next = last["date"] + pd.DateOffset(months=3)
    target_yoy = target_next - pd.DateOffset(years=1)

    def next_quarter_prior_year(field):
        candidates=[]
        for r in past:
            if r.get(field) is None: continue
            dist=abs((r["date"]-target_yoy).days)
            if dist <= 75: candidates.append((dist,r[field]))
        return min(candidates,key=lambda x:x[0])[1] if candidates else None

    def last_year_same_quarter(field):
        candidates=[]
        target=last["date"]-pd.DateOffset(years=1)
        for r in past:
            if r.get(field) is None: continue
            dist=abs((r["date"]-target).days)
            if dist <= 75: candidates.append((dist,r[field]))
        return min(candidates,key=lambda x:x[0])[1] if candidates else None

    def yoy_growth(field):
        cur=last.get(field)
        ly=last_year_same_quarter(field)
        if cur is not None and ly not in (None,0): return (cur/ly-1)*100
        return None

    def recent_growth(field):
        vals = [r[field] for r in past[-5:] if r.get(field) is not None]
        if len(vals) >= 3:
            gs = [(vals[i] / vals[i-1] - 1) * 100 for i in range(1, len(vals)) if vals[i-1] not in (None, 0)]
            if gs: return float(pd.Series(gs).median())
        return None

    rev_yoy = yoy_growth("revenue")
    eps_yoy = yoy_growth("eps")
    rev_recent = recent_growth("revenue")
    eps_recent = recent_growth("eps")

    # Blend seasonal YoY with recent trend; clamp to prevent absurd extrapolation.
    rev_growth = _clamp((rev_yoy if rev_yoy is not None else rev_recent if rev_recent is not None else 0) * 0.70 +
                        (rev_recent if rev_recent is not None else rev_yoy if rev_yoy is not None else 0) * 0.30, -35, 60)
    eps_growth = _clamp((eps_yoy if eps_yoy is not None else eps_recent if eps_recent is not None else 0) * 0.70 +
                        (eps_recent if eps_recent is not None else eps_yoy if eps_yoy is not None else 0) * 0.30, -60, 100)

    prior_next_rev = next_quarter_prior_year("revenue")
    prior_next_eps = next_quarter_prior_year("eps")
    base_rev = prior_next_rev if prior_next_rev is not None else last.get("revenue")
    base_eps = prior_next_eps if prior_next_eps is not None else last.get("eps")
    if base_rev is not None:
        base_rev = base_rev * (1 + rev_growth / 100)
    if base_eps is not None:
        base_eps = base_eps * (1 + eps_growth / 100)

    return {
        "available": base_rev is not None or base_eps is not None,
        "revenue": base_rev,
        "eps": base_eps,
        "revenue_growth_pct": rev_growth,
        "eps_growth_pct": eps_growth,
        "last_quarter": last.get("date").date().isoformat() if last.get("date") is not None else None,
        "source": "AEL historical-quarter model",
        "history_count": len(past),
    }


def _consensus_earnings_whispers(symbol):
    key = _env("EW_API_KEY")
    out = {"available": False, "source": "Earnings Whispers Data API", "error": None}
    if not key: return out
    js, err = _safe_json("https://www.earningswhispers.com/api/v1/earnings",
                         headers={"X-EW-Key": key, "User-Agent": "AEL/2.5.28"},
                         params={"tickers": symbol, "detail": "expanded", "limit": 10})
    if err: out["error"] = err; return out
    rows = (js or {}).get("data") or []
    row = next((x for x in rows if str(x.get("Ticker", "")).upper() == symbol.upper()), None)
    if not row: return out
    out.update({"eps": _finite(row.get("EarningsEst")), "revenue": _finite(row.get("RevenueEst")),
                "whisper_eps": _finite(row.get("Whisper")), "revision": _finite(row.get("Revision")),
                "implied_move": _finite(row.get("ImpliedMove")), "earnings_date": row.get("EPSDate"),
                "grade": row.get("Grade"), "score": row.get("Score"), "sentiment": row.get("Sentiment")})
    out["available"] = out["eps"] is not None or out["revenue"] is not None
    return out


def _consensus_fmp(symbol):
    key = _env("FMP_API_KEY")
    out = {"available": False, "source": "Financial Modeling Prep", "error": None}
    if not key: return out
    # The earnings endpoint is a useful low-friction consensus fallback.
    js, err = _safe_json("https://financialmodelingprep.com/stable/earnings",
                         params={"symbol": symbol, "apikey": key})
    if err: out["error"] = err; return out
    rows = js if isinstance(js, list) else (js or {}).get("data") or []
    today = pd.Timestamp.now(tz="UTC").date()
    rows = sorted(rows, key=lambda x: str(x.get("date") or ""))
    row = None
    for x in rows:
        try:
            if pd.Timestamp(x.get("date")).date() >= today and (x.get("epsEstimated") is not None or x.get("revenueEstimated") is not None):
                row = x; break
        except Exception: pass
    if row is None and rows:
        # Some FMP plans return only the next record without a future date flag.
        row = rows[-1]
    if not row: return out
    out.update({"eps": _finite(row.get("epsEstimated")), "revenue": _finite(row.get("revenueEstimated")),
                "earnings_date": row.get("date")})
    out["available"] = out["eps"] is not None or out["revenue"] is not None
    return out


def _consensus_alpha(symbol):
    key = _env("ALPHAVANTAGE_API_KEY")
    out = {"available": False, "source": "Alpha Vantage Earnings Estimates", "error": None}
    if not key: return out
    js, err = _safe_json("https://www.alphavantage.co/query", params={"function": "EARNINGS_ESTIMATES", "symbol": symbol, "apikey": key})
    if err: out["error"] = err; return out
    rows = (js or {}).get("estimates") or (js or {}).get("data") or []
    row = None
    for x in rows:
        # Alpha Vantage uses fiscalDateEnding.
        try:
            dt = pd.Timestamp(x.get("fiscalDateEnding"))
            if dt.tzinfo is None: dt = dt.tz_localize("UTC")
            if dt >= pd.Timestamp.now(tz="UTC") and (x.get("epsAvg") is not None or x.get("revenueAvg") is not None): row = x; break
        except Exception: pass
    if row is None and rows: row = rows[0]
    if not row: return out
    out.update({"eps": _finite(row.get("epsAvg")), "revenue": _finite(row.get("revenueAvg")),
                "eps_low": _finite(row.get("epsLow")), "eps_high": _finite(row.get("epsHigh")),
                "revenue_low": _finite(row.get("revenueLow")), "revenue_high": _finite(row.get("revenueHigh")),
                "analysts_eps": _finite(row.get("numberAnalystsEps")), "analysts_revenue": _finite(row.get("numberAnalystsRevenue"))})
    out["available"] = out["eps"] is not None or out["revenue"] is not None
    return out


def _consensus_finnhub_calendar(symbol):
    key = _env("FINNHUB_API_KEY")
    out = {"available": False, "source": "Finnhub earnings calendar", "error": None}
    if not key: return out
    now = datetime.now(timezone.utc).date()
    js, err = _safe_json("https://finnhub.io/api/v1/calendar/earnings", params={"from": now.isoformat(), "to": (now + timedelta(days=120)).isoformat(), "symbol": symbol, "token": key})
    if err: out["error"] = err; return out
    rows = (js or {}).get("earningsCalendar") or []
    row = next((x for x in rows if str(x.get("symbol", "")).upper() == symbol.upper()), None)
    if not row: return out
    out.update({"eps": _finite(row.get("epsEstimate")), "revenue": _finite(row.get("revenueEstimate")), "earnings_date": row.get("date")})
    out["available"] = out["eps"] is not None or out["revenue"] is not None
    return out


def _yahoo_consensus(symbol):
    out = {"available": False, "source": "Yahoo Finance earningsTrend", "error": None,
           "eps": None, "revenue": None, "revision_7d_pct": None, "revision_30d_pct": None, "revision_90d_pct": None, "earnings_date": None}
    try:
        t = yf.Ticker(symbol)
        trend = getattr(t, "earnings_trend", None)
        if trend is not None and hasattr(trend, "iterrows") and not trend.empty:
            df = trend.copy().reset_index()
            for _, r in df.iterrows():
                period = str(r.get("period") or "")
                if period not in ("0q", "+0q"): continue
                def g(group, field):
                    for k, v in r.to_dict().items():
                        if isinstance(k, tuple) and len(k) >= 2 and str(k[0]).lower() == group.lower() and str(k[1]).lower() == field.lower(): return _finite(v)
                    x = r.get(group)
                    return _finite(x.get(field)) if isinstance(x, dict) else None
                eps = g("earningsEstimate", "avg") or g("earningsEstimate", "current")
                rev = g("revenueEstimate", "avg") or g("revenueEstimate", "current")
                cur = g("epsTrend", "current")
                e7 = g("epsTrend", "7daysAgo"); e30 = g("epsTrend", "30daysAgo"); e90 = g("epsTrend", "90daysAgo")
                out.update({"eps": eps, "revenue": rev,
                            "revision_7d_pct": (cur/e7-1)*100 if cur is not None and e7 not in (None,0) else None,
                            "revision_30d_pct": (cur/e30-1)*100 if cur is not None and e30 not in (None,0) else None,
                            "revision_90d_pct": (cur/e90-1)*100 if cur is not None and e90 not in (None,0) else None})
                break
        dates = t.get_earnings_dates(limit=12)
        if dates is not None and not dates.empty:
            now = pd.Timestamp.now(tz="UTC")
            future=[]
            for idx in dates.index:
                try:
                    dt=pd.Timestamp(idx); dt=dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
                    if dt>now: future.append(dt)
                except Exception: pass
            if future: out["earnings_date"] = min(future).isoformat()
        out["available"] = out["eps"] is not None or out["revenue"] is not None
    except Exception as e: out["error"] = str(e)[:180]
    return out


def _consensus_bundle(symbol):
    funcs = [_consensus_earnings_whispers, _consensus_fmp, _consensus_alpha, _consensus_finnhub_calendar, _yahoo_consensus]
    results=[]
    with ThreadPoolExecutor(max_workers=len(funcs)) as ex:
        fs={ex.submit(fn, symbol):fn.__name__ for fn in funcs}
        for f in as_completed(fs):
            try: results.append(f.result())
            except Exception as e: results.append({"available":False,"error":str(e)[:180],"source":fs[f]})
    # Preserve preferred source order, but use whichever provider has the
    # individual field. This avoids losing revenue just because an EPS source failed.
    rank={"Earnings Whispers Data API":0,"Financial Modeling Prep":1,"Alpha Vantage Earnings Estimates":2,"Finnhub earnings calendar":3,"Yahoo Finance earningsTrend":4}
    results.sort(key=lambda x:rank.get(x.get("source"),99))
    eps_src=next((x for x in results if x.get("eps") is not None), None)
    rev_src=next((x for x in results if x.get("revenue") is not None), None)
    chosen=next((x for x in results if x.get("available")), None) or {}
    return {"results":results, "eps":eps_src, "revenue":rev_src,
            "earnings_date": next((x.get("earnings_date") for x in results if x.get("earnings_date")), None),
            "chosen":chosen}


def _historical_surprise(symbol):
    out={"available":False,"eps_mean_pct":None,"eps_median_pct":None,"eps_recent_weighted_pct":None,"count":0,"error":None,"source":"Yahoo historical earnings"}
    try:
        t=yf.Ticker(symbol); d=t.get_earnings_dates(limit=12)
        if d is None or d.empty: return out
        vals=[]
        for _,r in d.iterrows():
            v=None
            for k in ("Surprise(%)","surprisePercent","Surprise"):
                if k in r.index: v=_finite(r.get(k));
                if v is not None: break
            if v is not None and abs(v)<100: vals.append(v)
        if not vals: return out
        out["count"]=len(vals); out["eps_mean_pct"]=sum(vals)/len(vals); out["eps_median_pct"]=float(pd.Series(vals).median())
        weights=list(range(len(vals),0,-1)); out["eps_recent_weighted_pct"]=sum(v*w for v,w in zip(vals,weights))/sum(weights); out["available"]=True
    except Exception as e: out["error"]=str(e)[:180]
    return out


def _market_signal(symbol, earnings_date=None):
    out={"available":False,"momentum_5d_pct":None,"volume_ratio":None,"event_implied_move_pct":None,"iv_skew_pct":None,"error":None,"source":"Yahoo price + listed options"}
    try:
        t=yf.Ticker(symbol); h=t.history(period="6mo",interval="1d",auto_adjust=False)
        if h is not None and not h.empty:
            c=pd.to_numeric(h.get("Close"),errors="coerce").dropna(); v=pd.to_numeric(h.get("Volume"),errors="coerce").reindex(c.index).fillna(0)
            if len(c)>=10:
                out["momentum_5d_pct"]=(float(c.iloc[-1])/float(c.iloc[-6])-1)*100
                base=float(v.tail(60).head(max(1,min(40,len(v)-20))).mean()); cur=float(v.tail(20).mean())
                if base>0: out["volume_ratio"]=cur/base
        exps=list(t.options or []); chosen=None
        ed=None
        try: ed=pd.Timestamp(earnings_date) if earnings_date else None
        except Exception: ed=None
        for x in exps:
            try:
                xd=pd.Timestamp(x)
                if ed is None or xd>=ed.tz_localize(None): chosen=x; break
            except Exception: pass
        chosen=chosen or (exps[0] if exps else None)
        if chosen:
            ch=t.option_chain(chosen); calls=ch.calls; puts=ch.puts
            px=None
            try: px=_finite((t.fast_info or {}).get("last_price"))
            except Exception: pass
            if px is None and h is not None and not h.empty: px=float(c.iloc[-1])
            if px and calls is not None and puts is not None and not calls.empty and not puts.empty:
                for df in (calls,puts):
                    for col in ("strike","impliedVolatility","volume","openInterest"): df[col]=pd.to_numeric(df[col],errors="coerce")
                calls["dist"]=(calls.strike-px).abs(); puts["dist"]=(puts.strike-px).abs()
                civ=_finite(calls.sort_values("dist").iloc[0].get("impliedVolatility")); piv=_finite(puts.sort_values("dist").iloc[0].get("impliedVolatility"))
                if civ and piv:
                    try: dte=max(1,(pd.Timestamp(chosen)-pd.Timestamp.now()).days)
                    except Exception: dte=1
                    out["event_implied_move_pct"]=(civ+piv)/2*math.sqrt(dte/365)*100
                    c5=calls.iloc[(calls.strike-px*1.05).abs().argsort()[:1]]; p5=puts.iloc[(puts.strike-px*.95).abs().argsort()[:1]]
                    if len(c5) and len(p5):
                        a=_finite(c5.iloc[0].get("impliedVolatility")); b=_finite(p5.iloc[0].get("impliedVolatility"))
                        if a is not None and b is not None: out["iv_skew_pct"]=(b-a)*100
        out["available"]=any(out[k] is not None for k in ("momentum_5d_pct","volume_ratio","event_implied_move_pct","iv_skew_pct"))
    except Exception as e: out["error"]=str(e)[:180]
    return out


def _whisper_adjustment(hist, consensus, market, kind):
    """AEL's whisper-like uplift over a base estimate.

    This is deliberately not a copy of Earnings Whispers' proprietary method.
    It approximates the observable 'consensus is not the whole expectation'
    effect using historical surprise + estimate drift + price/options signals.
    """
    parts=[]
    h=hist.get("eps_recent_weighted_pct")
    if h is not None: parts.append(("历史财报惊喜", _clamp(h,-12,12), 0.40))
    rev=(consensus or {}).get("revision_30d_pct")
    if rev is None: rev=(consensus or {}).get("revision_7d_pct")
    if rev is not None: parts.append(("预测修正", _clamp(rev,-8,8), 0.20))
    mom=market.get("momentum_5d_pct")
    if mom is not None: parts.append(("财报前价格动量", _clamp(mom/2.5,-4,4), 0.12))
    skew=market.get("iv_skew_pct")
    if skew is not None: parts.append(("期权IV偏斜", _clamp(-skew/3.5,-4,4), 0.13))
    vr=market.get("volume_ratio")
    if vr is not None: parts.append(("财报前成交量", _clamp((vr-1)*2.5,-4,4), 0.08))
    iv=market.get("event_implied_move_pct")
    if iv is not None:
        # High event uncertainty widens the expected beat band, but does not
        # mechanically predict direction. Only a small positive calibration.
        parts.append(("事件隐含波动", _clamp((iv-5)*0.10,-1.5,1.5), 0.07))
    if not parts: return 0.0, [], 0
    wsum=sum(w for _,_,w in parts); raw=sum(v*w for _,v,w in parts)/wsum
    if kind=="revenue": raw*=0.62
    return _clamp(raw,-10,10), [{"factor":n,"contribution_pct":round(v*w/wsum,3)} for n,v,w in parts], len(parts)


def _confidence(evidence_n, market_available, base_source, exact_ew=False):
    score=35
    score += min(25, evidence_n*7)
    if market_available: score += 15
    if base_source and base_source != "AEL historical-quarter model": score += 15
    if exact_ew: score += 10
    return int(_clamp(score, 35, 95))


def analyze_whisper(symbol):
    requested=str(symbol or "").strip().upper()
    if not requested: return {"ok":False,"error":"缺少标的"}
    now=datetime.now(timezone.utc).timestamp()
    with _LOCK:
        c=_CACHE.get(requested)
        if c and now-c[0]<TTL: return c[1]

    # Independent source fan-out: one failing provider cannot block the others.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_cons=ex.submit(_consensus_bundle, requested)
        f_hist=ex.submit(_historical_surprise, requested)
        f_qtr=ex.submit(_quarterly_fallback, requested)
        cons=f_cons.result(); hist=f_hist.result(); qtr=f_qtr.result()

    earnings_date=cons.get("earnings_date")
    with ThreadPoolExecutor(max_workers=1) as ex:
        market=ex.submit(_market_signal, requested, earnings_date).result()

    eps_src=cons.get("eps") or {}; rev_src=cons.get("revenue") or {}
    eps_cons=_finite(eps_src.get("eps")); rev_cons=_finite(rev_src.get("revenue"))
    eps_base=eps_cons if eps_cons is not None else _finite(qtr.get("eps"))
    rev_base=rev_cons if rev_cons is not None else _finite(qtr.get("revenue"))
    eps_base_source=eps_src.get("source") if eps_cons is not None else qtr.get("source")
    rev_base_source=rev_src.get("source") if rev_cons is not None else qtr.get("source")

    eps_adj,eps_parts,eps_n=_whisper_adjustment(hist, eps_src, market, "eps")
    rev_adj,rev_parts,rev_n=_whisper_adjustment(hist, rev_src, market, "revenue")
    eps_implied=eps_base*(1+eps_adj/100) if eps_base is not None else None
    rev_implied=rev_base*(1+rev_adj/100) if rev_base is not None else None

    exact_ew=bool(_env("EW_API_KEY") and any(x.get("source")=="Earnings Whispers Data API" and x.get("available") for x in cons.get("results",[])))
    confidence=max(_confidence(eps_n,market.get("available"),eps_base_source,exact_ew), _confidence(rev_n,market.get("available"),rev_base_source,exact_ew))

    # If an EW key is present, expose the licensed EW number only as a reference;
    # AEL's own market-implied value remains independently calculated.
    ew=next((x for x in cons.get("results",[]) if x.get("source")=="Earnings Whispers Data API" and x.get("available")), {})

    out={
      "ok":True,"symbol":requested,"as_of":datetime.now(timezone.utc).isoformat(),
      "next_earnings_date":earnings_date,
      "model":"AEL Market-Implied Whisper v2",
      "status":"inferred",
      "confidence_pct":confidence,
      "data_mode":"provider-consensus+market-inference" if (eps_cons is not None or rev_cons is not None) else "historical-base+market-inference",
      "revenue":{
        "consensus":rev_cons,"base":rev_base,"implied":rev_implied,"implied_surprise_pct":rev_adj,
        "base_source":rev_base_source,"low":rev_src.get("revenue_low"),"high":rev_src.get("revenue_high"),
        "evidence":rev_parts,"status":"inferred","reason":"AEL 以可验证共识优先；共识缺失时以历史季度季节性建立基准，再叠加财报前市场信号。"
      },
      "eps":{
        "consensus":eps_cons,"base":eps_base,"implied":eps_implied,"implied_surprise_pct":eps_adj,
        "base_source":eps_base_source,"low":eps_src.get("eps_low"),"high":eps_src.get("eps_high"),
        "evidence":eps_parts,"status":"inferred","reason":"AEL 以可验证共识优先；共识缺失时以历史季度季节性建立基准，再叠加财报前市场信号。"
      },
      "market_beat_threshold":{"revenue":rev_implied,"eps":eps_implied},
      "market_signals":market,
      "historical_surprise":hist,
      "historical_base":qtr,
      "provider_status":[{"source":x.get("source"),"available":bool(x.get("available")),"error":x.get("error")} for x in cons.get("results",[])],
      "licensed_reference":{"available":exact_ew,"whisper_eps":ew.get("whisper_eps"),"consensus_eps":ew.get("eps"),"consensus_revenue":ew.get("revenue"),"source":"Earnings Whispers Data API" if exact_ew else None},
      "sources":{"consensus":eps_src.get("source") or rev_src.get("source") or qtr.get("source"),"history":"Yahoo historical earnings","market":"Yahoo price + listed options"},
      "method_note":"AEL v2 采用‘共识基准 → 历史惊喜 → 预测修正 → 财报前价格/成交量 → 事件期权IV/偏斜’的可解释推断。共识接口失败时不再输出空白，而是使用历史季度季节性建立 Base Estimate，再计算 Market-Implied Whisper。该数值是 AEL 的公开数据模型推断，不声称等同于 Earnings Whispers 私有 Whisper。若配置合法 EW_API_KEY，可同时显示官方 Whisper 作为独立校准参考。"
    }
    with _LOCK: _CACHE[requested]=(now,out)
    return out
