"""AEL MARKET-IMPLIED WHISPER™ v2.6

AEL's earnings expectation stack is deliberately split into three layers:
1) SELL-SIDE CONSENSUS (observed / cross-validated)
2) AEL WHISPER ESTIMATE (analyst-like estimate built from revisions, guidance,
   historical surprise and a fundamental nowcast)
3) AEL MARKET-IMPLIED (the result of applying price/volume/options price-in
   pressure to the AEL Whisper estimate).

The model does not claim access to private analyst whispers or a private
buy-side order book. It tries to reproduce the *observable economics* behind
an earnings whisper and then separately estimates what the market price has
already demanded. External providers are optional; every provider is isolated.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import math, os, re, threading
import requests
import pandas as pd
import yfinance as yf

_CACHE = {}
_LOCK = threading.Lock()
TTL = 900
TIMEOUT = 6
UA = "AEL/2.6 (public-data research)"


def _finite(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace("$", "").strip()
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_json(url, *, headers=None, params=None, timeout=TIMEOUT):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": UA}, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)[:180]


def _safe_text(url, *, headers=None, params=None, timeout=TIMEOUT):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": UA}, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, str(e)[:180]


def _env(name):
    return os.getenv(name, "").strip()


def _money_token(s):
    if s is None: return None
    m = re.search(r"\$?([\d,.]+)\s*([KMBT])?", str(s), re.I)
    if not m: return None
    v = _finite(m.group(1))
    if v is None: return None
    mul = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}.get((m.group(2) or "").upper(), 1)
    return v * mul


def _period_rows_from_yahoo(symbol):
    out = {"available": False, "rows": [], "source": "Yahoo quarterly financials", "error": None}
    try:
        t = yf.Ticker(symbol)
        df = getattr(t, "quarterly_income_stmt", None)
        if df is None or df.empty:
            df = t.quarterly_financials
        if df is None or df.empty:
            return out
        for dt in list(df.columns):
            revenue = net_income = eps = None
            for k in ("Total Revenue", "Operating Revenue"):
                if k in df.index:
                    revenue = _finite(df.loc[k, dt])
                    if revenue is not None: break
            for k in ("Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"):
                if k in df.index:
                    net_income = _finite(df.loc[k, dt])
                    if net_income is not None: break
            for k in ("Diluted EPS", "Basic EPS", "Diluted EPS from Continuing Operations"):
                if k in df.index:
                    eps = _finite(df.loc[k, dt])
                    if eps is not None: break
            if revenue is not None or net_income is not None or eps is not None:
                ts = pd.Timestamp(dt)
                if ts.tzinfo is not None: ts = ts.tz_localize(None)
                out["rows"].append({"date": ts, "revenue": revenue, "net_income": net_income, "eps": eps})
        out["rows"].sort(key=lambda x: x["date"])
        out["available"] = bool(out["rows"])
    except Exception as e:
        out["error"] = str(e)[:180]
    return out


def _quarterly_fallback(symbol):
    """Build a forward fundamental nowcast without reusing the wrong fiscal year.

    The old model looked for the same calendar quarter one year back and then
    applied the latest quarter's growth to it. That can be badly wrong for
    companies with fiscal calendars that do not line up with calendar months.
    This version uses the latest actual quarter as the anchor and blends a
    robust sequential trend with an available same-period YoY trend. It is a
    *nowcast*, not a substitute for analyst consensus.
    """
    hist = _period_rows_from_yahoo(symbol)
    rows = hist.get("rows") or []
    if not rows:
        return {"available": False, "error": hist.get("error"), "source": hist.get("source")}
    last = rows[-1]
    past = rows

    def growth_series(field):
        vals = [(r["date"], r.get(field)) for r in past if r.get(field) is not None and r.get(field) > 0]
        gs=[]
        for i in range(1, len(vals)):
            if vals[i-1][1] > 0:
                gs.append((vals[i][0], vals[i][1]/vals[i-1][1]-1))
        return gs

    def yoy_growth(field):
        vals=[(r["date"],r.get(field)) for r in past if r.get(field) is not None and r.get(field)>0]
        if len(vals)<4: return None
        target=last["date"]-pd.DateOffset(years=1)
        candidates=[(abs((d-target).days),v) for d,v in vals[:-1] if abs((d-target).days)<=100]
        if not candidates: return None
        ly=min(candidates,key=lambda z:z[0])[1]
        cur=last.get(field)
        return cur/ly-1 if cur and ly else None

    def robust_seq(field):
        gs=[g for _,g in growth_series(field)][-4:]
        if not gs: return None
        # Median of recent sequential growth, but suppress one-off explosive jumps.
        med=float(pd.Series(gs).median())
        return _clamp(med, -0.15, 0.25)

    rev_seq=robust_seq("revenue"); eps_seq=robust_seq("eps")
    rev_yoy=yoy_growth("revenue"); eps_yoy=yoy_growth("eps")
    def blend(seq,yoy,lo,hi):
        vals=[]; ws=[]
        if seq is not None: vals.append(seq); ws.append(0.55)
        if yoy is not None: vals.append(_clamp(yoy,lo,hi)); ws.append(0.45)
        if not vals: return 0.0
        return _clamp(sum(v*w for v,w in zip(vals,ws))/sum(ws),lo,hi)

    rev_g=blend(rev_seq,rev_yoy,-0.25,0.35)
    eps_g=blend(eps_seq,eps_yoy,-0.35,0.35)
    base_rev=last.get("revenue")*(1+rev_g) if last.get("revenue") is not None else None
    base_eps=last.get("eps")*(1+eps_g) if last.get("eps") is not None else None
    return {
        "available": base_rev is not None or base_eps is not None,
        "revenue": base_rev, "eps": base_eps,
        "revenue_growth_pct": rev_g*100, "eps_growth_pct": eps_g*100,
        "last_quarter": last.get("date").date().isoformat(),
        "source":"AEL fundamental nowcast", "history_count":len(past),
    }


def _target_period_from_history(qtr):
    """Infer the next fiscal period from the latest reported fiscal period.

    This is deliberately a *guard rail*, not an estimate.  If a provider does
    not expose fiscalDateEnding, its estimate may only be admitted when its
    earnings-report date also lines up with the same upcoming report window.
    """
    last = qtr.get("last_quarter") if isinstance(qtr, dict) else None
    if not last:
        return {"target_end": None, "last_actual_end": None}
    try:
        last_dt = pd.Timestamp(last).normalize()
        target = (last_dt + pd.DateOffset(months=3)).normalize()
        return {"target_end": target.date().isoformat(), "last_actual_end": last_dt.date().isoformat()}
    except Exception:
        return {"target_end": None, "last_actual_end": last}


def _date_distance_days(a, b):
    try:
        return abs((pd.Timestamp(a).date() - pd.Timestamp(b).date()).days)
    except Exception:
        return None


def _period_matches(row, target_period, target_earnings_date=None):
    """Hard fiscal-quarter gate for consensus rows.

    1. Prefer an explicit fiscalDateEnding/period_ending.
    2. If absent, require the provider's earnings date to be close to the
       target earnings date (when one is known).
    3. If neither exists, reject the row rather than mixing quarters.
    """
    if not row or not target_period:
        return False
    target_end = target_period.get("target_end")
    explicit = row.get("fiscal_date_ending") or row.get("period_ending")
    if explicit and target_end:
        d = _date_distance_days(explicit, target_end)
        return d is not None and d <= 35
    ed = row.get("earnings_date")
    if ed and target_earnings_date:
        d = _date_distance_days(ed, target_earnings_date)
        return d is not None and d <= 7
    # No period evidence = unsafe.  This is what prevents a stale historical
    # EPS (e.g. SNDK's old 33.28) from contaminating the current quarter.
    return False


def _public_web_estimates(symbol):
    """Keyless public estimate fallback.

    GNG Research exposes current-quarter consensus, revision history and
    revenue consensus in server-rendered HTML. This is intentionally a fallback
    behind licensed APIs, not the primary source. MarketBeat is a second public
    fallback for basic consensus/date data.
    """
    out={"available":False,"source":"Public estimate pages","error":None,
         "eps":None,"revenue":None,"revision_7d_pct":None,"revision_30d_pct":None,"revision_90d_pct":None,
         "analysts_eps":None,"analysts_revenue":None,"earnings_date":None,
         "eps_low":None,"eps_high":None,"revenue_low":None,"revenue_high":None}
    text,err=_safe_text(f"https://www.gngresearch.com/stock/{symbol}/")
    if text:
        # Upcoming estimate block: current period is the first one before the
        # next period. Regexes are deliberately narrow to avoid annual values.
        m=re.search(r"Period ending[^<]{0,100}.*?EPS Consensus\s*\$?([\d,.]+).*?Range\s*\$?([\d,.]+)\s*-\s*\$?([\d,.]+).*?(\d+)\s*analysts.*?Revenue Consensus\s*\$?([\d,.]+)\s*([KMBT])?", text, re.I|re.S)
        if m:
            out["eps"]=_finite(m.group(1)); out["eps_low"]=_finite(m.group(2)); out["eps_high"]=_finite(m.group(3)); out["analysts_eps"]=_finite(m.group(4));
            out["revenue"]=_money_token(m.group(5)+(m.group(6) or "")); out["analysts_revenue"]=out["analysts_eps"]; out["available"]=out["eps"] is not None or out["revenue"] is not None
        # Revision history commonly appears as 90d/60d/30d/7d/current.
        for days in (90,60,30,7):
            mm=re.search(rf"{days}d ago\s*\$?([\d,.]+)",text,re.I)
            if mm and out["eps"]:
                old=_finite(mm.group(1));
                if old not in (None,0): out[f"revision_{days}d_pct"]=(out["eps"]/old-1)*100
        md=re.search(r"Period ending\s*([A-Za-z0-9 ,/-]+?)\s*EPS Consensus",text,re.I)
        if md:
            try: out["period_ending"]=str(pd.Timestamp(md.group(1)).date())
            except Exception: pass
    if not out["available"]:
        text2,err2=_safe_text(f"https://www.marketbeat.com/stocks/NASDAQ/{symbol}/earnings/")
        if text2:
            me=re.search(r"Consensus EPS.*?\$([\d,.]+)",text2,re.I|re.S)
            mr=re.search(r"Consensus Revenue.*?\$([\d,.]+)([KMBT])?",text2,re.I|re.S)
            if me: out["eps"]=_finite(me.group(1))
            if mr: out["revenue"]=_money_token(mr.group(1)+(mr.group(2) or ""))
            if out["eps"] is not None or out["revenue"] is not None: out["available"]=True
    if not out["available"] and (err or err2 if 'err2' in locals() else err): out["error"]=(err2 if 'err2' in locals() and err2 else err)
    return out


def _consensus_earnings_whispers(symbol):
    key=_env("EW_API_KEY"); out={"available":False,"source":"Earnings Whispers Data API","error":None}
    if not key:return out
    js,err=_safe_json("https://www.earningswhispers.com/api/v1/earnings",headers={"X-EW-Key":key,"User-Agent":UA},params={"tickers":symbol,"detail":"expanded","limit":10})
    if err:out["error"]=err;return out
    rows=(js or {}).get("data") or []; row=next((x for x in rows if str(x.get("Ticker","")).upper()==symbol.upper()),None)
    if not row:return out
    out.update({"eps":_finite(row.get("EarningsEst")),"revenue":_finite(row.get("RevenueEst")),"whisper_eps":_finite(row.get("Whisper")),"revision":_finite(row.get("Revision")),"implied_move":_finite(row.get("ImpliedMove")),"earnings_date":row.get("EPSDate"),"period_ending":row.get("FiscalDateEnding") or row.get("FiscalDate") or row.get("PeriodEnding"),"grade":row.get("Grade"),"score":row.get("Score"),"sentiment":row.get("Sentiment")})
    out["available"]=out["eps"] is not None or out["revenue"] is not None
    return out


def _consensus_fmp(symbol):
    key=_env("FMP_API_KEY"); out={"available":False,"source":"Financial Modeling Prep","error":None}
    if not key:return out
    js,err=_safe_json("https://financialmodelingprep.com/stable/earnings",params={"symbol":symbol,"apikey":key})
    if err:out["error"]=err;return out
    rows=js if isinstance(js,list) else (js or {}).get("data") or []
    today=pd.Timestamp.now(tz="UTC").date(); rows=sorted(rows,key=lambda x:str(x.get("date") or ""))
    row=None
    for x in rows:
        try:
            if pd.Timestamp(x.get("date")).date()>=today and (x.get("epsEstimated") is not None or x.get("revenueEstimated") is not None): row=x;break
        except Exception:pass
    if row is None and rows: row=rows[-1]
    if not row:return out
    out.update({"eps":_finite(row.get("epsEstimated")),"revenue":_finite(row.get("revenueEstimated")),"earnings_date":row.get("date"),"period_ending":row.get("fiscalDateEnding") or row.get("fiscalDate")}); out["available"]=out["eps"] is not None or out["revenue"] is not None; return out


def _consensus_alpha(symbol):
    key=_env("ALPHAVANTAGE_API_KEY"); out={"available":False,"source":"Alpha Vantage Earnings Estimates","error":None}
    if not key:return out
    js,err=_safe_json("https://www.alphavantage.co/query",params={"function":"EARNINGS_ESTIMATES","symbol":symbol,"apikey":key})
    if err:out["error"]=err;return out
    rows=(js or {}).get("estimates") or (js or {}).get("data") or []; row=None
    for x in rows:
        try:
            dt=pd.Timestamp(x.get("fiscalDateEnding"))
            if dt.tzinfo is None:dt=dt.tz_localize("UTC")
            if dt>=pd.Timestamp.now(tz="UTC") and (x.get("epsAvg") is not None or x.get("revenueAvg") is not None):row=x;break
        except Exception:pass
    if row is None and rows:row=rows[0]
    if not row:return out
    out.update({"eps":_finite(row.get("epsAvg")),"revenue":_finite(row.get("revenueAvg")),"eps_low":_finite(row.get("epsLow")),"eps_high":_finite(row.get("epsHigh")),"revenue_low":_finite(row.get("revenueLow")),"revenue_high":_finite(row.get("revenueHigh")),"analysts_eps":_finite(row.get("numberAnalystsEps")),"analysts_revenue":_finite(row.get("numberAnalystsRevenue")),"fiscal_date_ending":str(row.get("fiscalDateEnding")) if row.get("fiscalDateEnding") else None}); out["available"]=out["eps"] is not None or out["revenue"] is not None; return out


def _consensus_finnhub_calendar(symbol):
    key=_env("FINNHUB_API_KEY"); out={"available":False,"source":"Finnhub earnings calendar","error":None}
    if not key:return out
    now=datetime.now(timezone.utc).date(); js,err=_safe_json("https://finnhub.io/api/v1/calendar/earnings",params={"from":now.isoformat(),"to":(now+timedelta(days=120)).isoformat(),"symbol":symbol,"token":key})
    if err:out["error"]=err;return out
    rows=(js or {}).get("earningsCalendar") or []; row=next((x for x in rows if str(x.get("symbol","")).upper()==symbol.upper()),None)
    if not row:return out
    out.update({"eps":_finite(row.get("epsEstimate")),"revenue":_finite(row.get("revenueEstimate")),"earnings_date":row.get("date")});out["available"]=out["eps"] is not None or out["revenue"] is not None;return out


def _yahoo_consensus(symbol):
    out={"available":False,"source":"Yahoo Finance earningsTrend","error":None,"eps":None,"revenue":None,"revision_7d_pct":None,"revision_30d_pct":None,"revision_90d_pct":None,"earnings_date":None}
    try:
        t=yf.Ticker(symbol); trend=getattr(t,"earnings_trend",None)
        if trend is not None and hasattr(trend,"iterrows") and not trend.empty:
            df=trend.copy().reset_index()
            for _,r in df.iterrows():
                period=str(r.get("period") or "")
                if period not in ("0q","+0q"):continue
                def g(group,field):
                    for k,v in r.to_dict().items():
                        if isinstance(k,tuple) and len(k)>=2 and str(k[0]).lower()==group.lower() and str(k[1]).lower()==field.lower():return _finite(v)
                    x=r.get(group);return _finite(x.get(field)) if isinstance(x,dict) else None
                eps=g("earningsEstimate","avg") or g("earningsEstimate","current");rev=g("revenueEstimate","avg") or g("revenueEstimate","current");cur=g("epsTrend","current");e7=g("epsTrend","7daysAgo");e30=g("epsTrend","30daysAgo");e90=g("epsTrend","90daysAgo")
                out.update({"eps":eps,"revenue":rev,"revision_7d_pct":(cur/e7-1)*100 if cur is not None and e7 not in (None,0) else None,"revision_30d_pct":(cur/e30-1)*100 if cur is not None and e30 not in (None,0) else None,"revision_90d_pct":(cur/e90-1)*100 if cur is not None and e90 not in (None,0) else None,"period_ending":str(r.get("endDate") or r.get("periodEnd") or r.get("fiscalDateEnding") or "") or None});break
        dates=t.get_earnings_dates(limit=12)
        if dates is not None and not dates.empty:
            now=pd.Timestamp.now(tz="UTC");future=[]
            for idx in dates.index:
                try:
                    dt=pd.Timestamp(idx);dt=dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
                    if dt>now:future.append(dt)
                except Exception:pass
            if future:out["earnings_date"]=min(future).isoformat()
        out["available"]=out["eps"] is not None or out["revenue"] is not None
    except Exception as e:out["error"]=str(e)[:180]
    return out


def _consensus_bundle(symbol, target_period=None):
    funcs=[_consensus_earnings_whispers,_consensus_fmp,_consensus_alpha,_consensus_finnhub_calendar,_yahoo_consensus,_public_web_estimates]
    results=[]
    with ThreadPoolExecutor(max_workers=len(funcs)) as ex:
        fs={ex.submit(fn,symbol):fn.__name__ for fn in funcs}
        for f in as_completed(fs):
            try: results.append(f.result())
            except Exception as e: results.append({"available":False,"error":str(e)[:180],"source":fs[f]})
    rank={"Earnings Whispers Data API":0,"Financial Modeling Prep":1,"Alpha Vantage Earnings Estimates":2,"Finnhub earnings calendar":3,"Yahoo Finance earningsTrend":4,"Public estimate pages":5}
    results.sort(key=lambda x:rank.get(x.get("source"),99))

    # Determine the most credible upcoming report date from rows that survived
    # the fiscal-period gate.  First pass uses explicit fiscal period metadata.
    period_rows=[x for x in results if x.get("available") and _period_matches(x,target_period, None)] if target_period else []
    target_earnings_date=next((x.get("earnings_date") for x in period_rows if x.get("earnings_date")),None)

    # If no explicit fiscal end is exposed, Yahoo/Finnhub/EW report dates can
    # establish the common event date. Use the earliest future date among those
    # providers, then re-run the gate for date-only providers.
    if not target_earnings_date:
        future_dates=[]
        now=pd.Timestamp.now(tz="UTC")
        for x in results:
            ed=x.get("earnings_date")
            if ed:
                try:
                    d=pd.Timestamp(ed)
                    if d.tzinfo is None:d=d.tz_localize("UTC")
                    if d>=now-pd.Timedelta(days=1): future_dates.append(d)
                except Exception: pass
        if future_dates: target_earnings_date=min(future_dates).isoformat()

    valid=[]
    for x in results:
        if not x.get("available"): continue
        if _period_matches(x,target_period,target_earnings_date):
            y=dict(x); y["period_validated"]=True; valid.append(y)
        else:
            y=dict(x); y["period_validated"]=False; y["rejected_reason"]="未通过目标财季锁定"; valid.append(y)

    rank_valid=[x for x in valid if x.get("period_validated")]
    eps_candidates=[x for x in rank_valid if x.get("eps") is not None]
    rev_candidates=[x for x in rank_valid if x.get("revenue") is not None]
    def choose(cands,key):
        if not cands:return None
        preferred=[x for x in cands if x.get("source") in ("Earnings Whispers Data API","Financial Modeling Prep","Alpha Vantage Earnings Estimates","Finnhub earnings calendar","Yahoo Finance earningsTrend")]
        pool=preferred if preferred else cands
        vals=[_finite(x.get(key)) for x in pool if _finite(x.get(key)) is not None]
        if len(vals)>=2:
            med=float(pd.Series(vals).median())
            return min(pool,key=lambda x:abs(float(x[key])-med))
        return pool[0]
    eps_src=choose(eps_candidates,"eps"); rev_src=choose(rev_candidates,"revenue")
    return {"results":valid,"eps":eps_src,"revenue":rev_src,
            "earnings_date":target_earnings_date,"target_period":target_period,
            "period_lock":"strict","chosen":eps_src or rev_src or {}}


def _historical_surprise(symbol):
    out={"available":False,"eps_mean_pct":None,"eps_median_pct":None,"eps_recent_weighted_pct":None,"count":0,"error":None,"source":"Yahoo historical earnings"}
    try:
        t=yf.Ticker(symbol);d=t.get_earnings_dates(limit=12)
        if d is None or d.empty:return out
        vals=[]
        for _,r in d.iterrows():
            v=None
            for k in ("Surprise(%)","surprisePercent","Surprise"):
                if k in r.index:v=_finite(r.get(k))
                if v is not None:break
            if v is not None and abs(v)<100:vals.append(v)
        if not vals:return out
        weights=list(range(len(vals),0,-1));out["count"]=len(vals);out["eps_mean_pct"]=sum(vals)/len(vals);out["eps_median_pct"]=float(pd.Series(vals).median());out["eps_recent_weighted_pct"]=sum(v*w for v,w in zip(vals,weights))/sum(weights);out["available"]=True
    except Exception as e:out["error"]=str(e)[:180]
    return out


def _market_signal(symbol, earnings_date=None):
    out={"available":False,"momentum_5d_pct":None,"momentum_20d_pct":None,"volume_ratio":None,"event_implied_move_pct":None,"iv_skew_pct":None,"error":None,"source":"Yahoo price + listed options"}
    try:
        t=yf.Ticker(symbol);h=t.history(period="6mo",interval="1d",auto_adjust=False)
        if h is not None and not h.empty:
            c=pd.to_numeric(h.get("Close"),errors="coerce").dropna();v=pd.to_numeric(h.get("Volume"),errors="coerce").reindex(c.index).fillna(0)
            if len(c)>=10:
                out["momentum_5d_pct"]=(float(c.iloc[-1])/float(c.iloc[-6])-1)*100
                if len(c)>=21:out["momentum_20d_pct"]=(float(c.iloc[-1])/float(c.iloc[-21])-1)*100
                base=float(v.tail(60).head(max(1,min(40,len(v)-20))).mean());cur=float(v.tail(20).mean());out["volume_ratio"]=cur/base if base>0 else None
        exps=list(t.options or []);chosen=None
        try:ed=pd.Timestamp(earnings_date) if earnings_date else None
        except Exception:ed=None
        for x in exps:
            try:
                xd=pd.Timestamp(x)
                if ed is None or xd>=ed.tz_localize(None):chosen=x;break
            except Exception:pass
        chosen=chosen or (exps[0] if exps else None)
        if chosen:
            ch=t.option_chain(chosen);calls=ch.calls;puts=ch.puts;px=None
            try:px=_finite((t.fast_info or {}).get("last_price"))
            except Exception:pass
            if px is None and h is not None and not h.empty:px=float(c.iloc[-1])
            if px and calls is not None and puts is not None and not calls.empty and not puts.empty:
                for df in (calls,puts):
                    for col in ("strike","impliedVolatility","volume","openInterest"):df[col]=pd.to_numeric(df[col],errors="coerce")
                calls["dist"]=(calls.strike-px).abs();puts["dist"]=(puts.strike-px).abs();civ=_finite(calls.sort_values("dist").iloc[0].get("impliedVolatility"));piv=_finite(puts.sort_values("dist").iloc[0].get("impliedVolatility"))
                if civ and piv:
                    try:dte=max(1,(pd.Timestamp(chosen)-pd.Timestamp.now()).days)
                    except Exception:dte=1
                    out["event_implied_move_pct"]=(civ+piv)/2*math.sqrt(dte/365)*100
                    c5=calls.iloc[(calls.strike-px*1.05).abs().argsort()[:1]];p5=puts.iloc[(puts.strike-px*.95).abs().argsort()[:1]]
                    if len(c5) and len(p5):
                        a=_finite(c5.iloc[0].get("impliedVolatility"));b=_finite(p5.iloc[0].get("impliedVolatility"));out["iv_skew_pct"]=(b-a)*100 if a is not None and b is not None else None
        out["available"]=any(out[k] is not None for k in ("momentum_5d_pct","momentum_20d_pct","volume_ratio","event_implied_move_pct","iv_skew_pct"))
    except Exception as e:out["error"]=str(e)[:180]
    return out


def _guidance_from_public_web(symbol):
    """Extract company guidance when a public earnings page states it.

    This is intentionally a soft signal. It is used only to position the AEL
    Whisper between management's stated range and the analyst consensus.
    """
    out={"available":False,"eps_low":None,"eps_high":None,"revenue_low":None,"revenue_high":None,"source":"Public company guidance text","error":None}
    text,err=_safe_text(f"https://www.benzinga.com/quote/{symbol}/earnings-forecasts")
    if not text:
        out["error"]=err;return out
    plain=re.sub(r"<[^>]+>"," ",text);plain=re.sub(r"\s+"," ",plain)
    me=re.search(r"guided.*?earnings per share.*?range of\s+\$?([\d,.]+)\s*-\s*\$?([\d,.]+)",plain,re.I)
    mr=re.search(r"guided.*?revenue.*?range of\s+\$?([\d,.]+)([KMBT])?\s*-\s*\$?([\d,.]+)([KMBT])?",plain,re.I)
    if me:out["eps_low"]=_finite(me.group(1));out["eps_high"]=_finite(me.group(2))
    if mr:out["revenue_low"]=_money_token(mr.group(1)+(mr.group(2) or ""));out["revenue_high"]=_money_token(mr.group(3)+(mr.group(4) or ""))
    out["available"]=any(v is not None for v in (out["eps_low"],out["eps_high"],out["revenue_low"],out["revenue_high"]))
    return out


def _analyst_revision(consensus):
    for key in ("revision_30d_pct","revision_7d_pct","revision_90d_pct"):
        v=_finite(consensus.get(key))
        if v is not None:return v,key
    return None,None


def _whisper_estimate(base, consensus, hist, qtr, guidance, kind):
    """Analyst-like whisper estimator.

    Consensus is the anchor. Revision drift, guidance positioning, historical
    surprise and the independent fundamental nowcast create a bounded whisper
    premium/discount. The effect is intentionally small: AEL is approximating a
    private analyst update, not mechanically extrapolating past beats.
    """
    if base is None:return None,[],None
    factors=[]
    drift,key=_analyst_revision(consensus or {})
    if drift is not None:
        factors.append(("预测修正",_clamp(drift,-12,12),0.30))
    hs=_finite(hist.get("eps_recent_weighted_pct")) if kind=="eps" else None
    if hs is not None:
        factors.append(("历史惊喜偏差",_clamp(hs,-15,15),0.18))
    g_lo=_finite(guidance.get(f"{kind}_low"));g_hi=_finite(guidance.get(f"{kind}_high"))
    if g_lo is not None and g_hi is not None and g_hi>=g_lo:
        mid=(g_lo+g_hi)/2; pos=(mid/base-1)*100 if base else 0
        factors.append(("公司指引定位",_clamp(pos,-12,12),0.25))
    qv=_finite(qtr.get(kind))
    if qv is not None and base:
        nowcast_gap=(qv/base-1)*100
        factors.append(("基本面Nowcast",_clamp(nowcast_gap,-15,15),0.27))
    if not factors:return base,[],"consensus-only"
    raw=sum(v*w for _,v,w in factors)/sum(w for _,_,w in factors)
    # Revenue is generally less noisy than EPS; keep its whisper premium tighter.
    cap=8.0 if kind=="eps" else 5.0
    adj=_clamp(raw,-cap,cap)
    return base*(1+adj/100),[{"factor":n,"contribution_pct":round(v*w/sum(ww for _,_,ww in factors),3)} for n,v,w in factors],key


def _market_implied_adjustment(whisper, market, hist, kind):
    if whisper is None:return None,[]
    parts=[]
    mom=market.get("momentum_20d_pct")
    if mom is not None:parts.append(("20D价格动量",_clamp(mom/4,-5,5),0.22))
    vr=market.get("volume_ratio")
    if vr is not None:parts.append(("异常成交量",_clamp((vr-1)*3,-4,4),0.12))
    skew=market.get("iv_skew_pct")
    if skew is not None:parts.append(("期权IV偏斜",_clamp(-skew/4,-4,4),0.18))
    move=market.get("event_implied_move_pct")
    if move is not None:parts.append(("事件隐含波动",_clamp((move-8)*0.20,-3,3),0.10))
    hs=hist.get("eps_recent_weighted_pct")
    if hs is not None:parts.append(("历史市场惊喜",_clamp(hs/4,-4,4),0.18))
    # The market-implied premium is deliberately smaller than the analyst-like
    # whisper premium. Price embeds many future quarters, not just this one.
    if not parts:return whisper,[]
    raw=sum(v*w for _,v,w in parts)/sum(w for _,_,w in parts)
    if kind=="revenue":raw*=0.55
    adj=_clamp(raw,-7,7)
    return whisper*(1+adj/100),[{"factor":n,"contribution_pct":round(v*w/sum(ww for _,_,ww in parts),3)} for n,v,w in parts]


def _confidence(evidence_n,market_available,base_source,exact_ew=False):
    score=42+min(20,evidence_n*4)
    if market_available:score+=12
    if base_source and base_source not in ("AEL fundamental nowcast","AEL historical-quarter model"):score+=14
    if exact_ew:score+=10
    return int(_clamp(score,42,96))


def analyze_whisper(symbol):
    requested=str(symbol or "").strip().upper()
    if not requested:return {"ok":False,"error":"缺少标的"}
    now=datetime.now(timezone.utc).timestamp()
    with _LOCK:
        c=_CACHE.get(requested)
        if c and now-c[0]<TTL:return c[1]
    # Quarter lock is established BEFORE consensus selection.  This prevents
    # a stale EPS from a just-reported quarter from being paired with the next
    # quarter's revenue (the exact failure seen on SNDK v2.6).
    qtr=_quarterly_fallback(requested)
    target_period=_target_period_from_history(qtr)
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_cons=ex.submit(_consensus_bundle,requested,target_period)
        f_hist=ex.submit(_historical_surprise,requested)
        f_guid=ex.submit(_guidance_from_public_web,requested)
        cons=f_cons.result();hist=f_hist.result();guid=f_guid.result()
    earnings_date=cons.get("earnings_date")
    market=_market_signal(requested,earnings_date)
    eps_src=cons.get("eps") or {};rev_src=cons.get("revenue") or {}
    eps_cons=_finite(eps_src.get("eps"));rev_cons=_finite(rev_src.get("revenue"))
    eps_base=eps_cons if eps_cons is not None else _finite(qtr.get("eps"));rev_base=rev_cons if rev_cons is not None else _finite(qtr.get("revenue"))
    eps_base_source=eps_src.get("source") if eps_cons is not None else qtr.get("source");rev_base_source=rev_src.get("source") if rev_cons is not None else qtr.get("source")
    eps_whisper,eps_wparts,eps_revision_key=_whisper_estimate(eps_base,eps_src,hist,qtr,guid,"eps")
    rev_whisper,rev_wparts,rev_revision_key=_whisper_estimate(rev_base,rev_src,hist,qtr,guid,"revenue")
    eps_implied,eps_mparts=_market_implied_adjustment(eps_whisper,market,hist,"eps")
    rev_implied,rev_mparts=_market_implied_adjustment(rev_whisper,market,hist,"revenue")
    eps_pricein=(eps_implied/eps_cons-1)*100 if eps_implied is not None and eps_cons else None
    rev_pricein=(rev_implied/rev_cons-1)*100 if rev_implied is not None and rev_cons else None
    exact_ew=bool(_env("EW_API_KEY") and any(x.get("source")=="Earnings Whispers Data API" and x.get("available") and x.get("period_validated") for x in cons.get("results",[])))
    ew=next((x for x in cons.get("results",[]) if x.get("source")=="Earnings Whispers Data API" and x.get("available") and x.get("period_validated")),{})
    confidence=max(_confidence(len(eps_wparts)+len(eps_mparts),market.get("available"),eps_base_source,exact_ew),_confidence(len(rev_wparts)+len(rev_mparts),market.get("available"),rev_base_source,exact_ew))
    out={
      "ok":True,"symbol":requested,"as_of":datetime.now(timezone.utc).isoformat(),"next_earnings_date":earnings_date,
      "model":"AEL Market-Implied Whisper v2.6.1","status":"inferred","confidence_pct":confidence,
      "target_period":target_period.get("target_end"),"last_actual_period":target_period.get("last_actual_end"),"period_lock":"strict",
      "data_mode":"observed-consensus→AEL-whisper→market-implied" if (eps_cons is not None or rev_cons is not None) else "fundamental-nowcast→AEL-whisper→market-implied",
      "revenue":{"consensus":rev_cons,"base":rev_base,"whisper":rev_whisper,"implied":rev_implied,"whisper_premium_pct":(rev_whisper/rev_cons-1)*100 if rev_whisper is not None and rev_cons else None,"pricein_pct":rev_pricein,"base_source":rev_base_source,"low":rev_src.get("revenue_low"),"high":rev_src.get("revenue_high"),"whisper_evidence":rev_wparts,"market_evidence":rev_mparts,"status":"inferred","reason":"共识 → 修正 → 指引定位 → 基本面Nowcast → 市场价格/期权Price-in。"},
      "eps":{"consensus":eps_cons,"base":eps_base,"whisper":eps_whisper,"implied":eps_implied,"whisper_premium_pct":(eps_whisper/eps_cons-1)*100 if eps_whisper is not None and eps_cons else None,"pricein_pct":eps_pricein,"base_source":eps_base_source,"low":eps_src.get("eps_low"),"high":eps_src.get("eps_high"),"whisper_evidence":eps_wparts,"market_evidence":eps_mparts,"status":"inferred","reason":"共识 → 修正 → 指引定位 → 基本面Nowcast → 市场价格/期权Price-in。"},
      "market_beat_threshold":{"revenue":rev_implied,"eps":eps_implied},
      "market_signals":market,"historical_surprise":hist,"fundamental_nowcast":qtr,"guidance":guid,
      "provider_status":[{"source":x.get("source"),"available":bool(x.get("available")) and bool(x.get("period_validated")),"period_validated":bool(x.get("period_validated")),"error":x.get("error") or x.get("rejected_reason")} for x in cons.get("results",[])],
      "licensed_reference":{"available":exact_ew,"whisper_eps":ew.get("whisper_eps"),"consensus_eps":ew.get("eps"),"consensus_revenue":ew.get("revenue"),"source":"Earnings Whispers Data API" if exact_ew else None},
      "sources":{"consensus":eps_src.get("source") or rev_src.get("source") or qtr.get("source"),"history":"Yahoo historical earnings","market":"Yahoo price + listed options","guidance":guid.get("source") if guid.get("available") else None},
      "method_note":"AEL v2.6.1 固定三层：① SELL-SIDE CONSENSUS：多源交叉验证，且所有 EPS/Revenue 必须先通过同一目标财季的严格 Quarter Lock；② AEL WHISPER ESTIMATE：以锁定后的共识为锚，叠加预测修正、公司指引定位、历史惊喜偏差与基本面Nowcast；③ AEL MARKET-IMPLIED：再叠加价格动量、异常成交量、事件期权IV、IV偏斜与历史市场惊喜，估计市场已经 Price-in 的本季门槛。任何无法证明属于目标财季的数据都被拒绝，绝不允许跨季度拼接。AEL 不声称看到 Earnings Whispers 私有模型或私人买方订单簿。配置合法 EW_API_KEY 后，官方 Whisper 仅作为独立校准参考。"
    }
    with _LOCK:_CACHE[requested]=(now,out)
    return out
