"""AEL Market-Implied Whisper Engine.

This is an explicit public-data inference layer. It does NOT claim access to
Earnings Whispers' proprietary Whisper number or private buy-side models.
Every numeric output is either OBSERVED from a public source or INFERRED from
observable inputs; missing inputs stay unavailable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import math, os, re, threading
import requests
import pandas as pd
import yfinance as yf

_CACHE = {}
_LOCK = threading.Lock()
TTL = 900


def _finite(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _nested(row, *keys):
    """Extract Yahoo earnings-trend nested values defensively."""
    cur=row
    for key in keys:
        if isinstance(cur, dict):
            cur=cur.get(key)
        else:
            try:
                cur=cur[key]
            except Exception:
                return None
    return cur


def _metric(row, *paths):
    for path in paths:
        if isinstance(path, str): path=(path,)
        v=_nested(row, *path)
        if v is not None:
            return _finite(v)
    return None


def _unwrap_quote_value(v):
    if isinstance(v, dict):
        if 'raw' in v: return _finite(v.get('raw'))
        if 'fmt' in v: return v.get('fmt')
    return v


def _yahoo_quote_summary(symbol):
    """Lightweight fallback for earningsTrend when yfinance parsing is unavailable.
    Returns only public Yahoo quoteSummary fields; failures remain unavailable.
    """
    out={"trend":[],"earnings_dates":[],"error":None}
    url=f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{requests.utils.quote(symbol)}"
    params={"modules":"earningsTrend,calendarEvents"}
    try:
        r=requests.get(url,params=params,headers={"User-Agent":"Mozilla/5.0"},timeout=8)
        r.raise_for_status()
        js=r.json()
        result=((js.get('quoteSummary') or {}).get('result') or [])
        if not result: return out
        obj=result[0] or {}
        et=obj.get('earningsTrend') or {}
        out['trend']=et.get('trend') or []
        ce=obj.get('calendarEvents') or {}
        for k in ('earnings','earningsDate'):
            vals=ce.get(k) or []
            if isinstance(vals,list):
                out['earnings_dates'].extend(vals)
            elif vals: out['earnings_dates'].append(vals)
    except Exception as e:
        out['error']=str(e)[:180]
    return out


def _trend_from_yahoo_summary(symbol):
    raw=_yahoo_quote_summary(symbol)
    out={"available":False,"source":"Yahoo Finance quoteSummary / earningsTrend",
         "next_earnings_date":None,"eps":{},"revenue":{},"error":raw.get('error')}
    trend=raw.get('trend') or []
    for row in trend:
        period=str(row.get('period') or '').strip()
        if period not in ('0q','+0q','0y','+0y'): continue
        ee=row.get('earningsEstimate') or {}; re_=row.get('revenueEstimate') or {}; et=row.get('epsTrend') or {}
        eps_cur=_unwrap_quote_value(et.get('current'))
        eps_7=_unwrap_quote_value(et.get('7daysAgo')); eps_30=_unwrap_quote_value(et.get('30daysAgo')); eps_90=_unwrap_quote_value(et.get('90daysAgo'))
        eps_avg=_unwrap_quote_value(ee.get('avg') or ee.get('average') or ee.get('current'))
        eps_low=_unwrap_quote_value(ee.get('low')); eps_high=_unwrap_quote_value(ee.get('high'))
        rev_avg=_unwrap_quote_value(re_.get('avg') or re_.get('average') or re_.get('current'))
        rev_low=_unwrap_quote_value(re_.get('low')); rev_high=_unwrap_quote_value(re_.get('high'))
        n=_unwrap_quote_value(ee.get('numberOfAnalysts') or ee.get('numberOfAnalystsCurrent'))
        if eps_avg is not None or eps_cur is not None:
            out['eps']={"consensus":eps_avg if eps_avg is not None else eps_cur,"low":eps_low,"high":eps_high,
              "revision_7d_pct":((float(eps_cur)/float(eps_7)-1)*100 if eps_cur is not None and eps_7 not in (None,0) else None),
              "revision_30d_pct":((float(eps_cur)/float(eps_30)-1)*100 if eps_cur is not None and eps_30 not in (None,0) else None),
              "revision_90d_pct":((float(eps_cur)/float(eps_90)-1)*100 if eps_cur is not None and eps_90 not in (None,0) else None),
              "analyst_count":n,"period":period}
        if rev_avg is not None:
            out['revenue']={"consensus":rev_avg,"low":rev_low,"high":rev_high,"growth":_unwrap_quote_value(re_.get('growth')),"period":period}
        if out['eps'] or out['revenue']: break
    dates=raw.get('earnings_dates') or []
    future=[]
    for x in dates:
        val=x.get('raw') if isinstance(x,dict) else x
        try:
            dt=pd.Timestamp(val)
            dt=dt.tz_localize('UTC') if dt.tzinfo is None else dt.tz_convert('UTC')
            if dt>pd.Timestamp.now(tz='UTC'): future.append(dt)
        except Exception: pass
    if future: out['next_earnings_date']=min(future).isoformat()
    out['available']=bool(out['eps'].get('consensus') is not None or out['revenue'].get('consensus') is not None)
    return out


def _parse_trend(symbol):
    out={"available":False,"source":"Yahoo Finance / earnings trend",
         "next_earnings_date":None,"eps":{},"revenue":{},"error":None}
    try:
        t=yf.Ticker(symbol)
        trend=getattr(t,"earnings_trend",None)
        if trend is not None and hasattr(trend,"iterrows") and not trend.empty:
            df=trend.copy()
            # yfinance versions expose either a flat DataFrame with nested
            # dict cells or expanded columns. Handle both.
            if "period" not in df.columns:
                df=df.reset_index().rename(columns={df.index.name or "index":"period"})
            for _,r in df.iterrows():
                period=str(r.get("period") or "").strip()
                if not period: continue
                rec=r.to_dict()
                def get_nested(group, field):
                    if group in rec:
                        g=rec.get(group)
                        if isinstance(g,dict): return _finite(g.get(field))
                        # Expanded MultiIndex columns can arrive as tuple keys.
                    for k,v in rec.items():
                        if isinstance(k,tuple) and len(k)>=2 and str(k[0]).lower()==group.lower() and str(k[1]).lower()==field.lower():
                            return _finite(v)
                    return None
                eps_avg=(get_nested("earningsEstimate","avg") or get_nested("earningsEstimate","average")
                         or get_nested("earningsEstimate","current"))
                eps_low=get_nested("earningsEstimate","low")
                eps_high=get_nested("earningsEstimate","high")
                rev_avg=(get_nested("revenueEstimate","avg") or get_nested("revenueEstimate","average")
                         or get_nested("revenueEstimate","current"))
                rev_low=get_nested("revenueEstimate","low")
                rev_high=get_nested("revenueEstimate","high")
                eps_cur=get_nested("epsTrend","current")
                eps_7=get_nested("epsTrend","7daysAgo")
                eps_30=get_nested("epsTrend","30daysAgo")
                eps_60=get_nested("epsTrend","60daysAgo")
                eps_90=get_nested("epsTrend","90daysAgo")
                rev_cur=get_nested("revenueEstimate","avg") or rev_avg
                rev_growth=get_nested("revenueEstimate","growth")
                n=get_nested("earningsEstimate","numberOfAnalysts") or get_nested("numberOfAnalysts","current")
                if any(v is not None for v in (eps_avg,eps_cur,rev_avg)):
                    if period in ("0q","+0q","0y","+0y"):
                        out["eps"]={"consensus":eps_avg or eps_cur,"low":eps_low,"high":eps_high,
                                     "revision_7d_pct":((eps_cur/eps_7-1)*100 if eps_cur is not None and eps_7 not in (None,0) else None),
                                     "revision_30d_pct":((eps_cur/eps_30-1)*100 if eps_cur is not None and eps_30 not in (None,0) else None),
                                     "revision_90d_pct":((eps_cur/eps_90-1)*100 if eps_cur is not None and eps_90 not in (None,0) else None),
                                     "analyst_count":n,"period":period}
                    if rev_avg is not None and period in ("0q","+0q","0y","+0y"):
                        out["revenue"]={"consensus":rev_avg,"low":rev_low,"high":rev_high,
                                        "growth":rev_growth,"period":period}
        dates=t.get_earnings_dates(limit=12)
        if dates is not None and not dates.empty:
            now=pd.Timestamp.now(tz="UTC")
            future=[]
            for idx in dates.index:
                try:
                    dt=pd.Timestamp(idx)
                    dt=dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
                    if dt>now: future.append(dt)
                except Exception: pass
            if future: out["next_earnings_date"]=min(future).isoformat()
        out["available"]=bool(out["eps"].get("consensus") is not None or out["revenue"].get("consensus") is not None)
    except Exception as e:
        out["error"]=str(e)[:180]
    if not out["available"]:
        fb=_trend_from_yahoo_summary(symbol)
        if fb.get("available") or fb.get("next_earnings_date"):
            return fb
    return out


def _historical_surprise(symbol):
    out={"available":False,"eps_mean_pct":None,"eps_median_pct":None,"eps_recent_weighted_pct":None,"count":0,"error":None,"source":"Yahoo historical earnings dates"}
    try:
        t=yf.Ticker(symbol)
        d=t.get_earnings_dates(limit=12)
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
        weights=list(range(len(vals),0,-1)); out["eps_recent_weighted_pct"]=sum(v*w for v,w in zip(vals,weights))/sum(weights)
        out["available"]=True
    except Exception as e: out["error"]=str(e)[:180]
    return out


def _market_signal(symbol):
    out={"available":False,"momentum_5d_pct":None,"volume_ratio":None,"event_implied_move_pct":None,"iv_skew_pct":None,"error":None,"source":"Yahoo price + listed options"}
    try:
        t=yf.Ticker(symbol); h=t.history(period="6mo",interval="1d",auto_adjust=False)
        if h is not None and not h.empty:
            c=pd.to_numeric(h.get("Close"),errors="coerce").dropna(); v=pd.to_numeric(h.get("Volume"),errors="coerce").reindex(c.index).fillna(0)
            if len(c)>=10:
                out["momentum_5d_pct"]=(float(c.iloc[-1])/float(c.iloc[-6])-1)*100
                base=float(v.tail(60).head(max(1,min(40,len(v)-20))).mean()); cur=float(v.tail(20).mean())
                if base>0: out["volume_ratio"]=cur/base
        exps=list(t.options or []); next_date=None
        if exps:
            # Prefer an expiry near the next earnings date.
            ed=None
            try:
                ed=_parse_trend(symbol).get("next_earnings_date")
                ed=pd.Timestamp(ed) if ed else None
            except Exception: ed=None
            chosen=None
            for x in exps:
                try:
                    xd=pd.Timestamp(x)
                    if ed is None or xd>=ed.tz_localize(None): chosen=x; break
                except Exception: pass
            chosen=chosen or exps[0]
            ch=t.option_chain(chosen); calls=ch.calls; puts=ch.puts
            px=None
            try: px=_finite((t.fast_info or {}).get("last_price"))
            except Exception: pass
            if px is None and h is not None and not h.empty: px=float(c.iloc[-1])
            if px and calls is not None and puts is not None and not calls.empty and not puts.empty:
                for df in (calls,puts):
                    for col in ("strike","impliedVolatility","volume"): df[col]=pd.to_numeric(df[col],errors="coerce")
                calls["dist"]=(calls.strike-px).abs(); puts["dist"]=(puts.strike-px).abs()
                civ=_finite(calls.sort_values("dist").iloc[0].get("impliedVolatility")); piv=_finite(puts.sort_values("dist").iloc[0].get("impliedVolatility"))
                if civ and piv:
                    iv=(civ+piv)/2
                    try:
                        dte=max(1,(pd.Timestamp(chosen)-pd.Timestamp.now()).days)
                    except Exception: dte=1
                    out["event_implied_move_pct"]=iv*math.sqrt(dte/365)*100
                    c5=calls.iloc[(calls.strike-px*1.05).abs().argsort()[:1]]; p5=puts.iloc[(puts.strike-px*.95).abs().argsort()[:1]]
                    if len(c5) and len(p5):
                        a=_finite(c5.iloc[0].get("impliedVolatility")); b=_finite(p5.iloc[0].get("impliedVolatility"))
                        if a is not None and b is not None: out["iv_skew_pct"]=(b-a)*100
        out["available"]=any(out[k] is not None for k in ("momentum_5d_pct","volume_ratio","event_implied_move_pct","iv_skew_pct"))
    except Exception as e: out["error"]=str(e)[:180]
    return out


def _clamp(x,lo,hi): return max(lo,min(hi,x))


def _infer_surprise(hist,trend,market,kind):
    """Return an explicit, bounded market-implied surprise percentage."""
    parts=[]
    # Historical earnings surprise is the strongest observable anchor.
    h=hist.get("eps_recent_weighted_pct")
    if h is not None: parts.append(("历史财报惊喜",_clamp(h,-15,15),0.45))
    rev=trend.get("eps",{}).get("revision_30d_pct") if kind=="eps" else None
    if rev is not None: parts.append(("30D盈利预测修正",_clamp(rev,-10,10),0.20))
    mom=market.get("momentum_5d_pct")
    if mom is not None: parts.append(("财报前价格动量",_clamp(mom/2,-5,5),0.15))
    skew=market.get("iv_skew_pct")
    if skew is not None: parts.append(("期权IV偏斜",_clamp(-skew/3,-5,5),0.10))
    vr=market.get("volume_ratio")
    if vr is not None: parts.append(("财报前成交量",_clamp((vr-1)*3,-5,5),0.10))
    if not parts: return None,[],0
    raw=sum(v*w for _,v,w in parts)/sum(w for _,_,w in parts)
    # Revenue tends to be less sensitive than EPS to historical EPS surprise;
    # damp the inferred revenue surprise to avoid pretending revenue precision.
    if kind=="revenue": raw*=0.65
    return _clamp(raw,-12,12),[{"factor":n,"contribution_pct":round(v*w/sum(w2 for _,_,w2 in parts),3)} for n,v,w in parts],len(parts)


def _confidence(parts, observed, has_consensus):
    base=25 if has_consensus else 0
    base += min(45, len(parts)*10)
    if observed: base+=20
    return int(_clamp(base,0,95))


def analyze_whisper(symbol):
    requested=str(symbol or "").strip().upper()
    if not requested: return {"ok":False,"error":"缺少标的"}
    now=datetime.now(timezone.utc).timestamp()
    with _LOCK:
        c=_CACHE.get(requested)
        if c and now-c[0]<TTL: return c[1]
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs={ex.submit(_parse_trend,requested):"trend",ex.submit(_historical_surprise,requested):"history",ex.submit(_market_signal,requested):"market"}
        r={}
        for f in as_completed(fs):
            try:r[fs[f]]=f.result()
            except Exception as e:r[fs[f]]={"available":False,"error":str(e)[:180]}
    trend=r.get("trend",{}); hist=r.get("history",{}); market=r.get("market",{})
    eps=trend.get("eps",{}); rev=trend.get("revenue",{})
    eps_surprise,eps_parts,eps_n=_infer_surprise(hist,trend,market,"eps")
    rev_surprise,rev_parts,rev_n=_infer_surprise(hist,trend,market,"revenue")
    eps_cons=eps.get("consensus"); rev_cons=rev.get("consensus")
    eps_implied=eps_cons*(1+eps_surprise/100) if eps_cons is not None and eps_surprise is not None else None
    rev_implied=rev_cons*(1+rev_surprise/100) if rev_cons is not None and rev_surprise is not None else None
    # "Market beat threshold" = the AEL implied result, not a guarantee or forecast.
    # It is the numerical threshold the engine estimates is already embedded in price.
    confidence=max(_confidence(eps_parts,market.get("available"),eps_cons is not None),
                   _confidence(rev_parts,market.get("available"),rev_cons is not None))
    out={
      "ok":True,"symbol":requested,"as_of":datetime.now(timezone.utc).isoformat(),
      "next_earnings_date":trend.get("next_earnings_date"),
      "model":"AEL Market-Implied Whisper v1",
      "status":"inferred" if (eps_implied is not None or rev_implied is not None) else "unavailable",
      "confidence_pct":confidence,
      "revenue":{"consensus":rev_cons,"implied":rev_implied,"implied_surprise_pct":rev_surprise,"low":rev.get("low"),"high":rev.get("high"),"growth":rev.get("growth"),"evidence":rev_parts,"status":"inferred" if rev_implied is not None else "unavailable","reason":"基于可验证一致预期与公开市场信号推断。" if rev_implied is not None else "缺少可验证的下一季营收一致预期，暂不生成数值。"},
      "eps":{"consensus":eps_cons,"implied":eps_implied,"implied_surprise_pct":eps_surprise,"low":eps.get("low"),"high":eps.get("high"),"evidence":eps_parts,"status":"inferred" if eps_implied is not None else "unavailable","reason":"基于可验证一致预期与公开市场信号推断。" if eps_implied is not None else "缺少可验证的下一季 EPS 一致预期，暂不生成数值。"},
      "guidance":{"status":"unavailable","revenue":None,"eps":None,"margin":None,"reason":"当前版本未取得可验证的公司管理层下一季度指导；不以卖方共识或模型值冒充 Guidance。"},
      "market_beat_threshold":{"revenue":rev_implied,"eps":eps_implied},
      "market_signals":market,
      "historical_surprise":hist,
      "sources":{"consensus":"Yahoo earnings trend","history":"Yahoo historical earnings dates","market":"Yahoo price + listed options"},
      "method_note":"AEL Implied Whisper 是公开数据推断，不是 Earnings Whispers 私有 Whisper，也不是私人买方模型。核心输出由一致预期 + 历史实际财报惊喜 + 近期预测修正 + 财报前价格/成交量 + 事件期权IV偏斜共同推断；每个贡献可拆解。缺失项不补值。Guidance 只有在取得可验证管理层指引时才显示。"
    }
    with _LOCK:_CACHE[requested]=(now,out)
    return out
