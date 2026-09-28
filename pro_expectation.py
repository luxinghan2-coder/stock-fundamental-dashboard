"""AEL Pro Buy-Side Expectation Engine.

Optional research layer only. It is never imported by Lite core scoring or scans.
Every source is isolated: source failure => None + diagnostics, never a whole-request
failure unless the primary symbol history itself is unavailable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import math
import threading
import os
import requests
import yfinance as yf
import pandas as pd

_CACHE = {}
_LOCK = threading.Lock()
TTL = 900


def _finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_history(symbol):
    try:
        h = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None, "历史行情为空"
        h = h.dropna(subset=["Close"])
        if len(h) < 30:
            return None, "历史行情不足30个交易日"
        return h, None
    except Exception as exc:
        return None, str(exc)[:180]


def _price_signal(h):
    close = pd.to_numeric(h["Close"], errors="coerce").dropna()
    vol = pd.to_numeric(h.get("Volume"), errors="coerce").reindex(close.index).fillna(0)
    if len(close) < 30:
        return None
    px = float(close.iloc[-1])
    ret20 = (px / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
    ret60 = (px / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else None
    ma200 = float(close.tail(min(200, len(close))).mean())
    vol20 = float(vol.tail(min(20, len(vol))).mean())
    vol_prev = float(vol.tail(min(60, len(vol))).head(max(1, min(40, len(vol)-20))).mean())
    vol_ratio = vol20 / vol_prev if vol_prev > 0 else None
    dd = (px / float(close.cummax().iloc[-1]) - 1) * 100
    return {
        "price": px, "momentum_20d_pct": ret20, "momentum_60d_pct": ret60,
        "above_ma200": px >= ma200 if ma200 else None,
        "volume_ratio_20d": vol_ratio, "drawdown_from_high_pct": dd,
    }



def _finra_ats_source(symbol):
    """Optional FINRA OTC/ATS evidence. Requires FINRA_API_TOKEN.

    FINRA publishes weekly ATS/non-ATS aggregate data with a delay. This is
    evidence about reported off-exchange activity, not a real-time dark-pool
    order book and not proof of buy/sell direction.
    """
    out = {"available": False, "weeks": [], "ats_share_pct": None,
           "ats_share_change_pct": None, "ats_volume_z": None,
           "block_or_flow_direction": "unknown", "error": None,
           "source": "FINRA OTC Transparency / Weekly Summary"}
    token = os.getenv("FINRA_API_TOKEN", "").strip()
    if not token:
        out["error"] = "未配置 FINRA_API_TOKEN；不会使用第三方代理冒充暗池数据"
        return out
    try:
        url = "https://api.finra.org/data/group/OTCMarket/name/weeklySummary"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
                   "Data-API-Version": "1"}
        fields = ["issueSymbolIdentifier", "issueName", "weekStartDate",
                  "summaryStartDate", "summaryTypeCode", "reportTypeCode", "tierIdentifier",
                  "totalWeeklyShareQuantity", "totalTradesCountSum", "lastUpdateDate"]
        # Ask for the recent 12 weeks for this symbol in one request.
        payload = {"limit": 200, "fields": fields, "compareFilters": [
            {"compareType": "equal", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol},
            {"compareType": "equal", "fieldName": "tierIdentifier", "fieldValue": "T1"},
        ]}
        r = requests.post(url, headers=headers, json=payload, timeout=3.0)
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            out["error"] = "FINRA 未返回该标的 ATS/OTC 周度数据"
            return out
        weekly = {}
        for row in rows:
            week = row.get("weekStartDate") or row.get("summaryStartDate")
            if not week:
                continue
            code = str(row.get("summaryTypeCode") or row.get("reportTypeCode") or row.get("reportType") or "").upper()
            shares = _finite(row.get("totalWeeklyShareQuantity")) or 0.0
            if code.startswith("ATS_W_") or code.startswith("ATS"):
                bucket = "ats"
            elif code.startswith("OTC_W_") or "NON" in code or code.startswith("OTC"):
                bucket = "non_ats"
            else:
                continue
            weekly.setdefault(str(week), {"ats": 0.0, "non_ats": 0.0})[bucket] += shares
        ordered = []
        for week, x in sorted(weekly.items()):
            total = x["ats"] + x["non_ats"]
            if total <= 0:
                continue
            ordered.append({"week": week, "ats_shares": x["ats"],
                            "non_ats_shares": x["non_ats"],
                            "ats_share_pct": x["ats"] / total * 100})
        if not ordered:
            out["error"] = "FINRA 周度记录缺少可计算的 ATS/Non-ATS 成交量"
            return out
        latest = ordered[-1]
        out["weeks"] = ordered[-12:]
        out["ats_share_pct"] = latest["ats_share_pct"]
        if len(ordered) >= 2:
            out["ats_share_change_pct"] = latest["ats_share_pct"] - ordered[-2]["ats_share_pct"]
        vals = [x["ats_shares"] for x in ordered[-8:]]
        if len(vals) >= 4:
            mu = sum(vals[:-1]) / max(1, len(vals)-1)
            sd = (sum((v-mu)**2 for v in vals[:-1]) / max(1, len(vals)-2)) ** 0.5 if len(vals) > 2 else 0
            out["ats_volume_z"] = (latest["ats_shares"] - mu) / sd if sd > 0 else 0.0
        out["available"] = True
        return out
    except Exception as exc:
        out["error"] = str(exc)[:180]
        return out


def _dark_expectation(ats, options, analyst):
    """Infer an expectation *gap* only from observable inputs.

    Direction is intentionally unknown when ATS data alone cannot identify
    whether trades were buys or sells. The result is a pressure/attention
    measure, not a directional trading signal.
    """
    if not ats.get("available"):
        return {"status": "unavailable", "data": {"score": None, "direction": "unknown"},
                "reason": ats.get("error") or "ATS数据不可用"}
    share = ats.get("ats_share_pct")
    change = ats.get("ats_share_change_pct")
    z = ats.get("ats_volume_z")
    # Activity/attention score: higher means stronger reported ATS activity,
    # not bullishness. Keep it bounded and fully decomposable.
    raw = 0.0
    parts = []
    if share is not None:
        p = max(-1.0, min(1.0, (share - 25.0) / 20.0))
        raw += p * 0.5; parts.append(("ATS占比", p, 0.5))
    if change is not None:
        p = max(-1.0, min(1.0, change / 5.0))
        raw += p * 0.3; parts.append(("ATS占比变化", p, 0.3))
    if z is not None:
        p = max(-1.0, min(1.0, z / 3.0))
        raw += p * 0.2; parts.append(("ATS成交异常度", p, 0.2))
    score = round(50 + 50 * raw, 1)
    return {"status": "inferred", "data": {
        "activity_score": score, "direction": "unknown",
        "ats_share_pct": share, "ats_share_change_pct": change,
        "ats_volume_z": z,
        "decomposition": [{"factor": n, "normalized": round(v, 3), "weight": w} for n,v,w in parts],
        "interpretation": "分数越高代表报告的ATS活动/异常度越高，不代表买入或卖出方向。"
    }, "reason": "由FINRA报告的ATS/Non-ATS周度成交痕迹推断；不是实时暗池订单簿。"}


def _analyst_source(symbol):
    out = {"available": False, "target_mean": None, "target_median": None,
           "recommendation_mean": None, "number_of_analysts": None, "error": None}
    try:
        info = yf.Ticker(symbol).info or {}
        out["target_mean"] = _finite(info.get("targetMeanPrice"))
        out["target_median"] = _finite(info.get("targetMedianPrice"))
        out["recommendation_mean"] = _finite(info.get("recommendationMean"))
        out["number_of_analysts"] = _finite(info.get("numberOfAnalystOpinions"))
        out["available"] = any(out[k] is not None for k in ("target_mean", "target_median", "recommendation_mean"))
    except Exception as exc:
        out["error"] = str(exc)[:180]
    return out


def _options_source(symbol):
    out = {"available": False, "implied_move_pct": None, "atm_iv_pct": None,
           "put_call_oi_ratio": None, "error": None}
    try:
        t = yf.Ticker(symbol)
        expiries = list(t.options or [])
        if not expiries:
            out["error"] = "期权到期日不可用"
            return out
        # Keep this bounded: nearest expiry only, one chain request per call.
        exp = expiries[0]
        chain = t.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        if calls is None or calls.empty or puts is None or puts.empty:
            out["error"] = "期权链为空"
            return out
        px = _finite((t.fast_info or {}).get("last_price"))
        if px is None:
            px = _finite(t.info.get("currentPrice"))
        if px is None:
            out["error"] = "标的现价不可用"
            return out
        calls = calls.assign(dist=(pd.to_numeric(calls["strike"], errors="coerce")-px).abs()).sort_values("dist")
        puts = puts.assign(dist=(pd.to_numeric(puts["strike"], errors="coerce")-px).abs()).sort_values("dist")
        c = calls.iloc[0]; p = puts.iloc[0]
        civ = _finite(c.get("impliedVolatility")); piv = _finite(p.get("impliedVolatility"))
        ivs = [x for x in (civ, piv) if x is not None and x > 0]
        if ivs:
            iv = sum(ivs)/len(ivs)
            out["atm_iv_pct"] = iv * 100
            # Approximate one-expiry expected move using IV * sqrt(T). This is a
            # market-implied statistic, not a forecast; use calendar days.
            try:
                dte = max(1, (pd.Timestamp(exp) - pd.Timestamp.utcnow().tz_localize(None)).days)
            except Exception:
                dte = 1
            out["implied_move_pct"] = iv * math.sqrt(dte/365) * 100
        coi = _finite(c.get("openInterest")) or 0
        poi = _finite(p.get("openInterest")) or 0
        if coi > 0:
            out["put_call_oi_ratio"] = poi / coi
        out["available"] = any(out[k] is not None for k in ("implied_move_pct", "atm_iv_pct", "put_call_oi_ratio"))
    except Exception as exc:
        out["error"] = str(exc)[:180]
    return out



def _crypto_onchain_source(symbol):
    """Optional crypto on-chain dark-flow evidence.

    This is deliberately limited to observable exchange/entity flows. It does
    not claim to see private OTC orders or identify the owner of an address.
    CryptoQuant access is optional and isolated behind CRYPTOQUANT_API_KEY.
    """
    out = {"available": False, "asset": symbol, "direction": "unknown",
           "netflow_latest": None, "netflow_7d_avg": None,
           "reserve_change_7d_pct": None, "whale_inflow_ratio_pct": None,
           "activity_score": None, "error": None,
           "source": "CryptoQuant on-chain exchange/entity flows"}
    token = os.getenv("CRYPTOQUANT_API_KEY", "").strip()
    if not token:
        out["error"] = "未配置 CRYPTOQUANT_API_KEY；链上暗盘仅在接入可验证链上数据源后显示"
        return out
    asset = {"BTCUSD":"btc", "ETHUSD":"eth", "SOLUSD":"sol"}.get(symbol.upper())
    if not asset:
        out["error"] = "当前链上暗盘仅对支持的加密资产启用；商品/股票不适用"
        return out
    base = "https://api.cryptoquant.com/v1"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def get_metric(path, params):
        r = requests.get(base + path, headers=headers, params=params, timeout=3.0)
        r.raise_for_status()
        data = r.json() or {}
        if _finite(data.get("status", {}).get("code")) not in (None, 200):
            raise RuntimeError(str(data.get("status", {}).get("message") or "CryptoQuant返回错误"))
        return (data.get("result") or {}).get("data") or []

    try:
        # Keep to three small daily series. No block-level fan-out in the Pro page.
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_net = ex.submit(get_metric, f"/{asset}/exchange-flows/netflow", {"exchange":"all_exchange","window":"day","limit":14})
            f_res = ex.submit(get_metric, f"/{asset}/exchange-flows/reserve", {"exchange":"all_exchange","window":"day","limit":14})
            f_in = ex.submit(get_metric, f"/{asset}/exchange-flows/inflow", {"exchange":"all_exchange","window":"day","limit":14})
            net = f_net.result(); reserve = f_res.result(); inflow = f_in.result()
        net_vals=[_finite(x.get("netflow_total")) for x in net if _finite(x.get("netflow_total")) is not None]
        if net_vals:
            out["netflow_latest"] = net_vals[-1]
            out["netflow_7d_avg"] = sum(net_vals[-7:]) / min(7, len(net_vals))
        res_vals=[_finite(x.get("reserve")) for x in reserve if _finite(x.get("reserve")) is not None]
        if len(res_vals)>=2 and res_vals[0] not in (None,0):
            # API returns oldest -> newest in normal usage; compare last 7 points.
            old=res_vals[max(0,len(res_vals)-7)]
            new=res_vals[-1]
            if old:
                out["reserve_change_7d_pct"]=(new/old-1)*100
        latest_in=inflow[-1] if inflow else {}
        it=_finite(latest_in.get("inflow_total")); top=_finite(latest_in.get("inflow_top10"))
        if it and it>0 and top is not None:
            out["whale_inflow_ratio_pct"] = top/it*100

        # Evidence score is activity/pressure, not a buy/sell oracle.
        parts=[]; raw=0.0
        if out["whale_inflow_ratio_pct"] is not None:
            p=max(0,min(1,out["whale_inflow_ratio_pct"]/60.0)); raw += p*0.4; parts.append(("Top10大额流入占比",p,0.4))
        if out["netflow_latest"] is not None and out["netflow_7d_avg"] is not None:
            base_abs=max(abs(out["netflow_7d_avg"]),1e-9); p=max(-1,min(1,out["netflow_latest"]/base_abs)); raw += ((p+1)/2)*0.35; parts.append(("最新净流入相对7日均值",(p+1)/2,0.35))
        if out["reserve_change_7d_pct"] is not None:
            p=max(-1,min(1,out["reserve_change_7d_pct"]/5.0)); raw += ((-p+1)/2)*0.25; parts.append(("交易所储备变化",(-p+1)/2,0.25))
        out["activity_score"]=round(raw*100,1) if parts else None
        if out["netflow_latest"] is not None and out["reserve_change_7d_pct"] is not None:
            if out["netflow_latest"] < 0 and out["reserve_change_7d_pct"] < 0:
                out["direction"]="偏积累"
            elif out["netflow_latest"] > 0 and (out["whale_inflow_ratio_pct"] or 0) >= 30:
                out["direction"]="偏分配"
            else:
                out["direction"]="混合/中性"
        out["decomposition"]=[{"factor":n,"normalized":round(v,3),"weight":w} for n,v,w in parts]
        out["available"]=bool(net_vals or res_vals or inflow)
        return out
    except Exception as exc:
        out["error"]=str(exc)[:180]
        return out


def _next_earnings_dark_source(symbol, ats=None):
    """Infer *buy-side/event positioning* around the next earnings event.

    IMPORTANT: this function deliberately excludes sell-side analyst targets,
    EPS consensus revisions, and analyst recommendation fields. It is a proxy
    for observable market positioning before the next report, built from:
    event-dated options, call/put activity, IV skew, pre-event price/volume,
    and (when available) FINRA ATS activity. It never claims access to private
    buy-side books or investment-committee expectations.
    """
    out={"available":False,"status":"unavailable","next_earnings_date":None,
         "event_implied_move_pct":None,"historical_abs_move_median_pct":None,
         "implied_vs_history_ratio":None,"atm_iv_pct":None,
         "call_put_volume_ratio":None,"put_call_oi_ratio":None,
         "otm_skew_proxy_pct":None,"call_volume_oi_ratio":None,
         "put_volume_oi_ratio":None,"pre_event_momentum_5d_pct":None,
         "pre_event_volume_ratio":None,"ats_share_pct":None,
         "ats_share_change_pct":None,"expectation_score":None,
         "positioning_bias":"混合/中性","decomposition":[],"error":None,
         "source":"event-dated listed options + price/volume + optional FINRA ATS"}
    try:
        t=yf.Ticker(symbol)
        dates=t.get_earnings_dates(limit=16)
        if dates is None or dates.empty:
            out["error"]="无法取得下一份财报日期"
            return out
        now=pd.Timestamp.now(tz="UTC")
        future=[]
        past=[]
        for idx,row in dates.iterrows():
            try: dt=pd.Timestamp(idx)
            except Exception: continue
            if dt.tzinfo is None: dt=dt.tz_localize("UTC")
            else: dt=dt.tz_convert("UTC")
            (future if dt>now else past).append((dt,row))
        if not future:
            out["error"]="当前没有可验证的下一份未来财报日期"
            return out
        event_dt,_=future[0]
        out["next_earnings_date"]=event_dt.isoformat()

        # One bounded historical request; never blocks the core Lite path.
        h=t.history(period="1y",interval="1d",auto_adjust=False)
        if h is not None and not h.empty:
            close=pd.to_numeric(h.get("Close"),errors="coerce").dropna()
            vol=pd.to_numeric(h.get("Volume"),errors="coerce").reindex(close.index).fillna(0)
            if len(close)>=30:
                out["pre_event_momentum_5d_pct"]=(float(close.iloc[-1])/float(close.iloc[-6])-1)*100
                base=float(vol.tail(60).head(max(1,min(40,len(vol)-20))).mean())
                cur=float(vol.tail(20).mean())
                if base>0: out["pre_event_volume_ratio"]=cur/base

        # Historical post-earnings absolute move, only from observable prices.
        try:
            moves=[]
            d0=close.index.tz_localize(None) if getattr(close.index,"tz",None) is not None else close.index
            for dt,_ in past[:10]:
                dn=dt.tz_convert(None)
                pos=int(d0.searchsorted(dn))
                if pos<1 or pos>=len(close): continue
                before=float(close.iloc[pos-1]); after=float(close.iloc[pos])
                if before: moves.append(abs(after/before-1)*100)
            if moves: out["historical_abs_move_median_pct"]=float(pd.Series(moves).median())
        except Exception:
            pass

        # Choose the first listed expiry on/after the event date. This is the
        # critical difference from generic options data: we study the actual
        # earnings event window, not the nearest weekly expiry.
        exps=list(t.options or [])
        chosen=None
        for ex in exps:
            try:
                ed=pd.Timestamp(ex).tz_localize("UTC")
                if ed>=event_dt: chosen=ex; break
            except Exception: continue
        if chosen is None and exps:
            chosen=exps[0]
        if chosen:
            chain=t.option_chain(chosen); calls=chain.calls; puts=chain.puts
            px=_finite((t.fast_info or {}).get("last_price"))
            if px is None and h is not None and not h.empty: px=float(close.iloc[-1])
            if px and calls is not None and puts is not None and not calls.empty and not puts.empty:
                for df in (calls,puts):
                    for col in ("volume","openInterest","impliedVolatility","strike"):
                        if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
                cv=float(calls["volume"].fillna(0).sum()); pv=float(puts["volume"].fillna(0).sum())
                co=float(calls["openInterest"].fillna(0).sum()); po=float(puts["openInterest"].fillna(0).sum())
                if pv>0: out["call_put_volume_ratio"]=cv/pv
                if co>0: out["put_call_oi_ratio"]=po/co
                if co>0: out["call_volume_oi_ratio"]=cv/co
                if po>0: out["put_volume_oi_ratio"]=pv/po

                calls["dist"]=(calls["strike"]-px).abs(); puts["dist"]=(puts["strike"]-px).abs()
                c_atm=calls.sort_values("dist").iloc[0]; p_atm=puts.sort_values("dist").iloc[0]
                ivs=[_finite(c_atm.get("impliedVolatility")),_finite(p_atm.get("impliedVolatility"))]
                ivs=[v for v in ivs if v and v>0]
                if ivs:
                    iv=sum(ivs)/len(ivs); out["atm_iv_pct"]=iv*100
                    try:
                        dte=max(1,(pd.Timestamp(chosen)-now.tz_convert(None)).days)
                    except Exception: dte=1
                    out["event_implied_move_pct"]=iv*math.sqrt(dte/365)*100
                # 5% OTM IV skew proxy. This is deliberately called a proxy,
                # because free chains do not reliably expose option delta.
                c5=calls.iloc[(calls["strike"]-px*1.05).abs().argsort()[:1]]
                p5=puts.iloc[(puts["strike"]-px*0.95).abs().argsort()[:1]]
                if len(c5) and len(p5):
                    civ=_finite(c5.iloc[0].get("impliedVolatility")); piv=_finite(p5.iloc[0].get("impliedVolatility"))
                    if civ is not None and piv is not None: out["otm_skew_proxy_pct"]=(piv-civ)*100

        if ats and ats.get("available"):
            out["ats_share_pct"]=ats.get("ats_share_pct")
            out["ats_share_change_pct"]=ats.get("ats_share_change_pct")

        if out["historical_abs_move_median_pct"] and out["event_implied_move_pct"]:
            out["implied_vs_history_ratio"]=out["event_implied_move_pct"]/out["historical_abs_move_median_pct"]

        # Buy-side/event positioning score. No analyst or EPS-consensus inputs.
        parts=[]; score=50.0
        def add(name,val,weight):
            nonlocal score
            if val is None:return
            p=max(-1,min(1,val)); score += p*weight; parts.append((name,p,weight))
        # Call volume pressure: ratio 1 means neutral, 2+ increasingly call-heavy.
        if out["call_put_volume_ratio"] is not None:
            r=out["call_put_volume_ratio"]; add("财报期Call/Put成交压力", max(-1,min(1,(r-1)/1.0)), 25)
        if out["otm_skew_proxy_pct"] is not None:
            # negative put-minus-call skew => relatively stronger call demand.
            add("5%OTM IV偏斜代理", max(-1,min(1,-out["otm_skew_proxy_pct"]/15.0)), 20)
        if out["pre_event_momentum_5d_pct"] is not None:
            add("财报前5D价格动量", max(-1,min(1,out["pre_event_momentum_5d_pct"]/10.0)), 15)
        if out["pre_event_volume_ratio"] is not None:
            # Volume acceleration is attention, not direction; center it at 1x.
            add("当前成交量异常（距下一财报仍有时间时作为当前定位代理）", max(-1,min(1,(out["pre_event_volume_ratio"]-1)/1.5)), 10)
        if out["ats_share_change_pct"] is not None:
            # ATS share change = attention/off-exchange activity, not direction.
            add("ATS活动变化", max(-1,min(1,out["ats_share_change_pct"]/5.0)), 10)
        if out["implied_vs_history_ratio"] is not None:
            # Magnitude only; never converts high implied move into bullishness.
            parts.append(("财报隐含/历史波动", max(-1,min(1,(out["implied_vs_history_ratio"]-1))), 20))
        if parts:
            # Normalize by the actually observed weights so missing sources do not
            # mechanically drag the score toward 50. The score is still an event
            # positioning proxy, not a buy/sell forecast.
            total_w=sum(w for _,_,w in parts)
            if total_w>0:
                score=50.0 + sum(v*w for _,v,w in parts) / total_w * 50.0
        out["expectation_score"]=round(max(0,min(100,score)),1) if parts else None
        bullish=sum(w for n,v,w in parts if v>0.25 and n not in ("财报隐含/历史波动","当前成交量异常（距下一财报仍有时间时作为当前定位代理）"))
        bearish=sum(w for n,v,w in parts if v<-0.25 and n not in ("财报隐含/历史波动","当前成交量异常（距下一财报仍有时间时作为当前定位代理）"))
        out["positioning_bias"]="偏正向" if bullish>=bearish+15 else ("偏负向" if bearish>=bullish+15 else "混合/中性")
        out["decomposition"]=[{"factor":n,"normalized":round(v,3),"weight":w} for n,v,w in parts]
        out["available"]=bool(out["next_earnings_date"] and any(out[k] is not None for k in ("event_implied_move_pct","call_put_volume_ratio","otm_skew_proxy_pct","pre_event_momentum_5d_pct")))
        out["status"]="inferred" if out["available"] else "unavailable"
        return out
    except Exception as exc:
        out["error"]=str(exc)[:180]
        return out

def _xstock_source(symbol):
    """Optional public xStocks metadata for an on-chain US equity token.

    Public endpoint; no API key required. This is metadata validation only and
    is isolated from the underlying US-equity research chain.
    """
    out={"available":False,"symbol":symbol,"underlying_symbol":None,
         "name":None,"logo":None,"trading_halted":None,"error":None,
         "source":"xStocks public Assets API"}
    try:
        r=requests.get(f"https://api.xstocks.fi/api/v2/public/assets/{requests.utils.quote(symbol, safe='')}", timeout=2.5,
                       headers={"User-Agent":"AEL/2.5 Pro Onchain Equity"})
        if r.status_code==404:
            out["error"]="该链上美股资产不存在或未公开"
            return out
        r.raise_for_status(); d=r.json() or {}
        under=d.get("underlying") or d.get("underlyingAsset") or {}
        if not isinstance(under, dict): under={}
        out["symbol"]=d.get("symbol") or symbol
        out["name"]=d.get("name")
        out["logo"]=d.get("logo")
        out["underlying_symbol"]=(under.get("symbol") or under.get("ticker") or
                                   d.get("underlyingSymbol") or d.get("underlyingTicker"))
        out["trading_halted"]=d.get("isTradingHalted")
        out["listing_country"] = under.get("listingCountry") or under.get("country")
        out["available"]=bool(out["underlying_symbol"])
        return out
    except Exception as exc:
        out["error"]=str(exc)[:180]
        return out

def analyze_expectation(symbol: str):
    requested_symbol = str(symbol or "").strip().upper()
    if not requested_symbol:
        return {"ok": False, "symbol": requested_symbol, "error": "缺少标的"}
    onchain_equity = {"available":False,"symbol":requested_symbol}
    # xStocks uses an x-suffixed token symbol such as AAPLx. If the user
    # searches the on-chain version, validate it through the public API, then
    # run the research engine against the real underlying equity ticker.
    if requested_symbol.endswith("X") and len(requested_symbol)>1:
        token_symbol = requested_symbol[:-1] + "x"
        base_symbol = requested_symbol[:-1]
        onchain_equity = _xstock_source(token_symbol)
        symbol = str(onchain_equity.get("underlying_symbol") or base_symbol).upper()
    else:
        symbol = requested_symbol
    now = time()
    with _LOCK:
        cached = _CACHE.get(symbol)
        if cached and now - cached[0] < TTL:
            return cached[1]

    h, hist_err = _safe_history(symbol)
    if h is None:
        return {"ok": False, "symbol": symbol, "error": hist_err or "暂无足够历史数据",
                "data_quality": {"primary_history": False}}

    # Optional sources are parallel and independently fail-safe.
    results = {}
    # Keep the independent sources parallel, but resolve ATS before the earnings
    # event model so its optional evidence can actually be included.
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_analyst_source, symbol): "analyst", ex.submit(_options_source, symbol): "options", ex.submit(_finra_ats_source, symbol): "ats_dark_pool", ex.submit(_crypto_onchain_source, symbol): "onchain_dark"}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as exc:
                results[k] = {"available": False, "error": str(exc)[:180]}

    ats = results.get("ats_dark_pool", {"available": False})
    try:
        results["next_earnings_dark"] = _next_earnings_dark_source(symbol, ats=ats)
    except Exception as exc:
        results["next_earnings_dark"] = {"available": False, "status": "unavailable", "error": str(exc)[:180]}

    price = _price_signal(h)
    analyst = results.get("analyst", {"available": False})
    options = results.get("options", {"available": False})
    next_earnings = results.get("next_earnings_dark", {"available": False, "status":"unavailable"})
    onchain = results.get("onchain_dark", {"available": False, "status":"unavailable"})
    dark = _dark_expectation(ats, options, analyst)
    current = _finite(price.get("price")) if price else None
    target = _finite(analyst.get("target_mean"))
    target_gap = ((target/current)-1)*100 if current and target else None

    # Explicitly keep unavailable institutional sources separate. No proxy is
    # presented as actual dark-pool/ATS/13F/CFTC observation.
    sources = {
        "earnings_revision": {"status": "unavailable", "reason": "当前版本未接入稳定的实时盈利预测修正数据源"},
        "ats_dark_pool": {"status": "observed" if ats.get("available") else "unavailable", "data": ats if ats.get("available") else {}, "reason": ats.get("error") or "FINRA ATS数据可用"},
        "dark_expectation": dark,
        "institutional_13f": {"status": "unavailable", "reason": "未接入13F历史披露解析；避免把滞后披露伪装成实时仓位"},
        "cftc_positioning": {"status": "unavailable", "reason": "当前标的未按期货品种接入CFTC持仓源"},
        "onchain_dark": {"status": "observed" if onchain.get("available") else "unavailable", "data": onchain if onchain.get("available") else {}, "reason": onchain.get("error") or "链上交易所/大额流量数据可用"},
        "next_earnings_dark": {"status": next_earnings.get("status", "unavailable"), "data": next_earnings if next_earnings.get("available") else {}, "reason": next_earnings.get("error") or "下一份财报的买方事件预期推断可用"},
        "onchain_equity": {"status": "observed" if onchain_equity.get("available") else "unavailable", "data": onchain_equity if onchain_equity.get("available") else {}, "reason": onchain_equity.get("error") or "xStocks公开资产元数据可用"},
        "analyst": {"status": "observed" if analyst.get("available") else "unavailable", "data": analyst},
        "options": {"status": "observed" if options.get("available") else "unavailable", "data": options},
        "price_volume": {"status": "observed", "data": price},
    }
    available = [k for k,v in sources.items() if v.get("status") in ("observed", "inferred")]
    out = {
        "ok": True, "symbol": symbol, "requested_symbol": requested_symbol, "as_of": pd.Timestamp.utcnow().isoformat(),
        "current_price": current,
        "analyst_target_gap_pct": target_gap,
        "sources": sources,
        "observed_count": len(available),
        "inferred_count": sum(1 for v in sources.values() if v.get("status") == "inferred"),
        "data_quality": {
            "primary_history": True,
            "optional_sources": {k:v.get("status") for k,v in sources.items()},
            "errors": {k:v.get("data",{}).get("error") for k,v in sources.items() if isinstance(v.get("data"), dict) and v.get("data",{}).get("error")},
        },
        "method_note": "AEL Pro 的“下一份财报买方暗盘预期”不使用卖方目标价、分析师评级或EPS共识修正作为核心输入；只从财报事件期权成交/持仓、IV偏斜代理、财报前价格与成交量、以及可用的FINRA ATS活动痕迹推断市场参与者的事件定位。结果全部标记为inferred，绝不声称拥有私人买方订单簿。链上美股通过xStocks公开Assets API做资产存在性/底层股票映射，失败只影响该扩展卡。",
    }
    with _LOCK:
        _CACHE[symbol] = (now, out)
    return out
