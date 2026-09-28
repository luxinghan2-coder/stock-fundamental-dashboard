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
                  "summaryStartDate", "reportTypeCode", "tierIdentifier",
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
            code = str(row.get("reportTypeCode") or "").upper()
            shares = _finite(row.get("totalWeeklyShareQuantity")) or 0.0
            if code.startswith("ATS"):
                bucket = "ats"
            elif "NON" in code or code.startswith("OTC"):
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


def analyze_expectation(symbol: str):
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return {"ok": False, "symbol": symbol, "error": "缺少标的"}
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
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_analyst_source, symbol): "analyst", ex.submit(_options_source, symbol): "options", ex.submit(_finra_ats_source, symbol): "ats_dark_pool"}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as exc:
                results[k] = {"available": False, "error": str(exc)[:180]}

    price = _price_signal(h)
    analyst = results.get("analyst", {"available": False})
    options = results.get("options", {"available": False})
    ats = results.get("ats_dark_pool", {"available": False})
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
        "analyst": {"status": "observed" if analyst.get("available") else "unavailable", "data": analyst},
        "options": {"status": "observed" if options.get("available") else "unavailable", "data": options},
        "price_volume": {"status": "observed", "data": price},
    }
    available = [k for k,v in sources.items() if v.get("status") in ("observed", "inferred")]
    out = {
        "ok": True, "symbol": symbol, "as_of": pd.Timestamp.utcnow().isoformat(),
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
        "method_note": "AEL 买方预期引擎只展示可验证的公开市场痕迹。ATS层优先使用FINRA官方周度ATS/Non-ATS数据；数据有发布延迟且不提供逐笔买卖方向，因此只能推断活动/异常度，不能声称知道暗池净买入。13F、CFTC等未接入时明确标记为unavailable，不用其他指标冒充。",
    }
    with _LOCK:
        _CACHE[symbol] = (now, out)
    return out
