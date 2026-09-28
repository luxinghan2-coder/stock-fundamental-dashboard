from __future__ import annotations

import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/pro/options", tags=["AEL Pro Options"])

DATA_BASE = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
TRADING_BASE = os.getenv(
    "ALPACA_TRADING_BASE_URL",
    "https://paper-api.alpaca.markets" if os.getenv("ALPACA_PAPER_TRADE", "true").lower() in {"1","true","yes"} else "https://api.alpaca.markets",
)
FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative")
TIMEOUT = float(os.getenv("AEL_PRO_OPTIONS_TIMEOUT", "12"))
MAX_SNAPSHOTS = max(100, min(5000, int(os.getenv("AEL_PRO_OPTIONS_MAX_SNAPSHOTS", "2500"))))
MAX_CONTRACTS = max(100, min(10000, int(os.getenv("AEL_PRO_OPTIONS_MAX_CONTRACTS", "5000"))))


def _headers() -> dict[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise HTTPException(
            status_code=503,
            detail="Pro 期权数据源未配置 Alpaca API Key。Lite 不受影响；请在 Railway 环境变量中配置，不要把密钥发到聊天里。",
        )
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"期权数据请求失败：{str(exc)[:180]}")
    if not r.ok:
        msg = ""
        try:
            msg = (r.json() or {}).get("message") or (r.json() or {}).get("error") or ""
        except Exception:
            pass
        raise HTTPException(status_code=r.status_code, detail=f"Alpaca 返回 {r.status_code}：{msg or r.text[:180]}")
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="期权数据返回不是有效 JSON")


def _finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _underlying_symbol(raw: str) -> str:
    s = str(raw or "").strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", s):
        raise HTTPException(status_code=400, detail="Pro 期权驾驶舱目前仅支持美股股票代码")
    return s


def _date_window(dte_min: int, dte_max: int) -> tuple[str, str]:
    today = date.today()
    return (
        (today + timedelta(days=dte_min)).isoformat(),
        (today + timedelta(days=dte_max)).isoformat(),
    )


def _contracts(symbol: str, exp_gte: str, exp_lte: str) -> dict[str, dict[str, Any]]:
    params = {
        "underlying_symbols": symbol,
        "status": "active",
        "expiration_date_gte": exp_gte,
        "expiration_date_lte": exp_lte,
        "limit": MAX_CONTRACTS,
    }
    data = _get(f"{TRADING_BASE}/v2/options/contracts", params)
    items = data.get("option_contracts") or data.get("contracts") or []
    return {str(x.get("symbol")): x for x in items if x.get("symbol")}


def _snapshots(
    symbol: str,
    exp_gte: str,
    exp_lte: str,
    option_type: str | None,
) -> tuple[dict[str, Any], str]:
    params: dict[str, Any] = {
        "feed": FEED,
        "limit": min(1000, MAX_SNAPSHOTS),
        "expiration_date_gte": exp_gte,
        "expiration_date_lte": exp_lte,
    }
    if option_type in {"call", "put"}:
        params["type"] = option_type

    all_snapshots: dict[str, Any] = {}
    token = None
    pages = 0
    while len(all_snapshots) < MAX_SNAPSHOTS and pages < 10:
        if token:
            params["page_token"] = token
        data = _get(f"{DATA_BASE}/v1beta1/options/snapshots/{symbol}", params)
        snaps = data.get("snapshots") or {}
        all_snapshots.update(snaps)
        token = data.get("next_page_token")
        pages += 1
        if not token or not snaps:
            break
    return all_snapshots, FEED


def _normalize(
    snapshots: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    underlying_price: float | None,
) -> list[dict[str, Any]]:
    today = date.today()
    out = []
    for contract_symbol, snap in snapshots.items():
        meta = contracts.get(contract_symbol) or {}
        exp = meta.get("expiration_date")
        strike = _finite(meta.get("strike_price"))
        typ = str(meta.get("type") or "").lower()
        if not exp or strike is None or typ not in {"call", "put"}:
            m = re.match(r"^([A-Z0-9.-]+?)(\d{6})([CP])(\d{8})$", contract_symbol)
            if not m:
                continue
            ds, typ_code, strike_code = m.group(2), m.group(3), m.group(4)
            try:
                exp = datetime.strptime(ds, "%y%m%d").date().isoformat()
            except ValueError:
                continue
            typ = "call" if typ_code == "C" else "put"
            strike = int(strike_code) / 1000.0
        try:
            dte = (date.fromisoformat(str(exp)) - today).days
        except Exception:
            continue
        if dte < 0:
            continue

        q = snap.get("latestQuote") or snap.get("latest_quote") or {}
        t = snap.get("latestTrade") or snap.get("latest_trade") or {}
        g = snap.get("greeks") or {}
        bid = _finite(q.get("bp") if q.get("bp") is not None else q.get("bid_price"))
        ask = _finite(q.get("ap") if q.get("ap") is not None else q.get("ask_price"))
        mid = (bid + ask) / 2 if bid is not None and ask is not None and bid >= 0 and ask >= bid else None
        delta = _finite(g.get("delta"))
        gamma = _finite(g.get("gamma"))
        theta = _finite(g.get("theta"))
        vega = _finite(g.get("vega"))
        rho = _finite(g.get("rho"))
        iv = _finite(g.get("impliedVolatility") if g.get("impliedVolatility") is not None else g.get("iv"))
        oi = _finite(meta.get("open_interest"))
        volume = _finite(t.get("s") if t.get("s") is not None else t.get("size"))
        spread = (ask - bid) if bid is not None and ask is not None else None
        spread_pct = (spread / mid * 100) if spread is not None and mid and mid > 0 else None
        capital = strike * 100 if typ == "put" else ((underlying_price or strike) * 100)
        premium = mid * 100 if mid is not None else None
        premium_yield = (mid / strike * 100) if mid is not None and strike > 0 and typ == "put" else (
            (mid / (underlying_price or strike) * 100) if mid is not None and (underlying_price or strike) > 0 else None
        )
        annualized = (premium_yield * 365 / dte) if premium_yield is not None and dte > 0 else None
        out.append({
            "contract": contract_symbol,
            "type": typ,
            "expiration": exp,
            "dte": dte,
            "strike": strike,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "iv": iv * 100 if iv is not None and iv <= 3 else iv,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_pct": spread_pct,
            "open_interest": oi,
            "volume": volume,
            "premium": premium,
            "capital_required": capital,
            "premium_yield": premium_yield,
            "annualized_yield": annualized,
            "underlying_price": underlying_price,
        })
    return out


def _underlying_price(symbol: str) -> float | None:
    data = _get(
        f"{DATA_BASE}/v2/stocks/{symbol}/snapshot",
        {"feed": os.getenv("ALPACA_STOCK_FEED", "iex")},
    )
    snap = data or {}
    trade = snap.get("latestTrade") or {}
    quote = snap.get("latestQuote") or {}
    return _finite(trade.get("p")) or (
        ((_finite(quote.get("bp")) or 0) + (_finite(quote.get("ap")) or 0)) / 2
        if _finite(quote.get("bp")) is not None and _finite(quote.get("ap")) is not None
        else None
    )


def _strategy_score(row: dict[str, Any], strategy: str) -> float:
    # Transparent heuristic score: liquidity + usable premium + target delta.
    # This is not a claim of expected return and is intentionally decomposable.
    oi = row.get("open_interest")
    vol = row.get("volume")
    spread_pct = row.get("spread_pct")
    delta = row.get("delta")
    yld = row.get("premium_yield")
    score = 0.0
    if oi is not None:
        score += min(25.0, math.log10(max(1.0, oi)) * 4.0)
    if vol is not None:
        score += min(15.0, math.log10(max(1.0, vol)) * 3.0)
    if spread_pct is not None:
        score += max(0.0, 25.0 - min(25.0, spread_pct * 2.0))
    if yld is not None:
        score += min(25.0, max(0.0, yld * 8.0))
    if delta is not None:
        target = -0.25 if strategy == "CSP" else 0.25
        score += max(0.0, 10.0 - abs(abs(delta) - target) * 40.0)
    return round(min(100.0, score), 1)


@router.get("/health")
def pro_options_health():
    configured = bool(
        (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID"))
        and (os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY"))
    )
    return {
        "ok": True,
        "configured": configured,
        "feed": FEED,
        "module": "AEL Pro Options",
        "execution": False,
        "message": "仅行情/研究，不包含下单执行",
    }


@router.get("/chain/{symbol}")
def pro_options_chain(
    symbol: str,
    dte_min: int = Query(7, ge=0, le=365),
    dte_max: int = Query(60, ge=1, le=730),
    option_type: str = Query("all"),
    delta_min: float | None = Query(None, ge=-1, le=1),
    delta_max: float | None = Query(None, ge=-1, le=1),
    limit: int = Query(200, ge=10, le=500),
):
    if dte_max < dte_min:
        raise HTTPException(status_code=400, detail="DTE 上限必须大于等于下限")
    sym = _underlying_symbol(symbol)
    exp_gte, exp_lte = _date_window(dte_min, dte_max)
    typ = option_type.lower()
    if typ not in {"all", "call", "put"}:
        raise HTTPException(status_code=400, detail="type 只能是 all / call / put")
    underlying = _underlying_price(sym)
    contracts = _contracts(sym, exp_gte, exp_lte)
    snaps, feed = _snapshots(sym, exp_gte, exp_lte, None if typ == "all" else typ)
    rows = _normalize(snaps, contracts, underlying)

    def keep(x):
        if delta_min is not None and (x["delta"] is None or x["delta"] < delta_min):
            return False
        if delta_max is not None and (x["delta"] is None or x["delta"] > delta_max):
            return False
        return True

    rows = [x for x in rows if keep(x)]
    rows.sort(key=lambda x: (
        x["dte"],
        abs((x["delta"] if x["delta"] is not None else 99) - (-0.25 if x["type"] == "put" else 0.25)),
        abs((x["strike"] / underlying - 1) if underlying else 99),
    ))
    rows = rows[:limit]
    return {
        "ok": True,
        "module": "AEL Pro Options",
        "symbol": sym,
        "underlying_price": underlying,
        "feed": feed,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "dte_min": dte_min, "dte_max": dte_max,
            "type": typ, "delta_min": delta_min, "delta_max": delta_max,
            "limit": limit,
        },
        "count": len(rows),
        "results": rows,
        "data_quality": {
            "real_quotes": True,
            "missing_greeks_not_filled": True,
            "missing_values": "暂无数据",
            "note": "Greeks/IV 由数据源提供；缺失时不自行估算。",
        },
    }


@router.get("/scanner/{symbol}")
def pro_options_scanner(
    symbol: str,
    strategy: str = Query("CSP"),
    dte_min: int = Query(20, ge=1, le=365),
    dte_max: int = Query(45, ge=1, le=730),
    delta_abs_min: float = Query(0.15, ge=0, le=1),
    delta_abs_max: float = Query(0.35, ge=0, le=1),
    limit: int = Query(20, ge=1, le=100),
):
    strategy = strategy.upper()
    if strategy not in {"CSP", "CC"}:
        raise HTTPException(status_code=400, detail="strategy 只能是 CSP / CC")
    typ = "put" if strategy == "CSP" else "call"
    delta_lo = -delta_abs_max if strategy == "CSP" else delta_abs_min
    delta_hi = -delta_abs_min if strategy == "CSP" else delta_abs_max
    data = pro_options_chain(
        symbol=symbol, dte_min=dte_min, dte_max=dte_max, option_type=typ,
        delta_min=delta_lo, delta_max=delta_hi, limit=500,
    )
    rows = data["results"]
    for row in rows:
        row["strategy"] = strategy
        row["strategy_score"] = _strategy_score(row, strategy)
    rows.sort(key=lambda x: (
        -(x.get("strategy_score") if x.get("strategy_score") is not None else -1),
        -(x.get("premium_yield") if x.get("premium_yield") is not None else -1),
        x["dte"],
    ))
    data["results"] = rows[:limit]
    data["strategy"] = strategy
    data["strategy_formula"] = "流动性 + 买卖价差 + 权利金收益率 + Delta目标接近度；仅用于候选排序，不代表预期收益。"
    return data
