import math
import re
from typing import Any
import numpy as np
import yfinance as yf

def clean_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if re.fullmatch(r"\d{6}", s):
        return s + (".SS" if s.startswith(("6", "68", "9")) else ".SZ")
    if re.fullmatch(r"\d{1,5}", s):
        return s.zfill(4) + ".HK"
    return s

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def pct(v):
    x = finite(v)
    return None if x is None else x * 100

def latest_row(df, name):
    if df is None or df.empty or name not in df.index:
        return None
    return finite(df.loc[name].iloc[0])

def series_value(df, row, col):
    try:
        if row not in df.index:
            return None
        return finite(df.loc[row, col])
    except Exception:
        return None

def annual_roe(ticker):
    inc = ticker.financials
    bs = ticker.balance_sheet
    if inc is None or inc.empty or bs is None or bs.empty:
        return []

    bs_cols = list(bs.columns)
    rows = []

    for col in list(inc.columns):
        year = getattr(col, "year", None)
        if year is None:
            continue

        net_income = series_value(inc, "Net Income", col)
        if net_income is None:
            net_income = series_value(inc, "Net Income Common Stockholders", col)
        if net_income is None:
            continue

        equity_end = series_value(bs, "Stockholders Equity", col)
        if equity_end is None:
            equity_end = series_value(bs, "Common Stock Equity", col)

        prior = [c for c in bs_cols if getattr(c, "year", None) == year - 1]
        equity_begin = None
        if prior:
            equity_begin = series_value(bs, "Stockholders Equity", prior[0])
            if equity_begin is None:
                equity_begin = series_value(bs, "Common Stock Equity", prior[0])

        if equity_begin is None or equity_end is None:
            continue

        avg_equity = (equity_begin + equity_end) / 2
        if avg_equity == 0:
            continue

        roe = net_income / avg_equity * 100
        if math.isfinite(roe):
            rows.append({
                "year": int(year),
                "net_income": net_income,
                "equity_begin": equity_begin,
                "equity_end": equity_end,
                "roe": roe
            })

    rows.sort(key=lambda x: x["year"])
    return rows[-15:]

def stats(values):
    vals = [finite(x) for x in values if finite(x) is not None]
    if not vals:
        return {"count":0,"average":None,"median":None,"std_dev":None,"range":None,"max":None,"min":None}
    arr = np.array(vals, dtype=float)
    return {
        "count": len(vals),
        "average": float(arr.mean()),
        "median": float(np.median(arr)),
        "std_dev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "range": float(arr.max() - arr.min()),
        "max": float(arr.max()),
        "min": float(arr.min())
    }

def safe_ratio(a, b):
    a, b = finite(a), finite(b)
    if a is None or b in (None, 0):
        return None
    return a / b

def build_dashboard(raw_symbol: str) -> dict[str, Any]:
    symbol = clean_symbol(raw_symbol)
    t = yf.Ticker(symbol)

    try:
        info = t.info or {}
    except Exception:
        info = {}

    price = finite(info.get("currentPrice") or info.get("regularMarketPrice"))
    pe = finite(info.get("trailingPE"))
    pb = finite(info.get("priceToBook"))
    roe_info = finite(info.get("returnOnEquity"))

    roe15 = annual_roe(t)
    roe_values = [r["roe"] for r in roe15]
    roe_stats = stats(roe_values)
    current_roe = roe_info * 100 if roe_info is not None else (roe15[-1]["roe"] if roe15 else None)

    revenue = finite(info.get("totalRevenue"))
    net_income = finite(info.get("netIncomeToCommon"))
    gross_margin = pct(info.get("grossMargins"))
    dividend_yield = pct(info.get("dividendYield"))
    market_cap = finite(info.get("marketCap"))

    debt_ratio = None
    debt_to_equity = finite(info.get("debtToEquity"))
    if debt_to_equity is not None:
        # Yahoo的debtToEquity不是严格意义上的负债/资产，因此V1不将它冒充为资产负债率。
        debt_ratio = None

    fcf = None
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            ocf = latest_row(cf, "Operating Cash Flow")
            capex = latest_row(cf, "Capital Expenditure")
            if ocf is not None and capex is not None:
                fcf = ocf + capex
    except Exception:
        pass

    return {
        "query": raw_symbol,
        "symbol": symbol,
        "company": info.get("longName") or info.get("shortName") or symbol,
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
        "market": info.get("market"),
        "market_data": {"price": price, "market_cap": market_cap},
        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": current_roe,
            "roe_pb": safe_ratio(current_roe, pb),
            "pe_roe": safe_ratio(pe, current_roe)
        },
        "fundamentals": {
            "revenue": revenue,
            "net_income": net_income,
            "gross_margin": gross_margin,
            "free_cash_flow": fcf,
            "debt_ratio": debt_ratio,
            "dividend_yield": dividend_yield
        },
        "roe_15y": {
            "years": roe15,
            "stats": roe_stats,
            "definition": "ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)",
            "std_definition": "15年有效年度ROE的样本标准差"
        },
        "source": {
            "provider": "Yahoo Finance via yfinance",
            "note": "V1原型；后续接入Alpha Vantage / SEC / A股及港股专用源进行交叉校验"
        }
    }
