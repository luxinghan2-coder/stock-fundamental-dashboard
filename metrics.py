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


def latest_row(df, names):
    if df is None or df.empty:
        return None
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in df.index:
            try:
                return finite(df.loc[name].iloc[0])
            except Exception:
                pass
    return None


def series_value(df, names, col):
    if isinstance(names, str):
        names = [names]
    try:
        for row in names:
            if row in df.index:
                return finite(df.loc[row, col])
    except Exception:
        pass
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
        net_income = series_value(inc, ["Net Income", "Net Income Common Stockholders"], col)
        equity_end = series_value(bs, ["Stockholders Equity", "Common Stock Equity"], col)
        prior = [c for c in bs_cols if getattr(c, "year", None) == year - 1]
        equity_begin = None
        if prior:
            equity_begin = series_value(bs, ["Stockholders Equity", "Common Stock Equity"], prior[0])
        if net_income is None or equity_begin is None or equity_end is None:
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
                "roe": roe,
            })
    rows.sort(key=lambda x: x["year"])
    return rows[-15:]


def stats(values):
    vals = [finite(x) for x in values if finite(x) is not None]
    if not vals:
        return {"count": 0, "average": None, "median": None, "std_dev": None,
                "range": None, "max": None, "min": None}
    arr = np.array(vals, dtype=float)
    return {
        "count": len(vals),
        "average": float(arr.mean()),
        "median": float(np.median(arr)),
        "std_dev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "range": float(arr.max() - arr.min()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def safe_ratio(a, b):
    a, b = finite(a), finite(b)
    if a is None or b in (None, 0):
        return None
    return a / b


def _annual_values(df, names):
    out = []
    if df is None or df.empty:
        return out
    for col in df.columns:
        year = getattr(col, "year", None)
        if year is None:
            continue
        value = series_value(df, names, col)
        if value is not None:
            out.append((int(year), value))
    return sorted(out)


def _last_value(df, names):
    return latest_row(df, names)


def dividend_metrics(ticker, price, fcf, net_income):
    result = {
        "ttm_dividend_per_share": None,
        "dividend_yield": None,
        "history": [],
        "cagr_3y": None,
        "cagr_5y": None,
        "cagr_10y": None,
        "consecutive_years": 0,
        "dividend_payout_ratio": safe_ratio(None, None),
        "fcf_payout_ratio": safe_ratio(None, None),
        "buybacks": None,
        "shareholder_payout": None,
        "shareholder_payout_ratio": None,
        "trend": "暂无数据",
    }
    try:
        divs = ticker.get_dividends(period="max")
        if divs is None or divs.empty:
            return result
        divs = divs.dropna()
        annual = divs.groupby(divs.index.year).sum().astype(float)
        result["history"] = [{"year": int(y), "dividend_per_share": float(v)} for y, v in annual.items()]
        if len(annual):
            last_year = annual.index[-1]
            recent = annual.loc[annual.index >= last_year - 0]
            # TTM: use the most recent four quarters when possible; for annual-only display use latest full year.
            result["ttm_dividend_per_share"] = float(recent.iloc[-1])
            if price:
                result["dividend_yield"] = result["ttm_dividend_per_share"] / price * 100

        def cagr(years):
            if len(annual) < years + 1:
                return None
            end_year = annual.index[-1]
            start_year = end_year - years
            if start_year not in annual.index:
                eligible = [y for y in annual.index if y <= start_year]
                if not eligible:
                    return None
                start_year = eligible[-1]
                years_actual = end_year - start_year
            else:
                years_actual = years
            start, end = float(annual.loc[start_year]), float(annual.loc[end_year])
            if start <= 0 or end < 0 or years_actual <= 0:
                return None
            return ((end / start) ** (1 / years_actual) - 1) * 100

        result["cagr_3y"] = cagr(3)
        result["cagr_5y"] = cagr(5)
        result["cagr_10y"] = cagr(10)
        positive_years = [y for y, v in annual.items() if v > 0]
        if positive_years:
            last = annual.index[-1]
            count = 0
            for y in range(last, last - len(annual) - 1, -1):
                if y in annual.index and annual.loc[y] > 0:
                    count += 1
                else:
                    break
            result["consecutive_years"] = count
        if result["cagr_5y"] is not None:
            result["trend"] = "上升" if result["cagr_5y"] > 1 else ("下降" if result["cagr_5y"] < -1 else "基本稳定")

        # Annual cash-flow payout ratios, using the latest available cash-flow period.
        cf = ticker.cashflow
        dividends_paid = _last_value(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid", "Common Stock Payments"])
        buybacks = _last_value(cf, ["Repurchase Of Capital Stock", "Repurchase Of Capital Stock"])
        if dividends_paid is not None:
            dividends_paid = abs(dividends_paid)
        if buybacks is not None:
            buybacks = abs(buybacks)
        result["buybacks"] = buybacks
        if dividends_paid is not None and net_income not in (None, 0):
            result["dividend_payout_ratio"] = dividends_paid / abs(net_income) * 100
        if dividends_paid is not None and fcf not in (None, 0):
            result["fcf_payout_ratio"] = dividends_paid / abs(fcf) * 100
        if dividends_paid is not None or buybacks is not None:
            total = (dividends_paid or 0) + (buybacks or 0)
            result["shareholder_payout"] = total
            if fcf not in (None, 0):
                result["shareholder_payout_ratio"] = total / abs(fcf) * 100
    except Exception:
        pass
    return result


def technical_analysis(ticker):
    out = {"score": None, "state": "暂无数据", "signals": [], "indicators": {}, "history": []}
    try:
        h = ticker.history(period="2y", interval="1d", auto_adjust=False)
        if h is None or h.empty or "Close" not in h:
            return out
        close = h["Close"].astype(float)
        volume = h["Volume"].astype(float) if "Volume" in h else None
        latest = float(close.iloc[-1])
        ma20, ma60, ma120, ma250 = [float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None for n in (20,60,120,250)]
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if rs.notna().any() else None
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val, signal_val = float(macd.iloc[-1]), float(signal.iloc[-1])
        mid = close.rolling(20).mean()
        sd = close.rolling(20).std()
        upper, lower = mid + 2*sd, mid - 2*sd
        bb_pos = float((latest - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if pd_valid(upper.iloc[-1]) and pd_valid(lower.iloc[-1]) and upper.iloc[-1] != lower.iloc[-1] else None
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else None
        vol_ratio = None
        if volume is not None and len(volume) >= 20:
            avg20 = float(volume.rolling(20).mean().iloc[-1])
            if avg20:
                vol_ratio = float(volume.iloc[-1] / avg20)
        high52 = float(close.tail(252).max()) if len(close) else None
        low52 = float(close.tail(252).min()) if len(close) else None
        pos52 = float((latest-low52)/(high52-low52)*100) if high52 is not None and low52 is not None and high52 != low52 else None

        score = 50.0
        signals = []
        for ma, weight, label in [(ma20, 10, "MA20"),(ma60, 10, "MA60"),(ma120, 10, "MA120"),(ma250, 15, "MA250")]:
            if ma is not None:
                if latest > ma:
                    score += weight
                    signals.append(f"{label}之上")
                else:
                    score -= weight
                    signals.append(f"{label}之下")
        if rsi is not None:
            if 50 <= rsi <= 70: score += 8
            elif rsi > 75: score -= 5
            elif rsi < 30: score += 2
            else: score -= 2
        if macd_val > signal_val: score += 8; signals.append("MACD金叉/强于信号线")
        else: score -= 8; signals.append("MACD弱于信号线")
        if bb_pos is not None:
            if 0.2 <= bb_pos <= 0.8: score += 4
            elif bb_pos > 0.95: score -= 3
        if ret20 is not None:
            score += max(-5, min(5, ret20 / 4))
        score = max(0, min(100, round(score)))
        state = "偏强" if score >= 65 else ("中性" if score >= 45 else "偏弱")
        out.update({
            "score": score, "state": state, "signals": signals,
            "indicators": {"ma20": ma20, "ma60": ma60, "ma120": ma120, "ma250": ma250,
                           "rsi14": rsi, "macd": macd_val, "macd_signal": signal_val,
                           "bollinger_position": bb_pos, "momentum_20d": ret20,
                           "volume_ratio_20d": vol_ratio, "52w_high": high52, "52w_low": low52, "52w_position": pos52},
            "history": [{"date": str(i.date()), "close": float(v)} for i, v in close.tail(120).items()]
        })
    except Exception:
        pass
    return out


def pd_valid(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def analyst_view(ticker):
    out = {"available": False, "rating": {}, "targets": {}, "earnings": {}, "revenue": {}, "changes": []}
    try:
        rec = ticker.recommendations
        if rec is not None and not rec.empty:
            last = rec.iloc[-1]
            out["rating"] = {str(k): finite(v) for k, v in last.to_dict().items() if finite(v) is not None}
            out["available"] = True
    except Exception:
        pass
    try:
        targets = ticker.get_analyst_price_targets() or {}
        out["targets"] = {k: finite(v) for k, v in targets.items() if finite(v) is not None}
        out["available"] = out["available"] or bool(out["targets"])
    except Exception:
        pass
    try:
        ee = ticker.earnings_estimate
        if ee is not None and not ee.empty:
            row = ee.loc["0y"] if "0y" in ee.index else ee.iloc[0]
            out["earnings"] = {str(k): finite(v) for k, v in row.to_dict().items() if finite(v) is not None}
            out["available"] = True
    except Exception:
        pass
    try:
        re_ = ticker.revenue_estimate
        if re_ is not None and not re_.empty:
            row = re_.loc["0y"] if "0y" in re_.index else re_.iloc[0]
            out["revenue"] = {str(k): finite(v) for k, v in row.to_dict().items() if finite(v) is not None}
            out["available"] = True
    except Exception:
        pass
    try:
        changes = ticker.upgrades_downgrades
        if changes is not None and not changes.empty:
            cols = [c for c in ["Firm", "firm", "FromGrade", "fromGrade", "ToGrade", "toGrade", "Action", "action"] if c in changes.columns]
            out["changes"] = changes.tail(8).reset_index().to_dict("records") if not cols else changes.tail(8)[cols].reset_index().to_dict("records")
            out["available"] = True
    except Exception:
        pass
    return out


def build_dashboard(raw_symbol: str) -> dict[str, Any]:
    symbol = clean_symbol(raw_symbol)
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    try:
        h1 = t.history(period="5d", interval="1d", auto_adjust=False)
        price = finite(h1["Close"].dropna().iloc[-1]) if h1 is not None and not h1.empty else None
    except Exception:
        price = None
    if price is None:
        price = finite(info.get("currentPrice") or info.get("regularMarketPrice"))

    inc = t.financials
    bs = t.balance_sheet
    cf = t.cashflow
    revenue = _last_value(inc, ["Total Revenue", "Operating Revenue"])
    net_income = _last_value(inc, ["Net Income", "Net Income Common Stockholders"])
    gross_profit = _last_value(inc, ["Gross Profit"])
    gross_margin = gross_profit / revenue * 100 if gross_profit is not None and revenue not in (None, 0) else pct(info.get("grossMargins"))
    total_assets = _last_value(bs, ["Total Assets"])
    total_liabilities = _last_value(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
    debt_ratio = total_liabilities / total_assets * 100 if total_liabilities is not None and total_assets not in (None, 0) else None
    equity = _last_value(bs, ["Stockholders Equity", "Common Stock Equity"])
    roe_info = finite(info.get("returnOnEquity"))
    current_roe = roe_info * 100 if roe_info is not None else (net_income / equity * 100 if net_income is not None and equity not in (None, 0) else None)

    fcf = None
    ocf = _last_value(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _last_value(cf, ["Capital Expenditure", "Capital Expenditures"])
    if ocf is not None and capex is not None:
        fcf = ocf + capex

    pe = finite(info.get("trailingPE"))
    pb = finite(info.get("priceToBook"))
    if pe is None and price is not None and net_income is not None:
        shares = _last_value(bs, ["Ordinary Shares Number", "Share Issued"])
        eps = safe_ratio(net_income, shares)
        pe = safe_ratio(price, eps)
    if pb is None and price is not None and equity is not None:
        shares = _last_value(bs, ["Ordinary Shares Number", "Share Issued"])
        bvps = safe_ratio(equity, shares)
        pb = safe_ratio(price, bvps)

    roe15 = annual_roe(t)
    roe_values = [r["roe"] for r in roe15]
    roe_stats = stats(roe_values)
    dividends = dividend_metrics(t, price, fcf, net_income)
    tech = technical_analysis(t)
    analysts = analyst_view(t)

    return {
        "query": raw_symbol,
        "symbol": symbol,
        "company": info.get("longName") or info.get("shortName") or symbol,
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
        "market": info.get("market"),
        "market_data": {"price": price, "market_cap": finite(info.get("marketCap"))},
        "valuation": {"pe": pe, "pb": pb, "roe": current_roe,
                      "roe_pb": safe_ratio(current_roe, pb), "pe_roe": safe_ratio(pe, current_roe)},
        "fundamentals": {"revenue": revenue, "net_income": net_income, "gross_margin": gross_margin,
                         "free_cash_flow": fcf, "debt_ratio": debt_ratio,
                         "debt_to_equity": finite(info.get("debtToEquity"))},
        "dividends": dividends,
        "roe_15y": {"years": roe15, "stats": roe_stats,
                    "definition": "ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)",
                    "std_definition": "15年有效年度ROE的样本标准差"},
        "technical": tech,
        "analysts": analysts,
        "source": {"provider": "Yahoo Finance via yfinance（免费原型数据源）",
                   "note": "V2：分红、技术指标、部分财务比率本地计算；分析师数据仅在源站提供时显示，不人为填充。"}
    }
