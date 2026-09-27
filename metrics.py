import math
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SEC_UA = os.getenv(
    "SEC_USER_AGENT",
    "stock-fundamental-dashboard/2.2 contact@example.com",
)
SEC_HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
_SEC_TICKERS = None


def clean_symbol(raw: str) -> str:
    s = str(raw or "").strip().upper().replace(" ", "")
    if re.fullmatch(r"\d{6}", s):
        return s + (".SS" if s.startswith(("5", "6", "68", "9")) else ".SZ")
    if re.fullmatch(r"\d{1,5}", s):
        return s.zfill(4) + ".HK"
    # Yahoo uses hyphens for share classes such as BRK.B.
    if "." in s and re.fullmatch(r"[A-Z]{1,6}\.[A-Z]", s):
        s = s.replace(".", "-")
    return s


def is_us_symbol(symbol: str) -> bool:
    return not symbol.endswith((".HK", ".SS", ".SZ")) and bool(re.fullmatch(r"[A-Z0-9-]+", symbol))


def finite(v):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def pct(v):
    x = finite(v)
    return None if x is None else x * 100


def _names(names):
    return [names] if isinstance(names, str) else list(names)


def latest_row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in _names(names):
        if name in df.index:
            try:
                s = df.loc[name]
                for v in list(s):
                    x = finite(v)
                    if x is not None:
                        return x
            except Exception:
                pass
    return None


def series_value(df, names, col):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        for row in _names(names):
            if row in df.index:
                return finite(df.loc[row, col])
    except Exception:
        pass
    return None


def safe_ratio(a, b):
    a, b = finite(a), finite(b)
    if a is None or b in (None, 0):
        return None
    return a / b


def _safe_get(obj, attr, errors, default=None, *args, **kwargs):
    try:
        value = getattr(obj, attr)
        if callable(value):
            value = value(*args, **kwargs)
        return value
    except Exception as exc:
        errors[attr] = str(exc)[:240]
        return default


def normalize_history(df):
    if df is None or getattr(df, "empty", True):
        return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yf.download can return (Price, Ticker) columns.
        if out.columns.nlevels == 2:
            first = out.columns.get_level_values(0)
            if "Close" in first:
                out.columns = first
            else:
                out.columns = out.columns.get_level_values(-1)
    if "Close" not in out.columns:
        return None
    out = out.dropna(subset=["Close"])
    return out if not out.empty else None


def yahoo_chart_history(symbol, errors):
    """Direct Yahoo chart endpoint fallback; avoids yfinance history-cache failures."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "2y", "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        payload = r.json()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            raise RuntimeError(((payload.get("chart") or {}).get("error") or {}).get("description", "Yahoo chart returned no result"))
        result = result[0]
        ts = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        if not ts or not quote.get("close"):
            raise RuntimeError("Yahoo chart returned no daily prices")
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
        data = {"Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"), "Close": quote.get("close"), "Volume": quote.get("volume")}
        return pd.DataFrame(data, index=idx).dropna(subset=["Close"])
    except Exception as exc:
        errors["yahoo_chart"] = str(exc)[:240]
        return None


def get_history(ticker, symbol, errors):
    try:
        h = ticker.history(period="2y", interval="1d", auto_adjust=False, repair=False, raise_errors=False)
        h = normalize_history(h)
        if h is not None and len(h) >= 30:
            return h
    except Exception as exc:
        errors["history"] = str(exc)[:240]
    try:
        h = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False, repair=False)
        h = normalize_history(h)
        if h is not None and len(h) >= 30:
            return h
    except Exception as exc:
        errors["download"] = str(exc)[:240]
    return yahoo_chart_history(symbol, errors)


def annual_roe(inc, bs):
    if inc is None or inc.empty or bs is None or bs.empty:
        return []
    bs_cols = list(bs.columns)
    rows = []
    for col in list(inc.columns):
        year = getattr(col, "year", None)
        if year is None:
            continue
        net_income = series_value(inc, ["Net Income", "Net Income Common Stockholders"], col)
        equity_end = series_value(bs, ["Stockholders Equity", "Common Stock Equity", "Stockholders Equity Including Minority Interest"], col)
        prior = [c for c in bs_cols if getattr(c, "year", None) == year - 1]
        equity_begin = None
        if prior:
            equity_begin = series_value(bs, ["Stockholders Equity", "Common Stock Equity", "Stockholders Equity Including Minority Interest"], prior[0])
        if net_income is None or equity_begin is None or equity_end is None:
            continue
        avg_equity = (equity_begin + equity_end) / 2
        if avg_equity == 0:
            continue
        roe = net_income / avg_equity * 100
        if math.isfinite(roe):
            rows.append({"year": int(year), "net_income": net_income, "equity_begin": equity_begin, "equity_end": equity_end, "roe": roe})
    rows.sort(key=lambda x: x["year"])
    return rows[-15:]


def stats(values):
    vals = [finite(x) for x in values if finite(x) is not None]
    if not vals:
        return {"count": 0, "average": None, "median": None, "std_dev": None, "range": None, "max": None, "min": None}
    arr = np.array(vals, dtype=float)
    return {"count": len(vals), "average": float(arr.mean()), "median": float(np.median(arr)), "std_dev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "range": float(arr.max() - arr.min()), "max": float(arr.max()), "min": float(arr.min())}


def dividend_metrics(divs, cf, price, fcf, net_income, dividend_error=None):
    out = {
        "status": "unavailable" if dividend_error else "no_dividend",
        "ttm_dividend_per_share": 0.0 if not dividend_error else None,
        "dividend_yield": 0.0 if not dividend_error else None,
        "history": [], "cagr_3y": None, "cagr_5y": None, "cagr_10y": None,
        "consecutive_years": 0, "dividend_payout_ratio": None,
        "fcf_payout_ratio": None, "buybacks": None,
        "shareholder_payout": None, "shareholder_payout_ratio": None,
        "trend": "无现金股息" if not dividend_error else "暂无数据",
        "latest_complete_year": None,
        "latest_period": None,
    }
    if divs is not None and not getattr(divs, "empty", True):
        try:
            divs = divs.dropna().astype(float)
            if not isinstance(divs.index, pd.DatetimeIndex):
                divs.index = pd.to_datetime(divs.index)
            annual = divs.groupby(divs.index.year).sum().sort_index()
            out["status"] = "has_dividend"

            now = pd.Timestamp.now()
            current_year = int(now.year)
            complete_years = [int(y) for y in annual.index if int(y) < current_year]
            latest_complete = max(complete_years) if complete_years else None
            out["latest_complete_year"] = latest_complete
            out["history"] = [
                {"year": int(y), "dividend_per_share": float(v),
                 "complete": int(y) < current_year}
                for y, v in annual.items()
            ]

            cutoff = now - pd.Timedelta(days=365)
            ttm = float(divs[divs.index >= cutoff].sum())
            if ttm <= 0 and latest_complete is not None:
                ttm = float(annual.loc[latest_complete])
            out["ttm_dividend_per_share"] = ttm
            if price:
                out["dividend_yield"] = ttm / price * 100

            if annual.index.max() == current_year:
                out["latest_period"] = f"{current_year} YTD"
            elif annual.index.max() is not None:
                out["latest_period"] = str(int(annual.index.max()))

            # CAGR must use completed annual observations only. This prevents
            # a partial current year (e.g. 2026 YTD) from distorting CAGR.
            completed = annual.loc[annual.index.astype(int) < current_year]
            def cagr(years):
                if completed.empty:
                    return None
                end_year = int(completed.index[-1]); target = end_year - years
                eligible = [int(y) for y in completed.index if int(y) <= target]
                if not eligible:
                    return None
                start_year = max(eligible); actual = end_year - start_year
                if actual <= 0:
                    return None
                start, endv = float(completed.loc[start_year]), float(completed.loc[end_year])
                if start <= 0 or endv <= 0:
                    return None
                return ((endv / start) ** (1 / actual) - 1) * 100
            out["cagr_3y"], out["cagr_5y"], out["cagr_10y"] = cagr(3), cagr(5), cagr(10)

            count = 0
            if latest_complete is not None:
                for y in range(latest_complete, latest_complete - 30, -1):
                    if y in completed.index and float(completed.loc[y]) > 0:
                        count += 1
                    else:
                        break
            out["consecutive_years"] = count
            if out["cagr_5y"] is not None:
                out["trend"] = "上升" if out["cagr_5y"] > 1 else ("下降" if out["cagr_5y"] < -1 else "基本稳定")
        except Exception as exc:
            out["history_error"] = str(exc)[:240]

    dividends_paid = latest_row(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid", "Common Stock Payments", "Payment Of Dividends"])
    buybacks = latest_row(cf, ["Repurchase Of Capital Stock", "Repurchase Of Capital Stock Issuance", "Common Stock Payments"])
    dividends_paid = abs(dividends_paid) if dividends_paid is not None else (0.0 if out["status"] == "no_dividend" else None)
    buybacks = abs(buybacks) if buybacks is not None else None
    out["buybacks"] = buybacks
    if dividends_paid is not None and net_income not in (None, 0):
        out["dividend_payout_ratio"] = dividends_paid / abs(net_income) * 100
    if dividends_paid is not None and fcf not in (None, 0):
        out["fcf_payout_ratio"] = dividends_paid / abs(fcf) * 100
    if dividends_paid is not None or buybacks is not None:
        total = (dividends_paid or 0) + (buybacks or 0)
        out["shareholder_payout"] = total
        if fcf not in (None, 0):
            out["shareholder_payout_ratio"] = total / abs(fcf) * 100
    return out


def fibonacci_levels(history):
    out = {"available": False, "swing_high": None, "swing_low": None, "levels": {}, "nearest_support": None, "nearest_resistance": None}
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return out
    try:
        h = history.tail(252)
        close = h["Close"].dropna().astype(float)
        if len(close) < 30:
            return out
        swing_high = float(h["High"].dropna().astype(float).max()) if "High" in h else float(close.max())
        swing_low = float(h["Low"].dropna().astype(float).min()) if "Low" in h else float(close.min())
        if swing_high <= swing_low:
            return out
        current = float(close.iloc[-1])
        diff = swing_high - swing_low
        ratios = {"0.0%": 0.0, "23.6%": 0.236, "38.2%": 0.382, "50.0%": 0.5, "61.8%": 0.618, "78.6%": 0.786, "100.0%": 1.0}
        levels = {label: swing_high - diff * ratio for label, ratio in ratios.items()}
        below = [(label, value) for label, value in levels.items() if value < current]
        above = [(label, value) for label, value in levels.items() if value > current]
        support = max(below, key=lambda x: x[1]) if below else None
        resistance = min(above, key=lambda x: x[1]) if above else None
        out.update({"available": True, "swing_high": swing_high, "swing_low": swing_low,
                    "levels": levels,
                    "nearest_support": {"level": support[0], "price": support[1]} if support else None,
                    "nearest_resistance": {"level": resistance[0], "price": resistance[1]} if resistance else None})
    except Exception:
        pass
    return out


def technical_price_chart(history, fib):
    out = {"available": False, "points": [], "current": None, "support": None, "resistance": None}
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return out
    try:
        close = history["Close"].dropna().astype(float).tail(120)
        if len(close) < 10:
            return out
        support = fib.get("nearest_support") if fib else None
        resistance = fib.get("nearest_resistance") if fib else None
        out["points"] = [{"date": str(idx.date()), "price": float(v)} for idx, v in close.items()]
        out["current"] = float(close.iloc[-1])
        out["support"] = support
        out["resistance"] = resistance
        out["available"] = True
    except Exception:
        pass
    return out


def technical_analysis(history):
    out = {"score": None, "state": "暂无数据", "signals": [], "indicators": {}, "history": []}
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return out
    try:
        close = history["Close"].dropna().astype(float)
        if len(close) < 30: return out
        volume = history["Volume"].dropna().astype(float) if "Volume" in history else None
        latest = float(close.iloc[-1])
        mas = {n: (float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None) for n in (20, 60, 120, 250)}
        delta = close.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan); rsi = float((100 - 100/(1+rs)).iloc[-1]) if finite(rs.iloc[-1]) is not None else None
        ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
        macd, signal = ema12-ema26, (ema12-ema26).ewm(span=9, adjust=False).mean()
        macd_val, signal_val = float(macd.iloc[-1]), float(signal.iloc[-1])
        mid, sd = close.rolling(20).mean(), close.rolling(20).std(); upper, lower = mid+2*sd, mid-2*sd
        bb_pos = safe_ratio(latest-float(lower.iloc[-1]), float(upper.iloc[-1])-float(lower.iloc[-1])) if finite(upper.iloc[-1]) is not None and finite(lower.iloc[-1]) is not None else None
        ret20 = (latest/float(close.iloc[-21])-1)*100 if len(close)>21 else None
        vol_ratio = None
        if volume is not None and len(volume)>=20:
            av=float(volume.rolling(20).mean().iloc[-1]); vol_ratio=float(volume.iloc[-1]/av) if av else None
        high52, low52 = float(close.tail(252).max()), float(close.tail(252).min()); pos52 = (latest-low52)/(high52-low52)*100 if high52 != low52 else None
        score=50.0; signals=[]
        for n,w in [(20,10),(60,10),(120,10),(250,15)]:
            ma=mas[n]
            if ma is not None:
                score += w if latest>ma else -w; signals.append(f"MA{n}之上" if latest>ma else f"MA{n}之下")
        if rsi is not None: score += 8 if 50<=rsi<=70 else (-5 if rsi>75 else (2 if rsi<30 else -2))
        score += 8 if macd_val>signal_val else -8; signals.append("MACD强于信号线" if macd_val>signal_val else "MACD弱于信号线")
        if bb_pos is not None: score += 4 if 0.2<=bb_pos<=0.8 else (-3 if bb_pos>0.95 else 0)
        if ret20 is not None: score += max(-5,min(5,ret20/4))
        score=max(0,min(100,round(score))); state="偏强" if score>=65 else ("中性" if score>=45 else "偏弱")
        out.update({"score":score,"state":state,"signals":signals,"indicators":{"ma20":mas[20],"ma60":mas[60],"ma120":mas[120],"ma250":mas[250],"rsi14":rsi,"macd":macd_val,"macd_signal":signal_val,"bollinger_position":bb_pos,"momentum_20d":ret20,"volume_ratio_20d":vol_ratio,"52w_high":high52,"52w_low":low52,"52w_position":pos52},"history":[{"date":str(i.date()),"close":float(v)} for i,v in close.tail(120).items()]})
    except Exception:
        pass
    return out


def analyst_view(ticker):
    out={"available":False,"rating":{},"targets":{},"earnings":{},"revenue":{},"changes":[]}
    try:
        rec=ticker.get_recommendations()
        if rec is not None and not rec.empty:
            row=rec.iloc[-1]; out["rating"]={str(k):finite(v) for k,v in row.to_dict().items() if finite(v) is not None}; out["available"]=True
    except Exception: pass
    try:
        x=ticker.get_analyst_price_targets() or {}; out["targets"]={k:finite(v) for k,v in x.items() if finite(v) is not None}; out["available"]|=bool(out["targets"])
    except Exception: pass
    for attr,key,index in [("get_earnings_estimate","earnings","0y"),("get_revenue_estimate","revenue","0y")]:
        try:
            df=getattr(ticker,attr)()
            if df is not None and not df.empty:
                row=df.loc[index] if index in df.index else df.iloc[0]; out[key]={str(k):finite(v) for k,v in row.to_dict().items() if finite(v) is not None}; out["available"]=True
        except Exception: pass
    try:
        df=ticker.get_upgrades_downgrades()
        if df is not None and not df.empty:
            out["changes"]=df.tail(8).reset_index().to_dict("records"); out["available"]=True
    except Exception: pass
    return out


def sec_cik_for_ticker(ticker):
    global _SEC_TICKERS
    if _SEC_TICKERS is None:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": SEC_UA}, timeout=12)
        r.raise_for_status()
        data = r.json()
        _SEC_TICKERS = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in data.values() if v.get("ticker")}
    return _SEC_TICKERS.get(ticker.upper())


def _sec_fact(facts, tags):
    for taxonomy, tag in tags:
        unit = facts.get(taxonomy, {}).get(tag)
        if not unit:
            continue
        # Prefer USD facts; then first available unit.
        if "USD" in unit: return unit["USD"]
        first = next(iter(unit.values()), None)
        if first: return first
    return []


def sec_annual_roe(symbol, errors):
    """Build up to 15 annual ROEs from SEC Company Facts.

    SEC duration facts (net income) and instant facts (equity) are matched by
    fiscal-period end dates, not merely by the SEC ``fy`` field. This avoids
    the common Apple-style fiscal-year/context mismatch where an equity fact
    from a prior balance sheet is accidentally paired with the current year.
    """
    if not is_us_symbol(symbol):
        return None
    try:
        cik = sec_cik_for_ticker(symbol)
        if not cik:
            errors["sec"] = "ticker not found in SEC company_tickers"
            return None

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        facts = data.get("facts", {})

        def pick_units(taxonomy, tag_candidates):
            for tag in tag_candidates:
                unit_map = facts.get(taxonomy, {}).get(tag, {}).get("units", {})
                if not unit_map:
                    continue
                if "USD" in unit_map:
                    return unit_map["USD"]
                first = next(iter(unit_map.values()), None)
                if first:
                    return first
            return []

        # Prefer consolidated net income and total shareholders' equity.
        ni_items = pick_units("us-gaap", [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "NetIncomeLossAvailableToCommonStockholdersDiluted",
        ])
        eq_items = pick_units("us-gaap", [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "CommonStockholdersEquity",
            "PartnersCapital",
        ])
        if not ni_items or not eq_items:
            errors["sec"] = "SEC facts missing usable net income or equity tags"
            return []

        forms = {"10-K", "20-F", "40-F"}

        def annual_flows(items):
            """Return one duration fact per fiscal year, preserving period dates."""
            candidates = []
            for x in items:
                val = finite(x.get("val"))
                start, end = x.get("start"), x.get("end")
                form, fy = x.get("form"), x.get("fy")
                if val is None or not start or not end or form not in forms:
                    continue
                try:
                    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                except Exception:
                    continue
                # Annual 10-K duration; allow 52/53-week fiscal years.
                if not 250 <= days <= 400:
                    continue
                try:
                    year = int(fy) if fy else pd.Timestamp(end).year
                except Exception:
                    year = pd.Timestamp(end).year
                score = (
                    1 if x.get("fp") == "FY" else 0,
                    1 if form == "10-K" else 0,
                    str(x.get("filed", "")),
                )
                candidates.append((year, str(end), score, x))

            by_year = {}
            for year, end, score, x in candidates:
                old = by_year.get(year)
                # Prefer the filing with the strongest annual designation and
                # latest filing date, while retaining its exact fiscal end date.
                if old is None or score > old[0]:
                    by_year[year] = (score, end, x)
            return {y: item for y, (_, _, item) in by_year.items()}

        def instant_equity(items):
            """Return the best equity fact for each exact balance-sheet end date."""
            by_end = {}
            for x in items:
                val = finite(x.get("val"))
                end = x.get("end")
                form = x.get("form")
                if val is None or not end or form not in forms:
                    continue
                try:
                    end_date = str(pd.Timestamp(end).date())
                except Exception:
                    continue
                # Instant facts must not require a start date.
                score = (
                    1 if x.get("fp") == "FY" else 0,
                    1 if form == "10-K" else 0,
                    str(x.get("filed", "")),
                )
                old = by_end.get(end_date)
                if old is None or score > old[0]:
                    by_end[end_date] = (score, x)
            return {end: item for end, (_, item) in by_end.items()}

        ni_by_year = annual_flows(ni_items)
        eq_by_end = instant_equity(eq_items)
        eq_dates = sorted(eq_by_end.keys())

        def equity_on_or_before(target_date, max_days=450):
            """Find the latest balance-sheet equity at/before target date."""
            if not target_date:
                return None
            try:
                target = pd.Timestamp(target_date)
            except Exception:
                return None
            eligible = [d for d in eq_dates if pd.Timestamp(d) <= target]
            if not eligible:
                return None
            chosen = eligible[-1]
            if (target - pd.Timestamp(chosen)).days > max_days:
                return None
            return eq_by_end[chosen]

        rows = []
        for year in sorted(ni_by_year):
            ni = ni_by_year[year]
            n = finite(ni.get("val"))
            end = ni.get("end")
            start = ni.get("start")
            if n is None or not start or not end:
                continue

            # Exact fiscal-period matching first. Beginning equity is the
            # balance immediately preceding the fiscal-period start.
            cur = eq_by_end.get(str(pd.Timestamp(end).date()))
            prev = equity_on_or_before(start)
            if cur is None or prev is None:
                continue

            eb = finite(prev.get("val"))
            ee = finite(cur.get("val"))
            if eb is None or ee is None:
                continue
            avg_equity = (eb + ee) / 2
            if avg_equity == 0:
                continue

            rows.append({
                "year": int(year),
                "fiscal_year": f"FY{int(year)}",
                "fiscal_period_end": str(pd.Timestamp(end).date()),
                "net_income": n,
                "equity_begin": eb,
                "equity_begin_date": str(pd.Timestamp(prev.get("end")).date()) if prev.get("end") else None,
                "equity_end": ee,
                "equity_end_date": str(pd.Timestamp(cur.get("end")).date()) if cur.get("end") else str(pd.Timestamp(end).date()),
                "roe": n / avg_equity * 100,
            })

        rows.sort(key=lambda x: x["year"])
        if not rows:
            errors["sec"] = "SEC facts found, but no matching annual income/equity pairs"
            return []
        return rows[-15:]
    except Exception as exc:
        errors["sec"] = str(exc)[:240]
        return None


def build_dashboard(raw_symbol: str) -> dict[str, Any]:
    symbol=clean_symbol(raw_symbol); t=yf.Ticker(symbol); errors={}
    history=get_history(t,symbol,errors)
    price=None
    if history is not None and not history.empty:
        try: price=finite(history["Close"].dropna().iloc[-1])
        except Exception: pass
    if price is None:
        fi=_safe_get(t,"fast_info",errors,{}) or {}
        try: price=finite(fi.get("last_price"))
        except Exception: pass

    inc=_safe_get(t,"financials",errors,None); bs=_safe_get(t,"balance_sheet",errors,None); cf=_safe_get(t,"cashflow",errors,None)
    info=_safe_get(t,"info",errors,{}) or {}
    revenue=latest_row(inc,["Total Revenue","Operating Revenue"]); net_income=latest_row(inc,["Net Income","Net Income Common Stockholders"])
    gross_profit=latest_row(inc,["Gross Profit"]); gross_margin=(gross_profit/revenue*100 if gross_profit is not None and revenue not in (None,0) else pct(info.get("grossMargins")))
    assets=latest_row(bs,["Total Assets"]); liabilities=latest_row(bs,["Total Liabilities Net Minority Interest","Total Liabilities"]); equity=latest_row(bs,["Stockholders Equity","Common Stock Equity","Stockholders Equity Including Minority Interest"])
    debt_ratio=safe_ratio(liabilities,assets); debt_ratio=debt_ratio*100 if debt_ratio is not None else None
    roe_info=finite(info.get("returnOnEquity")); current_roe=roe_info*100 if roe_info is not None else (net_income/equity*100 if net_income is not None and equity not in (None,0) else None)
    ocf=latest_row(cf,["Operating Cash Flow","Total Cash From Operating Activities"]); capex=latest_row(cf,["Capital Expenditure","Capital Expenditures"]); fcf=ocf+capex if ocf is not None and capex is not None else None
    shares=latest_row(bs,["Ordinary Shares Number","Share Issued","Common Stock Shares Outstanding"])
    pe=finite(info.get("trailingPE")); pb=finite(info.get("priceToBook"))
    if pe is None and price is not None and net_income is not None and shares not in (None,0): pe=safe_ratio(price,net_income/shares)
    if pb is None and price is not None and equity is not None and shares not in (None,0): pb=safe_ratio(price,equity/shares)

    dividend_error=None
    try: divs=t.get_dividends(period="max")
    except Exception as exc: divs=None; dividend_error=str(exc)[:240]; errors["dividends"] = dividend_error
    dividends=dividend_metrics(divs,cf,price,fcf,net_income,dividend_error)
    fib=fibonacci_levels(history); tech=technical_analysis(history); tech["fibonacci"]=fib; tech["price_chart"]=technical_price_chart(history,fib); analysts=analyst_view(t)

    # US: SEC/EDGAR is authoritative for long-history ROE; fallback to Yahoo only if SEC unavailable.
    sec_roe=sec_annual_roe(symbol,errors) if is_us_symbol(symbol) else None
    if sec_roe is not None and len(sec_roe) >= 2:
        roe15=sec_roe; roe_source="SEC EDGAR / XBRL Company Facts"
        # Keep the headline ROE definition consistent with the 15-year table
        # for U.S. issuers instead of mixing Yahoo's proprietary calculation.
        current_roe=finite(roe15[-1].get("roe")) if roe15 else current_roe
    else:
        roe15=annual_roe(inc,bs); roe_source="Yahoo Finance via yfinance"

    company=info.get("longName") or info.get("shortName") or symbol
    exchange=info.get("exchange") or ""; currency=info.get("currency") or ""
    source_status={"history":bool(history is not None and not getattr(history,"empty",True)),"financials":bool(inc is not None and not getattr(inc,"empty",True)),"balance_sheet":bool(bs is not None and not getattr(bs,"empty",True)),"cashflow":bool(cf is not None and not getattr(cf,"empty",True)),"dividends": dividends["status"],"analysts":analysts["available"],"roe_long_history": len(roe15) >= 10}
    return {"query":raw_symbol,"symbol":symbol,"company":company,"exchange":exchange,"currency":currency,"market":info.get("market"),"market_data":{"price":price,"market_cap":finite(info.get("marketCap"))},
            "valuation":{"pe":pe,"pb":pb,"roe":current_roe,"roe_pb":safe_ratio(current_roe,pb),"pe_roe":safe_ratio(pe,current_roe)},
            "fundamentals":{"revenue":revenue,"net_income":net_income,"gross_margin":gross_margin,"free_cash_flow":fcf,"debt_ratio":debt_ratio,"debt_to_equity":finite(info.get("debtToEquity"))},
            "dividends":dividends,"roe_15y":{"years":roe15,"stats":stats([r["roe"] for r in roe15]),"definition":"ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)","source":roe_source},
            "technical":tech,"analysts":analysts,"source":{"provider":"免费数据源：Yahoo Finance/yfinance + SEC EDGAR（美股长期ROE）","status":source_status,"errors":errors,"note":"行情、财报、分红、技术面、分析师及长期ROE独立取数；单模块失败不会让整个页面失效。"}}
