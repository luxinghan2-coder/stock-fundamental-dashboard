import math
from statistics import median, stdev

import pandas as pd
import yfinance as yf


def clean_number(value):
    """安全转换为数字。"""
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def safe_div(a, b):
    """安全除法。"""
    a = clean_number(a)
    b = clean_number(b)

    if a is None or b is None or b == 0:
        return None

    result = a / b

    if not math.isfinite(result):
        return None

    return result


def normalize_symbol(symbol):
    """标准化股票代码。"""
    symbol = str(symbol or "").strip().upper()

    # 常见 A 股代码
    if symbol.isdigit():
        if len(symbol) == 6:
            return symbol + ".SS" if symbol.startswith(("6", "9")) else symbol + ".SZ"

    # 港股
    if symbol.isdigit() and len(symbol) <= 5:
        return symbol.zfill(4) + ".HK"

    return symbol


def get_row(df, names):
    """从财务报表中寻找指定项目。"""
    if df is None or df.empty:
        return None

    for name in names:
        if name in df.index:
            try:
                row = df.loc[name]
                return row
            except Exception:
                pass

    return None


def calculate_dividend_yield(ticker, price):
    """
    使用最近12个月实际分红记录计算 TTM 股息率。

    公式：

    TTM股息率 =
    最近12个月实际每股分红合计
    ÷ 当前股价
    × 100

    注意：
    不使用 Yahoo 的 dividendYield 字段，
    避免出现 32% 这类单位错误。
    """

    price = clean_number(price)

    if price is None or price <= 0:
        return None

    try:
        dividends = ticker.dividends

        if dividends is None:
            return None

        if dividends.empty:
            return None

        dividends = dividends.dropna()

        if len(dividends) == 0:
            return None

        try:
            dividends.index = pd.to_datetime(dividends.index)
        except Exception:
            return None

        latest_date = dividends.index.max()

        if pd.isna(latest_date):
            return None

        one_year_ago = latest_date - pd.Timedelta(days=365)

        recent_dividends = dividends[
            dividends.index > one_year_ago
        ]

        if recent_dividends.empty:
            return None

        ttm_dividend = clean_number(
            recent_dividends.sum()
        )

        if ttm_dividend is None:
            return None

        if ttm_dividend < 0:
            return None

        dividend_yield = (
            ttm_dividend / price
        ) * 100

        if not math.isfinite(dividend_yield):
            return None

        return dividend_yield

    except Exception:
        return None


def build_current_metrics(ticker):
    """
    获取当前股票基本面数据。
    """

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # 当前价格
    price = clean_number(
        info.get("currentPrice")
    )

    if price is None:
        price = clean_number(
            info.get("regularMarketPrice")
        )

    # 公司信息
    company = (
        info.get("longName")
        or info.get("shortName")
        or ticker.ticker
    )

    exchange = (
        info.get("exchange")
        or info.get("fullExchangeName")
    )

    currency = (
        info.get("currency")
        or ""
    )

    # 市值
    market_cap = clean_number(
        info.get("marketCap")
    )

    # PE
    pe = clean_number(
        info.get("trailingPE")
    )

    if pe is None:
        pe = clean_number(
            info.get("forwardPE")
        )

    # PB
    pb = clean_number(
        info.get("priceToBook")
    )

    # ROE
    roe_raw = clean_number(
        info.get("returnOnEquity")
    )

    roe = None

    if roe_raw is not None:
        roe = roe_raw * 100

    # ROE / PB
    roe_pb = safe_div(
        roe,
        pb
    )

    # PE / ROE
    pe_roe = safe_div(
        pe,
        roe
    )

    # 营收
    revenue = clean_number(
        info.get("totalRevenue")
    )

    # 净利润
    net_income = clean_number(
        info.get("netIncomeToCommon")
    )

    if net_income is None:
        net_income = clean_number(
            info.get("netIncome")
        )

    # 毛利率
    gross_margin_raw = clean_number(
        info.get("grossMargins")
    )

    gross_margin = None

    if gross_margin_raw is not None:
        gross_margin = gross_margin_raw * 100

    # 自由现金流
    free_cash_flow = clean_number(
        info.get("freeCashflow")
    )

    # 负债率
    debt_ratio = None

    total_debt = clean_number(
        info.get("totalDebt")
    )

    total_assets = clean_number(
        info.get("totalAssets")
    )

    if (
        total_debt is not None
        and total_assets is not None
        and total_assets != 0
    ):
        debt_ratio = (
            total_debt / total_assets
        ) * 100

    # 股息率
    # 只使用实际分红记录计算
    dividend_yield = calculate_dividend_yield(
        ticker,
        price
    )

    return {
        "price": price,
        "market_cap": market_cap,
        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "roe_pb": roe_pb,
            "pe_roe": pe_roe
        },
        "fundamentals": {
            "revenue": revenue,
            "net_income": net_income,
            "gross_margin": gross_margin,
            "free_cash_flow": free_cash_flow,
            "dividend_yield": dividend_yield,
            "debt_ratio": debt_ratio
        },
        "company": company,
        "exchange": exchange,
        "currency": currency
    }


def empty_roe_result():
    """没有足够历史数据时返回标准空结构。"""

    return {
        "years": [],
        "stats": {
            "count": 0,
            "average": None,
            "median": None,
            "std_dev": None,
            "range": None,
            "max": None,
            "min": None
        },
        "definition": (
            "ROE = 年度净利润 / "
            "((期初股东权益 + 期末股东权益) / 2)"
        ),
        "std_definition": "15年有效年度ROE的样本标准差"
    }


def build_roe_history(ticker):
    """
    根据年度财务报表计算历史 ROE。

    ROE =
    年度净利润
    ÷
    ((期初股东权益 + 期末股东权益) / 2)

    最多返回最近15个有效年度。
    """

    try:
        income = ticker.income_stmt
    except Exception:
        income = None

    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = None

    if income is None or balance is None:
        return empty_roe_result()

    if income.empty or balance.empty:
        return empty_roe_result()

    # 净利润
    net_income_row = get_row(
        income,
        [
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders"
        ]
    )

    # 股东权益
    equity_row = get_row(
        balance,
        [
            "Stockholders Equity",
            "StockholdersEquity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
    )

    if net_income_row is None or equity_row is None:
        return empty_roe_result()

    try:
        dates = list(income.columns)
    except Exception:
        return empty_roe_result()

    records = []

    # 按日期排序
    try:
        dates = sorted(
            dates,
            key=lambda x: pd.Timestamp(x)
        )
    except Exception:
        pass

    for current_date in dates:

        try:
            current_ts = pd.Timestamp(current_date)
            year = current_ts.year
        except Exception:
            continue

        try:
            net_income = clean_number(
                net_income_row.get(current_date)
            )
        except Exception:
            net_income = None

        if net_income is None:
            continue

        # 当前年度期末权益
        try:
            equity_end = clean_number(
                equity_row.get(current_date)
            )
        except Exception:
            equity_end = None

        if equity_end is None:
            continue

        # 找前一个年度的权益
        previous_dates = [
            d for d in dates
            if pd.Timestamp(d) < current_ts
        ]

        if not previous_dates:
            continue

        previous_date = previous_dates[-1]

        try:
            equity_begin = clean_number(
                equity_row.get(previous_date)
            )
        except Exception:
            equity_begin = None

        if equity_begin is None:
            continue

        average_equity = (
            equity_begin + equity_end
        ) / 2

        if average_equity == 0:
            continue

        roe = (
            net_income /
            average_equity
        ) * 100

        if not math.isfinite(roe):
            continue

        records.append(
            {
                "year": year,
                "net_income": net_income,
                "equity_begin": equity_begin,
                "equity_end": equity_end,
                "roe": roe
            }
        )

    if not records:
        return empty_roe_result()

    # 最近15年
    records = records[-15:]

    roe_values = [
        x["roe"]
        for x in records
        if x.get("roe") is not None
    ]

    if not roe_values:
        return empty_roe_result()

    average_roe = sum(roe_values) / len(roe_values)

    median_roe = median(roe_values)

    if len(roe_values) >= 2:
        std_dev = stdev(roe_values)
    else:
        std_dev = None

    roe_max = max(roe_values)
    roe_min = min(roe_values)
    roe_range = roe_max - roe_min

    return {
        "years": records,
        "stats": {
            "count": len(roe_values),
            "average": average_roe,
            "median": median_roe,
            "std_dev": std_dev,
            "range": roe_range,
            "max": roe_max,
            "min": roe_min
        },
        "definition": (
            "ROE = 年度净利润 / "
            "((期初股东权益 + 期末股东权益) / 2)"
        ),
        "std_definition": "15年有效年度ROE的样本标准差"
    }


def build_dashboard(symbol):
    """
    构建完整股票基本面驾驶舱数据。
    """

    original_symbol = str(symbol or "").strip()

    if not original_symbol:
        raise ValueError("请输入股票代码")

    normalized_symbol = normalize_symbol(
        original_symbol
    )

    try:
        ticker = yf.Ticker(
            normalized_symbol
        )
    except Exception as exc:
        raise ValueError(
            f"无法创建股票对象：{exc}"
        )

    current = build_current_metrics(
        ticker
    )

    roe_history = build_roe_history(
        ticker
    )

    return {
        "query": original_symbol,
        "symbol": normalized_symbol,
        "company": current["company"],
        "exchange": current["exchange"],
        "currency": current["currency"],

        "market": {
            "price": current["price"],
            "market_cap": current["market_cap"]
        },

        "valuation": current["valuation"],

        "fundamentals": current["fundamentals"],

        "roe_15y": roe_history,

        "source": {
            "provider": "Yahoo Finance via yfinance",
            "note": (
                "V1原型；后续接入 Alpha Vantage / SEC "
                "/ A股及港股专用数据源进行交叉校验"
            )
        }
    }
