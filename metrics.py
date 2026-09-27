from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf


def clean_number(value):
    """
    将 pandas / numpy / NaN / inf
    转换为 JSON 可以接受的数字。
    """

    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def safe_div(a, b):
    """
    安全除法。
    """

    a = clean_number(a)
    b = clean_number(b)

    if a is None or b is None or b == 0:
        return None

    return a / b


def normalize_symbol(symbol: str) -> tuple[str, str]:
    """
    自动识别常见 A 股、港股、美股代码。

    A股:
        600519
        000001
        300750

    港股:
        0700
        700
        9988
        0700.HK

    美股:
        AAPL
        MSFT
        BRK.B
    """

    original = symbol.strip().upper()

    if not original:
        raise ValueError("股票代码不能为空")

    if original.endswith(".HK"):
        return original, "港股"

    if original.endswith(".SS"):
        return original, "A股"

    if original.endswith(".SZ"):
        return original, "A股"

    if original.isdigit():

        if len(original) <= 5:
            padded = original.zfill(4)
            return f"{padded}.HK", "港股"

        if len(original) == 6:

            if original.startswith(
                ("600", "601", "603", "605", "688")
            ):
                return f"{original}.SS", "A股"

            if original.startswith(
                ("000", "001", "002", "003", "300", "301")
            ):
                return f"{original}.SZ", "A股"

            return f"{original}.SS", "A股"

    return original, "美股"


def get_row(df, names):
    """
    从财务报表中寻找指定项目。
    """

    if df is None or df.empty:
        return None

    for name in names:

        if name in df.index:
            return df.loc[name]

    return None


def calculate_dividend_yield(ticker, price):
    """
    使用最近12个月实际分红记录计算 TTM 股息率。

    公式：

    TTM股息率 =
    最近12个月每股实际分红合计
    ÷
    当前股价
    × 100

    注意：
    不使用 Yahoo Finance 的 dividendYield 字段。
    不使用 dividendRate 字段。

    这样可以避免 Yahoo 字段单位变化
    导致股息率出现 32% 之类的问题。
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

        # 确保索引是时间格式
        try:
            dividends.index = pd.to_datetime(
                dividends.index
            )
        except Exception:
            return None

        latest_date = dividends.index.max()

        if pd.isna(latest_date):
            return None

        # 最近12个月
        one_year_ago = (
            latest_date
            -
            pd.Timedelta(days=365)
        )

        recent_dividends = dividends[
            dividends.index > one_year_ago
        ]

        if recent_dividends.empty:
            return None

        # 最近12个月实际每股分红
        ttm_dividend = clean_number(
            recent_dividends.sum()
        )

        if (
            ttm_dividend is None
            or ttm_dividend < 0
        ):
            return None

        # 核心公式
        dividend_yield = (
            ttm_dividend
            /
            price
        ) * 100

        if not math.isfinite(dividend_yield):
            return None

        return dividend_yield

    except Exception:
        return None


def build_current_metrics(ticker):
    """
    获取当前估值及基本面。
    """

    info = {}

    try:
        info = ticker.info

    except Exception:
        info = {}

    # ==============================
    # 当前价格
    # ==============================

    price = clean_number(
        info.get("currentPrice")
        or
        info.get("regularMarketPrice")
    )

    # ==============================
    # 市值
    # ==============================

    market_cap = clean_number(
        info.get("marketCap")
    )

    # ==============================
    # PE
    # ==============================

    pe = clean_number(
        info.get("trailingPE")
    )

    # ==============================
    # PB
    # ==============================

    pb = clean_number(
        info.get("priceToBook")
    )

    # ==============================
    # 当前ROE
    # ==============================

    roe = clean_number(
        info.get("returnOnEquity")
    )

    if roe is not None:
        roe *= 100

    # ==============================
    # 营收
    # ==============================

    revenue = clean_number(
        info.get("totalRevenue")
    )

    # ==============================
    # 净利润
    # ==============================

    net_income = clean_number(
        info.get("netIncomeToCommon")
    )

    # ==============================
    # 毛利率
    # ==============================

    gross_margin = clean_number(
        info.get("grossMargins")
    )

    if gross_margin is not None:
        gross_margin *= 100

    # ==============================
    # 自由现金流
    # ==============================

    free_cash_flow = clean_number(
        info.get("freeCashflow")
    )

    # ==============================
    # 股息率
    # ==============================

    # 完全不使用：
    #
    # info["dividendYield"]
    #
    # 也不使用：
    #
    # info["dividendRate"]
    #
    # 直接从最近12个月实际分红计算。

    dividend_yield = calculate_dividend_yield(
        ticker,
        price
    )

    # ==============================
    # 负债率
    # ==============================

    debt_ratio = None

    total_assets = clean_number(
        info.get("totalAssets")
    )

    total_debt = clean_number(
        info.get("totalDebt")
    )

    if (
        total_assets is not None
        and total_assets != 0
        and total_debt is not None
    ):

        debt_ratio = safe_div(
            total_debt,
            total_assets
        )

        if debt_ratio is not None:
            debt_ratio *= 100

    return {

        "market_data": {

            "price": price,

            "market_cap": market_cap

        },

        "valuation": {

            "pe": pe,

            "pb": pb,

            "roe": roe,

            "roe_pb": safe_div(
                roe,
                pb
            ),

            "pe_roe": safe_div(
                pe,
                roe
            )

        },

        "fundamentals": {

            "revenue": revenue,

            "net_income": net_income,

            "gross_margin": gross_margin,

            "free_cash_flow": free_cash_flow,

            "dividend_yield": dividend_yield,

            "debt_ratio": debt_ratio

        }

    }


def build_roe_history(ticker):
    """
    计算年度 ROE。

    ROE =
    年度净利润
    /
    ((期初股东权益 + 期末股东权益) / 2)

    只有同时存在：

    1. 当年净利润
    2. 当年期末股东权益
    3. 上一年期末股东权益

    才计算该年度 ROE。
    """

    income = None
    balance = None

    # ==============================
    # 获取利润表
    # ==============================

    try:

        income = ticker.get_income_stmt(
            freq="yearly"
        )

    except Exception:

        try:
            income = ticker.income_stmt

        except Exception:
            income = None

    # ==============================
    # 获取资产负债表
    # ==============================

    try:

        balance = ticker.get_balance_sheet(
            freq="yearly"
        )

    except Exception:

        try:
            balance = ticker.balance_sheet

        except Exception:
            balance = None

    if (
        income is None
        or income.empty
        or balance is None
        or balance.empty
    ):
        return empty_roe_result()

    # ==============================
    # 找净利润
    # ==============================

    net_income_row = get_row(
        income,
        [
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests"
        ]
    )

    # ==============================
    # 找股东权益
    # ==============================

    equity_row = get_row(
        balance,
        [
            "Stockholders Equity",
            "Stockholders' Equity",
            "Total Stockholder Equity",
            "Common Stock Equity"
        ]
    )

    if (
        net_income_row is None
        or equity_row is None
    ):
        return empty_roe_result()

    # ==============================
    # 建立年度净利润数据
    # ==============================

    income_map = {}

    for col in net_income_row.index:

        try:
            year = pd.Timestamp(col).year

        except Exception:
            continue

        value = clean_number(
            net_income_row[col]
        )

        if value is not None:
            income_map[year] = value

    # ==============================
    # 建立年度股东权益数据
    # ==============================

    equity_map = {}

    for col in equity_row.index:

        try:
            year = pd.Timestamp(col).year

        except Exception:
            continue

        value = clean_number(
            equity_row[col]
        )

        if value is not None:
            equity_map[year] = value

    # ==============================
    # 找共同年份
    # ==============================

    years = sorted(
        set(income_map.keys())
        &
        set(equity_map.keys())
    )

    rows = []

    # ==============================
    # 计算年度ROE
    # ==============================

    for year in years:

        beginning_equity = equity_map.get(
            year - 1
        )

        ending_equity = equity_map.get(
            year
        )

        net_income = income_map.get(
            year
        )

        if (
            beginning_equity is None
            or ending_equity is None
            or net_income is None
        ):
            continue

        average_equity = (
            beginning_equity
            +
            ending_equity
        ) / 2

        if average_equity == 0:
            continue

        roe = (
            net_income
            /
            average_equity
        ) * 100

        if not math.isfinite(roe):
            continue

        rows.append({

            "year": year,

            "net_income": net_income,

            "equity_begin":
                beginning_equity,

            "equity_end":
                ending_equity,

            "roe": roe

        })

    # ==============================
    # 最近15个有效年度
    # ==============================

    rows = rows[-15:]

    values = [

        float(row["roe"])

        for row in rows

        if row.get("roe") is not None

    ]

    # ==============================
    # ROE统计
    # ==============================

    if values:

        average = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        if len(values) >= 2:

            std_dev = float(
                np.std(
                    values,
                    ddof=1
                )
            )

        else:

            std_dev = None

        maximum = float(
            np.max(values)
        )

        minimum = float(
            np.min(values)
        )

        value_range = (
            maximum
            -
            minimum
        )

    else:

        average = None

        median = None

        std_dev = None

        maximum = None

        minimum = None

        value_range = None

    return {

        "years": rows,

        "stats": {

            "count":
                len(values),

            "average":
                average,

            "median":
                median,

            "std_dev":
                std_dev,

            "range":
                value_range,

            "max":
                maximum,

            "min":
                minimum

        },

        "definition":
            "ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)",

        "std_definition":
            "15年有效年度ROE的样本标准差"

    }


def empty_roe_result():

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

        "definition":
            "ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)",

        "std_definition":
            "15年有效年度ROE的样本标准差"

    }


def build_dashboard(symbol: str):

    # ==============================
    # 自动识别市场
    # ==============================

    yahoo_symbol, exchange = normalize_symbol(
        symbol
    )

    # ==============================
    # 创建 Yahoo Finance 对象
    # ==============================

    ticker = yf.Ticker(
        yahoo_symbol
    )

    # ==============================
    # 当前基本面
    # ==============================

    current = build_current_metrics(
        ticker
    )

    # ==============================
    # 15年 ROE
    # ==============================

    roe_15y = build_roe_history(
        ticker
    )

    # ==============================
    # 公司信息
    # ==============================

    info = {}

    try:

        info = ticker.info

    except Exception:

        info = {}

    company = (

        info.get("longName")

        or

        info.get("shortName")

        or

        symbol.upper()

    )

    # ==============================
    # 返回完整数据
    # ==============================

    return {

        "query":
            symbol.upper(),

        "symbol":
            yahoo_symbol,

        "company":
            company,

        "exchange":
            exchange,

        "currency":
            info.get("currency"),

        "market":
            info.get("market"),

        "market_data":
            current["market_data"],

        "valuation":
            current["valuation"],

        "fundamentals":
            current["fundamentals"],

        "roe_15y":
            roe_15y,

        "source": {

            "provider":
                "Yahoo Finance via yfinance",

            "note":
                "V1原型；股息率使用最近12个月实际分红计算；后续接入Alpha Vantage / SEC / A股及港股专用源进行交叉校验"

        }

    }
