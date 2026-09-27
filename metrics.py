import math
from statistics import median, stdev

import pandas as pd
import yfinance as yf


def clean_number(value):
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
    a = clean_number(a)
    b = clean_number(b)

    if a is None or b is None or b == 0:
        return None

    result = a / b

    return result if math.isfinite(result) else None


def normalize_symbol(symbol):
    symbol = str(symbol or "").strip().upper()

    # A股
    if symbol.isdigit():
        if len(symbol) == 6:
            if symbol.startswith(("6", "9")):
                return symbol + ".SS"
            return symbol + ".SZ"

        # 港股
        if len(symbol) <= 5:
            return symbol.zfill(4) + ".HK"

    return symbol


def find_row(df, names):
    if df is None or df.empty:
        return None

    for name in names:
        if name in df.index:
            return df.loc[name]

    return None


def get_latest_value(row):
    if row is None:
        return None

    try:
        values = row.dropna()

        if len(values) == 0:
            return None

        return clean_number(values.iloc[0])

    except Exception:
        return None


def get_latest_price(ticker):
    """
    不使用 ticker.info。
    直接从 history 获取最新价格。
    """

    try:
        history = ticker.history(
            period="5d",
            auto_adjust=False
        )

        if history is None or history.empty:
            return None

        close = history["Close"].dropna()

        if close.empty:
            return None

        return clean_number(close.iloc[-1])

    except Exception:
        return None


def get_price_history(ticker):
    try:
        history = ticker.history(
            period="1y",
            auto_adjust=False
        )

        if history is None or history.empty:
            return None

        return history

    except Exception:
        return None


def calculate_dividend_yield(ticker, price):
    """
    使用最近12个月实际现金分红计算股息率。
    不使用 ticker.info。
    """

    price = clean_number(price)

    if price is None or price <= 0:
        return None

    try:
        dividends = ticker.dividends

        if dividends is None or dividends.empty:
            return None

        dividends = dividends.dropna()

        if dividends.empty:
            return None

        dividends.index = pd.to_datetime(
            dividends.index
        )

        latest_date = dividends.index.max()

        start_date = (
            latest_date -
            pd.Timedelta(days=365)
        )

        recent = dividends[
            dividends.index > start_date
        ]

        if recent.empty:
            return None

        annual_dividend = clean_number(
            recent.sum()
        )

        if annual_dividend is None:
            return None

        return (
            annual_dividend /
            price *
            100
        )

    except Exception:
        return None


def get_income_statement(ticker):
    """
    独立获取年度利润表。
    """

    try:
        data = ticker.get_income_stmt(
            freq="yearly"
        )

        if data is None or data.empty:
            return None

        return data

    except Exception:
        return None


def get_balance_sheet(ticker):
    """
    独立获取年度资产负债表。
    """

    try:
        data = ticker.get_balance_sheet(
            freq="yearly"
        )

        if data is None or data.empty:
            return None

        return data

    except Exception:
        return None


def get_cashflow(ticker):
    """
    独立获取年度现金流量表。
    """

    try:
        data = ticker.get_cash_flow(
            freq="yearly"
        )

        if data is None or data.empty:
            return None

        return data

    except Exception:
        return None


def get_latest_financial_value(df, names):
    row = find_row(df, names)
    return get_latest_value(row)


def calculate_current_roe(
    income,
    balance
):
    """
    当前年度 ROE：

    净利润 /
    ((期初股东权益 + 期末股东权益) / 2)
    """

    if income is None or balance is None:
        return None

    net_income_row = find_row(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "NetIncome"
        ]
    )

    equity_row = find_row(
        balance,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
    )

    if net_income_row is None:
        return None

    if equity_row is None:
        return None

    try:
        income_dates = sorted(
            list(income.columns),
            key=lambda x: pd.Timestamp(x)
        )

        balance_dates = sorted(
            list(balance.columns),
            key=lambda x: pd.Timestamp(x)
        )

        if not income_dates:
            return None

        latest_income_date = income_dates[-1]

        net_income = clean_number(
            net_income_row.get(
                latest_income_date
            )
        )

        if net_income is None:
            return None

        # 找对应的期末权益
        matching_dates = [
            d for d in balance_dates
            if pd.Timestamp(d) <=
            pd.Timestamp(latest_income_date)
        ]

        if not matching_dates:
            return None

        end_date = matching_dates[-1]

        end_index = balance_dates.index(
            end_date
        )

        if end_index == 0:
            return None

        begin_date = balance_dates[
            end_index - 1
        ]

        equity_end = clean_number(
            equity_row.get(end_date)
        )

        equity_begin = clean_number(
            equity_row.get(begin_date)
        )

        if equity_end is None or equity_begin is None:
            return None

        average_equity = (
            equity_begin +
            equity_end
        ) / 2

        if average_equity == 0:
            return None

        return (
            net_income /
            average_equity *
            100
        )

    except Exception:
        return None


def calculate_roe_history(
    income,
    balance
):
    """
    计算最近15个有效年度 ROE。
    """

    if income is None or balance is None:
        return {
            "years": [],
            "stats": empty_stats()
        }

    net_income_row = find_row(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "NetIncome"
        ]
    )

    equity_row = find_row(
        balance,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
    )

    if net_income_row is None:
        return {
            "years": [],
            "stats": empty_stats()
        }

    if equity_row is None:
        return {
            "years": [],
            "stats": empty_stats()
        }

    try:
        dates = sorted(
            list(income.columns),
            key=lambda x: pd.Timestamp(x)
        )
    except Exception:
        return {
            "years": [],
            "stats": empty_stats()
        }

    records = []

    for date in dates:

        try:
            year = pd.Timestamp(date).year
        except Exception:
            continue

        net_income = clean_number(
            net_income_row.get(date)
        )

        if net_income is None:
            continue

        # 找当前年度对应的权益
        balance_dates = sorted(
            list(balance.columns),
            key=lambda x: pd.Timestamp(x)
        )

        matching = [
            d for d in balance_dates
            if pd.Timestamp(d) <=
            pd.Timestamp(date)
        ]

        if not matching:
            continue

        end_date = matching[-1]
        end_index = balance_dates.index(
            end_date
        )

        if end_index == 0:
            continue

        begin_date = balance_dates[
            end_index - 1
        ]

        equity_end = clean_number(
            equity_row.get(end_date)
        )

        equity_begin = clean_number(
            equity_row.get(begin_date)
        )

        if equity_end is None or equity_begin is None:
            continue

        average_equity = (
            equity_begin +
            equity_end
        ) / 2

        if average_equity == 0:
            continue

        roe = (
            net_income /
            average_equity *
            100
        )

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

    records = records[-15:]

    values = [
        x["roe"]
        for x in records
    ]

    if not values:
        return {
            "years": [],
            "stats": empty_stats()
        }

    average = sum(values) / len(values)

    med = median(values)

    std = (
        stdev(values)
        if len(values) >= 2
        else None
    )

    maximum = max(values)
    minimum = min(values)

    return {
        "years": records,
        "stats": {
            "count": len(values),
            "average": average,
            "median": med,
            "std_dev": std,
            "range": maximum - minimum,
            "max": maximum,
            "min": minimum
        }
    }


def empty_stats():
    return {
        "count": 0,
        "average": None,
        "median": None,
        "std_dev": None,
        "range": None,
        "max": None,
        "min": None
    }


def build_dashboard(symbol):

    original_symbol = str(
        symbol or ""
    ).strip()

    if not original_symbol:
        raise ValueError(
            "请输入股票代码"
        )

    symbol = normalize_symbol(
        original_symbol
    )

    ticker = yf.Ticker(symbol)

    # =========================
    # 1. 价格
    # =========================

    price = get_latest_price(
        ticker
    )

    # =========================
    # 2. 财务报表
    # =========================

    income = get_income_statement(
        ticker
    )

    balance = get_balance_sheet(
        ticker
    )

    cashflow = get_cashflow(
        ticker
    )

    # =========================
    # 3. 营收
    # =========================

    revenue = get_latest_financial_value(
        income,
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    # =========================
    # 4. 净利润
    # =========================

    net_income = get_latest_financial_value(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    )

    # =========================
    # 5. 毛利率
    # =========================

    gross_profit = get_latest_financial_value(
        income,
        [
            "Gross Profit"
        ]
    )

    gross_margin = None

    if (
        gross_profit is not None
        and revenue is not None
        and revenue != 0
    ):
        gross_margin = (
            gross_profit /
            revenue *
            100
        )

    # =========================
    # 6. 自由现金流
    # =========================

    operating_cashflow = get_latest_financial_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ]
    )

    capex = get_latest_financial_value(
        cashflow,
        [
            "Capital Expenditure",
            "Capital Expenditure Reported"
        ]
    )

    free_cash_flow = None

    if (
        operating_cashflow is not None
        and capex is not None
    ):
        # Yahoo通常把资本开支记为负数
        free_cash_flow = (
            operating_cashflow +
            capex
        )

    # =========================
    # 7. 资产负债率
    # =========================

    total_assets = get_latest_financial_value(
        balance,
        [
            "Total Assets"
        ]
    )

    total_debt = get_latest_financial_value(
        balance,
        [
            "Total Debt",
            "Total Debt And Equity"
        ]
    )

    debt_ratio = None

    if (
        total_debt is not None
        and total_assets is not None
        and total_assets != 0
    ):
        debt_ratio = (
            total_debt /
            total_assets *
            100
        )

    # =========================
    # 8. ROE
    # =========================

    roe = calculate_current_roe(
        income,
        balance
    )

    # =========================
    # 9. PE / PB
    # =========================

    pe = None
    pb = None

    # 不依赖 ticker.info
    # 使用市场价格 + 财务数据自行计算

    if (
        price is not None
        and net_income is not None
        and net_income != 0
    ):
        # 先尝试获取流通股本
        try:
            shares = ticker.get_shares_full(
                start=pd.Timestamp.now() -
                pd.Timedelta(days=30)
            )

            if shares is not None and not shares.empty:
                shares = shares.dropna()

                if not shares.empty:
                    latest_shares = clean_number(
                        shares.iloc[-1]
                    )

                    if (
                        latest_shares is not None
                        and latest_shares > 0
                    ):
                        market_cap = (
                            price *
                            latest_shares
                        )

                        pe = safe_div(
                            market_cap,
                            net_income
                        )

        except Exception:
            pass

    # =========================
    # 10. PB
    # =========================

    equity = get_latest_financial_value(
        balance,
        [
            "Stockholders Equity",
            "Common Stock Equity"
        ]
    )

    if (
        equity is not None
        and equity > 0
        and price is not None
    ):
        try:
            shares = ticker.get_shares_full(
                start=pd.Timestamp.now() -
                pd.Timedelta(days=30)
            )

            if shares is not None and not shares.empty:
                shares = shares.dropna()

                if not shares.empty:
                    latest_shares = clean_number(
                        shares.iloc[-1]
                    )

                    if (
                        latest_shares is not None
                        and latest_shares > 0
                    ):
                        market_cap = (
                            price *
                            latest_shares
                        )

                        pb = safe_div(
                            market_cap,
                            equity
                        )

        except Exception:
            pass

    # =========================
    # 11. 股息率
    # =========================

    dividend_yield = calculate_dividend_yield(
        ticker,
        price
    )

    # =========================
    # 12. 衍生指标
    # =========================

    roe_pb = safe_div(
        roe,
        pb
    )

    pe_roe = safe_div(
        pe,
        roe
    )

    # =========================
    # 13. 15年ROE
    # =========================

    roe_history = calculate_roe_history(
        income,
        balance
    )

    # =========================
    # 14. 公司名称
    # =========================

    company = symbol

    try:
        fast_info = ticker.fast_info

        if fast_info is not None:
            company = (
                getattr(
                    ticker,
                    "ticker",
                    None
                )
                or symbol
            )

    except Exception:
        pass

    # =========================
    # 返回
    # =========================

    market_cap = None

    try:
        shares = ticker.get_shares_full(
            start=pd.Timestamp.now() -
            pd.Timedelta(days=30)
        )

        if shares is not None and not shares.empty:
            shares = shares.dropna()

            if not shares.empty:
                latest_shares = clean_number(
                    shares.iloc[-1]
                )

                if (
                    latest_shares is not None
                    and price is not None
                ):
                    market_cap = (
                        price *
                        latest_shares
                    )

    except Exception:
        pass

    return {
        "query": original_symbol,
        "symbol": symbol,
        "company": company,
        "exchange": None,
        "currency": None,

        "market_data": {
            "price": price,
            "market_cap": market_cap
        },

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

        "roe_15y": {
            "years": roe_history["years"],
            "stats": roe_history["stats"],
            "definition": (
                "ROE = 年度净利润 / "
                "((期初股东权益 + 期末股东权益) / 2)"
            ),
            "std_definition": (
                "15年有效年度ROE的样本标准差"
            )
        },

        "source": {
            "provider": (
                "Yahoo Finance via yfinance"
            ),
            "note": (
                "V1：价格、利润表、资产负债表、"
                "现金流量表、分红分别取数；"
                "不依赖 ticker.info"
            )
        }
            }
