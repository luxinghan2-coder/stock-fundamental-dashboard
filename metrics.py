from fastapi import HTTPException
import requests
import statistics
from datetime import datetime, timezone


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
})


def num(value):
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, dict):
            if "raw" in value:
                return num(value["raw"])

            if "value" in value:
                return num(value["value"])

        return float(value)

    except Exception:
        return None


def first_number(*values):
    for value in values:
        n = num(value)
        if n is not None:
            return n

    return None


def get_json(url, params=None):
    response = SESSION.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Yahoo Chart API
# ============================================================

def yahoo_chart(symbol, range_="1y"):

    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    ]

    params = {
        "range": range_,
        "interval": "1d",
        "events": "div,splits",
    }

    last_error = None

    for url in urls:
        try:
            data = get_json(url, params)

            result = (
                data
                .get("chart", {})
                .get("result")
            )

            if result:
                return result[0]

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Yahoo Chart API 请求失败：{last_error}"
    )


# ============================================================
# Yahoo Fundamentals Time Series
# ============================================================

def yahoo_timeseries(
    symbol,
    types,
    period1,
    period2
):

    type_string = ",".join(types)

    urls = [
        f"https://query1.finance.yahoo.com/ws/"
        f"fundamentals-timeseries/v1/finance/timeseries/{symbol}",

        f"https://query2.finance.yahoo.com/ws/"
        f"fundamentals-timeseries/v1/finance/timeseries/{symbol}",
    ]

    params = {
        "symbol": symbol,
        "type": type_string,
        "period1": period1,
        "period2": period2,
    }

    last_error = None

    for url in urls:

        try:

            data = get_json(url, params)

            result = (
                data
                .get("timeseries", {})
                .get("result")
            )

            if result is not None:

                return {
                    "ok": True,
                    "result": result,
                    "error": None
                }

        except Exception as exc:

            last_error = exc

    return {
        "ok": False,
        "result": [],
        "error": str(last_error)
    }


# ============================================================
# 通用解析
# ============================================================

def extract_rows(result):

    rows = []

    for block in result or []:

        field_name = None

        for key in block.keys():

            if key in (
                "meta",
                "timestamp",
                "annualTotalRevenue",
                "annualOperatingRevenue",
                "annualGrossProfit",
                "annualOperatingIncome",
                "annualNetIncome",
                "annualNetIncomeCommonStockholders",
                "annualTotalAssets",
                "annualTotalLiabilitiesNetMinorityInterest",
                "annualStockholdersEquity",
                "annualCommonStockEquity",
                "annualTotalDebt",
                "annualOperatingCashFlow",
                "annualFreeCashFlow",
                "annualCapitalExpenditure",
            ):

                field_name = key

                if key not in ("meta", "timestamp"):
                    break

        if not field_name:
            continue

        values = block.get(field_name)

        if not isinstance(values, list):
            continue

        for item in values:

            if not isinstance(item, dict):
                continue

            value = first_number(
                item.get("reportedValue"),
                item.get("raw"),
                item.get("value")
            )

            as_of = (
                item.get("asOfDate")
                or item.get("period")
            )

            timestamp = item.get("timestamp")

            if value is None:
                continue

            year = None

            if as_of:
                try:
                    year = int(str(as_of)[:4])
                except Exception:
                    pass

            if year is None and timestamp:
                try:
                    year = datetime.fromtimestamp(
                        int(timestamp),
                        tz=timezone.utc
                    ).year
                except Exception:
                    pass

            rows.append({
                "field": field_name,
                "year": year,
                "date": as_of,
                "value": value
            })

    return rows


def financial_dict(result):

    output = {}

    rows = extract_rows(result)

    for row in rows:

        field = row["field"]
        year = row["year"]

        if year is None:
            continue

        output.setdefault(field, {})

        output[field][year] = row["value"]

    return output


# ============================================================
# 财务数据
# ============================================================

def get_financials(symbol):

    now = int(datetime.now(
        timezone.utc
    ).timestamp())

    fifteen_years_ago = now - (
        15 * 365 * 24 * 60 * 60
    )

    income_types = [
        "annualTotalRevenue",
        "annualOperatingRevenue",
        "annualGrossProfit",
        "annualOperatingIncome",
        "annualNetIncome",
        "annualNetIncomeCommonStockholders",
    ]

    balance_types = [
        "annualTotalAssets",
        "annualTotalLiabilitiesNetMinorityInterest",
        "annualStockholdersEquity",
        "annualCommonStockEquity",
        "annualTotalDebt",
    ]

    cashflow_types = [
        "annualOperatingCashFlow",
        "annualFreeCashFlow",
        "annualCapitalExpenditure",
    ]

    income_response = yahoo_timeseries(
        symbol,
        income_types,
        fifteen_years_ago,
        now
    )

    balance_response = yahoo_timeseries(
        symbol,
        balance_types,
        fifteen_years_ago,
        now
    )

    cashflow_response = yahoo_timeseries(
        symbol,
        cashflow_types,
        fifteen_years_ago,
        now
    )

    income = financial_dict(
        income_response["result"]
    )

    balance = financial_dict(
        balance_response["result"]
    )

    cashflow = financial_dict(
        cashflow_response["result"]
    )

    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,

        "status": {
            "income": income_response["ok"],
            "balance": balance_response["ok"],
            "cashflow": cashflow_response["ok"],
        }
    }


# ============================================================
# 最新年度值
# ============================================================

def latest_year_value(data, fields):

    all_years = set()

    for field in fields:

        values = data.get(field, {})

        all_years.update(
            values.keys()
        )

    if not all_years:
        return None, None

    latest_year = max(all_years)

    for field in fields:

        value = data.get(field, {}).get(
            latest_year
        )

        if value is not None:

            return latest_year, value

    return latest_year, None


# ============================================================
# 价格与公司信息
# ============================================================

def get_market_data(symbol):

    chart = yahoo_chart(
        symbol,
        "1y"
    )

    meta = chart.get(
        "meta",
        {}
    )

    price = first_number(
        meta.get("regularMarketPrice"),
        meta.get("previousClose"),
        meta.get("chartPreviousClose")
    )

    company = (
        meta.get("longName")
        or meta.get("shortName")
        or symbol
    )

    exchange = (
        meta.get("fullExchangeName")
        or meta.get("exchangeName")
    )

    currency = meta.get(
        "currency"
    )

    market_cap = first_number(
        meta.get("marketCap")
    )

    return {
        "price": price,
        "company": company,
        "exchange": exchange,
        "currency": currency,
        "market_cap": market_cap,
    }


# ============================================================
# 股息率
# ============================================================

def get_dividend_yield(symbol, price):

    if price is None:
        return None

    try:

        chart = yahoo_chart(
            symbol,
            "1y"
        )

        events = chart.get(
            "events",
            {}
        )

        dividends = events.get(
            "dividends",
            {}
        )

        total_dividend = 0.0

        for item in dividends.values():

            amount = first_number(
                item.get("amount")
            )

            if amount is not None:
                total_dividend += amount

        if total_dividend <= 0:
            return 0.0

        return (
            total_dividend
            / price
            * 100
        )

    except Exception:
        return None


# ============================================================
# ROE
# ============================================================

def calculate_roe(
    net_income,
    equity_begin,
    equity_end
):

    if (
        net_income is None
        or equity_begin is None
        or equity_end is None
    ):
        return None

    average_equity = (
        equity_begin
        + equity_end
    ) / 2

    if average_equity == 0:
        return None

    return (
        net_income
        / average_equity
        * 100
    )


# ============================================================
# 主函数
# ============================================================

def build_dashboard(symbol):

    symbol = symbol.strip().upper()

    market = get_market_data(
        symbol
    )

    financials = get_financials(
        symbol
    )

    income = financials["income"]
    balance = financials["balance"]
    cashflow = financials["cashflow"]

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    revenue_year, revenue = latest_year_value(
        income,
        [
            "annualTotalRevenue",
            "annualOperatingRevenue",
        ]
    )

    # --------------------------------------------------------
    # Net income
    # --------------------------------------------------------

    net_income_year, net_income = latest_year_value(
        income,
        [
            "annualNetIncome",
            "annualNetIncomeCommonStockholders",
        ]
    )

    # --------------------------------------------------------
    # Gross profit
    # --------------------------------------------------------

    gross_profit_year, gross_profit = latest_year_value(
        income,
        [
            "annualGrossProfit",
        ]
    )

    gross_margin = None

    if (
        gross_profit is not None
        and revenue is not None
        and revenue != 0
    ):

        gross_margin = (
            gross_profit
            / revenue
            * 100
        )

    # --------------------------------------------------------
    # FCF
    # --------------------------------------------------------

    fcf_year, free_cash_flow = latest_year_value(
        cashflow,
        [
            "annualFreeCashFlow",
        ]
    )

    # 如果 Yahoo 没直接给 FCF
    # 尝试 CFO - CapEx
    if free_cash_flow is None:

        cfo_year, cfo = latest_year_value(
            cashflow,
            [
                "annualOperatingCashFlow",
            ]
        )

        capex_year, capex = latest_year_value(
            cashflow,
            [
                "annualCapitalExpenditure",
            ]
        )

        if (
            cfo is not None
            and capex is not None
        ):

            free_cash_flow = (
                cfo + capex
            )

    # --------------------------------------------------------
    # Debt ratio
    # --------------------------------------------------------

    assets_year, assets = latest_year_value(
        balance,
        [
            "annualTotalAssets",
        ]
    )

    liabilities_year, liabilities = latest_year_value(
        balance,
        [
            "annualTotalLiabilitiesNetMinorityInterest",
        ]
    )

    debt_ratio = None

    if (
        liabilities is not None
        and assets is not None
        and assets != 0
    ):

        debt_ratio = (
            liabilities
            / assets
            * 100
        )

    # --------------------------------------------------------
    # ROE history
    # --------------------------------------------------------

    net_income_data = (
        income.get(
            "annualNetIncome",
            {}
        )
    )

    if not net_income_data:

        net_income_data = (
            income.get(
                "annualNetIncomeCommonStockholders",
                {}
            )
        )

    equity_data = (
        balance.get(
            "annualStockholdersEquity",
            {}
        )
    )

    if not equity_data:

        equity_data = (
            balance.get(
                "annualCommonStockEquity",
                {}
            )
        )

    years = sorted(
        net_income_data.keys()
    )

    roe_rows = []

    for year in years:

        previous_year = year - 1

        if previous_year not in equity_data:
            continue

        if year not in equity_data:
            continue

        roe = calculate_roe(
            net_income_data.get(year),
            equity_data.get(previous_year),
            equity_data.get(year)
        )

        if roe is None:
            continue

        roe_rows.append({
            "year": year,
            "roe": roe
        })

    # 最近15个有效年度
    roe_rows = roe_rows[-15:]

    roe_values = [
        row["roe"]
        for row in roe_rows
    ]

    # --------------------------------------------------------
    # ROE statistics
    # --------------------------------------------------------

    if roe_values:

        average_roe = statistics.mean(
            roe_values
        )

        median_roe = statistics.median(
            roe_values
        )

        std_dev = (
            statistics.stdev(
                roe_values
            )
            if len(roe_values) >= 2
            else 0
        )

        max_roe = max(
            roe_values
        )

        min_roe = min(
            roe_values
        )

        roe_range = (
            max_roe
            - min_roe
        )

    else:

        average_roe = None
        median_roe = None
        std_dev = None
        max_roe = None
        min_roe = None
        roe_range = None

    # --------------------------------------------------------
    # Current ROE
    # --------------------------------------------------------

    current_roe = None

    if roe_rows:

        current_roe = roe_rows[-1]["roe"]

    # --------------------------------------------------------
    # Valuation
    #
    # Yahoo fundamentals-timeseries 中不同股票可能返回
    # 不同 valuation 字段，因此这里优先尝试。
    # --------------------------------------------------------

    now = int(datetime.now(
        timezone.utc
    ).timestamp())

    valuation_response = yahoo_timeseries(
        symbol,
        [
            "trailingMarketCap",
            "trailingPeRatio",
            "trailingPbRatio",
            "trailingPsRatio",
        ],
        now - 365 * 24 * 60 * 60,
        now
    )

    valuation_rows = valuation_response[
        "result"
    ]

    valuation_data = financial_dict(
        valuation_rows
    )

    _, market_cap_ts = latest_year_value(
        valuation_data,
        [
            "trailingMarketCap",
        ]
    )

    _, pe = latest_year_value(
        valuation_data,
        [
            "trailingPeRatio",
        ]
    )

    _, pb = latest_year_value(
        valuation_data,
        [
            "trailingPbRatio",
        ]
    )

    if market["market_cap"] is None:
        market["market_cap"] = market_cap_ts

    # --------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------

    roe_pb = None

    if (
        current_roe is not None
        and pb is not None
        and pb != 0
    ):

        roe_pb = (
            current_roe
            / pb
        )

    pe_roe = None

    if (
        pe is not None
        and current_roe is not None
        and current_roe != 0
    ):

        pe_roe = (
            pe
            / current_roe
        )

    dividend_yield = get_dividend_yield(
        symbol,
        market["price"]
    )

    # --------------------------------------------------------
    # 数据日期
    # --------------------------------------------------------

    data_as_of = datetime.now(
        timezone.utc
    ).isoformat()

    return {

        "query": symbol,

        "symbol": symbol,

        "company": market["company"],

        "exchange": market["exchange"],

        "currency": market["currency"],

        "market_data": {

            "price": market["price"],

            "market_cap": market["market_cap"],
        },

        "valuation": {

            "pe": pe,

            "pb": pb,

            "roe": current_roe,

            "roe_pb": roe_pb,

            "pe_roe": pe_roe,
        },

        "fundamentals": {

            "revenue": revenue,

            "net_income": net_income,

            "gross_margin": gross_margin,

            "free_cash_flow": free_cash_flow,

            "dividend_yield": dividend_yield,

            "debt_ratio": debt_ratio,
        },

        "roe_15y": {

            "years": roe_rows,

            "stats": {

                "count": len(
                    roe_values
                ),

                "average": average_roe,

                "median": median_roe,

                "std_dev": std_dev,

                "range": roe_range,

                "max": max_roe,

                "min": min_roe,
            },

            "definition":
                "ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)",

            "std_definition":
                "15年有效年度ROE的样本标准差",
        },

        "source": {

            "provider":
                "Yahoo Finance HTTP API",

            "data_as_of":
                data_as_of,

            "financial_data_status":
                financials["status"],

            "valuation_data_status":
                valuation_response["ok"],

            "note":
                "市场数据与财务数据均通过 HTTP 请求获取；无法取得的数据保持为 null，不进行估算或虚构。",
        }
    }
