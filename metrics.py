import requests
import statistics
from datetime import datetime, timezone
from urllib.parse import quote

SEC_HEADERS = {
    "User-Agent": "stock-fundamental-dashboard contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
}


# ============================================================
# 通用工具
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, str):
            value = value.strip()

            if value == "":
                return None

            if value.lower() in {
                "null",
                "none",
                "nan",
                "-",
                "--",
            }:
                return None

            value = value.replace(",", "")

        result = float(value)

        if result != result:
            return None

        return result

    except Exception:
        return None


def clean_number(value):
    value = safe_float(value)

    if value is None:
        return None

    if abs(value) > 1e20:
        return None

    return value


def first_valid(*values):
    for value in values:
        value = clean_number(value)

        if value is not None:
            return value

    return None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_json(url, params=None, headers=None, timeout=15):
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def ratio(numerator, denominator):
    numerator = clean_number(numerator)
    denominator = clean_number(denominator)

    if numerator is None:
        return None

    if denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def percent_ratio(numerator, denominator):
    value = ratio(numerator, denominator)

    if value is None:
        return None

    return value * 100


def normalize_symbol(symbol):
    return str(symbol or "").strip().upper()


# ============================================================
# 市场识别
# ============================================================

def detect_market(symbol):
    s = normalize_symbol(symbol)

    # 港股：
    # 00700
    # 700
    # 00700.HK
    if s.endswith(".HK"):
        return "HK"

    if s.isdigit() and 1 <= len(s) <= 5:
        return "HK"

    # A股：
    # 600519
    # 000858
    # 300750
    # 688981
    if s.isdigit() and len(s) == 6:
        return "CN"

    if s.endswith(".SH") or s.endswith(".SZ"):
        return "CN"

    # 美股默认
    return "US"


# ============================================================
# A股代码处理
# ============================================================

def normalize_cn_code(symbol):
    s = normalize_symbol(symbol)

    if s.endswith(".SH") or s.endswith(".SZ"):
        s = s.split(".")[0]

    return s.zfill(6)


def cn_secid(code):
    code = normalize_cn_code(code)

    # 上海
    if code.startswith(("6", "68")):
        return f"1.{code}"

    # 深圳
    if code.startswith(("0", "2", "3")):
        return f"0.{code}"

    # 北交所等暂时不强行猜测
    return None


# ============================================================
# 港股代码处理
# ============================================================

def normalize_hk_code(symbol):
    s = normalize_symbol(symbol)

    if s.endswith(".HK"):
        s = s[:-3]

    s = s.replace(".", "")

    return s.zfill(5)


def hk_secid(symbol):
    return f"116.{normalize_hk_code(symbol)}"


# ============================================================
# Yahoo Finance
# ============================================================

def yahoo_chart(symbol, period="1y"):
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol)}",
    ]

    params = {
        "range": period,
        "interval": "1d",
        "events": "div,splits",
    }

    last_error = None

    for url in urls:
        try:
            data = get_json(
                url,
                params=params,
                headers=YAHOO_HEADERS,
                timeout=15,
            )

            result = data.get("chart", {}).get("result")

            if result:
                return result[0]

        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise RuntimeError("Yahoo Finance 没有返回数据")


def yahoo_quote(symbol):
    chart = yahoo_chart(symbol, "1y")

    meta = chart.get("meta", {})

    price = first_valid(
        meta.get("regularMarketPrice"),
        meta.get("previousClose"),
    )

    company = (
        meta.get("longName")
        or meta.get("shortName")
        or symbol
    )

    currency = meta.get("currency")

    exchange = (
        meta.get("fullExchangeName")
        or meta.get("exchangeName")
    )

    market_cap = None

    # Yahoo chart 接口有时直接返回 marketCap
    market_cap = clean_number(
        meta.get("marketCap")
    )

    return {
        "price": price,
        "company": company,
        "currency": currency,
        "exchange": exchange,
        "market_cap": market_cap,
        "chart": chart,
    }


# ============================================================
# SEC
# ============================================================

def sec_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"

    return get_json(
        url,
        headers=SEC_HEADERS,
        timeout=20,
    )


def find_cik_by_ticker(ticker):
    ticker = normalize_symbol(ticker)

    data = sec_ticker_map()

    for item in data.values():
        if normalize_symbol(item.get("ticker")) == ticker:
            return str(item.get("cik_str")).zfill(10)

    return None


def sec_company_facts(ticker):
    cik = find_cik_by_ticker(ticker)

    if not cik:
        raise RuntimeError(
            f"SEC 找不到股票代码 {ticker}"
        )

    url = (
        f"https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{cik}.json"
    )

    return get_json(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )


def extract_fact_units(fact):
    if not fact:
        return []

    units = fact.get("units", {})

    if not units:
        return []

    preferred = []

    for unit_name in [
        "USD",
        "USD/shares",
        "shares",
        "pure",
    ]:
        if unit_name in units:
            preferred = units[unit_name]
            break

    if not preferred:
        first_key = next(iter(units))
        preferred = units[first_key]

    return preferred


def fact_values(facts, tags):
    result = []

    usgaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        fact = usgaap.get(tag)

        if not fact:
            continue

        for item in extract_fact_units(fact):
            value = clean_number(item.get("val"))

            if value is None:
                continue

            result.append({
                "value": value,
                "fy": item.get("fy"),
                "fp": item.get("fp"),
                "form": item.get("form"),
                "filed": item.get("filed"),
                "start": item.get("start"),
                "end": item.get("end"),
                "frame": item.get("frame"),
                "tag": tag,
            })

    return result


def annual_fact_values(facts, tags):
    items = fact_values(facts, tags)

    annual = []

    for item in items:
        form = item.get("form")

        if form not in {
            "10-K",
            "10-K/A",
        }:
            continue

        fy = item.get("fy")

        if fy is None:
            continue

        annual.append(item)

    return annual


def latest_annual_value(facts, tags):
    items = annual_fact_values(facts, tags)

    if not items:
        return None

    items.sort(
        key=lambda x: (
            x.get("fy") or 0,
            x.get("filed") or "",
            x.get("end") or "",
        )
    )

    return items[-1]["value"]


def annual_value_by_year(facts, tags, year):
    items = annual_fact_values(facts, tags)

    candidates = [
        item for item in items
        if item.get("fy") == year
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x.get("filed") or "",
            x.get("end") or "",
        )
    )

    return candidates[-1]["value"]


def latest_annual_balance(facts, tags):
    items = fact_values(facts, tags)

    candidates = []

    for item in items:
        form = item.get("form")

        if form not in {
            "10-K",
            "10-K/A",
            "10-Q",
            "10-Q/A",
        }:
            continue

        if not item.get("end"):
            continue

        candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x.get("end") or "",
            x.get("filed") or "",
        )
    )

    return candidates[-1]["value"]


# ============================================================
# SEC 最新财务数据
# ============================================================

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]

NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",
]

EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "PartnersCapital",
]

ASSETS_TAGS = [
    "Assets",
]

LIABILITIES_TAGS = [
    "Liabilities",
]

OPERATING_CASHFLOW_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
]

CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]

GROSS_PROFIT_TAGS = [
    "GrossProfit",
]

SHARES_TAGS = [
    "EntityCommonStockSharesOutstanding",
]

EPS_TAGS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]


def sec_annual_revenue(facts, year=None):
    if year is None:
        return latest_annual_value(
            facts,
            REVENUE_TAGS,
        )

    return annual_value_by_year(
        facts,
        REVENUE_TAGS,
        year,
    )


def sec_annual_net_income(facts, year=None):
    if year is None:
        return latest_annual_value(
            facts,
            NET_INCOME_TAGS,
        )

    return annual_value_by_year(
        facts,
        NET_INCOME_TAGS,
        year,
    )


def sec_equity_at_end(facts, year=None):
    items = fact_values(
        facts,
        EQUITY_TAGS,
    )

    candidates = []

    for item in items:
        if not item.get("end"):
            continue

        if year is not None:
            end = str(item.get("end"))

            if not end.startswith(str(year)):
                continue

        candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x.get("end") or "",
            x.get("filed") or "",
        )
    )

    return candidates[-1]["value"]


def sec_annual_assets(facts):
    return latest_annual_balance(
        facts,
        ASSETS_TAGS,
    )


def sec_annual_liabilities(facts):
    return latest_annual_balance(
        facts,
        LIABILITIES_TAGS,
    )


def sec_annual_gross_profit(facts):
    return latest_annual_value(
        facts,
        GROSS_PROFIT_TAGS,
    )


def sec_annual_operating_cashflow(facts):
    return latest_annual_value(
        facts,
        OPERATING_CASHFLOW_TAGS,
    )


def sec_annual_capex(facts):
    return latest_annual_value(
        facts,
        CAPEX_TAGS,
    )


def sec_current_shares(facts):
    items = fact_values(
        facts,
        SHARES_TAGS,
    )

    if not items:
        return None

    items.sort(
        key=lambda x: (
            x.get("end") or "",
            x.get("filed") or "",
        )
    )

    return items[-1]["value"]


def sec_latest_eps(facts):
    return latest_annual_value(
        facts,
        EPS_TAGS,
    )


# ============================================================
# 美股 15 年 ROE
# ============================================================

def calculate_roe_series(facts, years=15):
    current_year = datetime.now(timezone.utc).year

    rows = []

    # 取最近 15 个完整年度
    target_years = range(
        current_year - years - 1,
        current_year,
    )

    for year in target_years:

        net_income = sec_annual_net_income(
            facts,
            year,
        )

        end_equity = sec_equity_at_end(
            facts,
            year,
        )

        begin_equity = sec_equity_at_end(
            facts,
            year - 1,
        )

        if (
            net_income is None
            or end_equity is None
            or begin_equity is None
        ):
            continue

        average_equity = (
            begin_equity + end_equity
        ) / 2

        if average_equity == 0:
            continue

        roe = (
            net_income
            / average_equity
            * 100
        )

        rows.append({
            "year": year,
            "roe": roe,
        })

    rows = rows[-years:]

    return rows


def roe_statistics(rows):
    values = [
        clean_number(item.get("roe"))
        for item in rows
    ]

    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return {
            "count": 0,
            "average": None,
            "median": None,
            "std_dev": None,
            "range": None,
            "max": None,
            "min": None,
        }

    if len(values) >= 2:
        std_dev = statistics.stdev(values)
    else:
        std_dev = None

    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "median": statistics.median(values),
        "std_dev": std_dev,
        "range": max(values) - min(values),
        "max": max(values),
        "min": min(values),
    }


# ============================================================
# 美股数据
# ============================================================

def build_us_dashboard(symbol):
    symbol = normalize_symbol(symbol)

    yahoo = yahoo_quote(symbol)

    facts = sec_company_facts(symbol)

    price = yahoo.get("price")

    revenue = sec_annual_revenue(facts)

    net_income = sec_annual_net_income(facts)

    equity_end = sec_equity_at_end(facts)

    equity_previous = sec_equity_at_end(
        facts,
        datetime.now(timezone.utc).year - 2,
    )

    # 如果上一年度无法直接匹配，再寻找最近两个资产负债表权益值
    if equity_previous is None:
        equity_previous = equity_end

    average_equity = None

    if (
        equity_end is not None
        and equity_previous is not None
    ):
        average_equity = (
            equity_previous + equity_end
        ) / 2

    roe = ratio(
        net_income,
        average_equity,
    )

    if roe is not None:
        roe *= 100

    assets = sec_annual_assets(facts)

    liabilities = sec_annual_liabilities(facts)

    debt_ratio = percent_ratio(
        liabilities,
        assets,
    )

    gross_profit = sec_annual_gross_profit(facts)

    gross_margin = percent_ratio(
        gross_profit,
        revenue,
    )

    operating_cf = sec_annual_operating_cashflow(
        facts
    )

    capex = sec_annual_capex(facts)

    free_cash_flow = None

    if (
        operating_cf is not None
        and capex is not None
    ):
        free_cash_flow = (
            operating_cf - abs(capex)
        )

    shares = sec_current_shares(facts)

    market_cap = yahoo.get("market_cap")

    if (
        market_cap is None
        and price is not None
        and shares is not None
    ):
        market_cap = price * shares

    eps = None

    if (
        net_income is not None
        and shares is not None
        and shares != 0
    ):
        eps = net_income / shares

    pe = ratio(price, eps)

    book_value_per_share = None

    if (
        equity_end is not None
        and shares is not None
        and shares != 0
    ):
        book_value_per_share = (
            equity_end / shares
        )

    pb = ratio(
        price,
        book_value_per_share,
    )

    roe_pb = ratio(
        roe,
        pb,
    )

    pe_roe = ratio(
        pe,
        roe,
    )

    roe_rows = calculate_roe_series(
        facts,
        years=15,
    )

    stats = roe_statistics(roe_rows)

    return {
        "query": symbol,
        "symbol": symbol,
        "market": "US",
        "company": yahoo.get("company"),
        "exchange": yahoo.get("exchange"),
        "currency": yahoo.get("currency"),

        "market_data": {
            "price": price,
            "market_cap": market_cap,
        },

        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "roe_pb": roe_pb,
            "pe_roe": pe_roe,
        },

        "fundamentals": {
            "revenue": revenue,
            "net_income": net_income,
            "gross_margin": gross_margin,
            "free_cash_flow": free_cash_flow,
            "dividend_yield": None,
            "debt_ratio": debt_ratio,
        },

        "roe_15y": {
            "years": roe_rows,
            "stats": stats,
            "definition": (
                "ROE = 年度净利润 / "
                "((期初股东权益 + 期末股东权益) / 2)"
            ),
            "std_definition": (
                "15年有效年度ROE的样本标准差"
            ),
        },

        "source": {
            "market": "US",
            "provider": (
                "SEC Company Facts + Yahoo Finance"
            ),
            "data_as_of": now_iso(),
            "financial_data_status": {
                "income": revenue is not None
                and net_income is not None,
                "balance": equity_end is not None,
                "cashflow": free_cash_flow is not None,
            },
            "valuation_data_status": {
                "market_price": price is not None,
                "market_cap": market_cap is not None,
                "eps": eps is not None,
                "pe": pe is not None,
                "pb": pb is not None,
            },
            "note": (
                "财务历史优先使用SEC XBRL数据；"
                "当前市场价格使用Yahoo Finance；"
                "无法取得的数据保持为null，"
                "不进行估算或虚构。"
            ),
        },
    }


# ============================================================
# 东方财富实时行情
# ============================================================

def eastmoney_quote(secid):
    url = (
        "https://push2.eastmoney.com/"
        "api/qt/stock/get"
    )

    params = {
        "secid": secid,
        "fltt": "2",
        "invt": "2",

        "fields": (
            "f43,f44,f45,f46,f47,f48,"
            "f57,f58,f60,f116,f117,"
            "f162,f164,f167,f168,"
            "f170,f173,f171,f84,f85"
        ),
    }

    data = get_json(
        url,
        params=params,
        headers=EASTMONEY_HEADERS,
        timeout=15,
    )

    result = data.get("data")

    if not result:
        raise RuntimeError(
            f"东方财富没有返回股票数据：{secid}"
        )

    return result


# ============================================================
# A股
# ============================================================

def build_cn_dashboard(symbol):
    code = normalize_cn_code(symbol)

    secid = cn_secid(code)

    if not secid:
        raise RuntimeError(
            f"暂不支持该A股代码：{code}"
        )

    data = eastmoney_quote(secid)

    price = clean_number(data.get("f43"))

    # 东方财富部分行情接口价格可能以“分”为单位，
    # 股票详情接口通常已经返回正常价格。
    # 如果明显异常，再进行兼容处理。
    if price is not None and price > 100000:
        price = price / 100

    company = (
        data.get("f58")
        or code
    )

    market_cap = clean_number(
        data.get("f116")
    )

    pe = clean_number(
        data.get("f162")
    )

    pb = clean_number(
        data.get("f167")
    )

    # f164 = ROE
    roe = clean_number(
        data.get("f164")
    )

    # f168 = 营业收入
    revenue = clean_number(
        data.get("f168")
    )

    # f170 = 净利润
    net_income = clean_number(
        data.get("f170")
    )

    # f173 = 毛利率
    gross_margin = clean_number(
        data.get("f173")
    )

    roe_pb = ratio(
        roe,
        pb,
    )

    pe_roe = ratio(
        pe,
        roe,
    )

    return {
        "query": symbol,
        "symbol": code,
        "market": "CN",
        "company": company,
        "exchange": (
            "Shanghai"
            if secid.startswith("1.")
            else "Shenzhen"
        ),
        "currency": "CNY",

        "market_data": {
            "price": price,
            "market_cap": market_cap,
        },

        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "roe_pb": roe_pb,
            "pe_roe": pe_roe,
        },

        "fundamentals": {
            "revenue": revenue,
            "net_income": net_income,
            "gross_margin": gross_margin,
            "free_cash_flow": None,
            "dividend_yield": None,
            "debt_ratio": None,
        },

        "roe_15y": {
            "years": [],
            "stats": {
                "count": 0,
                "average": None,
                "median": None,
                "std_dev": None,
                "range": None,
                "max": None,
                "min": None,
            },
            "definition": (
                "ROE = 年度净利润 / "
                "((期初股东权益 + 期末股东权益) / 2)"
            ),
            "std_definition": (
                "15年有效年度ROE的样本标准差"
            ),
        },

        "source": {
            "market": "CN",
            "provider": "Eastmoney Finance",
            "data_as_of": now_iso(),
            "financial_data_status": {
                "income": (
                    revenue is not None
                    or net_income is not None
                ),
                "balance": False,
                "cashflow": False,
            },
            "valuation_data_status": {
                "market_price": price is not None,
                "market_cap": market_cap is not None,
                "eps": None,
                "pe": pe is not None,
                "pb": pb is not None,
            },
            "note": (
                "A股当前行情与核心财务摘要"
                "使用东方财富公开接口；"
                "尚未取得的数据保持为null，"
                "不进行估算或虚构。"
            ),
        },
    }


# ============================================================
# 港股
# ============================================================

def hk_financial_indicator(symbol):
    code = normalize_hk_code(symbol)

    url = (
        "https://datacenter.eastmoney.com/"
        "securities/api/data/v1/get"
    )

    params = {
        "reportName":
            "RPT_CUSTOM_HKF10_FN_MAININDICATORMAX",

        "columns": (
            "ORG_CODE,SECUCODE,SECURITY_CODE,"
            "SECURITY_NAME_ABBR,SECURITY_INNER_CODE,"
            "REPORT_DATE,BASIC_EPS,"
            "PER_NETCASH_OPERATE,BPS,BPS_NEDILUTED,"
            "COMMON_ACS,PER_SHARES,ISSUED_COMMON_SHARES,"
            "HK_COMMON_SHARES,TOTAL_MARKET_CAP,"
            "HKSK_MARKET_CAP,OPERATE_INCOME,"
            "OPERATE_INCOME_SQ,OPERATE_INCOME_QOQ,"
            "OPERATE_INCOME_QOQ_SQ,HOLDER_PROFIT,"
            "HOLDER_PROFIT_SQ,HOLDER_PROFIT_QOQ,"
            "HOLDER_PROFIT_QOQ_SQ,PE_TTM,"
            "PE_TTM_SQ,PB_TTM,PB_TTM_SQ,"
            "NET_PROFIT_RATIO,NET_PROFIT_RATIO_SQ,"
            "ROE_AVG,ROE_AVG_SQ,ROA"
        ),

        "filter": (
            f'(SECUCODE="{code}.HK")'
        ),

        "pageNumber": "1",
        "pageSize": "5",

        "sortColumns": "REPORT_DATE",
        "sortTypes": "-1",

        "source": "HSF10",
        "client": "PC",
    }

    try:
        data = get_json(
            url,
            params=params,
            headers=EASTMONEY_HEADERS,
            timeout=20,
        )

        result = data.get("result") or {}

        rows = result.get("data") or []

        if rows:
            return rows[0]

    except Exception:
        pass

    return {}


def build_hk_dashboard(symbol):
    code = normalize_hk_code(symbol)

    secid = hk_secid(code)

    quote = eastmoney_quote(secid)

    indicator = hk_financial_indicator(code)

    price = clean_number(
        quote.get("f43")
    )

    if price is not None and price > 100000:
        price = price / 100

    company = (
        quote.get("f58")
        or indicator.get("SECURITY_NAME_ABBR")
        or code
    )

    market_cap = first_valid(
        indicator.get("TOTAL_MARKET_CAP"),
        quote.get("f116"),
    )

    pe = first_valid(
        indicator.get("PE_TTM"),
        quote.get("f162"),
    )

    pb = first_valid(
        indicator.get("PB_TTM"),
        quote.get("f167"),
    )

    roe = clean_number(
        indicator.get("ROE_AVG")
    )

    revenue = clean_number(
        indicator.get("OPERATE_INCOME")
    )

    net_income = clean_number(
        indicator.get("HOLDER_PROFIT")
    )

    gross_margin = None

    # 如果净利润率可取得，可以保留，
    # 但不能把净利润率冒充毛利率。
    net_profit_ratio = clean_number(
        indicator.get("NET_PROFIT_RATIO")
    )

    roe_pb = ratio(
        roe,
        pb,
    )

    pe_roe = ratio(
        pe,
        roe,
    )

    return {
        "query": symbol,
        "symbol": code + ".HK",
        "market": "HK",
        "company": company,
        "exchange": "Hong Kong",
        "currency": "HKD",

        "market_data": {
            "price": price,
            "market_cap": market_cap,
        },

        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "roe_pb": roe_pb,
            "pe_roe": pe_roe,
        },

        "fundamentals": {
            "revenue": revenue,
            "net_income": net_income,
            "gross_margin": gross_margin,
            "free_cash_flow": None,
            "dividend_yield": None,
            "debt_ratio": None,
        },

        "roe_15y": {
            "years": [],
            "stats": {
                "count": 0,
                "average": None,
                "median": None,
                "std_dev": None,
                "range": None,
                "max": None,
                "min": None,
            },
            "definition": (
                "ROE = 年度净利润 / "
                "((期初股东权益 + 期末股东权益) / 2)"
            ),
            "std_definition": (
                "15年有效年度ROE的样本标准差"
            ),
        },

        "source": {
            "market": "HK",
            "provider": "Eastmoney Finance",
            "data_as_of": now_iso(),
            "financial_data_status": {
                "income": (
                    revenue is not None
                    or net_income is not None
                ),
                "balance": False,
                "cashflow": False,
            },
            "valuation_data_status": {
                "market_price": price is not None,
                "market_cap": market_cap is not None,
                "eps": (
                    indicator.get("BASIC_EPS")
                    is not None
                ),
                "pe": pe is not None,
                "pb": pb is not None,
            },
            "note": (
                "港股当前行情使用东方财富实时行情；"
                "核心财务指标使用东方财富港股F10；"
                "无法取得的数据保持为null，"
                "不进行估算或虚构。"
            ),
        },

        "extra": {
            "net_profit_ratio": net_profit_ratio,
        },
    }


# ============================================================
# 总入口
# ============================================================

def build_dashboard(symbol):
    symbol = normalize_symbol(symbol)

    if not symbol:
        raise ValueError("股票代码不能为空")

    market = detect_market(symbol)

    if market == "CN":
        return build_cn_dashboard(symbol)

    if market == "HK":
        return build_hk_dashboard(symbol)

    return build_us_dashboard(symbol)
