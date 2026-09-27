from fastapi import HTTPException
import requests
import statistics
from datetime import datetime, timezone


SEC_HEADERS = {
    "User-Agent": "Stock Fundamental Dashboard contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(YAHOO_HEADERS)

_sec_ticker_cache = None


# ============================================================
# 基础工具
# ============================================================

def clean_number(value):
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        return float(value)

    except Exception:
        return None


def safe_div(a, b):
    a = clean_number(a)
    b = clean_number(b)

    if a is None or b is None or b == 0:
        return None

    return a / b


def http_get(url, headers=None, params=None, timeout=20):
    response = session.get(
        url,
        headers=headers,
        params=params,
        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# SEC Ticker → CIK
# ============================================================

def get_sec_ticker_map():
    global _sec_ticker_cache

    if _sec_ticker_cache is not None:
        return _sec_ticker_cache

    url = "https://www.sec.gov/files/company_tickers.json"

    data = http_get(
        url,
        headers={
            **SEC_HEADERS,
            "Host": "www.sec.gov"
        }
    )

    mapping = {}

    for item in data.values():
        ticker = str(
            item.get("ticker", "")
        ).upper().strip()

        cik = item.get("cik_str")
        name = item.get("title", "")

        if ticker and cik:
            mapping[ticker] = {
                "cik": int(cik),
                "name": name
            }

    _sec_ticker_cache = mapping

    return mapping


def ticker_to_cik(symbol):
    mapping = get_sec_ticker_map()

    item = mapping.get(
        symbol.upper().strip()
    )

    if not item:
        return None

    return item["cik"]


# ============================================================
# SEC Company Facts
# ============================================================

def get_company_facts(cik):
    url = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{int(cik):010d}.json"
    )

    return http_get(
        url,
        headers=SEC_HEADERS
    )


# ============================================================
# SEC Fact 查找
# ============================================================

def find_fact(facts, concepts):
    us_gaap = (
        facts
        .get("facts", {})
        .get("us-gaap", {})
    )

    for concept in concepts:
        fact = us_gaap.get(concept)

        if fact:
            return fact

    return None


# ============================================================
# 提取 SEC 年度数据
# ============================================================

def annual_facts(
    facts,
    concepts,
    unit="USD"
):
    fact = find_fact(
        facts,
        concepts
    )

    if not fact:
        return {}

    units = fact.get(
        "units",
        {}
    )

    values = units.get(unit)

    if not values:
        if units:
            values = next(
                iter(units.values())
            )
        else:
            return {}

    result = {}

    for item in values:

        value = clean_number(
            item.get("val")
        )

        if value is None:
            continue

        end = item.get("end")

        if not end:
            continue

        start = item.get("start")
        form = item.get("form", "")
        fp = item.get("fp", "")
        filed = item.get("filed", "")

        # ----------------------------------------------------
        # Duration 数据
        # ----------------------------------------------------

        if start:

            try:
                start_date = datetime.fromisoformat(
                    start
                )

                end_date = datetime.fromisoformat(
                    end
                )

                days = (
                    end_date - start_date
                ).days

            except Exception:
                days = 0

            # 只接受接近完整年度的数据
            if days < 300:
                continue

            if (
                fp != "FY"
                and form not in (
                    "10-K",
                    "10-K/A",
                    "20-F",
                    "20-F/A"
                )
            ):
                continue

        # ----------------------------------------------------
        # 年份
        # ----------------------------------------------------

        try:
            year = int(
                end[:4]
            )
        except Exception:
            continue

        previous = result.get(
            year
        )

        # 同一年优先使用最后提交版本
        if (
            previous is None
            or filed >= previous.get(
                "filed",
                ""
            )
        ):
            result[year] = {
                "value": value,
                "filed": filed,
                "form": form,
                "start": start,
                "end": end,
                "accn": item.get("accn")
            }

    return result


# ============================================================
# 财务数据
# ============================================================

def build_sec_financials(facts):

    revenue = annual_facts(
        facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet"
        ]
    )

    net_income = annual_facts(
        facts,
        [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic"
        ]
    )

    gross_profit = annual_facts(
        facts,
        [
            "GrossProfit"
        ]
    )

    assets = annual_facts(
        facts,
        [
            "Assets"
        ]
    )

    liabilities = annual_facts(
        facts,
        [
            "Liabilities"
        ]
    )

    equity = annual_facts(
        facts,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ]
    )

    operating_cash_flow = annual_facts(
        facts,
        [
            "NetCashProvidedByUsedInOperatingActivities"
        ]
    )

    capex = annual_facts(
        facts,
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets"
        ]
    )

    return {
        "revenue": revenue,
        "net_income": net_income,
        "gross_profit": gross_profit,
        "assets": assets,
        "liabilities": liabilities,
        "equity": equity,
        "operating_cash_flow": operating_cash_flow,
        "capex": capex
    }


# ============================================================
# 15年 ROE
# ============================================================

def calculate_roe_history(financials):

    net_income = financials["net_income"]
    equity = financials["equity"]

    candidate_years = sorted(
        set(net_income.keys())
        &
        set(equity.keys())
    )

    rows = []

    for year in candidate_years:

        previous_year = year - 1

        if previous_year not in equity:
            continue

        ni = clean_number(
            net_income[year]["value"]
        )

        beginning_equity = clean_number(
            equity[previous_year]["value"]
        )

        ending_equity = clean_number(
            equity[year]["value"]
        )

        if (
            ni is None
            or beginning_equity is None
            or ending_equity is None
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
            ni
            /
            average_equity
        ) * 100

        rows.append({
            "year": year,
            "roe": roe,
            "net_income": ni,
            "beginning_equity": beginning_equity,
            "ending_equity": ending_equity
        })

    # 最近15个有效年度
    rows = rows[-15:]

    values = [
        row["roe"]
        for row in rows
        if row["roe"] is not None
    ]

    if not values:

        stats = {
            "count": 0,
            "average": None,
            "median": None,
            "std_dev": None,
            "range": None,
            "max": None,
            "min": None
        }

    else:

        average = statistics.mean(
            values
        )

        median = statistics.median(
            values
        )

        if len(values) >= 2:
            std_dev = statistics.stdev(
                values
            )
        else:
            std_dev = None

        stats = {
            "count": len(values),
            "average": average,
            "median": median,
            "std_dev": std_dev,
            "range": (
                max(values)
                -
                min(values)
            ),
            "max": max(values),
            "min": min(values)
        }

    return rows, stats


# ============================================================
# Yahoo 当前价格
# ============================================================

def yahoo_chart(symbol):

    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    ]

    params = {
        "range": "5d",
        "interval": "1d",
        "events": "div,splits"
    }

    last_error = None

    for url in urls:

        try:

            response = session.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            data = response.json()

            results = (
                data
                .get("chart", {})
                .get("result")
            )

            if results:
                return results[0]

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Yahoo价格数据获取失败：{last_error}"
    )


def get_market_data(symbol):

    chart = yahoo_chart(
        symbol
    )

    meta = chart.get(
        "meta",
        {}
    )

    price = clean_number(
        meta.get(
            "regularMarketPrice"
        )
    )

    if price is None:
        price = clean_number(
            meta.get(
                "previousClose"
            )
        )

    return {
        "price": price,
        "currency": meta.get(
            "currency"
        ),
        "exchange": (
            meta.get("fullExchangeName")
            or meta.get("exchangeName")
        ),
        "company": (
            meta.get("longName")
            or meta.get("shortName")
        )
    }


# ============================================================
# Yahoo 市值
# ============================================================

def get_market_cap(symbol):

    urls = [
        "https://query1.finance.yahoo.com/v7/finance/quote",
        "https://query2.finance.yahoo.com/v7/finance/quote"
    ]

    params = {
        "symbols": symbol
    }

    for url in urls:

        try:

            response = session.get(
                url,
                params=params,
                timeout=15
            )

            if response.status_code != 200:
                continue

            data = response.json()

            results = (
                data
                .get("quoteResponse", {})
                .get("result", [])
            )

            if not results:
                continue

            market_cap = clean_number(
                results[0].get(
                    "marketCap"
                )
            )

            if market_cap is not None:
                return market_cap

        except Exception:
            continue

    return None


# ============================================================
# Yahoo TTM EPS
# ============================================================

def get_ttm_eps(symbol):

    urls = [
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/",
        "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
    ]

    for base in urls:

        try:

            url = base + symbol

            response = session.get(
                url,
                params={
                    "modules": "defaultKeyStatistics"
                },
                timeout=15
            )

            if response.status_code != 200:
                continue

            data = response.json()

            results = (
                data
                .get("quoteSummary", {})
                .get("result")
            )

            if not results:
                continue

            item = results[0]

            eps = (
                item
                .get("defaultKeyStatistics", {})
                .get("trailingEps")
            )

            if isinstance(eps, dict):
                eps = eps.get("raw")

            eps = clean_number(
                eps
            )

            if eps is not None:
                return eps

        except Exception:
            continue

    return None


# ============================================================
# 股息率
# ============================================================

def get_dividend_yield(
    symbol,
    price
):

    if price is None:
        return None

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{symbol}"
    )

    params = {
        "range": "1y",
        "interval": "1d",
        "events": "div"
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = (
            data
            .get("chart", {})
            .get("result")
        )

        if not results:
            return None

        events = (
            results[0]
            .get("events", {})
            .get("dividends", {})
        )

        if not events:
            return 0

        total = 0

        for event in events.values():

            amount = clean_number(
                event.get("amount")
            )

            if amount is not None:
                total += amount

        return (
            total
            /
            price
            *
            100
        )

    except Exception:
        return None


# ============================================================
# 主函数
# ============================================================

def build_dashboard(symbol):

    symbol = symbol.upper().strip()

    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="请输入股票代码"
        )

    # --------------------------------------------------------
    # 当前市场数据
    # --------------------------------------------------------

    market = get_market_data(
        symbol
    )

    price = market.get(
        "price"
    )

    # --------------------------------------------------------
    # SEC CIK
    # --------------------------------------------------------

    cik = ticker_to_cik(
        symbol
    )

    # --------------------------------------------------------
    # 如果SEC没有找到
    # --------------------------------------------------------

    if cik is None:

        return {
            "query": symbol,
            "symbol": symbol,
            "company": market.get(
                "company"
            ),
            "exchange": market.get(
                "exchange"
            ),
            "currency": market.get(
                "currency"
            ),

            "market_data": {
                "price": price,
                "market_cap": get_market_cap(
                    symbol
                )
            },

            "valuation": {
                "pe": None,
                "pb": None,
                "roe": None,
                "roe_pb": None,
                "pe_roe": None
            },

            "fundamentals": {
                "revenue": None,
                "net_income": None,
                "gross_margin": None,
                "free_cash_flow": None,
                "dividend_yield": get_dividend_yield(
                    symbol,
                    price
                ),
                "debt_ratio": None
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
                    "min": None
                },
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
                    "SEC Company Facts + Yahoo Finance"
                ),
                "financial_data_status": False,
                "valuation_data_status": False,
                "data_as_of": datetime.now(
                    timezone.utc
                ).isoformat(),
                "note": (
                    "SEC未找到该股票对应的CIK，"
                    "因此没有虚构财务数据。"
                )
            }
        }

    # --------------------------------------------------------
    # SEC 财务数据
    # --------------------------------------------------------

    facts = get_company_facts(
        cik
    )

    financials = build_sec_financials(
        facts
    )

    # --------------------------------------------------------
    # 15年ROE
    # --------------------------------------------------------

    roe_rows, roe_stats = (
        calculate_roe_history(
            financials
        )
    )

    # --------------------------------------------------------
    # 最新年度数据
    # --------------------------------------------------------

    def latest_value(name):

        data = financials.get(
            name,
            {}
        )

        if not data:
            return None

        year = max(
            data.keys()
        )

        return (
            year,
            data[year]["value"]
        )

    revenue_item = latest_value(
        "revenue"
    )

    net_income_item = latest_value(
        "net_income"
    )

    gross_profit_item = latest_value(
        "gross_profit"
    )

    assets_item = latest_value(
        "assets"
    )

    liabilities_item = latest_value(
        "liabilities"
    )

    equity_item = latest_value(
        "equity"
    )

    ocf_item = latest_value(
        "operating_cash_flow"
    )

    capex_item = latest_value(
        "capex"
    )

    revenue = (
        revenue_item[1]
        if revenue_item
        else None
    )

    net_income = (
        net_income_item[1]
        if net_income_item
        else None
    )

    gross_profit = (
        gross_profit_item[1]
        if gross_profit_item
        else None
    )

    assets = (
        assets_item[1]
        if assets_item
        else None
    )

    liabilities = (
        liabilities_item[1]
        if liabilities_item
        else None
    )

    equity = (
        equity_item[1]
        if equity_item
        else None
    )

    operating_cash_flow = (
        ocf_item[1]
        if ocf_item
        else None
    )

    capex = (
        capex_item[1]
        if capex_item
        else None
    )

    # --------------------------------------------------------
    # 毛利率
    # --------------------------------------------------------

    gross_margin = safe_div(
        gross_profit,
        revenue
    )

    if gross_margin is not None:
        gross_margin *= 100

    # --------------------------------------------------------
    # 负债率
    # --------------------------------------------------------

    debt_ratio = safe_div(
        liabilities,
        assets
    )

    if debt_ratio is not None:
        debt_ratio *= 100

    # --------------------------------------------------------
    # 自由现金流
    # --------------------------------------------------------

    free_cash_flow = None

    if (
        operating_cash_flow is not None
        and capex is not None
    ):
        free_cash_flow = (
            operating_cash_flow
            -
            capex
        )

    # --------------------------------------------------------
    # 当前ROE
    # --------------------------------------------------------

    current_roe = None

    if roe_rows:
        current_roe = roe_rows[-1][
            "roe"
        ]

    # --------------------------------------------------------
    # 市值
    # --------------------------------------------------------

    market_cap = get_market_cap(
        symbol
    )

    # --------------------------------------------------------
    # EPS
    # --------------------------------------------------------

    eps = get_ttm_eps(
        symbol
    )

    # --------------------------------------------------------
    # PE
    # --------------------------------------------------------

    pe = None

    if (
        price is not None
        and eps is not None
        and eps > 0
    ):
        pe = (
            price
            /
            eps
        )

    # --------------------------------------------------------
    # PB
    # --------------------------------------------------------

    pb = None

    if (
        market_cap is not None
        and equity is not None
        and equity > 0
    ):
        pb = (
            market_cap
            /
            equity
        )

    # --------------------------------------------------------
    # ROE / PB
    # --------------------------------------------------------

    roe_pb = safe_div(
        current_roe,
        pb
    )

    # --------------------------------------------------------
    # PE / ROE
    # --------------------------------------------------------

    pe_roe = safe_div(
        pe,
        current_roe
    )

    # --------------------------------------------------------
    # 股息率
    # --------------------------------------------------------

    dividend_yield = (
        get_dividend_yield(
            symbol,
            price
        )
    )

    # --------------------------------------------------------
    # 返回结果
    # --------------------------------------------------------

    return {

        "query": symbol,

        "symbol": symbol,

        "company": (
            market.get("company")
            or facts.get("entityName")
        ),

        "exchange": market.get(
            "exchange"
        ),

        "currency": market.get(
            "currency"
        ),

        "market_data": {
            "price": price,
            "market_cap": market_cap
        },

        "valuation": {
            "pe": pe,
            "pb": pb,
            "roe": current_roe,
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

            "years": [
                {
                    "year": row["year"],
                    "roe": row["roe"]
                }
                for row in roe_rows
            ],

            "stats": roe_stats,

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
                "SEC Company Facts + Yahoo Finance"
            ),

            "data_as_of": datetime.now(
                timezone.utc
            ).isoformat(),

            "financial_data_status": {

                "income": bool(
                    financials["net_income"]
                ),

                "balance": bool(
                    financials["equity"]
                ),

                "cashflow": bool(
                    financials["operating_cash_flow"]
                )
            },

            "valuation_data_status": {

                "market_price": (
                    price is not None
                ),

                "market_cap": (
                    market_cap is not None
                ),

                "eps": (
                    eps is not None
                ),

                "pe": (
                    pe is not None
                ),

                "pb": (
                    pb is not None
                )
            },

            "note": (
                "财务历史优先使用SEC XBRL数据；"
                "当前市场数据使用Yahoo Finance。"
                "所有无法取得的数据保持为null，"
                "不进行估算或虚构。"
            )
        }
      }
