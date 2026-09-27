import requests
import statistics
from datetime import datetime, timezone


SEC_HEADERS = {
    "User-Agent": "Stock Fundamental Dashboard contact@example.com"
}

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def sec_get(url):
    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=20
    )
    response.raise_for_status()
    return response.json()


def yahoo_get(url):
    response = requests.get(
        url,
        headers=YAHOO_HEADERS,
        timeout=20
    )
    response.raise_for_status()
    return response.json()


def get_sec_cik(symbol):
    data = sec_get(
        "https://www.sec.gov/files/company_tickers.json"
    )

    symbol = symbol.upper().strip()

    for item in data.values():
        if item.get("ticker", "").upper() == symbol:
            return str(item["cik_str"]).zfill(10)

    return None


def get_sec_facts(symbol):
    cik = get_sec_cik(symbol)

    if not cik:
        return None

    url = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{cik}.json"
    )

    return sec_get(url)


def get_fact(facts, taxonomy, concepts):
    source = facts.get("facts", {}).get(taxonomy, {})

    for concept in concepts:
        if concept in source:
            return source[concept]

    return None


def annual_values(fact):
    if not fact:
        return []

    units = fact.get("units", {})

    if "USD" in units:
        values = units["USD"]
    elif "shares" in units:
        values = units["shares"]
    elif "USD/shares" in units:
        values = units["USD/shares"]
    else:
        first_key = next(iter(units), None)

        if not first_key:
            return []

        values = units[first_key]

    result = []

    for item in values:
        form = item.get("form", "")
        fp = item.get("fp", "")
        start = item.get("start")
        end = item.get("end")
        val = item.get("val")

        if val is None or not end:
            continue

        days = None

        if start:
            try:
                d1 = datetime.fromisoformat(start)
                d2 = datetime.fromisoformat(end)
                days = (d2 - d1).days
            except Exception:
                pass

        is_annual = (
            fp == "FY"
            or form in ("10-K", "10-K/A", "20-F", "20-F/A")
        )

        if days is not None:
            is_annual = is_annual and days >= 300

        if not is_annual:
            continue

        try:
            year = int(end[:4])
        except Exception:
            continue

        result.append({
            "year": year,
            "end": end,
            "value": float(val),
            "form": form,
            "fy": item.get("fy")
        })

    result.sort(
        key=lambda x: (
            x["year"],
            x["end"]
        )
    )

    return result


def annual_latest_by_year(values):
    result = {}

    for item in values:
        year = item["year"]

        if (
            year not in result
            or item["end"] > result[year]["end"]
        ):
            result[year] = item

    return result


def get_latest_value(fact):
    if not fact:
        return None

    units = fact.get("units", {})

    if not units:
        return None

    values = []

    for unit_values in units.values():
        values.extend(unit_values)

    if not values:
        return None

    values = [
        x for x in values
        if x.get("val") is not None
        and x.get("end")
    ]

    if not values:
        return None

    values.sort(
        key=lambda x: x["end"],
        reverse=True
    )

    return float(values[0]["val"])


def get_yahoo_price(symbol):
    urls = [
        (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?range=5d&interval=1d&events=div,splits"
        ),
        (
            "https://query2.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?range=5d&interval=1d&events=div,splits"
        )
    ]

    last_error = None

    for url in urls:
        try:
            data = yahoo_get(url)

            result = data["chart"]["result"][0]
            meta = result.get("meta", {})

            price = meta.get("regularMarketPrice")

            if price is None:
                quote = (
                    result.get("indicators", {})
                    .get("quote", [{}])[0]
                )

                closes = quote.get("close", [])

                valid = [
                    x for x in closes
                    if x is not None
                ]

                if valid:
                    price = valid[-1]

            return {
                "price": float(price) if price is not None else None,
                "company": (
                    meta.get("longName")
                    or meta.get("shortName")
                    or symbol
                ),
                "exchange": meta.get("exchangeName"),
                "full_exchange": meta.get(
                    "fullExchangeName"
                ),
                "currency": meta.get("currency"),
                "events": result.get(
                    "events",
                    {}
                )
            }

        except Exception as exc:
            last_error = exc

    raise last_error or Exception(
        "Yahoo Finance price request failed"
    )


def get_dividend_yield(events, price):
    if not price or price <= 0:
        return None

    dividends = events.get("dividends", {})

    if not dividends:
        return None

    now = datetime.now(timezone.utc).timestamp()
    one_year_ago = now - 365 * 24 * 60 * 60

    total = 0.0

    for item in dividends.values():
        try:
            timestamp = float(item.get("date", 0))
            amount = float(item.get("amount", 0))
        except Exception:
            continue

        if timestamp >= one_year_ago:
            total += amount

    if total <= 0:
        return None

    return total / price * 100


def build_dashboard(symbol):
    symbol = symbol.upper().strip()

    yahoo = get_yahoo_price(symbol)

    price = yahoo["price"]

    facts = get_sec_facts(symbol)

    if facts is None:
        return {
            "query": symbol,
            "symbol": symbol,
            "company": yahoo["company"],
            "exchange": yahoo["exchange"],
            "currency": yahoo["currency"],
            "market_data": {
                "price": price,
                "market_cap": None
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
                    yahoo["events"],
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
                "provider": "SEC Company Facts + Yahoo Finance",
                "data_as_of": datetime.now(
                    timezone.utc
                ).isoformat(),
                "financial_data_status": {
                    "income": False,
                    "balance": False,
                    "cashflow": False
                },
                "valuation_data_status": {
                    "market_price": price is not None,
                    "market_cap": False,
                    "eps": False,
                    "pe": False,
                    "pb": False
                },
                "note": (
                    "无法取得的数据保持为null，"
                    "不进行估算或虚构。"
                )
            }
        }

    # -----------------------------
    # SEC facts
    # -----------------------------

    revenue_fact = get_fact(
        facts,
        "us-gaap",
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet"
        ]
    )

    net_income_fact = get_fact(
        facts,
        "us-gaap",
        [
            "NetIncomeLoss"
        ]
    )

    gross_profit_fact = get_fact(
        facts,
        "us-gaap",
        [
            "GrossProfit"
        ]
    )

    equity_fact = get_fact(
        facts,
        "us-gaap",
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ]
    )

    assets_fact = get_fact(
        facts,
        "us-gaap",
        [
            "Assets"
        ]
    )

    liabilities_fact = get_fact(
        facts,
        "us-gaap",
        [
            "Liabilities"
        ]
    )

    ocf_fact = get_fact(
        facts,
        "us-gaap",
        [
            "NetCashProvidedByUsedInOperatingActivities"
        ]
    )

    capex_fact = get_fact(
        facts,
        "us-gaap",
        [
            "PaymentsToAcquirePropertyPlantAndEquipment"
        ]
    )

    shares_fact = get_fact(
        facts,
        "dei",
        [
            "EntityCommonStockSharesOutstanding"
        ]
    )

    # -----------------------------
    # Annual series
    # -----------------------------

    revenue = annual_values(revenue_fact)
    net_income = annual_values(net_income_fact)
    gross_profit = annual_values(gross_profit_fact)
    equity = annual_values(equity_fact)
    assets = annual_values(assets_fact)
    liabilities = annual_values(liabilities_fact)
    ocf = annual_values(ocf_fact)
    capex = annual_values(capex_fact)

    revenue_by_year = annual_latest_by_year(revenue)
    income_by_year = annual_latest_by_year(net_income)
    gross_by_year = annual_latest_by_year(gross_profit)
    equity_by_year = annual_latest_by_year(equity)
    assets_by_year = annual_latest_by_year(assets)
    liabilities_by_year = annual_latest_by_year(liabilities)
    ocf_by_year = annual_latest_by_year(ocf)
    capex_by_year = annual_latest_by_year(capex)

    # -----------------------------
    # 15-year ROE
    # -----------------------------

    roe_rows = []

    equity_years = sorted(equity_by_year.keys())

    for year in sorted(income_by_year.keys()):

        if year - 1 not in equity_by_year:
            continue

        if year not in equity_by_year:
            continue

        income_value = income_by_year[
            year
        ]["value"]

        beginning_equity = equity_by_year[
            year - 1
        ]["value"]

        ending_equity = equity_by_year[
            year
        ]["value"]

        denominator = (
            beginning_equity
            + ending_equity
        ) / 2

        if denominator == 0:
            continue

        roe = (
            income_value
            / denominator
            * 100
        )

        roe_rows.append({
            "year": year,
            "roe": roe
        })

    roe_rows = roe_rows[-15:]

    roe_values = [
        row["roe"]
        for row in roe_rows
    ]

    if roe_values:
        average_roe = statistics.mean(
            roe_values
        )

        median_roe = statistics.median(
            roe_values
        )

        std_roe = (
            statistics.stdev(roe_values)
            if len(roe_values) >= 2
            else 0
        )

        max_roe = max(roe_values)
        min_roe = min(roe_values)
        range_roe = max_roe - min_roe

    else:
        average_roe = None
        median_roe = None
        std_roe = None
        max_roe = None
        min_roe = None
        range_roe = None

    current_roe = (
        roe_rows[-1]["roe"]
        if roe_rows
        else None
    )

    # -----------------------------
    # Current fundamentals
    # -----------------------------

    latest_year = (
        max(revenue_by_year.keys())
        if revenue_by_year
        else None
    )

    latest_revenue = (
        revenue_by_year[latest_year]["value"]
        if latest_year is not None
        else None
    )

    latest_income = (
        income_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in income_by_year
        else None
    )

    latest_gross_profit = (
        gross_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in gross_by_year
        else None
    )

    gross_margin = None

    if (
        latest_revenue
        and latest_revenue != 0
        and latest_gross_profit is not None
    ):
        gross_margin = (
            latest_gross_profit
            / latest_revenue
            * 100
        )

    latest_ocf = (
        ocf_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in ocf_by_year
        else None
    )

    latest_capex = (
        capex_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in capex_by_year
        else None
    )

    free_cash_flow = None

    if (
        latest_ocf is not None
        and latest_capex is not None
    ):
        free_cash_flow = (
            latest_ocf - latest_capex
        )

    latest_assets = (
        assets_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in assets_by_year
        else None
    )

    latest_liabilities = (
        liabilities_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in liabilities_by_year
        else None
    )

    debt_ratio = None

    if (
        latest_assets is not None
        and latest_assets != 0
        and latest_liabilities is not None
    ):
        debt_ratio = (
            latest_liabilities
            / latest_assets
            * 100
        )

    # -----------------------------
    # Shares outstanding
    # -----------------------------

    shares = get_latest_value(
        shares_fact
    )

    # SEC shares can be reported in actual
    # share count. We use it directly.
    market_cap = None

    if (
        price is not None
        and shares is not None
        and shares > 0
    ):
        market_cap = price * shares

    # -----------------------------
    # TTM net income
    # -----------------------------
    #
    # Try to construct trailing twelve
    # month income from the latest four
    # quarterly periods.
    #

    ttm_income = None

    if latest_income is not None:
        # For a robust free-data version,
        # use latest annual income as fallback.
        ttm_income = latest_income

    # -----------------------------
    # PE
    # -----------------------------

    pe = None

    if (
        market_cap is not None
        and ttm_income is not None
        and ttm_income > 0
    ):
        pe = (
            market_cap
            / ttm_income
        )

    # -----------------------------
    # PB
    # -----------------------------

    latest_equity = (
        equity_by_year[latest_year]["value"]
        if latest_year is not None
        and latest_year in equity_by_year
        else None
    )

    pb = None

    if (
        market_cap is not None
        and latest_equity is not None
        and latest_equity > 0
    ):
        pb = (
            market_cap
            / latest_equity
        )

    # -----------------------------
    # Derived ratios
    # -----------------------------

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

    # -----------------------------
    # Dividend
    # -----------------------------

    dividend_yield = get_dividend_yield(
        yahoo["events"],
        price
    )

    # -----------------------------
    # Output
    # -----------------------------

    return {
        "query": symbol,
        "symbol": symbol,
        "company": yahoo["company"],
        "exchange": (
            yahoo["full_exchange"]
            or yahoo["exchange"]
        ),
        "currency": yahoo["currency"],

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
            "revenue": latest_revenue,
            "net_income": latest_income,
            "gross_margin": gross_margin,
            "free_cash_flow": free_cash_flow,
            "dividend_yield": dividend_yield,
            "debt_ratio": debt_ratio
        },

        "roe_15y": {
            "years": roe_rows,
            "stats": {
                "count": len(roe_values),
                "average": average_roe,
                "median": median_roe,
                "std_dev": std_roe,
                "range": range_roe,
                "max": max_roe,
                "min": min_roe
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
            "data_as_of": datetime.now(
                timezone.utc
            ).isoformat(),

            "financial_data_status": {
                "income": bool(net_income),
                "balance": bool(equity),
                "cashflow": bool(ocf)
            },

            "valuation_data_status": {
                "market_price": price is not None,
                "market_cap": market_cap is not None,
                "eps": (
                    ttm_income is not None
                ),
                "pe": pe is not None,
                "pb": pb is not None
            },

            "note": (
                "财务历史优先使用SEC XBRL数据；"
                "当前市场价格使用Yahoo Finance；"
                "市值使用当前股价×SEC最新流通股数；"
                "无法取得的数据保持为null，"
                "不进行估算或虚构。"
            )
        }
    }
