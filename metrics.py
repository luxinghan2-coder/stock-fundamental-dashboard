from __future__ import annotations

from datetime import datetime, timezone
from statistics import median, stdev
from typing import Any, Dict, Optional

import requests


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
})


# ============================================================
# 基础工具
# ============================================================

def num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        if isinstance(value, dict):
            if "raw" in value:
                value = value["raw"]
            elif "reportedValue" in value:
                value = value["reportedValue"]
            else:
                return None

        value = float(value)

        if value != value:
            return None

        return value

    except Exception:
        return None


def first_number(data: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = num(data.get(key))
        if value is not None:
            return value
    return None


# ============================================================
# Yahoo Chart
# ============================================================

def yahoo_chart(symbol: str, range_: str = "1y") -> Dict[str, Any]:

    symbol = symbol.upper().strip()

    for host in [
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    ]:

        url = (
            f"https://{host}/v8/finance/chart/"
            f"{symbol}"
        )

        params = {
            "range": range_,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=15,
            )

            response.raise_for_status()

            data = response.json()

            result = (
                data.get("chart", {})
                .get("result")
                or []
            )

            if result:
                return result[0]

        except Exception:
            continue

    raise RuntimeError(
        f"无法取得 Yahoo 行情数据：{symbol}"
    )


# ============================================================
# Yahoo Fundamentals Time Series
# ============================================================

def yahoo_timeseries(
    symbol: str,
    types: list[str],
    period1: int,
    period2: int,
) -> Dict[str, Any]:

    symbol = symbol.upper().strip()

    type_string = ",".join(types)

    hosts = [
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    ]

    last_error = None

    for host in hosts:

        url = (
            f"https://{host}/ws/fundamentals-timeseries/"
            f"v1/finance/timeseries/{symbol}"
        )

        params = {
            "symbol": symbol,
            "type": type_string,
            "period1": period1,
            "period2": period2,
        }

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            data = response.json()

            if data.get("timeseries"):
                return data

            last_error = str(data)

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"Yahoo 财务数据请求失败：{last_error}"
    )


# ============================================================
# 解析 Yahoo Time Series
# ============================================================

def parse_timeseries(
    response: Dict[str, Any],
) -> list[Dict[str, Any]]:

    result = (
        response
        .get("timeseries", {})
        .get("result")
        or []
    )

    rows = []

    for block in result:

        if not isinstance(block, dict):
            continue

        timestamps = block.get("timestamp")

        if not timestamps:
            continue

        dates = block.get("date") or []

        # Yahoo 的常见返回形式：
        #
        # {
        #   "timestamp": [...],
        #   "annualTotalRevenue": [...]
        # }
        #
        # 这里把每个 timestamp 展开成一行。

        for i, ts in enumerate(timestamps):

            row = {
                "timestamp": ts,
            }

            if i < len(dates):
                row["date"] = dates[i]

            for key, value in block.items():

                if key in (
                    "timestamp",
                    "meta",
                    "symbol",
                ):
                    continue

                if isinstance(value, list):

                    if i < len(value):
                        item = value[i]

                        if isinstance(item, dict):
                            if "raw" in item:
                                row[key] = item["raw"]
                            elif "reportedValue" in item:
                                rv = item["reportedValue"]

                                if isinstance(rv, dict):
                                    row[key] = rv.get("raw")
                                else:
                                    row[key] = rv
                            else:
                                row[key] = item
                        else:
                            row[key] = item

                elif isinstance(value, dict):

                    if "raw" in value:
                        row[key] = value["raw"]

            rows.append(row)

    return rows


def merge_rows(
    response: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:

    result = (
        response
        .get("timeseries", {})
        .get("result")
        or []
    )

    years: Dict[int, Dict[str, Any]] = {}

    for block in result:

        if not isinstance(block, dict):
            continue

        timestamps = block.get("timestamp") or []

        for i, ts in enumerate(timestamps):

            try:
                year = datetime.fromtimestamp(
                    ts,
                    tz=timezone.utc,
                ).year
            except Exception:
                continue

            if year not in years:
                years[year] = {
                    "year": year
                }

            target = years[year]

            for key, value in block.items():

                if key == "timestamp":
                    continue

                if not isinstance(value, list):
                    continue

                if i >= len(value):
                    continue

                item = value[i]

                if isinstance(item, dict):

                    if "raw" in item:
                        target[key] = num(item["raw"])

                    elif "reportedValue" in item:

                        rv = item["reportedValue"]

                        if isinstance(rv, dict):
                            target[key] = num(
                                rv.get("raw")
                            )
                        else:
                            target[key] = num(rv)

                else:
                    target[key] = num(item)

    return years


# ============================================================
# 获取公司基本财务数据
# ============================================================

def get_fundamentals(symbol: str) -> Dict[str, Any]:

    now = datetime.now(timezone.utc)

    start = datetime(
        now.year - 16,
        1,
        1,
        tzinfo=timezone.utc,
    )

    period1 = int(start.timestamp())
    period2 = int(now.timestamp())

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

    all_types = (
        income_types
        + balance_types
        + cashflow_types
    )

    response = yahoo_timeseries(
        symbol,
        all_types,
        period1,
        period2,
    )

    years = merge_rows(response)

    return {
        "years": years
    }


# ============================================================
# 取得估值数据
# ============================================================

def get_valuation(symbol: str) -> Dict[str, Any]:

    now = datetime.now(timezone.utc)

    start = datetime(
        now.year - 2,
        1,
        1,
        tzinfo=timezone.utc,
    )

    types = [
        "trailingMarketCap",
        "trailingEnterpriseValue",
        "trailingPeRatio",
        "trailingPbRatio",
        "trailingPsRatio",
        "annualMarketCap",
        "annualPeRatio",
        "annualPbRatio",
    ]

    try:

        response = yahoo_timeseries(
            symbol,
            types,
            int(start.timestamp()),
            int(now.timestamp()),
        )

        result = (
            response
            .get("timeseries", {})
            .get("result")
            or []
        )

        latest = {}

        for block in result:

            for key in types:

                values = block.get(key)

                if not values:
                    continue

                if isinstance(values, list):

                    item = values[-1]

                    if isinstance(item, dict):

                        if "reportedValue" in item:

                            rv = item["reportedValue"]

                            if isinstance(rv, dict):
                                latest[key] = num(
                                    rv.get("raw")
                                )
                            else:
                                latest[key] = num(rv)

                        elif "raw" in item:
                            latest[key] = num(item["raw"])

        return latest

    except Exception:

        return {}


# ============================================================
# 当前价格
# ============================================================

def get_price_data(
    symbol: str,
) -> Dict[str, Any]:

    result = yahoo_chart(
        symbol,
        "5d",
    )

    meta = result.get("meta") or {}

    price = num(
        meta.get("regularMarketPrice")
    )

    if price is None:

        indicators = (
            result
            .get("indicators", {})
            .get("quote")
            or []
        )

        if indicators:

            closes = (
                indicators[0]
                .get("close")
                or []
            )

            for value in reversed(closes):

                value = num(value)

                if value is not None:
                    price = value
                    break

    return {
        "price": price,
        "company": (
            meta.get("longName")
            or meta.get("shortName")
            or symbol
        ),
        "exchange": (
            meta.get("fullExchangeName")
            or meta.get("exchangeName")
        ),
        "currency": meta.get("currency"),
        "market_cap": num(
            meta.get("marketCap")
        ),
    }


# ============================================================
# 股息
# ============================================================

def get_dividend_yield(
    symbol: str,
    price: Optional[float],
) -> Optional[float]:

    if price is None or price <= 0:
        return None

    result = yahoo_chart(
        symbol,
        "1y",
    )

    events = result.get("events") or {}

    dividends = (
        events.get("dividends")
        or {}
    )

    total = 0.0

    one_year_ago = (
        datetime.now(timezone.utc).timestamp()
        - 365 * 24 * 60 * 60
    )

    for item in dividends.values():

        try:

            timestamp = float(
                item.get("date")
            )

            amount = num(
                item.get("amount")
            )

            if (
                amount is not None
                and timestamp >= one_year_ago
            ):
                total += amount

        except Exception:
            continue

    if total <= 0:
        return None

    return (
        total / price * 100
    )


# ============================================================
# 计算 ROE
# ============================================================

def calculate_roe(
    net_income: Optional[float],
    beginning_equity: Optional[float],
    ending_equity: Optional[float],
) -> Optional[float]:

    if (
        net_income is None
        or beginning_equity is None
        or ending_equity is None
    ):
        return None

    average_equity = (
        beginning_equity
        + ending_equity
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

def build_dashboard(
    symbol: str,
) -> Dict[str, Any]:

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError(
            "请输入股票代码"
        )

    # --------------------------------------------------------
    # 市场数据
    # --------------------------------------------------------

    market = get_price_data(symbol)

    price = market["price"]

    # --------------------------------------------------------
    # 财务数据
    # --------------------------------------------------------

    fundamentals_data = get_fundamentals(symbol)

    years = fundamentals_data["years"]

    # --------------------------------------------------------
    # 估值
    # --------------------------------------------------------

    valuation_data = get_valuation(symbol)

    pe = first_number(
        valuation_data,
        "trailingPeRatio",
        "annualPeRatio",
    )

    pb = first_number(
        valuation_data,
        "trailingPbRatio",
        "annualPbRatio",
    )

    # --------------------------------------------------------
    # 最新年度财务数据
    # --------------------------------------------------------

    valid_years = sorted(
        years.keys()
    )

    latest_year = (
        valid_years[-1]
        if valid_years
        else None
    )

    latest = (
        years[latest_year]
        if latest_year is not None
        else {}
    )

    revenue = first_number(
        latest,
        "annualTotalRevenue",
        "annualOperatingRevenue",
    )

    net_income = first_number(
        latest,
        "annualNetIncome",
        "annualNetIncomeCommonStockholders",
    )

    gross_profit = first_number(
        latest,
        "annualGrossProfit",
    )

    free_cash_flow = first_number(
        latest,
        "annualFreeCashFlow",
    )

    total_assets = first_number(
        latest,
        "annualTotalAssets",
    )

    total_liabilities = first_number(
        latest,
        "annualTotalLiabilitiesNetMinorityInterest",
    )

    total_debt = first_number(
        latest,
        "annualTotalDebt",
    )

    equity = first_number(
        latest,
        "annualStockholdersEquity",
        "annualCommonStockEquity",
    )

    # --------------------------------------------------------
    # 毛利率
    # --------------------------------------------------------

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
    # 负债率
    #
    # 定义：
    # 总负债 / 总资产
    # --------------------------------------------------------

    debt_ratio = None

    if (
        total_liabilities is not None
        and total_assets is not None
        and total_assets != 0
    ):
        debt_ratio = (
            total_liabilities
            / total_assets
            * 100
        )

    # --------------------------------------------------------
    # 最新 ROE
    # --------------------------------------------------------

    current_roe = None

    if latest_year is not None:

        previous_years = [
            y
            for y in valid_years
            if y < latest_year
        ]

        if previous_years:

            previous_year = previous_years[-1]

            previous = years[
                previous_year
            ]

            beginning_equity = first_number(
                previous,
                "annualStockholdersEquity",
                "annualCommonStockEquity",
            )

            current_roe = calculate_roe(
                net_income,
                beginning_equity,
                equity,
            )

    # --------------------------------------------------------
    # 15 年 ROE
    # --------------------------------------------------------

    roe_rows = []

    for year in valid_years:

        current = years[year]

        net = first_number(
            current,
            "annualNetIncome",
            "annualNetIncomeCommonStockholders",
        )

        eq_end = first_number(
            current,
            "annualStockholdersEquity",
            "annualCommonStockEquity",
        )

        previous_years = [
            y
            for y in valid_years
            if y < year
        ]

        if not previous_years:
            continue

        previous_year = previous_years[-1]

        previous = years[
            previous_year
        ]

        eq_begin = first_number(
            previous,
            "annualStockholdersEquity",
            "annualCommonStockEquity",
        )

        roe = calculate_roe(
            net,
            eq_begin,
            eq_end,
        )

        if roe is None:
            continue

        roe_rows.append({
            "year": year,
            "roe": roe,
        })

    # 最近 15 个有效年度

    roe_rows = roe_rows[-15:]

    roe_values = [
        row["roe"]
        for row in roe_rows
    ]

    stats = {
        "count": len(roe_values),
        "average": None,
        "median": None,
        "std_dev": None,
        "range": None,
        "max": None,
        "min": None,
    }

    if roe_values:

        stats["average"] = (
            sum(roe_values)
            / len(roe_values)
        )

        stats["median"] = median(
            roe_values
        )

        if len(roe_values) >= 2:
            stats["std_dev"] = stdev(
                roe_values
            )
        else:
            stats["std_dev"] = 0.0

        stats["max"] = max(
            roe_values
        )

        stats["min"] = min(
            roe_values
        )

        stats["range"] = (
            stats["max"]
            - stats["min"]
        )

    # --------------------------------------------------------
    # 派生指标
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

    dividend_yield = None

    try:
        dividend_yield = get_dividend_yield(
            symbol,
            price,
        )
    except Exception:
        dividend_yield = None

    # --------------------------------------------------------
    # 返回
    # --------------------------------------------------------

    return {
        "query": symbol,
        "symbol": symbol,

        "company": market["company"],
        "exchange": market["exchange"],
        "currency": market["currency"],

        "market_data": {
            "price": price,
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
            "provider": (
                "Yahoo Finance HTTP "
                "Chart + Fundamentals Time Series API"
            ),
            "note": (
                "直接通过 HTTP 获取 Yahoo "
                "Finance 财务数据；"
                "不依赖 yfinance 财务接口。"
            ),
        },

        "data_as_of": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }
