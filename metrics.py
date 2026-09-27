from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests


# ============================================================
# Yahoo Finance HTTP 数据层
# 不再通过 yfinance 获取数据
# ============================================================

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


def _number(value: Any) -> Optional[float]:
    """安全转换数字。"""
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        value = float(value)

        if value != value:  # NaN
            return None

        return value
    except Exception:
        return None


def _chart_request(symbol: str, range_: str = "5d", interval: str = "1d") -> Dict[str, Any]:
    """
    直接调用 Yahoo Finance Chart API。

    优先 query1，失败后尝试 query2。
    """
    symbol = symbol.strip().upper()

    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    ]

    params = {
        "range": range_,
        "interval": interval,
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    last_error = None

    for url in urls:
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=15,
            )

            response.raise_for_status()

            data = response.json()

            chart = data.get("chart") or {}

            if chart.get("error"):
                last_error = chart["error"]
                continue

            result = chart.get("result") or []

            if not result:
                last_error = "Yahoo Finance 返回空结果"
                continue

            return result[0]

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"Yahoo Finance 数据请求失败：{last_error}"
    )


def _latest_close(result: Dict[str, Any]) -> Optional[float]:
    """从 Chart 数据取得最近一个收盘价。"""

    indicators = result.get("indicators") or {}
    quote_list = indicators.get("quote") or []

    if not quote_list:
        return None

    closes = quote_list[0].get("close") or []

    valid = [
        _number(x)
        for x in closes
        if _number(x) is not None
    ]

    if not valid:
        return None

    return valid[-1]


def _regular_market_price(result: Dict[str, Any]) -> Optional[float]:
    """取得 Yahoo 返回的当前市场价格。"""

    meta = result.get("meta") or {}

    price = _number(meta.get("regularMarketPrice"))

    if price is not None:
        return price

    return _latest_close(result)


def _dividend_yield(result: Dict[str, Any], price: Optional[float]) -> Optional[float]:
    """
    根据 Yahoo Chart API 最近一年分红计算股息率。

    返回百分比，例如：
    3.2 表示 3.2%
    """

    if price is None or price <= 0:
        return None

    events = result.get("events") or {}
    dividends = events.get("dividends") or {}

    now = datetime.now(timezone.utc).timestamp()
    one_year_ago = now - 365 * 24 * 60 * 60

    total = 0.0

    for item in dividends.values():
        try:
            timestamp = float(item.get("date"))
            amount = _number(item.get("amount"))

            if amount is None:
                continue

            if timestamp >= one_year_ago:
                total += amount

        except Exception:
            continue

    if total <= 0:
        return None

    return total / price * 100.0


# ============================================================
# 空指标
# ============================================================

def _empty_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "average": None,
        "median": None,
        "std_dev": None,
        "range": None,
        "max": None,
        "min": None,
    }


# ============================================================
# 主函数
# ============================================================

def build_dashboard(symbol: str) -> Dict[str, Any]:
    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("请输入股票代码")

    # --------------------------------------------------------
    # 第一阶段：
    # 直接从 Yahoo Chart API 获取市场基础信息
    # --------------------------------------------------------

    result = _chart_request(
        symbol,
        range_="1y",
        interval="1d",
    )

    meta = result.get("meta") or {}

    price = _regular_market_price(result)

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

    # Yahoo Chart API 有时会直接提供 marketCap
    if meta.get("marketCap") is not None:
        market_cap = _number(meta.get("marketCap"))

    dividend_yield = _dividend_yield(
        result,
        price,
    )

    # --------------------------------------------------------
    # 当前阶段：
    # 基本面指标先保持暂无数据
    #
    # 原因：
    # 我们正在先验证 Railway → Yahoo 的 HTTP 数据通道。
    # 下一步再接财务报表数据。
    # --------------------------------------------------------

    valuation = {
        "pe": None,
        "pb": None,
        "roe": None,
        "roe_pb": None,
        "pe_roe": None,
    }

    fundamentals = {
        "revenue": None,
        "net_income": None,
        "gross_margin": None,
        "free_cash_flow": None,
        "dividend_yield": dividend_yield,
        "debt_ratio": None,
    }

    return {
        "query": symbol,
        "symbol": symbol,
        "company": company,
        "exchange": exchange,
        "currency": currency,

        "market_data": {
            "price": price,
            "market_cap": market_cap,
        },

        "valuation": valuation,

        "fundamentals": fundamentals,

        "roe_15y": {
            "years": [],
            "stats": _empty_stats(),
            "definition": (
                "ROE = 年度净利润 / "
                "((期初股东权益 + 期末股东权益) / 2)"
            ),
            "std_definition": "15年有效年度ROE的样本标准差",
        },

        "source": {
            "provider": "Yahoo Finance HTTP Chart API",
            "note": (
                "当前版本直接通过 HTTP 请求 Yahoo Finance，"
                "不依赖 yfinance 获取市场数据。"
            ),
        },
    }
