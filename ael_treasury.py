"""AEL U.S. Treasury & Long-End Intelligence.

Free/public-data edition. No paid market-data terminal or API key is required.
Primary sources:
- U.S. Treasury Fiscal Data auction API
- U.S. Treasury public rates feed via FRED CSV (Federal Reserve source)

All failures are isolated to this Pro-only module.
"""
from __future__ import annotations
from datetime import datetime, timezone
import time
import requests

TIMEOUT = 8
CACHE_TTL = 300
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TREASURY_AUCTIONS = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
TREASURY_QRA = "https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding/most-recent-quarterly-refunding-documents"
TREASURY_AUCTION_QUERY = "https://www.treasurydirect.gov/auctions/auction-query/"

_CACHE = {"ts": 0.0, "data": None}


def _get(url, params=None):
    r = requests.get(url, params=params, timeout=TIMEOUT,
                     headers={"User-Agent": "AEL-Pro/2.6.8 (public-data)"})
    r.raise_for_status()
    return r


def _num(v):
    if v is None or v == "" or str(v).lower() in {"null", "none", "n/a", "na", "."}:
        return None
    try:
        return float(str(v).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _pct(accepted, denominator):
    a, d = _num(accepted), _num(denominator)
    if a is None or d in (None, 0):
        return None
    return a / d * 100.0


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _fred(series):
    try:
        r = _get(FRED_CSV.format(series=series))
        rows = []
        for line in r.text.strip().splitlines()[1:]:
            # FRED CSV is simple date,value; avoid assuming comma-free values.
            p = line.rsplit(",", 1)
            if len(p) >= 2 and p[1] not in ("", ".", "NA"):
                value = _num(p[1])
                if value is not None:
                    rows.append((p[0], value))
        if not rows:
            return {"error": "无有效观测", "series": series}
        return {"date": rows[-1][0], "value": rows[-1][1], "series": series,
                "source": "FRED / Federal Reserve (免费公开数据)", "history": rows[-40:]}
    except Exception as exc:
        return {"error": str(exc)[:160], "series": series}


def _auction_rows():
    # Treasury Fiscal Data uses security_type=Bond and security_term=30-Year.
    # No API key is required.
    params = {
        "fields": "cusip,auction_date,issue_date,maturity_date,security_type,security_term,reopening,offering_amt,accepted_comp_bid_rate_amt,high_yield,bid_to_cover_ratio,total_tendered,total_accepted,comp_accepted,primary_dealer_accepted,direct_bidder_accepted,indirect_bidder_accepted",
        "filter": "security_type:eq:Bond,security_term:eq:30-Year",
        "sort": "-auction_date",
        "page[number]": "1",
        "page[size]": "100",
        "format": "json",
    }
    try:
        data = _get(TREASURY_AUCTIONS, params).json().get("data", [])
        return data
    except Exception as exc:
        return [{"_error": str(exc)[:180]}]


def _auction_clean(x):
    def pick(*names):
        for n in names:
            if x.get(n) not in (None, "", "null"):
                return x.get(n)
        return None

    comp_accepted = pick("comp_accepted", "competitive_accepted")
    total_accepted = pick("total_accepted", "total_accepted_amt")
    total_tendered = pick("total_tendered", "total_tendered_amt")

    # FiscalData auction fields for dealer/direct/indirect are accepted dollar
    # amounts, not percentages. Convert to percentages of competitive accepted.
    indirect = pick("indirect_bidder_accepted", "indirect_bidder_accepted_amt")
    direct = pick("direct_bidder_accepted", "direct_bidder_accepted_amt")
    dealer = pick("primary_dealer_accepted", "primary_dealer_accepted_amt")
    denominator = comp_accepted if _num(comp_accepted) not in (None, 0) else total_accepted

    high_yield = pick("high_yield", "accepted_comp_bid_rate_amt", "high_rate")
    offering = pick("offering_amt", "offering_amount", "offering_amt_thousands")

    return {
        "auction_date": pick("auction_date", "record_date"),
        "issue_date": pick("issue_date"),
        "maturity_date": pick("maturity_date"),
        "security_term": pick("security_term"),
        "offering_amount": (_num(offering) * 1000 if _num(offering) is not None else None),
        "high_yield": _num(high_yield),
        "bid_to_cover": _num(pick("bid_to_cover_ratio", "bid_to_cover")),
        "indirect_pct": _pct(indirect, denominator),
        "direct_pct": _pct(direct, denominator),
        "dealer_pct": _pct(dealer, denominator),
        "comp_accepted": _num(comp_accepted),
        "total_accepted": _num(total_accepted),
        "total_tendered": _num(total_tendered),
        "raw": x,
    }


def _rank_label(value, median, higher_is_better=True):
    if value is None or median is None:
        return ("暂无数据", "unknown")
    if higher_is_better:
        d = value - median
        if d >= 0.12 * median:
            return ("偏强", "good")
        if d <= -0.12 * median:
            return ("偏弱", "bad")
    else:
        if value <= median * 0.75:
            return ("偏强", "good")
        if value >= median * 1.5:
            return ("偏弱", "bad")
    return ("一般", "neutral")


def _build():
    now = datetime.now(timezone.utc).isoformat()
    errors = []
    series = {s: _fred(s) for s in ("DGS30", "DGS20", "DGS10", "DGS5", "DGS2")}
    for s, v in series.items():
        if v and v.get("error"):
            errors.append(f"{s}: {v['error']}")

    vals = {s: (v.get("value") if v and not v.get("error") else None) for s, v in series.items()}
    spreads = {
        "30y_10y_bp": (vals["DGS30"] - vals["DGS10"]) * 100 if vals["DGS30"] is not None and vals["DGS10"] is not None else None,
        "10y_2y_bp": (vals["DGS10"] - vals["DGS2"]) * 100 if vals["DGS10"] is not None and vals["DGS2"] is not None else None,
    }

    raw = _auction_rows()
    auction_error = raw[0].get("_error") if raw and raw[0].get("_error") else None
    if auction_error:
        errors.append(f"Treasury Auction: {auction_error}")
    auctions = [_auction_clean(x) for x in raw if not x.get("_error")][:20]
    latest = auctions[0] if auctions else {}

    btc_med = _median([x.get("bid_to_cover") for x in auctions])
    ind_med = _median([x.get("indirect_pct") for x in auctions])
    btc_label = _rank_label(latest.get("bid_to_cover"), btc_med, True)
    ind_label = _rank_label(latest.get("indirect_pct"), ind_med, True)

    auction_score = []
    if btc_label[1] == "good": auction_score.append(-1)
    elif btc_label[1] == "bad": auction_score.append(1)
    if ind_label[1] == "good": auction_score.append(-1)
    elif ind_label[1] == "bad": auction_score.append(1)
    stress = "一般"
    if auction_score and sum(auction_score) >= 1: stress = "偏高"
    elif auction_score and sum(auction_score) <= -1: stress = "偏低"

    # Transparent rule layer. It does not predict prices.
    pressure = 0
    drivers = []
    if spreads["30y_10y_bp"] is not None:
        if spreads["30y_10y_bp"] > 35:
            pressure += 1; drivers.append("30年相对10年利率偏高")
        elif spreads["30y_10y_bp"] < -5:
            pressure -= 1; drivers.append("30年相对10年利率偏低")
    if btc_label[1] == "bad": pressure += 1; drivers.append("最近30年国债投标需求低于近期中位数")
    if btc_label[1] == "good": pressure -= 1; drivers.append("最近30年国债投标需求高于近期中位数")
    if ind_label[1] == "bad": pressure += 1; drivers.append("间接买家需求低于近期中位数")
    if ind_label[1] == "good": pressure -= 1; drivers.append("间接买家需求高于近期中位数")
    pressure_label = "偏高" if pressure >= 2 else ("偏低" if pressure <= -2 else "中性")

    return {
        "ok": True,
        "as_of": now,
        "free_data": True,
        "yields": vals,
        "yield_dates": {s: (v.get("date") if v else None) for s, v in series.items()},
        "spreads": spreads,
        "latest_auction": latest,
        "auction_history": auctions,
        "auction_stats": {"bid_to_cover_median": btc_med, "indirect_median": ind_med, "sample_size": len(auctions)},
        "auction_assessment": {"bid_to_cover": btc_label[0], "indirect": ind_label[0], "stress": stress},
        "long_end_pressure": {"label": pressure_label, "score": pressure, "drivers": drivers},
        "qra": {"official_url": TREASURY_QRA, "auction_query_url": TREASURY_AUCTION_QUERY,
                "next_refunding_note": "财政部季度再融资文件按官方日程更新；AEL不使用付费终端数据。"},
        "sources": {
            "rates": "FRED / Federal Reserve（免费）",
            "auctions": "U.S. Treasury Fiscal Data / TreasuryDirect（免费）",
            "qra": "U.S. Department of the Treasury（免费）",
        },
        "errors": errors,
        "method_note": "全部使用公开免费数据，无需Bloomberg/Refinitiv等付费终端。收益率来自FRED；拍卖来自美国财政部。间接/直接/交易商占比由官方接受金额按竞争性投标接受额计算。Tail需要可靠的WI基准，缺失时不估算、不伪造。长期压力为透明规则层，不是买卖建议。",
    }


def analyze_treasury():
    # 5-minute in-process cache keeps the Treasury page cheap while preserving
    # freshness. This module is only called by the Pro Treasury route.
    now = time.time()
    if _CACHE["data"] is not None and now - _CACHE["ts"] < CACHE_TTL:
        return _CACHE["data"]
    data = _build()
    _CACHE["ts"] = now
    _CACHE["data"] = data
    return data
