"""AEL U.S. Treasury & Long-End Intelligence.

Optional Pro-only module. Official/public sources first; failures are isolated
and never block Lite/SINGLE/MARKET SCAN.
"""
from __future__ import annotations
from datetime import datetime, timezone
import re
import requests

TIMEOUT = 8
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TREASURY_AUCTIONS = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
TREASURY_QRA = "https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding/most-recent-quarterly-refunding-documents"


def _get(url, params=None):
    r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": "AEL-Pro/2.6.7"})
    r.raise_for_status()
    return r


def _num(v):
    if v is None or v == "": return None
    try: return float(str(v).replace(",", "").replace("%", ""))
    except Exception: return None


def _fred(series):
    try:
        r = _get(FRED_CSV.format(series=series))
        lines = r.text.strip().splitlines()
        rows=[]
        for line in lines[1:]:
            p=line.split(",")
            if len(p)>=2 and p[1] not in ("", ".", "NA"):
                rows.append((p[0], _num(p[1])))
        if not rows: return None
        return {"date": rows[-1][0], "value": rows[-1][1], "series": series,
                "source": "FRED / Federal Reserve", "history": rows[-40:]}
    except Exception as exc:
        return {"error": str(exc)[:160], "series": series}


def _auction_rows():
    params={
        "filter":"security_type:eq:Treasury Bond",
        "sort":"-auction_date",
        "page[size]":"100",
    }
    try:
        data=_get(TREASURY_AUCTIONS, params).json().get("data", [])
        rows=[]
        for x in data:
            term=str(x.get("security_term") or "").lower()
            if "30-year" not in term and "30 year" not in term and "30" not in term:
                continue
            rows.append(x)
        return rows
    except Exception as exc:
        return [{"_error": str(exc)[:180]}]


def _auction_clean(x):
    def pick(*names):
        for n in names:
            if x.get(n) not in (None, ""):
                return x.get(n)
        return None
    return {
        "auction_date": pick("auction_date", "record_date"),
        "issue_date": pick("issue_date"),
        "offering_amount": _num(pick("offering_amount", "offering_amt")),
        "high_yield": _num(pick("high_yield", "high_yield_percent")),
        "bid_to_cover": _num(pick("bid_to_cover_ratio", "bid_to_cover")),
        "indirect_pct": _num(pick("indirect_bidder_accepted", "indirect_bidder_accepted_pct")),
        "direct_pct": _num(pick("direct_bidder_accepted", "direct_bidder_accepted_pct")),
        "dealer_pct": _num(pick("primary_dealer_accepted", "primary_dealer_accepted_pct")),
        "raw": x,
    }


def _rank_label(value, median, higher_is_better=True):
    if value is None or median is None: return ("暂无数据", "unknown")
    if higher_is_better:
        d=value-median
        if d >= 0.12*median: return ("偏强", "good")
        if d <= -0.12*median: return ("偏弱", "bad")
    else:
        if value <= median*0.75: return ("偏强", "good")
        if value >= median*1.5: return ("偏弱", "bad")
    return ("一般", "neutral")


def analyze_treasury():
    now=datetime.now(timezone.utc).isoformat()
    errors=[]
    series={s:_fred(s) for s in ("DGS30","DGS20","DGS10","DGS5","DGS2")}
    for s,v in series.items():
        if v and v.get("error"): errors.append(f"{s}: {v['error']}")

    vals={s:(v.get("value") if v and not v.get("error") else None) for s,v in series.items()}
    spreads={
        "10y_30y_bp": (vals["DGS30"]-vals["DGS10"])*100 if vals["DGS30"] is not None and vals["DGS10"] is not None else None,
        "2y_10y_bp": (vals["DGS10"]-vals["DGS2"])*100 if vals["DGS10"] is not None and vals["DGS2"] is not None else None,
    }

    raw=_auction_rows()
    auction_error=raw[0].get("_error") if raw and raw[0].get("_error") else None
    if auction_error: errors.append(f"Treasury Auction: {auction_error}")
    auctions=[_auction_clean(x) for x in raw if not x.get("_error")][:20]
    latest=auctions[0] if auctions else {}
    btc=[x["bid_to_cover"] for x in auctions if x.get("bid_to_cover") is not None]
    ind=[x["indirect_pct"] for x in auctions if x.get("indirect_pct") is not None]
    btc_med=sorted(btc)[len(btc)//2] if btc else None
    ind_med=sorted(ind)[len(ind)//2] if ind else None
    latest_btc_label=_rank_label(latest.get("bid_to_cover"),btc_med,True)
    latest_ind_label=_rank_label(latest.get("indirect_pct"),ind_med,True)

    # Tail is deliberately not fabricated. Treasury auction results do not
    # themselves provide a universal WI reference, so AEL leaves it unfilled.
    auction_score=[]
    if latest_btc_label[1] == "good": auction_score.append(-1)
    elif latest_btc_label[1] == "bad": auction_score.append(1)
    if latest_ind_label[1] == "good": auction_score.append(-1)
    elif latest_ind_label[1] == "bad": auction_score.append(1)
    stress="一般"
    if auction_score and sum(auction_score)>=1: stress="偏高"
    elif auction_score and sum(auction_score)<=-1: stress="偏低"

    # Simple transparent long-end pressure, not a trading signal.
    pressure=0; drivers=[]
    if vals["DGS30"] is not None and vals["DGS10"] is not None and spreads["10y_30y_bp"] is not None:
        if spreads["10y_30y_bp"] > 35: pressure+=1; drivers.append("30年相对10年偏高")
        elif spreads["10y_30y_bp"] < -5: pressure-=1; drivers.append("长端相对10年偏低")
    if latest_btc_label[1]=="bad": pressure+=1; drivers.append("最近30年国债投标需求偏弱")
    if latest_btc_label[1]=="good": pressure-=1; drivers.append("最近30年国债投标需求偏强")
    if latest_ind_label[1]=="bad": pressure+=1; drivers.append("间接买家需求低于近期中位数")
    if latest_ind_label[1]=="good": pressure-=1; drivers.append("间接买家需求高于近期中位数")
    pressure_label="偏高" if pressure>=2 else ("偏低" if pressure<=-2 else "中性")

    return {
        "ok": True,
        "as_of": now,
        "yields": vals,
        "yield_dates": {s:(v.get("date") if v else None) for s,v in series.items()},
        "spreads": spreads,
        "latest_auction": latest,
        "auction_history": auctions,
        "auction_stats": {"bid_to_cover_median":btc_med,"indirect_median":ind_med,"sample_size":len(auctions)},
        "auction_assessment": {"bid_to_cover":latest_btc_label[0],"indirect":latest_ind_label[0],"stress":stress},
        "long_end_pressure": {"label":pressure_label,"score":pressure,"drivers":drivers},
        "qra": {"official_url":TREASURY_QRA,"next_refunding_note":"财政部季度再融资文件按官方日程更新；AEL不从二手媒体推断发债规模。"},
        "sources": {
            "rates":"FRED / Federal Reserve",
            "auctions":"U.S. Treasury Fiscal Data / Treasury Auction Query",
            "qra":"U.S. Department of the Treasury Quarterly Refunding",
        },
        "errors":errors,
        "method_note":"官方数据优先；收益率来自FRED，拍卖来自美国财政部。尾部利差需要可靠的WI基准，缺失时不估算、不伪造。长期压力只是AEL透明规则层，不是买卖建议。",
    }
