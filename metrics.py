import math
import re
from typing import Any
import numpy as np
import yfinance as yf


def clean_symbol(raw: str) -> str:
    s = str(raw or '').strip().upper().replace(' ', '')
    if re.fullmatch(r'\d{6}', s):
        return s + ('.SS' if s.startswith(('5','6','68','9')) else '.SZ')
    if re.fullmatch(r'\d{1,5}', s):
        return s.zfill(4) + '.HK'
    return s


def finite(v):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def pct(v):
    x = finite(v)
    return None if x is None else x * 100


def _names(names):
    return [names] if isinstance(names, str) else list(names)


def latest_row(df, names):
    if df is None or getattr(df, 'empty', True):
        return None
    for name in _names(names):
        if name in df.index:
            try:
                s = df.loc[name]
                for v in list(s):
                    x = finite(v)
                    if x is not None:
                        return x
            except Exception:
                pass
    return None


def series_value(df, names, col):
    if df is None or getattr(df, 'empty', True):
        return None
    try:
        for row in _names(names):
            if row in df.index:
                return finite(df.loc[row, col])
    except Exception:
        pass
    return None


def safe_ratio(a, b):
    a, b = finite(a), finite(b)
    if a is None or b in (None, 0):
        return None
    return a / b


def _safe_get(obj, attr, errors, default=None, *args, **kwargs):
    try:
        value = getattr(obj, attr)
        if callable(value):
            value = value(*args, **kwargs)
        return value
    except Exception as exc:
        errors[attr] = str(exc)[:240]
        return default


def annual_roe(inc, bs):
    if inc is None or inc.empty or bs is None or bs.empty:
        return []
    bs_cols = list(bs.columns)
    rows = []
    for col in list(inc.columns):
        year = getattr(col, 'year', None)
        if year is None:
            continue
        net_income = series_value(inc, ['Net Income', 'Net Income Common Stockholders'], col)
        equity_end = series_value(bs, ['Stockholders Equity', 'Common Stock Equity', 'Stockholders Equity Including Minority Interest'], col)
        prior = [c for c in bs_cols if getattr(c, 'year', None) == year - 1]
        equity_begin = None
        if prior:
            equity_begin = series_value(bs, ['Stockholders Equity', 'Common Stock Equity', 'Stockholders Equity Including Minority Interest'], prior[0])
        if net_income is None or equity_begin is None or equity_end is None:
            continue
        avg_equity = (equity_begin + equity_end) / 2
        if avg_equity == 0:
            continue
        roe = net_income / avg_equity * 100
        if math.isfinite(roe):
            rows.append({'year': int(year), 'net_income': net_income, 'equity_begin': equity_begin, 'equity_end': equity_end, 'roe': roe})
    rows.sort(key=lambda x: x['year'])
    return rows[-15:]


def stats(values):
    vals = [finite(x) for x in values if finite(x) is not None]
    if not vals:
        return {'count': 0, 'average': None, 'median': None, 'std_dev': None, 'range': None, 'max': None, 'min': None}
    arr = np.array(vals, dtype=float)
    return {'count': len(vals), 'average': float(arr.mean()), 'median': float(np.median(arr)),
            'std_dev': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'range': float(arr.max() - arr.min()), 'max': float(arr.max()), 'min': float(arr.min())}


def dividend_metrics(divs, cf, price, fcf, net_income):
    out = {'ttm_dividend_per_share': None, 'dividend_yield': None, 'history': [],
           'cagr_3y': None, 'cagr_5y': None, 'cagr_10y': None, 'consecutive_years': 0,
           'dividend_payout_ratio': None, 'fcf_payout_ratio': None, 'buybacks': None,
           'shareholder_payout': None, 'shareholder_payout_ratio': None, 'trend': '暂无数据'}
    if divs is None or getattr(divs, 'empty', True):
        return out
    try:
        divs = divs.dropna().astype(float)
        annual = divs.groupby(divs.index.year).sum()
        out['history'] = [{'year': int(y), 'dividend_per_share': float(v)} for y, v in annual.items()]
        if len(annual):
            # TTM = all cash dividends paid in the last 365 days.
            cutoff = divs.index.max() - __import__('pandas').Timedelta(days=365)
            ttm = float(divs[divs.index > cutoff].sum())
            out['ttm_dividend_per_share'] = ttm if ttm > 0 else float(annual.iloc[-1])
            if price:
                out['dividend_yield'] = out['ttm_dividend_per_share'] / price * 100

            def cagr(years):
                end_year = int(annual.index[-1]); target = end_year - years
                eligible = [int(y) for y in annual.index if int(y) <= target]
                if not eligible: return None
                start_year = max(eligible); actual = end_year - start_year
                if actual <= 0: return None
                start, end = float(annual.loc[start_year]), float(annual.iloc[-1])
                if start <= 0 or end <= 0: return None
                return ((end / start) ** (1 / actual) - 1) * 100
            out['cagr_3y'], out['cagr_5y'], out['cagr_10y'] = cagr(3), cagr(5), cagr(10)
            last = int(annual.index[-1]); count = 0
            for y in range(last, last - 30, -1):
                if y in annual.index and annual.loc[y] > 0: count += 1
                else: break
            out['consecutive_years'] = count
            if out['cagr_5y'] is not None:
                out['trend'] = '上升' if out['cagr_5y'] > 1 else ('下降' if out['cagr_5y'] < -1 else '基本稳定')
    except Exception:
        pass

    dividends_paid = latest_row(cf, ['Cash Dividends Paid', 'Common Stock Dividend Paid', 'Common Stock Payments', 'Payment Of Dividends'])
    buybacks = latest_row(cf, ['Repurchase Of Capital Stock', 'Repurchase Of Capital Stock Issuance', 'Common Stock Payments'])
    dividends_paid = abs(dividends_paid) if dividends_paid is not None else None
    buybacks = abs(buybacks) if buybacks is not None else None
    out['buybacks'] = buybacks
    if dividends_paid is not None and net_income not in (None, 0):
        out['dividend_payout_ratio'] = dividends_paid / abs(net_income) * 100
    if dividends_paid is not None and fcf not in (None, 0):
        out['fcf_payout_ratio'] = dividends_paid / abs(fcf) * 100
    if dividends_paid is not None or buybacks is not None:
        total = (dividends_paid or 0) + (buybacks or 0)
        out['shareholder_payout'] = total
        if fcf not in (None, 0):
            out['shareholder_payout_ratio'] = total / abs(fcf) * 100
    return out


def technical_analysis(history):
    out = {'score': None, 'state': '暂无数据', 'signals': [], 'indicators': {}, 'history': []}
    if history is None or getattr(history, 'empty', True) or 'Close' not in history:
        return out
    try:
        close = history['Close'].dropna().astype(float)
        if len(close) < 30: return out
        volume = history['Volume'].dropna().astype(float) if 'Volume' in history else None
        latest = float(close.iloc[-1])
        mas = {n: (float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None) for n in (20,60,120,250)}
        delta = close.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan); rsi = float((100 - 100/(1+rs)).iloc[-1]) if finite(rs.iloc[-1]) is not None else None
        ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
        macd, signal = ema12-ema26, (ema12-ema26).ewm(span=9, adjust=False).mean()
        macd_val, signal_val = float(macd.iloc[-1]), float(signal.iloc[-1])
        mid, sd = close.rolling(20).mean(), close.rolling(20).std(); upper, lower = mid+2*sd, mid-2*sd
        bb_pos = safe_ratio(latest-float(lower.iloc[-1]), float(upper.iloc[-1])-float(lower.iloc[-1])) if finite(upper.iloc[-1]) is not None and finite(lower.iloc[-1]) is not None else None
        ret20 = (latest/float(close.iloc[-21])-1)*100 if len(close)>21 else None
        vol_ratio = None
        if volume is not None and len(volume)>=20:
            av=float(volume.rolling(20).mean().iloc[-1]); vol_ratio=float(volume.iloc[-1]/av) if av else None
        high52, low52 = float(close.tail(252).max()), float(close.tail(252).min()); pos52 = (latest-low52)/(high52-low52)*100 if high52 != low52 else None
        score=50.0; signals=[]
        for n,w in [(20,10),(60,10),(120,10),(250,15)]:
            ma=mas[n]
            if ma is not None:
                score += w if latest>ma else -w; signals.append(f'MA{n}之上' if latest>ma else f'MA{n}之下')
        if rsi is not None:
            score += 8 if 50<=rsi<=70 else (-5 if rsi>75 else (2 if rsi<30 else -2))
        score += 8 if macd_val>signal_val else -8; signals.append('MACD强于信号线' if macd_val>signal_val else 'MACD弱于信号线')
        if bb_pos is not None:
            score += 4 if 0.2<=bb_pos<=0.8 else (-3 if bb_pos>0.95 else 0)
        if ret20 is not None: score += max(-5,min(5,ret20/4))
        score=max(0,min(100,round(score))); state='偏强' if score>=65 else ('中性' if score>=45 else '偏弱')
        out.update({'score':score,'state':state,'signals':signals,'indicators':{'ma20':mas[20],'ma60':mas[60],'ma120':mas[120],'ma250':mas[250],'rsi14':rsi,'macd':macd_val,'macd_signal':signal_val,'bollinger_position':bb_pos,'momentum_20d':ret20,'volume_ratio_20d':vol_ratio,'52w_high':high52,'52w_low':low52,'52w_position':pos52},'history':[{'date':str(i.date()),'close':float(v)} for i,v in close.tail(120).items()]})
    except Exception:
        pass
    return out


def analyst_view(ticker):
    out={'available':False,'rating':{},'targets':{},'earnings':{},'revenue':{},'changes':[]}
    try:
        rec=ticker.get_recommendations()
        if rec is not None and not rec.empty:
            row=rec.iloc[-1]; out['rating']={str(k):finite(v) for k,v in row.to_dict().items() if finite(v) is not None}; out['available']=True
    except Exception: pass
    try:
        x=ticker.get_analyst_price_targets() or {}; out['targets']={k:finite(v) for k,v in x.items() if finite(v) is not None}; out['available']|=bool(out['targets'])
    except Exception: pass
    for attr,key,index in [('get_earnings_estimate','earnings','0y'),('get_revenue_estimate','revenue','0y')]:
        try:
            df=getattr(ticker,attr)()
            if df is not None and not df.empty:
                row=df.loc[index] if index in df.index else df.iloc[0]; out[key]={str(k):finite(v) for k,v in row.to_dict().items() if finite(v) is not None}; out['available']=True
        except Exception: pass
    try:
        df=ticker.get_upgrades_downgrades()
        if df is not None and not df.empty:
            out['changes']=df.tail(8).reset_index().to_dict('records'); out['available']=True
    except Exception: pass
    return out


def build_dashboard(raw_symbol: str) -> dict[str, Any]:
    symbol=clean_symbol(raw_symbol); t=yf.Ticker(symbol); errors={}
    # Fetch independently. info is deliberately last/fallback because it is a large, failure-prone endpoint.
    history=_safe_get(t,'history',errors,None,period='2y',interval='1d',auto_adjust=False,repair=True)
    price=None
    if history is not None and not history.empty and 'Close' in history:
        try: price=finite(history['Close'].dropna().iloc[-1])
        except Exception: pass
    if price is None:
        fi=_safe_get(t,'fast_info',errors,{}) or {}
        price=finite(fi.get('last_price') if hasattr(fi,'get') else None)

    inc=_safe_get(t,'financials',errors,None); bs=_safe_get(t,'balance_sheet',errors,None); cf=_safe_get(t,'cashflow',errors,None)
    info=_safe_get(t,'info',errors,{}) or {}
    revenue=latest_row(inc,['Total Revenue','Operating Revenue']); net_income=latest_row(inc,['Net Income','Net Income Common Stockholders'])
    gross_profit=latest_row(inc,['Gross Profit']); gross_margin=(gross_profit/revenue*100 if gross_profit is not None and revenue not in (None,0) else pct(info.get('grossMargins')))
    assets=latest_row(bs,['Total Assets']); liabilities=latest_row(bs,['Total Liabilities Net Minority Interest','Total Liabilities']); equity=latest_row(bs,['Stockholders Equity','Common Stock Equity','Stockholders Equity Including Minority Interest'])
    debt_ratio=safe_ratio(liabilities,assets); debt_ratio=debt_ratio*100 if debt_ratio is not None else None
    roe=current_roe=None
    roe_info=finite(info.get('returnOnEquity')); current_roe=roe_info*100 if roe_info is not None else (net_income/equity*100 if net_income is not None and equity not in (None,0) else None)
    ocf=latest_row(cf,['Operating Cash Flow','Total Cash From Operating Activities']); capex=latest_row(cf,['Capital Expenditure','Capital Expenditures']); fcf=ocf+capex if ocf is not None and capex is not None else None
    shares=latest_row(bs,['Ordinary Shares Number','Share Issued','Common Stock Shares Outstanding'])
    pe=finite(info.get('trailingPE')); pb=finite(info.get('priceToBook'))
    if pe is None and price is not None and net_income is not None and shares not in (None,0): pe=safe_ratio(price,net_income/shares)
    if pb is None and price is not None and equity is not None and shares not in (None,0): pb=safe_ratio(price,equity/shares)
    divs=_safe_get(t,'dividends',errors,None); dividends=dividend_metrics(divs,cf,price,fcf,net_income)
    roe15=annual_roe(inc,bs); tech=technical_analysis(history); analysts=analyst_view(t)
    company=info.get('longName') or info.get('shortName') or symbol
    exchange=info.get('exchange') or ''; currency=info.get('currency') or ''
    source_status={'history':bool(history is not None and not getattr(history,'empty',True)),'financials':bool(inc is not None and not getattr(inc,'empty',True)),'balance_sheet':bool(bs is not None and not getattr(bs,'empty',True)),'cashflow':bool(cf is not None and not getattr(cf,'empty',True)),'dividends':bool(divs is not None and not getattr(divs,'empty',True)),'analysts':analysts['available']}
    return {'query':raw_symbol,'symbol':symbol,'company':company,'exchange':exchange,'currency':currency,'market':info.get('market'),'market_data':{'price':price,'market_cap':finite(info.get('marketCap'))},'valuation':{'pe':pe,'pb':pb,'roe':current_roe,'roe_pb':safe_ratio(current_roe,pb),'pe_roe':safe_ratio(pe,current_roe)},'fundamentals':{'revenue':revenue,'net_income':net_income,'gross_margin':gross_margin,'free_cash_flow':fcf,'debt_ratio':debt_ratio,'debt_to_equity':finite(info.get('debtToEquity'))},'dividends':dividends,'roe_15y':{'years':roe15,'stats':stats([r['roe'] for r in roe15]),'definition':'ROE = 年度净利润 / ((期初股东权益 + 期末股东权益) / 2)','std_definition':'15年有效年度ROE的样本标准差'},'technical':tech,'analysts':analysts,'source':{'provider':'Yahoo Finance via yfinance 1.7.0（免费）','status':source_status,'errors':errors,'note':'各模块独立取数；单个数据模块失败不会让整个页面失效。'}}
