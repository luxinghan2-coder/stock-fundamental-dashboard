from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
import math
import re
import threading
import time
from typing import Any

import pandas as pd
import requests

FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
FED_FUNDS_RATE_URL = 'https://www.federalreserve.gov/FOMC/fundsrate.htm'
FOMC_CALENDAR_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

SERIES = {
    'fedfunds': ('FEDFUNDS', '联邦基金有效利率', '%', 'daily'),
    'dgs2': ('DGS2', '美国2年期国债收益率', '%', 'daily'),
    'dgs10': ('DGS10', '美国10年期国债收益率', '%', 'daily'),
    'curve_10y2y': ('T10Y2Y', '10Y-2Y期限利差', '%', 'daily'),
    'cpi': ('CPIAUCSL', 'CPI消费者价格指数', 'index', 'monthly'),
    'unemployment': ('UNRATE', '失业率', '%', 'monthly'),
    'payrolls': ('PAYEMS', '非农就业人数', 'thousand', 'monthly'),
    'industrial_production': ('INDPRO', '工业生产指数', 'index', 'monthly'),
    'hy_spread': ('BAMLH0A0HYM2', '美国高收益债利差', '%', 'daily'),
    'broad_usd': ('DTWEXBGS', '美元广义指数', 'index', 'daily'),
}

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 3600
_HTTP_TIMEOUT = 12


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _clean_html(text: str) -> str:
    text = re.sub(r'<script.*?</script>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _cached_get(key: str, loader):
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and now - item['ts'] < _CACHE_TTL:
            return item['value']
    value = loader()
    with _CACHE_LOCK:
        _CACHE[key] = {'ts': now, 'value': value}
    return value


def _fred_series(series_id: str, limit_years: int = 12) -> pd.DataFrame:
    def load():
        params = {'id': series_id, 'cosd': f'{datetime.now().year - limit_years}-01-01'}
        r = requests.get(FRED_BASE, params=params, timeout=_HTTP_TIMEOUT,
                         headers={'User-Agent': 'AEL-Macro-Lab/1.0'})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if 'DATE' not in df.columns or series_id not in df.columns:
            raise RuntimeError(f'FRED {series_id} 返回字段不完整')
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna(subset=['DATE', series_id]).sort_values('DATE')
        if df.empty:
            raise RuntimeError(f'FRED {series_id} 无有效数据')
        return df[['DATE', series_id]].reset_index(drop=True)
    return _cached_get(f'fred:{series_id}:{limit_years}', load)


def _latest_pair(df: pd.DataFrame, col: str):
    x = df[['DATE', col]].dropna()
    if x.empty:
        return None
    last = x.iloc[-1]
    prev = x.iloc[-2] if len(x) >= 2 else None
    return {
        'date': str(last['DATE'].date()),
        'value': _num(last[col]),
        'previous_date': str(prev['DATE'].date()) if prev is not None else None,
        'previous': _num(prev[col]) if prev is not None else None,
    }


def _period_change(df: pd.DataFrame, col: str, periods: int):
    x = df[['DATE', col]].dropna()
    if len(x) <= periods:
        return None
    return _num(x.iloc[-1][col] - x.iloc[-1 - periods][col])


def _yoy(df: pd.DataFrame, col: str, observations: int):
    x = df[['DATE', col]].dropna()
    if len(x) <= observations:
        return None
    old = _num(x.iloc[-1 - observations][col])
    new = _num(x.iloc[-1][col])
    if old in (None, 0) or new is None:
        return None
    return (new / old - 1) * 100


def _fred_payload(key: str, series_id: str, unit: str, frequency: str) -> dict[str, Any]:
    df = _fred_series(series_id)
    latest = _latest_pair(df, series_id)
    if latest is None:
        raise RuntimeError(f'{series_id} 无最新观测值')
    if frequency == 'monthly':
        observations_per_year = 12
        mom = _period_change(df, series_id, 1)
        yoy = _yoy(df, series_id, observations_per_year)
        change_3m = _period_change(df, series_id, 3)
    else:
        observations_per_year = 252
        mom = _period_change(df, series_id, 21)
        yoy = _period_change(df, series_id, 252)
        change_3m = _period_change(df, series_id, 63)
    return {
        'series_id': series_id,
        'name': SERIES[key][1],
        'unit': unit,
        'frequency': frequency,
        'latest': latest,
        'change_3m': change_3m,
        'change_yoy': yoy,
        'observations': int(len(df)),
        'source': 'FRED',
        'source_url': f'{FRED_BASE}?id={series_id}',
    }


def _fed_policy() -> dict[str, Any]:
    def load():
        r = requests.get(FED_FUNDS_RATE_URL, timeout=_HTTP_TIMEOUT,
                         headers={'User-Agent': 'AEL-Macro-Lab/1.0'})
        r.raise_for_status()
        text = _clean_html(r.text)
        # The current-year table is rendered as date / increase / decrease / level.
        year = datetime.now().year
        year_pos = text.find(str(year))
        scope = text[year_pos:year_pos + 12000] if year_pos >= 0 else text[:12000]
        months = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        matches = re.findall(rf'({months})\s+(\d{{1,2}})\s+(\d{{1,3}})\s+(\d{{1,3}})\s+([\d.]+-[\d.]+)', scope)
        if not matches:
            raise RuntimeError('Federal Reserve funds-rate page 未解析到当前政策区间')
        month_no = {m: i for i, m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'], 1)}
        rows=[]
        for mon, day, inc, dec, level in matches:
            try:
                dt = datetime(year, month_no[mon], int(day), tzinfo=timezone.utc)
            except Exception:
                continue
            rows.append({'date': dt.date().isoformat(), 'increase_bp': int(inc), 'decrease_bp': int(dec), 'level': level})
        if not rows:
            raise RuntimeError('Federal Reserve funds-rate page 日期解析失败')
        latest=max(rows, key=lambda x:x['date'])
        return {'latest':latest,'history':rows[-8:], 'source':'Federal Reserve', 'source_url':FED_FUNDS_RATE_URL}
    return _cached_get('fed:policy', load)


def _fomc_calendar() -> dict[str, Any]:
    def load():
        r = requests.get(FOMC_CALENDAR_URL, timeout=_HTTP_TIMEOUT,
                         headers={'User-Agent': 'AEL-Macro-Lab/1.0'})
        r.raise_for_status()
        text = _clean_html(r.text)
        year = datetime.now().year
        start_marker = f'{year} FOMC Meetings'
        year_pos = text.find(start_marker)
        scope = text[year_pos:] if year_pos >= 0 else text[:16000]
        next_year_marker = f'{year-1} FOMC Meetings'
        next_pos = scope.find(next_year_marker)
        if next_pos > 0:
            scope = scope[:next_pos]
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        pattern = r'(' + '|'.join(months) + r')\s+(\d{1,2})(?:-(\d{1,2}))?'
        meetings=[]
        for mon, d1, d2 in re.findall(pattern, scope):
            dt1=datetime(year, months.index(mon)+1, int(d1)).date()
            dt2=datetime(year, months.index(mon)+1, int(d2 or d1)).date()
            meetings.append({'start':dt1.isoformat(),'end':dt2.isoformat(),'label':f'{mon} {d1}' + (f'-{d2}' if d2 else '')})
        # Keep unique meeting entries; the calendar page contains other date-like text too.
        unique=[]
        seen=set()
        for m in meetings:
            if m['start'] in seen: continue
            seen.add(m['start']); unique.append(m)
        unique=sorted(unique, key=lambda x:x['start'])
        today=datetime.now(timezone.utc).date()
        upcoming=next((m for m in unique if datetime.fromisoformat(m['end']).date() >= today), None)
        return {'year':year,'meetings':unique,'next_meeting':upcoming,'source':'Federal Reserve FOMC calendar','source_url':FOMC_CALENDAR_URL}
    return _cached_get('fed:fomc-calendar', load)


def _regime(series: dict[str, Any]) -> dict[str, Any]:
    def val(k):
        try: return series[k]['latest']['value']
        except Exception: return None
    def ch(k, field):
        try: return series[k][field]
        except Exception: return None
    cpi_yoy=ch('cpi','change_yoy')
    unemp=val('unemployment')
    curve=val('curve_10y2y')
    hy=val('hy_spread')
    fed_change=ch('fedfunds','change_yoy')
    inflation = '暂无数据'
    if cpi_yoy is not None:
        inflation = '通胀偏高' if cpi_yoy >= 3 else ('通胀温和' if cpi_yoy >= 2 else '通胀偏低')
    labor = '暂无数据'
    if unemp is not None:
        labor = '就业偏弱' if unemp >= 5 else ('就业正常' if unemp >= 4 else '就业偏紧')
    curve_state = '暂无数据'
    if curve is not None:
        curve_state = '曲线倒挂' if curve < 0 else '曲线正斜率'
    credit = '暂无数据'
    if hy is not None:
        credit = '信用利差偏高' if hy >= 5 else ('信用利差中性' if hy >= 3 else '信用利差较低')
    policy = '暂无数据'
    if fed_change is not None:
        policy = '政策利率较一年前上升' if fed_change > 0.05 else ('政策利率较一年前下降' if fed_change < -0.05 else '政策利率大致持平')
    return {'inflation':inflation,'labor':labor,'yield_curve':curve_state,'credit':credit,'policy':policy}


def analyze_macro() -> dict[str, Any]:
    errors=[]
    data={}
    # Independent series retrieval: one unavailable series must not erase the rest.
    def fetch(item):
        key, meta = item
        sid, _, unit, freq = meta
        return key, _fred_payload(key, sid, unit, freq)
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix='ael-macro') as pool:
        futures={pool.submit(fetch,item):(item[0]) for item in SERIES.items()}
        for future in as_completed(futures):
            key=futures[future]
            try:
                k,payload=future.result(); data[k]=payload
            except Exception as exc:
                errors.append(f'{key}: {str(exc)[:180]}')
    policy=None; calendar=None
    try: policy=_fed_policy()
    except Exception as exc: errors.append(f'Fed政策: {str(exc)[:180]}')
    try: calendar=_fomc_calendar()
    except Exception as exc: errors.append(f'FOMC日历: {str(exc)[:180]}')
    regime=_regime(data)
    return {
        'as_of':datetime.now(timezone.utc).isoformat(),
        'policy':policy,
        'fomc_calendar':calendar,
        'series':data,
        'regime':regime,
        'data_quality':{
            'series_available':len(data),'series_total':len(SERIES),'missing_series':[k for k in SERIES if k not in data],
            'errors':errors,
        },
        'method_note':'宏观模块只展示真实的FRED与Federal Reserve数据，并做透明的环比/同比变化计算；不把宏观状态偷偷加减到Lite股票分数。暂无数据不会用估算值补齐。',
    }
