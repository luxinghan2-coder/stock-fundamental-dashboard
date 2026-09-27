from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from metrics import build_dashboard

BASE = Path(__file__).resolve().parent
app = FastAPI(title='AEL 股票基本面驾驶舱 V2.4.15', version='2.4.15')

# AEL MARKET SCAN：主动触发才执行。默认使用轻量、可控的跨市场候选池；
# 可通过环境变量 AEL_SCAN_UNIVERSE 覆盖，格式：AAPL,MSFT,600519.SS,0700.HK
DEFAULT_SCAN_UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','AVGO','AMD','TSLA','JPM','V','BRK-B',
    '0700.HK','9988.HK','3690.HK','1299.HK','0941.HK','0883.HK','1810.HK','9618.HK','0005.HK','0388.HK','2318.HK','2269.HK',
    '600519.SS','601318.SS','600036.SS','600900.SS','601888.SS','600276.SS','000858.SZ','000333.SZ','002594.SZ','300750.SZ','601398.SS','601288.SS'
]
_SCAN_CACHE = {}
_SCAN_CACHE_TTL = 300

app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/api/health')
def health():
    return {'ok': True, 'service': 'stock-fundamental-dashboard', 'version': '2.4.15'}

@app.get('/api/stock/core/{symbol}')
def stock_core(symbol: str):
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail='请输入股票代码')
    try:
        return build_dashboard(symbol, include_slow=False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'核心数据获取失败：{exc}')


@app.get('/api/stock/details/{symbol}')
def stock_details(symbol: str):
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail='请输入股票代码')
    try:
        return build_dashboard(symbol, include_slow=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'补充数据获取失败：{exc}')


def _scan_universe(markets: str):
    raw = os.getenv('AEL_SCAN_UNIVERSE', '').strip()
    universe = [x.strip() for x in raw.split(',') if x.strip()] if raw else DEFAULT_SCAN_UNIVERSE[:]
    selected = {x.strip().lower() for x in markets.split(',') if x.strip()} if markets else {'us','hk','cn'}
    def market(s):
        u=s.upper()
        if u.endswith('.HK'): return 'hk'
        if u.endswith('.SS') or u.endswith('.SZ'): return 'cn'
        return 'us'
    return [s for s in universe if market(s) in selected]


def _scan_one(symbol: str):
    try:
        d = build_dashboard(symbol, include_slow=False)
        v = d.get('valuation') or {}; f = d.get('fundamentals') or {}
        roe = v.get('roe'); pe = v.get('pe'); fcf = f.get('free_cash_flow')
        tech = d.get('technical') or {}
        strength = tech.get('score'); value = tech.get('value_score'); composite = tech.get('composite_score')
        fundamental_complete = all(x is not None for x in [roe, f.get('revenue'), f.get('net_income'), fcf])
        # 默认规则：基本面数据完整 + 至少一个技术维度达到可观察阈值。
        # 不人为估算缺失数据；缺失即不通过。
        eligible = fundamental_complete and strength is not None and value is not None and composite is not None
        if not eligible:
            return None
        return {
            'symbol': d.get('symbol') or symbol,
            'company': d.get('company') or symbol,
            'exchange': d.get('exchange') or '',
            'currency': d.get('currency') or '',
            'fundamental_ok': fundamental_complete,
            'roe': roe, 'pe': pe, 'fcf': fcf,
            'strength': strength, 'value_score': value, 'composite_score': composite,
            'state': tech.get('state') or '', 'value_state': tech.get('value_state') or '',
        }
    except Exception:
        return None


@app.get('/api/market-scan')
def market_scan(
    markets: str = Query('us,hk,cn'),
    limit: int = Query(20, ge=1, le=50),
    force: bool = Query(False),
):
    """主动市场扫描。只在用户点击时执行；结果短期缓存，避免重复扫描。"""
    universe = _scan_universe(markets)
    cache_key = f"{','.join(universe)}|{limit}"
    now = time()
    cached = _SCAN_CACHE.get(cache_key)
    if cached and not force and now - cached['ts'] < _SCAN_CACHE_TTL:
        return {**cached['data'], 'cached': True}

    results = []
    # 外部数据源请求有限并发，避免逐只串行造成等待，也避免瞬间打爆免费数据源。
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_scan_one, s): s for s in universe}
        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)

    # 固定按综合评分排序：技术强势分50% + 技术价值分50%。
    def rank_key(x):
        return x.get('composite_score') if isinstance(x.get('composite_score'), (int, float)) else -1
    results.sort(key=rank_key, reverse=True)
    results = results[:limit]
    data = {
        'ok': True,
        'scan': {
            'markets': [('美股' if m=='us' else '港股' if m=='hk' else 'A股') for m in ['us','hk','cn'] if m in {x.strip().lower() for x in markets.split(',')}],
            'universe_size': len(universe),
            'matched': len(results),
            'rules': '基本面数据完整 + 技术强势分、技术价值分、综合评分均可计算；缺失数据不估算',
            'ttl_seconds': _SCAN_CACHE_TTL,
        },
        'results': results,
        'cached': False,
    }
    _SCAN_CACHE[cache_key] = {'ts': now, 'data': data}
    return data

@app.get('/api/stock/{symbol}')
def stock_legacy(symbol: str):
    """兼容旧前端；新前端使用 core + details 两阶段加载。"""
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail='请输入股票代码')
    try:
        return build_dashboard(symbol, include_slow=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'数据获取失败：{exc}')
