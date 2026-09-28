from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from time import time
from datetime import datetime, timezone
import requests
import yfinance as yf
import pandas as pd
from metrics import build_dashboard, technical_analysis, fibonacci_levels, pivot_levels

BASE = Path(__file__).resolve().parent
APP_VERSION = '2.5.4'
app = FastAPI(title='AEL 股票基本面驾驶舱 V2.5.4', version=APP_VERSION)

# MARKET SCAN is deliberately separated from SINGLE. The scanner only pulls
# lightweight market-directory metadata plus batched daily history; it never
# calls build_dashboard() for thousands of symbols.
DEFAULT_SCAN_UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','AVGO','AMD','TSLA','JPM','V','BRK-B',
    '0700.HK','9988.HK','3690.HK','1299.HK','0941.HK','0883.HK','1810.HK','9618.HK','0005.HK','0388.HK','2318.HK','2269.HK',
    '600519.SS','601318.SS','600036.SS','600900.SS','601888.SS','600276.SS','000858.SZ','000333.SZ','002594.SZ','300750.SZ','601398.SS','601288.SS'
]

# V2.5.4: filter the universe before history download, then scan history in
# bounded parallel batches. Four workers avoids the latency of serial scanning
# without turning the Yahoo session into an uncontrolled request fan-out.
SCAN_BATCH_SIZE = max(50, min(250, int(os.getenv('AEL_SCAN_BATCH_SIZE', '250'))))
SCAN_WORKERS = max(1, min(6, int(os.getenv('AEL_SCAN_WORKERS', '4'))))
SCAN_CACHE_TTL = int(os.getenv('AEL_SCAN_CACHE_TTL', '900'))
UNIVERSE_CACHE_TTL = int(os.getenv('AEL_UNIVERSE_CACHE_TTL', '3600'))
SCREENER_PAGE_SIZE = 250
# 0 = follow Yahoo's reported total dynamically; no artificial 12,000-symbol cap.
# A positive value remains available as an emergency operator override.
SCREENER_MAX_SYMBOLS = int(os.getenv('AEL_SCAN_MAX_SYMBOLS', '0'))

# Primary MARKET SCAN market-cap gates. Yahoo's screener values are scoped
# to the selected regional market, so the thresholds below are expressed in
# that market's local currency: USD / HKD / CNY respectively.
MARKET_CAP_MIN = {'us': 20_000_000_000, 'hk': 30_000_000_000, 'cn': 30_000_000_000}
MARKET_CAP_CURRENCY = {'us': 'USD', 'hk': 'HKD', 'cn': 'CNY'}
# Yahoo exchange codes used for OTC/Pink Sheet listings. These are isolated
# from the primary US scan and therefore cannot consume TOP20 or scan time.
US_OTC_EXCHANGES = {'PNK', 'OQB', 'OQX', 'OTC'}

_SCAN_CACHE = {}
_UNIVERSE_CACHE = {}
_SCAN_JOBS = {}
_SCAN_LOCK = threading.Lock()

app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/api/health')
def health():
    return {'ok': True, 'service': 'stock-fundamental-dashboard', 'version': APP_VERSION}

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


def market_of(symbol: str):
    u = str(symbol).upper()
    if u.endswith('.HK'): return 'hk'
    if u.endswith('.SS') or u.endswith('.SZ'): return 'cn'
    return 'us'


def _scan_universe_override(markets: str):
    raw = os.getenv('AEL_SCAN_UNIVERSE', '').strip()
    if not raw:
        return None
    selected = {x.strip().lower() for x in markets.split(',') if x.strip()} if markets else {'us','hk','cn'}
    rows=[]
    for s in raw.split(','):
        symbol=s.strip()
        if symbol and market_of(symbol) in selected:
            rows.append({'symbol':symbol,'company':symbol,'sector':'未分类','industry':'未分类','exchange':'','currency':'','market_cap':None})
    return rows


def _yahoo_screener_page(region: str, offset: int = 0, size: int = SCREENER_PAGE_SIZE):
    """Fetch one Yahoo equity-screener page through yfinance's managed session.

    Do not call query2.finance.yahoo.com directly here. Yahoo binds crumb to
    the cookie/session that obtained it; yfinance's YfData manages that pair.
    This is especially important on stateless Railway containers, where a
    manually assembled crumb/cookie request can return HTTP 401 Invalid Crumb.
    """
    if size < 1 or size > 250:
        raise ValueError('Yahoo screener page size must be between 1 and 250')
    try:
        # EquityQuery is already an equity screener; `quoteType` is not a
        # valid EquityQuery field in current yfinance and causes Railway to
        # fail before the request is sent. Region is sufficient to scope the
        # equity universe, while the screener endpoint itself returns equity
        # quotes.
        query = yf.EquityQuery('eq', ['region', region.lower()])
        result = yf.screen(
            query,
            offset=int(offset),
            size=int(size),
            sortField='ticker',
            sortAsc=True,
        )
    except Exception as exc:
        raise RuntimeError(f'Yahoo screener authentication/query failed: {exc}') from exc
    if not isinstance(result, dict):
        raise RuntimeError('Yahoo screener returned an invalid response')
    quotes = result.get('quotes') or []
    if not isinstance(quotes, list):
        raise RuntimeError('Yahoo screener returned invalid quotes')
    return result

def _discover_market(region: str):
    cache=_UNIVERSE_CACHE.get(region)
    if cache and time()-cache['ts'] < UNIVERSE_CACHE_TTL:
        return cache['rows']
    rows=[]; offset=0; total=None
    while True:
        result=_yahoo_screener_page(region, offset, SCREENER_PAGE_SIZE)
        quotes=result.get('quotes') or []
        total=int(result.get('total') or 0)
        if not quotes: break
        for q in quotes:
            symbol=str(q.get('symbol') or '').strip().upper()
            if not symbol: continue
            rows.append({
                'symbol':symbol,
                'company':q.get('longName') or q.get('shortName') or symbol,
                'sector':q.get('sector') or '未分类',
                'industry':q.get('industry') or '未分类',
                'exchange':q.get('exchange') or '',
                'currency':q.get('currency') or '',
                'market_cap':q.get('marketCap'),
            })
        offset += len(quotes)
        if len(quotes) < SCREENER_PAGE_SIZE or (total and offset >= total): break
        if SCREENER_MAX_SYMBOLS > 0 and offset >= SCREENER_MAX_SYMBOLS: break
    # De-duplicate while preserving directory order.
    dedup={r['symbol']:r for r in rows}
    rows=list(dedup.values())
    if SCREENER_MAX_SYMBOLS > 0:
        rows=rows[:SCREENER_MAX_SYMBOLS]
    _UNIVERSE_CACHE[region]={'ts':time(),'rows':rows}
    return rows


def _is_us_otc(row):
    return str(row.get('exchange') or '').upper().strip() in US_OTC_EXCHANGES


def _passes_market_cap(row, market):
    value=row.get('market_cap')
    try:
        cap=float(value)
    except (TypeError, ValueError):
        return False
    return cap >= MARKET_CAP_MIN[market]


def _filter_primary_universe(rows, market):
    primary=[]
    otc=[]
    rejected_cap=0
    for raw in rows:
        row=dict(raw)
        if market == 'us' and _is_us_otc(row):
            otc.append(row)
            continue
        if not _passes_market_cap(row, market):
            rejected_cap += 1
            continue
        row['market']=market
        primary.append(row)
    return primary, otc, rejected_cap


def _scan_universe(markets: str):
    override=_scan_universe_override(markets)
    if override is not None:
        return override, 'environment override', {'us':0,'hk':0,'cn':0}, {'us':0,'hk':0,'cn':0}
    selected=[m.strip().lower() for m in markets.split(',') if m.strip() in {'us','hk','cn'}]
    if not selected: selected=['us','hk','cn']
    regions={'us':'us','hk':'hk','cn':'cn'}
    rows=[]
    otc_counts={'us':0,'hk':0,'cn':0}
    cap_rejected={'us':0,'hk':0,'cn':0}
    for mk in selected:
        market_rows=_discover_market(regions[mk])
        primary, otc, rejected_cap = _filter_primary_universe(market_rows, mk)
        otc_counts[mk]=len(otc)
        cap_rejected[mk]=rejected_cap
        for r in primary:
            r=dict(r); r['market']=mk; rows.append(r)
    return rows, 'Yahoo Finance screener directory → market-cap filter → OTC isolation', otc_counts, cap_rejected


def _extract_history(frame, symbol):
    if frame is None or getattr(frame,'empty',True): return None
    try:
        if isinstance(frame.columns, pd.MultiIndex):
            # group_by=ticker normally gives (ticker, field); tolerate the
            # alternative (field, ticker) layout as well.
            if symbol in frame.columns.get_level_values(0):
                frame=frame[symbol]
            elif symbol in frame.columns.get_level_values(-1):
                frame=frame.xs(symbol, axis=1, level=-1)
        if 'Close' not in frame.columns: return None
        return frame.dropna(subset=['Close']).copy()
    except Exception:
        return None


def _scan_history_batch(symbols):
    if not symbols: return {}, {}
    try:
        data=yf.download(symbols, period='2y', interval='1d', auto_adjust=False,
                         group_by='ticker', threads=False, progress=False, repair=False, timeout=20)
        histories={s:_extract_history(data,s) for s in symbols}
        errors={s:'历史行情为空或字段不完整' for s,h in histories.items() if h is None}
        return histories, errors
    except Exception as exc:
        msg=f'批量历史行情请求失败：{str(exc)[:180]}'
        return {s:None for s in symbols}, {s:msg for s in symbols}


def _scan_batch(rows):
    symbols=[r['symbol'] for r in rows]
    histories, errors=_scan_history_batch(symbols)
    out=[]
    for row in rows:
        symbol=row['symbol']; history=histories.get(symbol)
        if history is None or len(history)<30:
            errors.setdefault(symbol, '有效历史数据不足30个交易日')
            continue
        try:
            fib=fibonacci_levels(history)
            pivots=pivot_levels(history)
            tech=technical_analysis(history, fib, pivots)
            if tech.get('composite_score') is None:
                continue
            out.append({
                'symbol':symbol,'company':row.get('company') or symbol,
                'exchange':row.get('exchange') or '','currency':row.get('currency') or '',
                'market':row.get('market') or market_of(symbol),
                'sector':row.get('sector') or '未分类','industry':row.get('industry') or '未分类',
                'market_cap':row.get('market_cap'),
                'fundamental_ok':None,
                'fundamental_status':'扫描阶段未拉取完整财报',
                'strength':tech.get('score'),'value_score':tech.get('value_score'),
                'pullback_score':tech.get('pullback_score'),'composite_score':tech.get('composite_score'),
                'state':tech.get('state') or '暂无数据','value_state':tech.get('value_state') or '暂无数据',
                'pullback_state':tech.get('pullback_state') or '暂无数据',
                'score_breakdown':tech.get('score_breakdown') or {},
                'value_breakdown':tech.get('value_breakdown') or {},
                'pullback_breakdown':tech.get('pullback_breakdown') or {},
                'composite_breakdown':tech.get('composite_breakdown') or {},
            })
        except Exception as exc:
            errors[symbol]=f'技术指标计算失败：{str(exc)[:180]}'
            continue
    return out, errors


def _rank_results(rows, limit=20):
    groups={'us':[],'hk':[],'cn':[]}
    for row in rows:
        groups.setdefault(row.get('market') or market_of(row.get('symbol','')),[]).append(row)
    ranked={}
    for mk in ('us','hk','cn'):
        group=groups.get(mk,[])
        # Stable multi-pass sort keeps the documented priority while making
        # the final ticker tie-break ascending (A -> Z), not descending.
        group.sort(key=lambda x: str(x.get('symbol','')))
        group.sort(key=lambda x: float(x.get('value_score',-1)), reverse=True)
        group.sort(key=lambda x: float(x.get('strength',-1)), reverse=True)
        group.sort(key=lambda x: float(x.get('composite_score',-1)), reverse=True)
        for idx,row in enumerate(group[:limit],1):
            row['rank']=idx
        ranked[mk]=group[:limit]
    return ranked


def _sector_rank(rows, limit=10):
    by={}
    for row in rows:
        sector=row.get('sector') or '未分类'
        by.setdefault(sector,[]).append(row)
    result=[]
    for sector,items in by.items():
        scores=[float(x['composite_score']) for x in items if x.get('composite_score') is not None]
        pullbacks=[float(x['pullback_score']) for x in items if x.get('pullback_score') is not None]
        strengths=[float(x['strength']) for x in items if x.get('strength') is not None]
        if not scores: continue
        # Sector rotation is breadth-aware: average score + breadth of
        # high-quality pullbacks, rather than one superstar stock dominating.
        avg=sum(scores)/len(scores)
        breadth=sum(1 for x in pullbacks if x>=70)/max(1,len(pullbacks))*100
        strength_breadth=sum(1 for x in strengths if x>=70)/max(1,len(strengths))*100
        rotation=avg*0.50 + breadth*0.30 + strength_breadth*0.20
        result.append({'sector':sector,'stocks_scanned':len(items),'average_score':round(avg,1),
                       'pullback_breadth':round(breadth,1),'strength_breadth':round(strength_breadth,1),
                       'rotation_score':round(rotation,1)})
    result.sort(key=lambda x:(x['rotation_score'],x['average_score'],x['sector']),reverse=True)
    for i,x in enumerate(result[:limit],1): x['rank']=i
    return result[:limit]


def _job_snapshot(job):
    with _SCAN_LOCK:
        partial=job.get('rows',[])
        selected=job.get('selected_markets',[])
        return {
            'job_id':job['id'],'status':job['status'],'cancelled':job.get('cancelled',False),
            'started_at':job.get('started_at'),'updated_at':job.get('updated_at'),
            'total':job.get('total',0),'completed':job.get('completed',0),
            'current_market':job.get('current_market'),'current_sector':job.get('current_sector'),
            'current_sector_completed':job.get('current_sector_completed',0),'current_sector_total':job.get('current_sector_total',0),
            'otc_counts':job.get('otc_counts',{}),'cap_rejected':job.get('cap_rejected',{}),
            'scan_config':{'batch_size':SCAN_BATCH_SIZE,'workers':SCAN_WORKERS,'market_cap_min':MARKET_CAP_MIN},
            'universe_source':job.get('universe_source'),
            'results_by_market':_rank_results(partial,20),
            'sector_rotation':{mk:_sector_rank([x for x in partial if x.get('market')==mk],10) for mk in selected},
            'universe_size':job.get('total',0),'selected_markets':selected,
            'error':job.get('error'),'error_count':len(job.get('scan_errors',{})),'recent_errors':list(job.get('scan_errors',{}).items())[-8:]
        }


def _run_scan_job(job):
    try:
        rows,source,otc_counts,cap_rejected=_scan_universe(job['markets'])
        selected=job['selected_markets']
        rows=[r for r in rows if r.get('market') in selected]
        with _SCAN_LOCK:
            job['total']=len(rows); job['universe_source']=source
            job['otc_counts']=otc_counts; job['cap_rejected']=cap_rejected
            job['updated_at']=time()
        # Process market -> sector -> bounded batches. This makes sector
        # progress visible and limits peak data-source pressure.
        for mk in selected:
            if job['cancel_event'].is_set(): break
            market_rows=[r for r in rows if r.get('market')==mk]
            sectors={}
            for r in market_rows: sectors.setdefault(r.get('sector') or '未分类',[]).append(r)
            for sector, sector_rows in sorted(sectors.items(), key=lambda kv:(kv[0] != '未分类',kv[0])):
                if job['cancel_event'].is_set(): break
                with _SCAN_LOCK:
                    job['current_market']=mk; job['current_sector']=sector
                    job['current_sector_completed']=0; job['current_sector_total']=len(sector_rows); job['updated_at']=time()
                batches=[sector_rows[start:start+SCAN_BATCH_SIZE] for start in range(0,len(sector_rows),SCAN_BATCH_SIZE)]
                # Run a bounded number of history batches concurrently. Each
                # batch itself keeps yfinance threads disabled, preventing
                # nested fan-out. Results are merged as soon as a batch ends.
                with ThreadPoolExecutor(max_workers=SCAN_WORKERS, thread_name_prefix='ael-batch') as pool:
                    futures={}
                    for batch in batches:
                        if job['cancel_event'].is_set(): break
                        futures[pool.submit(_scan_batch, batch)]=(len(batch), batch)
                    for future in as_completed(futures):
                        batch_size,batch_rows=futures[future]
                        if job['cancel_event'].is_set():
                            continue
                        try:
                            batch_results,batch_errors=future.result()
                        except Exception as exc:
                            batch_results=[]
                            batch_errors={r['symbol']:f'批次扫描失败：{str(exc)[:180]}' for r in batch_rows}
                        with _SCAN_LOCK:
                            job['rows'].extend(batch_results)
                            job['scan_errors'].update(batch_errors)
                            job['completed'] += batch_size
                            job['current_sector_completed'] += batch_size
                            job['updated_at']=time()
        with _SCAN_LOCK:
            cancelled=job['cancel_event'].is_set()
            if cancelled:
                job['status']='cancelled'; job['cancelled']=True
            else:
                job['status']='completed'
            job['updated_at']=time()
            ranked=_rank_results(job['rows'],20)
            cache_key=job['cache_key']
            # Only a fully completed scan may populate the reusable cache.
            # Cancelled/failed/partial jobs are never presented as complete.
            if not cancelled:
                _SCAN_CACHE[cache_key]={'ts':time(),'data':{
                    'ok':True,'scan':{'markets':job['selected_markets'],'universe_size':job['total'],'matched':sum(len(v) for v in ranked.values()),
                                     'top_n':20,'rules':'全市场目录→板块分批→批量历史行情→技术/回踩评分→各市场独立TOP20；缺失数据不估算','ttl_seconds':SCAN_CACHE_TTL,
                                     'scanner':'background filtered + parallel batched scan','scan_config':{'batch_size':SCAN_BATCH_SIZE,'workers':SCAN_WORKERS,'market_cap_min':MARKET_CAP_MIN},'otc_counts':job.get('otc_counts',{}),'cap_rejected':job.get('cap_rejected',{})},
                    'results_by_market':ranked,'sector_rotation':{mk:_sector_rank([x for x in job['rows'] if x.get('market')==mk],10) for mk in job['selected_markets']},
                    'results':[row for mk in job['selected_markets'] for row in ranked.get(mk,[])], 'cached':False}}
    except Exception as exc:
        with _SCAN_LOCK:
            job['status']='failed'; job['error']=str(exc)[:500]; job['updated_at']=time()


def _cleanup_scan_jobs_locked(max_age_seconds=3600):
    now=time()
    keep_status={'running','cancelling'}
    stale=[jid for jid,j in _SCAN_JOBS.items()
           if j.get('status') not in keep_status and now-float(j.get('updated_at') or now)>max_age_seconds]
    for jid in stale:
        _SCAN_JOBS.pop(jid,None)


@app.post('/api/market-scan/start')
def market_scan_start(markets: str = Query('us,hk,cn'), force: bool = Query(False)):
    selected=[m.strip().lower() for m in markets.split(',') if m.strip() in {'us','hk','cn'}]
    if not selected: raise HTTPException(status_code=400, detail='至少选择一个市场')
    cache_key=','.join(selected)
    now=time()
    with _SCAN_LOCK:
        _cleanup_scan_jobs_locked()
        cached=_SCAN_CACHE.get(cache_key)
    if cached and not force and now-cached['ts']<SCAN_CACHE_TTL:
        return {'ok':True,'cached':True,'done':True,'data':cached['data']}
    with _SCAN_LOCK:
        _cleanup_scan_jobs_locked()
        for job in _SCAN_JOBS.values():
            if job['status']=='running' and job['cache_key']==cache_key:
                return {'ok':True,'job_id':job['id'],'status':'running'}
        job={'id':uuid.uuid4().hex[:12],'markets':cache_key,'selected_markets':selected,'status':'running','cancelled':False,
             'started_at':datetime.now(timezone.utc).isoformat(),'updated_at':time(),'total':0,'completed':0,'rows':[],
             'current_market':None,'current_sector':None,'current_sector_completed':0,'current_sector_total':0,
             'otc_counts':{'us':0,'hk':0,'cn':0},'cap_rejected':{'us':0,'hk':0,'cn':0},
             'scan_errors':{},
             'cancel_event':threading.Event(),'cache_key':cache_key,'error':None}
        _SCAN_JOBS[job['id']]=job
    threading.Thread(target=_run_scan_job,args=(job,),daemon=True,name=f'ael-scan-{job["id"]}').start()
    return {'ok':True,'job_id':job['id'],'status':'running'}


@app.get('/api/market-scan/status/{job_id}')
def market_scan_status(job_id: str):
    with _SCAN_LOCK:
        job=_SCAN_JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail='扫描任务不存在')
    return {'ok':True,**_job_snapshot(job)}


@app.post('/api/market-scan/cancel/{job_id}')
def market_scan_cancel(job_id: str):
    with _SCAN_LOCK:
        job=_SCAN_JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail='扫描任务不存在')
    job['cancel_event'].set()
    return {'ok':True,'job_id':job_id,'status':'cancelling'}


@app.get('/api/market-scan')
def market_scan_legacy(markets: str = Query('us,hk,cn'), limit: int = Query(20, ge=1, le=50), force: bool = Query(False)):
    """Compatibility endpoint. New UI uses start/status so scanning never blocks HTTP."""
    selected=[m.strip().lower() for m in markets.split(',') if m.strip() in {'us','hk','cn'}]
    cache_key=','.join(selected)
    with _SCAN_LOCK:
        cached=_SCAN_CACHE.get(cache_key)
    if cached and not force and time()-cached['ts']<SCAN_CACHE_TTL:
        data=dict(cached['data']); data['results_by_market']={m:(data.get('results_by_market') or {}).get(m,[])[:limit] for m in selected}; return data
    started=market_scan_start(markets=markets,force=force)
    if started.get('done'): return started['data']
    return JSONResponse(status_code=202,content={'ok':True,'status':'running','job_id':started['job_id'],'message':'扫描已在后台启动，请查询 status；不会阻塞单股查询'})


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
