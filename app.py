from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from metrics import build_dashboard

BASE = Path(__file__).resolve().parent
app = FastAPI(title='AEL 股票基本面驾驶舱 V2.4.14', version='2.4.14')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/api/health')
def health():
    return {'ok': True, 'service': 'stock-fundamental-dashboard', 'version': '2.4.14'}


@app.get('/api/search')
def stock_search(q: str = ""):
    """轻量候选搜索：优先本地索引；远程补充仅用于候选，不触发行情/财报查询。"""
    from metrics import search_symbols
    return {"query": q, "items": search_symbols(q, limit=5)}

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
