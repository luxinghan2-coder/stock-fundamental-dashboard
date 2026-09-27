from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from metrics import build_dashboard

BASE = Path(__file__).resolve().parent
app = FastAPI(title='AEL 股票基本面驾驶舱 V2.4.3', version='2.4.3')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/api/health')
def health():
    return {'ok': True, 'service': 'stock-fundamental-dashboard', 'version': '2.4.3'}

@app.get('/api/stock/{symbol}')
def stock(symbol: str):
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail='请输入股票代码')
    try:
        return build_dashboard(symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'数据获取失败：{exc}')
