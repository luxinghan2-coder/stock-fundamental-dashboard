"""Optional multi-asset Lite data layer.

Separate from stock fundamental/scan code so commodity/crypto support can never
change the existing stock core response path.
"""
from time import time
import math
import threading
import yfinance as yf
import pandas as pd

ASSET_INDEX = [
    {"symbol":"GC=F","name":"Gold Futures","cn":"黄金","asset_type":"商品","market":"COMEX","currency":"USD","icon":"🥇"},
    {"symbol":"SI=F","name":"Silver Futures","cn":"白银","asset_type":"商品","market":"COMEX","currency":"USD","icon":"🥈"},
    {"symbol":"CL=F","name":"Crude Oil Futures","cn":"原油","asset_type":"商品","market":"NYMEX","currency":"USD","icon":"🛢️"},
    {"symbol":"BZ=F","name":"Brent Crude Futures","cn":"布伦特原油","asset_type":"商品","market":"ICE","currency":"USD","icon":"🛢️"},
    {"symbol":"HG=F","name":"Copper Futures","cn":"铜","asset_type":"商品","market":"COMEX","currency":"USD","icon":"🔶"},
    {"symbol":"PL=F","name":"Platinum Futures","cn":"铂金","asset_type":"商品","market":"NYMEX","currency":"USD","icon":"⚪"},
    {"symbol":"BTC-USD","name":"Bitcoin USD","cn":"比特币","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"₿"},
    {"symbol":"ETH-USD","name":"Ethereum USD","cn":"以太坊","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"Ξ"},
    {"symbol":"SOL-USD","name":"Solana USD","cn":"Solana","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"S"},
    {"symbol":"DX-Y.NYB","name":"US Dollar Index","cn":"美元指数","asset_type":"外汇/宏观","market":"ICE","currency":"USD","icon":"$"},
]

_CACHE = {}
_LOCK = threading.Lock()
TTL = 300

def _finite(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def get_asset_index():
    return [dict(x) for x in ASSET_INDEX]

def find_asset(query):
    q=str(query or '').strip().lower()
    if not q: return None
    exact=next((x for x in ASSET_INDEX if x['symbol'].lower()==q or x['cn'].lower()==q or x['name'].lower()==q),None)
    if exact:return dict(exact)
    aliases={
      'gold':'GC=F','黄金期货':'GC=F','黄金现货':'GC=F','btc':'BTC-USD','bitcoin':'BTC-USD','比特币':'BTC-USD',
      'eth':'ETH-USD','ethereum':'ETH-USD','原油期货':'CL=F','wti':'CL=F','oil':'CL=F','crude oil':'CL=F',
      'brent':'BZ=F','布油':'BZ=F','silver':'SI=F','白银期货':'SI=F','copper':'HG=F','platinum':'PL=F',
    }
    s=aliases.get(q)
    return next((dict(x) for x in ASSET_INDEX if x['symbol']==s),None) if s else None

def _calc(symbol, meta, h):
    close=pd.to_numeric(h['Close'],errors='coerce').dropna()
    if len(close)<30:return None
    px=float(close.iloc[-1]); prev=float(close.iloc[-2]) if len(close)>=2 else None
    ret1=(px/prev-1)*100 if prev else None
    ret20=(px/float(close.iloc[-21])-1)*100 if len(close)>=21 else None
    ret60=(px/float(close.iloc[-61])-1)*100 if len(close)>=61 else None
    lo=float(close.tail(min(252,len(close))).min()); hi=float(close.tail(min(252,len(close))).max())
    pos=((px-lo)/(hi-lo)*100) if hi>lo else None
    daily=close.pct_change().dropna()
    vol20=float(daily.tail(min(20,len(daily))).std()*math.sqrt(252)*100) if len(daily)>=10 else None
    dd=(px/float(close.cummax().iloc[-1])-1)*100
    return {
      **meta,'available':True,'price':px,'change_1d_pct':ret1,'momentum_20d_pct':ret20,
      'momentum_60d_pct':ret60,'position_52w_pct':pos,'volatility_20d_annualized_pct':vol20,
      'max_drawdown_from_history_pct':dd,'history_rows':int(len(close)),
      'latest_trade_date':str(close.index[-1].date()) if hasattr(close.index[-1],'date') else str(close.index[-1]),
      'data_quality':{'history':True,'fundamentals':'not_applicable','reason':'该资产类别不套用股票ROE/PE/现金流硬门槛'},
    }

def get_asset(symbol):
    symbol=str(symbol or '').strip().upper()
    meta=next((dict(x) for x in ASSET_INDEX if x['symbol'].upper()==symbol),None)
    if not meta:return {'ok':False,'symbol':symbol,'error':'AEL当前多资产索引未收录该标的'}
    now=time()
    with _LOCK:
        c=_CACHE.get(symbol)
        if c and now-c[0]<TTL:return c[1]
    try:
        h=yf.Ticker(symbol).history(period='1y',interval='1d',auto_adjust=False)
        if h is None or h.empty:return {'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':'暂无足够行情数据'}
        out=_calc(symbol,meta,h)
        if not out:return {'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':'暂无足够历史数据'}
        out['ok']=True;out['method_note']='多资产只使用真实行情统计；股票基本面字段对商品/加密资产标记为不适用，不用0或估算值冒充。'
    except Exception as exc:
        out={'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':str(exc)[:180]}
    with _LOCK:_CACHE[symbol]=(now,out)
    return out
