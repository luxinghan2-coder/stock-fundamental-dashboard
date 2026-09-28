"""Optional multi-asset Lite data layer.

The asset layer is isolated from stock core/scan. It reuses the same technical
engine as stocks so commodities/crypto receive the same technical vocabulary,
while news is a separate lazy endpoint and can never block the quote response.
"""
from time import time
import math
import threading
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

from metrics import technical_analysis, fibonacci_levels, pivot_levels, resonance_levels, technical_price_chart

ASSET_INDEX = [
    {"symbol":"XAUUSD","provider_symbol":"XAUUSD=X","name":"Gold Spot / USD","cn":"黄金","asset_type":"商品","market":"Spot","currency":"USD","icon":"🥇"},
    {"symbol":"XAGUSD","provider_symbol":"SI=F","name":"Silver Spot / USD","cn":"白银","asset_type":"商品","market":"Spot","currency":"USD","icon":"🥈"},
    {"symbol":"WTI","provider_symbol":"CL=F","name":"WTI Crude Oil","cn":"WTI原油","asset_type":"商品","market":"NYMEX","currency":"USD","icon":"🛢️"},
    {"symbol":"BRENT","provider_symbol":"BZ=F","name":"Brent Crude Oil","cn":"布伦特原油","asset_type":"商品","market":"ICE","currency":"USD","icon":"🛢️"},
    {"symbol":"COPPER","provider_symbol":"HG=F","name":"Copper Futures","cn":"铜","asset_type":"商品","market":"COMEX","currency":"USD","icon":"🔶"},
    {"symbol":"PLATINUM","provider_symbol":"PL=F","name":"Platinum Futures","cn":"铂金","asset_type":"商品","market":"NYMEX","currency":"USD","icon":"⚪"},
    {"symbol":"BTCUSD","provider_symbol":"BTC-USD","name":"Bitcoin / USD","cn":"比特币","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"₿"},
    {"symbol":"ETHUSD","provider_symbol":"ETH-USD","name":"Ethereum / USD","cn":"以太坊","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"Ξ"},
    {"symbol":"SOLUSD","provider_symbol":"SOL-USD","name":"Solana / USD","cn":"Solana","asset_type":"加密资产","market":"Crypto","currency":"USD","icon":"S"},
    {"symbol":"DXY","provider_symbol":"DX-Y.NYB","name":"US Dollar Index","cn":"美元指数","asset_type":"外汇/宏观","market":"ICE","currency":"USD","icon":"$"},
]

_CACHE = {}
_NEWS_CACHE = {}
_LOCK = threading.Lock()
TTL = 300
NEWS_TTL = 900


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
    exact=next((x for x in ASSET_INDEX if x['symbol'].lower()==q or x['provider_symbol'].lower()==q or x['cn'].lower()==q or x['name'].lower()==q),None)
    if exact:return dict(exact)
    aliases={
      'gold':'XAUUSD','黄金期货':'XAUUSD','黄金现货':'XAUUSD','xau':'XAUUSD','xauusd':'XAUUSD','xauusd=x':'XAUUSD','gc=f':'XAUUSD',
      'btc':'BTCUSD','bitcoin':'BTCUSD','比特币':'BTCUSD','btcusd':'BTCUSD','btc-usd':'BTCUSD',
      'eth':'ETHUSD','ethereum':'ETHUSD','ethusd':'ETHUSD','eth-usd':'ETHUSD',
      'sol':'SOLUSD','solana':'SOLUSD','solusd':'SOLUSD','sol-usd':'SOLUSD',
      '原油期货':'WTI','wti':'WTI','oil':'WTI','crude oil':'WTI','石油':'WTI','原油':'WTI','cl=f':'WTI',
      'brent':'BRENT','布油':'BRENT','bz=f':'BRENT','silver':'XAGUSD','白银':'XAGUSD','si=f':'XAGUSD',
      'copper':'COPPER','铜':'COPPER','hg=f':'COPPER','platinum':'PLATINUM','铂金':'PLATINUM','pl=f':'PLATINUM',
      'dxy':'DXY','dx-y.nyb':'DXY','美元指数':'DXY',
    }
    s=aliases.get(q)
    return next((dict(x) for x in ASSET_INDEX if x['symbol']==s),None) if s else None

def get_asset(symbol):
    symbol=str(symbol or '').strip().upper()
    meta=next((dict(x) for x in ASSET_INDEX if x['symbol'].upper()==symbol),None)
    if not meta:return {'ok':False,'symbol':symbol,'error':'AEL当前多资产索引未收录该标的'}
    now=time()
    with _LOCK:
        c=_CACHE.get(symbol)
        if c and now-c[0]<TTL:return c[1]
    provider_symbol=meta.get('provider_symbol') or symbol
    h=None; errors=[]
    try:
        h=yf.Ticker(provider_symbol).history(period='2y',interval='1d',auto_adjust=False)
    except Exception as exc:
        errors.append('yfinance: '+str(exc)[:100])
    if h is None or h.empty:
        try:
            h=_yahoo_chart_history(provider_symbol,730)
        except Exception as exc:
            errors.append('Yahoo Chart: '+str(exc)[:100])
            h=None
    if h is None or h.empty:
        out={'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':'暂无足够行情数据'}
    else:
        try:
            out=_calc(symbol,meta,h)
            if not out: out={'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':'暂无足够历史数据'}
            else:
                out['ok']=True;out['provider_symbol']=provider_symbol
                out['method_note']='多资产复用股票技术分析引擎；行情优先使用 Yahoo Finance，yfinance 不可用时自动回退 Yahoo Chart；MA20/60/120/250、MACD、RSI、KDJ、BOLL、Pivot、Fibonacci、20D动量、52周位置、技术强势/高性价比/回踩质量全部保持；股票基本面字段对商品/加密资产标记为不适用。'
        except Exception as exc:
            out={'ok':False,'symbol':symbol,'asset_type':meta['asset_type'],'error':str(exc)[:180]}
    with _LOCK:_CACHE[symbol]=(now,out)
    return out


def _asset_news_query(meta):
    # Search by human-readable asset name, not provider ticker, for better news coverage.
    if meta['symbol']=='XAUUSD': return 'gold spot gold price'
    if meta['symbol']=='WTI': return 'WTI crude oil'
    if meta['symbol']=='BTCUSD': return 'Bitcoin BTC'
    if meta['symbol']=='ETHUSD': return 'Ethereum ETH'
    if meta['symbol']=='SOLUSD': return 'Solana SOL'
    return meta.get('name') or meta.get('cn') or meta['symbol']


def get_asset_news(symbol, limit=8):
    symbol=str(symbol or '').strip().upper()
    meta=next((dict(x) for x in ASSET_INDEX if x['symbol'].upper()==symbol),None)
    if not meta:
        return {'ok':False,'symbol':symbol,'news':[],'error':'未收录多资产'}
    limit=max(1,min(12,int(limit or 8)))
    key=f'{symbol}:{limit}'
    now=time()
    with _LOCK:
        c=_NEWS_CACHE.get(key)
        if c and now-c[0]<NEWS_TTL:return c[1]
    out={'ok':True,'symbol':symbol,'news':[],'source':'Yahoo Finance News','error':None}
    try:
        r=requests.get('https://query1.finance.yahoo.com/v1/finance/search',
                       params={'q':_asset_news_query(meta),'quotesCount':0,'newsCount':limit},
                       headers={'User-Agent':'Mozilla/5.0 (AEL; multi-asset news)'},timeout=4)
        r.raise_for_status(); data=r.json() or {}
        seen=set()
        for x in data.get('news') or []:
            title=(x.get('title') or '').strip(); link=(x.get('link') or x.get('canonicalUrl',{}).get('url') or '').strip()
            if not title or not link: continue
            k=link or title
            if k in seen: continue
            seen.add(k)
            ts=x.get('providerPublishTime'); dt=None
            if ts:
                try: dt=pd.to_datetime(ts,unit='s',utc=True).tz_convert(None).strftime('%Y-%m-%d %H:%M')
                except Exception: pass
            out['news'].append({'title':title,'publisher':str(x.get('publisher') or ''),'link':link,'published_at':dt})
    except Exception as exc:
        out['error']=str(exc)[:180]
    out['news']=out['news'][:limit]
    with _LOCK:_NEWS_CACHE[key]=(now,out)
    return out
