from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import yfinance as yf


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _series(hist):
    if hist is None or getattr(hist, 'empty', True) or 'Close' not in hist:
        return None
    s = hist['Close'].dropna()
    return s if not s.empty else None


def _returns(s):
    if s is None:
        return None
    r = s.pct_change().dropna()
    return r if not r.empty else None


def _vol(r, n):
    if r is None or len(r) < max(20, min(n, 60)):
        return None
    return float(r.tail(n).std() * math.sqrt(252) * 100)


def _max_dd(s, n=252):
    if s is None or len(s) < 2:
        return None
    x = s.tail(n)
    return float((x / x.cummax() - 1).min() * 100)


def _beta_corr(r, b):
    if r is None or b is None:
        return None, None
    df = __import__('pandas').concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    if len(df) < 30:
        return None, None
    cov = df['r'].cov(df['b'])
    var = df['b'].var()
    beta = cov / var if var and math.isfinite(var) else None
    corr = df['r'].corr(df['b'])
    return _num(beta), _num(corr)


def _var_cvar(r, alpha=.95):
    if r is None or len(r) < 30:
        return None, None
    q = float(r.quantile(1-alpha))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else None
    return q * 100, cvar * 100 if cvar is not None else None


def _sharpe(r, rf_annual=0.0):
    if r is None or len(r) < 30:
        return None
    daily_rf = rf_annual / 252.0
    ex = r - daily_rf
    sd = ex.std()
    if not sd or not math.isfinite(sd):
        return None
    return float(ex.mean() / sd * math.sqrt(252))


def analyze_risk(symbol: str, benchmark: str = 'SPY', period: str = '2y') -> dict[str, Any]:
    symbol = symbol.strip().upper()
    benchmark = benchmark.strip().upper() or 'SPY'
    if not symbol:
        raise ValueError('请输入股票代码')
    errors=[]
    try:
        hs = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    except Exception as exc:
        hs=None; errors.append(f'标的历史行情：{str(exc)[:180]}')
    try:
        hb = yf.Ticker(benchmark).history(period=period, auto_adjust=False)
    except Exception as exc:
        hb=None; errors.append(f'基准历史行情：{str(exc)[:180]}')
    s=_series(hs); b=_series(hb)
    r=_returns(s); br=_returns(b)
    beta,corr=_beta_corr(r,br)
    var95,cvar95=_var_cvar(r)
    downside=None
    if r is not None and len(r)>=30:
        neg=r[r<0]
        if len(neg)>=10:
            downside=float(neg.std()*math.sqrt(252)*100)
    result={
      'symbol':symbol,'benchmark':benchmark,'as_of':datetime.now(timezone.utc).isoformat(),
      'metrics':{
        'beta_2y':beta,'correlation_2y':corr,
        'volatility_20d_pct':_vol(r,20),'volatility_60d_pct':_vol(r,60),'volatility_252d_pct':_vol(r,252),
        'max_drawdown_252d_pct':_max_dd(s,252),'max_drawdown_2y_pct':_max_dd(s,504),
        'historical_var_95_1d_pct':var95,'historical_cvar_95_1d_pct':cvar95,
        'downside_volatility_annualized_pct':downside,
        'sharpe_0rf_2y':_sharpe(r,0.0),
      },
      'data_quality':{
        'symbol_history_rows':int(len(hs)) if hs is not None else 0,
        'benchmark_history_rows':int(len(hb)) if hb is not None else 0,
        'errors':errors,
      },
      'method':'historical_risk_v1',
      'method_note':'历史风险统计；VaR/CVaR为历史分位法，不是预测；Sharpe 暂按 0% 年化无风险利率计算。缺失数据不估算。'
    }
    result['missing_metrics']=[k for k,v in result['metrics'].items() if v is None]
    return result
