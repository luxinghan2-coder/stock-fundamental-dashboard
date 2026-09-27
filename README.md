# 股票基本面驾驶舱 V2.2

A股 / 港股 / 美股基本面 H5/PWA 原型。

## V2.2
- 行情采用 yfinance history + Yahoo chart endpoint fallback
- 股息独立取数；区分“无现金股息”和“数据不可用”
- 股息率、3/5/10Y CAGR、连续分红、净利润支付率、FCF支付率、回购、股东总支付率
- 技术分析本地计算：MA20/60/120/250、RSI14、MACD、布林带位置、20日动量、52周位置
- 分析师评级、目标价、EPS/营收预期、评级变化
- 美股长期 ROE 优先使用 SEC EDGAR XBRL Company Facts；SEC API 无需 API key
- A股 / 港股继续使用免费市场数据源；缺失不人为填充
- 单个数据模块失败不会拖垮整个页面

## Railway
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## SEC User-Agent
生产环境建议在 Railway Variables 中设置：
`SEC_USER_AGENT=StockFundamentalDashboard/2.2 your-email@example.com`

SEC data.sec.gov 的程序化访问需要遵守 SEC 的访问政策并提供可识别的 User-Agent。
