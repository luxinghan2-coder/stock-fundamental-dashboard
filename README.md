# 股票基本面驾驶舱 V2.3.2

A股 / 港股 / 美股基本面分析 H5/PWA。

## V2.3.2 修复
- 美股长期 ROE 使用 SEC EDGAR / XBRL Company Facts。
- SEC ROE 按 fiscal-period end date 配对：年度净利润 + 对应财年期初权益 + 对应财年期末权益，避免 Apple 等非自然年财年的年份错位。
- ROE 明细显示 FY 财年标签与财年截止日期。
- 美股当前 ROE 与 SEC 15 年 ROE 定义保持一致。
- 股息历史区分完整年度与当前年度 YTD。
- 3Y / 5Y / 10Y 股息 CAGR 只使用完整年度，不把未结束年度混入 CAGR。
- TTM 股息率继续按最近 365 天现金股息计算。
- 保留 V2.3.1 已有的基本面、分红、技术指标、Fibonacci、价格曲线、分析师图表与 K/M/B/T 金额格式。
- 数据缺失时显示“暂无数据”，不人为填充。

## 数据源
- Yahoo Finance / yfinance：行情、财报、分红、技术指标、分析师数据。
- SEC EDGAR / XBRL Company Facts：美股长期 ROE。

## Railway
启动命令：
`uvicorn app:app --host 0.0.0.0 --port $PORT`
