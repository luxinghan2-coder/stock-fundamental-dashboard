# 股票基本面驾驶舱 V1

独立于现有期权收入项目的股票基本面分析 H5/PWA 原型。

V1：
- A股 / 港股 / 美股代码输入与基础市场识别
- 当前价格、PE、PB、ROE、ROE/PB、PE/ROE
- 营收、净利润、毛利率、自由现金流、负债率、股息率
- 最近15个完整财年的ROE
- 15年ROE平均值、中位数、标准差、极差、最高/最低
- 自动绘制15年ROE折线图
- 数据缺失显示“暂无数据”，不人为填充

Railway启动：
uvicorn app:app --host 0.0.0.0 --port $PORT

V1默认使用Yahoo Finance via yfinance作为原型数据源。
后续版本接入Alpha Vantage、SEC及A股/港股专用公开数据源做路由与交叉校验。
