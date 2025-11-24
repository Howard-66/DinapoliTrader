# Features Engineering for DiNapoli Trading Strategy

## 原始市场数据 (Raw Market Data)
这些是金融时间序列的基础数据，直接反映了价格行为和交易活动。

- 开盘价 (Open Price) 、最高价 (High Price) 、最低价 (Low Price) 、收盘价 (Close Price) 
- 成交量 (Volume) 
- 调整后的收盘价 (Adjusted Close Price) 
- 订单簿深度 (Order Book Depth) 

## 传统技术指标 (Traditional Technical Indicators)
这些指标通过数学计算从原始价格和成交量数据中派生，用于识别趋势、动能、波动性和超买/超卖状况。
- 移动平均线 (Moving Averages - MA) 
- 简单移动平均线 (SMA) 
- 指数移动平均线 (EMA) 
- 位移移动平均线 (DMA)  (帝纳波利专有，但可作为特征)
- 成交量加权移动平均线 (VWMA) 
- 相对强弱指数 (Relative Strength Index - RSI) 
- 移动平均收敛/发散 (MACD) 
- 随机指标 (Stochastic Oscillator) 
- 布林带 (Bollinger Bands) 
- 变化率 (Rate of Change - ROC) 
- 商品通道指数 (Commodity Channel Index - CCI) 
- 平均真实波动范围 (Average True Range - ATR) 
- 强力指数 (Force Index) 
- 能量潮 (On-Balance Volume - OBV) 
- 平均趋向指数 (Average Directional Index - ADX) 
- 价格与移动平均线的偏差 (Price - MA Difference)  (例如，收盘价与SMA的偏差)
## 帝纳波利专有指标及派生特征 (DiNapoli Proprietary & Derived Features)
这些是帝纳波利方法独有的或高度相关的特征，具有先行或汇聚的特性。
### 帝纳波利斐波那契水平 (DiNapoli Fibonacci Levels) 
- 斐波那契回撤水平 (如 23.6%, 38.2%, 50%, 61.8%, 78.6%) 
- 斐波那契扩展水平 (如 100%, 161.8%, 261.8%) 
- 价格与斐波那契水平的距离 (当前价格与最近斐波那契水平的距离)
### 汇聚强度 (Confluence Strength / Fibonacci Clusters) 
- 量化不同时间框架或价格波动中斐波那契水平的重叠数量 。
- 可以考虑加权：重要斐波那契水平（如61.8%）或高时间框架的水平可以有更高权重
### 先行指标
- 震荡指标预测器 (Oscillator Predictor - DNOscP)  (旨在提前指示超买/超卖状况)
M- ACD预测器 (MACD Predictor - DMACDP)  (据称能追踪算法行为，预测行动点位)
- 位移移动平均线 (Displaced Moving Averages - DMA)  (高效的趋势识别能力)
- 长记忆 (Long Term Memory)  (KNN模型中的一个概念，反映长期S/R)
## 时间特征 (Time-Based Features)
捕捉市场行为中的周期性和时间依赖性。
- 星期几 (Day of the Week) 
- 月份 (Month of the Year) 
- 季节性 (Seasonality) 
- 滞后值 (Lagged Values)  (例如，前一天的收盘价、前一周的RSI等)
## 市场情绪与另类数据 (Market Sentiment & Alternative Data)
提供超越纯技术分析的市场背景信息。
- 新闻文章情感 (Sentiment from News Articles)  (通过NLP工具如FinBERT或ChatGPT-4o进行情感分析) 
- 社交媒体动态情感 (Sentiment from Social Media / Tweets) 
- 经济指标 (Economic Indicators) 
- 链上活动 (On-chain Activity) (针对加密货币市场) 
## 市场结构与模式 (Market Structure & Patterns)
捕捉价格行为的特定形态或结构。
- K线形态 (Candlestick Patterns) (例如，吞没形态) 
- 趋势方向 (Trend Direction)  (例如，通过移动平均线交叉或价格高低点序列判断)
- 市场状态 (Market State) (例如，趋势、震荡、高波动、低波动) 
- 价格波动性 (Volatility)  (例如，通过ATR或标准差衡量)
## 降维特征 (Dimensionality Reduction Features)
当原始特征数量过多或存在高度相关性时，可以使用降维技术来创建新的、更简洁的特征。
- 主成分分析 (Principal Component Analysis - PCA) 后的特征  (用于降低高度相关技术指标的维度，同时保留最大信息)