# DiNapoli 量化交易策略框架 (DiNapoli Algo-Trading Framework)

## 1. 基础数据与指标定义 (Base Indicators)

在编写策略前，必须定义好基础变量。

* **OHLC:** Open, High, Low, Close (当前周期)。
* **3x3 DMA:** 3周期简单移动平均线，向未来平移3个周期。
    * 公式: `DMA[t] = SMA(Close, 3)[t-3]`
    * *注意：代码中需处理平移后的数据对齐，当前K线对比的是3根K线前的SMA数值。*
* **Swing High/Low (波段高低点识别):** 量化的难点在于识别 A-B-C 浪。
    * *建议算法:* 使用 `ZigZag` 指标或 `Fractals` (分形) 逻辑来定义波段高点(High)和波段低点(Low)。
* **Fibonacci Retracement (回撤位):**
    * `Ret_382 = Low + (High - Low) * 0.382`
    * `Ret_618 = Low + (High - Low) * 0.618`
* **Fibonacci Expansion (目标位/扩展位):** 基于 A-B-C 结构 (A=起点, B=转折点, C=回调终点)。
    * `COP = C + (B - A) * 0.618`
    * `OP  = C + (B - A) * 1.0`
    * `XOP = C + (B - A) * 1.618`

---

## 2. 核心策略逻辑模块 (Core Strategy Logic)

我们将四大策略分为**反转类**和**趋势类**，每种策略包含具体的 `Signal` (信号), `Entry` (进场), `Stop` (止损), `Exit` (止盈)。

### 策略 A: Double Repo (顶部/底部反转)

* **适用场景:** 捕捉趋势耗尽时的反转。
* **逻辑条件 (伪代码):**
    1.  **趋势确认:** 价格在过去 $N$ 周期内主要位于 3x3 DMA 一侧。
    2.  **第一次穿透:** $Close_{t-k} < DMA_{t-k}$ (假设做空)，随后 $Close$ 返回上方。
    3.  **第二次穿透:** $High_{t} > High_{previous\_peak}$ (创出新高)，但 $Close_{t} < DMA_{t}$ (收盘跌回)。
    4.  **形态过滤:** 两次穿透之间的时间间隔 $Bars > Min\_Threshold$ (例如3根K线)，避免噪点。
* **进场 (Entry):**
    * 触发信号 K 线收盘后，下一 K 线市价入场；或
    * 在信号 K 线低点下方挂 Stop 单。
* **止损 (SL):** `Max(High_Peak1, High_Peak2) + Buffer`。
* **止盈 (TP):** 基于整个形态起点到终点的回撤：
    * TP1: 0.382 回撤。
    * TP2: 0.618 回撤。

### 策略 B: Bread & Butter (趋势回调)

* **适用场景:** 强趋势中的回调买入。
* **逻辑条件:**
    1.  **趋势强劲:** $Close > DMA$ 持续 $N$ 根 K 线。
    2.  **回调触发:** $Close < DMA$ (收盘价跌破均线)。
    3.  **支撑确认:** 当前价格区间包含波段 A-B 的 $0.382$ 回撤位。
* **进场 (Entry):** 在 $0.382$ 回撤位挂 Limit 单 (限价单)。
* **止损 (SL):** 置于 $0.618$ 回撤位下方。
* **止盈 (TP):** 计算 A-B-C (C为进场点) 的扩展位：
    * TP1: COP ($0.618$ 扩展)。
    * TP2: OP ($1.0$ 扩展)。

### 策略 C: L3 Confluence (高精度汇聚)

* **适用场景:** 狙击极高概率的反弹点（无需等待 K 线确认）。
* **逻辑条件:**
    1.  计算大周期波段 (Major Swing) 的回撤位集合 $R = \{0.382, 0.618\}$。
    2.  计算小周期回调浪 (Minor Swing) 的扩展位集合 $E = \{COP, OP, XOP\}$。
    3.  **汇聚判断:** 如果 $|r - e| < Tolerance$ (两者差值在极小范围内，如价格的0.05%)，则形成 L3 区域。
* **进场 (Entry):** 在汇聚价格区间挂 Limit 单。
* **止损 (SL):** 汇聚区间下方 + 固定 Buffer。
* **止盈 (TP):** 基于进场后新波段的 OP ($1.0$ 扩展)。

### 策略 D: Minesweeper A (强趋势跟随)

* **适用场景:** 价格贴着 DMA 走，拒绝回调。
* **逻辑条件:**
    1.  **均线乖离:** 价格一直位于 DMA 上方，且从未触及 0.382 回撤位。
    2.  **辅助指标:** Stochastic (8,3,3) 从超卖区金叉 (做多) 或超买区死叉 (做空)。
* **进场 (Entry):** Stochastic 信号出现时直接进场，或突破前一根 K 线高点进场。
* **止损 (SL):** 追踪止损 (Trailing Stop)，设在每日 3x3 DMA 数值下方。
* **止盈 (TP):** 不设固定目标，直到止损被触发。

---

## 3. 信号增强过滤器 (Trigger Filters)

在量化代码中，可以将 RRT 和 FTP 写成**布尔函数 (Boolean Functions)**，用于增强上述策略的进场确信度。

* **Check_RRT (铁轨形态):**
    * `Body1_Size > Threshold` AND `Body2_Size > Threshold`
    * `Direction1 != Direction2`
    * `Open2 ≈ Close1` AND `Close2 ≈ Open1` (吞没逻辑)
    * *用法:* 如果策略 A 或 B 的进场位出现了 `Check_RRT == True`，加仓或提高信号权重。

* **Check_FTP (穿透失败):**
    * `High > DMA` (做空时刺穿)
    * `Close < DMA` (收盘收回)
    * `Shadow_Upper > Body_Size` (长上影线)
    * *用法:* 用于策略 B 持仓过程中的加仓信号，或防止被假突破洗出。

---

## 4. 建议的量化开发路线图 (Implementation Roadmap)

为了将这些策略程序化，建议你按以下步骤进行：

### 第一步：波段识别算法 (Swing Detection)
这是最难的一步。你需要写一个健壮的函数来识别 A、B、C 点。
* *简单的:* 使用 `iHigh` 和 `iLow` 的极值（例如过去20根K线的最高点）。
* *复杂的:* 实现 ZigZag 算法，过滤掉微小的波动，锁定关键的高低点。**如果没有准确的 A-B 点，斐波那契计算就是错的。**

### 第二步：回测 Double Repo (Backtesting)
先写 Double Repo。因为它形态最独特，最容易用代码定义。
* 测试不同时间周期（H4, D1 效果通常最好）。
* 优化参数：两次穿透之间的最小/最大 K 线数量。

### 第三步：添加多周期过滤 (Multi-Timeframe)
写一个主函数：
* `Trend_Direction = Check_Weekly_Chart()` (判断周线是在 DMA 上方还是下方)。
* 如果周线看涨，日线策略只开多单 (Long Only)；反之只开空单。
* **DiNapoli 强调：这是提高胜率最核心的过滤条件。**

### 第四步：风险管理模块 (Risk Management)
* 统一止损逻辑：所有策略必须有硬止损。
* 资金管理：建议单笔交易风险不超过账户资金的 2%。

### 下一步你能做的事
你需要我为你提供一段 **Python (使用 Pandas/TA-Lib)** 的伪代码，来演示**如何计算 3x3 DMA 以及如何识别 Double Repo 的基本逻辑**吗？这将是你量化脚本的第一块基石。