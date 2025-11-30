# 交易策略详情文档

本文档详细描述了系统支持的两种核心交易模式：**Double Repo (双重穿透)** 和 **Single Penetration (单次穿透)**。

## 1. 模式一：Double Repo (双重穿透)

Double Repo 是一种“失败的突破”模式，通常发生在趋势末端，预示着趋势的反转。

### 1.1 进场规则 (Entry Rules)

以**看涨 (Buy)** 为例（看跌逻辑相反）：

1.  **初始状态**：价格位于 **3x3 DMA** (Displaced Moving Average) 之下，且趋势向下。
2.  **第一次穿透**：收盘价向上突破 3x3 DMA。
3.  **回落**：收盘价重新跌回 3x3 DMA 之下。
4.  **第二次穿透 (信号触发)**：收盘价再次向上突破 3x3 DMA。
5.  **过滤条件**：
    -   **25x5 DMA 过滤**：价格必须“远离” 25x5 DMA（即处于超卖状态），这增加了反转的可能性。
    -   **时间限制**：两次穿透之间的时间间隔不宜过长。

### 1.2 止损与止盈 (Exit Rules)
系统支持三种止损模式和两种止盈模式，可根据市场环境灵活选择：

*   **止损模式 (Stop Loss Modes)**：
    1.  **Pattern Based (形态止损)**：
        *   使用形态结构的最低点。
        *   对于 Double Repo：使用第一次穿透后形成的最低点 (Pattern Low)。
    2.  **ATR Based (波动率止损)**：
        *   基于 ATR (平均真实波幅) 动态调整。
        *   公式：`Entry Price - (ATR * Multiplier)`。
        *   适用于高波动市场，给予价格呼吸空间。
    3.  **Fixed Percentage (固定百分比)**：
        *   使用固定的百分比回撤。
        *   公式：`Entry Price * (1 - Stop Loss %)`。

*   **止盈模式 (Take Profit Modes)**：
    1.  **Pattern Based (Fibonacci)**：
        *   使用斐波那契扩展目标位。
        *   对于 Double Repo：默认目标为 OP (1.0 扩展) 或 COP (0.618 扩展)。
    2.  **Fixed Percentage (固定百分比)**：
        *   使用固定的百分比目标。
        *   公式：`Entry Price * (1 + Take Profit %)`。

*   **时间止损 (Time Exit)**：
    *   **Holding Period**：持仓超过指定 K 线数量后强制平仓，作为最后的风控手段。

### 1.3 头寸控制 (Position Sizing)

*   **动态仓位 (Dynamic Sizing)**：
    *   基于账户权益的固定百分比风险 (Risk % per Trade)。
    *   公式：$$ \text{Size} = \frac{\text{Equity} \times \text{Risk \%}}{|\text{Entry} - \text{SL}|} $$
    *   当止损幅度较小时，仓位自动放大；止损幅度大时，仓位自动缩小，确保每笔交易的金额风险恒定。

---

## 2. 模式二：Single Penetration (单次穿透 / Bread & Butter)

Single Penetration 是一种趋势跟踪模式，旨在捕捉强劲趋势中的回调机会。DiNapoli 称之为 "Bread & Butter" 交易。

### 2.1 进场规则 (Entry Rules)

以**看涨 (Buy)** 为例：

1.  **推力 (Thrust)**：
    *   收盘价必须连续 **N 根** (默认 8 根) K 线保持在 **3x3 DMA** 之上。
    *   这确认了强劲的上升趋势。
2.  **穿透 (Penetration)**：
    *   价格回调并触及 3x3 DMA。
    *   **触发点**：当 K 线的最低价 (Low) 小于或等于 3x3 DMA 时触发。
3.  **进场方式**：
    *   在价格触及 3x3 DMA 时生成买入信号。

### 2.2 止损与止盈 (Exit Rules)

*   **止损模式**：
    *   **Pattern Based**：使用推力启动时的低点 (Thrust Start Low) 或最近的摆动低点。
    *   同样支持 **ATR Based** 和 **Fixed Percentage**。

*   **止盈模式**：
    *   **Pattern Based**：通常以推力的高点 (Thrust High) 为初步目标。
    *   同样支持 **Fixed Percentage**。

### 2.3 头寸控制 (Position Sizing)

*   与 Double Repo 共享相同的动态仓位逻辑。

---

## 3. 模式三：Railroad Tracks (RRT / 铁轨形态)

Railroad Tracks 是一种强烈的反转模式，由两根方向相反、实体较长且长度相近的 K 线组成，形似铁轨。

### 3.1 进场规则 (Entry Rules)

1.  **形态构成**：
    *   **K 线 1**：实体较长（相对于近期平均水平）。
    *   **K 线 2**：实体较长，方向与 K 线 1 相反，且实体长度与 K 线 1 相近（通常在 70% - 130% 之间）。
2.  **信号触发**：
    *   **看跌 RRT (Top Reversal)**：K 线 1 为阳线 (Bullish)，K 线 2 为阴线 (Bearish)。
    *   **看涨 RRT (Bottom Reversal)**：K 线 1 为阴线 (Bearish)，K 线 2 为阳线 (Bullish)。
3.  **进场点**：K 线 2 收盘时。

### 3.2 止损与止盈 (Exit Rules)

*   **止损 (Stop Loss)**：
    *   使用形态的极值点。
    *   **看跌**：两根 K 线的最高价 (Max High)。
    *   **看涨**：两根 K 线的最低价 (Min Low)。

*   **止盈 (Take Profit)**：
    *   通常以形态高度 (Pattern Height) 为基准。
    *   **目标位**：进场价格 +/- (1.0 * 形态高度)。

---

## 4. 模式四：Failure to Penetrate (FTP / 穿透失败)

Failure to Penetrate 是一种利用支撑/阻力有效性的模式。当价格试图突破 3x3 DMA 但失败并收回时触发。

### 4.1 进场规则 (Entry Rules)

1.  **穿透尝试**：
    *   价格在盘中突破了 3x3 DMA（High > DMA 或 Low < DMA）。
2.  **收盘回归**：
    *   收盘价未能保持在突破方向，而是收回到了 3x3 DMA 的另一侧。
    *   **看涨 FTP**：最低价 < 3x3 DMA，但收盘价 > 3x3 DMA（且前一根 K 线也在 DMA 之上）。
    *   **看跌 FTP**：最高价 > 3x3 DMA，但收盘价 < 3x3 DMA（且前一根 K 线也在 DMA 之下）。

### 4.2 止损与止盈 (Exit Rules)

*   **止损 (Stop Loss)**：
    *   使用穿透 K 线的极值点（Wick High/Low）。
    *   **看涨**：该 K 线的最低价。
    *   **看跌**：该 K 线的最高价。

*   **止盈 (Take Profit)**：
    *   基于风险回报比 (Risk:Reward)。
    *   默认目标为 **2.0 倍风险** (2:1 R:R)。

---

## 5. 总结对比

| 特征 | Double Repo | Single Penetration | Railroad Tracks (RRT) | Failure to Penetrate (FTP) |
| :--- | :--- | :--- | :--- | :--- |
| **模式类型** | 趋势反转 (Reversal) | 趋势跟踪 (Trend Following) | 强力反转 (Reversal) | 支撑/阻力确认 (S/R Hold) |
| **核心指标** | 3x3 DMA, 25x5 DMA | 3x3 DMA | K 线形态 (Price Action) | 3x3 DMA |
| **进场信号** | 两次穿透 3x3 DMA | 强劲推力后首次回调触及 3x3 DMA | 双线反向并列 | 穿透后收回 |
| **风险偏好** | 较高 (摸顶抄底) | 中等 (顺势回调) | 高 (急剧反转) | 中等 (确认信号) |
| **主要止盈** | 斐波那契扩展 (OP) | 推力高点 (Thrust High) | 形态高度 (1.0x) | 风险回报比 (2.0x) |
