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

## 3. 总结对比

| 特征 | Double Repo | Single Penetration |
| :--- | :--- | :--- |
| **模式类型** | 趋势反转 (Reversal) | 趋势跟踪 (Trend Following) |
| **核心指标** | 3x3 DMA, 25x5 DMA | 3x3 DMA |
| **进场信号** | 两次穿透 3x3 DMA | 强劲推力后首次回调触及 3x3 DMA |
| **风险偏好** | 较高 (摸顶抄底) | 中等 (顺势回调) |
| **主要止盈** | 斐波那契扩展 (OP) | 推力高点 (Thrust High) |

