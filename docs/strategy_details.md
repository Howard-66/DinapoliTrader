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

*   **止损 (Stop Loss)**：
    *   **位置**：设置在整个形态的最低点下方（即第一次穿透后形成的低点）。
    *   **动态调整**：可结合 ATR (平均真实波幅) 进行动态止损，例如 `Entry - 2 * ATR`。

*   **止盈 (Take Profit)**：
    *   使用 **斐波那契扩展 (Fibonacci Expansions)** 工具计算目标位。
    *   基于 A-B-C 波段（A=起点, B=第一次高点, C=回调低点）。
    *   **目标位**：
        *   **COP (Contracted Objective Point)**: 0.618 扩展位。
        *   **OP (Objective Point)**: 1.000 扩展位。
        *   **XOP (Expanded Objective Point)**: 1.618 扩展位。

### 1.3 头寸控制 (Position Sizing)

*   **风险模型**：基于账户权益的固定百分比风险（例如每笔交易风险 2%）。
*   **计算公式**：
    $$ \text{头寸数量} = \frac{\text{账户权益} \times \text{风险百分比}}{|\text{进场价} - \text{止损价}|} $$

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
    *   激进：在价格触及 3x3 DMA 时直接挂单买入。
    *   保守：等待 K 线收盘，确认收盘价仍在 3x3 DMA 附近或之上（未完全崩盘）后进场。

### 2.2 止损与止盈 (Exit Rules)

*   **止损 (Stop Loss)**：
    *   设置在最近的一个显著摆动低点下方。
    *   或使用 ATR 止损来适应市场波动。

*   **止盈 (Take Profit)**：
    *   主要目标是趋势恢复。
    *   可以使用 **斐波那契回撤 (Fibonacci Retracements)** 的反向逻辑，或者前高点。
    *   通常目标较为保守，旨在获取“面包和黄油”般的稳定收益。

### 2.3 头寸控制 (Position Sizing)

*   采用与 Double Repo 相同的风险百分比模型。
*   由于是顺势交易，胜率通常较高，但盈亏比可能略低于反转交易。

---

## 3. 总结对比

| 特征 | Double Repo | Single Penetration |
| :--- | :--- | :--- |
| **模式类型** | 趋势反转 (Reversal) | 趋势跟踪 (Trend Following) |
| **核心指标** | 3x3 DMA, 25x5 DMA | 3x3 DMA |
| **进场信号** | 两次穿透 3x3 DMA | 强劲推力后首次回调触及 3x3 DMA |
| **风险偏好** | 较高 (摸顶抄底) | 中等 (顺势回调) |
| **主要止盈** | 斐波那契扩展 (COP/OP/XOP) | 前高或波段目标 |
