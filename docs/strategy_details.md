# 交易策略详情文档

本文档详细描述了系统支持的两种核心交易模式：**Double Repo (双重穿透)** 和 **Single Penetration (单次穿透)**。

> [!NOTE]
> **Railroad Tracks (RRT)** 和 **Failure to Penetrate (FTP)** 已被重新定义为**信号增强过滤器 (Signal Enhancement Filters)**，不再作为独立的进场策略，而是用于增强核心模式的信号质量。

## 1. 模式一：Double Repo (双重穿透)

Double Repo 是一种“失败的突破”模式，通常发生在趋势末端，预示着趋势的反转。

### 1.1 进场规则 (Entry Rules)

以**看涨 (Buy)** 为例（看跌逻辑相反）：

1.  **初始状态**：价格位于 **3x3 DMA** 之下，且趋势向下。
    -   **趋势确认**：在第一次穿透前，收盘价必须连续 `min_trend_bars` (默认 3 根) 位于 3x3 DMA 之下。
2.  **第一次穿透**：收盘价向上突破 3x3 DMA。
3.  **回落**：收盘价重新跌回 3x3 DMA 之下。
4.  **第二次穿透 (信号触发)**：收盘价再次向上突破 3x3 DMA。
5.  **过滤条件**：
    -   **25x5 DMA 过滤**：价格必须“远离” 25x5 DMA。
        -   **逻辑**：`(DMA25 - Close) / Close > Min Dist %` (对于 Buy)。
        -   **目的**：确保价格处于超卖状态，增加反转概率。默认最小距离为 0.5%。
    -   **信号增强**：如果同时检测到 **RRT** 或 **FTP** 形态，信号置信度增加。

### 1.2 止损与止盈 (Exit Rules)
系统支持三种止损模式和两种止盈模式，可根据市场环境灵活选择：

*   **止损模式 (Stop Loss Modes)**：
    1.  **Pattern Based (形态止损)**：
        *   使用形态结构的最低点。
        *   对于 Double Repo：使用第一次穿透后形成的最低点 (Pattern Low)。
    2.  **ATR Based (波动率止损)**：
        *   基于 ATR (平均真实波幅) 动态调整。
        *   公式：`Entry Price - (ATR * Multiplier)`。
    3.  **Fixed Percentage (固定百分比)**：
        *   使用固定的百分比回撤。

*   **止盈模式 (Take Profit Modes)**：
    1.  **Pattern Based (Fibonacci)**：
        *   使用斐波那契扩展目标位。
        *   对于 Double Repo：默认目标为 OP (1.0 扩展) 或 COP (0.618 扩展)。
    2.  **Fixed Percentage (固定百分比)**：
        *   使用固定的百分比目标。

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
3.  **进场方式 (Touch Entry)**：
    *   **限价单 (Limit Order)**：直接在 3x3 DMA 的价格位置挂单进场，无需等待 K 线收盘确认。
    *   *注意：这与之前的收盘确认逻辑不同，旨在获得更好的入场价格。*
4.  **信号增强**：
    *   检测回调过程中是否伴随 **FTP** (穿透失败) 或 **RRT** 形态，作为确认信号。

### 2.2 止损与止盈 (Exit Rules)

*   **止损模式**：
    *   **Pattern Based**：使用推力启动时的低点 (Thrust Start Low) 或最近的摆动低点。
    *   同样支持 **ATR Based** 和 **Fixed Percentage**。

*   **止盈模式**：
    *   **Pattern Based**：通常以推力的高点 (Thrust High) 为初步目标。
    *   同样支持 **Fixed Percentage**。

---

## 3. 信号增强过滤器 (Signal Enhancement Filters)

以下模式不再独立触发交易，而是作为上述核心策略的辅助确认指标。

### 3.1 Railroad Tracks (RRT / 铁轨形态)
*   **特征**：两根方向相反、实体较长且长度相近的 K 线并列。
*   **作用**：强烈的反转信号。如果在 Double Repo 的第二次穿透或 Single Penetration 的回调底部出现，极大地增加了成功的概率。

### 3.2 Failure to Penetrate (FTP / 穿透失败)
*   **特征**：价格盘中突破 3x3 DMA 但收盘收回（留长影线）。
*   **作用**：确认支撑/阻力的有效性。在 Single Penetration 的触及进场后，如果当日收盘形成 FTP，是极佳的确认信号。

---

## 4. 总结对比

| 特征 | Double Repo | Single Penetration |
| :--- | :--- | :--- |
| **模式类型** | 趋势反转 (Reversal) | 趋势跟踪 (Trend Following) |
| **核心指标** | 3x3 DMA, 25x5 DMA | 3x3 DMA |
| **进场信号** | 两次穿透 3x3 DMA (Close Confirmation) | 强劲推力后首次回调触及 3x3 DMA (**Touch Entry**) |
| **辅助过滤** | RRT, FTP | RRT, FTP |
| **主要止盈** | 斐波那契扩展 (OP) | 推力高点 (Thrust High) |
