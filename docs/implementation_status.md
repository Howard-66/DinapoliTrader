# 策略实现状态审计报告 (Strategy Implementation Status Audit)

本文档基于 `docs/strategy_details.md` 中的定义，对当前代码库 (`src/`) 进行逐项审计，确认功能的实现与集成状态。

## 1. 审计概览

| 功能模块 | 模式 | 实现状态 (Code Exists?) | 集成状态 (Running in Backtest?) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **进场规则** | Double Repo | ✅ 已实现 (`patterns.py`) | ✅ 已集成 | 回测引擎默认调用此模式生成信号。 |
| **进场规则** | Single Penetration | ✅ 已实现 (`patterns.py`) | ❌ **未集成** | 代码中有定义，但 `BacktestEngine` 未调用。 |
| **止损规则** | 通用 (ATR/Low) | ✅ 已实现 (`risk/manager.py`) | ❌ **未集成** | 策略执行时未设置止损单。 |
| **止盈规则** | 通用 (Fibonacci) | ✅ 已实现 (`fibonacci.py`) | ❌ **未集成** | 策略执行时未设置止盈单。 |
| **头寸控制** | 动态仓位 | ✅ 已实现 (`risk/manager.py`) | ❌ **未集成** | 回测目前使用 Backtrader 默认仓位（通常为 1 单位或全部资金，取决于 Sizer 配置）。 |

---

## 2. 详细分析

### 2.1 进场规则 (Entry Rules)

*   **Double Repo**:
    *   **代码位置**: `src/strategies/patterns.py` -> `detect_double_repo`
    *   **现状**: `src/backtest/engine.py` 中显式调用了 `recognizer.detect_double_repo()` 并将结果传入策略。
    *   **结论**: **完全工作**。

*   **Single Penetration**:
    *   **代码位置**: `src/strategies/patterns.py` -> `detect_single_penetration`
    *   **现状**: `src/backtest/engine.py` **没有**调用此方法。
    *   **结论**: **代码存在但未启用**。如需测试此模式，需要修改 Engine 代码。

### 2.2 止损与止盈 (Exit Rules)

*   **代码位置**:
    *   SL: `src/risk/manager.py` -> `calculate_atr_stop_loss`
    *   TP: `src/strategies/fibonacci.py` -> `calculate_expansions` / `calculate_retracements`
*   **现状**:
    *   `src/backtest/strategy.py` 中的 `DiNapoliStrategyWithSignals.next()` 方法仅执行 `self.buy()` 或 `self.sell()`。
    *   它**没有**计算止损/止盈价格。
    *   它**没有**提交附带的 `Stop` (止损) 或 `Limit` (止盈) 订单 (Bracket Orders)。
*   **当前执行规则**:
    *   **纯信号驱动**：只有当信号源发出反向信号（例如从 BUY 变为 SELL）时，或者策略内部逻辑平仓时才会离场。
    *   由于当前信号源主要是 Entry 信号，**目前的回测实际上可能没有止损止盈，或者仅在反手时平仓**。

### 2.3 头寸控制 (Position Sizing)

*   **代码位置**: `src/risk/manager.py` -> `calculate_position_size`
*   **现状**:
    *   `src/backtest/strategy.py` 中直接调用 `self.buy()`，未指定 `size` 参数。
    *   Backtrader 默认 Sizer 通常是 `FixedSize` (1股) 或者需要显式配置。
    *   `src/backtest/engine.py` 设置了初始资金 `setcash`，但未设置自定义 Sizer 来调用我们的 `RiskManager`。
*   **当前执行规则**:
    *   使用 Backtrader 的默认 Sizer（通常是每次交易 1 个单位，或者取决于全局配置）。**未利用风险管理模块计算的动态仓位。**

---

## 3. 修复建议 (Next Steps)

为了使回测符合文档描述，需要进行以下集成工作：

1.  **集成 Single Penetration**: 修改 `BacktestEngine` 以支持选择或混合两种策略模式。
2.  **集成 SL/TP**:
    *   在 `DiNapoliStrategyWithSignals` 中，当接收到开仓信号时：
        *   调用 `RiskManager` 计算 SL。
        *   调用 `FibonacciEngine` 计算 TP。
        *   使用 `self.buy_bracket(...)` 或手动提交 `Stop` 和 `Limit` 订单来执行带有保护的交易。
3.  **集成动态仓位**:
    *   在开仓前，调用 `RiskManager.calculate_position_size` 计算手数。
    *   将计算出的 `size` 传递给 `self.buy(size=...)`。

---

**总结**: 目前系统处于“信号生成逻辑完备，但执行逻辑简陋”的阶段。核心的 DiNapoli 交易理念（进场）已实现，但风控和出场（SL/TP/Sizing）尚未闭环。
