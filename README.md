# DiNapoli量化交易策略

开发一套基于乔·帝纳波利（Joe DiNapoli）斐波那契分析的量化交易策略。该策略旨在克服传统帝纳波利方法的主观性，通过算法客观识别交易模式，并利用传统技术指标、机器学习（ML）、深度学习（DL）和大模型（LLM）技术增强信号的准确性、优化风险管理，最终实现一个稳健、自适应、可视化的自动化交易、策略研究分析回测、模型优化、部署与迭代的完整系统。

## 核心功能 (Core Features)

### 1. 数据层 (Data Layer)
- **多源数据支持**: 集成 `Tushare` (A股) 和 `YFinance` (美股/全球)，并支持合成数据 (Synthetic) 作为回退。
- **自动缓存**: 数据本地缓存，减少 API 调用。

### 2. 策略层 (Strategy Layer)
- **DiNapoli 核心组件**:
    - **Displaced Moving Averages (DMA)**: 3x3, 7x5, 25x5 置换移动平均线。
    - **Fibonacci Analysis**: 自动识别 Swing Points，计算回撤 (Retracement) 和扩展 (Expansion) 目标位 (COP, OP, XOP)。
- **模式识别 (Pattern Recognition)**:
    - **Double Repo**: 经典的趋势反转模式。
    - **Single Penetration**: "面包与黄油" (Bread & Butter) 趋势延续模式。
- **信号过滤**:
    - **Trend Filter**: 基于 SMA 200 的大趋势过滤。
    - **ML Confidence**: 基于随机森林 (Random Forest) 的信号置信度评分。

### 3. 智能增强 (AI Enhancement)
- **ML Lab**: 内置机器学习实验室，支持一键训练模型，评估信号质量。
- **LLM Analyst**: 集成 Google Gemini Pro，提供基于市场语境 (Context-Aware) 的自然语言分析报告。

### 4. 回测与风控 (Backtest & Risk)
- **可视化仪表盘**: 基于 Streamlit 的交互式界面，集成 Plotly 图表。
- **性能分析**: 实时计算权益曲线 (Equity Curve)、回撤曲线 (Drawdown)、胜率、盈亏比等指标。
- **动态风控 (Dynamic Risk)**:
    - **ATR Stop Loss**: 基于波动率的动态止损。
    - **Position Sizing**: 基于风险敞口的动态仓位管理。
- **参数优化**: 内置网格搜索 (Grid Search) 优化器，寻找最佳止盈止损参数。

## 快速开始 (Quick Start)

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境**:
   复制 `.env.example` 为 `.env`，填入 Tushare Token 和 Gemini API Key。

3. **运行应用**:
   ```bash
   streamlit run app.py
   ```

## 项目结构 (Project Structure)
- `src/data`: 数据获取与处理
- `src/indicators`: 技术指标 (DMA, MACD, ZigZag)
- `src/strategies`: 策略逻辑与模式识别
- `src/ml`: 机器学习与 LLM 分析师
- `src/risk`: 风控管理
- `src/optimization`: 参数优化器
- `src/utils`: 性能分析与可视化
