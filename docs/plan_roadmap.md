# DiNapoli Trading System - Development Roadmap

## Phase 1: Foundation & Infrastructure (Weeks 1-2)
**Goal:** Build the data pipeline and basic backtesting engine.
- [ ] **Data Ingestion**: Implement `DataFeed` class to fetch OHLCV from CSV/API.
- [ ] **Data Storage**: Setup SQLite/Parquet storage for historical data.
- [ ] **Basic Indicators**: Implement SMA, EMA, MACD, RSI as baseline features.
- [ ] **Visualization**: Simple Matplotlib/Plotly chart with candlestick support.

## Phase 2: Core DiNapoli Strategy (Weeks 3-5)
**Goal:** Implement the specific DiNapoli analysis tools.
- [ ] **Displaced Moving Averages (DMA)**: Implement 3x3, 7x5, 25x5 DMAs.
- **Fibonacci Engine**:
    - [ ] **Swing Detection**: Algorithm to identify "Focus Numbers" (significant highs/lows).
    - [ ] **Retracements**: Calculate 0.382, 0.618 levels dynamically.
    - [ ] **Expansions**: Calculate COP, OP, XOP targets.
- **Pattern Recognition**:
    - [ ] **Double Repo**: Logic for "Close below 3x3 -> High -> Close above 3x3 -> New Low -> Close above 3x3".
    - [ ] **Bread & Butter**: Re-entry logic on strong trends.

## Phase 3: Advanced Intelligence (Weeks 6-8)
**Goal:** Integrate ML and LLM to filter signals.
- [ ] **Feature Engineering**: Create dataset with DiNapoli features + Volatility + Volume.
- [ ] **ML Classifier**: Train Random Forest/XGBoost to predict "Win/Loss" of a pattern.
    - *Input*: Pattern type, Distance to DMA, RSI divergence, Market Regime.
    - *Target*: Profit > 1R within N bars.
- [ ] **LLM Analyst**:
    - Build a prompt template: "Analyze this chart context: [Trend=Up, Volatility=High, News=Bearish]. Should we take a Long Double Repo?"
    - Connect to Gemini/OpenAI API for "Second Opinion".

## Phase 4: System Hardening & Deployment (Weeks 9-10)
**Goal:** Production-ready system.
- [ ] **Risk Management Module**: Kelly Criterion sizing, Max Drawdown limits.
- [ ] **Paper Trading**: Connect to exchange sandbox.
- [ ] **Dashboard**: Real-time web UI (Streamlit) showing active signals and account health.

## Phase 5: Strategy Optimization & Alpha Generation (Completed)
**Goal:** Enhance profitability through rigorous optimization and advanced filtering.
- [x] **Signal Filtering**:
    - **Trend Alignment**: Only take signals in direction of higher timeframe trend (SMA 200).
    - **Volatility Filter**: ATR-based filtering.
- [x] **Parameter Optimization**:
    - **Grid Search**: Find optimal Holding Period, Stop Loss, and Take Profit.
- [x] **Advanced Patterns**:
    - **Single Penetration (Bread & Butter)**: Implemented high-probability trend continuation pattern.
- [x] **Real ML Model Training**:
    - **Random Forest Classifier**: Trained on historical signals to predict success probability.
    - **ML Lab**: UI for training and evaluating models.
- [x] **Dynamic Risk Management**:
    - **ATR-based Stop Loss**: Adjust SL based on volatility.
    - **Dynamic Position Sizing**: Risk-based sizing (e.g., 1% risk per trade).

## Phase 6: Core Optimization & Advanced Analytics (Completed)
**Goal:** Deepen the strategy's robustness and analytical capabilities.
- [x] **Multi-Timeframe Analysis (MTF)**:
    - **Trend Confirmation**: Confirm Daily signals with Weekly trend direction.
    - **MTF Dashboard**: View Weekly/Monthly charts alongside Daily.
- [x] **Market Scanner (Batch Analysis)**:
    - **Sector Scan**: Scan lists of symbols (e.g., CSI 300, SP 500) for active signals.
    - **Opportunity Table**: Dashboard view ranking top opportunities by ML Confidence.
- [x] **Advanced ML Insights**:
    - **Feature Importance**: Visualize which factors (RSI, DMA, Volatility) drive the model's decisions.
    - **Model Management**: Save/Load distinct models for different asset classes.
- [x] **Robust Backtesting**:
    - **Walk-Forward Analysis**: Validate parameters over rolling time windows.
    - **Detailed Reporting**: Monthly returns heatmap, Sharpe/Sortino ratios.

## Phase 7: Strategy Expansion & Fundamentals (Weeks 19-21)
**Goal:** Diversify signal sources and integrate fundamental data.
- [ ] **Core Strategy Expansion**:
    - **Strategy C: L3 Confluence**: Implementation of high-precision Fibonacci intersection (Retracement + Expansion).
    - **Strategy D: Minesweeper A**: Trend following logic for markets with zero retracement (Stochastics filter).
- [ ] **Signal Filters & Confirmations**:
    - **Railroad Tracks (RRT)**: Reversal pattern logic as a signal booster.
    - **Failure to Penetrate (FTP)**: Logic to detect rejection at key levels (wicks).
    - **Consolidation Breakout**: Volatility squeeze breakouts.
- [ ] **Fundamental Analysis**:
    - **Data Integration**: Fetch PE, PB, ROE, Revenue Growth from Tushare/YFinance.
    - **Valuation Filter**: Filter out overvalued assets in Market Scanner.
    - **Quality Filter**: Ensure trading only high-quality companies (high ROE).
- [ ] **Advanced Trade Management**:
    - **Trailing Stop**: Implement Chandelier Exit or ATR Trailing Stop.
    - **Partial Take Profit**: Scale out positions to lock in gains.
    - **Limit Entry**: Optimize entry price with limit orders at retracement levels.

## Phase 8: Portfolio & Advanced Analytics (Weeks 22-25)
**Goal:** Move from single-asset to portfolio-level management.
- [ ] **Portfolio Backtesting**:
    - **Multi-Asset Engine**: Simulate trading a basket of assets with shared capital.
    - **Correlation Analysis**: Avoid over-exposure to correlated assets.
    - **Portfolio Optimization**: Mean-Variance or Risk Parity allocation.
- [ ] **Advanced Feature Engineering**:
    - **Macro Data**: Interest rates, GDP, Inflation as regime filters.
    - **Sentiment Analysis**: News/Social media sentiment score via LLM.
    - **Micro-Structure**: Order flow and volume profile features.

## Phase 9: Deep Intelligence (Weeks 26-29)
**Goal:** Leverage state-of-the-art AI for alpha generation.
- [ ] **Deep Learning**:
    - **LSTM/GRU**: Capture temporal dependencies in price sequences.
- [ ] **Reinforcement Learning (RL)**:
    - **DQN/PPO Agent**: Train agents to optimize risk-adjusted returns in a simulated environment.
- [ ] **Explainable AI (XAI)**:
    - **SHAP Values**: Interpret complex model decisions.
