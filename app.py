import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.data.feed import DataFeed
from src.indicators.basics import Indicators
from src.strategies.patterns import PatternRecognizer
from src.utils.visualization import Visualizer
from src.ml.classifier import SignalClassifier
from src.ml.llm_analyst import LLMAnalyst
from src.utils.performance import PerformanceAnalyzer
from src.utils.performance import PerformanceAnalyzer
from src.optimization.optimizer import StrategyOptimizer
from src.ml.trainer import ModelTrainer

st.set_page_config(page_title="DiNapoli Trader", layout="wide")

st.title("DiNapoli Quantitative Trading System")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol", "000001.SZ")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Strategy Parameters (Always visible)
st.sidebar.markdown("---")
st.sidebar.header("Strategy Parameters")
# Use session state keys for persistence and programmatic updates
if 'holding_period' not in st.session_state: st.session_state.holding_period = 5
if 'stop_loss' not in st.session_state: st.session_state.stop_loss = 2.0
if 'take_profit' not in st.session_state: st.session_state.take_profit = 5.0

holding_period = st.sidebar.number_input("Holding Period (Bars)", min_value=1, value=st.session_state.holding_period, key='holding_period_input')
stop_loss = st.sidebar.number_input("Stop Loss (%)", min_value=0.1, value=st.session_state.stop_loss, step=0.1, key='stop_loss_input') / 100
take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.1, value=st.session_state.take_profit, step=0.1, key='take_profit_input') / 100
enable_trend_filter = st.sidebar.checkbox("Enable Trend Filter (SMA 200)", value=False)

# Sync inputs back to session state (if changed manually)
st.session_state.holding_period = st.session_state.holding_period_input
st.session_state.stop_loss = st.session_state.stop_loss_input
st.session_state.take_profit = st.session_state.take_profit_input

# --- Optimizer Section ---
with st.sidebar.expander("⚙️ Strategy Optimizer"):
    st.write("Parameter Ranges:")
    opt_sl_min = st.number_input("Min Stop Loss %", 0.5, 5.0, 1.0, 0.5)
    opt_sl_max = st.number_input("Max Stop Loss %", 0.5, 5.0, 3.0, 0.5)
    opt_tp_min = st.number_input("Min Take Profit %", 1.0, 10.0, 3.0, 1.0)
    opt_tp_max = st.number_input("Max Take Profit %", 1.0, 10.0, 7.0, 1.0)
    
    if st.button("Run Optimization"):
        if st.session_state.analyzed and st.session_state.data is not None:
            with st.spinner("Optimizing..."):
                # Define Grid
                # Simple grid: 3 steps for each
                import numpy as np
                sl_range = np.linspace(opt_sl_min, opt_sl_max, 3) / 100
                tp_range = np.linspace(opt_tp_min, opt_tp_max, 3) / 100
                hp_range = [3, 5, 8]
                
                param_grid = {
                    'holding_period': hp_range,
                    'stop_loss': sl_range,
                    'take_profit': tp_range
                }
                
                # We need signals for optimization. 
                # Re-detect signals to be safe (fast enough)
                opt_df = st.session_state.data
                opt_recognizer = PatternRecognizer(opt_df)
                opt_signals = opt_recognizer.detect_double_repo() # Optimize on DR for now
                # Merge SP if needed, but let's stick to DR for core optimization or both
                opt_signals_sp = opt_recognizer.detect_single_penetration()
                opt_signals.update(opt_signals_sp)
                
                optimizer = StrategyOptimizer(opt_df, opt_signals)
                results = optimizer.grid_search(param_grid)
                
                st.session_state.opt_results = results
                st.toast("Optimization Complete!", icon="🚀")
        else:
            st.error("Please run analysis first.")

if 'opt_results' in st.session_state and not st.session_state.opt_results.empty:
    st.sidebar.markdown("---")
    st.sidebar.write("Top Result:")
    best = st.session_state.opt_results.iloc[0]
    st.sidebar.write(f"Return: {best['Total Return']:.2%}")
    st.sidebar.write(f"SL: {best['stop_loss']:.1%}, TP: {best['take_profit']:.1%}")
    
    def apply_best_params():
        st.session_state.holding_period_input = int(best['holding_period'])
        st.session_state.stop_loss_input = float(best['stop_loss'] * 100)
        st.session_state.take_profit_input = float(best['take_profit'] * 100)
        # Also update the sync variables
        st.session_state.holding_period = int(best['holding_period'])
        st.session_state.stop_loss = float(best['stop_loss'] * 100)
        st.session_state.take_profit = float(best['take_profit'] * 100)

    st.sidebar.button("Apply Best Params", on_click=apply_best_params)

# --- ML Lab Section ---
with st.sidebar.expander("🧪 ML Lab"):
    st.write("Train a model to filter signals.")
    if st.button("Train ML Model"):
        if st.session_state.analyzed and st.session_state.data is not None:
            with st.spinner("Training Model (Simulated)..."):
                # Use current data for training (in reality, should use historical database)
                train_df = st.session_state.data
                train_recognizer = PatternRecognizer(train_df)
                train_signals = train_recognizer.detect_double_repo()
                train_signals.update(train_recognizer.detect_single_penetration())
                
                trainer = ModelTrainer()
                result = trainer.train(train_df, train_signals)
                
                if result['status'] == 'success':
                    st.success(f"Trained on {result['samples']} signals.")
                    st.write(f"Accuracy: {result['accuracy']:.2f}")
                    st.write(f"Precision: {result['precision']:.2f}")
                else:
                    st.error(result['message'])
        else:
            st.error("Load data first.")
            
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

# --- Risk Management Section ---
with st.sidebar.expander("🛡️ Risk Management"):
    initial_capital = st.number_input("Initial Capital", value=100000.0, step=1000.0)
    use_dynamic_sizing = st.checkbox("Use Dynamic Position Sizing", value=False)
    risk_per_trade = st.number_input("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
    use_atr_sl = st.checkbox("Use ATR Stop Loss", value=False)
    atr_multiplier = st.number_input("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)

if st.sidebar.button("Analyze"):
    with st.spinner("Fetching Data..."):
        feed = DataFeed()
        try:
            df = feed.fetch_data(symbol, str(start_date), str(end_date))
            if not df.empty:
                st.session_state.data = df
                st.session_state.analyzed = True
                st.toast(f"Loaded {len(df)} bars for {symbol}", icon="✅")
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if st.session_state.analyzed and st.session_state.data is not None:
    df = st.session_state.data
    
    # Note: Synthetic data check (simple heuristic or flag if we had one)
    # For now just show the note if we have data
    # st.toast("Note: If external data fetch failed, synthetic data is displayed.", icon="ℹ️")
            

    
    # Calculate Indicators
    dma_3x3 = Indicators.displaced_ma(df['close'], 3, 3)
    dma_7x5 = Indicators.displaced_ma(df['close'], 7, 5)
    dma_25x5 = Indicators.displaced_ma(df['close'], 25, 5)
    sma_200 = Indicators.sma(df['close'], 200)
    
    # Detect Patterns
    recognizer = PatternRecognizer(df)
    signals_dr = recognizer.detect_double_repo()
    signals_sp = recognizer.detect_single_penetration()
    
    # Apply Trend Filter if enabled
    if enable_trend_filter:
        signals_dr = recognizer.apply_trend_filter(signals_dr, sma_200)
        signals_sp = recognizer.apply_trend_filter(signals_sp, sma_200)
    
    # Merge signals
    # Priority: Double Repo > Single Penetration (if overlap, though unlikely)
    signals = signals_dr.copy()
    signals.update(signals_sp) # This overwrites, which is fine. Or we can combine.
    
    # Combine for display
    # If DR has signal, keep DR. If DR is NaN and SP has signal, use SP.
    mask_sp = (signals_dr['signal'].isna()) & (signals_sp['signal'].notna())
    signals.loc[mask_sp] = signals_sp.loc[mask_sp]
    
    # Add markers for signals (moved up to define buy_signals earlier)
    buy_signals = signals[signals['signal'] == 'BUY']
    
    # ML Prediction
    if not buy_signals.empty:
        clf = SignalClassifier()
        if clf.is_trained:
            probs = []
            for idx in buy_signals.index:
                # Find integer location
                if idx in df.index:
                    prob = clf.predict_proba(df, idx)
                    probs.append(prob)
                else:
                    probs.append(0.5)
            buy_signals['confidence'] = probs
            
            # Filter by confidence
            # We need to update the main 'signals' dataframe to reflect this filter
            # First, ensure confidence column exists in signals
            signals['confidence'] = np.nan
            signals.loc[buy_signals.index, 'confidence'] = buy_signals['confidence']
            
            # Set signal to NaN for those that don't meet confidence
            low_conf_indices = buy_signals[buy_signals['confidence'] < min_confidence].index
            signals.loc[low_conf_indices, 'signal'] = np.nan
            signals.loc[low_conf_indices, 'pattern'] = np.nan
            
            # Re-filter buy_signals for display
            buy_signals = buy_signals[buy_signals['confidence'] >= min_confidence]
        else:
            buy_signals['confidence'] = 0.5 # Default if no model

    # Performance Metrics Calculation
    equity_curve = None
    drawdown_curve = None
    metrics = None
    
    if not buy_signals.empty:
        perf_analyzer = PerformanceAnalyzer(df, signals)
        metrics = perf_analyzer.calculate_metrics(
            holding_period=holding_period, 
            stop_loss_pct=stop_loss, 
            take_profit_pct=take_profit,
            initial_capital=initial_capital,
            use_dynamic_sizing=use_dynamic_sizing,
            risk_per_trade_pct=risk_per_trade,
            atr_multiplier=atr_multiplier if use_atr_sl else 0.0
        )
        equity_curve = metrics['Equity Curve']
        drawdown_curve = metrics['Drawdown Curve']

    # Visualization
    indicators = {
        'DMA 3x3': dma_3x3,
        'DMA 7x5': dma_7x5,
        'DMA 25x5': dma_25x5
    }
    if enable_trend_filter:
        indicators['SMA 200'] = sma_200
    
    fig = Visualizer.plot_chart(df, indicators, equity=equity_curve, drawdown=drawdown_curve, title=f"{symbol} DiNapoli Analysis")
    
    # Add markers for signals
    buy_signals_dr = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Double Repo')]
    buy_signals_sp = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Single Penetration')]
    
    if not buy_signals_dr.empty:
        fig.add_scatter(x=buy_signals_dr.index, y=df.loc[buy_signals_dr.index, 'low']*0.99, mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Double Repo Buy')
        
    if not buy_signals_sp.empty:
        fig.add_scatter(x=buy_signals_sp.index, y=df.loc[buy_signals_sp.index, 'low']*0.99, mode='markers', marker=dict(color='blue', size=8, symbol='triangle-up'), name='Single Pen. Buy')
        
    st.plotly_chart(fig, width='stretch')
    
    # Performance Metrics Display
    if metrics:
        st.markdown("---")
        st.subheader("Strategy Performance (Estimated)")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", metrics['Total Trades'])
        m2.metric("Win Rate", f"{metrics['Win Rate']:.1%}")
        m3.metric("Avg Return", f"{metrics['Avg Return']:.2%}")
        m4.metric("Total Return", f"{metrics['Total Return']:.2%}")
        m5.metric("Ann. Return", f"{metrics['Annualized Return']:.2%}")
        m6.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

    st.markdown("---")
    st.subheader("Detected Signals & Trade Log")
    
    if metrics and not metrics['Trade Log'].empty:
        st.write("Detailed Trade Log (Simulated):")
        st.dataframe(metrics['Trade Log'].style.format({
            'Entry Price': '{:.2f}',
            'Exit Price': '{:.2f}',
            'PnL Amount': '{:.2f}',
            'PnL %': '{:.2%}',
            'Confidence': '{:.2%}'
        }))
    elif not buy_signals.empty:
        st.write("Raw Signals (Filtered):")
        st.dataframe(buy_signals.style.format({'confidence': '{:.2%}'}))
    else:
        st.info("No signals to display.")

    st.markdown("---")
    st.subheader("🤖 AI Analyst (Gemini)")
    
    if st.button("Ask AI Analyst"):
        with st.spinner("Analyzing market context..."):
            analyst = LLMAnalyst()
            
            # Construct context
            last_close = df['close'].iloc[-1]
            last_dma25 = dma_25x5.iloc[-1]
            trend = "Bullish" if last_close > last_dma25 else "Bearish"
            
            # Volatility (ATR-like or simple std)
            vol = df['close'].pct_change().std() * 100
            vol_str = f"{vol:.2f}% (Daily)"
            
            # RSI
            rsi = Indicators.rsi(df['close'], 14).iloc[-1]
            
            # Pattern
            recent_pattern = "None"
            if not buy_signals.empty:
                recent_pattern = buy_signals['pattern'].iloc[-1]
            
            context = {
                'symbol': symbol,
                'trend': trend,
                'volatility': vol_str,
                'pattern': recent_pattern,
                'rsi': f"{rsi:.2f}",
                'close': f"{last_close:.2f}"
            }
            
            analysis = analyst.analyze_context(context)
            st.write(analysis)

st.sidebar.markdown("---")
st.sidebar.info("System Ready")
