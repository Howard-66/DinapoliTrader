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
from src.optimization.optimizer import StrategyOptimizer
from src.ml.trainer import ModelTrainer
from src.utils.scanner import MarketScanner

st.set_page_config(page_title="DiNapoli Trader", layout="wide")

# st.title("DiNapoli Quantitative Trading System")

# Sidebar
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Single Analysis", "Market Scanner"])

if mode == "Single Analysis":
    symbol = st.sidebar.text_input("Symbol", "601398.SH")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Strategy Parameters (Always visible)
st.sidebar.header("Exit Strategy")

# Use session state keys for persistence
if 'sl_mode' not in st.session_state: st.session_state.sl_mode = 'Pattern Based'
if 'tp_mode' not in st.session_state: st.session_state.tp_mode = 'Pattern Based (Fib)'
if 'holding_period' not in st.session_state: st.session_state.holding_period = 15
if 'stop_loss' not in st.session_state: st.session_state.stop_loss = 2.0
if 'take_profit' not in st.session_state: st.session_state.take_profit = 5.0
if 'atr_multiplier' not in st.session_state: st.session_state.atr_multiplier = 2.0

# Stop Loss Configuration
st.sidebar.subheader("Stop Loss")
sl_mode = st.sidebar.selectbox(
    "Stop Loss Mode",
    ["Pattern Based", "ATR Based", "Fixed Percentage"],
    index=["Pattern Based", "ATR Based", "Fixed Percentage"].index(st.session_state.sl_mode),
    key='sl_mode_input'
)

stop_loss = 0.02 # Default
atr_multiplier = 2.0 # Default

if sl_mode == "Fixed Percentage":
    stop_loss = st.sidebar.number_input("Stop Loss (%)", min_value=0.1, value=st.session_state.stop_loss, step=0.1, key='stop_loss_input') / 100
elif sl_mode == "ATR Based":
    atr_multiplier = st.sidebar.number_input("ATR Multiplier", 1.0, 5.0, st.session_state.atr_multiplier, 0.5, key='atr_multiplier_input')

# Take Profit Configuration
st.sidebar.subheader("Take Profit")
tp_mode = st.sidebar.selectbox(
    "Take Profit Mode",
    ["Pattern Based (Fib)", "Fixed Percentage"],
    index=["Pattern Based (Fib)", "Fixed Percentage"].index(st.session_state.tp_mode),
    key='tp_mode_input'
)

take_profit = 0.05 # Default

if tp_mode == "Fixed Percentage":
    take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.1, value=st.session_state.take_profit, step=0.1, key='take_profit_input') / 100

# Time Exit
st.sidebar.subheader("Time Exit")
holding_period = st.sidebar.number_input("Holding Period (Bars)", min_value=1, value=st.session_state.holding_period, key='holding_period_input')

# Risk Management (Sizing)
st.sidebar.header("Risk Management")
st.sidebar.subheader("Position Sizing")
initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0, step=1000.0)
use_dynamic_sizing = st.sidebar.checkbox("Use Dynamic Position Sizing", value=True)
risk_per_trade = 0.01
if use_dynamic_sizing:
    risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100

st.sidebar.header("Filter")
enable_trend_filter = st.sidebar.checkbox("Enable Trend Filter (SMA 200)", value=False)

# Sync inputs back to session state
st.session_state.sl_mode = sl_mode
st.session_state.tp_mode = tp_mode
st.session_state.holding_period = holding_period
if sl_mode == "Fixed Percentage":
    st.session_state.stop_loss = stop_loss * 100
if sl_mode == "ATR Based":
    st.session_state.atr_multiplier = atr_multiplier
if tp_mode == "Fixed Percentage":
    st.session_state.take_profit = take_profit * 100

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

if mode == "Single Analysis":
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

elif mode == "Market Scanner":
    st.title("Market Scanner 🔍")
    st.write("Scan multiple symbols for DiNapoli patterns.")
    
    # Input for symbols
    default_symbols = "688981.SH, 601398.SH, 601111.SH, 600036.SH, 002714.SZ, 300088.SZ, 601688.SH"
    symbols_input = st.text_area("Enter Symbols (comma separated)", default_symbols, height=100)
    
    c1, c2 = st.columns(2)
    with c1:
        lookback = st.number_input("Lookback Days", min_value=30, max_value=365, value=200)
    with c2:
        scan_window = st.number_input("Scan Window (Bars)", min_value=1, max_value=20, value=5, help="Number of recent bars to check for signals.")
    
    if st.button("Scan Market"):
        symbols_list = [s.strip() for s in symbols_input.split(',') if s.strip()]
        
        if symbols_list:
            scanner = MarketScanner()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # We can't easily hook into the scanner loop for progress without modifying scanner to yield
            # So we'll just show a spinner for now, or modify scanner later.
            # For now, just run it.
            
            with st.spinner(f"Scanning {len(symbols_list)} symbols..."):
                results = scanner.scan(symbols_list, lookback_days=lookback, scan_window=scan_window)
                
            if not results.empty:
                st.success(f"Found {len(results)} active signals!")
                
                # Style the dataframe
                st.dataframe(results.style.applymap(
                    lambda x: 'color: green' if x == 'BUY' else 'color: red', subset=['Signal']
                ).format({
                    'Close': '{:.2f}',
                    'SL': '{:.2f}',
                    'TP': '{:.2f}',
                    'Confidence': '{:.2%}'
                }))
            else:
                st.info("No active signals found in the provided list.")
        else:
            st.warning("Please enter at least one symbol.")
    
    st.stop()

# Only show Single Analysis content if mode is Single Analysis
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
    
    # Strategy Selection
    st.sidebar.header("Strategy Selection")
    selected_strategies = st.sidebar.multiselect(
        "Active Strategies",
        ["Double Repo", "Single Penetration"],
        default=["Double Repo", "Single Penetration"]
    )
    
    # Apply Trend Filter if enabled
    if enable_trend_filter:
        signals_dr = recognizer.apply_trend_filter(signals_dr, sma_200)
        signals_sp = recognizer.apply_trend_filter(signals_sp, sma_200)
    
    # Merge signals based on selection
    signals = pd.DataFrame(index=df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp'])
    
    if "Double Repo" in selected_strategies:
        # Merge DR signals
        mask_dr = signals_dr['signal'].notna()
        signals.loc[mask_dr] = signals_dr.loc[mask_dr]
        
    if "Single Penetration" in selected_strategies:
        # Merge SP signals. 
        # Strategy: If DR already has a signal, we can overwrite or keep.
        # Let's assume priority to DR if both happen (rare), or overwrite if we want latest.
        # Here we simply fill where empty or overwrite if SP is present.
        # To avoid conflict, let's say if DR exists, we keep DR.
        mask_sp = signals_sp['signal'].notna()
        # Only fill where signal is currently NaN to give DR priority
        # Or just overwrite? Let's overwrite to show all.
        # Actually, better to prioritize:
        # If "Double Repo" is selected and present, keep it.
        # If "Single Penetration" is selected and present, fill gaps.
        
        # Let's do:
        # 1. Start with empty
        # 2. If DR selected, fill DR
        # 3. If SP selected, fill SP where NaN
        
        if "Double Repo" in selected_strategies:
             mask_fill = (signals['signal'].isna()) & (signals_sp['signal'].notna())
             signals.loc[mask_fill] = signals_sp.loc[mask_fill]
        else:
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
            
            # Fix SettingWithCopyWarning
            buy_signals = buy_signals.copy()
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
            buy_signals = buy_signals.copy()
            buy_signals['confidence'] = 0.5 # Default if no model

    # Performance Metrics Calculation
    equity_curve = None
    drawdown_curve = None
    metrics = None
    
    if not buy_signals.empty:
        perf_analyzer = PerformanceAnalyzer(df, signals)
        metrics = perf_analyzer.calculate_metrics(
            holding_period=holding_period, 
            sl_mode='Pattern' if sl_mode == 'Pattern Based' else ('ATR' if sl_mode == 'ATR Based' else 'Fixed'),
            tp_mode='Pattern' if tp_mode == 'Pattern Based (Fib)' else 'Fixed',
            stop_loss_pct=stop_loss, 
            take_profit_pct=take_profit,
            initial_capital=initial_capital,
            use_dynamic_sizing=use_dynamic_sizing,
            risk_per_trade_pct=risk_per_trade,
            atr_multiplier=atr_multiplier
        )
        equity_curve = metrics['Equity Curve']
        drawdown_curve = metrics['Drawdown Curve']

    # Visualization
    indicators = {}
    if enable_trend_filter:
        indicators['SMA 200'] = sma_200
    
    fig = Visualizer.plot_chart(df, indicators, equity=equity_curve, drawdown=drawdown_curve, trades=metrics['Trade Log'] if metrics else None, title=f"{symbol} Backtesting Results")
    
    # Add markers for signals
    buy_signals_dr = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Double Repo')]
    buy_signals_sp = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Single Penetration')]
    
    if not buy_signals_dr.empty:
        fig.add_scatter(x=buy_signals_dr.index, y=df.loc[buy_signals_dr.index, 'low']*0.99, mode='markers', marker=dict(color='green', size=5, symbol='triangle-up'), name='Double Repo Buy')
        
    if not buy_signals_sp.empty:
        fig.add_scatter(x=buy_signals_sp.index, y=df.loc[buy_signals_sp.index, 'low']*0.99, mode='markers', marker=dict(color='blue', size=5, symbol='triangle-up'), name='Single Pen. Buy')
        
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
            'Stop Loss': '{:.2f}',
            'Take Profit': '{:.2f}',
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

    # st.markdown("---")
    # st.subheader("🤖 AI Analyst (Gemini)")
    
    # if st.button("Ask AI Analyst"):
    #     with st.spinner("Analyzing market context..."):
    #         analyst = LLMAnalyst()
            
    #         # Construct context
    #         last_close = df['close'].iloc[-1]
    #         last_dma25 = dma_25x5.iloc[-1]
    #         trend = "Bullish" if last_close > last_dma25 else "Bearish"
            
    #         # Volatility (ATR-like or simple std)
    #         vol = df['close'].pct_change().std() * 100
    #         vol_str = f"{vol:.2f}% (Daily)"
            
    #         # RSI
    #         rsi = Indicators.rsi(df['close'], 14).iloc[-1]
            
    #         # Pattern
    #         recent_pattern = "None"
    #         if not buy_signals.empty:
    #             recent_pattern = buy_signals['pattern'].iloc[-1]
            
    #         context = {
    #             'symbol': symbol,
    #             'trend': trend,
    #             'volatility': vol_str,
    #             'pattern': recent_pattern,
    #             'rsi': f"{rsi:.2f}",
    #             'close': f"{last_close:.2f}"
    #         }
            
    #         analysis = analyst.analyze_context(context)
    #         st.write(analysis)

# st.sidebar.markdown("---")
# st.sidebar.info("System Ready")
