import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import warnings
from streamlit_echarts5 import st_echarts

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
# Suppress pandas fillna method warning (from tushare)
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*fillna with 'method' is deprecated.*")

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

# Sidebar Configuration
st.sidebar.header("Configuration")

# 1. Analysis Mode: Segmented Control
analysis_mode = st.sidebar.segmented_control(
    "Analysis Mode", 
    ["Single Asset", "Portfolio"],
    default="Single Asset"
)

if analysis_mode == "Single Asset":
    st.sidebar.markdown("---")
    symbol = st.sidebar.text_input("Symbol", "601398.SH")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    # Fetch Data Button (Common)
    if st.sidebar.button("Load Data"):
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

    # Ensure data is loaded
    if st.session_state.analyzed and st.session_state.data is not None:
        df = st.session_state.data
        
        # --- Common Calculations ---
        # Calculate Indicators
        dma_3x3 = Indicators.displaced_ma(df['close'], 3, 3)
        dma_7x5 = Indicators.displaced_ma(df['close'], 7, 5)
        dma_25x5 = Indicators.displaced_ma(df['close'], 25, 5)
        sma_200 = Indicators.sma(df['close'], 200)
        
        # Detect Patterns
        recognizer = PatternRecognizer(df)
        signals_dr = recognizer.detect_double_repo()
        signals_sp = recognizer.detect_single_penetration()
        
        # --- Main Area Tabs ---
        tab_backtest, tab_robustness, tab_ml = st.tabs(["Strategy Backtest", "Walk-Forward Analysis", "Machine Learning Lab"])
        
        # --- Tab 1: Backtest ---
        # --- Tab 1: Backtest ---
        with tab_backtest:
            # st.header(f"Backtest: {symbol}")
            
            with st.expander("⚙️ Strategy Settings", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.write("Active Strategies")
                    selected_strategies = st.pills(
                        "Active Strategies",
                        ["Double Repo", "Single Penetration", "Railroad Tracks", "Failure to Penetrate"],
                        default=["Double Repo", "Single Penetration", "Railroad Tracks", "Failure to Penetrate"],
                        selection_mode="multi",
                        label_visibility="collapsed"
                    )
                    
                    st.write("Filters")
                    enable_trend_filter = st.checkbox("Enable Trend Filter (SMA 200)", value=False)
                    enable_mtf_filter = st.checkbox("Enable MTF Filter (Weekly Trend)", value=False, help="Filter signals based on Weekly 25x5 DMA Trend.")
                    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
                
                with c2:
                    st.write("Exit Strategy")
                    sl_mode = st.selectbox("Stop Loss Mode", ["Pattern Based", "ATR Based", "Fixed Percentage"], index=0)
                    tp_mode = st.selectbox("Take Profit Mode", ["Pattern Based (Fib)", "Fixed Percentage"], index=0)
                    holding_period = st.number_input("Holding Period (Bars)", min_value=1, value=30)
                
                with c3:
                    st.write("Parameters")
                    stop_loss = 0.02
                    atr_multiplier = 2.0
                    take_profit = 0.05
                    
                    if sl_mode == "Fixed Percentage":
                        stop_loss = st.number_input("Stop Loss (%)", 0.1, 20.0, 2.0, 0.1) / 100
                    elif sl_mode == "ATR Based":
                        atr_multiplier = st.number_input("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
                        
                    if tp_mode == "Fixed Percentage":
                        take_profit = st.number_input("Take Profit (%)", 0.1, 50.0, 5.0, 0.1) / 100
                        
                with c4:
                    # Risk Management
                    st.write("Risk Management")
                    initial_capital = st.number_input("Initial Capital", value=100000.0, step=1000.0)
                    use_dynamic_sizing = st.checkbox("Use Dynamic Position Sizing", value=True)
                    risk_per_trade = 0.01
                    if use_dynamic_sizing:
                        risk_per_trade = st.number_input("Risk per Trade (%)", 0.1, 5.0, 2.0, 0.1) / 100

            # --- Backtest Execution ---
            
            # Detect New Patterns
            signals_rrt = recognizer.detect_railroad_tracks()
            signals_ftp = recognizer.detect_failure_to_penetrate()

            # Apply Filters
            if enable_trend_filter:
                signals_dr = recognizer.apply_trend_filter(signals_dr, sma_200)
                signals_sp = recognizer.apply_trend_filter(signals_sp, sma_200)
                signals_rrt = recognizer.apply_trend_filter(signals_rrt, sma_200)
                signals_ftp = recognizer.apply_trend_filter(signals_ftp, sma_200)

            if enable_mtf_filter:
                weekly_df = DataFeed.resample_to_weekly(df)
                signals_dr = recognizer.apply_mtf_filter(signals_dr, weekly_df)
                signals_sp = recognizer.apply_mtf_filter(signals_sp, weekly_df)
                signals_rrt = recognizer.apply_mtf_filter(signals_rrt, weekly_df)
                signals_ftp = recognizer.apply_mtf_filter(signals_ftp, weekly_df)
            
            # Merge Signals
            signals = pd.DataFrame(index=df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
            
            # Helper to merge signals
            def merge_signals(target_df, source_df, strategy_name):
                if strategy_name in selected_strategies:
                    mask = source_df['signal'].notna()
                    # Only fill where target is empty to avoid overwriting (priority: DR > SP > RRT > FTP)
                    mask_fill = (target_df['signal'].isna()) & mask
                    target_df.loc[mask_fill] = source_df.loc[mask_fill]

            merge_signals(signals, signals_dr, "Double Repo")
            merge_signals(signals, signals_sp, "Single Penetration")
            merge_signals(signals, signals_rrt, "Railroad Tracks")
            merge_signals(signals, signals_ftp, "Failure to Penetrate")

            # ML Prediction Filter
            buy_signals = signals[signals['signal'] == 'BUY']
            if not buy_signals.empty:
                clf = SignalClassifier()
                if clf.is_trained:
                    probs = []
                    for idx in buy_signals.index:
                        if idx in df.index:
                            probs.append(clf.predict_proba(df, idx))
                        else:
                            probs.append(0.5)
                    
                    buy_signals = buy_signals.copy()
                    buy_signals['confidence'] = probs
                    
                    # Update main signals df with confidence
                    signals['confidence'] = np.nan
                    signals.loc[buy_signals.index, 'confidence'] = buy_signals['confidence']
                    
                    # Filter
                    low_conf_indices = buy_signals[buy_signals['confidence'] < min_confidence].index
                    signals.loc[low_conf_indices, 'signal'] = np.nan
                    signals.loc[low_conf_indices, 'pattern'] = np.nan
                    
                    buy_signals = buy_signals[buy_signals['confidence'] >= min_confidence]
                else:
                    buy_signals = buy_signals.copy()
                    buy_signals['confidence'] = 0.5

            # Calculate Metrics
            metrics = None
            equity_curve = None
            drawdown_curve = None
            
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
            
            # Prepare signal markers for visualization
            signal_markers = {}
            buy_signals_dr = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Double Repo')]
            buy_signals_sp = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Single Penetration')]
            buy_signals_rrt = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Railroad Tracks')]
            buy_signals_ftp = signals[(signals['signal'] == 'BUY') & (signals['pattern'] == 'Failure to Penetrate')]
            
            if not buy_signals_dr.empty:
                signal_markers['Double Repo'] = buy_signals_dr
            if not buy_signals_sp.empty:
                signal_markers['Single Penetration'] = buy_signals_sp
            if not buy_signals_rrt.empty:
                signal_markers['Railroad Tracks'] = buy_signals_rrt
            if not buy_signals_ftp.empty:
                signal_markers['Failure to Penetrate'] = buy_signals_ftp
            
            fig = Visualizer.plot_chart(df, indicators, 
                                       equity=equity_curve, 
                                       floating_equity=metrics.get('Floating Equity Curve') if metrics else None,
                                       drawdown=drawdown_curve, 
                                       trades=metrics['Trade Log'] if metrics else None, 
                                       signals=signal_markers if signal_markers else None,
                                       title=f"{symbol} Backtesting Results")
                
            st_echarts(options=fig, height="550px")
            
            # Metrics Display
            if metrics:
                st.markdown("---")
                st.subheader("Strategy Performance")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Trades", metrics['Total Trades'])
                m2.metric("Win Rate", f"{metrics['Win Rate']:.1%}")
                m3.metric("Avg Return", f"{metrics['Avg Return']:.2%}")
                m4.metric("Total Return", f"{metrics['Total Return']:.2%}")
                m5.metric("Ann. Return", f"{metrics['Annualized Return']:.2%}")
                m6.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                
                m7, m8, m9 = st.columns(3)
                m7.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0.0):.2f}")
                m8.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0.0):.2f}")
                m9.metric("Profit Factor", f"{metrics.get('Profit Factor', 0.0):.2f}")
                
                # Monthly Returns Heatmap
                st.markdown("---")
                st.subheader("Monthly Returns Heatmap")
                if 'Monthly Returns' in metrics and not metrics['Monthly Returns'].empty:
                    fig_heatmap = Visualizer.plot_heatmap(metrics['Monthly Returns'])
                    st_echarts(options=fig_heatmap, height="500px")
                
                # Strategy Breakdown
                st.markdown("---")
                st.subheader("Strategy Contribution Breakdown")
                if 'Strategy Breakdown' in metrics and not metrics['Strategy Breakdown'].empty:
                    breakdown_df = metrics['Strategy Breakdown']
                    
                    # Display table
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(breakdown_df.style.format({
                            'Total PnL': '{:.2f}',
                            'Win Rate': '{:.1%}',
                            'Contribution %': '{:.1f}%'
                        }), width='stretch')
                    
                    with col2:
                        # Create bar chart for contribution using ECharts
                        fig_contrib = Visualizer.plot_bar_chart(
                            breakdown_df,
                            x_col='Strategy',
                            y_col='Contribution %',
                            title="Profit Contribution by Strategy"
                        )
                        st_echarts(options=fig_contrib, height="300px")
                else:
                    st.info("No strategy breakdown available.")
                
                st.markdown("---")
                st.subheader("Trade Log")
                st.info("Click on a trade to view detailed visualization.")
                
                event = st.dataframe(
                    metrics['Trade Log'].style.format({
                        'Entry Price': '{:.2f}', 'Stop Loss': '{:.2f}', 'Take Profit': '{:.2f}', 
                        'Exit Price': '{:.2f}', 'PnL Amount': '{:.2f}', 'PnL %': '{:.2%}', 'Confidence': '{:.2%}'
                    }),
                    on_select="rerun",
                    selection_mode="single-row"
                )
                
                if len(event.selection.rows) > 0:
                    selected_row_idx = event.selection.rows[0]
                    trade_row = metrics['Trade Log'].iloc[selected_row_idx]
                    
                    @st.dialog("Trade Details", width="large")
                    def show_trade_details(trade_row, df):
                        st.write(f"**{trade_row['Pattern']}** | {trade_row['Entry Date'].date()} -> {trade_row['Exit Date'].date()}")
                        
                        # Generate Chart
                        chart_option = Visualizer.plot_trade_detail(df, trade_row)
                        if chart_option:
                            st_echarts(options=chart_option, height="500px")
                        else:
                            st.warning("Could not generate chart for this trade.")
                        
                        # Show Metadata Raw (Debug/Info)
                        with st.expander("Raw Signal Metadata"):
                            # Convert timestamps to str for JSON serialization
                            meta_copy = trade_row.get('Metadata', {}).copy() if isinstance(trade_row.get('Metadata'), dict) else {}
                            st.json(meta_copy)
                            
                    show_trade_details(trade_row, df)
            else:
                st.info("No trades executed.")

        # --- Tab 2: Robustness ---
        with tab_robustness:
            st.header("Walk-Forward Analysis 🔬")
            st.write("Test strategy robustness by optimizing on past data and testing on future data.")
            
            with st.expander("⚙️ WFA Settings", expanded=True):
                n_splits = st.number_input("Number of Splits", 2, 10, 5)
            
            if st.button("Run Walk-Forward Analysis"):
                # Prepare signals (using default logic for now, or we could expose strategy params here too)
                # For simplicity, we assume basic pattern signals are pre-calculated
                # We need to pass a grid of parameters to optimize
                
                # Merge signals (basic)
                signals = pd.DataFrame(index=df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
                mask_dr = signals_dr['signal'].notna()
                signals.loc[mask_dr] = signals_dr.loc[mask_dr]
                mask_sp = signals_sp['signal'].notna()
                mask_fill = (signals['signal'].isna()) & (signals_sp['signal'].notna())
                signals.loc[mask_fill] = signals_sp.loc[mask_fill]
                
                with st.spinner("Running Walk-Forward Analysis..."):
                    param_grid = {
                        'holding_period': [5, 10, 15, 20],
                        'stop_loss': [0.02, 0.05],
                        'take_profit': [0.05, 0.10]
                    }
                    
                    optimizer = StrategyOptimizer(df, signals)
                    wfa_results = optimizer.walk_forward_analysis(param_grid, n_splits=n_splits)
                    
                    if not wfa_results.empty:
                        st.write("Walk-Forward Results (Out-of-Sample):")
                        st.dataframe(wfa_results.style.format({
                            'OOS Return': '{:.2%}',
                            'OOS Sharpe': '{:.2f}',
                            'OOS Max DD': '{:.2%}'
                        }))
                        
                        avg_oos_return = wfa_results['OOS Return'].mean()
                        st.metric("Average OOS Return", f"{avg_oos_return:.2%}")
                    else:
                        st.warning("Not enough data for Walk-Forward Analysis.")

        # --- Tab 3: ML Lab ---
        with tab_ml:
            st.header("ML Lab 🧪")
            
            # Display Current Model Info
            st.subheader("📊 Current Model Information")
            clf = SignalClassifier()
            if clf.is_trained:
                metadata = clf.get_metadata()
                if metadata:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Symbol", metadata.get('symbol', 'N/A'))
                        st.metric("Training Samples", metadata.get('total_samples', 'N/A'))
                        
                    with col2:
                        start = metadata.get('start_date', 'N/A')
                        end = metadata.get('end_date', 'N/A')
                        if start != 'N/A' and end != 'N/A':
                            st.metric("Date Range", f"{start[:10]} to {end[:10]}")
                        else:
                            st.metric("Date Range", "N/A")
                        
                        metrics = metadata.get('performance_metrics', {})
                        accuracy = metrics.get('accuracy', 0)
                        st.metric("Accuracy", f"{accuracy:.2%}")
                        
                    with col3:
                        train_params = metadata.get('training_parameters', {})
                        st.metric("Holding Period", f"{train_params.get('holding_period', 'N/A')} bars")
                        precision = metrics.get('precision', 0)
                        st.metric("Precision", f"{precision:.2%}")
                    
                    # Additional details in expander
                    with st.expander("📋 Detailed Training Information"):
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.write("**Training Parameters:**")
                            st.write(f"- Stop Loss: {train_params.get('stop_loss_pct', 0):.1%}")
                            st.write(f"- Take Profit: {train_params.get('take_profit_pct', 0):.1%}")
                            st.write(f"- Holding Period: {train_params.get('holding_period', 'N/A')} bars")
                            
                            st.write("\n**Model Hyperparameters:**")
                            model_params = metadata.get('model_hyperparameters', {})
                            st.write(f"- Estimators: {model_params.get('n_estimators', 'N/A')}")
                            st.write(f"- Max Depth: {model_params.get('max_depth', 'N/A')}")
                            
                        with info_col2:
                            st.write("**Dataset Information:**")
                            st.write(f"- Total Samples: {metadata.get('total_samples', 'N/A')}")
                            st.write(f"- Training Samples: {metadata.get('training_samples', 'N/A')}")
                            st.write(f"- Test Samples: {metadata.get('test_samples', 'N/A')}")
                            st.write(f"- Positive Samples: {metadata.get('positive_samples', 'N/A')}")
                            st.write(f"- Negative Samples: {metadata.get('negative_samples', 'N/A')}")
                            st.write(f"- Feature Count: {metadata.get('feature_count', 'N/A')}")
                            
                            st.write("\n**Training Timestamp:**")
                            timestamp = metadata.get('training_timestamp', 'N/A')
                            if timestamp != 'N/A':
                                st.write(f"{timestamp[:19].replace('T', ' ')}")
                            else:
                                st.write("N/A")
                else:
                    st.info("Model is trained but metadata is not available. This may be an older model.")
            else:
                st.info("No trained model found. Train a model to see information here.")
            
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Model Training")
                if st.button("Train ML Model"):
                    with st.spinner("Training Model..."):
                        train_df = df
                        train_recognizer = PatternRecognizer(train_df)
                        train_signals = train_recognizer.detect_double_repo()
                        train_signals.update(train_recognizer.detect_single_penetration())
                        
                        trainer = ModelTrainer()
                        result = trainer.train(
                            train_df, 
                            train_signals,
                            symbol=symbol,
                            start_date=str(start_date),
                            end_date=str(end_date),
                            holding_period=5,
                            stop_loss_pct=0.02,
                            take_profit_pct=0.05
                        )
                        
                        if result['status'] == 'success':
                            st.success(f"Trained on {result['samples']} signals.")
                            st.write(f"Accuracy: {result['accuracy']:.2f}")
                            st.write(f"Precision: {result['precision']:.2f}")
                            st.rerun()
                        else:
                            st.error(result['message'])
            
            with c2:
                st.subheader("Feature Importance")
                if st.button("Show Feature Importance"):
                    clf_temp = SignalClassifier()
                    if clf_temp.is_trained:
                        importance_df = clf_temp.get_feature_importance()
                        if not importance_df.empty:
                            st.bar_chart(importance_df.set_index('Feature'))
                        else:
                            st.info("No feature importance available.")
                    else:
                        st.warning("Model not trained yet.")

            st.markdown("---")
            st.subheader("Model Management")
            
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                # Download Model
                clf = SignalClassifier()
                if clf.is_trained and os.path.exists(clf.model_path):
                    with open(clf.model_path, "rb") as f:
                        st.download_button(
                            label="Download Trained Model",
                            data=f,
                            file_name="signal_classifier.joblib",
                            mime="application/octet-stream"
                        )
                else:
                    st.info("No trained model found to download.")
                        
            with m_c2:
                # Upload Model
                uploaded_file = st.file_uploader("Upload Pre-trained Model", type=['joblib'])
                if uploaded_file is not None:
                    # Save uploaded file
                    with open(clf.model_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Model uploaded successfully! Reloading...")
                    # Force reload
                    clf = SignalClassifier()
                    st.experimental_rerun()

    else:
        st.info("Please load data to proceed.")

elif analysis_mode == "Portfolio":
    st.title("Portfolio Scanner 🔍")
    st.write("Scan multiple symbols for DiNapoli patterns.")
    
    # --- Sidebar: Scanner Params ---
    st.sidebar.header("Scanner Settings")
    
    default_symbols = "688981.SH, 601398.SH, 601111.SH, 600036.SH, 002714.SZ, 300088.SZ, 601688.SH"
    symbols_input = st.sidebar.text_area("Symbols (comma separated)", default_symbols, height=150)
    lookback = st.sidebar.number_input("Lookback Days", min_value=30, max_value=365, value=200)
    scan_window = st.sidebar.number_input("Scan Window (Bars)", min_value=1, max_value=20, value=5)
    
    if st.button("Scan Market"):
        symbols_list = [s.strip() for s in symbols_input.split(',') if s.strip()]
        
        if symbols_list:
            scanner = MarketScanner()
            
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
