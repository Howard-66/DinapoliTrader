import streamlit as st
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
holding_period = st.sidebar.number_input("Holding Period (Bars)", min_value=1, value=5)
stop_loss = st.sidebar.number_input("Stop Loss (%)", min_value=0.1, value=2.0, step=0.1) / 100
take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.1, value=5.0, step=0.1) / 100

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
    
    # Detect Patterns
    recognizer = PatternRecognizer(df)
    signals = recognizer.detect_double_repo()
    
    # Add markers for signals (moved up to define buy_signals earlier)
    buy_signals = signals[signals['signal'] == 'BUY']
    # Performance Metrics Calculation
    equity_curve = None
    drawdown_curve = None
    metrics = None
    
    if not buy_signals.empty:
        perf_analyzer = PerformanceAnalyzer(df, signals)
        metrics = perf_analyzer.calculate_metrics(holding_period, stop_loss, take_profit)
        equity_curve = metrics['Equity Curve']
        drawdown_curve = metrics['Drawdown Curve']

    # Visualization
    indicators = {
        'DMA 3x3': dma_3x3,
        'DMA 7x5': dma_7x5,
        'DMA 25x5': dma_25x5
    }
    
    fig = Visualizer.plot_chart(df, indicators, equity=equity_curve, drawdown=drawdown_curve, title=f"{symbol} DiNapoli Analysis")
    
    # Add markers for signals
    if not buy_signals.empty:
        fig.add_scatter(x=buy_signals.index, y=df.loc[buy_signals.index, 'low']*0.99, mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal')
        
    st.plotly_chart(fig, use_container_width=True)
    
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
            'PnL %': '{:.2%}'
        }))
    elif not buy_signals.empty:
        st.write("Raw Signals:")
        st.dataframe(buy_signals)
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
