import sys
import os
import pandas as pd
import backtrader as bt
import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.strategy import DiNapoliStrategyWithSignals

def run_verification():
    # 1. Create Synthetic Data (5 days)
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'open': [100, 102, 105, 103, 106],
        'high': [105, 106, 108, 105, 110],
        'low': [95, 98, 100, 98, 102],
        'close': [101, 104, 102, 104, 108],
        'volume': [1000, 1000, 1000, 1000, 1000]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'datetime'

    # 2. Create Signal on Day 2 (2023-01-02)
    # Signal generated at Close of Day 2.
    # Should enter at Open of Day 3 (2023-01-03).
    signals = pd.DataFrame(index=df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp'])
    signals.loc[dates[1], 'signal'] = 'BUY' # Signal on Day 2
    signals.loc[dates[1], 'pattern'] = 'Test Pattern'
    signals.loc[dates[1], 'pattern_sl'] = 90.0
    signals.loc[dates[1], 'pattern_tp'] = 120.0

    # 3. Setup Cerebro
    cerebro = bt.Cerebro()
    
    # Add Data
    bt_df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data_feed = bt.feeds.PandasData(dataname=bt_df)
    cerebro.adddata(data_feed)

    # Add Strategy
    cerebro.addstrategy(DiNapoliStrategyWithSignals, 
                        signals=signals,
                        stop_loss_pct=0.02,
                        take_profit_pct=0.05,
                        initial_capital=100000.0)

    # Set Broker (Slippage + Commission)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_slippage_perc(perc=0.0005) # 0.05%

    # Run
    print("Running Backtest...")
    strategies = cerebro.run()
    strat = strategies[0]

    # Verify
    print("\n--- Verification Results ---")
    
    # Check Trades
    # We expect 1 trade.
    # Entry Date: 2023-01-03
    # Entry Price: Open of Day 3 (105) + Slippage
    # Slippage = 105 * 0.0005 = 0.0525 -> Price = 105.0525
    
    expected_entry_date = dates[2].date() # 2023-01-03
    expected_raw_price = 105.0
    slippage = 0.0005
    expected_price_with_slippage = expected_raw_price * (1 + slippage) # For Buy, Price increases? 
    # Backtrader slippage logic: 
    # Buy: Price = Execution Price * (1 + slippage) ? Or Price + Slippage?
    # set_slippage_perc uses percentage.
    # Usually Buy executes at Ask (higher).
    
    # Let's check the logs or analyzer.
    # Since we don't have easy access to strat.orders history without recording it,
    # we rely on the print logs from the strategy which we can capture or just read.
    
    # But for automated check, let's look at the position or broker.
    # Since the backtest finished (Day 5), the position might still be open or closed (TP/SL).
    # Day 3 Open: 105. SL: 90. TP: 120.
    # Day 3 Low: 100. High: 108. Close: 102. -> No Hit.
    # Day 4 Open: 103. Low: 98. High: 105. Close: 104. -> No Hit.
    # Day 5 Open: 106. Low: 102. High: 110. Close: 108. -> No Hit.
    # Position should be open.
    
    pos_size = strat.position.size
    print(f"Final Position Size: {pos_size}")
    
    if pos_size > 0:
        print("PASS: Position is open.")
        entry_price = strat.position.price
        print(f"Entry Price: {entry_price:.4f}")
        print(f"Expected Price: {expected_price_with_slippage:.4f}")
        
        if abs(entry_price - expected_price_with_slippage) < 0.0001:
            print("PASS: Entry Price matches expected slippage.")
        else:
            print("FAIL: Entry Price mismatch.")
            
    else:
        print("FAIL: Position is closed (unexpected).")

if __name__ == "__main__":
    run_verification()
