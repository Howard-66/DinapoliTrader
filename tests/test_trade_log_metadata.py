import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.backtest_results import BacktestResultsManager

def test_trade_log_with_metadata():
    print("Testing Trade Log with Metadata containing numpy bools...")
    
    manager = BacktestResultsManager()
    
    # Simulate Trade Log DataFrame with Metadata column containing numpy bools
    trade_log = pd.DataFrame([
        {
            'Signal Date': pd.Timestamp('2023-01-01'),
            'Pattern': 'Double Repo',
            'Entry Date': pd.Timestamp('2023-01-02'),
            'Entry Price': 100.0,
            'Stop Loss': 95.0,
            'Take Profit': 110.0,
            'Exit Date': pd.Timestamp('2023-01-10'),
            'Exit Price': 108.0,
            'Quantity': 100,
            'PnL Amount': 800.0,
            'PnL %': 0.08,
            'Confidence': 0.75,
            'Metadata': {
                'rrt_detected': np.bool_(True),  # This is the problem!
                'ftp_detected': np.bool_(False),
                'A': {'date': pd.Timestamp('2023-01-01'), 'price': 95.0}
            }
        }
    ])
    
    # Simulate minimal metrics
    metrics = {
        'Total Trades': 1,
        'Win Rate': 1.0,
        'Trade Log': trade_log
    }
    
    parameters = {
        'selected_strategies': ['Double Repo'],
        'enable_trend_filter': True
    }
    
    # Try to save
    try:
        filename = manager.save_result(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            parameters=parameters,
            metrics=metrics,
            description='Test with numpy bools in metadata'
        )
        print(f"✅ Save successful: {filename}")
        
        # Try to load it back
        loaded = manager.load_result(filename)
        print(f"✅ Load successful")
        print(f"Loaded Trade Log Metadata: {loaded['metrics']['Trade Log'].iloc[0]['Metadata']}")
        
        # Clean up
        manager.delete_result(filename)
        print("✅ Cleanup successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_trade_log_with_metadata()
