import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.visualization import Visualizer

def test_visualization():
    print("Testing visualization...")
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'open': 100, 'high': 105, 'low': 95, 'close': 100, 'volume': 1000
    }, index=dates)
    
    # Dummy trade row with RRT metadata
    trade_row = pd.Series({
        'Entry Date': dates[50],
        'Exit Date': dates[60],
        'Entry Price': 100,
        'Exit Price': 110,
        'PnL Amount': 1000,
        'PnL %': 0.10,
        'Pattern': 'Double Repo',
        'Metadata': {
            'A': {'date': dates[40], 'price': 105},
            'B': {'date': dates[45], 'price': 95},
            'C': {'date': dates[50], 'price': 100},
            'rrt_detected': True,
            'ftp_detected': False
        }
    })
    
    try:
        option = Visualizer.plot_trade_detail(df, trade_row)
        # Check if RRT series is added
        rrt_series = [s for s in option['series'] if s['name'] == 'RRT Confirmation']
        if rrt_series:
            print("✅ RRT Confirmation series found in chart option.")
        else:
            print("❌ RRT Confirmation series NOT found.")
            
    except Exception as e:
        print(f"❌ Error in plot_trade_detail: {e}")
        raise e

    print("Visualization test complete.")

if __name__ == "__main__":
    test_visualization()
