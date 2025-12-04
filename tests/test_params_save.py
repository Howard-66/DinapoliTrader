import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.backtest_results import BacktestResultsManager

def test_parameters_serialization():
    print("Testing Parameters Dictionary Serialization...")
    
    manager = BacktestResultsManager()
    
    # Simulate the parameters dictionary from app.py with numpy booleans
    parameters = {
        'selected_strategies': ['Double Repo', 'Single Penetration'],
        'enable_trend_filter': np.bool_(True),  # This is the culprit
        'enable_mtf_filter': np.bool_(False),
        'min_confidence': 0.5,
        'require_rrt': np.bool_(False),
        'require_ftp': np.bool_(True)
    }
    
    # Simulate minimal metrics
    metrics = {
        'Total Trades': 10,
        'Win Rate': 0.6
    }
    
    # Try to save
    try:
        filename = manager.save_result(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-12-31',
            parameters=parameters,
            metrics=metrics,
            description='Test save with numpy booleans'
        )
        print(f"✅ Save successful: {filename}")
        
        # Try to load it back
        loaded = manager.load_result(filename)
        print(f"✅ Load successful")
        print(f"Loaded parameters: {loaded['parameters']}")
        
        # Clean up
        manager.delete_result(filename)
        print("✅ Cleanup successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

if __name__ == "__main__":
    test_parameters_serialization()
