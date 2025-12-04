import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.patterns import PatternRecognizer

def test_fib_target():
    print("Testing Fibonacci TP Targets...")
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'open': 100, 'high': 105, 'low': 95, 'close': 100, 'volume': 1000
    }, index=dates)
    
    recognizer = PatternRecognizer(df)
    
    # Mock DMA 25x5 to pass the distance filter
    recognizer.dma_25x5 = pd.Series(200.0, index=df.index)
    
    # Verify the method accepts the parameter
    try:
        signals = recognizer.detect_double_repo(fib_target='XOP')
        print("✅ detect_double_repo called successfully with fib_target='XOP'.")
    except Exception as e:
        print(f"❌ Error calling detect_double_repo: {e}")
        raise e

    print("Fib target test complete.")

if __name__ == "__main__":
    test_fib_target()
