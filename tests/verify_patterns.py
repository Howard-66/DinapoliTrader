import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.patterns import PatternRecognizer
from src.indicators.basics import Indicators

def create_synthetic_data(length=100):
    """Create synthetic OHLC data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close'])
    
    # Base price
    price = 100.0
    for i in range(length):
        change = np.random.normal(0, 1)
        price += change
        df.iloc[i] = [price, price + abs(change), price - abs(change), price]
        
    # Ensure numeric types
    df = df.astype(float)
    return df

def test_rrt():
    print("\nTesting Railroad Tracks (RRT)...")
    # Create RRT pattern: Bullish -> Bearish (Top Reversal)
    df = create_synthetic_data(20)
    
    # Candle 1: Bullish, large body
    df.iloc[10] = [100, 105, 99, 104] 
    # Candle 2: Bearish, large body, similar size
    df.iloc[11] = [104, 105, 99, 100]
    
    pr = PatternRecognizer(df)
    # Note: Current implementation returns DataFrame, future will return Series
    try:
        if hasattr(pr, 'check_railroad_tracks'):
            result = pr.check_railroad_tracks()
            print(f"New API (check_railroad_tracks) detected. Result at index 11: {result.iloc[11]}")
        else:
            result = pr.detect_railroad_tracks()
            print(f"Old API (detect_railroad_tracks) detected. Signal at index 11: {result['signal'].iloc[11]}")
    except Exception as e:
        print(f"Error testing RRT: {e}")

def test_ftp():
    print("\nTesting Failure to Penetrate (FTP)...")
    df = create_synthetic_data(20)
    
    # Setup: Price below 3x3 DMA, then penetrates and closes back
    # Force DMA to be around 100
    df['close'] = 100.0
    # Ensure previous close is below DMA (99 < 100)
    df.iloc[14] = [99, 100, 98, 99]
    
    # At t=15, DMA 3x3 is approx 100.
    # Bearish FTP: High > DMA, Close < DMA
    df.iloc[15] = [99, 102, 98, 99] # High 102 > 100, Close 99 < 100
    
    pr = PatternRecognizer(df)
    try:
        if hasattr(pr, 'check_failure_to_penetrate'):
            result = pr.check_failure_to_penetrate()
            print(f"New API (check_failure_to_penetrate) detected. Result at index 15: {result.iloc[15]}")
        else:
            result = pr.detect_failure_to_penetrate()
            print(f"Old API (detect_failure_to_penetrate) detected. Signal at index 15: {result['signal'].iloc[15]}")
    except Exception as e:
        print(f"Error testing FTP: {e}")

def test_single_penetration():
    print("\nTesting Single Penetration...")
    df = create_synthetic_data(50)
    
    # Create Thrust: Rising trend to ensure Close > DMA
    # DMA lags by 3, so if price rises, Close > DMA
    for i in range(10, 20):
        price = 100.0 + (i - 10) * 2 # 100, 102, 104...
        df.iloc[i]['close'] = price
        df.iloc[i]['low'] = price - 1
        df.iloc[i]['high'] = price + 1
        
    # Penetration at t=20
    # SMA(3) at t=17.
    # C[17]=114, C[16]=112, C[15]=110. Avg = 112.
    # So DMA[20] = 112.
    
    # Touch Entry: Low <= DMA
    df.iloc[20]['low'] = 111.0 # Touches DMA (112)
    df.iloc[20]['close'] = 115.0 
    
    pr = PatternRecognizer(df)
    result = pr.detect_single_penetration()
    print(f"Signal at index 20: {result['signal'].iloc[20]}")
    if not pd.isna(result['metadata'].iloc[20]):
        print(f"Metadata: {result['metadata'].iloc[20]}")

if __name__ == "__main__":
    test_rrt()
    test_ftp()
    test_single_penetration()
