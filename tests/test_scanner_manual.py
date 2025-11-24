import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.scanner import MarketScanner

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_scanner():
    print("Initializing Scanner...")
    scanner = MarketScanner()
    
    # Use some symbols that likely have data or use synthetic if offline
    # Tushare might need token, but feed handles synthetic fallback if fails
    symbols = ['AAPL', 'MSFT', 'BTC-USD'] 
    
    print(f"Scanning symbols: {symbols}")
    results = scanner.scan(symbols, lookback_days=100)
    
    print("\nScan Results:")
    if not results.empty:
        print(results)
    else:
        print("No active signals found (or data fetch failed).")
        
    # Check structure
    expected_cols = ['Symbol', 'Date', 'Signal', 'Pattern', 'Close', 'SL', 'TP']
    if all(col in results.columns for col in expected_cols):
        print("\nPASS: DataFrame structure is correct.")
    else:
        print(f"\nFAIL: DataFrame structure mismatch. Got: {results.columns}")

if __name__ == "__main__":
    test_scanner()
