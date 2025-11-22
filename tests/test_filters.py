import unittest
import pandas as pd
import numpy as np
from src.strategies.patterns import PatternRecognizer

class TestFilters(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame(index=dates)
        self.df['close'] = 100.0
        self.df['open'] = 100.0
        self.df['high'] = 101.0
        self.df['low'] = 99.0
        self.df['volume'] = 1000
        
        # Create a dummy signal at index 50
        self.signals = pd.DataFrame(index=dates, columns=['signal', 'pattern'])
        self.signals.iloc[50] = ['BUY', 'Test Pattern']
        
        # Create a Trend MA
        self.trend_ma = pd.Series(100.0, index=dates)
        
    def test_trend_filter_buy(self):
        recognizer = PatternRecognizer(self.df)
        
        # Case 1: Close > MA (Bullish) -> Signal should be kept
        self.df['close'].iloc[50] = 105.0
        self.trend_ma.iloc[50] = 100.0
        
        filtered = recognizer.apply_trend_filter(self.signals, self.trend_ma)
        self.assertEqual(filtered.iloc[50]['signal'], 'BUY')
        
        # Case 2: Close < MA (Bearish) -> Signal should be removed
        self.df['close'].iloc[50] = 95.0
        self.trend_ma.iloc[50] = 100.0
        
        filtered = recognizer.apply_trend_filter(self.signals, self.trend_ma)
        self.assertTrue(pd.isna(filtered.iloc[50]['signal']))

if __name__ == '__main__':
    unittest.main()
