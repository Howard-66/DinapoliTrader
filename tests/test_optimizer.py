import unittest
import pandas as pd
import numpy as np
from src.optimization.optimizer import StrategyOptimizer

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame(index=dates)
        self.df['close'] = 100.0
        self.df['open'] = 100.0
        self.df['high'] = 101.0
        self.df['low'] = 99.0
        self.df['volume'] = 1000
        
        # Create dummy signals
        self.signals = pd.DataFrame(index=dates, columns=['signal', 'pattern'])
        # Buy at 10, Sell at 20 (Profit)
        self.signals.iloc[10] = ['BUY', 'Test']
        self.df['close'].iloc[10] = 100
        self.df['close'].iloc[20] = 110 # 10% gain
        
        # Buy at 30, Sell at 40 (Loss)
        self.signals.iloc[30] = ['BUY', 'Test']
        self.df['close'].iloc[30] = 100
        self.df['close'].iloc[40] = 90 # 10% loss
        
    def test_grid_search(self):
        optimizer = StrategyOptimizer(self.df, self.signals)
        
        # Define grid
        # We expect tighter stop loss to catch the loss early?
        # Or wider take profit to catch the gain?
        
        param_grid = {
            'holding_period': [5, 10],
            'stop_loss': [0.05, 0.15], # 5%, 15%
            'take_profit': [0.05, 0.15] # 5%, 15%
        }
        
        results = optimizer.grid_search(param_grid)
        
        # Check columns
        self.assertIn('Total Return', results.columns)
        self.assertIn('holding_period', results.columns)
        
        # Check if results are sorted
        self.assertTrue(results['Total Return'].is_monotonic_decreasing)
        
        # Check number of combinations: 2 * 2 * 2 = 8
        self.assertEqual(len(results), 8)

if __name__ == '__main__':
    unittest.main()
