import unittest
import pandas as pd
import numpy as np
from src.utils.performance import PerformanceAnalyzer

class TestPerformanceAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create synthetic price data
        dates = pd.date_range('2023-01-01', periods=20)
        self.df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100, 101, 102, 103, 104, 105, 100, 99, 98, 97, 96, 95, 100, 100, 100, 100, 100, 100, 100, 100],
            'volume': [1000] * 20
        }, index=dates)
        
        # Create synthetic signals
        self.signals = pd.DataFrame(index=dates, columns=['signal', 'pattern'])
        self.signals['signal'] = None
        
        # Signal 1: Buy at index 0 (Price 100). 
        # Next 5 bars: 101, 102, 103, 104, 105.
        # If holding 5 bars, exit at 105. Return = 5%.
        self.signals.iloc[0] = ['BUY', 'Pattern']
        
        # Signal 2: Buy at index 6 (Price 100).
        # Next 5 bars: 99, 98, 97, 96, 95.
        # If SL is 2% (98), it hits SL at index 8 (Low 95 is < 98? No, low is constant 95 in my setup... wait)
        # My setup has constant high/low 105/95.
        # If SL is 2% (98). Low is 95. So it hits SL immediately next bar?
        # Yes, Low 95 <= 98. So exit at 98. Return = -2%.
        self.signals.iloc[6] = ['BUY', 'Pattern']

    def test_calculate_metrics(self):
        analyzer = PerformanceAnalyzer(self.df, self.signals)
        metrics = analyzer.calculate_metrics(holding_period=5, stop_loss_pct=0.02, take_profit_pct=0.10)
        
        # Trade 1: Entry 100. High 105. Low 95.
        # Wait, my synthetic data has constant High 105 and Low 95.
        # If SL is 2% (98), Low 95 triggers it immediately on next bar.
        # If TP is 5% (105), High 105 triggers it immediately on next bar.
        # Which one triggers first? The code checks SL then TP.
        # So both trades should hit SL immediately?
        
        # Let's adjust data to be more specific for testing logic
        # Trade 1: Success
        # Index 0: Close 100.
        # Index 1: Low 99, High 102, Close 101. (No SL/TP)
        # Index 2: Low 99, High 106, Close 105. (TP hits at 105 if TP=5%)
        
        dates = pd.date_range('2023-01-01', periods=10)
        df = pd.DataFrame(index=dates)
        df['close'] = [100, 101, 105, 100, 100, 100, 100, 90, 100, 100]
        df['high'] =  [100, 102, 106, 100, 100, 100, 100, 90, 100, 100]
        df['low'] =   [100, 99, 101, 100, 100, 100, 100, 80, 100, 100]
        
        signals = pd.DataFrame(index=dates, columns=['signal', 'pattern'])
        signals.iloc[0] = ['BUY', 'P1'] # Trade 1
        signals.iloc[6] = ['BUY', 'P2'] # Trade 2
        
        analyzer = PerformanceAnalyzer(df, signals)
        metrics = analyzer.calculate_metrics(holding_period=5, stop_loss_pct=0.02, take_profit_pct=0.05)
        
        # Trade 1 (Idx 0): Entry 100. TP 105. SL 98.
        # Idx 1: Low 99 (>98), High 102 (<105). Hold.
        # Idx 2: Low 101 (>98), High 106 (>105). TP Hit! Exit 105. Return 5%.
        
        # Trade 2 (Idx 6): Entry 100. TP 105. SL 98.
        # Idx 7: Low 80 (<98). SL Hit! Exit 98. Return -2%.
        
        # Total Trades: 2
        # Win Rate: 50%
        # Avg Return: (0.05 - 0.02) / 2 = 1.5%
        
        self.assertEqual(metrics['Total Trades'], 2)
        self.assertAlmostEqual(metrics['Win Rate'], 0.5)
        self.assertAlmostEqual(metrics['Avg Return'], 0.015)
        self.assertIn('Annualized Return', metrics)
        # 2 trades over 10 days. Total Return approx 3%.
        # Annualized: (1.03)^(365/10) - 1 ... huge number.
        # Let's just check it exists and is positive.
        self.assertGreater(metrics['Annualized Return'], 0.0)
        
        # Verify Drawdown is negative
        self.assertTrue((metrics['Drawdown Curve'] <= 0).all())
        self.assertLess(metrics['Max Drawdown'], 0)
        
        # Verify Trade Log
        self.assertIn('Trade Log', metrics)
        trade_log = metrics['Trade Log']
        self.assertEqual(len(trade_log), 2)
        self.assertIn('Entry Price', trade_log.columns)
        self.assertIn('PnL Amount', trade_log.columns)
        # Check first trade PnL Amount (Entry 100, Exit 105, Qty 100 -> 500 profit)
        self.assertAlmostEqual(trade_log.iloc[0]['PnL Amount'], 500.0)

if __name__ == '__main__':
    unittest.main()
