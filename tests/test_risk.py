import unittest
import pandas as pd
import numpy as np
from src.risk.manager import RiskManager
from src.utils.performance import PerformanceAnalyzer

class TestRisk(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame(index=dates)
        self.df['close'] = 100.0
        self.df['high'] = 102.0
        self.df['low'] = 98.0
        self.df['open'] = 100.0
        self.df['volume'] = 1000
        
        # Create dummy signals
        self.signals = pd.DataFrame(index=dates, columns=['signal', 'pattern'])
        self.signals.iloc[20] = ['BUY', 'Test']
        
    def test_atr_stop_loss(self):
        entry = 100.0
        atr = 2.0
        mult = 2.0
        sl = RiskManager.calculate_atr_stop_loss(entry, atr, mult, 'BUY')
        self.assertEqual(sl, 96.0) # 100 - (2 * 2)
        
    def test_position_sizing(self):
        equity = 100000.0
        risk_pct = 0.01 # 1% = 1000 risk
        entry = 100.0
        sl = 98.0 # Risk per share = 2
        
        shares = RiskManager.calculate_position_size(equity, risk_pct, entry, sl)
        # Expected: 1000 / 2 = 500 shares
        self.assertEqual(shares, 500)
        
    def test_performance_with_dynamic_risk(self):
        # Mock ATR in df for the analyzer to use?
        # The analyzer calculates ATR internally using Indicators.atr
        # We need high/low/close to vary to get valid ATR, or just trust it returns something.
        # My synthetic data has constant range, so ATR should be constant (approx 4.0? High-Low=4)
        
        analyzer = PerformanceAnalyzer(self.df, self.signals)
        metrics = analyzer.calculate_metrics(
            initial_capital=100000.0,
            use_dynamic_sizing=True,
            risk_per_trade_pct=0.01,
            atr_multiplier=2.0
        )
        
        # Check if quantity is not 100
        trade_log = metrics['Trade Log']
        if not trade_log.empty:
            qty = trade_log.iloc[0]['Quantity']
            self.assertNotEqual(qty, 100)
            # ATR approx 4.0 (High 102 - Low 98)
            # SL distance = 4 * 2 = 8
            # Risk = 1000
            # Qty = 1000 / 8 = 125
            # Let's check if it's roughly 125
            self.assertTrue(100 <= qty <= 150)

if __name__ == '__main__':
    unittest.main()
