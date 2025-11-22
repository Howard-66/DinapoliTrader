import unittest
import pandas as pd
import numpy as np
from src.indicators.basics import Indicators

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.data = pd.Series(np.random.randn(100) + 100, index=pd.date_range('2023-01-01', periods=100))

    def test_sma(self):
        sma = Indicators.sma(self.data, 10)
        self.assertEqual(len(sma), 100)
        self.assertTrue(np.isnan(sma[0])) # First 9 should be NaN

    def test_dma(self):
        # 3x3 DMA: 3 period SMA shifted 3 periods forward
        dma = Indicators.displaced_ma(self.data, 3, 3)
        self.assertEqual(len(dma), 100)
        # First 2 are NaN due to SMA(3), then shifted 3, so first 5 might be NaN?
        # SMA(3) valid at index 2. Shift(3) moves it to index 5.
        # So indices 0,1,2,3,4 should be NaN.
        self.assertTrue(np.isnan(dma[4]))
        
    def test_rsi(self):
        rsi = Indicators.rsi(self.data, 14)
        self.assertEqual(len(rsi), 100)

if __name__ == '__main__':
    unittest.main()
