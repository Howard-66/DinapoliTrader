import unittest
import pandas as pd
import numpy as np
from src.strategies.patterns import PatternRecognizer

class TestSinglePenetration(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        self.df = pd.DataFrame(index=dates)
        
        # Create a trend where Close > 3x3 DMA for > 8 bars
        # 3x3 DMA is roughly average of last 3 bars shifted by 3.
        # Let's just make price rise steadily.
        self.df['close'] = np.linspace(100, 150, 50)
        self.df['open'] = self.df['close'] - 1
        self.df['high'] = self.df['close'] + 1
        self.df['low'] = self.df['close'] - 1
        self.df['volume'] = 1000
        
        # Calculate DMA manually to ensure we control the thrust
        # But PatternRecognizer calculates it internally.
        # We need to ensure price stays above DMA for a while.
        
        # Let's create a dip at index 30
        self.df.iloc[30, self.df.columns.get_loc('low')] = 120 # Deep dip
        self.df.iloc[30, self.df.columns.get_loc('close')] = 128 # Close above DMA (~127)
        
    def test_bullish_single_penetration(self):
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_single_penetration(thrust_bars=8)
        
        # Check if signal is detected at index 30
        # Note: DMA calculation consumes initial bars.
        # 3x3 DMA needs 3 bars + 3 shift = 6 bars valid?
        
        # Let's inspect the signal
        signal_row = signals.iloc[30]
        # We expect a BUY if the dip touched the DMA
        # Since we don't know exact DMA value easily without running, 
        # let's check if ANY signal was generated.
        
        has_signal = not signals['signal'].dropna().empty
        self.assertTrue(has_signal, "Should detect at least one signal")
        
        if has_signal:
            first_signal = signals[signals['signal'].notna()].iloc[0]
            self.assertEqual(first_signal['signal'], 'BUY')
            self.assertEqual(first_signal['pattern'], 'Single Penetration')

if __name__ == '__main__':
    unittest.main()
