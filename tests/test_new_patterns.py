import unittest
import unittest.mock
import pandas as pd
import numpy as np
from src.strategies.patterns import PatternRecognizer

class TestNewPatterns(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=50)
        self.df = pd.DataFrame(index=dates)
        self.df['close'] = 100.0
        self.df['open'] = 100.0
        self.df['high'] = 100.0
        self.df['low'] = 100.0
        
    @unittest.mock.patch('src.indicators.basics.Indicators.displaced_ma')
    def test_detect_railroad_tracks(self, mock_dma):
        # Mock DMAs (not used for RRT but required for init)
        mock_dma.return_value = pd.Series([100] * 50, index=self.df.index)
        
        recognizer = PatternRecognizer(self.df)
        
        # Construct Bearish RRT (Top Reversal)
        # Candle 1: Bullish, Large Body
        # Candle 2: Bearish, Large Body, Similar Size
        
        idx = 10
        # Candle 1 (Bullish)
        self.df.iloc[idx-1, self.df.columns.get_loc('open')] = 100
        self.df.iloc[idx-1, self.df.columns.get_loc('close')] = 110
        self.df.iloc[idx-1, self.df.columns.get_loc('high')] = 111
        self.df.iloc[idx-1, self.df.columns.get_loc('low')] = 99
        
        # Candle 2 (Bearish)
        self.df.iloc[idx, self.df.columns.get_loc('open')] = 110
        self.df.iloc[idx, self.df.columns.get_loc('close')] = 100
        self.df.iloc[idx, self.df.columns.get_loc('high')] = 111
        self.df.iloc[idx, self.df.columns.get_loc('low')] = 99
        
        # Re-initialize to pick up data changes
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_railroad_tracks()
        
        self.assertEqual(signals['signal'].iloc[idx], 'SELL')
        self.assertEqual(signals['pattern'].iloc[idx], 'Railroad Tracks')
        
        # Construct Bullish RRT (Bottom Reversal)
        idx = 20
        # Candle 1 (Bearish)
        self.df.iloc[idx-1, self.df.columns.get_loc('open')] = 100
        self.df.iloc[idx-1, self.df.columns.get_loc('close')] = 90
        self.df.iloc[idx-1, self.df.columns.get_loc('high')] = 101
        self.df.iloc[idx-1, self.df.columns.get_loc('low')] = 89
        
        # Candle 2 (Bullish)
        self.df.iloc[idx, self.df.columns.get_loc('open')] = 90
        self.df.iloc[idx, self.df.columns.get_loc('close')] = 100
        self.df.iloc[idx, self.df.columns.get_loc('high')] = 101
        self.df.iloc[idx, self.df.columns.get_loc('low')] = 89
        
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_railroad_tracks()
        
        self.assertEqual(signals['signal'].iloc[idx], 'BUY')
        self.assertEqual(signals['pattern'].iloc[idx], 'Railroad Tracks')

    @unittest.mock.patch('src.indicators.basics.Indicators.displaced_ma')
    def test_detect_failure_to_penetrate(self, mock_dma):
        # Mock DMA 3x3 at 100
        # Use return_value so it works for multiple calls/inits
        mock_dma.return_value = pd.Series([100] * 50, index=self.df.index)
        
        recognizer = PatternRecognizer(self.df)
        
        # Bullish FTP (Support Hold)
        # Trend is Up (Previous Close > DMA)
        # Current Low < DMA, Current Close > DMA
        
        idx = 10
        self.df.iloc[idx-1, self.df.columns.get_loc('close')] = 102 # Above 100
        
        self.df.iloc[idx, self.df.columns.get_loc('open')] = 102
        self.df.iloc[idx, self.df.columns.get_loc('high')] = 103
        self.df.iloc[idx, self.df.columns.get_loc('low')] = 98 # Dip below 100
        self.df.iloc[idx, self.df.columns.get_loc('close')] = 101 # Close above 100
        
        # Re-init
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_failure_to_penetrate()
        
        self.assertEqual(signals['signal'].iloc[idx], 'BUY')
        self.assertEqual(signals['pattern'].iloc[idx], 'Failure to Penetrate')
        
        # Bearish FTP (Resistance Hold)
        # Trend is Down (Previous Close < DMA)
        # Current High > DMA, Current Close < DMA
        
        idx = 20
        self.df.iloc[idx-1, self.df.columns.get_loc('close')] = 98 # Below 100
        
        self.df.iloc[idx, self.df.columns.get_loc('open')] = 98
        self.df.iloc[idx, self.df.columns.get_loc('high')] = 102 # Spike above 100
        self.df.iloc[idx, self.df.columns.get_loc('low')] = 97
        self.df.iloc[idx, self.df.columns.get_loc('close')] = 99 # Close below 100
        
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_failure_to_penetrate()
        
        self.assertEqual(signals['signal'].iloc[idx], 'SELL')
        self.assertEqual(signals['pattern'].iloc[idx], 'Failure to Penetrate')

if __name__ == '__main__':
    unittest.main()
