import unittest
import unittest.mock
import pandas as pd
from src.data.feed import DataFeed

class TestDataFeed(unittest.TestCase):
    def setUp(self):
        self.feed = DataFeed()

    @unittest.mock.patch('yfinance.download')
    def test_fetch_yfinance_daily(self, mock_download):
        # Mock response
        mock_df = pd.DataFrame({
            'Open': [150.0] * 5,
            'High': [155.0] * 5,
            'Low': [149.0] * 5,
            'Close': [152.0] * 5,
            'Volume': [1000000] * 5
        }, index=pd.date_range('2023-01-01', periods=5))
        mock_download.return_value = mock_df

        df = self.feed.fetch_data('AAPL', '2023-01-01', '2023-01-10', interval='1d')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertTrue('close' in df.columns)
        self.assertEqual(len(df), 5)

    def test_fetch_invalid_symbol(self):
        df = self.feed.fetch_data('INVALID_SYMBOL_XYZ', '2023-01-01', '2023-01-10')
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
