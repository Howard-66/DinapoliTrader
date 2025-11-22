import unittest
import unittest.mock
import pandas as pd
from src.backtest.engine import BacktestEngine

class TestBacktestEngine(unittest.TestCase):
    
    @unittest.mock.patch('src.data.feed.DataFeed.fetch_data')
    def test_run_backtest(self, mock_fetch):
        # Mock data
        dates = pd.date_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000.0] * 100
        }, index=dates)
        mock_fetch.return_value = df
        
        engine = BacktestEngine('AAPL', '2023-01-01', '2023-04-01')
        # Just ensure it runs without error
        try:
            engine.run()
        except Exception as e:
            self.fail(f"BacktestEngine.run() raised {e} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
