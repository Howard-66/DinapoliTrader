import yfinance as yf
import tushare as ts
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Optional, List, Union
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFeed:
    """
    Handles data ingestion from various sources.
    Currently supports: yfinance
    """

    def __init__(self, source: str = 'tushare'):
        self.source = source
        if self.source == 'tushare':
            token = os.getenv('TUSHARE_TOKEN')
            if not token:
                logger.warning("TUSHARE_TOKEN not found in environment variables. Tushare may fail.")
            else:
                ts.set_token(token)
                self.pro = ts.pro_api()

    def fetch_data(self, 
                   symbol: str, 
                   start_date: str, 
                   end_date: Optional[str] = None, 
                   interval: str = '1d',
                   adj: str = 'qfq') -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol (str): Ticker symbol (e.g., 'AAPL', 'BTC-USD').
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None (now).
            interval (str): Data interval (e.g., '1d', '1h', '15m').
            adj (str): Adjustment type. 'qfq' (default), 'hfq', or None.
            
        Returns:
            pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume] and Datetime index.
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} ({interval}, adj={adj})")
        
        if self.source == 'yfinance':
            return self._fetch_yfinance(symbol, start_date, end_date, interval)
        elif self.source == 'tushare':
            return self._fetch_tushare(symbol, start_date, end_date, interval, adj)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _fetch_yfinance(self, symbol: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}. Using synthetic data.")
                return self.generate_synthetic_data(start, end)

            # Standardize columns
            # yfinance returns MultiIndex columns if multiple tickers, but we assume single ticker for now
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(symbol, axis=1, level=1)
            
            # Ensure standard column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in required_cols if c in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}. Using synthetic data.")
            return self.generate_synthetic_data(start, end)

    def _fetch_tushare(self, symbol: str, start: str, end: Optional[str], interval: str, adj: str) -> pd.DataFrame:
        """
        Fetch data from Tushare using pro_bar for adjusted prices.
        Symbol format: '000001.SZ' or '600000.SH'
        """
        try:
            if end is None:
                end = pd.Timestamp.now().strftime('%Y%m%d')
            
            # Convert dates to YYYYMMDD
            start_fmt = start.replace('-', '')
            end_fmt = end.replace('-', '')
            
            # Use ts.pro_bar to get adjusted data
            # api=self.pro ensures we use the authenticated instance
            df = ts.pro_bar(ts_code=symbol, 
                            api=self.pro, 
                            adj=adj, 
                            start_date=start_fmt, 
                            end_date=end_fmt)
            
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol} from Tushare. Using synthetic data.")
                return self.generate_synthetic_data(start, end)
                
            # Tushare returns date descending usually, and columns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
            
            # Sort by date ascending
            df = df.sort_values('trade_date')
            df.index = pd.to_datetime(df['trade_date'])
            
            # Rename columns
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in required_cols if c in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Tushare: {e}. Using synthetic data.")
            return self.generate_synthetic_data(start, end)

    def generate_synthetic_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing/fallback.
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        if n == 0:
            return pd.DataFrame()
            
        # Random walk
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        price_path = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['close'] = price_path
        df['open'] = df['close'] * (1 + np.random.normal(0, 0.005, n))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, n)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, n)))
        df['volume'] = np.random.randint(1000, 100000, n).astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily data to weekly.
        """
        if df.empty:
            return df
            
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to Weekly (W-FRI)
        weekly_df = df.resample('W-FRI').agg(logic)
        weekly_df = weekly_df.dropna()
        
        return weekly_df

if __name__ == "__main__":
    # Quick test
    feed = DataFeed()
    data = feed.fetch_data('AAPL', '2023-01-01', '2023-12-31')
    print(data.head())
    print(data.tail())
