import pandas as pd
import logging
import datetime
from src.data.feed import DataFeed
from src.strategies.patterns import PatternRecognizer
from src.ml.classifier import SignalClassifier

logger = logging.getLogger(__name__)

class MarketScanner:
    """
    Scans a list of symbols for DiNapoli patterns.
    """

    def __init__(self):
        self.feed = DataFeed()
        self.classifier = SignalClassifier()

    def scan(self, symbols: list, lookback_days: int = 365, scan_window: int = 5) -> pd.DataFrame:
        """
        Scan the list of symbols for active signals in the most recent data.
        
        Args:
            symbols (list): List of ticker symbols.
            lookback_days (int): Number of days of history to fetch.
            scan_window (int): Number of recent bars to check for signals.
            
        Returns:
            pd.DataFrame: DataFrame containing found signals.
        """
        results = []
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            symbol = symbol.strip()
            if not symbol:
                continue
                
            try:
                # Fetch data
                # Note: For Tushare, we might need to be careful with rate limits if list is long
                df = self.feed.fetch_data(symbol, start_date=start_date)
                
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                    
                # Recognize Patterns
                recognizer = PatternRecognizer(df)
                
                # Double Repo
                dr_signals = recognizer.detect_double_repo()
                
                # Single Penetration
                sp_signals = recognizer.detect_single_penetration()
                
                # Check the last N bars for signals
                recent_indices = df.index[-scan_window:]
                
                for idx in recent_indices:
                    # Check Double Repo
                    if idx in dr_signals.index:
                        sig = dr_signals.loc[idx]
                        if not pd.isna(sig['signal']):
                            confidence = self.classifier.predict_proba(df, idx)
                            results.append({
                                'Symbol': symbol,
                                'Date': idx,
                                'Signal': sig['signal'],
                                'Pattern': sig['pattern'],
                                'Close': df.loc[idx, 'close'],
                                'SL': sig['pattern_sl'],
                                'TP': sig['pattern_tp'],
                                'Confidence': confidence
                            })
                            continue # Prioritize DR if both exist (rare)

                    # Check Single Penetration
                    if idx in sp_signals.index:
                        sig = sp_signals.loc[idx]
                        if not pd.isna(sig['signal']):
                            confidence = self.classifier.predict_proba(df, idx)
                            results.append({
                                'Symbol': symbol,
                                'Date': idx,
                                'Signal': sig['signal'],
                                'Pattern': sig['pattern'],
                                'Close': df.loc[idx, 'close'],
                                'SL': sig['pattern_sl'],
                                'TP': sig['pattern_tp'],
                                'Confidence': confidence
                            })

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
                
        if not results:
            return pd.DataFrame(columns=['Symbol', 'Date', 'Signal', 'Pattern', 'Close', 'SL', 'TP', 'Confidence'])
            
        return pd.DataFrame(results)
