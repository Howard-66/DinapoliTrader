import pandas as pd
import numpy as np
from src.indicators.basics import Indicators

class FeatureExtractor:
    """
    Extracts technical features for ML model.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.indicators = self._calculate_indicators()
        
    def _calculate_indicators(self) -> pd.DataFrame:
        """Pre-calculate all necessary indicators."""
        df = self.df.copy()
        
        # Basic Indicators
        df['rsi'] = Indicators.rsi(df['close'], 14)
        df['sma_50'] = Indicators.sma(df['close'], 50)
        df['sma_200'] = Indicators.sma(df['close'], 200)
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], 14)
        
        # DiNapoli Specific
        df['dma_3x3'] = Indicators.displaced_ma(df['close'], 3, 3)
        df['dma_25x5'] = Indicators.displaced_ma(df['close'], 25, 5)
        
        # Derived Features
        # Distance to DMAs (normalized by price)
        df['dist_dma3'] = (df['close'] - df['dma_3x3']) / df['close']
        df['dist_dma25'] = (df['close'] - df['dma_25x5']) / df['close']
        
        # Trend Alignment
        df['trend_long'] = (df['close'] > df['sma_200']).astype(int)
        df['trend_med'] = (df['close'] > df['sma_50']).astype(int)
        
        # Volatility
        df['volatility'] = df['atr'] / df['close']
        
        # Momentum
        df['mom_5'] = df['close'].pct_change(5)
        
        return df
        
    def get_features(self, idx: int) -> dict:
        """
        Get features for a specific index.
        """
        if idx not in self.indicators.index:
            return {}
            
        row = self.indicators.loc[idx]
        
        return {
            'rsi': row['rsi'],
            'dist_dma3': row['dist_dma3'],
            'dist_dma25': row['dist_dma25'],
            'volatility': row['volatility'],
            'mom_5': row['mom_5'],
            'trend_long': row['trend_long']
        }
        
    def get_all_features(self) -> pd.DataFrame:
        """Return DataFrame of features only."""
        cols = ['rsi', 'dist_dma3', 'dist_dma25', 'volatility', 'mom_5', 'trend_long']
        return self.indicators[cols].fillna(0)
