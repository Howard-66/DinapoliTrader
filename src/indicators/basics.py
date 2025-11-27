import pandas as pd
import numpy as np

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

class Indicators:
    """
    Library of technical indicators.
    Uses TA-Lib for performance where possible, falls back to pandas/numpy.
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if HAS_TALIB:
            return pd.Series(talib.SMA(series.values, timeperiod=period), index=series.index)
        else:
            return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        if HAS_TALIB:
            return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)
        else:
            return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Moving Average Convergence Divergence"""
        if HAS_TALIB:
            macd, macdsignal, macdhist = talib.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(macd, index=series.index), pd.Series(macdsignal, index=series.index), pd.Series(macdhist, index=series.index)
        else:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macdsignal = macd.ewm(span=signal, adjust=False).mean()
            macdhist = macd - macdsignal
            return macd, macdsignal, macdhist

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if HAS_TALIB:
            return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)
        else:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    @staticmethod
    def displaced_ma(series: pd.Series, period: int, displacement: int) -> pd.Series:
        """
        Displaced Moving Average (DMA).
        Calculates SMA/EMA and shifts it forward.
        DiNapoli uses SMA typically.
        """
        if HAS_TALIB:
            ma = talib.SMA(series.values, timeperiod=period)
            return pd.Series(ma, index=series.index).shift(displacement)
        else:
            return series.rolling(window=period).mean().shift(displacement)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        if HAS_TALIB:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        else:
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()
