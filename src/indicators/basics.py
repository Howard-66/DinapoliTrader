import pandas as pd
import numpy as np
import talib

class Indicators:
    """
    Library of technical indicators.
    Uses TA-Lib for performance where possible, falls back to pandas/numpy.
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return pd.Series(talib.SMA(series.values, timeperiod=period), index=series.index)

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Moving Average Convergence Divergence"""
        macd, macdsignal, macdhist = talib.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(macd, index=series.index), pd.Series(macdsignal, index=series.index), pd.Series(macdhist, index=series.index)

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)

    @staticmethod
    def displaced_ma(series: pd.Series, period: int, displacement: int) -> pd.Series:
        """
        Displaced Moving Average (DMA).
        Calculates SMA/EMA and shifts it forward.
        DiNapoli uses SMA typically.
        """
        ma = talib.SMA(series.values, timeperiod=period)
        return pd.Series(ma, index=series.index).shift(displacement)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
