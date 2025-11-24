import pandas as pd
import numpy as np
from src.indicators.basics import Indicators

class PatternRecognizer:
    """
    DiNapoli模式识别器。
    目前支持：
    1. Double Repo (Double Repenetration) - 失败的突破模式
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # 预计算必要的指标
        self.dma_3x3 = Indicators.displaced_ma(df['close'], 3, 3)
        self.dma_7x5 = Indicators.displaced_ma(df['close'], 7, 5)
        self.dma_25x5 = Indicators.displaced_ma(df['close'], 25, 5)

    def detect_double_repo(self, lookback: int = 20) -> pd.DataFrame:
        """
        检测Double Repo模式。
        
        Returns:
            pd.DataFrame: 包含信号的DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp'])
        
        close = self.df['close']
        low = self.df['low']
        high = self.df['high']
        dma3 = self.dma_3x3
        
        # 标记穿透点：Cross Over (1) / Cross Under (-1)
        cross = pd.Series(0, index=self.df.index)
        cross[(close > dma3) & (close.shift(1) <= dma3.shift(1))] = 1 # Up Cross
        cross[(close < dma3) & (close.shift(1) >= dma3.shift(1))] = -1 # Down Cross
        
        for i in range(lookback, len(self.df)):
            # 检查当前是否是 Up Cross (Buy Signal Trigger)
            if cross.iloc[i] == 1:
                # 寻找最近的 Down Cross
                prev_down_idx = -1
                for j in range(i-1, i-lookback, -1):
                    if cross.iloc[j] == -1:
                        prev_down_idx = j
                        break
                
                if prev_down_idx != -1:
                    # 寻找更早的 Up Cross
                    prev_up_idx = -1
                    for k in range(prev_down_idx-1, i-lookback, -1):
                        if cross.iloc[k] == 1:
                            prev_up_idx = k
                            break
                            
                    if prev_up_idx != -1:
                        # 找到了 Up -> Down -> Up 序列
                        if close.iloc[i] < self.dma_25x5.iloc[i]:
                             # Double Repo Buy
                             # Pattern SL: Lowest Low during the formation (between first Up Cross and current)
                             pattern_low = low.iloc[prev_up_idx:i+1].min()
                             sl_price = pattern_low
                             
                             # Pattern TP: Fibonacci Expansion (COP = 0.618)
                             # A: Start of move (Low before 1st Cross) - Simplified: Use Pattern Low
                             # B: High of 1st thrust (Max High between 1st Cross and Down Cross)
                             # C: Low of retracement (Min Low between Down Cross and 2nd Cross)
                             
                             # Simplified Logic for TP (OP Target)
                             # Target = Entry + (Entry - SL) * 1.618 (Classic R:R) or Fib Expansion
                             # Let's use simple Fib Expansion logic if we can identify A-B-C
                             # A = pattern_low
                             # B = high.iloc[prev_up_idx:prev_down_idx+1].max()
                             # C = low.iloc[prev_down_idx:i+1].min()
                             
                             # If C < A, it invalidates, but let's assume valid for now
                             # OP = C + (B-A) * 1.0
                             
                             a_price = pattern_low
                             b_price = high.iloc[prev_up_idx:prev_down_idx+1].max()
                             c_price = low.iloc[prev_down_idx:i+1].min()
                             
                             op_price = c_price + (b_price - a_price) * 1.0
                             
                             signals.iloc[i] = ['BUY', 'Double Repo', sl_price, op_price]

        return signals

    def apply_trend_filter(self, signals: pd.DataFrame, trend_ma: pd.Series) -> pd.DataFrame:
        """
        Apply Trend Filter (e.g., SMA 200).
        BUY only if Close > Trend MA.
        SELL only if Close < Trend MA.
        """
        if trend_ma is None:
            return signals
            
        filtered_signals = signals.copy()
        close = self.df['close']
        
        # Filter BUYs
        buy_mask = (filtered_signals['signal'] == 'BUY')
        # Ensure alignment
        trend_aligned = close > trend_ma
        filtered_signals.loc[buy_mask & (~trend_aligned), 'signal'] = np.nan
        filtered_signals.loc[buy_mask & (~trend_aligned), 'pattern'] = np.nan
        
        # Filter SELLs
        sell_mask = (filtered_signals['signal'] == 'SELL')
        trend_aligned_sell = close < trend_ma
        filtered_signals.loc[sell_mask & (~trend_aligned_sell), 'signal'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_aligned_sell), 'pattern'] = np.nan
        
        return filtered_signals

    def apply_mtf_filter(self, signals: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Multi-Timeframe Filter.
        Filter Daily signals based on Weekly Trend (Close > Weekly 25x5 DMA).
        """
        if weekly_df is None or weekly_df.empty:
            return signals
            
        # Calculate Weekly DMA 25x5
        weekly_dma25 = Indicators.displaced_ma(weekly_df['close'], 25, 5)
        
        # Determine Weekly Trend
        # 1 = Bullish, -1 = Bearish
        weekly_trend = pd.Series(0, index=weekly_df.index)
        weekly_trend[weekly_df['close'] > weekly_dma25] = 1
        weekly_trend[weekly_df['close'] < weekly_dma25] = -1
        
        # Resample Weekly Trend to Daily (ffill)
        # Reindex to match daily df
        daily_trend = weekly_trend.reindex(self.df.index, method='ffill')
        
        filtered_signals = signals.copy()
        
        # Filter BUYs (Require Weekly Bullish)
        buy_mask = (filtered_signals['signal'] == 'BUY')
        trend_ok = (daily_trend == 1)
        filtered_signals.loc[buy_mask & (~trend_ok), 'signal'] = np.nan
        filtered_signals.loc[buy_mask & (~trend_ok), 'pattern'] = np.nan
        
        # Filter SELLs (Require Weekly Bearish)
        sell_mask = (filtered_signals['signal'] == 'SELL')
        trend_ok_sell = (daily_trend == -1)
        filtered_signals.loc[sell_mask & (~trend_ok_sell), 'signal'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_ok_sell), 'pattern'] = np.nan
        
        return filtered_signals

    def detect_single_penetration(self, thrust_bars: int = 8) -> pd.DataFrame:
        """
        检测 Single Penetration (Bread & Butter) 模式。
        
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp'])
        
        close = self.df['close']
        low = self.df['low']
        high = self.df['high']
        dma3 = self.dma_3x3
        
        current_bull_run = 0
        current_bear_run = 0
        
        # Store start of thrust for SL calculation
        thrust_start_idx = 0
        
        for i in range(1, len(self.df)):
            # Bullish Logic
            if close.iloc[i-1] > dma3.iloc[i-1]:
                if current_bull_run == 0:
                    thrust_start_idx = i-1
                current_bull_run += 1
            else:
                current_bull_run = 0
                
            # Check for Penetration after sufficient thrust
            if current_bull_run >= thrust_bars:
                if low.iloc[i] <= dma3.iloc[i] and close.iloc[i] > dma3.iloc[i]: 
                    if low.iloc[i] <= dma3.iloc[i]:
                         # BUY Signal
                         # Pattern SL: Low of the thrust start (or recent swing low)
                         # Simplified: Min Low during the thrust
                         sl_price = low.iloc[thrust_start_idx:i].min()
                         
                         # Pattern TP: Fibonacci Retracement of the thrust
                         # Thrust High = Max High during thrust
                         thrust_high = high.iloc[thrust_start_idx:i].max()
                         thrust_low = sl_price
                         
                         # Target: Usually a retest of highs or 0.618 retracement of the drop?
                         # For Single Pen (Trend Following), target is continuation.
                         # But DiNapoli often trades the reaction.
                         # Let's set TP at Thrust High (Conservative) or 1.618 extension
                         # Standard Bread & Butter: Target is simply a reaction.
                         # Let's use Thrust High as a logical target.
                         tp_price = thrust_high
                         
                         signals.iloc[i] = ['BUY', 'Single Penetration', sl_price, tp_price]
                         current_bull_run = 0 

            # Bearish Logic
            if close.iloc[i-1] < dma3.iloc[i-1]:
                if current_bear_run == 0:
                    thrust_start_idx = i-1
                current_bear_run += 1
            else:
                current_bear_run = 0
                
            if current_bear_run >= thrust_bars:
                if high.iloc[i] >= dma3.iloc[i]:
                    # SELL Signal
                    sl_price = high.iloc[thrust_start_idx:i].max()
                    thrust_low = low.iloc[thrust_start_idx:i].min()
                    tp_price = thrust_low
                    
                    signals.iloc[i] = ['SELL', 'Single Penetration', sl_price, tp_price]
                    current_bear_run = 0
                    
        return signals
