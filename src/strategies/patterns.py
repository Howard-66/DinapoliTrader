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
            Columns: [signal, pattern, pattern_sl, pattern_tp, metadata]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
        
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
                             a_price = pattern_low
                             b_price = high.iloc[prev_up_idx:prev_down_idx+1].max()
                             c_price = low.iloc[prev_down_idx:i+1].min()
                             
                             op_price = c_price + (b_price - a_price) * 1.0
                             
                             # Find indices for visualization
                             # We need the actual dates or integer indices relative to the dataframe
                             # Let's store the index labels (timestamps)
                             
                             # Find the exact bar where B and C occurred
                             # B: Max high in range [prev_up_idx, prev_down_idx]
                             b_range = high.iloc[prev_up_idx:prev_down_idx+1]
                             b_idx = b_range.idxmax()
                             
                             # C: Min low in range [prev_down_idx, i]
                             c_range = low.iloc[prev_down_idx:i+1]
                             c_idx = c_range.idxmin()
                             
                             # A: Min low around prev_up_idx (start of pattern)
                             # Usually A is the low before the first thrust.
                             a_range = low.iloc[max(0, prev_up_idx-5):prev_up_idx+1]
                             a_idx = a_range.idxmin()
                             
                             metadata = {
                                 'A': {'date': a_idx, 'price': a_price},
                                 'B': {'date': b_idx, 'price': b_price},
                                 'C': {'date': c_idx, 'price': c_price},
                                 'entry': {'date': self.df.index[i], 'price': close.iloc[i]}
                             }
                             
                             signals.iloc[i] = ['BUY', 'Double Repo', sl_price, op_price, metadata]

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
        filtered_signals.loc[buy_mask & (~trend_aligned), 'metadata'] = np.nan
        
        # Filter SELLs
        sell_mask = (filtered_signals['signal'] == 'SELL')
        trend_aligned_sell = close < trend_ma
        filtered_signals.loc[sell_mask & (~trend_aligned_sell), 'signal'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_aligned_sell), 'pattern'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_aligned_sell), 'metadata'] = np.nan
        
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
        filtered_signals.loc[buy_mask & (~trend_ok), 'metadata'] = np.nan
        
        # Filter SELLs (Require Weekly Bearish)
        sell_mask = (filtered_signals['signal'] == 'SELL')
        trend_ok_sell = (daily_trend == -1)
        filtered_signals.loc[sell_mask & (~trend_ok_sell), 'signal'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_ok_sell), 'pattern'] = np.nan
        filtered_signals.loc[sell_mask & (~trend_ok_sell), 'metadata'] = np.nan
        
        return filtered_signals

    def detect_single_penetration(self, thrust_bars: int = 8) -> pd.DataFrame:
        """
        检测 Single Penetration (Bread & Butter) 模式。
        
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp, metadata]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
        
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
                         
                         # Pattern TP: Thrust High
                         thrust_high = high.iloc[thrust_start_idx:i].max()
                         tp_price = thrust_high
                         
                         metadata = {
                             'thrust_start': {'date': self.df.index[thrust_start_idx], 'price': low.iloc[thrust_start_idx]},
                             'thrust_end': {'date': self.df.index[i-1], 'price': high.iloc[i-1]},
                             'penetration': {'date': self.df.index[i], 'price': low.iloc[i]}
                         }
                         
                         signals.iloc[i] = ['BUY', 'Single Penetration', sl_price, tp_price, metadata]
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
                    
                    metadata = {
                         'thrust_start': {'date': self.df.index[thrust_start_idx], 'price': high.iloc[thrust_start_idx]},
                         'thrust_end': {'date': self.df.index[i-1], 'price': low.iloc[i-1]},
                         'penetration': {'date': self.df.index[i], 'price': high.iloc[i]}
                    }
                    
                    signals.iloc[i] = ['SELL', 'Single Penetration', sl_price, tp_price, metadata]
                    current_bear_run = 0
                    
        return signals

    def detect_railroad_tracks(self, lookback: int = 5) -> pd.DataFrame:
        """
        检测 Railroad Tracks (RRT) 模式。
        Logic: Two adjacent candles with opposite directions and similar large bodies.
        
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp, metadata]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
        
        close = self.df['close']
        open_price = self.df['open']
        high = self.df['high']
        low = self.df['low']
        
        # Calculate body size and range
        body = (close - open_price).abs()
        total_range = high - low
        
        # Average body size for relative comparison
        avg_body = body.rolling(window=20).mean()
        
        for i in range(1, len(self.df)):
            # Check for sufficient body size (e.g., > 1.5 * avg_body)
            # This ensures "large bodies"
            if body.iloc[i] < 1.0 * avg_body.iloc[i] or body.iloc[i-1] < 1.0 * avg_body.iloc[i-1]:
                continue
                
            # Check for opposite directions
            # Candle 1 (Previous): Bullish, Candle 2 (Current): Bearish -> Bearish RRT
            is_prev_bull = close.iloc[i-1] > open_price.iloc[i-1]
            is_curr_bear = close.iloc[i] < open_price.iloc[i]
            
            # Candle 1: Bearish, Candle 2: Bullish -> Bullish RRT
            is_prev_bear = close.iloc[i-1] < open_price.iloc[i-1]
            is_curr_bull = close.iloc[i] > open_price.iloc[i]
            
            # Check for "Similar Length"
            # Body difference shouldn't be too large (e.g., within 30%)
            body_ratio = body.iloc[i] / body.iloc[i-1]
            is_similar_size = 0.7 <= body_ratio <= 1.3
            
            if not is_similar_size:
                continue
                
            if is_prev_bull and is_curr_bear:
                # Bearish RRT (Top Reversal)
                # Signal: SELL
                # SL: Max High of the two candles
                sl_price = max(high.iloc[i], high.iloc[i-1])
                
                # TP: Target 1:1 or 1.618 of the pattern height
                pattern_height = sl_price - min(low.iloc[i], low.iloc[i-1])
                tp_price = min(low.iloc[i], low.iloc[i-1]) - pattern_height * 1.0
                
                metadata = {
                    'bar1': {'date': self.df.index[i-1], 'open': open_price.iloc[i-1], 'close': close.iloc[i-1]},
                    'bar2': {'date': self.df.index[i], 'open': open_price.iloc[i], 'close': close.iloc[i]}
                }
                
                signals.iloc[i] = ['SELL', 'Railroad Tracks', sl_price, tp_price, metadata]
                
            elif is_prev_bear and is_curr_bull:
                # Bullish RRT (Bottom Reversal)
                # Signal: BUY
                # SL: Min Low of the two candles
                sl_price = min(low.iloc[i], low.iloc[i-1])
                
                # TP
                pattern_height = max(high.iloc[i], high.iloc[i-1]) - sl_price
                tp_price = max(high.iloc[i], high.iloc[i-1]) + pattern_height * 1.0
                
                metadata = {
                    'bar1': {'date': self.df.index[i-1], 'open': open_price.iloc[i-1], 'close': close.iloc[i-1]},
                    'bar2': {'date': self.df.index[i], 'open': open_price.iloc[i], 'close': close.iloc[i]}
                }
                
                signals.iloc[i] = ['BUY', 'Railroad Tracks', sl_price, tp_price, metadata]
                
        return signals

    def detect_failure_to_penetrate(self, lookback: int = 5) -> pd.DataFrame:
        """
        检测 Failure to Penetrate (FTP) 模式。
        Logic: Price penetrates DMA 3x3 (High > DMA for Sell, Low < DMA for Buy) 
               but closes back inside (Close < DMA for Sell, Close > DMA for Buy).
               
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp, metadata]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
        
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        dma3 = self.dma_3x3
        
        for i in range(1, len(self.df)):
            # Bullish FTP (Support Hold)
            if low.iloc[i] < dma3.iloc[i] and close.iloc[i] > dma3.iloc[i]:
                # Potential Bullish FTP
                if close.iloc[i-1] > dma3.iloc[i-1]: 
                    # BUY Signal
                    sl_price = low.iloc[i] # The wick low
                    
                    # TP: Recent High or fixed R:R
                    risk = close.iloc[i] - sl_price
                    tp_price = close.iloc[i] + risk * 2.0 # 2:1 Reward
                    
                    metadata = {
                        'penetration': {'date': self.df.index[i], 'low': low.iloc[i], 'close': close.iloc[i], 'dma3': dma3.iloc[i]}
                    }
                    
                    signals.iloc[i] = ['BUY', 'Failure to Penetrate', sl_price, tp_price, metadata]
            
            # Bearish FTP (Resistance Hold)
            if high.iloc[i] > dma3.iloc[i] and close.iloc[i] < dma3.iloc[i]:
                # Potential Bearish FTP
                if close.iloc[i-1] < dma3.iloc[i-1]:
                    # SELL Signal
                    sl_price = high.iloc[i] # The wick high
                    
                    risk = sl_price - close.iloc[i]
                    tp_price = close.iloc[i] - risk * 2.0
                    
                    metadata = {
                        'penetration': {'date': self.df.index[i], 'high': high.iloc[i], 'close': close.iloc[i], 'dma3': dma3.iloc[i]}
                    }
                    
                    signals.iloc[i] = ['SELL', 'Failure to Penetrate', sl_price, tp_price, metadata]
                    
        return signals
