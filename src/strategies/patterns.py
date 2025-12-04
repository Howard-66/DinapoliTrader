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

    def detect_double_repo(self, lookback: int = 20, min_dma25_dist_pct: float = 0.005, min_trend_bars: int = 3, fib_target: str = 'OP') -> pd.DataFrame:
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
        
        # Pre-calculate filters
        rrt_filter = self.check_railroad_tracks()
        ftp_filter = self.check_failure_to_penetrate()
        
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
                        # Check Initial Trend Condition
                        # Before prev_up_idx (First Penetration), price should be below 3x3 DMA for min_trend_bars
                        trend_valid = True
                        if min_trend_bars > 0:
                            # Check window [prev_up_idx - min_trend_bars : prev_up_idx]
                            # Note: prev_up_idx is the index where Close > DMA (Cross Over)
                            # So we check bars BEFORE it.
                            start_check = prev_up_idx - min_trend_bars
                            if start_check >= 0:
                                # For Buy: Close < DMA
                                pre_trend = (close.iloc[start_check:prev_up_idx] < dma3.iloc[start_check:prev_up_idx])
                                if not pre_trend.all():
                                    trend_valid = False
                            else:
                                trend_valid = False # Not enough data
                        
                        if trend_valid:
                            # 找到了 Up -> Down -> Up 序列
                            # Refined 25x5 Filter: Price should be "far" from 25x5
                            # Calculate distance percentage
                            dma25_val = self.dma_25x5.iloc[i]
                            close_val = close.iloc[i]
                            
                            # For Buy: DMA25 should be significantly above Close
                            dist_pct = (dma25_val - close_val) / close_val
                            
                            if dist_pct > min_dma25_dist_pct:
                             # Double Repo Buy
                             # Pattern SL: Lowest Low during the formation (between first Up Cross and current)
                             pattern_low = low.iloc[prev_up_idx:i+1].min()
                             sl_price = pattern_low
                             
                             # Pattern TP: Fibonacci Expansion
                             # COP = 0.618, OP = 1.0, XOP = 1.618
                             fib_ratios = {'COP': 0.618, 'OP': 1.0, 'XOP': 1.618}
                             ratio = fib_ratios.get(fib_target, 1.0)
                             
                             a_price = pattern_low
                             b_price = high.iloc[prev_up_idx:prev_down_idx+1].max()
                             c_price = low.iloc[prev_down_idx:i+1].min()
                             
                             op_price = c_price + (b_price - a_price) * ratio
                             
                             # Find indices for visualization
                             # B: Max high in range [prev_up_idx, prev_down_idx]
                             b_range = high.iloc[prev_up_idx:prev_down_idx+1]
                             b_idx = b_range.idxmax()
                             
                             # C: Min low in range [prev_down_idx, i]
                             c_range = low.iloc[prev_down_idx:i+1]
                             c_idx = c_range.idxmin()
                             
                             # A: Min low around prev_up_idx (start of pattern)
                             a_range = low.iloc[max(0, prev_up_idx-5):prev_up_idx+1]
                             a_idx = a_range.idxmin()
                             
                             metadata = {
                                 'A': {'date': a_idx, 'price': a_price},
                                 'B': {'date': b_idx, 'price': b_price},
                                 'C': {'date': c_idx, 'price': c_price},
                                 'entry': {'date': self.df.index[i], 'price': close.iloc[i]},
                                 'rrt_detected': rrt_filter.iloc[i],
                                 'ftp_detected': ftp_filter.iloc[i]
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
        daily_trend = weekly_trend.reindex(self.df.index).ffill()
        
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
        Refined Logic:
        1. Thrust: Close > 3x3 DMA for N bars.
        2. Penetration: Price touches 3x3 DMA (Low <= DMA).
        3. Entry: Limit order at 3x3 DMA.
        
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
            Columns: [signal, pattern, pattern_sl, pattern_tp, metadata]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern', 'pattern_sl', 'pattern_tp', 'metadata'])
        
        close = self.df['close']
        low = self.df['low']
        high = self.df['high']
        dma3 = self.dma_3x3
        
        # Pre-calculate filters
        rrt_filter = self.check_railroad_tracks()
        ftp_filter = self.check_failure_to_penetrate()
        
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
                # Touch Entry: Low <= DMA
                if low.iloc[i] <= dma3.iloc[i]:
                     # BUY Signal
                     entry_price = dma3.iloc[i]
                     
                     # Pattern SL: Low of the thrust start (or recent swing low)
                     # Simplified: Min Low during the thrust
                     sl_price = low.iloc[thrust_start_idx:i].min()
                     
                     # Pattern TP: Thrust High
                     thrust_high = high.iloc[thrust_start_idx:i].max()
                     tp_price = thrust_high
                     
                     metadata = {
                         'thrust_start': {'date': self.df.index[thrust_start_idx], 'price': low.iloc[thrust_start_idx]},
                         'thrust_end': {'date': self.df.index[i-1], 'price': high.iloc[i-1]},
                         'penetration': {'date': self.df.index[i], 'price': low.iloc[i]},
                         'rrt_detected': rrt_filter.iloc[i],
                         'ftp_detected': ftp_filter.iloc[i]
                     }
                     
                     signals.iloc[i] = ['BUY', 'Single Penetration', sl_price, tp_price, metadata]
                     current_bull_run = 0 # Reset after signal

            # Bearish Logic
            if close.iloc[i-1] < dma3.iloc[i-1]:
                if current_bear_run == 0:
                    thrust_start_idx = i-1
                current_bear_run += 1
            else:
                current_bear_run = 0
                
            if current_bear_run >= thrust_bars:
                # Touch Entry: High >= DMA
                if high.iloc[i] >= dma3.iloc[i]:
                    # SELL Signal
                    entry_price = dma3.iloc[i]
                    
                    sl_price = high.iloc[thrust_start_idx:i].max()
                    thrust_low = low.iloc[thrust_start_idx:i].min()
                    tp_price = thrust_low
                    
                    metadata = {
                         'thrust_start': {'date': self.df.index[thrust_start_idx], 'price': high.iloc[thrust_start_idx]},
                         'thrust_end': {'date': self.df.index[i-1], 'price': low.iloc[i-1]},
                         'penetration': {'date': self.df.index[i], 'price': high.iloc[i]},
                         'rrt_detected': rrt_filter.iloc[i],
                         'ftp_detected': ftp_filter.iloc[i]
                    }
                    
                    signals.iloc[i] = ['SELL', 'Single Penetration', sl_price, tp_price, metadata]
                    current_bear_run = 0
                    
        return signals

    def check_railroad_tracks(self) -> pd.Series:
        """
        Check for Railroad Tracks (RRT) pattern.
        Returns: Boolean Series (True where RRT is detected).
        """
        close = self.df['close']
        open_price = self.df['open']
        high = self.df['high']
        low = self.df['low']
        
        # Calculate body size
        body = (close - open_price).abs()
        
        # Average body size for relative comparison
        avg_body = body.rolling(window=20).mean()
        
        # Initialize result
        is_rrt = pd.Series(False, index=self.df.index)
        
        # Vectorized approach is harder for complex conditions, iterating is fine for now
        # or use shift() for vectorization
        
        # Condition 1: Large bodies (> 1.0 * avg)
        large_body = body > (1.0 * avg_body)
        prev_large_body = large_body.shift(1)
        
        # Condition 2: Opposite directions
        bullish = close > open_price
        bearish = close < open_price
        
        opp_dir_1 = bullish.shift(1) & bearish # Bull -> Bear (Top)
        opp_dir_2 = bearish.shift(1) & bullish # Bear -> Bull (Bottom)
        
        # Condition 3: Similar size (within 30%)
        body_ratio = body / body.shift(1)
        similar_size = (body_ratio >= 0.7) & (body_ratio <= 1.3)
        
        # Combine
        rrt_condition = (large_body & prev_large_body & (opp_dir_1 | opp_dir_2) & similar_size)
        is_rrt[rrt_condition] = True
        
        return is_rrt

    def check_failure_to_penetrate(self) -> pd.Series:
        """
        Check for Failure to Penetrate (FTP) pattern.
        Returns: Boolean Series (True where FTP is detected).
        """
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        dma3 = self.dma_3x3
        
        is_ftp = pd.Series(False, index=self.df.index)
        
        # Bullish FTP (Support Hold): Low < DMA but Close > DMA, and Prev Close > Prev DMA
        bull_ftp = (low < dma3) & (close > dma3) & (close.shift(1) > dma3.shift(1))
        
        # Bearish FTP (Resistance Hold): High > DMA but Close < DMA, and Prev Close < Prev DMA
        bear_ftp = (high > dma3) & (close < dma3) & (close.shift(1) < dma3.shift(1))
        
        is_ftp[bull_ftp | bear_ftp] = True
        
        return is_ftp
