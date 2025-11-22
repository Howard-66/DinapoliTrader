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
        
        逻辑（以看涨为例）：
        1. 价格在3x3 DMA下方（趋势向下）。
        2. 价格突破3x3 DMA向上（第一次穿透）。
        3. 价格回落到3x3 DMA下方。
        4. 价格再次突破3x3 DMA向上（第二次穿透）。
        5. 确认：第二次穿透后的收盘价必须高于第一次穿透的高点（可选，或仅需收盘在3x3上方）。
        6. 过滤：必须远离25x5 DMA（超卖）。
        
        Returns:
            pd.DataFrame: 包含信号的DataFrame。
            Columns: [signal_type, entry_price, stop_loss, target]
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern'])
        
        # 简化逻辑实现：
        # 遍历最近N根K线，寻找穿透事件
        # 这在向量化计算中比较复杂，这里使用循环遍历（性能较低但逻辑清晰）
        # 实际生产中应优化为向量化
        
        close = self.df['close']
        dma3 = self.dma_3x3
        
        # 标记穿透点：Cross Over (1) / Cross Under (-1)
        cross = pd.Series(0, index=self.df.index)
        cross[(close > dma3) & (close.shift(1) <= dma3.shift(1))] = 1 # Up Cross
        cross[(close < dma3) & (close.shift(1) >= dma3.shift(1))] = -1 # Down Cross
        
        # 寻找Double Repo Buy
        # 序列：Up Cross -> Down Cross -> Up Cross
        # 且发生在短时间内
        
        # 这里仅做简单演示逻辑，实际需要更严格的状态机
        
        for i in range(lookback, len(self.df)):
            # 检查当前是否是 Up Cross
            if cross.iloc[i] == 1:
                # 向前回溯寻找上一个 Up Cross
                # 在这两个 Up Cross 之间必须有一个 Down Cross
                
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
                        # 检查是否“远离”25x5 DMA (这里简化为距离判断)
                        # 假设当前价格低于 25x5 DMA (大趋势向下)
                        if close.iloc[i] < self.dma_25x5.iloc[i]:
                             signals.iloc[i] = ['BUY', 'Double Repo']
                             
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

    def detect_single_penetration(self, thrust_bars: int = 8) -> pd.DataFrame:
        """
        检测 Single Penetration (Bread & Butter) 模式。
        
        逻辑（以看涨为例）：
        1. Thrust: 价格连续 N 根 K 线收盘在 3x3 DMA 之上。
        2. Penetration: 价格回调并触及 3x3 DMA（Low <= 3x3）。
        3. 信号：在触及点生成买入信号。
        
        Args:
            thrust_bars (int): 定义 Thrust 所需的最少 K 线数量 (默认 8)。
            
        Returns:
            pd.DataFrame: 包含信号的 DataFrame。
        """
        signals = pd.DataFrame(index=self.df.index, columns=['signal', 'pattern'])
        
        close = self.df['close']
        low = self.df['low']
        high = self.df['high']
        dma3 = self.dma_3x3
        
        # 1. Identify Thrust
        # Bullish Thrust: Close > 3x3 DMA
        bull_thrust = (close > dma3).astype(int)
        # Bearish Thrust: Close < 3x3 DMA
        bear_thrust = (close < dma3).astype(int)
        
        # Calculate consecutive thrust bars
        # Group consecutive 1s and count
        # Simple loop for clarity (vectorization possible with cumsum-reset logic)
        
        current_bull_run = 0
        current_bear_run = 0
        
        for i in range(1, len(self.df)):
            # Bullish Logic
            if close.iloc[i-1] > dma3.iloc[i-1]:
                current_bull_run += 1
            else:
                current_bull_run = 0
                
            # Check for Penetration after sufficient thrust
            if current_bull_run >= thrust_bars:
                # Check if CURRENT bar penetrates (Low <= 3x3)
                # Note: The thrust condition applies to PREVIOUS bars.
                # If current bar opens above but touches 3x3, it's a buy.
                if low.iloc[i] <= dma3.iloc[i] and close.iloc[i] > dma3.iloc[i]: # Close > 3x3 ensures it's a bounce, not a breakdown? 
                    # DiNapoli definition: "The first time the market penetrates the 3x3 DMA after a thrust"
                    # We can just mark the penetration.
                    if low.iloc[i] <= dma3.iloc[i]:
                         signals.iloc[i] = ['BUY', 'Single Penetration']
                         # Reset run to avoid multiple signals in same pullback? 
                         # Usually we take the first one.
                         current_bull_run = 0 

            # Bearish Logic
            if close.iloc[i-1] < dma3.iloc[i-1]:
                current_bear_run += 1
            else:
                current_bear_run = 0
                
            if current_bear_run >= thrust_bars:
                if high.iloc[i] >= dma3.iloc[i]:
                    signals.iloc[i] = ['SELL', 'Single Penetration']
                    current_bear_run = 0
                    
        return signals
