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
