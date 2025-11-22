import backtrader as bt
import pandas as pd
from src.strategies.patterns import PatternRecognizer
from src.indicators.basics import Indicators

class DiNapoliStrategy(bt.Strategy):
    """
    Backtrader策略包装器。
    将自定义的PatternRecognizer逻辑集成到Backtrader中。
    """
    params = (
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.05),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        # 为了使用我们的PatternRecognizer，我们需要完整的历史数据
        # Backtrader通常是逐步运行的 (next)
        # 我们可以预先计算所有信号，或者在next中动态计算
        # 预计算效率更高，但为了模拟实时，我们在next中获取切片
        
        # 简单起见，我们在这里不进行复杂的动态计算，
        # 而是假设我们已经有了信号（在外部计算好传入，或者在start中计算）
        # 但为了演示Backtrader的整合，我们在next中做简单的逻辑
        
        # 实际上，为了复用 src/strategies/patterns.py，我们需要将BT的数据转换为Pandas
        pass

    def next(self):
        if self.order:
            return

        # 获取当前时间点的数据切片 (作为Pandas DataFrame)
        # 注意：这在回测中可能比较慢，优化方法是预计算指标
        # 这里为了演示集成，我们简化处理：
        # 仅使用简单的逻辑，或者假设我们有一个预计算的信号流
        
        # 让我们实现一个简单的逻辑：
        # 如果收盘价 > 3x3 DMA 且 之前 < 3x3 DMA -> Buy (简化版)
        
        # 计算 3x3 DMA
        # BT有内置指标，但我们想用自己的逻辑
        # 我们可以用 bt.indicators.SMA(period=3).plotinfo.plot = False
        # 然后 shift
        
        # 为了真正复用我们的 PatternRecognizer，最佳实践是：
        # 1. 在回测开始前，用 DataFeed 获取数据。
        # 2. 用 PatternRecognizer 计算所有信号。
        # 3. 将信号作为额外的 Data Feed 传入 Backtrader，或者在 Strategy 中直接读取预计算的信号表。
        
        # 这里我们采用“预计算信号”模式
        # 假设 self.datas[0] 有一个 'signal' line (我们稍后在 loader 中添加)
        # 或者我们通过 params 传入信号 dataframe
        pass

class DiNapoliStrategyWithSignals(bt.Strategy):
    """
    接收预计算信号的策略。
    """
    params = (
        ('signals', None), # pd.DataFrame with index as datetime and 'signal' column
    )
    
    def __init__(self):
        self.signals = self.params.signals
        self.order = None

    def next(self):
        if self.order:
            return
            
        dt = self.datas[0].datetime.datetime(0)
        
        if self.signals is not None and dt in self.signals.index:
            sig = self.signals.loc[dt]
            if isinstance(sig, pd.Series):
                sig = sig['signal'] # Handle duplicate index if any, though unlikely
            
            if hasattr(sig, 'values'): # If multiple signals
                 sig = sig.iloc[0]

            if sig == 'BUY':
                self.log(f'BUY CREATE, {self.datas[0].close[0]}')
                self.order = self.buy()
            elif sig == 'SELL':
                self.log(f'SELL CREATE, {self.datas[0].close[0]}')
                self.order = self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
