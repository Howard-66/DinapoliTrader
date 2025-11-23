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
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.05),
        ('use_atr_sl', False),
        ('atr_multiplier', 2.0),
        ('risk_per_trade_pct', 0.01),
        ('initial_capital', 100000.0),
        ('use_dynamic_sizing', False),
    )
    
    def __init__(self):
        self.signals = self.params.signals
        self.order = None
        self.stop_order = None
        self.limit_order = None
        
        # ATR Indicator for dynamic SL
        if self.params.use_atr_sl or self.params.use_dynamic_sizing:
            self.atr = bt.indicators.ATR(self.data, period=14)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Check if we have an open position
        # Note: With bracket orders, we might have pending child orders even if position is closed?
        # Backtrader handles this if we use buy_bracket.
        
        if self.order:
            return
            
        dt = self.datas[0].datetime.datetime(0)
        
        if self.signals is not None and dt in self.signals.index:
            sig = self.signals.loc[dt]
            if isinstance(sig, pd.Series):
                sig = sig['signal'] 
            
            if hasattr(sig, 'values'): 
                 sig = sig.iloc[0]

            if sig == 'BUY' and not self.position:
                current_price = self.datas[0].close[0]
                
                # Calculate SL Price
                sl_price = 0.0
                if self.params.use_atr_sl:
                    atr_val = self.atr[0]
                    sl_price = current_price - (atr_val * self.params.atr_multiplier)
                else:
                    sl_price = current_price * (1.0 - self.params.stop_loss_pct)
                
                # Calculate TP Price
                tp_price = current_price * (1.0 + self.params.take_profit_pct)
                
                # Calculate Position Size
                size = 100 # Default
                if self.params.use_dynamic_sizing:
                    # Use RiskManager logic (re-implemented here or imported)
                    # Risk Amount = Equity * Risk%
                    # Risk Per Share = |Entry - SL|
                    equity = self.broker.getvalue()
                    risk_amount = equity * self.params.risk_per_trade_pct
                    risk_per_share = abs(current_price - sl_price)
                    if risk_per_share > 0:
                        size = int(risk_amount / risk_per_share)
                    else:
                        size = 0
                
                if size > 0:
                    self.log(f'BUY CREATE, {current_price:.2f} (SL: {sl_price:.2f}, TP: {tp_price:.2f}, Size: {size})')
                    # Use Bracket Order
                    self.buy_bracket(size=size, price=current_price, stopprice=sl_price, limitprice=tp_price)
            
            elif sig == 'SELL' and self.position:
                 # If we have a sell signal and are long, close position
                 # Note: Bracket orders usually handle exit, but a reverse signal might force exit
                 self.log(f'SELL CREATE (Signal), {self.datas[0].close[0]}')
                 self.close()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
