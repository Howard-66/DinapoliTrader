import backtrader as bt
import pandas as pd
from src.backtest.strategy import DiNapoliStrategyWithSignals
from src.strategies.patterns import PatternRecognizer
from src.data.feed import DataFeed

class BacktestEngine:
    """
    回测引擎。
    """

    def __init__(self, symbol: str, start_date: str, end_date: str, initial_cash: float = 100000.0):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.cerebro = bt.Cerebro()

    def run(self):
        # 1. 获取数据
        feed = DataFeed()
        df = feed.fetch_data(self.symbol, self.start_date, self.end_date)
        
        if df.empty:
            print("No data found.")
            return

        # 2. 计算信号 (预计算)
        recognizer = PatternRecognizer(df)
        signals = recognizer.detect_double_repo() # Returns DF with 'signal' column
        
        # 3. 加载数据到 Cerebro
        # Backtrader expects 'Open', 'High', 'Low', 'Close', 'Volume'
        # Our df has lowercase. Rename for BT feed.
        bt_df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        data = bt.feeds.PandasData(dataname=bt_df)
        self.cerebro.adddata(data)

        # 4. 添加策略
        self.cerebro.addstrategy(DiNapoliStrategyWithSignals, signals=signals)

        # 5. 设置资金
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=0.001)

        # 6. 运行
        print(f'Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        self.cerebro.run()
        print(f'Final Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        
        # 7. 绘图 (Optional, BT's plot is matplotlib based)
        # self.cerebro.plot()
