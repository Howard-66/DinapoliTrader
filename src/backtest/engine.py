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

    def run(self, 
            strategies: list = ['Double Repo'], 
            stop_loss_pct: float = 0.02,
            take_profit_pct: float = 0.05,
            use_atr_sl: bool = False,
            atr_multiplier: float = 2.0,
            risk_per_trade_pct: float = 0.01,
            use_dynamic_sizing: bool = False):
        
        # 1. 获取数据
        feed = DataFeed()
        df = feed.fetch_data(self.symbol, self.start_date, self.end_date)
        
        if df.empty:
            print("No data found.")
            return

        # 2. 计算信号 (预计算)
        recognizer = PatternRecognizer(df)
        signals = pd.DataFrame(index=df.index, columns=['signal', 'pattern'])
        
        if 'Double Repo' in strategies:
            signals_dr = recognizer.detect_double_repo()
            mask_dr = signals_dr['signal'].notna()
            signals.loc[mask_dr] = signals_dr.loc[mask_dr]
            
        if 'Single Penetration' in strategies:
            signals_sp = recognizer.detect_single_penetration()
            # Merge logic: Prioritize existing signals (DR) or overwrite?
            # Let's fill gaps
            mask_fill = (signals['signal'].isna()) & (signals_sp['signal'].notna())
            signals.loc[mask_fill] = signals_sp.loc[mask_fill]
        
        # 3. 加载数据到 Cerebro
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
        self.cerebro.addstrategy(DiNapoliStrategyWithSignals, 
                                 signals=signals,
                                 stop_loss_pct=stop_loss_pct,
                                 take_profit_pct=take_profit_pct,
                                 use_atr_sl=use_atr_sl,
                                 atr_multiplier=atr_multiplier,
                                 risk_per_trade_pct=risk_per_trade_pct,
                                 initial_capital=self.initial_cash,
                                 use_dynamic_sizing=use_dynamic_sizing)

        # 5. 设置资金
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=0.001)
        self.cerebro.broker.set_slippage_perc(perc=0.0005) # 0.05% slippage

        # 6. 运行
        print(f'Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        self.cerebro.run()
        print(f'Final Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        
        # 7. 绘图 (Optional, BT's plot is matplotlib based)
        # self.cerebro.plot()
