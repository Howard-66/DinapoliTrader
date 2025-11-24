import pandas as pd
import itertools
from sklearn.model_selection import TimeSeriesSplit
from src.utils.performance import PerformanceAnalyzer

class StrategyOptimizer:
    """
    Optimizes strategy parameters using Grid Search.
    """
    
    def __init__(self, df: pd.DataFrame, signals: pd.DataFrame):
        self.df = df
        self.signals = signals
        self.analyzer = PerformanceAnalyzer(df, signals)
        
    def grid_search(self, param_grid: dict, df: pd.DataFrame = None, signals: pd.DataFrame = None) -> pd.DataFrame:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid (dict): Dictionary of parameters and their ranges.
            df (pd.DataFrame): Optional dataframe to run on.
            signals (pd.DataFrame): Optional signals to run on.
        """
        target_df = df if df is not None else self.df
        target_signals = signals if signals is not None else self.signals
        analyzer = PerformanceAnalyzer(target_df, target_signals)
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        results = []
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Extract specific parameters
            hp = params.get('holding_period', 5)
            sl = params.get('stop_loss', 0.02)
            tp = params.get('take_profit', 0.05)
            
            # Calculate metrics
            metrics = analyzer.calculate_metrics(holding_period=hp, stop_loss_pct=sl, take_profit_pct=tp)
            
            # Store result
            result_row = params.copy()
            result_row['Total Return'] = metrics['Total Return']
            result_row['Win Rate'] = metrics['Win Rate']
            result_row['Total Trades'] = metrics['Total Trades']
            result_row['Max Drawdown'] = metrics['Max Drawdown']
            result_row['Ann. Return'] = metrics.get('Annualized Return', 0.0)
            
            results.append(result_row)
            
        results_df = pd.DataFrame(results)
        
        # Sort by Total Return descending
        if not results_df.empty:
            results_df = results_df.sort_values(by='Total Return', ascending=False)
            
        return results_df

    def walk_forward_analysis(self, param_grid: dict, n_splits: int = 5) -> pd.DataFrame:
        """
        Perform Walk-Forward Analysis.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for fold, (train_index, test_index) in enumerate(tscv.split(self.df)):
            # Create Train/Test splits
            train_df = self.df.iloc[train_index]
            test_df = self.df.iloc[test_index]
            
            # Slice signals (ensure alignment)
            train_signals = self.signals.loc[train_df.index]
            test_signals = self.signals.loc[test_df.index]
            
            # 1. Optimize on Train
            train_results = self.grid_search(param_grid, df=train_df, signals=train_signals)
            
            if train_results.empty:
                continue
                
            # Get best params (by Total Return)
            best_params = train_results.iloc[0].to_dict()
            
            # 2. Test on Test (OOS)
            # Extract params
            hp = int(best_params.get('holding_period', 5))
            sl = float(best_params.get('stop_loss', 0.02))
            tp = float(best_params.get('take_profit', 0.05))
            
            analyzer = PerformanceAnalyzer(test_df, test_signals)
            metrics = analyzer.calculate_metrics(holding_period=hp, stop_loss_pct=sl, take_profit_pct=tp)
            
            # Record Result
            res = {
                'Fold': fold + 1,
                'Train Start': train_df.index[0].date(),
                'Train End': train_df.index[-1].date(),
                'Test Start': test_df.index[0].date(),
                'Test End': test_df.index[-1].date(),
                'Best Params': f"HP:{hp}, SL:{sl:.2f}, TP:{tp:.2f}",
                'OOS Return': metrics['Total Return'],
                'OOS Sharpe': metrics.get('Sharpe Ratio', 0.0),
                'OOS Max DD': metrics['Max Drawdown']
            }
            results.append(res)
            
        return pd.DataFrame(results)
