import pandas as pd
import itertools
from src.utils.performance import PerformanceAnalyzer

class StrategyOptimizer:
    """
    Optimizes strategy parameters using Grid Search.
    """
    
    def __init__(self, df: pd.DataFrame, signals: pd.DataFrame):
        self.df = df
        self.signals = signals
        self.analyzer = PerformanceAnalyzer(df, signals)
        
    def grid_search(self, param_grid: dict) -> pd.DataFrame:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid (dict): Dictionary of parameters and their ranges.
                e.g., {'holding_period': [3, 5, 8], 'stop_loss': [0.01, 0.02]}
                
        Returns:
            pd.DataFrame: Results of optimization, sorted by Total Return.
        """
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
            metrics = self.analyzer.calculate_metrics(holding_period=hp, stop_loss_pct=sl, take_profit_pct=tp)
            
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
