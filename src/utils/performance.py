import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    """
    Calculates performance metrics for trading signals.
    Assumes a simple execution model for estimation:
    - Enter on Signal Close
    - Hold for 'holding_period' bars OR until Stop Loss / Take Profit
    """

    def __init__(self, df: pd.DataFrame, signals: pd.DataFrame):
        self.df = df
        self.signals = signals

    def calculate_metrics(self, 
                          holding_period: int = 5, 
                          stop_loss_pct: float = 0.02, 
                          take_profit_pct: float = 0.05) -> dict:
        """
        Calculate trade metrics.
        
        Returns:
            dict: {
                'Total Trades': int,
                'Win Rate': float,
                'Avg Return': float,
                'Total Return': float,
                'Annualized Return': float,
                'Max Drawdown': float,
                'Equity Curve': pd.Series,
                'Drawdown Curve': pd.Series,
                'Trade Log': pd.DataFrame
            }
        """
        trades = []
        trade_log_data = []
        trade_returns = pd.Series(0.0, index=self.df.index)
        
        # Filter only Buy signals for now (as PatternRecognizer mainly does Buy Double Repo)
        buy_signals = self.signals[self.signals['signal'] == 'BUY']
        
        if buy_signals.empty:
            return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Avg Return': 0.0,
                'Total Return': 0.0,
                'Max Drawdown': 0.0,
                'Equity Curve': pd.Series(1.0, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index)
            }

        for date, row in buy_signals.iterrows():
            # Find index of signal
            if date not in self.df.index:
                continue
                
            idx = self.df.index.get_loc(date)
            if idx + 1 >= len(self.df):
                continue
                
            entry_price = self.df['close'].iloc[idx]
            
            # Simulate trade
            exit_price = entry_price
            status = 'HOLD'
            
            # If still holding after period, exit at close
            if status == 'HOLD':
                exit_idx = idx + min(holding_period, len(self.df) - idx - 1)
                exit_price = self.df['close'].iloc[exit_idx]
                exit_date = self.df.index[exit_idx]
            else:
                # Find the date corresponding to the exit
                # Since we broke the loop at i, the exit index is idx + i
                # But we need to be careful if i was the last step
                # Re-calculate exit index based on loop break
                # Actually, simpler: we know the price, but we need the date.
                # In the loop, we accessed idx + i.
                # Let's store exit_date in the loop.
                pass

            # Re-running logic to get exit_date correctly is messy.
            # Let's just use the index we found.
            # We need to capture 'i' from the loop or calculate it.
            # Let's refactor the loop slightly above to capture exit_date.
            pass 
            
            # REFACTORING THE LOOP LOGIC INLINE HERE FOR CLARITY
            # (The previous loop block is being replaced by this entire chunk)
            
            final_exit_date = None
            
            for i in range(1, holding_period + 1):
                current_idx = idx + i
                if current_idx >= len(self.df):
                    exit_price = self.df['close'].iloc[-1]
                    final_exit_date = self.df.index[-1]
                    break
                
                curr_high = self.df['high'].iloc[current_idx]
                curr_low = self.df['low'].iloc[current_idx]
                curr_close = self.df['close'].iloc[current_idx]
                curr_date = self.df.index[current_idx]
                
                # Check SL/TP
                if curr_low <= entry_price * (1 - stop_loss_pct):
                    exit_price = entry_price * (1 - stop_loss_pct)
                    final_exit_date = curr_date
                    break
                elif curr_high >= entry_price * (1 + take_profit_pct):
                    exit_price = entry_price * (1 + take_profit_pct)
                    final_exit_date = curr_date
                    break
                
                exit_price = curr_close
                final_exit_date = curr_date
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_amount = (exit_price - entry_price) * 100 # Assuming 100 shares
            trades.append(pnl_pct)
            
            # Record Trade Log
            trade_log_data.append({
                'Signal Date': date,
                'Pattern': row.get('pattern', 'Unknown'),
                'Entry Date': date, # Assuming entry on signal close
                'Entry Price': entry_price,
                'Exit Date': final_exit_date if final_exit_date else date,
                'Exit Price': exit_price,
                'Quantity': 100,
                'PnL Amount': pnl_amount,
                'PnL %': pnl_pct
            })
            
            # Add return to the exit date
            if final_exit_date:
                trade_returns[final_exit_date] += pnl_pct

        if not trades:
             return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Avg Return': 0.0,
                'Total Return': 0.0,
                'Max Drawdown': 0.0,
                'Equity Curve': pd.Series(1.0, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index),
                'Trade Log': pd.DataFrame()
            }

        trades_np = np.array(trades)
        win_rate = np.mean(trades_np > 0)
        avg_return = np.mean(trades_np)
        
        # Calculate Equity Curve
        equity_curve = (1 + trade_returns).cumprod()
        total_return = equity_curve.iloc[-1] - 1.0
        
        # Calculate Drawdown Curve
        peak = equity_curve.cummax()
        drawdown_curve = (equity_curve - peak) / peak
        max_drawdown = drawdown_curve.min() # Most negative value

        # Calculate Annualized Return
        days = (self.df.index[-1] - self.df.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        return {
            'Total Trades': len(trades),
            'Win Rate': win_rate,
            'Avg Return': avg_return,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Max Drawdown': max_drawdown,
            'Equity Curve': equity_curve,
            'Drawdown Curve': drawdown_curve,
            'Trade Log': pd.DataFrame(trade_log_data)
        }
