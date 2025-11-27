import pandas as pd
import numpy as np
from src.risk.manager import RiskManager
from src.indicators.basics import Indicators

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
                          sl_mode: str = 'Fixed',
                          tp_mode: str = 'Fixed',
                          stop_loss_pct: float = 0.02, 
                          take_profit_pct: float = 0.05,
                          initial_capital: float = 100000.0,
                          use_dynamic_sizing: bool = False,
                          risk_per_trade_pct: float = 0.01,
                          atr_multiplier: float = 2.0) -> dict:
        """
        Calculate trade metrics.
        """
        trades = []
        trade_log_data = []

        # For equity curve, we need to track daily portfolio value
        portfolio_value = pd.Series(initial_capital, index=self.df.index)
        current_capital = initial_capital
        
        # Pre-calculate ATR if needed
        atr_series = None
        if sl_mode == 'ATR' or use_dynamic_sizing:
            atr_series = Indicators.atr(self.df['high'], self.df['low'], self.df['close'], 14)
        
        # Filter only Buy signals for now (as PatternRecognizer mainly does Buy Double Repo)
        # Note: app.py should pass filtered signals, but we double check here if needed.
        # We assume 'signals' df has 'signal' column with 'BUY'/'SELL'
        buy_signals = self.signals[self.signals['signal'] == 'BUY']
        
        if buy_signals.empty:
            return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Avg Return': 0.0,
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Max Drawdown': 0.0,
                'Equity Curve': pd.Series(initial_capital, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index),
                'Trade Log': pd.DataFrame()
            }


        for date, row in buy_signals.iterrows():
            # Find index of signal
            if date not in self.df.index:
                continue
                
            idx = self.df.index.get_loc(date)
            # Enter on next bar
            entry_idx = idx + 1
            if entry_idx >= len(self.df):
                continue
                
            entry_date = self.df.index[entry_idx]
            entry_price = self.df['open'].iloc[entry_idx]
            
            # --- Determine Stop Loss Price ---
            sl_price = 0.0
            
            if sl_mode == 'Pattern':
                if 'pattern_sl' in row and not pd.isna(row['pattern_sl']):
                    sl_price = row['pattern_sl']
                else:
                    sl_price = entry_price * (1 - stop_loss_pct)
                    
            elif sl_mode == 'ATR' and atr_series is not None:
                current_atr = atr_series.iloc[idx]
                if np.isnan(current_atr):
                    current_atr = entry_price * 0.01 # Fallback
                sl_price = RiskManager.calculate_atr_stop_loss(entry_price, current_atr, atr_multiplier, 'BUY')
                
            else: # Fixed
                sl_price = entry_price * (1 - stop_loss_pct)
                
            # --- Determine Position Size ---
            quantity = 100 # Default
            if use_dynamic_sizing:
                quantity = RiskManager.calculate_position_size(current_capital, risk_per_trade_pct, entry_price, sl_price)
            
            # --- Determine Take Profit Price ---
            tp_price = 0.0
            
            if tp_mode == 'Pattern':
                if 'pattern_tp' in row and not pd.isna(row['pattern_tp']):
                    tp_price = row['pattern_tp']
                else:
                    tp_price = entry_price * (1 + take_profit_pct)
            else: # Fixed
                tp_price = entry_price * (1 + take_profit_pct)
            
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
                if curr_low <= sl_price:
                    exit_price = sl_price
                    final_exit_date = curr_date
                    break
                elif curr_high >= tp_price:
                    exit_price = tp_price
                    final_exit_date = curr_date
                    break
                
                exit_price = curr_close
                final_exit_date = curr_date
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_amount = (exit_price - entry_price) * quantity
            trades.append(pnl_pct)
            
            # Update Capital
            current_capital += pnl_amount
            
            # Record Trade Log
            trade_log_data.append({
                'Signal Date': date,
                'Pattern': row.get('pattern', 'Unknown'),
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Stop Loss': sl_price,
                'Take Profit': tp_price,
                'Exit Date': final_exit_date if final_exit_date else date,
                'Exit Price': exit_price,
                'Quantity': quantity,
                'PnL Amount': pnl_amount,
                'PnL %': pnl_pct,
                'Confidence': row.get('confidence', np.nan)
            })
            
            # Update Portfolio Value Curve (Simplified: Step change at exit)
            # In reality, equity changes daily. For estimation, we just add PnL at exit.
            if final_exit_date:
                # Find indices between entry and exit
                # Actually, simpler to just add pnl to all days AFTER exit
                # But we want a time series.
                # Let's just set the value from exit_date onwards to new capital
                # This is an approximation.
                mask = portfolio_value.index >= final_exit_date
                portfolio_value.loc[mask] = current_capital

        if not trades:
             return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Avg Return': 0.0,
                'Total Return': 0.0,
                'Max Drawdown': 0.0,
                'Total Return': 0.0,
                'Max Drawdown': 0.0,
                'Equity Curve': pd.Series(initial_capital, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index),
                'Trade Log': pd.DataFrame()
            }

        trades_np = np.array(trades)
        win_rate = np.mean(trades_np > 0)
        avg_return = np.mean(trades_np)
        
        # Calculate Equity Curve
        equity_curve = portfolio_value
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        
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

        # Calculate Advanced Metrics
        daily_returns = equity_curve.pct_change().dropna()
        
        # Sharpe Ratio (assuming 0 risk-free rate for simplicity)
        if daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        if downside_returns.std() != 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
            
        # Profit Factor
        gross_profit = trades_np[trades_np > 0].sum() if len(trades_np[trades_np > 0]) > 0 else 0.0
        gross_loss = abs(trades_np[trades_np < 0].sum()) if len(trades_np[trades_np < 0]) > 0 else 0.0
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0

        # Calculate Monthly Returns
        monthly_returns_series = equity_curve.resample('M').last().pct_change()
        # Handle first month (if it starts mid-month, pct_change might be NaN or relative to 0)
        # Actually, equity curve starts at initial capital.
        # pct_change() first element is NaN. We need to fill it.
        # The first month's return is (End Value / Initial Capital) - 1
        first_month_idx = monthly_returns_series.index[0]
        monthly_returns_series.iloc[0] = (equity_curve.resample('M').last().iloc[0] - initial_capital) / initial_capital
        
        # Create Pivot Table for Heatmap (Year x Month)
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns_series.index.year,
            'Month': monthly_returns_series.index.month,
            'Return': monthly_returns_series.values
        })
        monthly_returns_matrix = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        
        # Fill missing months with 0.0
        monthly_returns_matrix = monthly_returns_matrix.fillna(0.0)

        return {
            'Total Trades': len(trades),
            'Win Rate': win_rate,
            'Avg Return': avg_return,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Profit Factor': profit_factor,
            'Equity Curve': equity_curve,
            'Drawdown Curve': drawdown_curve,
            'Monthly Returns': monthly_returns_matrix,
            'Trade Log': pd.DataFrame(trade_log_data)
        }
