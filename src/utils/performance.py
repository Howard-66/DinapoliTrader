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
        
        # Track per-strategy statistics
        strategy_stats = {}

        # For equity curve, we need to track daily portfolio value
        # Realized Equity: Only includes closed trades
        # Floating Equity: Includes unrealized P&L from open positions
        realized_equity = pd.Series(initial_capital, index=self.df.index)
        floating_equity = pd.Series(initial_capital, index=self.df.index)
        current_capital = initial_capital
        
        # Track open positions for floating P&L calculation
        open_positions = []  # List of (entry_idx, entry_price, quantity, pattern)
        
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
                'Floating Equity Curve': pd.Series(initial_capital, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index),
                'Trade Log': pd.DataFrame(),
                'Strategy Breakdown': pd.DataFrame()
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
            
            # Add to open positions
            pattern = row.get('pattern', 'Unknown')
            open_positions.append({
                'entry_idx': entry_idx,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'quantity': quantity,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'pattern': pattern,
                'signal_date': date,
                'confidence': row.get('confidence', np.nan)
            })
            
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
            
            # Update Capital (Realized)
            current_capital += pnl_amount
            
            # Remove from open positions
            open_positions = [pos for pos in open_positions if pos['entry_date'] != entry_date]
            
            # Track per-strategy statistics
            if pattern not in strategy_stats:
                strategy_stats[pattern] = {
                    'trades': 0,
                    'pnl_amount': 0.0,
                    'wins': 0
                }
            strategy_stats[pattern]['trades'] += 1
            strategy_stats[pattern]['pnl_amount'] += pnl_amount
            if pnl_pct > 0:
                strategy_stats[pattern]['wins'] += 1
            
            # Record Trade Log
            trade_log_data.append({
                'Signal Date': date,
                'Pattern': pattern,
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
            
            # Update Realized Equity Curve (Step change at exit)
            if final_exit_date:
                mask = realized_equity.index >= final_exit_date
                realized_equity.loc[mask] = current_capital
        
        # Calculate Floating Equity Curve (daily)
        # Build a list of all trades with their entry and exit dates
        trade_positions = []
        for trade in trade_log_data:
            trade_positions.append({
                'entry_date': trade['Entry Date'],
                'exit_date': trade['Exit Date'],
                'entry_price': trade['Entry Price'],
                'quantity': trade['Quantity']
            })
        
        # For each day, calculate unrealized P&L from positions that are open
        for i, current_date in enumerate(self.df.index):
            unrealized_pnl = 0.0
            current_price = self.df['close'].iloc[i]
            
            # Check all trades to see if they're open on this date
            for trade in trade_positions:
                if trade['entry_date'] <= current_date < trade['exit_date']:
                    # Position is open on this date
                    unrealized_pnl += (current_price - trade['entry_price']) * trade['quantity']
            
            # Floating equity = realized capital at this point + unrealized P&L
            floating_equity.iloc[i] = realized_equity.iloc[i] + unrealized_pnl

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
                'Floating Equity Curve': pd.Series(initial_capital, index=self.df.index),
                'Drawdown Curve': pd.Series(0.0, index=self.df.index),
                'Trade Log': pd.DataFrame(),
                'Strategy Breakdown': pd.DataFrame()
            }

        trades_np = np.array(trades)
        win_rate = np.mean(trades_np > 0)
        avg_return = np.mean(trades_np)
        
        # Calculate Equity Curves
        equity_curve = realized_equity  # Realized equity (only closed trades)
        floating_equity_curve = floating_equity  # Floating equity (includes unrealized P&L)
        
        # Use floating equity for total return calculation (more accurate)
        total_return = (floating_equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        # Calculate Drawdown Curve (based on floating equity)
        peak = floating_equity_curve.cummax()
        drawdown_curve = (floating_equity_curve - peak) / peak
        max_drawdown = drawdown_curve.min() # Most negative value

        # Calculate Annualized Return
        days = (self.df.index[-1] - self.df.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        # Calculate Advanced Metrics (based on floating equity)
        daily_returns = floating_equity_curve.pct_change().dropna()
        
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

        # Calculate Monthly Returns (based on floating equity)
        monthly_returns_series = floating_equity_curve.resample('ME').last().pct_change()
        # Handle first month
        monthly_returns_series.iloc[0] = (floating_equity_curve.resample('ME').last().iloc[0] - initial_capital) / initial_capital
        
        # Create Pivot Table for Heatmap (Year x Month)
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns_series.index.year,
            'Month': monthly_returns_series.index.month,
            'Return': monthly_returns_series.values
        })
        monthly_returns_matrix = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        
        # Fill missing months with 0.0
        monthly_returns_matrix = monthly_returns_matrix.fillna(0.0)
        
        # Calculate Strategy Breakdown
        strategy_breakdown_data = []
        total_pnl = sum(stats['pnl_amount'] for stats in strategy_stats.values())
        
        for pattern, stats in strategy_stats.items():
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0.0
            # Calculate contribution percentage
            # Handle case where total PnL is zero or negative
            if total_pnl > 0:
                contribution_pct = (stats['pnl_amount'] / total_pnl) * 100
            elif total_pnl < 0:
                # For negative total PnL, losses contribute positively to the negative total
                contribution_pct = (stats['pnl_amount'] / total_pnl) * 100
            else:
                contribution_pct = 0.0
            
            strategy_breakdown_data.append({
                'Strategy': pattern,
                'Trades': stats['trades'],
                'Total PnL': stats['pnl_amount'],
                'Win Rate': win_rate,
                'Contribution %': contribution_pct
            })
        
        strategy_breakdown_df = pd.DataFrame(strategy_breakdown_data)
        # Sort by contribution percentage descending
        if not strategy_breakdown_df.empty:
            strategy_breakdown_df = strategy_breakdown_df.sort_values('Contribution %', ascending=False)

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
            'Floating Equity Curve': floating_equity_curve,
            'Drawdown Curve': drawdown_curve,
            'Monthly Returns': monthly_returns_matrix,
            'Trade Log': pd.DataFrame(trade_log_data),
            'Strategy Breakdown': strategy_breakdown_df
        }
