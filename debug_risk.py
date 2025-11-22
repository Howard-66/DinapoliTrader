import pandas as pd
import numpy as np
from src.risk.manager import RiskManager
from src.indicators.basics import Indicators

# Recreate test scenario
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
df = pd.DataFrame(index=dates)
df['close'] = 100.0
df['high'] = 102.0
df['low'] = 98.0
df['open'] = 100.0
df['volume'] = 1000

# Calculate ATR
atr_series = Indicators.atr(df['high'], df['low'], df['close'], 14)
print(f"ATR at index 10: {atr_series.iloc[10]}")

# Calculate SL and Qty
entry = 100.0
atr = atr_series.iloc[10]
if np.isnan(atr):
    print("ATR is NaN")
    atr = 4.0 # Fallback

sl = RiskManager.calculate_atr_stop_loss(entry, atr, 2.0, 'BUY')
print(f"SL Price: {sl}")

equity = 100000.0
risk_pct = 0.01
qty = RiskManager.calculate_position_size(equity, risk_pct, entry, sl)
print(f"Quantity: {qty}")
