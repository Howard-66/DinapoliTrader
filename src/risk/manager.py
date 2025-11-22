import pandas as pd
import numpy as np

class RiskManager:
    """
    Manages risk by calculating position sizes and dynamic stop losses.
    """
    
    @staticmethod
    def calculate_atr_stop_loss(entry_price: float, atr: float, multiplier: float = 2.0, direction: str = 'BUY') -> float:
        """
        Calculate Stop Loss price based on ATR.
        """
        if direction == 'BUY':
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)
            
    @staticmethod
    def calculate_position_size(account_equity: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float) -> int:
        """
        Calculate position size (number of shares) based on risk percentage.
        
        Risk Amount = Equity * Risk%
        Risk Per Share = |Entry - SL|
        Shares = Risk Amount / Risk Per Share
        """
        if account_equity <= 0:
            return 0
            
        risk_amount = account_equity * risk_per_trade_pct
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
            
        shares = int(risk_amount / risk_per_share)
        
        # Ensure we can afford it (margin check simplified)
        # Assuming 1:1 leverage for now
        cost = shares * entry_price
        if cost > account_equity:
            shares = int(account_equity / entry_price)
            
        return shares
