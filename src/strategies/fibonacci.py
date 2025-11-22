import pandas as pd
from typing import List, Dict, Optional

class FibonacciEngine:
    """
    斐波那契计算引擎。
    负责计算回撤（Retracements）和扩展（Expansions）。
    """

    @staticmethod
    def calculate_retracements(start_price: float, end_price: float) -> Dict[str, float]:
        """
        计算斐波那契回撤水平。
        DiNapoli主要关注 0.382 和 0.618。
        
        Args:
            start_price (float): 波动起始价格 (Focus Number)
            end_price (float): 波动结束价格 (Reaction Point)
            
        Returns:
            dict: {'0.382': price, '0.618': price}
        """
        diff = end_price - start_price
        # 如果是上升趋势（Start < End），回撤是向下的，所以减去 diff * ratio
        # 如果是下降趋势（Start > End），回撤是向上的，所以加上 |diff| * ratio
        # 统一公式：End - Diff * Ratio
        
        # 修正逻辑：
        # 上升趋势：100 -> 200. Diff = 100. Retracement 0.382 = 200 - 100*0.382 = 161.8
        # 下降趋势：200 -> 100. Diff = -100. Retracement 0.382 = 100 - (-100)*0.382 = 138.2
        
        return {
            '0.382': end_price - diff * 0.382,
            '0.618': end_price - diff * 0.618,
            '0.5': end_price - diff * 0.5 # Optional
        }

    @staticmethod
    def calculate_expansions(a: float, b: float, c: float) -> Dict[str, float]:
        """
        计算斐波那契扩展目标（COP, OP, XOP）。
        基于A-B-C波动。
        A: Start
        B: End of impulse
        C: End of retracement
        
        Target = C + (B-A) * Ratio
        
        Ratios:
        COP (Contracted Objective Point): 0.618
        OP (Objective Point): 1.000
        XOP (Expanded Objective Point): 1.618
        """
        impulse = b - a
        
        return {
            'COP': c + impulse * 0.618,
            'OP': c + impulse * 1.0,
            'XOP': c + impulse * 1.618
        }
