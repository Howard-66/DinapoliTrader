import unittest
import unittest.mock
import pandas as pd
import numpy as np
from src.strategies.patterns import PatternRecognizer

class TestPatternRecognizer(unittest.TestCase):
    def setUp(self):
        # 构造一个Double Repo形态的数据
        # 价格在3x3下方 -> 突破 -> 回落 -> 再次突破
        # 3x3 DMA大概是滞后的，所以我们直接构造价格和DMA的关系
        
        dates = pd.date_range('2023-01-01', periods=50)
        self.df = pd.DataFrame(index=dates)
        
        # 构造价格序列
        # 0-10: 下跌
        # 11: 突破 (Up Cross 1)
        # 12-15: 回落 (Down Cross)
        # 16: 再次突破 (Up Cross 2)
        
        price = np.linspace(100, 80, 10).tolist() # 0-9
        price.append(85) # 10: Up (assume DMA is around 82)
        price.extend([81, 80, 79]) # 11-13: Down
        price.append(86) # 14: Up again
        price.extend([87] * (50 - len(price)))
        
        self.df['close'] = price
        self.df['high'] = price # Simplify
        self.df['low'] = price
        self.df['open'] = price
        
        # Mock Indicators to control DMA values exactly
        # 我们不能轻易mock内部的Indicators调用，除非patch
        # 但我们可以构造足够长的数据让DMA自然计算出来，或者patch PatternRecognizer.dma_3x3
        
    @unittest.mock.patch('src.indicators.basics.Indicators.displaced_ma')
    def test_detect_double_repo(self, mock_dma):
        # Mock DMA 3x3 to be a constant line that price crosses
        # Price crosses 82
        mock_dma.return_value = pd.Series([82] * 50, index=self.df.index)
        
        # 我们还需要mock 25x5 DMA来满足过滤条件 (Price < 25x5 for Buy)
        # 让 25x5 = 100 (远高于价格)
        
        # PatternRecognizer __init__ calls displaced_ma 3 times
        # 1. 3x3
        # 2. 7x5
        # 3. 25x5
        
        mock_dma.side_effect = [
            pd.Series([82] * 50, index=self.df.index), # 3x3
            pd.Series([90] * 50, index=self.df.index), # 7x5
            pd.Series([100] * 50, index=self.df.index) # 25x5
        ]
        
        recognizer = PatternRecognizer(self.df)
        signals = recognizer.detect_double_repo(lookback=5)
        
        # 我们期望在第14个点左右检测到信号 (index 14 is the second cross up)
        # 0-9: < 82
        # 10: 85 > 82 (Up Cross 1)
        # 11: 81 < 82 (Down Cross)
        # 12: 80 < 82
        # 13: 79 < 82
        # 14: 86 > 82 (Up Cross 2) -> SIGNAL
        
        self.assertEqual(signals['signal'].iloc[14], 'BUY')
        self.assertEqual(signals['pattern'].iloc[14], 'Double Repo')

if __name__ == '__main__':
    unittest.main()
