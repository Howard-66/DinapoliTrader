import unittest
import pandas as pd
import numpy as np
from src.indicators.zigzag import ZigZag

class TestZigZag(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的“V”型数据
        # 100 -> 110 -> 90 -> 120
        dates = pd.date_range('2023-01-01', periods=4)
        self.high = pd.Series([100, 110, 95, 120], index=dates)
        self.low = pd.Series([90, 105, 90, 115], index=dates)
        
    def test_calculate_simple(self):
        # 这是一个非常简化的测试，实际ZigZag依赖于偏差
        # 假设偏差很小，应该能捕捉到所有转折
        pivots = ZigZag.calculate(self.high, self.low, deviation=1.0)
        # 期望捕捉到 110 (High), 90 (Low)
        # 100 (start), 110 (high), 90 (low), 120 (end)
        
        valid_pivots = pivots.dropna()
        self.assertTrue(len(valid_pivots) >= 2)
        
    def test_get_swing_points(self):
        points = ZigZag.get_swing_points(self.high, self.low, deviation=1.0)
        self.assertTrue(len(points) >= 2)
        # 检查高低点交替
        types = [p['type'] for p in points]
        for i in range(len(types)-1):
            self.assertNotEqual(types[i], types[i+1])

if __name__ == '__main__':
    unittest.main()
