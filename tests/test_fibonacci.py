import unittest
from src.strategies.fibonacci import FibonacciEngine

class TestFibonacciEngine(unittest.TestCase):
    
    def test_retracements_uptrend(self):
        # 100 -> 200
        levels = FibonacciEngine.calculate_retracements(100, 200)
        # 0.382 retracement of 100 move is 38.2. Price = 200 - 38.2 = 161.8
        self.assertAlmostEqual(levels['0.382'], 161.8)
        # 0.618 retracement of 100 move is 61.8. Price = 200 - 61.8 = 138.2
        self.assertAlmostEqual(levels['0.618'], 138.2)

    def test_retracements_downtrend(self):
        # 200 -> 100
        levels = FibonacciEngine.calculate_retracements(200, 100)
        # Diff = -100.
        # 0.382: 100 - (-100 * 0.382) = 100 + 38.2 = 138.2
        self.assertAlmostEqual(levels['0.382'], 138.2)
        
    def test_expansions_uptrend(self):
        # A=100, B=200, C=150
        # Impulse = 100.
        # COP = 150 + 100*0.618 = 211.8
        # OP = 150 + 100*1.0 = 250
        # XOP = 150 + 100*1.618 = 311.8
        levels = FibonacciEngine.calculate_expansions(100, 200, 150)
        self.assertAlmostEqual(levels['COP'], 211.8)
        self.assertAlmostEqual(levels['OP'], 250.0)
        self.assertAlmostEqual(levels['XOP'], 311.8)

if __name__ == '__main__':
    unittest.main()
