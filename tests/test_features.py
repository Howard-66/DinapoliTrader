import unittest
import pandas as pd
import numpy as np
from src.ml.features import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_generate_features(self):
        engineer = FeatureEngineer(self.df)
        features = engineer.generate_features()
        
        self.assertIn('rsi', features.columns)
        self.assertIn('dist_dma3', features.columns)
        self.assertIn('dist_dma25', features.columns)
        # Check that NaN rows are dropped
        self.assertTrue(len(features) < 100)

    def test_create_labels(self):
        engineer = FeatureEngineer(self.df)
        labels = engineer.create_labels(horizon=5, threshold=0.0)
        self.assertEqual(len(labels), 100)
        self.assertTrue(set(labels.unique()).issubset({0, 1}))

if __name__ == '__main__':
    unittest.main()
