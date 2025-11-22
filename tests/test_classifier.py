import unittest
import pandas as pd
import numpy as np
import os
from src.ml.classifier import SignalClassifier

class TestSignalClassifier(unittest.TestCase):
    def setUp(self):
        # Create synthetic features and labels
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        self.y = pd.Series(np.random.randint(0, 2, 100))
        self.model_path = 'tests/temp_model.pkl'

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_train_and_predict(self):
        clf = SignalClassifier(model_path=self.model_path)
        clf.train(self.X, self.y)
        
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(clf.is_trained)
        
        probs = clf.predict_proba(self.X.iloc[:5])
        self.assertEqual(len(probs), 5)
        self.assertTrue(all(0 <= p <= 1 for p in probs))

if __name__ == '__main__':
    unittest.main()
