import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os

class SignalClassifier:
    """
    信号分类器。
    使用Random Forest判断交易信号的成功概率。
    """

    def __init__(self, model_path: str = 'models/rf_classifier.pkl'):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_path = model_path
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        训练模型。
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        
        print(f"Model Trained. Accuracy: {acc:.2f}, Precision: {prec:.2f}")
        
        # Save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率。
        """
        if not self.is_trained:
            # Try load
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            else:
                raise Exception("Model not trained or found.")
                
        return self.model.predict_proba(X)[:, 1] # Return probability of class 1 (Success)
