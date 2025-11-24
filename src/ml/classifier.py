import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os

from src.ml.features import FeatureExtractor

class SignalClassifier:
    """
    Signal Classifier using Random Forest.
    """

    def __init__(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(os.path.dirname(__file__), 'signal_classifier.joblib')
        self.model = None
        self.is_trained = False
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            except Exception as e:
                print(f"Failed to load model: {e}")

    def predict_proba(self, df: pd.DataFrame, idx: int) -> float:
        """
        Predict probability of success for a signal at a given index.
        """
        if not self.is_trained or self.model is None:
            return 0.5 # Default neutral probability
            
        extractor = FeatureExtractor(df)
        features = extractor.get_features(idx)
        
        if not features:
            return 0.5
            
        # Convert to DataFrame (single row)
        X = pd.DataFrame([features])
        # Ensure column order matches training (handled by FeatureExtractor usually, but good to be safe)
        # Here we rely on FeatureExtractor returning consistent dict keys.
        
        try:
            # Check if model expects feature names (sklearn 1.0+)
            if hasattr(self.model, 'feature_names_in_'):
                # Model has feature names, pass DataFrame
                pass
            else:
                # Legacy model or trained on numpy array: convert to array to avoid warning
                X = X.values
                
            prob = self.model.predict_proba(X)[0][1] # Probability of class 1
            return prob
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5

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

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        """
        if not self.is_trained or self.model is None:
            return pd.DataFrame()
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Try to get feature names
                if hasattr(self.model, 'feature_names_in_'):
                    names = self.model.feature_names_in_
                else:
                    # Fallback if names not stored
                    names = [f"Feature {i}" for i in range(len(importances))]
                    
                df = pd.DataFrame({
                    'Feature': names,
                    'Importance': importances
                })
                return df.sort_values('Importance', ascending=False)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return pd.DataFrame()


