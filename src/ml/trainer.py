import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import json
import os
from datetime import datetime
from src.ml.features import FeatureExtractor

class ModelTrainer:
    """
    Trains a ML model to filter signals.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model_path = os.path.join(os.path.dirname(__file__), 'signal_classifier.joblib')
        self.metadata_path = os.path.join(os.path.dirname(__file__), 'signal_classifier_metadata.json')
        
    def save_metadata(self, metadata: dict):
        """
        Save training metadata to JSON file.
        """
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save metadata: {e}")
        
    def generate_dataset(self, df: pd.DataFrame, signals: pd.DataFrame, 
                         holding_period: int = 5, 
                         stop_loss_pct: float = 0.02, 
                         take_profit_pct: float = 0.05):
        """
        Generate X (features) and y (labels) from signals.
        Label 1 if trade hits TP before SL, else 0.
        """
        extractor = FeatureExtractor(df)
        features_df = extractor.get_all_features()
        
        X = []
        y = []
        
        # Filter for BUY signals only for now
        buy_signals = signals[signals['signal'] == 'BUY']
        
        for date, row in buy_signals.iterrows():
            if date not in df.index:
                continue
                
            idx_loc = df.index.get_loc(date)
            if idx_loc + 1 >= len(df):
                continue
                
            # Get Features
            feature_row = features_df.loc[date]
            X.append(feature_row.values)
            
            # Determine Label (Simulate Trade)
            entry_price = df['close'].iloc[idx_loc]
            label = 0 # Default Loss
            
            for i in range(1, holding_period + 1):
                curr_idx = idx_loc + i
                if curr_idx >= len(df):
                    break
                    
                curr_high = df['high'].iloc[curr_idx]
                curr_low = df['low'].iloc[curr_idx]
                
                # Check TP first (optimistic? or SL first? Let's check SL first for conservatism)
                if curr_low <= entry_price * (1 - stop_loss_pct):
                    label = 0
                    break
                elif curr_high >= entry_price * (1 + take_profit_pct):
                    label = 1
                    break
                
                # If held to end, check return
                if i == holding_period:
                    exit_price = df['close'].iloc[curr_idx]
                    if exit_price > entry_price: # Simple positive return
                        label = 1
                    else:
                        label = 0
                        
            y.append(label)
            
        return pd.DataFrame(X, columns=features_df.columns), np.array(y), features_df.columns
        
    def train(self, df: pd.DataFrame, signals: pd.DataFrame, 
              symbol: str = None, start_date: str = None, end_date: str = None,
              holding_period: int = 5, stop_loss_pct: float = 0.02, 
              take_profit_pct: float = 0.05):
        """
        Train the model and save it with metadata.
        """
        X, y, feature_names = self.generate_dataset(df, signals, holding_period, 
                                                     stop_loss_pct, take_profit_pct)
        
        if len(X) < 10:
            return {"status": "error", "message": "Not enough signals to train (need > 10)"}
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
        # Prepare and save metadata
        metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_samples": len(X),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_parameters": {
                "holding_period": holding_period,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct
            },
            "model_hyperparameters": {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "random_state": self.model.random_state
            },
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision
            },
            "feature_count": len(feature_names),
            "positive_samples": int(y.sum()),
            "negative_samples": int(len(y) - y.sum())
        }
        
        self.save_metadata(metadata)
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "precision": precision,
            "samples": len(X)
        }
