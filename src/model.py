"""
Model training and evaluation for crop recommendation
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import pandas as pd


class CropRecommender:
    """
    Machine learning model for recommending crops based on soil nutrients
    """
    
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_importance = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the model and return train/test split
        
        Args:
            X: Features (N, P, K, pH, Rainfall, Temperature, Humidity)
            y: Target (Crop_Type)
            test_size: Proportion of data to use for testing
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, X: pd.DataFrame):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def save(self, path: str = "models/crop_recommender.pkl"):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str = "models/crop_recommender.pkl"):
        """Load trained model"""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")
