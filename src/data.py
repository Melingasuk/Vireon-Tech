"""
Data loading and preprocessing module for soil health prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


class DataLoader:
    """Load and preprocess soil health data"""
    
    def __init__(self, data_path: str = "data/raw/soil_data.csv"):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        return pd.read_csv(self.data_path)
    
    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data: handle missing values, encode categorical variables, scale features
        
        Args:
            df: Raw dataframe with columns [N, P, K, pH, Rainfall, Temperature, Humidity, Crop_Type]
        
        Returns:
            X: Features (N, P, K, pH, Rainfall, Temperature, Humidity)
            y: Target (Crop_Type)
        """
        df = df.copy()
        
        # Check required columns
        required_cols = ['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature', 'Humidity', 'Crop_Type']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        X = df[['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature', 'Humidity']]
        y = df['Crop_Type']
        
        # Scale numeric features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Encode target variable
        self.label_encoders['Crop_Type'] = LabelEncoder()
        y = pd.Series(
            self.label_encoders['Crop_Type'].fit_transform(y),
            name='Crop_Type'
        )
        
        return X, y
    
    def get_feature_names(self) -> list[str]:
        """Get feature names in order"""
        return ['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature', 'Humidity']
