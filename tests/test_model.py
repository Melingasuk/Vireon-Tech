"""
Unit tests for the crop recommendation model
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import CropRecommender
from data import DataLoader


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'N': np.random.randint(20, 100, n_samples),
        'P': np.random.randint(5, 50, n_samples),
        'K': np.random.randint(10, 60, n_samples),
        'pH': np.random.uniform(4.5, 8.5, n_samples),
        'Rainfall': np.random.randint(500, 2000, n_samples),
        'Temperature': np.random.uniform(15, 35, n_samples),
        'Humidity': np.random.randint(20, 95, n_samples),
        'Crop_Type': np.random.choice(['Rice', 'Wheat', 'Corn', 'Cotton'], n_samples)
    }
    return pd.DataFrame(data)


def test_model_initialization():
    """Test model can be initialized"""
    model = CropRecommender()
    assert model.is_trained == False
    assert model.model is not None


def test_model_training(sample_data):
    """Test model training pipeline"""
    loader = DataLoader()
    X, y = loader.preprocess(sample_data)
    
    model = CropRecommender()
    X_train, X_test, y_train, y_test = model.train(X, y, test_size=0.2)
    
    assert model.is_trained == True
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == 7  # 7 features


def test_model_prediction(sample_data):
    """Test model can make predictions"""
    loader = DataLoader()
    X, y = loader.preprocess(sample_data)
    
    model = CropRecommender()
    model.train(X, y)
    
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert all(isinstance(p, (int, np.integer)) for p in predictions)


def test_prediction_before_training():
    """Test that prediction fails before training"""
    model = CropRecommender()
    X = pd.DataFrame(np.random.rand(10, 7))
    
    with pytest.raises(ValueError):
        model.predict(X)


def test_data_preprocessing(sample_data):
    """Test data preprocessing"""
    loader = DataLoader()
    X, y = loader.preprocess(sample_data)
    
    assert X.shape[0] == sample_data.shape[0]
    assert X.shape[1] == 7  # 7 features
    assert len(y) == sample_data.shape[0]
    assert not X.isna().any().any()  # No NaN values
