"""
Model evaluation metrics and visualization
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import numpy as np


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(y_true, y_pred) -> dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with accuracy, precision, recall, f1-score
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics
    
    @staticmethod
    def get_classification_report(y_true, y_pred, target_names=None) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def print_metrics(metrics: dict):
        """Print formatted metrics"""
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():.<30} {value:.4f}")
        print("="*50)
