"""
Main training script for soil health prediction model
"""
import sys
from data import DataLoader
from model import CropRecommender
from evaluate import ModelEvaluator


def main():
    """
    Complete training pipeline:
    1. Load and preprocess data
    2. Train model
    3. Evaluate on test set
    4. Save model
    """
    print("Starting Soil Health & Crop Recommendation Model Training...")
    
    try:
        # Step 1: Load and preprocess data
        print("\n[1/4] Loading and preprocessing data...")
        loader = DataLoader()
        df = loader.load_data()
        print(f"Loaded {len(df)} samples")
        
        X, y = loader.preprocess(df)
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {y.unique()}")
        
        # Step 2: Train model
        print("\n[2/4] Training model...")
        model = CropRecommender()
        X_train, X_test, y_train, y_test = model.train(X, y)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Step 3: Evaluate
        print("\n[3/4] Evaluating model...")
        y_pred = model.predict(X_test)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)
        evaluator.print_metrics(metrics)
        
        print("\nClassification Report:")
        print(evaluator.get_classification_report(y_test, y_pred))
        
        # Step 4: Save model
        print("\n[4/4] Saving model...")
        model.save()
        
        print("\n✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
