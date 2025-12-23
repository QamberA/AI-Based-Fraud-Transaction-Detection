
import logging
from model_manager import ModelManager
import numpy as np
import os

def verify_metrics():
    print("Verifying Metrics...")
    
    # 1. Setup Mock Data
    # 5 features to match test_model_manager but evaluation logic doesn't strictly depend on feature count for the keys check
    # But wait, evaluate_all needs X_test and y_test
    X_train = np.random.rand(50, 5) 
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 5)
    y_test = np.random.randint(0, 2, 10)
    
    # 2. Initialize ModelManager
    mm = ModelManager()
    
    # Train quickly (no grid search)
    mm.param_grids = {} # Disable grid search for speed if it was enabled
    mm.train_all(X_train, y_train)
    
    # 3. Evaluate
    results = mm.evaluate_all(X_test, y_test)
    
    # 4. Check for keys
    expected_metrics = ['R2', 'MSE', 'RMSE', 'F1-Score', 'Recall', 'Precision']
    
    for model_name, metrics in results.items():
        print(f"Checking metrics for {model_name}...")
        for m in expected_metrics:
            if m not in metrics:
                print(f"FAILED: {m} not found in {model_name} results")
                return # Fail fast
            else:
                print(f"  OK: {m}: {metrics[m]}")

    print("SUCCESS: All metrics found.")

if __name__ == "__main__":
    verify_metrics()
