
import logging
from model_manager import ModelManager
import numpy as np
import os
import shutil

def test_model_manager():
    print("Testing ModelManager...")
    
    # 1. Setup Mock Data
    X_train = np.random.rand(100, 5) # 5 features
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    
    # 2. Initialize ModelManager
    mm = ModelManager()
    
    # Clean up any existing model file for clean test
    if os.path.exists(mm.model_path):
        os.remove(mm.model_path)
    
    # 3. Train
    print("Training...")
    mm.train_all(X_train, y_train)
    assert mm.trained == True
    assert os.path.exists(mm.model_path), "Model file was not created"
    
    # 4. Predict
    print("Predicting...")
    preds = mm.predict(X_test[0:1])
    assert 'Logistic Regression' in preds
    
    # 5. Explain
    print("Explaining...")
    fig = mm.get_explanation(X_test[0:1])
    assert fig is not None, "SHAP figure was not returned"
    print("Shape figure generated.")
    
    # 6. Persistence
    print("Testing Persistence...")
    del mm
    mm2 = ModelManager() # Should load from disk
    assert mm2.trained == True, "Failed to load trained state"
    assert 'Logistic Regression' in mm2.models
    
    print("test_model_manager passed!")

if __name__ == "__main__":
    test_model_manager()
