from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, mean_squared_error, r2_score

class ModelManager:
    def __init__(self):
        # model architectures
        self.models = {
            'ANN (Neural Network)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42,class_weight='balanced'),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,class_weight='balanced')
        }

        # Hyperparameter Grids for Tuning
        self.param_grids = {
            'ANN (Neural Network)': {}, 
            'Decision Tree': {'max_depth': [5, 10, 20]},
            'Logistic Regression': {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']}
        }

        self.trained = False
        self.model_path = 'trained_models.pkl'
        
        # Attempt to load existing models
        self.load_models()

    def train_all(self, X_train, y_train):
        """Trains all defined models using GridSearchCV for hyperparameter tuning."""
        print("Starting training with Hyperparameter Tuning...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            grid = self.param_grids.get(name, {})
            
            if grid:
                # Perform Grid Search
                clf = GridSearchCV(model, grid, cv=3, scoring='f1', n_jobs=-1)
                clf.fit(X_train, y_train)
                self.models[name] = clf.best_estimator_
                print(f"  Best params for {name}: {clf.best_params_}")
            else:
                # Standard Fit
                model.fit(X_train, y_train)
                
        self.trained = True
        self.save_models()
        print("Training complete and models saved.")

    def evaluate_all(self, X_test, y_test):
        """Returns a dictionary of performance metrics for all models."""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            # Handle probability prediction for different model types
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = [0] * len(y_test) # Fallback
            

            # Calculate ROC Curve metrics
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'AUC': roc_auc_score(y_test, y_prob),
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'FPR': fpr,
                'TPR': tpr,
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        return results

    def predict(self, X_input):
        """Returns predictions from all models for a specific input."""
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1]
            predictions[name] = {'prediction': int(pred), 'probability': float(prob)}
        return predictions

    def save_models(self):
        """Persists trained models to disk."""
        joblib.dump(self.models, self.model_path)

    def load_models(self):
        """Loads models from disk if they exist."""
        if os.path.exists(self.model_path):
            try:
                self.models = joblib.load(self.model_path)
                self.trained = True
                print("Models loaded from disk.")
            except Exception as e:
                print(f"Failed to load models: {e}")

    def get_explanation(self, X_input):
        if 'Logistic Regression' not in self.models:
            return None
        
        model = self.models['Logistic Regression']
        
        # Feature Names 
        feature_names = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest',
            'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]
        
        # Explainer with feature names
        import pandas as pd
        X_df = pd.DataFrame(X_input, columns=feature_names)
        
        explainer = shap.LinearExplainer(model, X_df) 
        shap_values = explainer(X_df)
        
        fig = plt.figure(figsize=(8, 6))
        
        # 3. Plot with the names
        shap.plots.waterfall(shap_values[0], show=False, max_display=10)
        
        plt.tight_layout()
        return fig