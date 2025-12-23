import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import joblib
from imblearn.over_sampling import SMOTE

class DataManager:
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.categorical_features = ['type']
        self.preprocessor_path = 'preprocessor.pkl'
        self.load_preprocessor()

    def generate_synthetic_data(self, n_samples=5000):
        print('generating blah blah')
        """Generates realistic synthetic financial transaction data with logical patterns."""
        np.random.seed(42)
        
        # 1. Generate SAFE transactions (90% of data)
        n_safe = int(n_samples * 0.9)
        safe_data = {
            'step': np.random.randint(1, 744, n_safe), # 1 month of hours
            'type': np.random.choice(['PAYMENT', 'CASH_IN', 'DEBIT', 'CASH_OUT', 'TRANSFER'], n_safe, p=[0.4, 0.2, 0.1, 0.15, 0.15]),
            'amount': np.random.exponential(scale=1000, size=n_safe),
            'oldbalanceOrg': np.random.uniform(0, 50000, n_safe),
            'newbalanceOrig': np.zeros(n_safe),
            'oldbalanceDest': np.random.uniform(0, 50000, n_safe),
            'newbalanceDest': np.zeros(n_safe),
            'isFraud': np.zeros(n_safe, dtype=int)
        }
        
        # Calculate correct balances for safe transactions
        for i in range(n_safe):
            if safe_data['type'][i] in ['PAYMENT', 'DEBIT', 'CASH_OUT', 'TRANSFER']:
                 # Money leaving origin
                 safe_data['newbalanceOrig'][i] = max(0, safe_data['oldbalanceOrg'][i] - safe_data['amount'][i])
                 safe_data['newbalanceDest'][i] = safe_data['oldbalanceDest'][i] + safe_data['amount'][i]
            else:
                 # Money entering origin (CASH_IN)
                 safe_data['newbalanceOrig'][i] = safe_data['oldbalanceOrg'][i] + safe_data['amount'][i]
                 safe_data['newbalanceDest'][i] = max(0, safe_data['oldbalanceDest'][i] - safe_data['amount'][i])

        # 2. Generate FRAUD transactions (10% of data)
        n_fraud = n_samples - n_safe
        fraud_data = {
            'step': np.random.randint(1, 744, n_fraud),
            'type': np.random.choice(['CASH_OUT', 'TRANSFER'], n_fraud), # Fraud mostly happens here
            'amount': np.random.uniform(10000, 1000000, n_fraud),
            'oldbalanceOrg': np.random.uniform(10000, 1000000, n_fraud),
            'newbalanceOrig': np.zeros(n_fraud), # Fraudsters often empty accounts
            'oldbalanceDest': np.random.uniform(0, 10000, n_fraud),
            'newbalanceDest': np.zeros(n_fraud), # Destination often hides the money trail
            'isFraud': np.ones(n_fraud, dtype=int)
        }
        
        # Pattern: Fraud amount often equals the exact old balance (Account Takeover)
        fraud_data['amount'] = fraud_data['oldbalanceOrg'] 
        
        df_safe = pd.DataFrame(safe_data)
        df_fraud = pd.DataFrame(fraud_data)
        
        return pd.concat([df_safe, df_fraud]).sample(frac=1).reset_index(drop=True)

    def load_and_preprocess(self, filepath=None):
        """Loads data, adds engineered features, and splits it."""
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath, nrows=100000)
        else:
            df = self.generate_synthetic_data()

        # Feature Engineering: 
        # If newBalance != oldBalance - amount, something is suspicious.
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

        # Update features list to include engineered features
        numeric_features_extended = self.numeric_features + ['errorBalanceOrig', 'errorBalanceDest']
        X = df[numeric_features_extended + self.categorical_features]
        y = df['isFraud']
        # Pipeline Construction
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features_extended),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Fit preprocessor
        X_train_proc = self.preprocessor.fit_transform(X_train)
        self.save_preprocessor()
        
        # Apply SMOTE to training data
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)
        print(f"Original shape: {X_train_proc.shape}, Resampled shape: {X_train_resampled.shape}")

        X_test_proc = self.preprocessor.transform(X_test)
        return X_train_resampled, X_test_proc, y_train_resampled, y_test

    def save_preprocessor(self):
        """Saves the fitted preprocessor to disk."""
        try:
            joblib.dump(self.preprocessor, self.preprocessor_path)
            print("Preprocessor saved.")
        except Exception as e:
            print(f"Failed to save preprocessor: {e}")

    def load_preprocessor(self):
        """Loads preprocessor from disk."""
        if os.path.exists(self.preprocessor_path):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                print("Preprocessor loaded from disk.")
            except Exception as e:
                print(f"Failed to load preprocessor: {e}")

    def process_input(self, input_dict):
        """Prepares a single user input for the AI."""
        df = pd.DataFrame([input_dict])
        
        # Re-create engineered features on live input
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        
        return self.preprocessor.transform(df)