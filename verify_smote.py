import numpy as np
from data_manager import DataManager

def verify_smote():
    print("Verifying SMOTE implementation...")
    dm = DataManager()
    X_train, X_test, y_train, y_test = dm.load_and_preprocess()
    
    unique, counts = np.unique(y_train, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    print(f"Class distribution after SMOTE: {counts_dict}")
    
    if len(counts) != 2:
        print("FAILED: Expected 2 classes.")
        return

    count_0 = counts_dict.get(0, 0)
    count_1 = counts_dict.get(1, 0)
    
    if count_0 == count_1:
        print("SUCCESS: Classes are perfectly balanced.")
    args = (count_0, count_1)
    diff = abs(count_0 - count_1)
    if diff <= 1:
         print("SUCCESS: Classes are balanced (within 1 sample diff due to rounding potentially).")
    else:
        print(f"FAILED: Classes are not balanced. Diff: {diff}")

if __name__ == "__main__":
    verify_smote()
