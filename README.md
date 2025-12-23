Sentinel: AI Financial Fraud Detection System

Sentinel is an AI-powered desktop application for detecting financial fraud in transaction data.
The system combines multiple machine learning models, interactive visual analytics, and explainable AI (XAI) techniques within a modern GUI to support accurate and interpretable fraud detection.

📌 Project Overview

Financial fraud poses significant risks to digital payment systems. This project addresses the problem by implementing and comparing multiple machine learning models to classify transactions as fraudulent or safe. 
The application provides:
End-to-end fraud detection pipeline
Real-time transaction analysis
Model performance comparison
Explainable predictions using SHAP
User-friendly desktop interface
This project was developed as an academic project to demonstrate practical implementation of machine learning concepts.

🧠 Algorithms Implemented
Artificial Neural Network (MLPClassifier)
Decision Tree Classifier
Logistic Regression
Each model is trained, evaluated, and compared using industry-standard metrics.

⚙️ System Features
🔹 Functional Features
Load and preprocess transaction data
Train multiple ML models simultaneously
Compare model performance (Accuracy, AUC, ROC)
Live transaction fraud prediction
Explain predictions using SHAP (Explainable AI)
Interactive desktop GUI with multiple views

🔹 Non-Functional Features
Modular and maintainable code structure
Responsive and user-friendly interface
Scalable design for adding new models
Robust error handling and logging

📊 Model Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
ROC Curve
AUC Score
Confusion Matrix
Graphical analysis is available inside the application.

🖥️ Application Interface
The GUI is built using Tkinter and CustomTkinter and includes:
Dashboard – Model training & system logs
Analytics – Performance charts and comparisons
Live Prediction – Real-time fraud analysis
Explanation View – SHAP-based model interpretability

🛠️ Tools & Technologies
Programming Language: Python
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Explainable AI: SHAP
GUI Framework: Tkinter, CustomTkinter

📁 Project Structure.
├── main.py                # Application entry point (GUI)
├── data_manager.py        # Data loading & preprocessing
├── model_manager.py       # Model training, evaluation & prediction
├── requirements.txt       # Project dependencies
├── financial_data.csv     # Dataset (or synthetic data)
└── README.md              # Project documentation

▶️ How to Run
Clone the repository:
git clone https://github.com/your-username/sentinel-fraud-detection.git


Install dependencies:
pip install -r requirements.txt

Run the application:
python index.py

🔍 Explainable AI XAI)
The system integrates SHAP (SHapley Additive exPlanations) to explain model predictions, allowing users to understand which features contributed to fraud detection decisions.

