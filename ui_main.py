import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np

# Import our custom modules
from data_manager import DataManager
from model_manager import ModelManager

# Setup Plot Style
plt.style.use('dark_background')

class FraudDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Sentinel: AI Financial Fraud Detection System")
        self.geometry("1200x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # --- Logic Modules ---
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.is_trained = False

        # --- Layout Configuration ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Navigation) ---
        self.create_sidebar()

        # --- Main Content Area (Frames) ---
        self.frames = {}
        self.create_dashboard_frame()
        self.create_analysis_frame()
        self.create_prediction_frame()

        # Show Dashboard initially
        self.show_frame("Dashboard")

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="SENTINEL AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=30)

        self.btn_dashboard = ctk.CTkButton(self.sidebar, text="Dashboard & Train", command=lambda: self.show_frame("Dashboard"))
        self.btn_dashboard.pack(pady=10, padx=20, fill="x")

        self.btn_analysis = ctk.CTkButton(self.sidebar, text="Model Analytics", command=lambda: self.show_frame("Analysis"))
        self.btn_analysis.pack(pady=10, padx=20, fill="x")

        self.btn_predict = ctk.CTkButton(self.sidebar, text="Live Prediction", command=lambda: self.show_frame("Prediction"))
        self.btn_predict.pack(pady=10, padx=20, fill="x")
        
        # Status Indicator
        status_text = "Status: Loaded from Disk" if self.model_manager.trained else "Status: Not Trained"
        status_color = "#2ecc71" if self.model_manager.trained else "gray"
        self.status_label = ctk.CTkLabel(self.sidebar, text=status_text, text_color=status_color)
        self.status_label.pack(side="bottom", pady=20)

    def create_dashboard_frame(self):
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["Dashboard"] = frame

        # Header
        lbl = ctk.CTkLabel(frame, text="System Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        lbl.pack(pady=20, padx=20, anchor="w")

        # Control Panel
        control_frame = ctk.CTkFrame(frame)
        control_frame.pack(fill="x", padx=20, pady=10)

        self.train_btn = ctk.CTkButton(control_frame, text="Initialize System & Train Models", 
                                       command=self.start_training_thread, height=50, fg_color="#2ecc71", hover_color="#27ae60")
        self.train_btn.pack(pady=20, padx=20, fill="x")

        # Logs Console
        log_lbl = ctk.CTkLabel(frame, text="System Logs:", font=ctk.CTkFont(weight="bold"))
        log_lbl.pack(padx=20, anchor="w")
        
        self.log_textbox = ctk.CTkTextbox(frame, height=400, font=("Consolas", 12))
        self.log_textbox.pack(fill="both", expand=True, padx=20, pady=10)
        self.log("Welcome to Sentinel AI. Please initialize training to begin.")

    def create_analysis_frame(self):
        frame = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["Analysis"] = frame
        
        lbl = ctk.CTkLabel(frame, text="Model Performance Analytics", font=ctk.CTkFont(size=24, weight="bold"))
        lbl.pack(pady=20, padx=20, anchor="w")
        
        self.analysis_container = ctk.CTkFrame(frame)
        self.analysis_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.analysis_container, text="Train models to view analytics.", text_color="gray").pack(pady=50)

    def create_prediction_frame(self):
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["Prediction"] = frame

        lbl = ctk.CTkLabel(frame, text="Live Transaction Analysis", font=ctk.CTkFont(size=24, weight="bold"))
        lbl.pack(pady=20, padx=20, anchor="w")

        # Input Grid
        input_frame = ctk.CTkFrame(frame)
        input_frame.pack(fill="x", padx=20)

        self.entries = {}
        fields = [
            ("Step (Hour)", "step"), ("Amount ($)", "amount"),
            ("Old Balance Orig", "oldbalanceOrg"), ("New Balance Orig", "newbalanceOrig"),
            ("Old Balance Dest", "oldbalanceDest"), ("New Balance Dest", "newbalanceDest")
        ]

        for i, (label, key) in enumerate(fields):
            row, col = divmod(i, 2)
            ctk.CTkLabel(input_frame, text=label).grid(row=row, column=col*2, padx=10, pady=10, sticky="e")
            ent = ctk.CTkEntry(input_frame, placeholder_text="0.0")
            ent.grid(row=row, column=col*2+1, padx=10, pady=10, sticky="w")
            self.entries[key] = ent

        # Transaction Type Dropdown
        ctk.CTkLabel(input_frame, text="Transaction Type").grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.type_combo = ctk.CTkComboBox(input_frame, values=['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT'])
        self.type_combo.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.type_combo.set('TRANSFER')

        # Predict Button
        self.btn_run_pred = ctk.CTkButton(frame, text="Analyze Transaction", command=self.run_prediction, 
                                          height=50, fg_color="#e74c3c", hover_color="#c0392b", state="disabled")
        self.btn_run_pred.pack(pady=30, padx=20, fill="x")

        # Explain Button (Hidden/Disabled initially)
        self.btn_explain = ctk.CTkButton(frame, text="Explain Prediction (Why?)", command=self.explain_prediction,
                                         height=40, fg_color="#3498db", hover_color="#2980b9", state="disabled")
        self.btn_explain.pack(pady=5, padx=20, fill="x")

        # Result Display
        self.result_frame = ctk.CTkFrame(frame, fg_color="#1a1a1a")
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.result_lbl = ctk.CTkLabel(self.result_frame, text="Awaiting Input...", font=ctk.CTkFont(size=18))
        self.result_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def show_frame(self, name):
        # Hide all frames
        for frame in self.frames.values():
            frame.grid_forget()
        # Show selected frame
        self.frames[name].grid(row=0, column=1, sticky="nsew")

    def log(self, message):
        self.log_textbox.insert("end", f">> {message}\n")
        self.log_textbox.see("end")

    # --- Training Logic ---
    def start_training_thread(self):
        self.train_btn.configure(state="disabled", text="Training in progress...")
        self.status_label.configure(text="Status: Training...", text_color="orange")
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        try:
            self.log("Loading financial data...")
            X_train, X_test, y_train, y_test = self.data_manager.load_and_preprocess('financial_data.csv')
            
            self.log("Training Neural Network, Decision Tree, and Logical Regression...")
            self.model_manager.train_all(X_train, y_train)
            
            self.log("Evaluating models...")
            results = self.model_manager.evaluate_all(X_test, y_test)
            
            # Update UI from main thread
            self.after(0, lambda: self.finish_training(results))
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.after(0, lambda: self.train_btn.configure(state="normal", text="Retry Training"))

    def finish_training(self, results):
        self.is_trained = True
        self.train_btn.configure(text="Retrain Models", state="normal")
        self.btn_run_pred.configure(state="normal")
        self.btn_explain.configure(state="disabled") # reset explanation button
        self.status_label.configure(text="Status: System Ready", text_color="#2ecc71")
        
        # Log Results
        self.log("-" * 30)
        for model, metrics in results.items():
            self.log(f"--- {model} ---")
            if "Decision Tree" in model:
                 self.log(f"F1: {metrics['F1-Score']:.4f}, Recall: {metrics['Recall']:.4f}, Precision: {metrics['Precision']:.4f}")
            elif "Logistic Regression" in model:
                 self.log(f"R2: {metrics['R2']:.4f}, MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            elif "ANN" in model:
                 self.log(f"F1: {metrics['F1-Score']:.4f}")
            else:
                 self.log(f"Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
        self.log("-" * 30)
        self.log("Training Complete. Check 'Model Analytics' tab for graphs.")

        # Render Analytics Charts
        self.render_analytics(results)

    def render_analytics(self, results):
        # Clear previous widgets
        for widget in self.analysis_container.winfo_children():
            widget.destroy()

        # Create Figure with larger size for more plots
        fig = plt.Figure(figsize=(10, 12), dpi=100)
        fig.patch.set_facecolor('#2b2b2b') # Match background
        
        # Use GridSpec for custom layout: 3 Rows
        # Row 1: Bar Chart (Metrics)
        # Row 2: ROC Curves (Combined)
        # Row 3: Confusion Matrices (One per model)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

        models = list(results.keys())

        # 1. Bar Chart: Model Comparison
        ax1 = fig.add_subplot(gs[0, :])
        accs = [results[m]['Accuracy'] for m in models]
        aucs = [results[m]['AUC'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width/2, accs, width, label='Accuracy', color='#3498db')
        ax1.bar(x + width/2, aucs, width, label='AUC Score', color='#e67e22')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', color='white')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.set_facecolor('#2b2b2b')

        # 2. ROC Curves (All Models)
        ax2 = fig.add_subplot(gs[1, :])
        for name in models:
            fpr = results[name].get('FPR', [0, 1]) 
            tpr = results[name].get('TPR', [0, 1])
            auc = results[name]['AUC']
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', color='gray') 
        ax2.set_xlabel('False Positive Rate', color='white')
        ax2.set_ylabel('True Positive Rate', color='white')
        ax2.set_title('ROC Curves', color='white')
        ax2.legend()
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.set_facecolor('#2b2b2b')

        # 3. Confusion Matrices (Individual)
        for i, name in enumerate(models):
            if i >= 3: break 
            
            ax = fig.add_subplot(gs[2, i])
            cm = results[name]['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{name}', color='white', fontsize=10)
            ax.set_ylabel('True', color='white')
            ax.set_xlabel('Pred', color='white')
            ax.tick_params(colors='white')

        fig.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Prediction Lo gic ---
    def run_prediction(self):
        try:
            # Gather In put
            input_data = {
                'step': float(self.entries['step'].get()),
                'amount': float(self.entries['amount'].get()),
                'oldbalanceOrg': float(self.entries['oldbalanceOrg'].get()),
                'newbalanceOrig': float(self.entries['newbalanceOrig'].get()),
                'oldbalanceDest': float(self.entries['oldbalanceDest'].get()),
                'newbalanceDest': float(self.entries['newbalanceDest'].get()),
                'type': self.type_combo.get()
            }
            
            # Process & Predict
            processed_data = self.data_manager.process_input(input_data)
            predictions = self.model_manager.predict(processed_data)
            is_fraud = any(p['prediction'] == 1 for p in predictions.values())
            

            details_lines = []
            for m, p in predictions.items():
                is_fraud_model = p['prediction'] == 1
                prob_fraud = p['probability']
                
                # Calculate display confidence
                # If model says Fraud, confidence is prob_fraud
                # If model says Safe, confidence is 1 - prob_fraud
                display_conf = prob_fraud if is_fraud_model else (1.0 - prob_fraud)
                
                status_str = "Fraud" if is_fraud_model else "Safe"
                details_lines.append(f"{m}: {status_str} ({display_conf:.1%} conf)")
            
            details = "\n".join(details_lines)
            
            # Update UI Colors and Text
            res_color = "#e74c3c" if is_fraud else "#2ecc71"
            
            if is_fraud:
                res_text = "🚨 FRAUD DETECTED"
                # Add a note if it was a split decision
                fraud_votes = sum(1 for p in predictions.values() if p['prediction'] == 1)
                if fraud_votes < len(predictions):
                    res_text += " (Flagged by Security Protocols)"
            else:
                res_text = "✅ SAFE TRANSACTION"
            
            self.result_lbl.configure(text=f"{res_text}\n\nModel Breakdown:\n{details}", text_color=res_color)
            
            # Enable Explanation Button and store last processed data
            self.btn_explain.configure(state="normal")
            self.last_processed_data = processed_data

        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all numeric fields contain valid numbers.")
    def explain_prediction(self):
        """Generates and displays SHAP explanation for the last prediction."""
        try:
            if not hasattr(self, 'last_processed_data'):
                return

            # Generate Figure
            fig = self.model_manager.get_explanation(self.last_processed_data)
            
            if fig is None:
                messagebox.showwarning("Explanation Unavailable", "SHAP explanation is currently only available for Logistic Regression.")
                return

            # Create Popup Window
            top = ctk.CTkToplevel(self)
            top.title("Prediction Explanation (SHAP)")
            top.geometry("700x600")
            
            # Instructions
            ctk.CTkLabel(top, text="Feature Contribution to Prediction (waterfall plot)\nRed = Pushes towards Fraud, Blue = Pushes towards Safe", 
                         font=ctk.CTkFont(size=14)).pack(pady=10)

            # Embed Figure
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Explanation Error", str(e))