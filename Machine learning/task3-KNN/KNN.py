import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Load and prepare the data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = [target_names[i] for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classification")
        self.root.geometry("1000x700")
        
        # Create notebook (tab controller)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_prediction_tab()
        self.create_analysis_tab()
        
    def create_prediction_tab(self):
        # Prediction tab
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Prediction")
        
        # Title
        title_label = tk.Label(self.prediction_frame, text="Iris Flower Classifier", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = tk.Frame(self.prediction_frame)
        input_frame.pack(pady=20, padx=20, fill='x')
        
        # Create input fields
        self.input_vars = {}
        for i, feature in enumerate(feature_names):
            frame = tk.Frame(input_frame)
            frame.pack(fill='x', pady=5)
            
            label = tk.Label(frame, text=f"{feature}:", width=20, anchor='w')
            label.pack(side='left')
            
            var = tk.DoubleVar(value=0.0)
            entry = tk.Entry(frame, textvariable=var, width=20)
            entry.pack(side='left', padx=10)
            
            self.input_vars[feature] = var
        
        # Buttons frame
        button_frame = tk.Frame(self.prediction_frame)
        button_frame.pack(pady=20)
        
        predict_btn = tk.Button(button_frame, text="Predict", 
                               command=self.predict_species,
                               bg='lightblue', font=('Arial', 12))
        predict_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(button_frame, text="Clear", 
                             command=self.clear_inputs,
                             bg='lightcoral', font=('Arial', 12))
        clear_btn.pack(side='left', padx=10)
        
        # Result display
        self.result_frame = tk.Frame(self.prediction_frame, relief='groove', bd=2)
        self.result_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        result_title = tk.Label(self.result_frame, text="Prediction Result", 
                               font=('Arial', 14, 'bold'))
        result_title.pack(pady=10)
        
        self.result_text = tk.Text(self.result_frame, height=8, width=60, 
                                  font=('Arial', 11), state='disabled')
        self.result_text.pack(pady=10, padx=10, fill='both', expand=True)
        
    def create_analysis_tab(self):
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Data Analysis")
        
        # Create a canvas and scrollbar for the analysis tab
        canvas = tk.Canvas(self.analysis_frame)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add content to analysis tab
        self.add_analysis_content()
        
    def add_analysis_content(self):
        # Accuracy and metrics
        metrics_frame = tk.Frame(self.scrollable_frame, relief='groove', bd=2)
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        metrics_title = tk.Label(metrics_frame, text="Model Performance Metrics", 
                                font=('Arial', 14, 'bold'))
        metrics_title.pack(pady=10)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        acc_label = tk.Label(metrics_frame, 
                            text=f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)",
                            font=('Arial', 12))
        acc_label.pack(pady=5)
        
        # Confusion Matrix
        cm_frame = tk.Frame(self.scrollable_frame, relief='groove', bd=2)
        cm_frame.pack(fill='x', padx=10, pady=10)
        
        cm_title = tk.Label(cm_frame, text="Confusion Matrix", 
                           font=('Arial', 14, 'bold'))
        cm_title.pack(pady=10)
        
        # Create confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = Figure(figsize=(6, 4))
        ax_cm = fig_cm.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, 
                   yticklabels=target_names, ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix')
        
        canvas_cm = FigureCanvasTkAgg(fig_cm, cm_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(pady=10)
        
        # Classification Report
        report_frame = tk.Frame(self.scrollable_frame, relief='groove', bd=2)
        report_frame.pack(fill='x', padx=10, pady=10)
        
        report_title = tk.Label(report_frame, text="Classification Report", 
                               font=('Arial', 14, 'bold'))
        report_title.pack(pady=10)
        
        report_text = tk.Text(report_frame, height=10, width=80, font=('Courier', 10))
        report_text.pack(pady=10, padx=10, fill='both')
        
        # Get classification report as string
        report_str = classification_report(y_test, y_pred, target_names=target_names)
        report_text.insert('1.0', report_str)
        report_text.config(state='disabled')
        
        # Scatter plots
        scatter_frame = tk.Frame(self.scrollable_frame, relief='groove', bd=2)
        scatter_frame.pack(fill='x', padx=10, pady=10)
        
        scatter_title = tk.Label(scatter_frame, text="Data Visualization", 
                                font=('Arial', 14, 'bold'))
        scatter_title.pack(pady=10)
        
        # Create scatter plot
        fig_scatter = Figure(figsize=(10, 8))
        
        # Plot 1: Sepal length vs Sepal width
        ax1 = fig_scatter.add_subplot(221)
        colors = ['red', 'blue', 'green']
        for i, species in enumerate(target_names):
            mask = df['species'] == species
            ax1.scatter(df[mask]['sepal length (cm)'], df[mask]['sepal width (cm)'], 
                       c=colors[i], label=species, alpha=0.7)
        ax1.set_xlabel('Sepal Length (cm)')
        ax1.set_ylabel('Sepal Width (cm)')
        ax1.legend()
        ax1.set_title('Sepal Length vs Sepal Width')
        
        # Plot 2: Petal length vs Petal width
        ax2 = fig_scatter.add_subplot(222)
        for i, species in enumerate(target_names):
            mask = df['species'] == species
            ax2.scatter(df[mask]['petal length (cm)'], df[mask]['petal width (cm)'], 
                       c=colors[i], label=species, alpha=0.7)
        ax2.set_xlabel('Petal Length (cm)')
        ax2.set_ylabel('Petal Width (cm)')
        ax2.legend()
        ax2.set_title('Petal Length vs Petal Width')
        
        # Plot 3: Sepal length vs Petal length
        ax3 = fig_scatter.add_subplot(223)
        for i, species in enumerate(target_names):
            mask = df['species'] == species
            ax3.scatter(df[mask]['sepal length (cm)'], df[mask]['petal length (cm)'], 
                       c=colors[i], label=species, alpha=0.7)
        ax3.set_xlabel('Sepal Length (cm)')
        ax3.set_ylabel('Petal Length (cm)')
        ax3.legend()
        ax3.set_title('Sepal Length vs Petal Length')
        
        # Plot 4: Sepal width vs Petal width
        ax4 = fig_scatter.add_subplot(224)
        for i, species in enumerate(target_names):
            mask = df['species'] == species
            ax4.scatter(df[mask]['sepal width (cm)'], df[mask]['petal width (cm)'], 
                       c=colors[i], label=species, alpha=0.7)
        ax4.set_xlabel('Sepal Width (cm)')
        ax4.set_ylabel('Petal Width (cm)')
        ax4.legend()
        ax4.set_title('Sepal Width vs Petal Width')
        
        fig_scatter.tight_layout()
        
        canvas_scatter = FigureCanvasTkAgg(fig_scatter, scatter_frame)
        canvas_scatter.draw()
        canvas_scatter.get_tk_widget().pack(pady=10)
        
    def predict_species(self):
        try:
            # Get input values
            input_data = []
            for feature in feature_names:
                value = self.input_vars[feature].get()
                input_data.append(value)
            
            # Convert to numpy array and reshape
            input_array = np.array(input_data).reshape(1, -1)
            
            # Scale the input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = knn.predict(input_scaled)[0]
            probabilities = knn.predict_proba(input_scaled)[0]
            
            # Display results
            self.result_text.config(state='normal')
            self.result_text.delete('1.0', tk.END)
            
            self.result_text.insert(tk.END, f"Predicted Species: {target_names[prediction]}\n\n")
            self.result_text.insert(tk.END, "Prediction Probabilities:\n")
            for i, prob in enumerate(probabilities):
                self.result_text.insert(tk.END, f"  {target_names[i]}: {prob:.4f} ({prob*100:.2f}%)\n")
            
            self.result_text.insert(tk.END, f"\nInput Features:\n")
            for i, feature in enumerate(feature_names):
                self.result_text.insert(tk.END, f"  {feature}: {input_data[i]}\n")
            
            self.result_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Please check your input values:\n{str(e)}")
    
    def clear_inputs(self):
        for var in self.input_vars.values():
            var.set(0.0)
        
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.config(state='disabled')

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()