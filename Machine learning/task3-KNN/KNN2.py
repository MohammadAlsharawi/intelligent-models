import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

class KNNVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Classifier Visualization")
        self.root.geometry("1200x800")
        
        # Create datasets
        self.datasets = {
            'Moons': make_moons(noise=0.3, random_state=0),
            'Blobs': make_blobs(n_samples=200, centers=2, random_state=8, cluster_std=1.5)
        }
        
        # Store models and results
        self.models = {}
        self.results = {}
        
        # Train models
        self.train_models()
        
        # Create notebook (tab controller)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs for each dataset
        self.create_moons_tab()
        self.create_blobs_tab()
        self.create_comparison_tab()
        
    def train_models(self):
        """Train KNN models for both datasets"""
        for name, (X, y) in self.datasets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train KNN model
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)
            
            # Store results
            self.models[name] = {
                'model': clf,
                'scaler': scaler,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'X_original': X,
                'y_original': y
            }
            
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
    
    def create_moons_tab(self):
        """Create tab for Moons dataset"""
        moons_frame = ttk.Frame(self.notebook)
        self.notebook.add(moons_frame, text="Moons Dataset")
        self.create_dataset_tab(moons_frame, 'Moons')
    
    def create_blobs_tab(self):
        """Create tab for Blobs dataset"""
        blobs_frame = ttk.Frame(self.notebook)
        self.notebook.add(blobs_frame, text="Blobs Dataset")
        self.create_dataset_tab(blobs_frame, 'Blobs')
    
    def create_comparison_tab(self):
        """Create comparison tab"""
        comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(comparison_frame, text="Dataset Comparison")
        
        # Create main frames
        top_frame = tk.Frame(comparison_frame)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        bottom_frame = tk.Frame(comparison_frame)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Comparison metrics
        metrics_frame = tk.LabelFrame(bottom_frame, text="Performance Comparison", 
                                    font=('Arial', 12, 'bold'))
        metrics_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Comparison visualizations
        viz_frame = tk.LabelFrame(bottom_frame, text="Dataset Comparison", 
                                font=('Arial', 12, 'bold'))
        viz_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.add_comparison_content(metrics_frame, viz_frame)
    
    def create_dataset_tab(self, parent, dataset_name):
        """Create a standardized tab for each dataset"""
        # Create main frames
        top_frame = tk.Frame(parent)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        bottom_frame = tk.Frame(parent)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side: Metrics
        metrics_frame = tk.LabelFrame(bottom_frame, text="Model Performance Metrics", 
                                    font=('Arial', 12, 'bold'))
        metrics_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right side: Visualizations
        viz_frame = tk.LabelFrame(bottom_frame, text="Visualizations", 
                                font=('Arial', 12, 'bold'))
        viz_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Add content to metrics frame
        self.add_metrics_content(metrics_frame, dataset_name)
        
        # Add content to visualizations frame
        self.add_visualizations_content(viz_frame, dataset_name)
    
    def add_metrics_content(self, parent, dataset_name):
        """Add performance metrics to the frame"""
        results = self.results[dataset_name]
        
        # Accuracy
        acc_frame = tk.Frame(parent)
        acc_frame.pack(fill='x', padx=10, pady=5)
        
        acc_label = tk.Label(acc_frame, text=f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)",
                           font=('Arial', 14, 'bold'), fg='blue')
        acc_label.pack(anchor='w')
        
        # Confusion Matrix
        cm_frame = tk.LabelFrame(parent, text="Confusion Matrix", font=('Arial', 11, 'bold'))
        cm_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        fig_cm = Figure(figsize=(5, 4))
        ax_cm = fig_cm.add_subplot(111)
        
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix')
        
        canvas_cm = FigureCanvasTkAgg(fig_cm, cm_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Classification Report
        report_frame = tk.LabelFrame(parent, text="Classification Report", font=('Arial', 11, 'bold'))
        report_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        report_text = tk.Text(report_frame, height=12, width=50, font=('Courier', 9))
        report_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Format classification report
        report_str = self.format_classification_report(results['classification_report'])
        report_text.insert('1.0', report_str)
        report_text.config(state='disabled')
    
    def add_visualizations_content(self, parent, dataset_name):
        """Add visualizations to the frame"""
        model_data = self.models[dataset_name]
        X = model_data['X_original']
        y = model_data['y_original']
        clf = model_data['model']
        scaler = model_data['scaler']
        
        # Create main figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot 1: Decision Boundary
        ax1 = fig.add_subplot(221)
        self.plot_decision_boundary(ax1, clf, scaler, X, y)
        ax1.set_title(f'Decision Boundary - {dataset_name}')
        
        # Plot 2: Training Data
        ax2 = fig.add_subplot(222)
        self.plot_dataset(ax2, model_data['X_train'], model_data['y_train'], 'Training Data')
        
        # Plot 3: Test Data Predictions
        ax3 = fig.add_subplot(223)
        y_pred = self.results[dataset_name]['y_pred']
        self.plot_predictions(ax3, model_data['X_test'], model_data['y_test'], y_pred, 'Test Data Predictions')
        
        # Plot 4: Probability Distribution
        ax4 = fig.add_subplot(224)
        self.plot_probabilities(ax4, dataset_name, 'Class Probability Distribution')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def add_comparison_content(self, metrics_frame, viz_frame):
        """Add comparison content"""
        # Performance comparison table
        table_frame = tk.Frame(metrics_frame)
        table_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create comparison table
        columns = ('Dataset', 'Accuracy', 'Precision_0', 'Precision_1', 'Recall_0', 'Recall_1', 'F1_0', 'F1_1')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=5)
        
        # Define headings
        tree.heading('Dataset', text='Dataset')
        tree.heading('Accuracy', text='Accuracy')
        tree.heading('Precision_0', text='Precision Class 0')
        tree.heading('Precision_1', text='Precision Class 1')
        tree.heading('Recall_0', text='Recall Class 0')
        tree.heading('Recall_1', text='Recall Class 1')
        tree.heading('F1_0', text='F1-Score Class 0')
        tree.heading('F1_1', text='F1-Score Class 1')
        
        # Set column widths
        tree.column('Dataset', width=100)
        for col in columns[1:]:
            tree.column(col, width=120)
        
        # Add data
        for dataset_name in ['Moons', 'Blobs']:
            results = self.results[dataset_name]
            report = results['classification_report']
            
            tree.insert('', 'end', values=(
                dataset_name,
                f"{results['accuracy']:.4f}",
                f"{report['0']['precision']:.4f}",
                f"{report['1']['precision']:.4f}",
                f"{report['0']['recall']:.4f}",
                f"{report['1']['recall']:.4f}",
                f"{report['0']['f1-score']:.4f}",
                f"{report['1']['f1-score']:.4f}"
            ))
        
        tree.pack(fill='both', expand=True)
        
        # Comparison visualization
        fig_comp = Figure(figsize=(8, 6))
        ax_comp = fig_comp.add_subplot(111)
        
        # Plot both datasets
        colors = ['red', 'blue']
        for i, (dataset_name, (X, y)) in enumerate(self.datasets.items()):
            ax_comp.scatter(X[y == 0, 0], X[y == 0, 1], c=colors[i], alpha=0.6, 
                          label=f'{dataset_name} - Class 0', marker='o')
            ax_comp.scatter(X[y == 1, 0], X[y == 1, 1], c=colors[i], alpha=0.6, 
                          label=f'{dataset_name} - Class 1', marker='s')
        
        ax_comp.set_xlabel('Feature 1')
        ax_comp.set_ylabel('Feature 2')
        ax_comp.set_title('Dataset Comparison - Original Data Distribution')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        canvas_comp = FigureCanvasTkAgg(fig_comp, viz_frame)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def plot_decision_boundary(self, ax, clf, scaler, X, y, h=0.02):
        """Plot decision boundary"""
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Scale the mesh grid points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Predict and plot
        Z = clf.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']), alpha=0.6)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=30, cmap=ListedColormap(['#FF0000', '#00FF00']))
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(*scatter.legend_elements(), title="Classes")
    
    def plot_dataset(self, ax, X, y, title):
        """Plot dataset points"""
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=30, 
                           cmap=ListedColormap(['#FF0000', '#00FF00']))
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        ax.legend(*scatter.legend_elements(), title="Classes")
    
    def plot_predictions(self, ax, X_test, y_test, y_pred, title):
        """Plot test predictions with correct/incorrect markers"""
        correct = y_pred == y_test
        colors = ['green' if c else 'red' for c in correct]
        
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=50, alpha=0.7)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Correct'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Incorrect')
        ]
        ax.legend(handles=legend_elements, title="Predictions")
    
    def plot_probabilities(self, ax, dataset_name, title):
        """Plot class probability distribution"""
        probas = self.results[dataset_name]['y_pred_proba']
        
        x = range(len(probas))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], probas[:, 0], width, label='Class 0', alpha=0.7, color='red')
        ax.bar([i + width/2 for i in x], probas[:, 1], width, label='Class 1', alpha=0.7, color='green')
        
        ax.set_xlabel('Test Samples')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def format_classification_report(self, report_dict):
        """Format classification report as string"""
        lines = []
        lines.append("              precision    recall  f1-score   support\n")
        lines.append("                                                    \n")
        
        for class_name in ['0', '1']:
            metrics = report_dict[class_name]
            line = f"     {class_name}        {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}         {int(metrics['support'])}\n"
            lines.append(line)
        
        lines.append("\n")
        lines.append("    accuracy" + " " * 21 + f"{report_dict['accuracy']:.2f}" + " " * 11 + f"{int(report_dict['macro avg']['support'])}\n")
        lines.append("   macro avg" + " " * 10 + f"{report_dict['macro avg']['precision']:.2f}" + " " * 7 + 
                    f"{report_dict['macro avg']['recall']:.2f}" + " " * 7 + 
                    f"{report_dict['macro avg']['f1-score']:.2f}" + " " * 7 + 
                    f"{int(report_dict['macro avg']['support'])}\n")
        lines.append("weighted avg" + " " * 10 + f"{report_dict['weighted avg']['precision']:.2f}" + " " * 7 + 
                    f"{report_dict['weighted avg']['recall']:.2f}" + " " * 7 + 
                    f"{report_dict['weighted avg']['f1-score']:.2f}" + " " * 7 + 
                    f"{int(report_dict['weighted avg']['support'])}\n")
        
        return "".join(lines)

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = KNNVisualizationApp(root)
    root.mainloop()