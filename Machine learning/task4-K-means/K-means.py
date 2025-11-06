import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

class KMeansClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering Analysis")
        self.root.geometry("1200x800")
        
        # Generate synthetic data
        self.X, self.y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
        
        # Train K-Means model
        self.kmeans = KMeans(n_clusters=3, random_state=0)
        self.y_kmeans = self.kmeans.fit_predict(self.X)
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_true, self.y_kmeans)
        self.cm = confusion_matrix(self.y_true, self.y_kmeans)
        self.report = classification_report(self.y_true, self.y_kmeans, output_dict=True)
        
        # Calculate inertia for elbow method
        self.inertia = []
        for k in range(1, 10):
            kmeans_temp = KMeans(n_clusters=k, random_state=0)
            kmeans_temp.fit(self.X)
            self.inertia.append(kmeans_temp.inertia_)
        
        # Create notebook (tab controller)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_clustering_tab()
        self.create_elbow_tab()
        self.create_metrics_tab()
        self.create_prediction_tab()
    
    def create_clustering_tab(self):
        """Create tab for K-Means clustering visualization"""
        clustering_frame = ttk.Frame(self.notebook)
        self.notebook.add(clustering_frame, text="K-Means Clustering")
        
        # Title
        title_label = tk.Label(clustering_frame, text="K-Means Clustering Results", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create figure for clustering
        fig = Figure(figsize=(10, 6))
        
        # Plot 1: True labels
        ax1 = fig.add_subplot(121)
        scatter1 = ax1.scatter(self.X[:, 0], self.X[:, 1], c=self.y_true, s=30, cmap='viridis')
        ax1.set_title('True Clusters')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot 2: K-Means results
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(self.X[:, 0], self.X[:, 1], c=self.y_kmeans, s=30, cmap='viridis')
        centers = self.kmeans.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        ax2.set_title('K-Means Clustering\n(Red X = Cluster Centers)')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        plt.colorbar(scatter2, ax=ax2)
        
        fig.tight_layout()
        
        # Embed figure in tkinter
        canvas = FigureCanvasTkAgg(fig, clustering_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_elbow_tab(self):
        """Create tab for Elbow Method visualization"""
        elbow_frame = ttk.Frame(self.notebook)
        self.notebook.add(elbow_frame, text="Elbow Method")
        
        # Title
        title_label = tk.Label(elbow_frame, text="Elbow Method for Optimal K Selection", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create figure for elbow method
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(range(1, 10), self.inertia, marker='o', linewidth=2, markersize=8)
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.grid(True, alpha=0.3)
        
        # Highlight the chosen k (3)
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Chosen k=3')
        ax.legend()
        
        # Add explanation
        explanation = """
        The Elbow Method helps determine the optimal number of clusters.
        Look for the 'elbow' point where inertia stops decreasing significantly.
        In this case, k=3 was chosen as it represents the optimal balance.
        """
        
        text_widget = tk.Text(elbow_frame, height=4, width=80, font=('Arial', 10))
        text_widget.pack(pady=10)
        text_widget.insert('1.0', explanation)
        text_widget.config(state='disabled')
        
        # Embed figure in tkinter
        canvas = FigureCanvasTkAgg(fig, elbow_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_metrics_tab(self):
        """Create tab for performance metrics"""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Performance Metrics")
        
        # Create main frames
        top_frame = tk.Frame(metrics_frame)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        bottom_frame = tk.Frame(metrics_frame)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side: Accuracy and Confusion Matrix
        left_frame = tk.LabelFrame(bottom_frame, text="Classification Metrics", 
                                 font=('Arial', 12, 'bold'))
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right side: Classification Report
        right_frame = tk.LabelFrame(bottom_frame, text="Detailed Report", 
                                  font=('Arial', 12, 'bold'))
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Add content to left frame
        self.add_classification_metrics(left_frame)
        
        # Add content to right frame
        self.add_classification_report(right_frame)
    
    def add_classification_metrics(self, parent):
        """Add accuracy and confusion matrix to the frame"""
        # Accuracy
        acc_frame = tk.Frame(parent)
        acc_frame.pack(fill='x', padx=10, pady=10)
        
        acc_label = tk.Label(acc_frame, 
                           text=f"Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)",
                           font=('Arial', 14, 'bold'), fg='blue')
        acc_label.pack(anchor='w')
        
        # Confusion Matrix
        cm_frame = tk.Frame(parent)
        cm_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        fig_cm = Figure(figsize=(6, 5))
        ax_cm = fig_cm.add_subplot(111)
        
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
                   yticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
        ax_cm.set_xlabel('Predicted Cluster')
        ax_cm.set_ylabel('True Cluster')
        ax_cm.set_title('Confusion Matrix')
        
        canvas_cm = FigureCanvasTkAgg(fig_cm, cm_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill='both', expand=True)
    
    def add_classification_report(self, parent):
        """Add classification report to the frame"""
        report_text = tk.Text(parent, height=20, width=60, font=('Courier', 10))
        report_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Format classification report as string
        report_str = self.format_classification_report(self.report)
        report_text.insert('1.0', report_str)
        report_text.config(state='disabled')
    
    def format_classification_report(self, report_dict):
        """Format classification report as string"""
        lines = []
        lines.append("              precision    recall  f1-score   support\n")
        lines.append("                                                    \n")
        
        for class_name in ['0', '1', '2']:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                line = f"     {class_name}        {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}         {int(metrics['support'])}\n"
                lines.append(line)
        
        lines.append("\n")
        if 'accuracy' in report_dict:
            lines.append("    accuracy" + " " * 21 + f"{report_dict['accuracy']:.2f}" + " " * 11 + f"{len(self.X)}\n")
        if 'macro avg' in report_dict:
            lines.append("   macro avg" + " " * 10 + f"{report_dict['macro avg']['precision']:.2f}" + " " * 7 + 
                        f"{report_dict['macro avg']['recall']:.2f}" + " " * 7 + 
                        f"{report_dict['macro avg']['f1-score']:.2f}" + " " * 7 + 
                        f"{int(report_dict['macro avg']['support'])}\n")
        if 'weighted avg' in report_dict:
            lines.append("weighted avg" + " " * 10 + f"{report_dict['weighted avg']['precision']:.2f}" + " " * 7 + 
                        f"{report_dict['weighted avg']['recall']:.2f}" + " " * 7 + 
                        f"{report_dict['weighted avg']['f1-score']:.2f}" + " " * 7 + 
                        f"{int(report_dict['weighted avg']['support'])}\n")
        
        return "".join(lines)
    
    def create_prediction_tab(self):
        """Create tab for making predictions on new data"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="Make Prediction")
        
        # Title
        title_label = tk.Label(prediction_frame, text="Predict Cluster for New Data", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = tk.Frame(prediction_frame)
        input_frame.pack(pady=20, padx=20, fill='x')
        
        # Create input fields for features
        self.input_vars = {}
        features = ['Feature 1', 'Feature 2']
        
        for i, feature in enumerate(features):
            frame = tk.Frame(input_frame)
            frame.pack(fill='x', pady=8)
            
            label = tk.Label(frame, text=f"{feature}:", width=15, anchor='w', font=('Arial', 11))
            label.pack(side='left')
            
            var = tk.DoubleVar()
            entry = tk.Entry(frame, textvariable=var, width=20, font=('Arial', 11))
            entry.pack(side='left', padx=10)
            
            # Set some default values based on data range
            default_value = np.mean(self.X[:, i])
            var.set(f"{default_value:.2f}")
            
            self.input_vars[feature] = var
        
        # Buttons frame
        button_frame = tk.Frame(prediction_frame)
        button_frame.pack(pady=20)
        
        predict_btn = tk.Button(button_frame, text="Predict Cluster", 
                               command=self.predict_cluster,
                               bg='lightgreen', font=('Arial', 12, 'bold'),
                               width=15, height=2)
        predict_btn.pack(side='left', padx=10)
        
        random_btn = tk.Button(button_frame, text="Random Sample", 
                              command=self.use_random_sample,
                              bg='lightblue', font=('Arial', 12),
                              width=15, height=2)
        random_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(button_frame, text="Clear", 
                             command=self.clear_inputs,
                             bg='lightcoral', font=('Arial', 12),
                             width=15, height=2)
        clear_btn.pack(side='left', padx=10)
        
        # Result display
        result_frame = tk.LabelFrame(prediction_frame, text="Prediction Result", 
                                   font=('Arial', 12, 'bold'))
        result_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, width=80, 
                                  font=('Arial', 11), state='disabled')
        self.result_text.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Add some sample data suggestions
        sample_frame = tk.Frame(prediction_frame)
        sample_frame.pack(pady=10)
        
        sample_label = tk.Label(sample_frame, text="Sample data ranges:", 
                               font=('Arial', 10, 'italic'))
        sample_label.pack()
        
        ranges_text = f"Feature 1: [{self.X[:, 0].min():.2f}, {self.X[:, 0].max():.2f}], "
        ranges_text += f"Feature 2: [{self.X[:, 1].min():.2f}, {self.X[:, 1].max():.2f}]"
        
        ranges_label = tk.Label(sample_frame, text=ranges_text, 
                               font=('Arial', 9), fg='gray')
        ranges_label.pack()
    
    def predict_cluster(self):
        """Predict cluster for input data"""
        try:
            # Get input values
            input_data = []
            for feature in ['Feature 1', 'Feature 2']:
                value = self.input_vars[feature].get()
                input_data.append(value)
            
            # Convert to numpy array and reshape
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            prediction = self.kmeans.predict(input_array)[0]
            
            # Calculate distances to all cluster centers
            distances = self.kmeans.transform(input_array)[0]
            
            # Display results
            self.result_text.config(state='normal')
            self.result_text.delete('1.0', tk.END)
            
            self.result_text.insert(tk.END, f"Predicted Cluster: {prediction}\n\n")
            self.result_text.insert(tk.END, "Distance to Cluster Centers:\n")
            for i, distance in enumerate(distances):
                self.result_text.insert(tk.END, f"  Cluster {i}: {distance:.4f}\n")
            
            self.result_text.insert(tk.END, f"\nInput Features:\n")
            self.result_text.insert(tk.END, f"  Feature 1: {input_data[0]}\n")
            self.result_text.insert(tk.END, f"  Feature 2: {input_data[1]}\n")
            
            # Add cluster center information
            center = self.kmeans.cluster_centers_[prediction]
            self.result_text.insert(tk.END, f"\nCluster Center {prediction}:\n")
            self.result_text.insert(tk.END, f"  Feature 1: {center[0]:.4f}\n")
            self.result_text.insert(tk.END, f"  Feature 2: {center[1]:.4f}\n")
            
            self.result_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Please check your input values:\n{str(e)}")
    
    def use_random_sample(self):
        """Use a random sample from the dataset"""
        random_index = np.random.randint(0, len(self.X))
        sample = self.X[random_index]
        true_label = self.y_true[random_index]
        pred_label = self.y_kmeans[random_index]
        
        # Set input values
        self.input_vars['Feature 1'].set(f"{sample[0]:.2f}")
        self.input_vars['Feature 2'].set(f"{sample[1]:.2f}")
        
        # Display sample info
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, f"Random Sample #{random_index}\n")
        self.result_text.insert(tk.END, f"True Cluster: {true_label}\n")
        self.result_text.insert(tk.END, f"K-Means Cluster: {pred_label}\n")
        self.result_text.insert(tk.END, "\nClick 'Predict Cluster' to verify prediction\n")
        self.result_text.config(state='disabled')
    
    def clear_inputs(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set("0.0")
        
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.config(state='disabled')

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansClusteringApp(root)
    root.mainloop()