import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
data = load_breast_cancer()
X = data.data[:, :5]  # Use first 5 features
y = data.target
feature_names = data.feature_names[:5]
target_names = data.target_names

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.where(y == 0, -1, 1)  # Convert to -1 and 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Average Perceptron implementation
class AveragePerceptron:
    def __init__(self, epochs=10):
        self.epochs = epochs
        self.avg_weights = None

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        avg_w = np.zeros(X.shape[1])
        count = 0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                if yi * np.dot(w, xi) <= 0:
                    w += yi * xi
                avg_w += w
                count += 1
        self.avg_weights = avg_w / count

    def predict(self, X):
        return np.where(np.dot(X, self.avg_weights) > 0, 1, -1)

# Train model
model = AveragePerceptron(epochs=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# GUI setup
root = tk.Tk()
root.title("Average Perceptron Classifier")
root.geometry("1000x700")

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text='Predict')
notebook.add(tab2, text='Visualization')
notebook.pack(expand=True, fill='both')

# Prediction tab
tk.Label(tab1, text="Enter feature values:", font=('Arial', 14)).pack(pady=10)
entries = []
for i, feature in enumerate(feature_names):
    frame = tk.Frame(tab1)
    frame.pack(pady=5)
    tk.Label(frame, text=feature, width=25, anchor='w').pack(side='left')
    var = tk.DoubleVar()
    entry = tk.Entry(frame, textvariable=var, width=20)
    entry.pack(side='left')
    entries.append(var)

def predict():
    try:
        input_data = [var.get() for var in entries]
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        label = target_names[0] if prediction == -1 else target_names[1]
        messagebox.showinfo("Prediction", f"Predicted Class: {label}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(tab1, text="Predict", command=predict, bg='lightgreen', font=('Arial', 12)).pack(pady=20)

# Visualization tab
tk.Label(tab2, text="Model Performance Metrics", font=('Arial', 14, 'bold')).pack(pady=10)
tk.Label(tab2, text=f"Accuracy: {acc:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Precision: {prec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Recall: {rec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"F1 Score: {f1:.4f}", font=('Arial', 12)).pack()

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter((y_test + 1) // 2, (y_pred + 1) // 2, color='purple', alpha=0.6)
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.set_title('True vs Predicted Labels')
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=tab2)
canvas.draw()
canvas.get_tk_widget().pack(pady=20)

root.mainloop()
