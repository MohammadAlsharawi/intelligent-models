import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Perceptron model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

root = tk.Tk()
root.title("Perceptron Classifier")
root.geometry("1000x700")

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text='Predict')
notebook.add(tab2, text='Visualization')
notebook.pack(expand=True, fill='both')

tk.Label(tab1, text="Enter feature values:", font=('Arial', 14)).pack(pady=10)
entries = []
for i, feature in enumerate(feature_names[:5]):  # Use first 5 features for simplicity
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
        input_scaled = scaler.transform([input_data + [0]*(X.shape[1]-5)])  
        prediction = model.predict(input_scaled)[0]
        label = target_names[prediction]
        messagebox.showinfo("Prediction", f"Predicted Class: {label}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(tab1, text="Predict", command=predict, bg='lightgreen', font=('Arial', 12)).pack(pady=20)

tk.Label(tab2, text="Model Performance Metrics", font=('Arial', 14, 'bold')).pack(pady=10)
tk.Label(tab2, text=f"Accuracy: {acc:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Precision: {prec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Recall: {rec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"F1 Score: {f1:.4f}", font=('Arial', 12)).pack()

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred, color='purple', alpha=0.6)
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.set_title('True vs Predicted Labels')
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=tab2)
canvas.draw()
canvas.get_tk_widget().pack(pady=20)

root.mainloop()
