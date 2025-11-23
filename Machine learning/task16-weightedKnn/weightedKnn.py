import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load and clean dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X = X.iloc[:, :5]
X.fillna(X.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Weighted KNN model
model = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

# Metrics
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Train final model
model.fit(X_scaled, y)

# GUI setup
root = tk.Tk()
root.title("Weighted KNN Classifier - Breast Cancer")
root.geometry("1100x750")

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text='Predict')
notebook.add(tab2, text='Visualization')
notebook.pack(expand=True, fill='both')

# Tab 1: Prediction
tk.Label(tab1, text="Enter Feature Values", font=('Arial', 16, 'bold')).pack(pady=10)
entries = []
for feature in X.columns:
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
        label = data.target_names[prediction]
        messagebox.showinfo("Prediction", f"Predicted Class: {label}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(tab1, text="Predict", command=predict, bg='lightgreen', font=('Arial', 12)).pack(pady=20)

# Tab 2: Visualization
tk.Label(tab2, text="Model Performance Metrics", font=('Arial', 16, 'bold')).pack(pady=10)
tk.Label(tab2, text=f"Accuracy : {acc:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Precision: {prec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"Recall   : {rec:.4f}", font=('Arial', 12)).pack()
tk.Label(tab2, text=f"F1 Score : {f1:.4f}", font=('Arial', 12)).pack()

# Class distribution plot
fig, ax = plt.subplots(figsize=(6, 4))
counts = [list(y_pred).count(i) for i in range(len(data.target_names))]
ax.bar(data.target_names, counts, color='teal')
ax.set_title("Predicted Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=tab2)
canvas.draw()
canvas.get_tk_widget().pack(pady=20)

root.mainloop()
