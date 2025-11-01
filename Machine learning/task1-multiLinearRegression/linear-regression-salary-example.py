import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

X_encoded = pd.get_dummies(X, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)
#we make encode because the linear regression only accept numerical values

# Train-test split 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

root = tk.Tk()
root.title("Salary Predictor")

notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text='Predict Salary')
notebook.add(tab2, text='Visualization')
notebook.pack(expand=True, fill='both')

def predict_salary():
    try:
        age = float(age_entry.get())
        gender = gender_entry.get()
        education = education_entry.get()
        job = job_entry.get()
        experience = float(experience_entry.get())

        user_input = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Education Level': education,
            'Job Title': job,
            'Years of Experience': experience
        }])

        user_encoded = pd.get_dummies(user_input)
        user_encoded = user_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        predicted_salary = model.predict(user_encoded)[0]
        messagebox.showinfo("Predicted Salary", f"Estimated Salary: ${predicted_salary:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

tk.Label(tab1, text="Age").grid(row=0, column=0, padx=5, pady=5)
tk.Label(tab1, text="Gender (Male/Female)").grid(row=1, column=0, padx=5, pady=5)
tk.Label(tab1, text="Education Level").grid(row=2, column=0, padx=5, pady=5)
tk.Label(tab1, text="Job Title").grid(row=3, column=0, padx=5, pady=5)
tk.Label(tab1, text="Years of Experience").grid(row=4, column=0, padx=5, pady=5)

age_entry = tk.Entry(tab1)
gender_entry = tk.Entry(tab1)
education_entry = tk.Entry(tab1)
job_entry = tk.Entry(tab1)
experience_entry = tk.Entry(tab1)

age_entry.grid(row=0, column=1, padx=5, pady=5)
gender_entry.grid(row=1, column=1, padx=5, pady=5)
education_entry.grid(row=2, column=1, padx=5, pady=5)
job_entry.grid(row=3, column=1, padx=5, pady=5)
experience_entry.grid(row=4, column=1, padx=5, pady=5)

tk.Button(tab1, text="Predict Salary", command=predict_salary).grid(row=5, columnspan=2, pady=10)

metrics_frame = tk.Frame(tab2)
metrics_frame.pack(pady=10)

tk.Label(metrics_frame, text=f"R² Score: {r2:.4f}", font=('Arial', 12)).pack(anchor='w')
tk.Label(metrics_frame, text=f"Mean Squared Error (MSE): {mse:.2f}", font=('Arial', 12)).pack(anchor='w')
tk.Label(metrics_frame, text=f"Root Mean Squared Error (RMSE): {rmse:.2f}", font=('Arial', 12)).pack(anchor='w')
tk.Label(metrics_frame, text=f"Mean Absolute Error (MAE): {mae:.2f}", font=('Arial', 12)).pack(anchor='w')


fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred, color='purple')
ax.set_xlabel('Actual Salary')
ax.set_ylabel('Predicted Salary')
ax.set_title('Actual vs Predicted Salary')
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=tab2)
canvas.draw()
canvas.get_tk_widget().pack(expand=True, fill='both')


root.mainloop()


# R² Score (Coefficient of Determination) => Measures how well the model explains the variance in the target variable. Ranges from 0 to 1 (higher is better).
# Mean Squared Error (MSE) => Average of squared differences between predicted and actual values. Lower is better.
# Root Mean Squared Error (RMSE) => Square root of MSE. Easier to interpret in the same units as the target.
# Mean Absolute Error (MAE) => Average of absolute differences between predicted and actual values. Lower is better.

