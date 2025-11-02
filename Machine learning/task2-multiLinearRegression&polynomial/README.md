# Salary Predictor with Polynomial Regression

This project demonstrates a salary prediction system using **Polynomial Regression**, built with Python and a Tkinter-based GUI. The model estimates salaries based on user input features such as age, gender, education level, job title, and years of experience. It also visualizes model performance and compares it with standard Linear Regression.

---

## 🧠 Project Overview

- **Goal**: Predict salaries based on structured input data using regression techniques.
- **Model Used**: Polynomial Regression (degree = 2)
- **Interface**: Interactive GUI built with Tkinter
- **Visualization**: Actual vs Predicted salary scatter plot + performance metrics

---

## 📊 Model Performance

| Metric                  | Linear Regression | Polynomial Regression |
|------------------------|-------------------|------------------------|
| R² Score               | 0.8316            | 0.8694                 |
| Mean Squared Error     | 461,536,104.80    | 357,946,294.10         |
| Root Mean Squared Error| 21,483.39         | 18,919.47              |
| Mean Absolute Error    | 15,741.89         | 11,503.84              |

✅ **Polynomial Regression outperforms Linear Regression** across all metrics, indicating better fit and more accurate predictions.

---

## 🔍 Key Features

- Predict salary from user input via GUI
- Encode categorical features using one-hot encoding
- Polynomial feature expansion for non-linear relationships
- Performance metrics: R², MSE, RMSE, MAE
- Visualization of prediction accuracy

---

## 🧪 How It Works

1. **Data Preprocessing**
   - Load salary dataset
   - Remove duplicates and missing values
   - Encode categorical variables (`Gender`, `Education Level`, `Job Title`)

2. **Model Training**
   - Split data into training and testing sets (80/20)
   - Apply polynomial transformation (degree = 2)
   - Train `LinearRegression` on transformed features

3. **Prediction**
   - GUI accepts user input
   - Input is encoded and transformed
   - Model predicts salary and displays result

4. **Evaluation**
   - Metrics displayed in GUI
   - Scatter plot shows actual vs predicted salaries

---

## 🖥️ Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib
- Tkinter

---

