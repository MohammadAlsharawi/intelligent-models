# Logistic Regression Classifier with Tkinter GUI

This project demonstrates a binary classification system using **Logistic Regression**, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🧠 Dataset: Breast Cancer Wisconsin

- Source: `sklearn.datasets.load_breast_cancer`
- Type: Real-world medical dataset
- Samples: 569
- Classes:
  - `malignant` (label 0)
  - `benign` (label 1)
- Features: 30 numeric features describing cell nuclei (e.g., radius, texture, perimeter)
- In this project, we use the first 5 features for simplicity in the GUI:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness

This dataset is widely used for binary classification tasks and medical diagnostics.

---

## 📈 Algorithm: Logistic Regression

**Logistic Regression** is a linear model used for binary classification. It estimates the probability that a given input belongs to a particular class using the logistic (sigmoid) function.

### 🔄 How It Works:
- Computes a weighted sum of input features
- Applies the sigmoid function to produce a probability between 0 and 1
- Classifies based on a threshold (typically 0.5)

### ✅ Advantages:
- Fast and efficient
- Interpretable coefficients
- Performs well on linearly separable data
- Outputs probabilities

### ❌ Disadvantages:
- Assumes linear relationship between features and log-odds
- May underperform on complex or non-linear data
- Sensitive to outliers

---

## ⚙️ Preprocessing: StandardScaler

We apply `StandardScaler` to normalize the input features before training the model.

### What it does:
- Transforms each feature to have **zero mean** and **unit variance**
- Formula:  
  

\[
  z = \frac{x - \mu}{\sigma}
  \]



### Why it's used:
- Ensures consistent input scale
- Improves convergence and stability of the logistic regression model

---

## 🔁 Evaluation: Cross-Validation

We use **Stratified K-Fold Cross-Validation** to evaluate the model's performance reliably.

### How it works:
- Splits data into `k` folds (here, 5)
- Maintains class distribution across folds
- Trains and tests the model on different folds
- Aggregates performance metrics

### Benefits:
- Reduces bias from a single train-test split
- Provides more stable and generalizable metrics

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9297  |
| Precision  | 0.9342  |
| Recall     | 0.9562  |
| F1 Score   | 0.9446  |
| R² Score   | 0.0993  |

These results reflect strong performance in classifying tumor types. The high recall score indicates excellent sensitivity to detecting malignant cases, while the precision confirms the model's reliability in positive predictions. The R² score is low because it's not ideal for classification tasks, but included for completeness.

---

## 📉 Feature Coefficients

Logistic Regression provides interpretable coefficients that indicate the influence of each feature on the prediction.

| Feature          | Coefficient |
|------------------|-------------|
| Mean smoothness  | -1.75       |
| Mean area        | -1.50       |
| Mean perimeter   | -1.50       |
| Mean texture     | -1.25       |
| Mean radius      | -1.00       |

Negative coefficients suggest that higher values of these features are associated with the malignant class.

---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 5 features
- Predict button with result popup

### Tab 2: Visualization
- Display of performance metrics
- Bar chart showing feature coefficients

---
