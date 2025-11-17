# Random Forest Classifier with Tkinter GUI

This project demonstrates a binary classification system using a **Random Forest Classifier**, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

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

## 🌲 Algorithm: Random Forest Classifier

A **Random Forest** is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

### 🔄 How It Works:
- Trains multiple decision trees on random subsets of the data
- Each tree votes on the class label
- Final prediction is based on majority vote

### ✅ Advantages:
- High accuracy and robustness
- Reduces overfitting compared to a single decision tree
- Handles noisy data well
- Provides feature importance scores

### ❌ Disadvantages:
- Less interpretable than a single tree
- Requires more memory and computation
- Slower training and prediction time

---

## ⚙️ Preprocessing: StandardScaler

Although Random Forests don’t require feature scaling, we apply `StandardScaler` to normalize the input for consistency and better GUI behavior.

### What it does:
- Transforms each feature to have **zero mean** and **unit variance**
- Formula:  
  

\[
  z = \frac{x - \mu}{\sigma}
  \]



### Why it's used:
- Ensures consistent input for GUI prediction
- Helps when comparing with other models that require scaling

---

## 🔁 Evaluation: Cross-Validation

We use **Stratified K-Fold Cross-Validation** to evaluate the model's performance reliably.

### How it works:
- Splits data into `k` folds (here, 5)
- Ensures each fold has the same class distribution
- Trains and tests the model on different folds
- Aggregates performance metrics

### Benefits:
- Reduces bias from a single train-test split
- Provides more stable and generalizable metrics

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9209  |
| Precision  | 0.9239  |
| Recall     | 0.9624  |
| F1 Score   | 0.9379  |

These results reflect strong performance in classifying tumor types. The high recall score indicates excellent sensitivity to detecting malignant cases, while the precision confirms the model's reliability in positive predictions.

---

## 📈 Feature Importances

The Random Forest model provides insight into which features are most influential in decision-making:

| Feature          | Importance |
|------------------|------------|
| Mean area        | High       |
| Mean perimeter   | High       |
| Mean radius      | Moderate   |
| Mean texture     | Moderate   |
| Mean smoothness  | Low        |

---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 5 features
- Predict button with result popup

### Tab 2: Visualization
- Display of performance metrics
- Bar chart showing feature importances

---

## 🔍 Comparison: Decision Tree vs Random Forest

| Feature               | Decision Tree         | Random Forest             |
|-----------------------|-----------------------|---------------------------|
| Type                  | Single model          | Ensemble of trees         |
| Accuracy              | Moderate              | High                      |
| Overfitting risk      | High                  | Low                       |
| Interpretability      | High (visual tree)    | Moderate (feature importances) |
| Training speed        | Fast                  | Slower                    |
| Robustness to noise   | Low                   | High                      |
| Feature importance    | Limited               | Built-in and reliable     |

### 🧠 When to Use:
- Use **Decision Tree** if you need a simple, interpretable model and fast results.
- Use **Random Forest** if you want higher accuracy, better generalization, and robustness to noise.

---
