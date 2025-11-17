# Decision Tree Classifier with Tkinter GUI

This project demonstrates a binary classification system using a **Decision Tree Classifier**, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

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

## 🌳 Algorithm: Decision Tree Classifier

A **Decision Tree** is a supervised learning algorithm used for both classification and regression. It splits the data into branches based on feature thresholds, forming a tree-like structure.

### 🔄 How It Works:
- Starts at the root node
- Splits data based on feature values to maximize class separation (using criteria like Gini or entropy)
- Continues splitting until stopping conditions are met (e.g., max depth, pure leaf)

### ✅ Advantages:
- Easy to interpret and visualize
- Handles both numerical and categorical data
- No need for feature scaling

### ❌ Disadvantages:
- Prone to overfitting
- Sensitive to small changes in data
- Less accurate than ensemble methods

---

## ⚙️ Preprocessing: StandardScaler

Although Decision Trees don’t require feature scaling, we apply `StandardScaler` to normalize the input for consistency and better visualization.

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
| Accuracy   | 0.8805  |
| Precision  | 0.8981  |
| Recall     | 0.9132  |
| F1 Score   | 0.9066  |

These results reflect the model's ability to classify breast cancer cases with high reliability. The recall score of 91.32% indicates strong sensitivity to detecting malignant cases, while the precision of 89.81% shows the model's confidence in its positive predictions. The F1 score balances both metrics, confirming the model's robustness.


---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 5 features
- Predict button with result popup

### Tab 2: Visualization
- Display of performance metrics
- Decision tree plot showing feature splits and class labels

---

