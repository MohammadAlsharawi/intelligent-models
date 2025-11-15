# Voted Perceptron Classifier with Tkinter GUI

This project demonstrates a binary classification system using the **Voted Perceptron algorithm**, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🧠 Dataset: Breast Cancer Wisconsin

- Source: `sklearn.datasets.load_breast_cancer`
- Type: Real-world medical dataset
- Classes:
  - `malignant` (label -1)
  - `benign` (label +1)
- Features: 30 numeric features describing cell nuclei (e.g., radius, texture, perimeter)
- In this project, we use the first 5 features for simplicity in the GUI:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness

---

## 🔍 Algorithm: Voted Perceptron

The **Voted Perceptron** is an extension of the classic Perceptron algorithm.  
Instead of keeping a single weight vector, it stores multiple weight vectors and their associated vote counts. During prediction, each vector "votes" on the classification, and the final decision is based on the weighted majority.

### 🔄 How It Works:
1. Initialize weight vector `w` and vote count `c = 1`
2. For each training sample:
   - If the prediction is incorrect, store `w` and `c`, then update `w`
   - If correct, increment `c`
3. During prediction, each stored vector votes, and the majority decides the final class

---

## ⚙️ StandardScaler

Before training, we apply `StandardScaler` to normalize the features.  
This ensures that all features have **zero mean** and **unit variance**, which helps the Perceptron converge faster and perform better.

### Why it's important:
- Prevents features with large scales from dominating the learning process
- Improves model stability and accuracy

---

## 🔬 Perceptron vs Voted Perceptron

| Feature              | Perceptron                  | Voted Perceptron                      |
|----------------------|-----------------------------|---------------------------------------|
| Weight vectors       | Single                      | Multiple (with vote counts)           |
| Decision strategy    | Based on current weights    | Majority vote from stored vectors     |
| Robustness           | Lower                       | Higher (less sensitive to noise)      |
| Performance          | Good for linearly separable | Better generalization in practice     |

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9474  |
| Precision  | 0.9710  |
| Recall     | 0.9437  |
| F1 Score   | 0.9571  |

These metrics indicate strong performance in classifying tumor types using the voted perceptron approach.

---

## 🖥️ GUI Features

- **Tab 1: Predict**
  - Input form for 5 features
  - Predict button with result popup
- **Tab 2: Visualization**
  - Display of performance metrics
  - Scatter plot: True vs Predicted labels

---
