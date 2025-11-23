# Weighted KNN Classifier with Tkinter GUI – Breast Cancer Dataset

This project demonstrates a binary classification system using a **Weighted K-Nearest Neighbors (KNN)** model, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🧠 Dataset: Breast Cancer Wisconsin

- Source: `sklearn.datasets.load_breast_cancer`
- Type: Real-world medical dataset
- Samples: 569
- Classes:
  - `malignant` (label 0)
  - `benign` (label 1)
- Features: 30 numeric features describing cell nuclei
- GUI uses the first 5 features for simplicity:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness

This dataset is widely used for binary classification tasks and medical diagnostics.

---

## 🧹 Data Cleaning

Before training:
- Missing values are handled using mean imputation (`fillna`)
- Features are standardized using `StandardScaler` to ensure consistent scale

---

## 📈 Algorithm: Weighted K-Nearest Neighbors (KNN)

**KNN** is a non-parametric algorithm that classifies a sample based on the majority class of its nearest neighbors.

### 🔄 How It Works:
- Calculates distance between input and training samples
- Selects the `k` nearest neighbors
- **Weighted version** gives more influence to closer neighbors using inverse distance

### ✅ Advantages:
- Simple and intuitive
- No training phase (lazy learning)
- Effective for well-separated classes

### ❌ Disadvantages:
- Sensitive to feature scaling
- Slow with large datasets
- Performance depends on choice of `k` and distance metric

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
| Accuracy   | 0.9467  |
| Precision  | 0.9472  |
| Recall     | 0.9632  |
| F1 Score   | 0.9551  |

These results reflect strong performance in classifying tumor types. The high recall score indicates excellent sensitivity to detecting malignant cases, while the precision confirms the model's reliability in positive predictions.

---

## 📈 Visualization

The GUI includes a bar chart showing the distribution of predicted classes across the dataset:
- Helps visualize model bias or class imbalance
- Confirms that all classes are being predicted correctly

---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 5 features
- Predict button with result popup
- Displays the predicted class name (`malignant` or `benign`)

### Tab 2: Visualization
- Display of model performance metrics
- Bar chart showing predicted class distribution

---
