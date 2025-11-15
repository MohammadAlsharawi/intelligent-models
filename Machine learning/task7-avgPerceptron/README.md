# Average Perceptron Classifier with Tkinter GUI

This project demonstrates a binary classification system using the **Average Perceptron algorithm**, applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🧠 Dataset: Breast Cancer Wisconsin

- Source: `sklearn.datasets.load_breast_cancer`
- Type: Real-world medical dataset
- Classes:
  - `malignant` (label -1)
  - `benign` (label +1)
- Features used in GUI:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness

---

## 🔍 Algorithm: Average Perceptron

The **Average Perceptron** is a variant of the standard Perceptron algorithm.  
Instead of using the final weight vector for prediction, it keeps track of the **cumulative sum of all weight updates** during training and uses their **average** for prediction.

### 🔄 How It Works:
1. Initialize weight vector `w` and cumulative vector `avg_w`
2. For each training sample:
   - If prediction is incorrect, update `w`
   - Add `w` to `avg_w` after every sample
3. Final prediction uses `avg_w / total_updates`

This averaging helps reduce fluctuations and improves generalization.

---

## ⚙️ StandardScaler

Before training, we apply `StandardScaler` to normalize the features.  
This ensures that all features have **zero mean** and **unit variance**, which helps the Perceptron converge faster and perform better.

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9474  |
| Precision  | 0.9710  |
| Recall     | 0.9437  |
| F1 Score   | 0.9571  |

These metrics indicate strong performance in classifying tumor types using the Average Perceptron.

---

## 📈 Comparison: Perceptron vs Voted vs Average

| Feature              | Perceptron                  | Voted Perceptron                      | Average Perceptron                   |
|----------------------|-----------------------------|---------------------------------------|--------------------------------------|
| Weight vectors       | Single                      | Multiple (with vote counts)           | Single (average of all updates)      |
| Decision strategy    | Final weight                | Majority vote from stored vectors     | Average of all weight updates        |
| Robustness           | Moderate                    | High                                  | High                                 |
| Memory usage         | Low                         | High                                  | Moderate                             |
| Performance          | Good for linearly separable | Better generalization                 | Smooth and stable predictions        |

---

## 🖥️ GUI Features

- **Tab 1: Predict**
  - Input form for 5 features
  - Predict button with result popup
- **Tab 2: Visualization**
  - Display of performance metrics
  - Scatter plot: True vs Predicted labels

---
