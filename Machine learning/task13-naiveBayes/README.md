# Naive Bayes Classifier with Tkinter GUI – Iris Dataset

This project demonstrates a multi-class classification system using a **Naive Bayes classifier**, applied to the **Iris dataset**. The model predicts the species of an iris flower based on its physical measurements. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🌸 Dataset: Iris

- Source: `sklearn.datasets.load_iris`
- Type: Real-world botanical dataset
- Samples: 150
- Classes:
  - `setosa`
  - `versicolor`
  - `virginica`
- Features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)

This dataset is widely used for classification tasks and educational purposes due to its simplicity and well-separated classes.

---

## 📈 Algorithm: Naive Bayes Classifier

**Naive Bayes** is a probabilistic classifier based on Bayes' Theorem, assuming independence between features.

### 🔄 How It Works:
- Calculates the probability of each class given the input features
- Assumes each feature contributes independently to the outcome
- Chooses the class with the highest posterior probability

### ✅ Advantages:
- Fast and efficient
- Works well with small datasets
- Handles multi-class problems naturally
- Robust to irrelevant features

### ❌ Disadvantages:
- Assumes feature independence (often unrealistic)
- May underperform on correlated or complex data
- Sensitive to how probabilities are estimated

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
- Improves numerical stability and GUI behavior

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
| Accuracy   | 0.9733  |

The model achieves high accuracy on the Iris dataset, confirming its effectiveness for this classification task.

---

## 📈 Visualization

The GUI includes a bar chart showing the distribution of predicted classes across the dataset:
- Helps visualize model bias or class imbalance
- Confirms that all classes are being predicted correctly

---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 4 features
- Predict button with result popup

### Tab 2: Visualization
- Display of model accuracy
- Bar chart showing predicted class distribution

---
