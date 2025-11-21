# SVM Classifier with Tkinter GUI – Wine Dataset

This project demonstrates a multi-class classification system using a **Support Vector Machine (SVM)**, applied to the **Wine dataset**. The model predicts the wine cultivar based on chemical properties. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🍷 Dataset: Wine

- Source: `sklearn.datasets.load_wine`
- Type: Real-world chemical analysis dataset
- Samples: 178
- Classes:
  - Class 0: Cultivar A
  - Class 1: Cultivar B
  - Class 2: Cultivar C
- Features: 13 chemical attributes (e.g., alcohol, malic acid, ash)
- In this project, we use the first 5 features for simplicity in the GUI:
  - Alcohol
  - Malic acid
  - Ash
  - Alcalinity of ash
  - Magnesium

This dataset is commonly used for classification tasks and demonstrates how chemical composition can distinguish wine types.

---

## 🧹 Data Cleaning

Before training:
- Missing values are handled using mean imputation (`fillna`)
- Features are standardized using `StandardScaler` to ensure consistent scale

---

## 📈 Algorithm: Support Vector Machine (SVM)

**SVM** is a powerful supervised learning algorithm that finds the optimal hyperplane to separate classes in feature space.

### 🔄 How It Works:
- Maximizes the margin between classes
- Uses kernel functions to handle non-linear boundaries
- In this project, we use a **linear kernel**

### ✅ Advantages:
- Effective in high-dimensional spaces
- Robust to overfitting (especially with regularization)
- Works well for both binary and multi-class classification

### ❌ Disadvantages:
- Sensitive to feature scaling
- Can be slow with large datasets
- Requires careful tuning of kernel and parameters

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
| Accuracy   | 0.8596  |
| Precision  | 0.8580  |
| Recall     | 0.8696  |
| F1 Score   | 0.8587  |

These results reflect solid performance in classifying wine types. The balanced precision and recall indicate that the model is reliable across all classes.

---

## 📈 Visualization

The GUI includes a bar chart showing the distribution of predicted classes across the dataset:
- Helps visualize model bias or class imbalance
- Confirms that all classes are being predicted

---

## 🖥️ GUI Features

Built with **Tkinter**, the GUI includes:

### Tab 1: Predict
- Input form for 5 features
- Predict button with result popup
- Displays the predicted wine cultivar name

### Tab 2: Visualization
- Display of model performance metrics
- Bar chart showing predicted class distribution

---

