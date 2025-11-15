# Perceptron Classifier with Tkinter GUI

This project demonstrates a binary classification system using the **Perceptron algorithm** applied to the **Breast Cancer Wisconsin dataset**. The model predicts whether a tumor is malignant or benign based on input features. The application includes a user-friendly GUI built with Tkinter, featuring two tabs: one for prediction and one for performance visualization.

---

## 🧠 Project Overview

- **Algorithm**: Perceptron (binary linear classifier)
- **Dataset**: Breast Cancer Wisconsin (from `sklearn.datasets`)
- **Interface**: Tkinter GUI with two tabs:
  - **Predict**: Enter feature values and get classification result
  - **Visualization**: View model performance metrics and scatter plot

---

## 🔍 Features

- Input form for five key features:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness
- Real-time prediction with result popup
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Scatter plot of true vs predicted labels

---

## 📊 Model Performance

- **Accuracy**: 0.9737  
- **Precision**: 0.9857  
- **Recall**: 0.9718  
- **F1 Score**: 0.9787  

These metrics indicate strong performance in classifying tumor types.

---

## 🧪 How It Works

1. Load and preprocess the Breast Cancer dataset
2. Standardize features using `StandardScaler`
3. Train a Perceptron model on 80% of the data
4. Evaluate predictions on the remaining 20%
5. Build a GUI for:
   - User input and prediction
   - Displaying evaluation metrics and scatter plot

---

## 🖥️ Technologies Used

- Python
- scikit-learn
- pandas, numpy
- matplotlib
- Tkinter

---
