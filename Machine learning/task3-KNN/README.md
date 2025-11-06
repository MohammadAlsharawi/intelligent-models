# Iris Flower Classification with KNN and Tkinter GUI

This project demonstrates a machine learning application that classifies iris flowers into three species—**setosa**, **versicolor**, and **virginica**—based on their physical features. It uses the well-known Iris dataset and a K-Nearest Neighbors (KNN) classifier, wrapped in an interactive GUI built with Tkinter.

---

## 🌸 Project Overview

- **Dataset**: Iris dataset from `sklearn.datasets`
- **Model**: K-Nearest Neighbors (KNN)
- **Interface**: Tkinter GUI with two tabs:
  - **Prediction**: Input features and predict species
  - **Data Analysis**: View model accuracy, confusion matrix, classification report, and scatter plots

---

## 🔍 Features

- Input sepal and petal measurements to predict flower species
- Display prediction probabilities for all classes
- View model performance metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Visualize feature relationships using scatter plots

---

## 📊 Model Performance

- **Accuracy**: 100.00%  
- **Confusion Matrix**: Perfect classification across all three species  
- **Classification Report**: Precision, recall, and F1-score all at 1.00

---

## 🧪 How It Works

1. Load Iris dataset and convert to DataFrame
2. Split into training and testing sets (80/20)
3. Standardize features using `StandardScaler`
4. Train a KNN classifier with `n_neighbors=5`
5. Build a GUI for:
   - Real-time prediction from user input
   - Displaying evaluation metrics and visualizations

---

## 🖥️ Technologies Used

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Tkinter

---
