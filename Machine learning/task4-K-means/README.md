# K-Means Clustering Analysis with Tkinter GUI

This project demonstrates an interactive application for performing **K-Means clustering** on synthetic data. It includes visualizations, performance metrics, and a prediction interface—all built using Python and Tkinter.

---

## 📊 Project Overview

- **Algorithm**: K-Means Clustering
- **Data**: Synthetic dataset generated with `make_blobs`
- **Interface**: Tkinter GUI with multiple tabs:
  - **Clustering Visualization**: Compare true vs predicted clusters
  - **Elbow Method**: Determine optimal number of clusters (k)
  - **Performance Metrics**: Accuracy, confusion matrix, classification report
  - **Prediction**: Input new data and predict cluster assignment

---

## 🔍 Features

- Visual comparison between true clusters and K-Means results
- Elbow Method plot to select optimal `k`
- Confusion matrix and classification report
- Predict cluster for new data points
- Random sample selection and distance to cluster centers

---

## 📈 Model Performance

- **Accuracy**: 33.33%  
- **Precision/Recall/F1-score**: Varies by cluster  
- **Confusion Matrix**: Perfect diagonal alignment, but label mismatch affects accuracy

> Note: K-Means is unsupervised, so label alignment may differ from true labels. Accuracy is computed for illustrative purposes.

---

## 🧪 How It Works

1. Generate synthetic data with 3 clusters
2. Train K-Means model with `n_clusters=3`
3. Evaluate clustering performance using:
   - Accuracy score
   - Confusion matrix
   - Classification report
4. Visualize cluster assignments and Elbow Method
5. Predict cluster for new input using GUI

---

## 🖥️ Technologies Used

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Tkinter

---
