# Salary Predictor using Linear Regression

This project demonstrates a practical application of machine learning to predict salaries based on user input features such as age, gender, education level, job title, and years of experience. It uses a Linear Regression model trained on structured salary data and provides both interactive prediction and performance visualization through a Tkinter-based GUI.

## 🔍 Features
- Predict salary based on custom user input
- Interactive GUI with input fields and result pop-up
- Visualization tab showing model performance metrics:
  - R² Score
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- Scatter plot comparing actual vs predicted salaries

## 🧠 Machine Learning Workflow
- Data cleaning: remove duplicates and missing values
- Feature encoding: convert categorical variables to numerical using one-hot encoding
- Train-test split: 80% training, 20% testing
- Model training: Linear Regression using `scikit-learn`
- Evaluation: R², MSE, RMSE, MAE

## 📊 Model Performance
- **R² Score**: 0.8316  
- **MSE**: 461,536,104.80  
- **RMSE**: 21,483.39  
- **MAE**: 15,741.89  

These metrics indicate that the model performs well in capturing salary trends based on the input features.

## 🖥️ Technologies Used
- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib
- Tkinter


