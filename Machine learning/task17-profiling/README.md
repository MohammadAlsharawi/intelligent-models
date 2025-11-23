# Profiling in Machine Learning Projects

Profiling is a critical step in machine learning and data science workflows. It helps developers understand the structure, quality, and performance of their data and code. This README explains the different types of profiling, their purposes, and how to apply them effectively.

---

## 📊 What Is Profiling?

**Profiling** refers to the process of analyzing either:
- The **data** used in a machine learning pipeline
- The **code** and **execution performance** of the pipeline itself

Profiling helps answer questions like:
- What does my dataset look like?
- Are there missing or duplicate values?
- Which parts of my code are slow or memory-intensive?

---

## 🧮 1. Data Profiling

**Data profiling** is the process of examining the dataset to understand its structure, quality, and statistical properties.

### 🔍 Key Tasks:
- Checking for missing values
- Identifying data types and ranges
- Detecting outliers and duplicates
- Summarizing distributions and correlations

### 🛠️ Common Tools:
- `pandas_profiling` / `ydata-profiling`
- `sweetviz`
- `dtale`
- Manual inspection using `pandas.describe()`, `info()`, `isnull().sum()`

### 📦 Example:
```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data.csv")
profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file("profiling_report.html")
