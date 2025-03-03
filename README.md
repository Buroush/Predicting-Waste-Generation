### **Project: Solid Waste Prediction Using LSTM and XGBoost**  

This project aims to predict daily solid waste generation using machine learning techniques. It leverages **Long Short-Term Memory (LSTM)** networks to capture time-dependent patterns and **XGBoost** for structured feature analysis. By blending these models, the system achieves improved accuracy in forecasting waste amounts. The dataset is preprocessed with cyclic encoding for seasonal trends, and predictions are evaluated using **RMSE, R² Score, MAE, and Explained Variance (EV).** The goal is to help municipalities optimize waste management and resource allocation effectively.

Data Set Link : https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset (This dataset contains the daily solid waste amounts collected by garbage trucks within the city limits of 5 cities over the world.)
``` python
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ivantha/daily-solid-waste-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
```
# *Author : Pankaj Mondal*
[LinkedIn](https://www.linkedin.com/in/buroush/)  
[GitHub](https://github.com/Buroush)  
[LeetCode](https://leetcode.com/Buroush)  
