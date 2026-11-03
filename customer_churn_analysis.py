import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Path for the data on the GitHub repository
path = Path(__file__).parent / "data" / "churn_bank.csv"
df = pd.read_csv(path)

print(df.head())

# Check that the dataset is clean

# Obtain a technical summary of the dataset
print(df.info)
# Print the column names
print(df.columns)
# Check the data types of the variables 
print(df.dtypes)
# Obtain a technical summary of the numerical variables
print(df.select_dtypes(include = np.number).describe().T.to_string())
# Check whether there are any missing data 
print(df.isna().sum()) # no missing data
# Check wether there are any duplicated rows in the dataset
print(df.duplicated().sum())

# Question 1: What is the churn rate among the bank's customers?

# We define churn rate as follows: customers who exited / total number of customers 

churn_rate = df["Exited"].sum() / len(df)
print(churn_rate)

# The churn rate among the bank's customers irs 20.37 %. 

# Question 2: How are the different variables (gender, age, geography, etc.) distributed in the dataset?

# For the numerical variables: 
print(df.select_dtypes(include = np.number).describe().T.to_string())


