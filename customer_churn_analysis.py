import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so

sns.set_style("ticks") 
sns.set_context("talk")
sns.set_theme(style="whitegrid")

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

<<<<<<< HEAD
=======
# Question 2: How are the different variables (gender, age, geography, etc.) distributed in the dataset?

# For the numerical variables: 
print(df.select_dtypes(include = np.number).describe().T.to_string())

#For the categorical variables

categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Exited"]

plt.suptitle("Distributions of categorical variables", fontsize=18, y=0.95)

for i, var in enumerate(categorical_var):

   #create subplots
   plt.subplot(3,2, i + 1)

   plot = sns.countplot(data = df, x = var, stat = "proportion")
   #add percentage labels
   plot.bar_label(plot.containers[0], fmt="%.2f", padding = 3, label_type="center")
   #clearer than label "proportion"
   plot.set_ylabel("Percentage")
   #so that each plot has the same scale
   plot.set_ylim(0, 1)
   #plot.set_title(f"Distribution of {var}")

plt.tight_layout()
plt.show()


#%%
# NUMERICAL VARIABLES
# For the numerical variables: 

numerical_var = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
#descriptive statistics
print(df[numerical_var].describe)

#histograms
for var in numerical_var:
   plot = sns.histplot(data = df, x = var, binwidth=5 if var == "Age" else None)
   plot.set_title(f"Distribution of {var}")
   plt.show()

#boxplots
#shows: min, 1st quartile, median, 3rd quartile, max
for var in numerical_var:
   plot = sns.boxplot(data = df, x = var)
   plot.set_title(f"Distribution of {var}")
   plt.show()


#%%
#Distribution of exited and non exited customers

## all the plots on one grid
plt.suptitle("Distributions of exited and non exited", fontsize=18, y=0.95)

for i, var in enumerate(categorical_var):

   #create subplots
   plt.subplot(3,2, i + 1)

   plot = sns.countplot(data = df, x = var, hue = "Exited")
   #add percentage labels
   plot.bar_label(plot.containers[0], fmt="%.2f", padding = 3, label_type="center")

plt.tight_layout()
plt.show()



#%%
#Question 5: How are different indicators distributed by countries


# double loop for explained and explanatory var

#loop for each categorical variable
categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Exited"]

for var1 in [v for v in categorical_var]:

    #go through each categorical variable
    for var2 in [v for v in categorical_var if v != var1] :
        shares = (
            df.groupby(var1)[var2]
            .value_counts(normalize=True)
            .round(2)
            .unstack()
        )

        #the species are the indexes of the shares dataframe
        values_var1 = tuple(shares.index)
        weight_counts = {category: shares[category].to_numpy() for category in shares.columns}

        #each part of the column is a subplot
        fig, ax = plt.subplots()
        #the bottom of the bar starts at 0, want nb of columns corresponding to the nb of values
        bottom = np.zeros(len(values_var1))

        #convert values of var 1 in str for the legend
        values_var1 = [str(v) for v in shares.index]

        #go through each value of weight_counts
        for boolean, weight_count in weight_counts.items():
            p = ax.bar(values_var1, weight_count, label=boolean, bottom=bottom)
            ax.bar_label(p, fmt="%.2f", label_type="center")
            #first add values for male, then for female
            bottom += weight_count

        ax.set_title(f"Share of {var2} in each {var1}")
        ax.legend(loc="upper right")

        plt.show()



#%% la même chose mais en grid
categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Exited"]

for var1 in [v for v in categorical_var]:

    fig, axes = plt.subplots(3,2)
    axes = axes.flatten()

    for i, var2 in enumerate([v for v in categorical_var if v != var1]) :

        ax = axes[i]
        
        shares = (
            df.groupby(var1)[var2]
            .value_counts(normalize=True)
            .round(2)
            .unstack()
        )

        #the species are the indexes of the shares dataframe
        values_var1 = tuple(shares.index)
        weight_counts = {category: shares[category].to_numpy() for category in shares.columns}

        #each part of the column is a subplot
        #ax = ax[i]
        #the bottom of the bar starts at 0, want nb of columns corresponding to the nb of values
        bottom = np.zeros(len(values_var1))

        #convert values of var 1 in str for the legend
        values_var1 = [str(v) for v in shares.index]

        #go through each value of weight_counts
        for boolean, weight_count in weight_counts.items():
            p = ax.bar(values_var1, weight_count, label=boolean, bottom=bottom)
            ax.bar_label(p, fmt="%.2f", label_type="center")
            #first add values for male, then for female
            bottom += weight_count

        ax.set_title(f"Share of {var2} in each {var1}")
        ax.legend(loc="upper right")

    plt.tight_layout()        
    plt.show()
>>>>>>> 02a657cec0d069d149bea883cb7a8402d53bfd1a
