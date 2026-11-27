#%%
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so

# For the machine learning part
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier

sns.set_style("ticks") 
sns.set_context("talk")
sns.set_theme(style="whitegrid")

# Path for the data on the GitHub repository
path = Path(__file__).parent / "data" / "churn_bank.csv"
df = pd.read_csv(path)

print(df.head())

# First part of the script contains many exploratory data analysis graphs and tables, 
# and the second part contains the code for the cleaned graphs of the report, as well as the 
# code for the random forest model that predicts churn based on a customer's features.

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

# The churn rate among the bank's customers is 20.37 %. 

#%% Question 2: How are the different variables (gender, age, geography, etc.) distributed in the dataset?

# For the numerical variables: 
print(df.select_dtypes(include = np.number).describe().T.to_string())

#For the categorical variables
categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Exited"]

# Shares of all the categorical variables (bar plots)
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
# For the numerical variables: 

numerical_var = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
#descriptive statistics
print(df[numerical_var].describe())

#histograms
for var in numerical_var:
   plot = sns.histplot(data = df, x = var, binwidth=5 if var == "Age" else None)
   plot.set_title(f"Distribution of {var}")
   plt.show()

#%%
# Distribution of exited and non exited customers by categorical variables (bar plots)

## all the plots on one grid
plt.suptitle("Distributions of exited and non exited", fontsize=18, y=0.95)

for i, var in enumerate(categorical_var):

   #create subplots
   plt.subplot(3,2, i + 1)

   plot = sns.countplot(data = df, x = var, hue = "Exited", palette="viridis")
   #add percentage labels
   plot.bar_label(plot.containers[0], fmt="%.2f", padding = 3, label_type="center")

plt.tight_layout()
plt.show()


#%%
# How are different indicators distributed by countries

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
            p = ax.bar(values_var1, weight_count, label=boolean, bottom=bottom, color = sns.color_palette("viridis", len(weight_counts))[list(weight_counts).index(boolean)])
            ax.bar_label(p, fmt="%.2f", label_type="center")
            #first add values for male, then for female
            bottom += weight_count

        ax.set_title(f"Share of {var2} in each {var1}")
        ax.legend(loc="upper right")

    plt.tight_layout()  

    plt.show()

# %%
# Clean graphs after having explored the data 

# Distributions of the numerical variables
numerical_var = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

# clean names for the x-axis labels
num_labels = {
    "CreditScore": "Credit score",
    "Age": "Age (years)",
    "Tenure": "Tenure (years)",
    "Balance": "Balance (€)",
    "EstimatedSalary": "Estimated salary (€)",
}

print(df[numerical_var].describe())   

fig = plt.figure(figsize=(20, 10))  

for i, var in enumerate(numerical_var):
    plt.subplot(2, 3, i + 1)
    plot = sns.histplot(data=df, x=var,
                        binwidth=5 if var == "Age" else None,
                        discrete=(var == "Tenure"))   # Tenure = full years
    plot.set_xlabel(num_labels[var])
    plot.set_ylabel("Number of customers")

plt.tight_layout()
plt.show()

#%%
# Distribution of the categorical variables (shares)

# Rename the categorical variables for the legends of the graphs
df_plot = df.copy()
df_plot["Status"] = df_plot["Exited"].map({0: "Stayed", 1: "Exited"})
df_plot["HasCrCard"] = df_plot["HasCrCard"].map({0: "No", 1: "Yes"})
df_plot["IsActiveMember"] = df_plot["IsActiveMember"].map({0: "Inactive", 1: "Active"})

# names for the x-axis labels
x_labels = {
    "Gender": "Gender",
    "Geography": "Country",
    "HasCrCard": "Has a credit card",
    "IsActiveMember": "Activity status",
    "NumOfProducts": "Number of products",
    "Status": "Customer status",
}

#For the categorical variables
plt.figure(figsize=(20, 11))

for i, var in enumerate(x_labels):

   #create subplots
   plt.subplot(2, 3, i + 1)

   plot = sns.countplot(data = df_plot, x = var, stat = "proportion", color = "C0")
   #add percentage labels
   plot.bar_label(plot.containers[0], fmt="%.2f", label_type="center", fontsize=16)
   plot.set_xlabel(x_labels[var])
   #clearer than label "proportion"
   plot.set_ylabel("Percentage")
   #so that each plot has the same scale
   plot.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
# Shares of the different categorical variables by customer status (exited or not) and country
x_labels["Status"] = "Customer status"
categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Status"]
x_axis_var = ["Status", "Geography"] 

for var1 in x_axis_var:
    for var2 in [v for v in categorical_var if v != var1]:
        shares = (
            df_plot.groupby(var1)[var2]
            .value_counts(normalize=True)
            .round(2)
            .unstack()
        )

        # create an empty figure
        fig, ax = plt.subplots()
        # each bar starts at 0, segments are stacked on top
        bottom = np.zeros(len(shares.index))
        # x-axis values as text so they are treated as categories
        values_var1 = [str(v) for v in shares.index]

        for category in shares.columns:
            weight_count = shares[category].to_numpy()
            p = ax.bar(values_var1, weight_count, label=category, bottom=bottom, 
                       color=sns.color_palette("viridis", len(shares.columns))[list(shares.columns).index(category)])
            ax.bar_label(p, fmt="%.2f", label_type="center")
            bottom += weight_count

        ax.set_title(f"Share of {x_labels[var2]} by {x_labels[var1]}")
        ax.set_xlabel(x_labels[var1])
        ax.set_ylabel("Percentage")
        ax.legend(title=x_labels[var2], loc="upper right")
        plt.show()

x_labels["Status"] = "Customer status"
categorical_var = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "NumOfProducts", "Status"]
x_axis_var = ["Status", "Geography"]

for var1 in x_axis_var:

    fig = plt.figure(figsize=(20, 10))

    for i, var2 in enumerate([v for v in categorical_var if v != var1]):
        ax = plt.subplot(2, 3, i + 1)

        shares = (
            df_plot.groupby(var1)[var2]
            .value_counts(normalize=True)
            .round(2)
            .unstack()
        )

        bottom = np.zeros(len(shares.index))
        values_var1 = [str(v) for v in shares.index]

        for category in shares.columns:
            weight_count = shares[category].to_numpy()
            p = ax.bar(values_var1, weight_count, label=category, bottom=bottom, 
                       color=sns.color_palette("viridis", len(shares.columns))[list(shares.columns).index(category)])
            ax.bar_label(p, fmt="%.2f", label_type="center")
            bottom += weight_count

        ax.set_title(x_labels[var2])
        ax.set_ylabel("Share")
        ax.set_ylim(0, 1)
        # legend on the right side of the plot
        ax.legend(title=x_labels[var2], loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    plt.show()

# %%
numerical_var = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
num_labels = {
    "CreditScore": "Credit score",
    "Age": "Age (years)",
    "Tenure": "Tenure (years)",
    "Balance": "Balance (€)",
    "EstimatedSalary": "Estimated salary (€)",
}

# Densities by customer status 
fig = plt.figure(figsize=(18, 10))
for i, var in enumerate(numerical_var):
    plt.subplot(2, 3, i + 1)
    plot = sns.kdeplot(data=df_plot, x=var, hue="Status", hue_order=["Stayed", "Exited"],
                       fill=True, common_norm=False, palette="viridis")
    plot.set_xlabel(num_labels[var])
    plot.set_ylabel("Density")
    handles = plot.legend_.legend_handles
    plot.legend_.remove()

fig.legend(handles, ["Stayed", "Exited"], title="Customer status", loc="center",
           bbox_to_anchor=(0.83, 0.28), fontsize=14, title_fontsize=14)
plt.tight_layout()
plt.show()

# Mean of each numerical variable by status (printed table)
print(df_plot.groupby("Status")[numerical_var].mean().round(1))

# %%
# We focus on the variables that matter for churn.
# Each bar = churn rate of the group (barplot with y="Exited" shows the mean,
# and the mean of a 0/1 column is the share of customers who left).

# Age and balance groups (IsActiveMember is already "Active" / "Inactive" in df_plot)
df_plot["AgeGroup"] = pd.cut(df_plot["Age"], bins=[17, 30, 40, 50, 60, 100],
                             labels=["18-30", "31-40", "41-50", "51-60", "60+"])
df_plot["BalanceGroup"] = pd.cut(df_plot["Balance"], bins=[-1, 0, 100000, 150000, 300000],
                                 labels=["empty", "0-100k", "100-150k", "150k+"])
df_plot["Products"] = df_plot["NumOfProducts"].astype(str)

fig, axes = plt.subplots(2, 2, figsize=(20, 13))

# a) Age x activity: being inactive matters more and more with age
sns.barplot(data=df_plot, x="AgeGroup", y="Exited", hue="IsActiveMember",
            errorbar=None, palette="viridis", edgecolor="black", ax=axes[0, 0])
axes[0, 0].set(title="a) Age x activity", xlabel="Age group")
axes[0, 0].legend(title="Activity status")

# b) Age x gender: women churn more at every age 
sns.barplot(data=df_plot, x="AgeGroup", y="Exited", hue="Gender",
            errorbar=None, palette="viridis", edgecolor="black", ax=axes[0, 1])
axes[0, 1].set(title="b) Age x gender", xlabel="Age group")

# c) Balance x country
sns.barplot(data=df_plot, x="BalanceGroup", y="Exited", hue="Geography",
            errorbar=None, palette="viridis", edgecolor="black", ax=axes[1, 0])
axes[1, 0].set(title="c) Balance x country", xlabel="Balance (€)")
axes[1, 0].legend(title="Country")

# d) Balance x products (1 or 2 products = 97% of customers)
sns.barplot(data=df_plot[df_plot["NumOfProducts"] <= 2], x="BalanceGroup", y="Exited",
            hue="Products", errorbar=None, palette="viridis", edgecolor="black", ax=axes[1, 1])
axes[1, 1].set(title="d) Balance x number of products", xlabel="Balance (€)")
axes[1, 1].legend(title="Number of products")

for ax in axes.flat:
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, 1)
    ax.axhline(df["Exited"].mean(), color="black", ls="--", lw=1, zorder=0)   # bank average 20%
    for bars in ax.containers:
        ax.bar_label(bars, fmt="{:.0%}", fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()

# %%
# We now rank the segments from the riskiest to the safest, based on churn rate. 
# Ranking: churn rate of each country x gender x activity group,
# from the riskiest to the safest (one color per country)
seg = df_plot.groupby(["Geography", "Gender", "IsActiveMember"])["Exited"].mean().reset_index()
seg["Segment"] = seg["Geography"] + ", " + seg["Gender"] + ", " + seg["IsActiveMember"]
seg = seg.sort_values("Exited", ascending=False)

plt.figure(figsize=(11, 8))
plot = sns.barplot(data=seg, x="Exited", y="Segment", hue="Geography",
                   palette="viridis", edgecolor="black")
for bars in plot.containers:
    plot.bar_label(bars, fmt="{:.0%}", padding=5)
plot.axvline(df["Exited"].mean(), color="black", ls="--", lw=1, zorder=0, label="bank average")
plot.set(xlabel="Churn rate", ylabel="")
plot.legend(title="Country", loc="lower right")
sns.despine()
plt.tight_layout()
plt.show()

# %% Inputs lucile - analysis of churn
# we saw that gender and country play the most role in explaining the churn rate, let's look at the interaction between the 2

#computing the table with the churn rates
churn_country_gender = df.pivot_table(
    index = "Geography",
    columns = "Gender",
    values = "Exited",
    #mean because exit is binary, *100 to get %
    aggfunc = "mean") * 100

#esthetics
churn_country_gender = churn_country_gender.round(2)
churn_country_gender.columns = ["Female churn (%)", "Male churn (%)"]
churn_country_gender.index.name = "Country"

print(churn_country_gender)

# Germany has the highest churn rate for both gender
# Female have higher churn rate in all countries

# %% In the explanatory analysis we saw an interesting behaviour of number of products with respect to exiting or not
#computing the table with the churn rates
churn_nb_products = df.pivot_table(
    index = "NumOfProducts",
    values = "Exited",
    #mean because exit is binary
    aggfunc = ["mean", "count"])

#esthetics
churn_nb_products = churn_nb_products.round(2)
churn_nb_products.columns = ["Churn rate (%)", "Count"]
churn_nb_products["Churn rate (%)"] = churn_nb_products["Churn rate (%)"] * 100
churn_nb_products.index.name = "Number of Products"

print(churn_nb_products)

# intersting to see that most churn is for 3 and 4 products BUT have to keep in mind 
# that little sample in those 2 categories

# look at among those who exited, what is the repartition of the nb of products 
plot = sns.countplot(data = df[df["Exited"] == 1], x = "Exited", hue = "NumOfProducts", palette="viridis")
plt.show()

# %% In the explanatory analysis we saw an interesting behaviour with active/inactive customers
#computing the table with the churn rates
churn_active = df.pivot_table(
    index = "IsActiveMember",
    values = "Exited",
    #mean because exit is binary, *100 to get %
    aggfunc = "mean")*100

#esthetics
churn_active = churn_active.round(2)
churn_active.columns = ["Churn rate (%)"]
churn_active.index.name = "Active member"

print(churn_active)

#so mainly inactive members, in explanatory statistics we saw that the distribution of this variable was even so we can compare both values
# %% Age of the members
df["AgeGroup"] = pd.cut(df["Age"],
                        bins=[17, 30, 40, 50, 60, 100],
                        labels=["18-30", "31-40", "41-50", "51-60", "60+"])

#computing the table with the churn rates
churn_age = df.pivot_table(
    index = "AgeGroup",
    values = "Exited",
    #mean because exit is binary, *100 to get %
    aggfunc = "mean")*100

#esthetics
churn_age = churn_age.round(2)
churn_age.columns = ["Churn rate (%)"]
churn_age.index.name = "Age"

print(churn_age)

#%% Age and country
#computing the table with the churn rates
churn_country_age = df.pivot_table(
    index = "AgeGroup",
    columns = "Geography",
    values = "Exited",
    #mean because exit is binary, *100 to get %
    aggfunc = "mean") * 100

#esthetics
churn_country_age = churn_country_age.round(2)
churn_country_age.columns = ["France churn (%)", "Germany churn (%)", "Spain churn (%)"]
churn_country_age.index.name = "Age "

print(churn_country_age)

# %% Age and Gender
#computing the table with the churn rates
churn_gender_age = df.pivot_table(
    index = "AgeGroup",
    columns = "Gender",
    values = "Exited",
    #mean because exit is binary, *100 to get %
    aggfunc = "mean") * 100

#esthetics
churn_gender_age = churn_gender_age.round(2)
churn_gender_age.columns = ["Female churn (%)", "Male churn (%)"]
churn_gender_age.index.name = "Age "

print(churn_gender_age)

# %%
# Question 7 (optional): A simple model that predicts churn (random forest model)

# Prepare the data: drop the ids, turn text columns into 0/1 columns
df_model = df.drop(columns=["CustomerId", "Surname", "AgeGroup"])
df_model = pd.get_dummies(df_model, columns=["Geography", "Gender"], drop_first=True, dtype=int)
X = df_model.drop(columns="Exited")
Y = df_model["Exited"]

# 70% of the customers to learn, 30% to test (and same churn rate in both parts)
X_learn, X_test, Y_learn, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.3, random_state=0, stratify=Y)

# Random forest of 300 trees:
# Only 20% of the customers leave: accuracy is misleading (predicting "stays" for 
# everyone already gives 80%), so we give more weight to churners (class_weight="balanced") 
# and choose the tree size with the balanced accuracy, by stratified 5-fold cross-validation 
# on the learning set.
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
leaf_grid = {"min_samples_leaf": [1, 5, 10, 20, 50, 100, 200]}
cv_folds = model_selection.StratifiedKFold(5, shuffle=True, random_state=0)
rf_tuned = model_selection.GridSearchCV(rf, leaf_grid, cv=cv_folds, n_jobs=-1,
                                        scoring="balanced_accuracy")
rf_tuned.fit(X_learn, Y_learn)
print("Best parameter:", rf_tuned.best_params_,
      "CV balanced accuracy:", round(rf_tuned.best_score_, 3))

# Evaluation on the test customers (never seen during the learning phase)
Y_pred = rf_tuned.predict(X_test)
Y_proba = rf_tuned.predict_proba(X_test)[:, 1]
print("Baseline accuracy (predict 'stays' for everyone):", round(1 - Y_test.mean(), 3))
print("Accuracy:         ", round(metrics.accuracy_score(Y_test, Y_pred), 3))
print("Balanced accuracy:", round(metrics.balanced_accuracy_score(Y_test, Y_pred), 3))
print("Precision:        ", round(metrics.precision_score(Y_test, Y_pred), 3))
print("Recall:           ", round(metrics.recall_score(Y_test, Y_pred), 3))
print("AUC:              ", round(metrics.roc_auc_score(Y_test, Y_proba), 3))

# Confusion matrix (rows = true status, columns = predicted status) and ROC curve
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
metrics.ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred, ax=axes[0],
                                                display_labels=["Stayed", "Exited"],
                                                cmap="Blues", colorbar=False)
axes[0].set_title("Confusion matrix")
axes[0].grid(False)   # no whitegrid lines on top of the matrix
metrics.RocCurveDisplay.from_predictions(Y_test, Y_proba, ax=axes[1], name="Random forest")
axes[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
axes[1].set(title="ROC curve",
            xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)")
plt.tight_layout()
plt.show()

# What does the model use? (importance = how much the variable helps to separate the groups)
var_names = {
    "CreditScore": "Credit score",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "NumOfProducts": "Number of products",
    "HasCrCard": "Has a credit card",
    "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary",
    "Geography_Germany": "Germany (vs France)",
    "Geography_Spain": "Spain (vs France)",
    "Gender_Male": "Male (vs Female)",
}
results = pd.DataFrame({"Variable": X.columns.map(var_names),
                        "Importance": rf_tuned.best_estimator_.feature_importances_})
results = results.sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(data=results, x="Importance", y="Variable", color="C0", edgecolor="black")
plt.ylabel("")
sns.despine()
plt.tight_layout()
plt.show()

# %%
