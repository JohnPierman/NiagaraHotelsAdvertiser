import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../bruteforcemodelselection/Updated_Data.csv'
data = pd.read_csv(file_path)

# Remove formatting from numeric columns
data = data.replace({',': '', '\$': '', '%': ''}, regex=True)  # Remove formatting
data = data.apply(pd.to_numeric)  # Convert all columns to numeric type

# Select only relevant columns for the analysis
selected_columns = [6, 8, 11]  # Second column (y) and the selected features
data_filtered = data.iloc[:, selected_columns]

# Separate features (X) and target (y)
y = data_filtered.iloc[:, 0].values  # First column in the filtered data
X = data_filtered.iloc[:, 1:].values  # Remaining columns in the filtered data

# Standardize the features (subtract mean and divide by std)
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

# Add intercept term for regression
X_with_const_standardized = sm.add_constant(X_standardized)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = ["Intercept"] + [data.columns[i] for i in selected_columns[1:]]
vif_data["VIF"] = [variance_inflation_factor(X_with_const_standardized, i) for i in range(X_with_const_standardized.shape[1])]

# Print VIF
print("Variance Inflation Factors (VIF) after standardization:\n", vif_data)

# Linear regression summary
model = sm.OLS(y, X_with_const_standardized)
results = model.fit()

print("\nLinear Regression Summary:\n")
print(results.summary())

# Descriptive Statistics for filtered data
desc_stats = data_filtered.describe()
print("\nDescriptive Statistics:\n")
print(desc_stats)

# Correlation Matrix for filtered data
correlation_matrix = data_filtered.corr()
print("\nCorrelation Matrix:\n")
print(correlation_matrix)
