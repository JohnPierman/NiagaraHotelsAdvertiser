import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../bruteforcemodelselection/Updated_Data.csv'  # Replace with your updated dataset path
data = pd.read_csv(file_path)


data = data.replace({',': '', '\$': '', '%': ''}, regex=True)  # Remove formatting from numeric columns
data = data.apply(pd.to_numeric)  # Convert all columns to numeric type

# Create the interaction term (target variable)
data['Interaction'] = data['Distance to Niagara Falls (miles)'] * data['Average Pleasure Trip Distance']
y = data['Interaction'].values  # Set the interaction term as the target

# Specify the indices of the columns to use as features (omit the original target columns)
#input_indices = [3, 4, 5, 6, 7, 8, 9, 10]  # Indices of remaining feature columns
input_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10,11,12]  # Indices of remaining feature columns
X = data.iloc[:, input_indices].values  # Extract specified columns as features

# Standardize the features (subtract mean and divide by std)
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

# Add intercept term for regression
X_with_const_standardized = sm.add_constant(X_standardized)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = ["Intercept"] + [data.columns[i] for i in input_indices]
vif_data["VIF"] = [variance_inflation_factor(X_with_const_standardized, i) for i in range(X_with_const_standardized.shape[1])]

# Print VIF
print("Variance Inflation Factors (VIF) after standardization:\n", vif_data)

# Linear regression summary
model = sm.OLS(y, X_with_const_standardized)
results = model.fit()

print("\nLinear Regression Summary:\n")
print(results.summary())

# Descriptive Statistics
desc_stats = data.describe()
print("\nDescriptive Statistics:\n")
print(desc_stats)

# Correlation Matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:\n")
print(correlation_matrix)

# Heatmap for Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()
