import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from itertools import combinations


class BruteForceSubsetSelection:
    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_splits = n_splits  # Number of splits for cross-validation

    def calculate_cross_validated_mse(self, subset):
        """Calculate the cross-validated MSE for a given subset of features."""
        if len(subset) == 0:
            return float('inf')

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        mse_scores = []

        for train_index, val_index in kf.split(self.X):
            X_train, X_val = self.X[train_index][:, subset], self.X[val_index][:, subset]
            y_train, y_val = self.y[train_index], self.y[val_index]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse_scores.append(mean_squared_error(y_val, y_pred))

        return np.mean(mse_scores)  # Return the average MSE over all folds

    def brute_force_search(self):
        """Perform brute-force search to find the best subset of features with the lowest cross-validated MSE."""
        best_subset = None
        best_mse = float('inf')
        all_results = []

        # Iterate over all possible non-empty subsets
        for r in range(1, self.n_features + 1):
            for subset in combinations(range(self.n_features), r):
                mse = self.calculate_cross_validated_mse(subset)
                all_results.append((list(subset), mse))
                if mse < best_mse:
                    best_mse = mse
                    best_subset = subset

        return list(best_subset), best_mse, all_results

    @staticmethod
    def load_data(filename):
        """Load data from a CSV file."""
        data = pd.read_csv(filename)
        X = data.iloc[:, 2:].values
        y = data.iloc[:, 1].values
        return X, y

    @staticmethod
    def save_results(filename, best_subset, best_mse, all_results):
        """Save the best subset, cross-validated MSE, and all subset MSEs to a CSV file."""
        results_df = pd.DataFrame(all_results, columns=['Subset', 'Cross-Validated MSE'])
        results_df['Best_Subset'] = str(best_subset)
        results_df['Best_MSE'] = best_mse
        results_df.to_csv(filename, index=False)


# Load the data
X, y = BruteForceSubsetSelection.load_data('Updated_Data.csv')

# Perform brute-force subset selection with cross-validation
selector = BruteForceSubsetSelection(X, y)
best_subset, best_mse, all_results = selector.brute_force_search()

# Save the results including all subset cross-validated MSEs
BruteForceSubsetSelection.save_results('best_subset_results.csv', best_subset, best_mse, all_results)
