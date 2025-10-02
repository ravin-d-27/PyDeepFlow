# cross_validator.py
import numpy as np
from sklearn.model_selection import KFold


class CrossValidator:
    def __init__(self, n_splits=5):
        """
        Initialize the CrossValidator with the number of splits.

        Args:
            n_splits (int): The number of folds for cross-validation.
        """
        self.n_splits = n_splits

    def split(self, X, y):
        """
        Generate indices for k-fold cross-validation.

        Args:
            X (array-like): Feature data.
            y (array-like): Labels.

        Returns:
            generator: A generator yielding train and validation indices.
        """
        kf = KFold(n_splits=self.n_splits)
        for train_index, val_index in kf.split(X):
            yield train_index, val_index

    def get_metrics(self, y_true, y_pred, metrics):
        """
        Calculate and return specified metrics.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            metrics (list): List of metrics to calculate.

        Returns:
            dict: A dictionary containing the requested metrics.
        """
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results["accuracy"] = np.mean(y_true == y_pred)

        return results
