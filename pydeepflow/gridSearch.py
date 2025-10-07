import numpy as np
from itertools import product
from pydeepflow.cross_validator import CrossValidator


class GridSearchCV:
    def __init__(self, model_class, param_grid, scoring='accuracy', cv=3):
        """
        Initializes the GridSearchCV class.

        Parameters:
            model_class: The model class to be used for fitting.
            param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
            scoring (str): Scoring method to evaluate the model. Options are 'accuracy', 'loss', etc.
            cv (int): Number of cross-validation folds.
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params = None
        self.best_score = -np.inf

    def _get_score(self, results):
        """
        Extract the scoring metric from the model's evaluation results.

        Parameters:
            results (dict): Evaluation results returned by the model.

        Returns:
            float: The computed score for the specified scoring metric.
        """
        if self.scoring == 'accuracy':
            return results['accuracy']
        elif self.scoring == 'loss':
            # Lower loss is better, so negate it for consistency
            return -results['loss']
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring}")

    def fit(self, X, y):
        """
        Fit the model with the best hyperparameters using Grid Search.

        Parameters:
            X (array-like): Feature data.
            y (array-like): Target data.
        """
        # Generate all combinations of parameters
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        cross_validator = CrossValidator(n_splits=self.cv)

        for params in param_combinations:
            params_dict = dict(zip(param_names, params))
            print(f"Testing parameters: {params_dict}")
            
            fold_scores = []
            # Iterating over the pre-defined folds
            for train_index, val_index in cross_validator.split(X, y):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                model = self.model_class(X_train, y_train, **params_dict)
                model.fit(epochs=10)
                
                results = model.evaluate(X_val, y_val, metrics=[self.scoring])
                fold_scores.append(self._get_score(results))

            avg_score = np.mean(fold_scores)
            print(f"Average score for parameters {params_dict}: {avg_score:.4f}\n")
            
            # Update best score and parameters if applicable
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params_dict

        print(f"Best parameters: {self.best_params} with score: {self.best_score:.4f}")
