import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

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

        for params in param_combinations:
            params_dict = dict(zip(param_names, params))
            print(f"Testing parameters: {params_dict}")
            
            # Perform cross-validation
            scores = []
            for _ in range(self.cv):
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)
                
                # Initialize model with current parameters
                model = self.model_class(X_train, y_train, **params_dict)
                model.fit(epochs=10)  # Adjust epochs as needed
                
                # Evaluate the model
                val_loss, val_accuracy = model.evaluate(X_val, y_val)
                
                # Store the score based on the scoring metric
                if self.scoring == 'accuracy':
                    scores.append(val_accuracy)
                elif self.scoring == 'loss':
                    scores.append(-val_loss)  # Assuming lower loss is better

            avg_score = np.mean(scores)
            print(f"Average score for parameters {params_dict}: {avg_score:.4f}")
            print()
            
            # Update best score and parameters if applicable
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params_dict

        print(f"Best parameters: {self.best_params} with score: {self.best_score:.4f}")
