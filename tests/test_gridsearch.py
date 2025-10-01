import unittest
import numpy as np
from pydeepflow.gridSearch import GridSearchCV
from pydeepflow.model import Multi_Layer_ANN

class TestGridSearchCV(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # for reproducibility
        self.X = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y = np.random.randint(0, 2, size=(100, 1))  # Binary classification

    def test_fit_and_finds_best_params(self):
        """
        Test if GridSearchCV runs, finds best parameters, and the process is consistent.
        """
        param_grid = {
            'hidden_layers': [[5], [10]],
            'activations': [['relu']],
            'l2_lambda': [0.0, 0.01]
        }

        grid_search = GridSearchCV(
            model_class=Multi_Layer_ANN,
            param_grid=param_grid,
            scoring='accuracy',
            cv=2 
        )

        grid_search.fit(self.X, self.y)

        self.assertIsNotNone(grid_search.best_params, "best_params should be set after fitting.")
        self.assertNotEqual(grid_search.best_score, -np.inf, "best_score should be updated after fitting.")
        self.assertIn(grid_search.best_params['hidden_layers'], param_grid['hidden_layers'])
        self.assertIn(grid_search.best_params['l2_lambda'], param_grid['l2_lambda'])

        print(f"\nGridSearchCV test passed with best params: {grid_search.best_params}")
        print(f"Best score: {grid_search.best_score:.4f}")

if __name__ == '__main__':
    unittest.main()
