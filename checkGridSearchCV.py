
import numpy as np
from pydeepflow.gridSearch import GridSearchCV
from pydeepflow.model import Multi_Layer_ANN
    # Assuming X and Y are your training data and labels
X_train = np.random.rand(100, 20)  # Example feature data
Y_train = np.random.randint(0, 2, size=(100, 1))  # Example binary labels

# Define the parameter grid
param_grid = {
    'hidden_layers': [[5,5], [20,20], [10, 10]],  # Different configurations of hidden layers
    'activations': [['relu','relu'], ['tanh','relu'], ['sigmoid','relu']],  # Different activation functions
    'l2_lambda': [0.0, 0.01],  # Different regularization strengths
    'dropout_rate': [0.0, 0.5]  # Different dropout rates
}

# Initialize GridSearchCV
grid_search = GridSearchCV(Multi_Layer_ANN, param_grid, scoring='accuracy', cv=3)

# Fit Grid Search
grid_search.fit(X_train, Y_train)