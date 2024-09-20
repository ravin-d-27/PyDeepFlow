import numpy as np

def activation(x, func):
    if func == 'relu':
        return np.maximum(0, x)
    elif func == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"Unsupported activation function: {func}")

def activation_derivative(x, func):
    if func == 'relu':
        return np.where(x > 0, 1, 0)
    elif func == 'sigmoid':
        return x * (1 - x)
    else:
        raise ValueError(f"Unsupported activation derivative: {func}")
