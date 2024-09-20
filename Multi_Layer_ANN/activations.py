import numpy as np

def activation(x, func):
    if func == 'relu':
        return np.maximum(0, x)
    elif func == 'leaky_relu':
        return np.where(x > 0, x, 0.01 * x)
    elif func == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif func == 'tanh':
        return np.tanh(x)
    elif func == 'softmax':
        exp_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported activation function: {func}")

def activation_derivative(x, func):
    if func == 'relu':
        return np.where(x > 0, 1, 0)
    elif func == 'leaky_relu':
        return np.where(x > 0, 1, 0.01)
    elif func == 'sigmoid':
        return x * (1 - x)
    elif func == 'tanh':
        return 1 - np.tanh(x) ** 2
    elif func == 'softmax':
        # Softmax derivative is more complex and usually handled in combination with cross-entropy loss.
        # If needed, this can be customized for specific cases.
        return x * (1 - x)
    else:
        raise ValueError(f"Unsupported activation derivative: {func}")
