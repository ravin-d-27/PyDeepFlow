def activation(x, func, device):
    if func == 'relu':
        # ReLU activation: returns max(0, x)
        return device.maximum(0, x)
    elif func == 'leaky_relu':
        # Leaky ReLU: allows small gradients for negative inputs (0.01 * x)
        return device.where(x > 0, x, 0.01 * x)
    elif func == 'sigmoid':
        # Sigmoid: 1 / (1 + exp(-x))
        return 1 / (1 + device.exp(-x))
    elif func == 'tanh':
        # Tanh: hyperbolic tangent
        return device.tanh(x)
    elif func == 'softmax':
        # Softmax: for numerical stability, subtract max(x) from x
        exp_x = device.exp(x - device.max(x, axis=-1, keepdims=True))
        return exp_x / device.sum(exp_x, axis=-1, keepdims=True)
    else:
        # Raise error if unsupported activation is passed
        raise ValueError(f"Unsupported activation function: {func}")

def activation_derivative(x, func, device):
    if func == 'relu':
        # Derivative of ReLU: 1 if x > 0, otherwise 0
        return device.where(x > 0, 1, 0)
    elif func == 'leaky_relu':
        # Derivative of Leaky ReLU: 1 if x > 0, otherwise 0.01
        return device.where(x > 0, 1, 0.01)
    elif func == 'sigmoid':
        # Derivative of Sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        return x * (1 - x)
    elif func == 'tanh':
        # Derivative of Tanh: tanh'(x) = 1 - tanh(x)^2
        return 1 - x ** 2
    elif func == 'softmax':
        # Derivative of softmax: softmax'(x) = softmax(x) * (1 - softmax(x))
        # Often used in combination with cross-entropy, so exact derivative may vary based on loss function.
        return x * (1 - x)
    else:
        # Raise error if unsupported activation is passed
        raise ValueError(f"Unsupported activation derivative: {func}")
