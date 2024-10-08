def activation(x, func, device):
    """
    Applies the specified activation function to the input data.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data to which the activation function will be applied.
    func : str
        The activation function to apply. Supported values include 'relu', 'leaky_relu', 'sigmoid', 'tanh', and 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.

    Returns:
    --------
    np.ndarray
        The result of applying the specified activation function to the input data.

    Raises:
    -------
    ValueError
        If the specified activation function is unsupported.

    Notes:
    ------
    - ReLU returns max(0, x).
    - Leaky ReLU allows gradients for negative inputs (0.01 * x).
    - Sigmoid returns 1 / (1 + exp(-x)).
    - Tanh returns the hyperbolic tangent of x.
    - Softmax is numerically stabilized by subtracting the max value in x before computing the exponentials.
    """
    if func == 'relu':
        return device.maximum(0, x)
    elif func == 'leaky_relu':
        return device.where(x > 0, x, 0.01 * x)
    elif func == 'sigmoid':
        return 1 / (1 + device.exp(-x))
    elif func == 'tanh':
        return device.tanh(x)
    elif func == 'softmax':
        exp_x = device.exp(x - device.max(x, axis=-1, keepdims=True))
        return exp_x / device.sum(exp_x, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported activation function: {func}")


def activation_derivative(x, func, device):
    """
    Computes the derivative of the specified activation function.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data on which the derivative will be computed.
    func : str
        The activation function whose derivative is to be computed. Supported values include 'relu', 'leaky_relu', 'sigmoid', 'tanh', and 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.

    Returns:
    --------
    np.ndarray
        The derivative of the activation function applied to the input data.

    Raises:
    -------
    ValueError
        If the specified activation function's derivative is unsupported.

    Notes:
    ------
    - The ReLU derivative is 1 if x > 0, otherwise 0.
    - The Leaky ReLU derivative is 1 if x > 0, otherwise 0.01.
    - The Sigmoid derivative is sigmoid(x) * (1 - sigmoid(x)).
    - The Tanh derivative is 1 - tanh(x)^2.
    - The Softmax derivative assumes usage with cross-entropy loss, resulting in softmax(x) * (1 - softmax(x)).
    """
    if func == 'relu':
        return device.where(x > 0, 1, 0)
    elif func == 'leaky_relu':
        return device.where(x > 0, 1, 0.01)
    elif func == 'sigmoid':
        return x * (1 - x)
    elif func == 'tanh':
        return 1 - x ** 2
    elif func == 'softmax':
        return x * (1 - x)
    else:
        raise ValueError(f"Unsupported activation derivative: {func}")
