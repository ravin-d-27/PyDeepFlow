import numpy as np

def activation(x, func, device, alpha=0.01):
    """
    Applies the specified activation function to the input data.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data to which the activation function will be applied.
    func : str
        The activation function to apply.
        Supported values: 'relu', 'leaky_relu', 'prelu', 'elu', 'gelu', 'swish', 'selu',
                          'softplus', 'mish', 'rrelu', 'hardswish', 'sigmoid', 'softsign',
                          'tanh', 'hardtanh', 'hardsigmoid', 'tanhshrink', 'softshrink',
                          'hardshrink', 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.
    alpha : float, optional (default=0.01)
        Parameter used for activations like Leaky ReLU, PReLU, and RReLU.

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
    - ReLU (Rectified Linear Unit): Returns 0 for negative inputs, otherwise returns the input value.
    - Leaky ReLU: Similar to ReLU, but allows a small negative slope (alpha * x) for x < 0.
    - PReLU (Parametric ReLU): Similar to Leaky ReLU but with a learnable parameter for the negative slope.
    - ELU (Exponential Linear Unit): Applies exponential transformation for x < 0, linear for x > 0.
    - GELU (Gaussian Error Linear Unit): Approximates a Gaussian error function. Smooth curve activation.
    - Swish: Uses sigmoid(x) * x, introducing smooth non-linearity.
    - SELU: Scaled ELU with fixed scaling factors to promote self-normalization in neural networks.
    - Softplus: Smooth approximation of ReLU, calculated as log(1 + exp(x)).
    - Mish: A newer activation that uses x * tanh(softplus(x)).
    - RReLU (Randomized Leaky ReLU): A randomized variant of Leaky ReLU used mainly for regularization.
    - HardSwish: Similar to Swish but uses a piecewise linear approximation for faster computation.
    - Sigmoid: S-shaped curve that maps any input to the range (0, 1).
    - Softsign: Another S-shaped function, but uses x / (1 + |x|) for smoother transitions.
    - Tanh: Maps input to the range (-1, 1) with a hyperbolic tangent curve.
    - HardTanh: Similar to Tanh but clamped to the range [-1, 1].
    - HardSigmoid: A faster approximation of sigmoid, producing values in the range (0, 1).
    - Tanhshrink: Subtracts the tanh activation from the input: x - tanh(x).
    - Softshrink: Shrinks the values towards zero by a threshold of alpha.
    - Hardshrink: Similar to Softshrink but with hard cutoffs at alpha and -alpha.
    - Softmax: Maps input to a probability distribution by exponentiating and normalizing the inputs.
    """
    if func == 'relu':
        return device.maximum(0, x)
    elif func == 'leaky_relu':
        return device.where(x > 0, x, alpha * x)
    elif func == 'prelu':
        return device.where(x > 0, x, alpha * x)
    elif func == 'elu':
        return device.where(x > 0, x, alpha * (device.exp(x) - 1))
    elif func == 'gelu':
        return 0.5 * x * (1 + device.tanh(device.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) 
    elif func == 'swish':
        return x / (1 + device.exp(-x))
    elif func == 'selu':
        lam = 1.0507
        alpha_selu = 1.67326
        return lam * device.where(x > 0, x, alpha_selu * (device.exp(x) - 1))
    elif func == 'softplus':
        return device.log(1 + device.exp(x))
    elif func == 'mish':
        return x * device.tanh(device.log(1 + device.exp(x)))
    elif func == 'rrelu':
        return device.where(x > 0, x, alpha * x)
    elif func == 'hardswish':
        return x * device.where(x > 3, 1, device.where(x < -3, 0, (x + 3) / 6))
    elif func == 'sigmoid':
        return 1 / (1 + device.exp(-x))
    elif func == 'softsign':
        return x / (1 + device.abs(x))
    elif func == 'tanh':
        return device.tanh(x)
    elif func == 'hardtanh':
        return device.where(x > 1, 1, device.where(x < -1, -1, x))
    elif func == 'hardsigmoid':
        return device.where(x > 1, 1, device.where(x < -1, 0, (x + 1) / 2))
    elif func == 'tanhshrink':
        return x - device.tanh(x)
    elif func == 'softshrink':
        return device.where(device.abs(x) > alpha, x - alpha * device.sign(x), 0)
    elif func == 'hardshrink':
        return device.where(device.abs(x) > alpha, x, 0)
    elif func == 'softmax':
        exp_x = device.exp(x - device.max(x, axis=-1, keepdims=True))
        return exp_x / device.sum(exp_x, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported activation function: {func}")

def activation_derivative(x, func, device, alpha=0.01):
    """
    Computes the derivative of the specified activation function.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data on which the derivative will be computed.
    func : str
        The activation function whose derivative is to be computed.
        Supported values: 'relu', 'leaky_relu', 'prelu', 'elu', 'gelu', 'swish', 'selu',
                          'softplus', 'mish', 'rrelu', 'hardswish', 'sigmoid', 'softsign',
                          'tanh', 'hardtanh', 'hardsigmoid', 'tanhshrink', 'softshrink',
                          'hardshrink', 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.
    alpha : float, optional (default=0.01)
        Parameter used for activations like Leaky ReLU, PReLU, and RReLU.

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
    - The Leaky ReLU derivative is 1 if x > 0, otherwise alpha.
    - The PReLU derivative is 1 if x > 0, otherwise alpha.
    - The ELU derivative is 1 if x > 0, otherwise alpha * exp(x).
    - The GELU derivative is complex but approximated as a smooth function using tanh and polynomials.
    - The Swish derivative is sigmoid(x) + x * sigmoid'(x).
    - The SELU derivative is lam if x > 0, otherwise lam * alpha_selu * exp(x).
    - The Softplus derivative is sigmoid(x).
    - The Mish derivative is a combination of tanh(softplus(x)) and x.
    - The RReLU derivative is 1 if x > 0, otherwise alpha.
    - The HardSwish derivative is a piecewise function that ranges from 0 to 1, depending on x.
    - The Sigmoid derivative is sigmoid(x) * (1 - sigmoid(x)).
    - The Softsign derivative is 1 / (1 + |x|)^2.
    - The Tanh derivative is 1 - tanh(x)^2.
    - The HardTanh derivative is 1 in the range [-1, 1], otherwise 0.
    - The HardSigmoid derivative is 0.5 for x in [-1, 1], otherwise 0.
    - The Tanhshrink derivative is 1 - tanh(x)^2.
    - The Softshrink and Hardshrink derivatives are 1 where |x| > alpha, otherwise 0.
    - The Softmax derivative assumes usage with cross-entropy loss, resulting in softmax(x) * (1 - softmax(x)).
    """
    if func == 'relu':
        return device.where(x > 0, 1, 0)
    elif func == 'leaky_relu':
        return device.where(x > 0, 1, alpha)
    elif func == 'prelu':
        return device.where(x > 0, 1, alpha)
    elif func == 'elu':
        return device.where(x > 0, 1, alpha * device.exp(x))
    elif func == 'gelu':
        return 0.5 * (1 + device.tanh(device.sqrt(2 / device.pi) * (x + 0.044715 * x ** 3))) + \
               0.5 * x * (1 - device.tanh(device.sqrt(2 / device.pi) * (x + 0.044715 * x ** 3)) ** 2)
    elif func == 'swish':
        sigma = 1 / (1 + device.exp(-x))
        return sigma + x * sigma * (1 - sigma)
    elif func == 'selu':
        lam = 1.0507
        alpha_selu = 1.67326
        return lam * device.where(x > 0, 1, alpha_selu * device.exp(x))
    elif func == 'softplus':
        return 1 / (1 + device.exp(-x))
    elif func == 'mish':
        sp = device.log(1 + device.exp(x))
        tanh_sp = device.tanh(sp)
        return device.exp(x) * (tanh_sp + x * (1 - tanh_sp ** 2) / sp) / (1 + device.exp(-x))
    elif func == 'rrelu':
        return device.where(x > 0, 1, alpha)
    elif func == 'hardswish':
        return device.where(x > -3, device.where(x < 3, x / 3 + 0.5, 1), 0)
    elif func == 'sigmoid':
        return x * (1 - x)
    elif func == 'softsign':
        return 1 / (1 + device.abs(x)) ** 2
    elif func == 'tanh':
        return 1 - x ** 2
    elif func == 'hardtanh':
        return device.where(device.abs(x) <= 1, 1, 0)
    elif func == 'hardsigmoid':
        return device.where(device.abs(x) <= 1, 0.5, 0)
    elif func == 'tanhshrink':
        return 1 - device.tanh(x) ** 2
    elif func == 'softshrink':
        return device.where(device.abs(x) > alpha, 1, 0)
    elif func == 'hardshrink':
        return device.where(device.abs(x) > alpha, 1, 0)
    elif func == 'softmax':
        return x * (1 - x)
    else:
        raise ValueError(f"Unsupported activation derivative: {func}")
