import numpy as np
"""
    Implements various neural network weight initialization strategies.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix to initialize, e.g., (input_dim, output_dim).

    Notes
    -----
    - Avoid initializing weights to zero or any constant value to prevent symmetry.
    - Choose initialization methods based on the activation functions used.
    - Consider network depth to avoid vanishing/exploding gradients.
    - Avoid weights that are too large (exploding gradients) or too small (vanishing gradients).
    - Avoid weights that are too sparse (dead neurons) or too dense (overfitting).

    Methods
    -------
    zeros()
        Initializes weights to all zeros.
    ones()
        Initializes weights to all ones.
    random_initialize(method='normal', low=-0.1, high=0.1)
        Initializes weights randomly using a uniform or normal distribution.
    xavier_initialize(method='normal')
        Initializes weights using Xavier/Glorot initialization (best for tanh/sigmoid).
    he_initialize(method='normal')
        Initializes weights using He initialization (best for relu).
    """

def zeros(shape):
    """
    Initialize weights to all zeros.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (e.g., (input_dim, output_dim)).

    Returns
    -------
    np.ndarray
        Weight matrix of zeros.
    """
    return np.zeros(shape)

def ones(shape):
    """
    Initialize weights to all ones.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (e.g., (input_dim, output_dim)).

    Returns
    -------
    np.ndarray
        Weight matrix of ones.
    """
    return np.ones(shape)

def random_initialize(shape, method='normal', low=-0.1, high=0.1):
    """
    Randomly initialize weights using a uniform or normal distribution.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix.
    method : str, optional
        'uniform' or 'normal' (default is 'normal').
    low : float, optional
        Lower bound for uniform distribution (default is -0.1).
    high : float, optional
        Upper bound for uniform distribution (default is 0.1).

    Returns
    -------
    np.ndarray
        Randomly initialized weight matrix.

    Raises
    ------
    ValueError
        If method is not 'uniform' or 'normal'.
    """
    if method == 'uniform':
        return np.random.uniform(low, high, shape)
    elif method == 'normal':
        return np.random.normal(0, 0.01, shape)
    else:
        raise ValueError("Method must be 'uniform' or 'normal'")

def xavier_initialize(shape, method='normal'):
    """
    Initialize weights using Xavier/Glorot initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).
    method : str, optional
        'normal' or 'uniform' (default is 'normal').

    Returns
    -------
    np.ndarray
        Xavier-initialized weight matrix.

    Raises
    ------
    ValueError
        If shape is not 2D or method is unsupported.
    """
    if len(shape) != 2:
        raise ValueError("Xavier initialization works for 2D layers only (e.g., fully connected layers).")
    
    fan_in, fan_out = shape
    if method == 'normal':
        std = np.sqrt(2. / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std
    elif method == 'uniform':
        limit = np.sqrt(6. / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    else:
        raise ValueError("Unsupported method. Use 'normal' or 'uniform'.")

def he_initialize(shape, method='normal'):
    """
    Initialize weights using He initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).
    method : str, optional
        'normal' or 'uniform' (default is 'normal').

    Returns
    -------
    np.ndarray
        He-initialized weight matrix.

    Raises
    ------
    ValueError
        If shape is not 2D or method is unsupported.
    """
    if len(shape) != 2:
        raise ValueError("He initialization works for 2D layers only (e.g., fully connected layers).")
    
    fan_in, fan_out = shape
    if method == 'normal':
        std = np.sqrt(2. / fan_in)
        return np.random.randn(fan_in, fan_out) * std
    elif method == 'uniform':
        limit = np.sqrt(6. / fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    else:
        raise ValueError("Unsupported method. Use 'normal' or 'uniform'.")



