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

def RandomNormal(shape, mean=0.0, std=0.01):
    """
    Randomly initialize weights using a normal (Gaussian) distribution.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix.
    mean : float, optional
        Mean of the normal distribution (default is 0.0).
    std : float, optional
        Standard deviation of the normal distribution (default is 0.01).

    Returns
    -------
    np.ndarray
        Randomly initialized weight matrix.
    """
    return np.random.normal(mean, std, shape)

def RandomUniform(shape, low=-0.1, high=0.1):
    """
    Randomly initialize weights using a uniform distribution.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix.
    low : float, optional
        Lower bound for uniform distribution (default is -0.1).
    high : float, optional
        Upper bound for uniform distribution (default is 0.1).

    Returns
    -------
    np.ndarray
        Randomly initialized weight matrix.
    """
    return np.random.uniform(low, high, shape)

def XavierNormal(shape):
    """
    Initialize weights using Xavier/Glorot normal initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).

    Returns
    -------
    np.ndarray
        Xavier-initialized weight matrix (normal distribution).

    Raises
    ------
    ValueError
        If shape is not 2D.
    """
    if len(shape) != 2:
        raise ValueError("Xavier initialization works for 2D layers only (e.g., fully connected layers).")
    fan_in, fan_out = shape
    std = np.sqrt(2. / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def XavierUniform(shape):
    """
    Initialize weights using Xavier/Glorot uniform initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).

    Returns
    -------
    np.ndarray
        Xavier-initialized weight matrix (uniform distribution).

    Raises
    ------
    ValueError
        If shape is not 2D.
    """
    if len(shape) != 2:
        raise ValueError("Xavier initialization works for 2D layers only (e.g., fully connected layers).")
    fan_in, fan_out = shape
    limit = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

glorot_normal = XavierNormal  # Alias for Xavier normal initialization
glorot_uniform = XavierUniform  # Alias for Xavier uniform initialization

def HeNormal(shape):
    """
    Initialize weights using He normal initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).

    Returns
    -------
    np.ndarray
        He-initialized weight matrix (normal distribution).

    Raises
    ------
    ValueError
        If shape is not 2D.
    """
    if len(shape) != 2:
        raise ValueError("He initialization works for 2D layers only (e.g., fully connected layers).")
    fan_in, fan_out = shape
    std = np.sqrt(2. / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def HeUniform(shape):
    """
    Initialize weights using He uniform initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must be 2D).

    Returns
    -------
    np.ndarray
        He-initialized weight matrix (uniform distribution).

    Raises
    ------
    ValueError
        If shape is not 2D.
    """
    if len(shape) != 2:
        raise ValueError("He initialization works for 2D layers only (e.g., fully connected layers).")
    fan_in, fan_out = shape
    limit = np.sqrt(6. / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))



