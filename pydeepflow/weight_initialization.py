import numpy as np

class WeightInializer:
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

    def __init__(self, shape):
        """
        Initialize the WeightInializer with the desired shape.

        Parameters
        ----------
        shape : tuple
            Shape of the weight matrix (e.g., (input_dim, output_dim)).
        """
        self.shape = shape

    def zeros(self):
        """
        Initialize weights to all zeros.

        Returns
        -------
        np.ndarray
            Weight matrix of zeros.
        """
        return np.zeros(self.shape)
    
    def ones(self):
        """
        Initialize weights to all ones.

        Returns
        -------
        np.ndarray
            Weight matrix of ones.
        """
        return np.ones(self.shape)
    
    def random_initialize(self, method='normal', low=-0.1, high=0.1):
        """
        Randomly initialize weights using a uniform or normal distribution.

        Parameters
        ----------
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
            return np.random.uniform(low, high, self.shape)
        elif method == 'normal':
            return np.random.normal(0, 0.01, self.shape)
        else:
            raise ValueError("Method must be 'uniform' or 'normal'")

    def xavier_initialize(self, method='normal'):
        """
        Initialize weights using Xavier/Glorot initialization.

        Parameters
        ----------
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
        if len(self.shape) != 2:
            raise ValueError("Xavier initialization works for 2D layers only (e.g., fully connected layers).")
        
        fan_in = self.shape[0]
        fan_out = self.shape[1]
        
        if method == 'normal':
            std = np.sqrt(2. / (fan_in + fan_out))
            return np.random.randn(fan_in, fan_out) * std
        elif method == 'uniform':
            limit = np.sqrt(6. / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))
        else:
            raise ValueError("Unsupported method. Use 'normal' or 'uniform'.")

    def he_initialize(self, method='normal'):
        """
        Initialize weights using He initialization.

        Parameters
        ----------
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
        fan_in = self.shape[0]
        fan_out = self.shape[1]

        if len(self.shape) != 2:
            raise ValueError("He initialization works for 2D layers only (e.g., fully connected layers).")

        if method == 'normal':
            std = np.sqrt(2. / fan_in)
            return np.random.randn(fan_in, fan_out) * std
        elif method == 'uniform':
            limit = np.sqrt(6. / fan_in)
            return np.random.uniform(-limit, limit, (fan_in, fan_out))
        else:
            raise ValueError("Unsupported method. Use 'normal' or 'uniform'.")



