import numpy as np
from dataclasses import dataclass
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


@dataclass
class InitializationMetadata:
    """
    Metadata about layer weight initialization.
    
    Stores comprehensive information about how a layer's weights and biases
    were initialized, including the method used, activation function, shape,
    and scaling factors.
    
    Attributes
    ----------
    layer_index : int
        Index of the layer in the network (0-based).
    layer_type : str
        Type of layer ('dense' for fully connected, 'conv' for convolutional).
    method : str
        Initialization method used (e.g., 'he_normal', 'xavier_uniform').
    activation : str
        Activation function for this layer (e.g., 'relu', 'sigmoid').
    shape : tuple
        Shape of the weight matrix/tensor.
        - For dense layers: (input_dim, output_dim)
        - For conv layers: (kernel_h, kernel_w, in_channels, out_channels)
    bias_value : float
        Value used for bias initialization.
    fan_in : int
        Number of input connections to each neuron.
    fan_out : int
        Number of output connections from each neuron.
    scale : float
        Scaling factor used in the initialization (e.g., std for normal, limit for uniform).
    
    Examples
    --------
    >>> metadata = InitializationMetadata(
    ...     layer_index=0,
    ...     layer_type='dense',
    ...     method='he_normal',
    ...     activation='relu',
    ...     shape=(784, 128),
    ...     bias_value=0.01,
    ...     fan_in=784,
    ...     fan_out=128,
    ...     scale=0.0535
    ... )
    >>> print(metadata)
    Layer 0 (dense): he_normal for relu, shape=(784, 128), scale=0.0535
    """
    layer_index: int
    layer_type: str
    method: str
    activation: str
    shape: tuple
    bias_value: float
    fan_in: int
    fan_out: int
    scale: float
    
    def __str__(self):
        """
        Return a human-readable string representation of the metadata.
        
        Returns
        -------
        str
            Formatted string with key initialization details.
        
        Examples
        --------
        >>> metadata = InitializationMetadata(0, 'dense', 'he_normal', 'relu',
        ...                                   (784, 128), 0.01, 784, 128, 0.0535)
        >>> str(metadata)
        'Layer 0 (dense): he_normal for relu, shape=(784, 128), scale=0.0535'
        """
        return (f"Layer {self.layer_index} ({self.layer_type}): "
                f"{self.method} for {self.activation}, "
                f"shape={self.shape}, scale={self.scale:.4f}")


# Mapping from activation functions to optimal initialization methods
ACTIVATION_INIT_MAP = {
    # ReLU family - use He initialization
    'relu': 'he_normal',
    'leaky_relu': 'he_normal',
    'prelu': 'he_normal',
    'elu': 'he_normal',
    'rrelu': 'he_normal',
    
    # SELU - use LeCun initialization
    'selu': 'lecun_normal',
    
    # Sigmoid/Tanh family - use Xavier initialization
    'sigmoid': 'xavier_normal',
    'tanh': 'xavier_normal',
    'softsign': 'xavier_normal',
    'hardtanh': 'xavier_normal',
    'hardsigmoid': 'xavier_normal',
    'tanhshrink': 'xavier_normal',
    
    # Modern activations - use He initialization
    'gelu': 'he_normal',
    'swish': 'he_normal',
    'mish': 'he_normal',
    'hardswish': 'he_normal',
    
    # Other activations - use Xavier as safe default
    'softplus': 'xavier_normal',
    'softshrink': 'xavier_normal',
    'hardshrink': 'xavier_normal',
    'softmax': 'xavier_normal',
    'linear': 'xavier_normal',
}

# Bias initialization based on activation function
ACTIVATION_BIAS_MAP = {
    'relu': 0.01,  # Small positive to prevent dead neurons
    'leaky_relu': 0.01,
    'prelu': 0.01,
    'elu': 0.0,
    'selu': 0.0,
    'sigmoid': 0.0,
    'tanh': 0.0,
    'gelu': 0.0,
    'swish': 0.0,
    'mish': 0.0,
    'softmax': 0.0,
    'linear': 0.0,
    # Default for others
    'default': 0.0
}

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
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers

    Returns
    -------
    np.ndarray
        Xavier-initialized weight matrix/tensor (normal distribution).

    Raises
    ------
    ValueError
        If shape is not 2D or 4D.
    """
    fan_in, fan_out = calculate_fan(shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def XavierUniform(shape):
    """
    Initialize weights using Xavier/Glorot uniform initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers

    Returns
    -------
    np.ndarray
        Xavier-initialized weight matrix/tensor (uniform distribution).

    Raises
    ------
    ValueError
        If shape is not 2D or 4D.
    """
    fan_in, fan_out = calculate_fan(shape)
    limit = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)



def HeNormal(shape):
    """
    Initialize weights using He normal initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers

    Returns
    -------
    np.ndarray
        He-initialized weight matrix/tensor (normal distribution).

    Raises
    ------
    ValueError
        If shape is not 2D or 4D.
    """
    fan_in, fan_out = calculate_fan(shape)
    std = np.sqrt(2. / fan_in)
    return np.random.randn(*shape) * std

def HeUniform(shape):
    """
    Initialize weights using He uniform initialization.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers

    Returns
    -------
    np.ndarray
        He-initialized weight matrix/tensor (uniform distribution).

    Raises
    ------
    ValueError
        If shape is not 2D or 4D.
    """
    fan_in, fan_out = calculate_fan(shape)
    limit = np.sqrt(6. / fan_in)
    return np.random.uniform(-limit, limit, shape)


def calculate_fan(shape):
    """
    Calculate fan_in and fan_out for any weight shape (2D or 4D).
    
    For 2D shapes (dense layers): shape = (fan_in, fan_out)
    For 4D shapes (conv layers): shape = (kernel_h, kernel_w, in_channels, out_channels)
    
    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_height, kernel_width, in_channels, out_channels) for conv layers
    
    Returns
    -------
    tuple
        (fan_in, fan_out) where:
        - For 2D: fan_in = shape[0], fan_out = shape[1]
        - For 4D: fan_in = kernel_h * kernel_w * in_channels
                  fan_out = kernel_h * kernel_w * out_channels
    
    Raises
    ------
    ValueError
        If shape is not 2D or 4D.
    """
    if len(shape) == 2:
        # Dense layer: (input_dim, output_dim)
        fan_in, fan_out = shape
    elif len(shape) == 4:
        # Convolutional layer: (kernel_h, kernel_w, in_channels, out_channels)
        kernel_h, kernel_w, in_channels, out_channels = shape
        receptive_field_size = kernel_h * kernel_w
        fan_in = receptive_field_size * in_channels
        fan_out = receptive_field_size * out_channels
    else:
        raise ValueError(
            f"Cannot calculate fan for shape {shape}. "
            f"Expected 2D (dense) or 4D (conv) shape, got {len(shape)}D."
        )
    
    return fan_in, fan_out


def LeCunNormal(shape):
    """
    Initialize weights using LeCun normal initialization.
    
    Optimal for SELU activation function. Uses variance = 1 / fan_in.
    
    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers
    
    Returns
    -------
    np.ndarray
        LeCun-initialized weight matrix (normal distribution).
    
    Notes
    -----
    LeCun initialization is designed for SELU activations and helps maintain
    self-normalizing properties of the network.
    """
    fan_in, fan_out = calculate_fan(shape)
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


def LeCunUniform(shape):
    """
    Initialize weights using LeCun uniform initialization.
    
    Optimal for SELU activation function. Uses uniform distribution with
    limit = sqrt(3 / fan_in).
    
    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix/tensor.
        - 2D: (input_dim, output_dim) for dense layers
        - 4D: (kernel_h, kernel_w, in_channels, out_channels) for conv layers
    
    Returns
    -------
    np.ndarray
        LeCun-initialized weight matrix (uniform distribution).
    
    Notes
    -----
    LeCun initialization is designed for SELU activations and helps maintain
    self-normalizing properties of the network.
    """
    fan_in, fan_out = calculate_fan(shape)
    limit = np.sqrt(3.0 / fan_in)
    return np.random.uniform(-limit, limit, shape)


def get_weight_initializer(name, shape):
    """
    Retrieve a weight initialization function by name.

    Parameters
    ----------
    name : str
        Name of the weight initialization method. Supported names include:
        'zeros', 'ones', 'random_normal', 'random_uniform', 'xavier_normal',
        'xavier_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'.
    shape : tuple
        Shape of the weight matrix/tensor to initialize.

    Returns
    -------
    np.ndarray
        Initialized weight matrix/tensor.

    Raises
    ------
    ValueError
        If the provided name does not match any supported initialization method.
    """
    initializers = {
        'zeros': zeros(shape),
        'ones': ones(shape),
        'random_normal': RandomNormal(shape),
        'random_uniform': RandomUniform(shape),
        'xavier_normal': XavierNormal(shape),
        'xavier_uniform': XavierUniform(shape),
        'he_normal': HeNormal(shape),
        'he_uniform': HeUniform(shape),
        'lecun_normal': LeCunNormal(shape),
        'lecun_uniform': LeCunUniform(shape),
        'glorot_normal': XavierNormal(shape),  # Alias for Xavier normal initialization
        'glorot_uniform': XavierUniform(shape)  # Alias for Xavier uniform initialization
    }
    if name not in initializers:
        raise ValueError(f"Unsupported weight initializer: {name}. Supported initializers are: {list(initializers.keys())}")
    return initializers[name]


def calculate_scale(method, fan_in, fan_out):
    """
    Calculate the scaling factor used by an initialization method.
    
    Parameters
    ----------
    method : str
        Initialization method name.
    fan_in : int
        Number of input connections.
    fan_out : int
        Number of output connections.
    
    Returns
    -------
    float
        Scaling factor (std for normal distributions, limit for uniform distributions).
    
    Notes
    -----
    - He Normal: std = sqrt(2 / fan_in)
    - He Uniform: limit = sqrt(6 / fan_in)
    - Xavier Normal: std = sqrt(2 / (fan_in + fan_out))
    - Xavier Uniform: limit = sqrt(6 / (fan_in + fan_out))
    - LeCun Normal: std = sqrt(1 / fan_in)
    - LeCun Uniform: limit = sqrt(3 / fan_in)
    - Others: 0.0 (not applicable)
    """
    if method in ['he_normal']:
        return np.sqrt(2.0 / fan_in)
    elif method in ['he_uniform']:
        return np.sqrt(6.0 / fan_in)
    elif method in ['xavier_normal', 'glorot_normal']:
        return np.sqrt(2.0 / (fan_in + fan_out))
    elif method in ['xavier_uniform', 'glorot_uniform']:
        return np.sqrt(6.0 / (fan_in + fan_out))
    elif method in ['lecun_normal']:
        return np.sqrt(1.0 / fan_in)
    elif method in ['lecun_uniform']:
        return np.sqrt(3.0 / fan_in)
    elif method == 'random_normal':
        return 0.01  # Default std
    elif method == 'random_uniform':
        return 0.1  # Default limit
    else:
        return 0.0  # Not applicable for zeros, ones, etc.


class WeightInitializer:
    """
    Orchestrates weight initialization for neural network layers.
    
    Supports automatic selection based on activation functions or manual
    specification of initialization methods.
    
    Parameters
    ----------
    device : Device
        Device object for array operations (CPU or GPU).
    mode : str, optional
        Initialization mode. Either 'auto' for activation-based selection
        or 'manual' for explicit method specification (default is 'auto').
    method : str or list, optional
        Specific initialization method(s) when mode='manual'.
        Can be a single string for all layers or a list for layer-specific methods.
    bias_init : str or float, optional
        Bias initialization strategy. Options:
        - 'auto': Activation-aware defaults (default)
        - 'zeros': All zeros
        - float: Constant value for all biases
    
    Attributes
    ----------
    device : Device
        Device object for array operations.
    mode : str
        Initialization mode ('auto' or 'manual').
    method : str or list
        Initialization method(s) to use.
    bias_init : str or float
        Bias initialization strategy.
    layer_index : int
        Counter for tracking layer indices during initialization.
    
    Examples
    --------
    >>> from pydeepflow.device import Device
    >>> device = Device(use_gpu=False)
    >>> 
    >>> # Automatic initialization based on activation
    >>> initializer = WeightInitializer(device, mode='auto')
    >>> weights, biases, metadata = initializer.initialize_dense_layer(784, 128, 'relu')
    >>> 
    >>> # Manual initialization with specific method
    >>> initializer = WeightInitializer(device, mode='manual', method='xavier_normal')
    >>> weights, biases, metadata = initializer.initialize_dense_layer(784, 128, 'sigmoid')
    """
    
    def __init__(self, device, mode='auto', method=None, bias_init='auto'):
        """
        Initialize the WeightInitializer.
        
        Parameters
        ----------
        device : Device
            Device object for array operations.
        mode : str, optional
            'auto' for activation-based selection, 'manual' for explicit method.
        method : str or list, optional
            Specific initialization method (when mode='manual').
        bias_init : str or float, optional
            Bias initialization strategy ('auto', 'zeros', or float value).
        """
        self.device = device
        self.mode = mode
        self.method = method
        self.bias_init = bias_init
        self.layer_index = 0
    
    def get_method_for_activation(self, activation):
        """
        Determine the best initialization method for an activation function.
        
        Uses the ACTIVATION_INIT_MAP to select optimal initialization based on
        the activation function. Falls back to Xavier initialization for unknown
        activations with a warning.
        
        Parameters
        ----------
        activation : str
            Activation function name (e.g., 'relu', 'sigmoid', 'tanh').
        
        Returns
        -------
        str
            Initialization method name (e.g., 'he_normal', 'xavier_normal').
        
        Notes
        -----
        If the activation function is not recognized, the method defaults to
        'xavier_normal' as a safe fallback and issues a warning.
        
        Examples
        --------
        >>> initializer = WeightInitializer(device)
        >>> initializer.get_method_for_activation('relu')
        'he_normal'
        >>> initializer.get_method_for_activation('sigmoid')
        'xavier_normal'
        >>> initializer.get_method_for_activation('selu')
        'lecun_normal'
        """
        if activation in ACTIVATION_INIT_MAP:
            return ACTIVATION_INIT_MAP[activation]
        else:
            # Fallback to Xavier for unknown activations
            import warnings
            warnings.warn(
                f"Unknown activation '{activation}' for automatic initialization. "
                f"Defaulting to Xavier initialization.",
                UserWarning
            )
            return 'xavier_normal'
    
    def get_bias_value(self, activation):
        """
        Determine bias initialization value based on activation function.
        
        Uses the ACTIVATION_BIAS_MAP to select appropriate bias initialization.
        For ReLU-family activations, uses small positive values (0.01) to prevent
        dead neurons. For most other activations, uses zeros.
        
        Parameters
        ----------
        activation : str
            Activation function name (e.g., 'relu', 'sigmoid', 'tanh').
        
        Returns
        -------
        float
            Bias initialization value.
        
        Notes
        -----
        - ReLU, LeakyReLU, PReLU: 0.01 (small positive to prevent dead neurons)
        - Most other activations: 0.0
        - Unknown activations: 0.0 (default)
        
        Examples
        --------
        >>> initializer = WeightInitializer(device)
        >>> initializer.get_bias_value('relu')
        0.01
        >>> initializer.get_bias_value('sigmoid')
        0.0
        """
        if self.bias_init == 'auto':
            # Use activation-aware bias initialization
            return ACTIVATION_BIAS_MAP.get(activation, ACTIVATION_BIAS_MAP['default'])
        elif self.bias_init == 'zeros':
            return 0.0
        elif isinstance(self.bias_init, (int, float, np.number)):
            return float(self.bias_init)
        else:
            # Fallback to zeros for invalid values
            return 0.0
    
    def initialize_dense_layer(self, input_dim, output_dim, activation):
        """
        Initialize weights and biases for a dense (fully connected) layer.
        
        Selects the appropriate initialization method based on the mode:
        - 'auto': Automatically selects based on activation function
        - 'manual': Uses the specified method
        
        Parameters
        ----------
        input_dim : int
            Number of input neurons.
        output_dim : int
            Number of output neurons.
        activation : str
            Activation function name for this layer.
        
        Returns
        -------
        tuple
            (weight_matrix, bias_vector, metadata) where:
            - weight_matrix: np.ndarray of shape (input_dim, output_dim)
            - bias_vector: np.ndarray of shape (output_dim,)
            - metadata: InitializationMetadata object with initialization details
        
        Raises
        ------
        ValueError
            If the initialization method is unsupported or produces invalid values.
        
        Examples
        --------
        >>> initializer = WeightInitializer(device, mode='auto')
        >>> W, b, meta = initializer.initialize_dense_layer(784, 128, 'relu')
        >>> W.shape
        (784, 128)
        >>> b.shape
        (128,)
        """
        # Determine which initialization method to use
        if self.mode == 'auto':
            method = self.get_method_for_activation(activation)
        elif self.mode == 'manual':
            if isinstance(self.method, list):
                # Layer-specific initialization (will be used in task 9)
                if self.layer_index < len(self.method):
                    method = self.method[self.layer_index]
                else:
                    raise ValueError(
                        f"Layer index {self.layer_index} exceeds method list length {len(self.method)}"
                    )
            else:
                method = self.method
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'auto' or 'manual'.")
        
        # Initialize weights using the selected method
        shape = (input_dim, output_dim)
        try:
            weight_matrix = get_weight_initializer(method, shape)
        except ValueError as e:
            raise ValueError(
                f"Failed to initialize layer {self.layer_index} with method '{method}': {str(e)}"
            )
        
        # Validate for numerical stability
        if np.isnan(weight_matrix).any() or np.isinf(weight_matrix).any():
            raise RuntimeError(
                f"Weight initialization produced invalid values (NaN or Inf) "
                f"for layer {self.layer_index} using method '{method}'"
            )
        
        # Initialize biases
        bias_value = self.get_bias_value(activation)
        bias_vector = np.full(output_dim, bias_value)
        
        # Create metadata
        fan_in, fan_out = calculate_fan(shape)
        scale = calculate_scale(method, fan_in, fan_out)
        metadata = InitializationMetadata(
            layer_index=self.layer_index,
            layer_type='dense',
            method=method,
            activation=activation,
            shape=shape,
            bias_value=bias_value,
            fan_in=fan_in,
            fan_out=fan_out,
            scale=scale
        )
        
        # Increment layer index for next layer
        self.layer_index += 1
        
        return weight_matrix, bias_vector, metadata
    
    def initialize_conv_layer(self, kernel_h, kernel_w, in_channels, out_channels, activation):
        """
        Initialize weights and biases for a convolutional layer.
        
        Properly calculates fan_in and fan_out for 4D convolutional weights
        and applies the selected initialization method with appropriate scaling.
        
        Parameters
        ----------
        kernel_h : int
            Kernel height.
        kernel_w : int
            Kernel width.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (filters).
        activation : str
            Activation function name for this layer.
        
        Returns
        -------
        tuple
            (weight_tensor, bias_vector, metadata) where:
            - weight_tensor: np.ndarray of shape (kernel_h, kernel_w, in_channels, out_channels)
            - bias_vector: np.ndarray of shape (out_channels,)
            - metadata: InitializationMetadata object with initialization details
        
        Raises
        ------
        ValueError
            If the initialization method is unsupported or produces invalid values.
        
        Examples
        --------
        >>> initializer = WeightInitializer(device, mode='auto')
        >>> W, b, meta = initializer.initialize_conv_layer(3, 3, 1, 32, 'relu')
        >>> W.shape
        (3, 3, 1, 32)
        >>> b.shape
        (32,)
        """
        # Determine which initialization method to use
        if self.mode == 'auto':
            method = self.get_method_for_activation(activation)
        elif self.mode == 'manual':
            if isinstance(self.method, list):
                # Layer-specific initialization (will be used in task 9)
                if self.layer_index < len(self.method):
                    method = self.method[self.layer_index]
                else:
                    raise ValueError(
                        f"Layer index {self.layer_index} exceeds method list length {len(self.method)}"
                    )
            else:
                method = self.method
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'auto' or 'manual'.")
        
        # Initialize weights using the selected method
        # Conv layer shape: (kernel_h, kernel_w, in_channels, out_channels)
        shape = (kernel_h, kernel_w, in_channels, out_channels)
        try:
            weight_tensor = get_weight_initializer(method, shape)
        except ValueError as e:
            raise ValueError(
                f"Failed to initialize conv layer {self.layer_index} with method '{method}': {str(e)}"
            )
        
        # Validate for numerical stability
        if np.isnan(weight_tensor).any() or np.isinf(weight_tensor).any():
            raise RuntimeError(
                f"Conv weight initialization produced invalid values (NaN or Inf) "
                f"for layer {self.layer_index} using method '{method}'"
            )
        
        # Initialize biases
        bias_value = self.get_bias_value(activation)
        bias_vector = np.full(out_channels, bias_value)
        
        # Create metadata
        fan_in, fan_out = calculate_fan(shape)
        scale = calculate_scale(method, fan_in, fan_out)
        metadata = InitializationMetadata(
            layer_index=self.layer_index,
            layer_type='conv',
            method=method,
            activation=activation,
            shape=shape,
            bias_value=bias_value,
            fan_in=fan_in,
            fan_out=fan_out,
            scale=scale
        )
        
        # Increment layer index for next layer
        self.layer_index += 1
        
        return weight_tensor, bias_vector, metadata


# Temporary stub functions for backward compatibility
# These will be replaced by WeightInitializer class in subsequent tasks
def initialize_weights(shape, method='he_normal'):
    """
    Temporary stub function for backward compatibility.
    Will be replaced by WeightInitializer class.
    """
    return get_weight_initializer(method, shape)


def initialize_biases(shape, value=0.0):
    """
    Temporary stub function for backward compatibility.
    Will be replaced by WeightInitializer class.
    """
    if isinstance(value, (int, float)):
        return np.full(shape, value)
    return np.zeros(shape)


def get_initializer_for_activation(activation):
    """
    Temporary stub function for backward compatibility.
    Returns the optimal initialization method for a given activation function.
    Will be replaced by WeightInitializer class.
    """
    return ACTIVATION_INIT_MAP.get(activation, 'xavier_normal')
