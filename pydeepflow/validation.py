"""
Model validation utilities for PyDeepFlow.

This module provides comprehensive input validation functionality that can be used
across different model types (ANN, CNN, etc.) to ensure consistent validation
standards and reduce code duplication.
"""

import numpy as np
import warnings


class ModelValidator:
    """
    Comprehensive input validation for PyDeepFlow models.
    
    This class provides reusable validation methods that can be used across
    different model types to ensure consistent validation standards and
    reduce code duplication.
    """
    
    def __init__(self, device=None):
        """
        Initialize the ModelValidator.
        
        Args:
            device: Device object for array operations (optional)
        """
        self.device = device
    
    def validate_training_data(self, data, data_name, max_dimensions=4):
        """
        Validate training data (X_train or Y_train).
        
        Args:
            data: Training data to validate
            data_name (str): Name of the data for error messages
            max_dimensions (int): Maximum allowed dimensions (2 for ANN, 4 for CNN)
            
        Raises:
            ValueError: If data is invalid
            TypeError: If data type is incorrect
        """
        # Check if data exists
        if data is None:
            raise ValueError(f"{data_name} cannot be None")
            
        # Convert to numpy array if not already
        try:
            data_array = np.asarray(data)
        except Exception as e:
            raise TypeError(f"{data_name} must be convertible to numpy array. Error: {e}")
        
        # Check if empty
        if data_array.size == 0:
            raise ValueError(f"{data_name} cannot be empty")
            
        # Check for valid dimensions
        if data_array.ndim == 0:
            raise ValueError(f"{data_name} must be at least 1-dimensional")
        elif max_dimensions == 2 and data_array.ndim > 2:
            raise ValueError(f"{data_name} must be 1D or 2D array for ANN models, got {data_array.ndim}D")
        elif data_array.ndim > max_dimensions:
            raise ValueError(f"{data_name} must be at most {max_dimensions}D array, got {data_array.ndim}D")
            
        # Check for numeric data
        if not np.issubdtype(data_array.dtype, np.number):
            raise TypeError(f"{data_name} must contain numeric data, got dtype: {data_array.dtype}")
            
        # Check for NaN values
        if np.isnan(data_array).any():
            nan_indices = np.where(np.isnan(data_array))
            if len(nan_indices[0]) <= 10:  # Show first 10 NaN locations
                locations = list(zip(*nan_indices))
                raise ValueError(f"{data_name} contains NaN values at indices: {locations}")
            else:
                raise ValueError(f"{data_name} contains {np.isnan(data_array).sum()} NaN values")
                
        # Check for infinite values
        if np.isinf(data_array).any():
            inf_indices = np.where(np.isinf(data_array))
            if len(inf_indices[0]) <= 10:  # Show first 10 inf locations
                locations = list(zip(*inf_indices))
                raise ValueError(f"{data_name} contains infinite values at indices: {locations}")
            else:
                raise ValueError(f"{data_name} contains {np.isinf(data_array).sum()} infinite values")
    
    def validate_data_compatibility(self, X_train, Y_train):
        """
        Validate that X_train and Y_train are compatible.
        
        Args:
            X_train: Training input data
            Y_train: Training target data
            
        Raises:
            ValueError: If data is incompatible
        """
        X_array = np.asarray(X_train)
        Y_array = np.asarray(Y_train)
        
        # Check sample count compatibility
        if X_array.shape[0] != Y_array.shape[0]:
            raise ValueError(f"X_train and Y_train must have the same number of samples. "
                           f"Got X_train: {X_array.shape[0]}, Y_train: {Y_array.shape[0]}")
                           
        # Check minimum sample count
        if X_array.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples for training, got {X_array.shape[0]}")
            
        # Validate feature count for 2D input (ANN)
        if X_array.ndim == 2 and X_array.shape[1] == 0:
            raise ValueError("X_train must have at least 1 feature")
            
        # Validate target format for classification
        if Y_array.ndim == 2:
            if Y_array.shape[1] == 0:
                raise ValueError("Y_train must have at least 1 output dimension")
            # Check for valid one-hot encoding
            if Y_array.shape[1] > 1:
                # Should be one-hot encoded for multi-class
                row_sums = np.sum(Y_array, axis=1)
                if not np.allclose(row_sums, 1.0):
                    raise ValueError("For multi-class classification, Y_train should be one-hot encoded "
                                   "(each row should sum to 1)")
    
    def validate_hidden_layers(self, hidden_layers):
        """
        Validate hidden layer configuration.
        
        Args:
            hidden_layers: List of hidden layer sizes
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If types are incorrect
        """
        if not isinstance(hidden_layers, (list, tuple)):
            raise TypeError(f"hidden_layers must be a list or tuple, got {type(hidden_layers)}")
            
        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers cannot be empty. Specify at least one hidden layer.")
            
        for i, layer_size in enumerate(hidden_layers):
            if not isinstance(layer_size, (int, np.integer)):
                raise TypeError(f"All hidden layer sizes must be integers. "
                              f"Layer {i} has type {type(layer_size)}")
                              
            if layer_size <= 0:
                raise ValueError(f"All hidden layer sizes must be positive integers. "
                               f"Layer {i} has size {layer_size}")
                               
            if layer_size > 10000:  # Reasonable upper limit
                raise ValueError(f"Hidden layer size seems too large: {layer_size} neurons in layer {i}. "
                               f"Consider using smaller layers for better performance.")
    
    def validate_activations(self, activations, hidden_layers):
        """
        Validate activation functions.
        
        Args:
            activations: List of activation function names
            hidden_layers: List of hidden layer sizes
            
        Raises:
            ValueError: If activations are invalid
            TypeError: If types are incorrect
        """
        if not isinstance(activations, (list, tuple)):
            raise TypeError(f"activations must be a list or tuple, got {type(activations)}")
            
        if len(activations) != len(hidden_layers):
            raise ValueError(f"Number of activation functions ({len(activations)}) must match "
                           f"number of hidden layers ({len(hidden_layers)})")
                           
        # Valid activation functions (from activations.py)
        valid_activations = {
            'relu', 'leaky_relu', 'prelu', 'elu', 'gelu', 'swish', 'selu',
            'softplus', 'mish', 'rrelu', 'hardswish', 'sigmoid', 'softsign',
            'tanh', 'hardtanh', 'hardsigmoid', 'tanhshrink', 'softshrink',
            'hardshrink', 'softmax'
        }
        
        for i, activation in enumerate(activations):
            if not isinstance(activation, str):
                raise TypeError(f"Activation functions must be strings. "
                              f"Activation {i} has type {type(activation)}")
                              
            if activation not in valid_activations:
                raise ValueError(f"Unsupported activation function '{activation}' at index {i}. "
                               f"Supported activations: {sorted(valid_activations)}")
    
    def validate_loss_function(self, loss):
        """
        Validate loss function.
        
        Args:
            loss (str): Loss function name
            
        Raises:
            ValueError: If loss function is invalid
            TypeError: If type is incorrect
        """
        if not isinstance(loss, str):
            raise TypeError(f"Loss function must be a string, got {type(loss)}")
            
        # Valid loss functions (from losses.py)
        valid_losses = {'binary_crossentropy', 'mse', 'categorical_crossentropy', 'hinge', 'huber'}
        
        if loss not in valid_losses:
            raise ValueError(f"Unsupported loss function '{loss}'. "
                           f"Supported losses: {sorted(valid_losses)}")
    
    def validate_regularization_params(self, l2_lambda, dropout_rate):
        """
        Validate regularization parameters.
        
        Args:
            l2_lambda (float): L2 regularization parameter
            dropout_rate (float): Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If types are incorrect
        """
        # Validate L2 lambda
        if not isinstance(l2_lambda, (int, float, np.number)):
            raise TypeError(f"l2_lambda must be a number, got {type(l2_lambda)}")
            
        if l2_lambda < 0:
            raise ValueError(f"l2_lambda must be non-negative, got {l2_lambda}")
            
        if l2_lambda > 1:
            raise ValueError(f"l2_lambda seems too large: {l2_lambda}. "
                           f"Typical values are between 0 and 0.1")
            
        # Validate dropout rate
        if not isinstance(dropout_rate, (int, float, np.number)):
            raise TypeError(f"dropout_rate must be a number, got {type(dropout_rate)}")
            
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in range [0, 1), got {dropout_rate}")
    
    def validate_optimizer(self, optimizer):
        """
        Validate optimizer.
        
        Args:
            optimizer: Optimizer name (str) or optimizer object
            
        Raises:
            ValueError: If optimizer is invalid
            TypeError: If type is incorrect
        """
        # Allow both string names and optimizer objects for backward compatibility
        if isinstance(optimizer, str):
            valid_optimizers = {'sgd', 'adam', 'rmsprop'}
            if optimizer not in valid_optimizers:
                raise ValueError(f"Unsupported optimizer '{optimizer}'. "
                               f"Supported optimizers: {sorted(valid_optimizers)}")
        else:
            # Check if it's a valid optimizer object
            from pydeepflow.optimizers import Adam, RMSprop
            if not isinstance(optimizer, (Adam, RMSprop)) and optimizer != 'sgd':
                raise TypeError(f"optimizer must be a string or valid optimizer object, got {type(optimizer)}")
    
    def validate_training_hyperparameters(self, learning_rate, epochs, batch_size, X_train):
        """
        Validate training hyperparameters (learning rate, epochs, batch size).
        
        Args:
            learning_rate (float): Learning rate for optimization
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            X_train: Training data for batch size validation
            
        Returns:
            int: Adjusted batch size (if auto-adjusted)
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If types are incorrect
        """
        # Validate learning rate
        if not isinstance(learning_rate, (int, float, np.number)):
            raise TypeError(f"learning_rate must be a number, got {type(learning_rate)}")
            
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
            
        if learning_rate > 1.0:
            raise ValueError(f"learning_rate seems too large: {learning_rate}. "
                        f"Typical values are between 0.0001 and 0.1")
        
        if learning_rate < 1e-8:
            raise ValueError(f"learning_rate seems too small: {learning_rate}. "
                        f"Values below 1e-8 may prevent the model from learning effectively")
            
        # Validate epochs
        if not isinstance(epochs, (int, np.integer)):
            raise TypeError(f"epochs must be an integer, got {type(epochs)}")
            
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
            
        if epochs > 10000:
            raise ValueError(f"epochs seems too large: {epochs}. "
                        f"Training for more than 10000 epochs is rarely necessary. "
                        f"Consider using early stopping instead")
            
        # Validate batch size
        if not isinstance(batch_size, (int, np.integer)):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size)}")
            
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            
        # Get number of training samples
        X_array = np.asarray(X_train)
        n_samples = X_array.shape[0]
        
        # Auto-adjust batch_size if larger than dataset
        adjusted_batch_size = batch_size
        if batch_size > n_samples:
            warnings.warn(
                f"batch_size ({batch_size}) is larger than the number of training samples ({n_samples}). "
                f"Automatically adjusting batch_size to {n_samples}.",
                UserWarning
            )
            adjusted_batch_size = n_samples
            
        if batch_size > 1024:
            raise ValueError(f"batch_size seems too large: {batch_size}. "
                        f"Typical values are 16, 32, 64, 128, 256, or 512. "
                        f"Large batch sizes may cause memory issues and poor generalization")
            
        # Warning for very small batch sizes (not an error, but good to know)
        if batch_size == 1 and n_samples >= 100:
            warnings.warn(
                f"batch_size of 1 (online learning) with {n_samples} samples "
                f"may result in very slow and unstable training. "
                f"Consider using a larger batch size like 16, 32, or 64.",
                UserWarning,
                stacklevel=3
            )
        
        return adjusted_batch_size
    
    def validate_cnn_layers(self, layers_config):
        """
        Validate CNN layer configuration.
        
        Args:
            layers_config (list): List of layer configuration dictionaries
            
        Raises:
            ValueError: If layer configuration is invalid
            TypeError: If types are incorrect
        """
        if not isinstance(layers_config, (list, tuple)):
            raise TypeError(f"layers_config must be a list or tuple, got {type(layers_config)}")
            
        if len(layers_config) == 0:
            raise ValueError("layers_config cannot be empty. Specify at least one layer.")
        
        valid_layer_types = {'conv', 'flatten', 'dense', 'maxpool', 'avgpool'}
        
        for i, layer_config in enumerate(layers_config):
            if not isinstance(layer_config, dict):
                raise TypeError(f"Layer {i} configuration must be a dictionary, got {type(layer_config)}")
            
            if 'type' not in layer_config:
                raise ValueError(f"Layer {i} must specify a 'type' field")
            
            layer_type = layer_config['type'].lower()
            if layer_type not in valid_layer_types:
                raise ValueError(f"Unsupported layer type '{layer_type}' at layer {i}. "
                               f"Supported types: {sorted(valid_layer_types)}")
            
            # Validate layer-specific parameters
            if layer_type == 'conv':
                self._validate_conv_layer_config(layer_config, i)
            elif layer_type == 'dense':
                self._validate_dense_layer_config(layer_config, i)
    
    def _validate_conv_layer_config(self, config, layer_index):
        """Validate convolutional layer configuration."""
        required_fields = ['out_channels', 'kernel_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Conv layer {layer_index} missing required field '{field}'")
        
        # Validate out_channels
        if not isinstance(config['out_channels'], (int, np.integer)):
            raise TypeError(f"Conv layer {layer_index} 'out_channels' must be integer")
        if config['out_channels'] <= 0:
            raise ValueError(f"Conv layer {layer_index} 'out_channels' must be positive")
        
        # Validate kernel_size
        if not isinstance(config['kernel_size'], (int, np.integer)):
            raise TypeError(f"Conv layer {layer_index} 'kernel_size' must be integer")
        if config['kernel_size'] <= 0:
            raise ValueError(f"Conv layer {layer_index} 'kernel_size' must be positive")
        
        # Validate optional fields
        if 'stride' in config:
            if not isinstance(config['stride'], (int, np.integer)) or config['stride'] <= 0:
                raise ValueError(f"Conv layer {layer_index} 'stride' must be positive integer")
        
        if 'padding' in config:
            if not isinstance(config['padding'], (int, np.integer)) or config['padding'] < 0:
                raise ValueError(f"Conv layer {layer_index} 'padding' must be non-negative integer")
    
    def _validate_dense_layer_config(self, config, layer_index):
        """Validate dense layer configuration."""
        required_fields = ['neurons', 'activation']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Dense layer {layer_index} missing required field '{field}'")
        
        # Validate neurons
        if not isinstance(config['neurons'], (int, np.integer)):
            raise TypeError(f"Dense layer {layer_index} 'neurons' must be integer")
        if config['neurons'] <= 0:
            raise ValueError(f"Dense layer {layer_index} 'neurons' must be positive")
        
        # Validate activation
        if not isinstance(config['activation'], str):
            raise TypeError(f"Dense layer {layer_index} 'activation' must be string")
        
        # Use existing activation validation
        self.validate_activations([config['activation']], [config['neurons']])
    
    def validate_initial_weights(self, initial_weights):
        """
        Validate initial weights parameter.
        
        Args:
            initial_weights (str): Weight initialization strategy
            
        Raises:
            ValueError: If initial_weights is invalid
            TypeError: If type is incorrect
        """
        if not isinstance(initial_weights, str):
            raise TypeError(f"initial_weights must be a string, got {type(initial_weights)}")
            
        # Valid weight initialization strategies
        valid_weights = {'auto', 'he', 'xavier', 'glorot', 'lecun', 'random'}
        
        if initial_weights not in valid_weights:
            raise ValueError(f"Unsupported initial_weights '{initial_weights}'. "
                           f"Supported strategies: {sorted(valid_weights)}")