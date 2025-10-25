import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from pydeepflow.activations import activation, activation_derivative
from pydeepflow.losses import get_loss_function, get_loss_derivative
from pydeepflow.metrics import precision_score, recall_score, f1_score, confusion_matrix,mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from pydeepflow.device import Device
from pydeepflow.regularization import Regularization
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.cross_validator import CrossValidator
from pydeepflow.batch_normalization import BatchNormalization
from pydeepflow.weight_initialization import (
    get_weight_initializer, 
    initialize_weights, 
    initialize_biases,
    get_initializer_for_activation, 
    WeightInitializer
)

from pydeepflow.optimizers import Adam, RMSprop
from tqdm import tqdm
from pydeepflow.validation import ModelValidator
from pydeepflow.introspection import create_introspector, ModelSummaryFormatter
from pydeepflow.preprocessing import ImageDataGenerator # Added for data augmentation

# ====================================================================
# IM2COL / COL2IM helper functions (USER'S TESTED WORKING VERSIONS)
# ====================================================================

def get_im2col_indices(X_shape, filter_height, filter_width, padding=0, stride=1):
    """
    Creates indices to use with fancy indexing to get column matrix from a 4D tensor (N, H, W, C).
    """
    N, H, W, C = X_shape
    Fh, Fw = filter_height, filter_width

    # Calculate output dimensions
    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1
    
    # Check if dimensions are valid (no fractional output size)
    if not isinstance(H_out, int) or not isinstance(W_out, int):
        raise ValueError("Invalid stride or filter size for input dimensions.")

    # Indices of the output grid
    i0 = np.repeat(np.arange(Fh), Fw)
    i1 = np.tile(np.arange(Fw), Fh)
    
    # Indices in the original padded image
    i = i0.reshape(-1, 1) + np.arange(H_out) * stride
    j = i1.reshape(-1, 1) + np.arange(W_out) * stride

    # Indices for all filters, rows, columns, and channels
    k = np.repeat(np.arange(C), Fh * Fw).reshape(-1, 1)
    
    # Repeat and tile for the output feature map size
    i = np.repeat(i, W_out, axis=1)
    j = np.tile(j, H_out).reshape(i.shape)

    return (k, i, j)

def im2col_indices(X, filter_height, filter_width, padding=0, stride=1):
    """
    Transforms the input tensor X into a column matrix.
    Input X shape: (N, H, W, C) -> Output: (N * H_out * W_out, Fh * Fw * C)
    """
    N, H, W, C = X.shape
    Fh, Fw = filter_height, filter_width
    
    # Pad input
    X_padded = X
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Calculate output dimensions
    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1

    # Create patches using proper indexing
    X_col = np.zeros((N, H_out, W_out, Fh * Fw * C))
    
    col_idx = 0
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            # Extract patch for all channels at position (y, x) in the filter
            # Shape of patch: (N, H_out, W_out, C)
            patch = X_padded[:, y:y_max:stride, x:x_max:stride, :]
            # Place in correct column positions for all channels
            X_col[:, :, :, col_idx * C:(col_idx + 1) * C] = patch
            col_idx += 1
    
    # Reshape to (N * H_out * W_out, Fh * Fw * C)
    X_col = X_col.reshape(N * H_out * W_out, -1)
    return X_col


def col2im_indices(cols, X_shape, filter_height, filter_width, padding=0, stride=1):
    """
    Inverse of im2col. Converts columns back to the image shape. Returns shape (N, H, W, C).
    """
    N, H, W, C = X_shape
    Fh, Fw = filter_height, filter_width

    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1

    # Initialize gradient container
    if padding > 0:
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        X_grad = np.zeros((N, H_padded, W_padded, C))
    else:
        X_grad = np.zeros(X_shape)

    # Reshape cols back to (N, H_out, W_out, Fh * Fw * C)
    cols_reshaped = cols.reshape(N, H_out, W_out, Fh * Fw * C)

    # Scatter-add contributions back to input positions
    col_idx = 0
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            # Extract gradients for this filter position across all channels
            grad_patch = cols_reshaped[:, :, :, col_idx * C:(col_idx + 1) * C]
            # Accumulate to the correct positions in the gradient
            X_grad[:, y:y_max:stride, x:x_max:stride, :] += grad_patch
            col_idx += 1

    # Remove padding if it was added
    if padding > 0:
        return X_grad[:, padding:-padding, padding:-padding, :]
    return X_grad


# ====================================================================
# ConvLayer and Flatten (The new components)
# ====================================================================

class ConvLayer:
    """
    2D convolutional layer (channels-last) implemented via im2col.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : int
        Size of the convolutional kernel (assumes square kernel).
    stride : int, optional
        Stride for the convolution operation (default is 1).
    padding : int, optional
        Padding to add to the input (default is 0).
    device : Device, optional
        Device object for array operations (default is CPU).
    activation : str, optional
        Activation function to be used after this layer (default is 'relu').
        Used for automatic weight initialization selection.
    weight_init : str, optional
        Weight initialization strategy. Options:
        - 'auto': Automatically select based on activation function (default)
        - 'he_normal', 'he_uniform': He/Kaiming initialization
        - 'xavier_normal', 'xavier_uniform': Xavier/Glorot initialization
        - 'lecun_normal', 'lecun_uniform': LeCun initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 device=None, activation='relu', weight_init='auto'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Fh = kernel_size
        self.Fw = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device if device is not None else Device(use_gpu=False)
        self.activation = activation

        # Initialize using WeightInitializer for proper activation-aware initialization
        initializer = WeightInitializer(
            device=self.device,
            mode='auto' if weight_init == 'auto' else 'manual',
            method=weight_init if weight_init != 'auto' else None,
            bias_init='auto'
        )
        
        # Initialize weights and biases using the new system
        W_init, b_init, metadata = initializer.initialize_conv_layer(
            kernel_h=self.Fh,
            kernel_w=self.Fw,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            activation=activation
        )
        
        # Convert to device arrays and reshape bias for broadcasting
        # Bias shape needs to be (1, 1, 1, out_channels) for proper broadcasting
        W_init = self.device.array(W_init)
        b_init = self.device.array(b_init.reshape(1, 1, 1, self.out_channels))

        # Store params in a dict for clarity
        self.params = {'W': W_init, 'b': b_init}
        self.grads = {'dW': None, 'db': None}
        self.cache = None
        
        # Store initialization metadata for introspection
        self.init_metadata = metadata

    def forward(self, X):
        """ X shape: (N, H, W, C_prev) """
        N, H, W, C = X.shape
        Fh, Fw = self.Fh, self.Fw

        # Validate input channels match expected
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        H_out = (H + 2 * self.padding - Fh) // self.stride + 1
        W_out = (W + 2 * self.padding - Fw) // self.stride + 1

        # Validate output dimensions are positive
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Invalid output dimensions H_out={H_out}, W_out={W_out}. "
                           f"Check filter size, stride, and padding settings.")

        # Convert input to columns
        X_col = im2col_indices(X, Fh, Fw, self.padding, self.stride) 

        # Reshape weights to (Fh*Fw*C_prev, out_channels)
        W_col = self.params['W'].reshape(-1, self.out_channels)

        # GEMM: (N*H_out*W_out, out_channels)
        out_col = self.device.dot(X_col, W_col)

        # reshape back to image format
        out = out_col.reshape(N, H_out, W_out, self.out_channels)

        # add bias (broadcasts over N, H_out, W_out)
        out = out + self.params['b']

        # cache for backward pass
        self.cache = (X.shape, X_col, self.params['W'], self.params['b'])

        return out

    def backward(self, dOut):
        """ dOut shape: (N, H_out, W_out, out_channels) """
        X_shape, X_col, W, b = self.cache
        
        # flatten dOut to columns: (N*H_out*W_out, out_channels)
        dOut_col = dOut.reshape(-1, self.out_channels)

        # bias gradient: sum over batch and spatial dims -> shape (out_channels,)
        db = self.device.sum(dOut, axis=(0, 1, 2))

        # weight gradient: X_col.T @ dOut_col -> (Fh*Fw*C_prev, out_channels)
        dW_col = self.device.dot(X_col.T, dOut_col)
        dW = dW_col.reshape(self.Fh, self.Fw, self.in_channels, self.out_channels)

        # store grads correctly
        self.grads = {'dW': dW, 'db': db}

        # input gradient: dX_col = dOut_col @ W_col.T
        W_col = W.reshape(-1, self.out_channels)
        dX_col = self.device.dot(dOut_col, W_col.T)

        # convert columns back to image gradient
        dX = col2im_indices(dX_col, X_shape, self.Fh, self.Fw, self.padding, self.stride)

        return dX


class Flatten:
    """
    Flattens (N, H, W, C) -> (N, H*W*C)
    """
    def __init__(self):
        self.cache = None
        self.params = {}
        self.grads = {}

    def forward(self, X):
        self.cache = X.shape
        # X.shape is N (batch size), flatten the rest
        return X.reshape(X.shape[0], -1)

    def backward(self, dOut):
        original_shape = self.cache
        return dOut.reshape(original_shape)

class MaxPooling2D:
    """A Max Pooling layer for 2D inputs."""
    def __init__(self, pool_size=(2, 2), stride=2):

        if isinstance(pool_size, int):
            self.pool_height = self.pool_width = pool_size
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            self.pool_height, self.pool_width = pool_size
        else:
            raise ValueError(f"Invalid pool_size '{pool_size}'. Must be int or tuple/list of 2 ints.")

        self.stride = stride
        self.cache = {}
        self.params = {}
        self.grads = {}

    def forward(self, X):
        self.cache['X'] = X
        N, H, W, C = X.shape
        out_h = (H - self.pool_height) // self.stride + 1
        out_w = (W - self.pool_width) // self.stride + 1
        out = np.zeros((N, out_h, out_w, C))
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * self.stride, i * self.stride + self.pool_height
                w_start, w_end = j * self.stride, j * self.stride + self.pool_width
                window = X[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
        return out

    def backward(self, dOut):
        X = self.cache['X']
        N, H, W, C = X.shape
        _, out_h, out_w, _ = dOut.shape
        dX = np.zeros_like(X)
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i * self.stride, i * self.stride + self.pool_height
                        w_start, w_end = j * self.stride, j * self.stride + self.pool_width
                        window = X[n, h_start:h_end, w_start:w_end, c]
                        mask = (window == np.max(window))
                        dX[n, h_start:h_end, w_start:w_end, c] += mask * dOut[n, i, j, c]
        return dX

class AveragePooling2D:
    """An Average Pooling layer for 2D inputs."""
    def __init__(self, pool_size=(2, 2), stride=2):

        if isinstance(pool_size, int):
            self.pool_height = self.pool_width = pool_size
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            self.pool_height, self.pool_width = pool_size
        else:
            raise ValueError(f"Invalid pool_size '{pool_size}'. Must be int or tuple/list of 2 ints.")

        self.stride = stride
        self.cache = {}
        self.params = {}
        self.grads = {}

    def forward(self, X):
        self.cache['X_shape'] = X.shape
        N, H, W, C = X.shape
        out_h = (H - self.pool_height) // self.stride + 1
        out_w = (W - self.pool_width) // self.stride + 1
        out = np.zeros((N, out_h, out_w, C))
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * self.stride, i * self.stride + self.pool_height
                w_start, w_end = j * self.stride, j * self.stride + self.pool_width
                window = X[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.mean(window, axis=(1, 2))
        return out

    def backward(self, dOut):
        X_shape = self.cache['X_shape']
        _, out_h, out_w, _ = dOut.shape
        dX = np.zeros(X_shape)
        pool_area = self.pool_height * self.pool_width
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * self.stride, i * self.stride + self.pool_height
                w_start, w_end = j * self.stride, j * self.stride + self.pool_width
                grad = dOut[:, i, j, :][:, np.newaxis, np.newaxis, :]
                dX[:, h_start:h_end, w_start:w_end, :] += grad / pool_area
        return dX


# ====================================================================
# Multi_Layer_ANN (Dense-only training logic) - UNMODIFIED
# ====================================================================

class Multi_Layer_ANN:
    """
    A Multi-Layer Artificial Neural Network (ANN) for classification tasks.

    This class implements a multi-layer feedforward neural network that can be used for both
    binary and multi-class classification problems. It supports various features such as
    customizable hidden layers, activation functions, loss functions, GPU acceleration,
    L2 regularization, dropout, and batch normalization.

    Attributes:
        device (Device): An object to manage computation on either CPU or GPU.
        regularization (Regularization): An object to handle L2 and dropout regularization.
        layers (list): A list of integers defining the number of neurons in each layer,
                       including the input and output layers.
        output_activation (str): The activation function for the output layer ('sigmoid' for binary
                                 classification, 'softmax' for multi-class).
        activations (list): A list of activation functions for the hidden layers.
        weights (list): A list of weight matrices for each layer.
        biases (list): A list of bias vectors for each layer.
        loss (str): The name of the loss function.
        loss_func (function): The loss function.
        loss_derivative (function): The derivative of the loss function.
        X_train (array): The training data features.
        y_train (array): The training data labels.
        training (bool): A flag indicating whether the model is in training or inference mode.
        history (dict): A dictionary to store training and validation metrics over epochs.
        use_batch_norm (bool): A flag to enable or disable batch normalization.
        batch_norm_layers (list): A list of BatchNormalization layers.
    """
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy',
            use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, use_batch_norm=False, 
            optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32, 
            weight_init='auto', bias_init='auto'):
        """
        Initializes the Multi_Layer_ANN.

        Args:
            X_train (np.ndarray): The training input data.
            Y_train (np.ndarray): The training target data.
            hidden_layers (list): A list of integers specifying the number of neurons in each hidden layer.
            activations (list): A list of strings specifying the activation function for each hidden layer.
            loss (str, optional): The loss function to use. Defaults to 'categorical_crossentropy'.
            use_gpu (bool, optional): If True, computations will be performed on the GPU. Defaults to False.
            l2_lambda (float, optional): The L2 regularization parameter. Defaults to 0.0.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.0.
            use_batch_norm (bool, optional): If True, batch normalization will be applied. Defaults to False.
            weight_init : str or list, optional
                Weight initialization strategy. Options:
                - 'auto': Automatically select based on activation functions (default)
                - 'he_normal', 'he_uniform': He/Kaiming initialization (good for ReLU)
                - 'xavier_normal', 'xavier_uniform': Xavier/Glorot initialization (good for sigmoid/tanh)
                - 'lecun_normal', 'lecun_uniform': LeCun initialization (good for SELU)
                - 'random_normal', 'random_uniform': Simple random initialization
                - list: Layer-specific initialization methods
            bias_init : str or float, optional
                Bias initialization strategy. Options:
                - 'auto': Activation-aware initialization (default)
                - 'zeros': All zeros
                - float: Custom constant value
        

        Raises:
            ValueError: If the number of activation functions does not match the number of hidden layers.
        """
        # Validate inputs before proceeding with initialization
        validator = ModelValidator(device=None)  # Device not needed for validation
        validator.validate_training_data(X_train, "X_train", max_dimensions=2)
        validator.validate_training_data(Y_train, "Y_train", max_dimensions=2)
        validator.validate_data_compatibility(X_train, Y_train)
        validator.validate_hidden_layers(hidden_layers)
        validator.validate_activations(activations, hidden_layers)
        validator.validate_loss_function(loss)
        validator.validate_regularization_params(l2_lambda, dropout_rate)
        validator.validate_optimizer(optimizer)
        
        # Validate and adjust batch_size
        self.batch_size = validator.validate_training_hyperparameters(
            learning_rate, epochs, batch_size, X_train
        )
        
        # Initialize device FIRST (needed for weight initialization)
        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)

        # Determine the network architecture based on the classification task (binary or multi-class)
        if Y_train.ndim == 1 or Y_train.shape[1] == 1:
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else:
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'

        self.activations = activations
        
        if len(self.activations) != len(hidden_layers):
            raise ValueError("The number of activation functions must match the number of hidden layers.")

        # Setup loss function
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # Move training data to the device (GPU or CPU)
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)
        
        # Validate weight_init and bias_init parameters
        num_weight_layers = len(self.layers) - 1
        validator.validate_weight_init(weight_init, num_layers=num_weight_layers)
        validator.validate_bias_init(bias_init)
        
        # Determine initialization mode and create WeightInitializer
        if weight_init == 'auto':
            weight_initializer = WeightInitializer(
                device=self.device, 
                mode='auto',
                bias_init=bias_init
            )
        elif isinstance(weight_init, list):
            # Layer-specific initialization
            weight_initializer = WeightInitializer(
                device=self.device,
                mode='manual',
                method=weight_init,
                bias_init=bias_init
            )
        else:
            # Manual mode with single method for all layers
            weight_initializer = WeightInitializer(
                device=self.device,
                mode='manual',
                method=weight_init,
                bias_init=bias_init
            )
        
        # Initialize weights and biases using the new WeightInitializer system
        self.weights = []
        self.biases = []
        self.init_metadata = []  # Store initialization metadata for introspection
        
        for i in range(len(self.layers) - 1):
            # Get the activation function for this layer
            if i < len(self.activations):
                activation_fn = self.activations[i]
            else:
                # Output layer uses output_activation
                activation_fn = self.output_activation
            
            # Initialize weights and biases using WeightInitializer
            W, b, metadata = weight_initializer.initialize_dense_layer(
                input_dim=self.layers[i],
                output_dim=self.layers[i + 1],
                activation=activation_fn
            )
            
            # Convert to device arrays and reshape bias to (1, n) for broadcasting
            self.weights.append(self.device.array(W))
            self.biases.append(self.device.array(b.reshape(1, -1)))
            self.init_metadata.append(metadata)

        # Initialize training attribute
        self.training = False

        # Store metrics for plotting
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        # Batch Normalization setup
        self.use_batch_norm = use_batch_norm
        self.batch_norm_layers = []
        
        if self.use_batch_norm:
            for i in range(len(self.layers) - 2):  # Exclude input and output layers
                self.batch_norm_layers.append(BatchNormalization(self.layers[i+1], device=self.device))

        # Optimizer setup
        if optimizer == 'adam':
            self.optimizer = Adam()
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop()
        else:
            self.optimizer = None  # Default to SGD

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            tuple: A tuple containing:
                - activations (list): A list of activation values for each layer.
                - Z_values (list): A list of the pre-activation (linear) values for each layer.
        """
        activations = [X]
        Z_values = []

        # Forward pass through hidden layers with dropout and batch normalization
        for i in range(len(self.weights) - 1):
            Z = self.device.dot(activations[-1], self.weights[i]) + self.biases[i]
            if self.use_batch_norm:
                Z = self.batch_norm_layers[i].normalize(Z, training=self.training)
            Z_values.append(Z)
            A = activation(Z, self.activations[i], self.device)
            A = self.regularization.apply_dropout(A, training=self.training)
            activations.append(A)

        # Forward pass through the output layer
        Z_output = self.device.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, self.output_activation, self.device)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate, clip_value=None):
        """
        Performs backpropagation to compute gradients and update model weights and biases.

        Args:
            X (np.ndarray): The input data for the current batch.
            y (np.ndarray): The true labels for the current batch.
            activations (list): The list of activations from the forward pass.
            Z_values (list): The list of pre-activation values from the forward pass.
            learning_rate (float): The learning rate for weight updates.
            clip_value (float, optional): The value to clip gradients to, preventing exploding gradients.
                                          Defaults to None.
        """
        # Calculate the error in the output layer
        output_error = activations[-1] - y
        d_output = output_error * activation_derivative(activations[-1], self.output_activation, self.device)

        # Backpropagate through the network
        deltas = [d_output]
        for i in reversed(range(len(self.weights) - 1)):
            error = self.device.dot(deltas[-1], self.weights[i + 1].T)
            if self.use_batch_norm:
                error = self.batch_norm_layers[i].backprop(Z_values[i], error, learning_rate)
            delta = error * activation_derivative(activations[i + 1], self.activations[i], self.device)
            deltas.append(delta)

        deltas.reverse()

        # Update weights and biases with L2 regularization
        gradient = {'weights': [], 'biases': []}
        for i in range(len(self.weights)):
            grad_weights = self.device.dot(activations[i].T, deltas[i])
            grad_biases = self.device.sum(deltas[i], axis=0, keepdims=True)

            # Clip gradients if clip_value is specified
            if clip_value is not None:
                # Clip weights gradients
                grad_weights_norm = self.device.norm(grad_weights)
                if grad_weights_norm > clip_value:
                    grad_weights = grad_weights * (clip_value / grad_weights_norm)

                # Clip bias gradients
                grad_biases_norm = self.device.norm(grad_biases)
                if grad_biases_norm > clip_value:
                    grad_biases = grad_biases * (clip_value / grad_biases_norm)

            gradient['weights'].append(grad_weights)
            gradient['biases'].append(grad_biases)

        if self.optimizer:
            params = self.weights + self.biases
            grads = gradient['weights'] + gradient['biases']
            self.optimizer.update(params, grads)
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= gradient['weights'][i] * learning_rate
                self.biases[i] -= gradient['biases'][i] * learning_rate

        # Apply L2 regularization to the weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.regularization.apply_l2_regularization(self.weights[i], learning_rate, X.shape)

    def fit(self, epochs=100, learning_rate=0.01, generator=None, steps_per_epoch=None, lr_scheduler=None,
            early_stop=None, X_val=None, y_val=None, checkpoint=None, verbose=False, clipping_threshold=None):
        """
        Trains the neural network model using either a static dataset or a data generator.

        Args:
            epochs (int): The number of epochs to train the model. Defaults to 100.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            generator (ImageDataGenerator, optional): A data generator. If provided, training uses batches from this generator. Defaults to None.
            steps_per_epoch (int, optional): Batches per epoch when using a generator. Required if generator is provided. Defaults to None.
            lr_scheduler (object, optional): A learning rate scheduler. Defaults to None.
            early_stop (object, optional): An early stopping callback. Defaults to None.
            X_val (np.ndarray, optional): Validation features. Defaults to None.
            y_val (np.ndarray, optional): Validation labels. Defaults to None.
            checkpoint (object, optional): A model checkpointing callback. Defaults to None.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
            clipping_threshold (float, optional): The value for gradient clipping. Defaults to None.
        """
        if early_stop and (X_val is None or y_val is None):
            raise ValueError("Validation set (X_val, y_val) is required for early stopping.")

        if generator and not isinstance(generator, ImageDataGenerator):
            raise TypeError("`generator` must be an instance of ImageDataGenerator or None.")

        if generator and steps_per_epoch is None:
            raise ValueError("`steps_per_epoch` must be specified when using a generator.")

        # Determine number of epochs based on input
        num_epochs = epochs if epochs is not None else 100  # Default epochs if none provided

        for epoch in tqdm(range(num_epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green',
                          disable=not verbose):
            start_time = time.time()
            current_lr = lr_scheduler.get_lr(epoch) if lr_scheduler else learning_rate

            epoch_train_loss = 0.0
            epoch_train_accuracy = 0.0

            # --- Training Loop ---
            if generator:
                # Training with Data Generator
                for step in range(steps_per_epoch):
                    X_batch, y_batch = next(generator)
                    X_batch_device = self.device.array(X_batch)
                    y_batch_device = self.device.array(y_batch)

                    self.training = True
                    activations, Z_values = self.forward_propagation(X_batch_device)
                    self.backpropagation(X_batch_device, y_batch_device, activations, Z_values, current_lr,
                                         clip_value=clipping_threshold)
                    self.training = False

                    # Calculate batch metrics
                    loss = self.loss_func(y_batch_device, activations[-1], self.device)
                    # Convert to numpy for accuracy calculation if needed
                    preds_np = self.device.asnumpy(activations[-1])
                    y_batch_np = self.device.asnumpy(y_batch_device)
                    accuracy = np.mean(np.argmax(preds_np, axis=1) == np.argmax(y_batch_np,
                                                                                axis=1)) if self.output_activation != 'sigmoid' else np.mean(
                        (preds_np >= 0.5).astype(int) == y_batch_np)

                    epoch_train_loss += loss
                    epoch_train_accuracy += accuracy

                # Average metrics over steps
                train_loss = epoch_train_loss / steps_per_epoch
                train_accuracy = epoch_train_accuracy / steps_per_epoch

            else:
                # Standard Training (without generator)
                self.training = True
                activations, Z_values = self.forward_propagation(self.X_train)
                self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr,
                                     clip_value=clipping_threshold)
                self.training = False

                train_loss = self.loss_func(self.y_train, activations[-1], self.device)
                preds_np = self.device.asnumpy(activations[-1])
                y_train_np = self.device.asnumpy(self.y_train)
                train_accuracy = np.mean(np.argmax(preds_np, axis=1) == np.argmax(y_train_np,
                                                                                  axis=1)) if self.output_activation != 'sigmoid' else np.mean(
                    (preds_np >= 0.5).astype(int) == y_train_np)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)

            # --- Validation Step ---
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                X_val_device = self.device.array(X_val)
                y_val_device = self.device.array(y_val)
                val_activations, _ = self.forward_propagation(X_val_device)
                val_loss = self.loss_func(y_val_device, val_activations[-1], self.device)

                val_preds_np = self.device.asnumpy(val_activations[-1])
                # Ensure y_val is numpy for comparison
                y_val_np = y_val if isinstance(y_val, np.ndarray) else self.device.asnumpy(y_val_device)

                val_accuracy = np.mean(np.argmax(val_preds_np, axis=1) == np.argmax(y_val_np,
                                                                                    axis=1)) if self.output_activation != 'sigmoid' else np.mean(
                    (val_preds_np >= 0.5).astype(int) == y_val_np)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

                # Checkpoint saving
                if checkpoint is not None:
                    if checkpoint.should_save(epoch, val_loss):
                        checkpoint.save_weights(epoch, self.weights, self.biases, val_loss)

                # Early stopping check
                if early_stop:
                    early_stop(val_loss)
                    if early_stop.early_stop:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break  # Exit epoch loop

            # --- Verbose Output ---
            if verbose:  # Print at the end of each epoch
                log_msg = f"\rEpoch {epoch + 1}/{num_epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}"
                if val_loss is not None:
                    log_msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                log_msg += f" - lr: {current_lr:.6f} "
                # Use print instead of sys.stdout.write for tqdm compatibility
                tqdm.write(log_msg)

        print("\nTraining Completed!")

    def predict(self, X):
        """
        Makes predictions on new data.

        Args:
            X (np.ndarray): The input data for which to make predictions.

        Returns:
            np.ndarray: The predicted probabilities or class labels.
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def evaluate(self, X, y, metrics=['loss', 'accuracy']):
        """
        Evaluates the model's performance on a given dataset.

        This method computes various performance metrics to assess the model's accuracy and effectiveness.
        It can calculate loss, accuracy, precision, recall, F1-score, and a confusion matrix.

        Args:
            X (np.ndarray): The input features for evaluation.
            y (np.ndarray): The true labels for evaluation.
            metrics (list, optional): A list of metrics to calculate.
                                      Defaults to ['loss', 'accuracy'].
                                      Available metrics: 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'root_mean_squared_error'.

        Returns:
            dict: A dictionary where keys are the metric names and values are the computed scores.
        """
        predictions = self.predict(X)
        results = {}

        if 'loss' in metrics:
            results['loss'] = self.loss_func(y, predictions, self.device)

        y_pred_classes = (predictions >= 0.5).astype(int) if self.output_activation == 'sigmoid' else np.argmax(predictions, axis=1)
        y_true_classes = y if self.output_activation == 'sigmoid' else np.argmax(y, axis=1)

        if 'accuracy' in metrics:
            results['accuracy'] = np.mean(y_pred_classes == y_true_classes)
        
        if 'precision' in metrics:
            results['precision'] = precision_score(y_true_classes, y_pred_classes)

        if 'recall' in metrics:
            results['recall'] = recall_score(y_true_classes, y_pred_classes)

        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_true_classes, y_pred_classes)
        
        if 'confusion_matrix' in metrics:
            num_classes = self.layers[-1]
            results['confusion_matrix'] = confusion_matrix(y_true_classes, y_pred_classes, num_classes)
             
        if 'mean_absolute_error' in metrics:
          results['mean_absolute_error'] = mean_absolute_error(y, predictions)

        if 'mean_squared_error' in metrics:
          results['mean_squared_error'] = mean_squared_error(y, predictions)

        if 'r2_score' in metrics:
         results['r2_score'] = r2_score(y, predictions)

        if 'root_mean_squared_error' in metrics:
            results['root_mean_squared_error'] = root_mean_squared_error(y, predictions)



        return results
       
    def load_checkpoint(self, checkpoint_path):
        """
        Loads model weights and biases from a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
        """
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = ModelCheckpoint(checkpoint_path)
        self.weights, self.biases = checkpoint.load_weights()

    def save_model(self, file_path):
        """
        Saves the entire model to a file.

        This includes weights, biases, layer architecture, and activation functions.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        model_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'layers': self.layers,
            'activations': self.activations,
            'output_activation': self.output_activation
        }
        np.save(file_path, model_data)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Loads a model from a file.

        This restores the weights, biases, layer architecture, and activation functions.

        Args:
            file_path (str): The path to the file from which to load the model.
        """
        model_data = np.load(file_path, allow_pickle=True).item()
        self.weights = [self.device.array(w) for w in model_data['weights']]
        self.biases = [self.device.array(b) for b in model_data['biases']]
        self.layers = model_data['layers']
        self.activations = model_data['activations']
        self.output_activation = model_data['output_activation']
        print(f"Model loaded from {file_path}")

    def summary(self):
        """
        Displays a summary of the model architecture using the introspection module.
        
        This method provides a comprehensive overview of the neural network structure,
        similar to Keras model.summary(), showing:
        - Layer-by-layer breakdown with input/output shapes
        - Parameter count for each layer
        - Total trainable parameters
        - Estimated memory usage
        - Model configuration details
        """
        introspector = create_introspector(self)
        layer_info = introspector.get_layer_info()
        param_counts = introspector.calculate_parameters()
        memory_usage = introspector.estimate_memory_usage()
        configuration = introspector.get_model_configuration()
        
        summary_text = ModelSummaryFormatter.format_summary(
            layer_info, param_counts, memory_usage, configuration, "Multi_Layer_ANN"
        )
        print(summary_text)

    def get_model_info(self):
        """
        Returns a dictionary containing detailed model information using the introspection module.
        
        This method provides programmatic access to model architecture details,
        useful for automated analysis or integration with other tools.
        
        Returns:
            dict: A dictionary containing:
                - layer_info: List of dictionaries with layer details
                - total_params: Total number of parameters
                - memory_usage: Estimated memory usage in MB
                - configuration: Model configuration details
        """
        introspector = create_introspector(self)
        layer_info = introspector.get_layer_info()
        param_counts = introspector.calculate_parameters()
        memory_usage = introspector.estimate_memory_usage()
        configuration = introspector.get_model_configuration()
        
        return ModelSummaryFormatter.format_model_info(
            layer_info, param_counts, memory_usage, configuration
        )
        return introspector.get_model_info()

    def get_initialization_info(self):
        """
        Retrieve initialization metadata for all layers.
        
        Returns detailed information about how each layer's weights and biases
        were initialized, including the method used, activation function, shape,
        and scaling factors.
        
        Returns
        -------
        list of InitializationMetadata
            List of metadata objects, one for each layer in the network.
            Each metadata object contains:
            - layer_index: Index of the layer (0-based)
            - layer_type: Type of layer ('dense' or 'conv')
            - method: Initialization method used (e.g., 'he_normal', 'xavier_uniform')
            - activation: Activation function for the layer
            - shape: Shape of the weight matrix/tensor
            - bias_value: Value used for bias initialization
            - fan_in: Number of input connections
            - fan_out: Number of output connections
            - scale: Scaling factor used in initialization
        
        Examples
        --------
        >>> model = Multi_Layer_ANN(X_train, y_train, [128, 64], ['relu', 'relu'])
        >>> init_info = model.get_initialization_info()
        >>> for metadata in init_info:
        ...     print(metadata)
        Layer 0 (dense): he_normal for relu, shape=(784, 128), scale=0.0535
        Layer 1 (dense): he_normal for relu, shape=(128, 64), scale=0.1768
        Layer 2 (dense): he_normal for softmax, shape=(64, 10), scale=0.2500
        
        Notes
        -----
        This method is useful for:
        - Debugging initialization issues
        - Understanding model configuration
        - Verifying that appropriate initialization methods were selected
        - Documenting model architecture for reproducibility
        """
        return self.init_metadata

    def _validate_inputs(self, X_train, Y_train, hidden_layers, activations, loss, 
                        l2_lambda, dropout_rate, optimizer, learning_rate, epochs, batch_size, initial_weights):
        """
        Comprehensive validation of all input parameters for model initialization.
        
        Args:
        X_train (np.ndarray): Training input data, shape (n_samples, n_features)
        Y_train (np.ndarray): Training target data, shape (n_samples, n_outputs)
        hidden_layers (list[int]): List of hidden layer sizes, all positive integers
        activations (list[str]): List of activation function names (e.g., 'relu', 'sigmoid')
        loss (str): Loss function name (e.g., 'mse', 'cross_entropy')
        l2_lambda (float): L2 regularization parameter, non-negative
        dropout_rate (float): Dropout rate, value between 0.0 and 1.0
        optimizer (str): Optimizer name (e.g., 'adam', 'sgd')
        learning_rate (float): Learning rate for optimization, positive value
        epochs (int): Number of training epochs, positive integer
        batch_size (int): Batch size for training, positive integer
        initial_weights (str): Weight initialization strategy (e.g., 'auto', 'he', 'xavier')

            
        Raises:
            ValueError: If any input parameter is invalid
            TypeError: If input types are incorrect
        """
        # Validate X_train
        self._validate_training_data(X_train, "X_train")
        
        # Validate Y_train  
        self._validate_training_data(Y_train, "Y_train")
        
        # Validate data compatibility
        self._validate_data_compatibility(X_train, Y_train)
        
        # Validate hidden layers
        self._validate_hidden_layers(hidden_layers)
        
        # Validate activations
        self._validate_activations(activations, hidden_layers)
        
        # Validate loss function
        self._validate_loss_function(loss)
        
        # Validate regularization parameters
        self._validate_regularization_params(l2_lambda, dropout_rate)
        
        # Validate optimizer
        self._validate_optimizer(optimizer)

        # Validate the training parameters
        self._validate_training_hyperparameters(learning_rate, epochs, batch_size, X_train)
        
        # Validate initial weights parameter
        self._validate_initial_weights(initial_weights)

    def _validate_training_data(self, data, data_name):
        """Validate training data (X_train or Y_train)."""
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
            
        # Check for valid dimensions (1D or 2D)
        if data_array.ndim == 0:
            raise ValueError(f"{data_name} must be at least 1_dimensional")
        elif data_array.ndim > 2:
            raise ValueError(f"{data_name} must be 1D or 2D array, got {data_array.ndim}D")
            
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

    def _validate_data_compatibility(self, X_train, Y_train):
        """Validate that X_train and Y_train are compatible."""
        X_array = np.asarray(X_train)
        Y_array = np.asarray(Y_train)
        
        # Check sample count compatibility
        if X_array.shape[0] != Y_array.shape[0]:
            raise ValueError(f"X_train and Y_train must have the same number of samples. "
                           f"Got X_train: {X_array.shape[0]}, Y_train: {Y_array.shape[0]}")
                           
        # Check minimum sample count
        if X_array.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples for training, got {X_array.shape[0]}")
            
        # Validate feature count
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

    def _validate_hidden_layers(self, hidden_layers):
        """Validate hidden layer configuration."""
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

    def _validate_activations(self, activations, hidden_layers):
        """Validate activation functions."""
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

    def _validate_loss_function(self, loss):
        """Validate loss function."""
        if not isinstance(loss, str):
            raise TypeError(f"Loss function must be a string, got {type(loss)}")
            
        # Valid loss functions (from losses.py)
        valid_losses = {'binary_crossentropy', 'mse', 'categorical_crossentropy', 'hinge', 'huber'}
        
        if loss not in valid_losses:
            raise ValueError(f"Unsupported loss function '{loss}'. "
                           f"Supported losses: {sorted(valid_losses)}")

    def _validate_regularization_params(self, l2_lambda, dropout_rate):
        """Validate regularization parameters."""
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

    def _validate_optimizer(self, optimizer):
        """Validate optimizer."""
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
                # Note: 'sgd' is handled as string since there's no SGD class

    def _validate_training_hyperparameters(self, learning_rate, epochs, batch_size, X_train):
        """Validate training hyperparameters (learning rate, epochs, batch size)."""
        
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
        if batch_size > n_samples:
            import warnings
            warnings.warn(
                f"batch_size ({batch_size}) is larger than the number of training samples ({n_samples}). "
                f"Automatically adjusting batch_size to {n_samples}.",
                UserWarning
            )
            # Store adjusted batch size
            self.batch_size = n_samples
        else:
            self.batch_size = batch_size
            
        if batch_size > 1024:
            raise ValueError(f"batch_size seems too large: {batch_size}. "
                        f"Typical values are 16, 32, 64, 128, 256, or 512. "
                        f"Large batch sizes may cause memory issues and poor generalization")
            
        # Warning for very small batch sizes (not an error, but good to know)
        if batch_size == 1 and n_samples >=100:
            import warnings
            warnings.warn(
                f"batch_size of 1 (online learning) with {n_samples} samples "
                f"may result in very slow and unstable training. "
                f"Consider using a larger batch size like 16, 32, or 64.",
                UserWarning,
                stacklevel=3  # Adjust stacklevel to point to the right caller
            )

    def _validate_initial_weights(self, initial_weights):
        """Validate initial weights parameter."""
        if not isinstance(initial_weights, str):
            raise TypeError(f"initial_weights must be a string, got {type(initial_weights)}")
            
        valid_initial_weights = {'auto', 'he', 'xavier', 'glorot', 'lecun', 'random'}
        
        if initial_weights not in valid_initial_weights:
            raise ValueError(f"Unsupported initial_weights '{initial_weights}'. "
                           f"Supported values: {sorted(valid_initial_weights)}")

# ====================================================================
# Plotting utilities (UNMODIFIED)
# ====================================================================

class Plotting_Utils:
    def plot_training_history(self, history, metrics=('loss', 'accuracy'), figure='history.png'):
        """
        Plot training history using a non-interactive backend so this works in
        headless environments (CI, containers, servers) where GUI backends like
        GTK are unavailable.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        num_metrics = len(metrics)
        fig = Figure(figsize=(6 * num_metrics, 5))
        canvas = FigureCanvas(fig)

        if num_metrics == 1:
            ax = [fig.add_subplot(1, 1, 1)]
        else:
            ax = [fig.add_subplot(1, num_metrics, i + 1) for i in range(num_metrics)]

        for i, metric in enumerate(metrics):
            ax[i].plot(history.get(f'train_{metric}', []), label=f'Train {metric.capitalize()}')
            if f'val_{metric}' in history:
                ax[i].plot(history.get(f'val_{metric}', []), label=f'Validation {metric.capitalize()}')
            ax[i].set_title(f"{metric.capitalize()} over Epochs")
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(metric.capitalize())
            ax[i].legend()

        fig.tight_layout()
        fig.savefig(figure)
        # Do not call plt.show() — this avoids GUI backend imports in headless envs

    def plot_learning_curve(self, train_sizes, train_scores, val_scores, figure='learning_curve.png'):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title("Learning Curve")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                             val_scores_mean + val_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        ax.plot(train_sizes, val_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure)
        # Do not call plt.show() to avoid GUI backend usage


# ====================================================================
# NEW CLASS: Multi_Layer_CNN (The Structural Fix)
# ====================================================================

class Multi_Layer_CNN:
    """
    A Sequential Model wrapper that chains Convolutional, Flatten, and Dense layers.
    It implements end-to-end forward propagation and backpropagation for CNN training.
    """

    def __init__(self, layers_list, X_train, Y_train, loss='categorical_crossentropy',
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, optimizer='sgd'):

        validator = ModelValidator()
        validator.validate_training_data(X_train, "X_train", max_dimensions=4)
        if np.asarray(X_train).ndim != 4:
            raise ValueError("X_train must be a 4D array (N, H, W, C) for CNN models.")
        validator.validate_training_data(Y_train, "Y_train", max_dimensions=2)
        validator.validate_data_compatibility(X_train, Y_train)
        validator.validate_cnn_layers(layers_list)
        validator.validate_loss_function(loss)
        validator.validate_regularization_params(l2_lambda, dropout_rate)
        validator.validate_optimizer(optimizer)

        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)

        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)
        self.training = False
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        self.layers_list = []
        self.trainable_params = []

        current_input_shape = X_train.shape[1:]  # (H, W, C)

        for layer_config in layers_list:
            layer_type = layer_config['type'].lower()

            if layer_type == 'conv':
                in_c = current_input_shape[-1]
                conv_layer = ConvLayer(in_channels=in_c, out_channels=layer_config['out_channels'], kernel_size=layer_config['kernel_size'], stride=layer_config.get('stride', 1), padding=layer_config.get('padding', 0), device=self.device)
                self.layers_list.append(conv_layer)
                H, W, _ = current_input_shape
                k, s, p = conv_layer.Fh, conv_layer.stride, conv_layer.padding
                out_h, out_w = (H + 2*p - k)//s + 1, (W + 2*p - k)//s + 1
                current_input_shape = (out_h, out_w, conv_layer.out_channels)
                self.trainable_params.extend([conv_layer.params['W'], conv_layer.params['b']])

            elif layer_type == 'maxpool':
                # Get pool_size and stride from config, providing defaults
                pool_size_config = layer_config.get('pool_size', (2, 2))
                stride = layer_config.get('stride', 2)

                # Add the layer instance
                self.layers_list.append(MaxPooling2D(pool_size=pool_size_config, stride=stride))

                # Calculate output shape
                H, W, C = current_input_shape

                # *** FIX: Handle int or tuple for pool_size BEFORE calculating shape ***
                if isinstance(pool_size_config, int):
                    pool_h = pool_w = pool_size_config
                elif isinstance(pool_size_config, (tuple, list)) and len(pool_size_config) == 2:
                    pool_h, pool_w = pool_size_config
                else:
                    raise ValueError(
                        f"Invalid pool_size '{pool_size_config}' for MaxPooling2D. Must be int or tuple/list of 2 ints.")

                # Calculate output height and width
                out_h = (H - pool_h) // stride + 1
                out_w = (W - pool_w) // stride + 1

                # Update current shape for the next layer
                current_input_shape = (out_h, out_w, C)

            elif layer_type == 'avgpool':
                # Get pool_size and stride from config, providing defaults
                pool_size_config = layer_config.get('pool_size', (2, 2))
                stride = layer_config.get('stride', 2)

                # Add the layer instance
                self.layers_list.append(AveragePooling2D(pool_size=pool_size_config, stride=stride))

                # Calculate output shape
                H, W, C = current_input_shape

                # *** FIX: Handle int or tuple for pool_size BEFORE calculating shape ***
                if isinstance(pool_size_config, int):
                    pool_h = pool_w = pool_size_config
                elif isinstance(pool_size_config, (tuple, list)) and len(pool_size_config) == 2:
                    pool_h, pool_w = pool_size_config
                else:
                    raise ValueError(
                        f"Invalid pool_size '{pool_size_config}' for AveragePooling2D. Must be int or tuple/list of 2 ints.")

                # Calculate output height and width
                out_h = (H - pool_h) // stride + 1
                out_w = (W - pool_w) // stride + 1

                # Update current shape for the next layer
                current_input_shape = (out_h, out_w, C)

            elif layer_type == 'flatten':
                self.layers_list.append(Flatten())
                current_input_shape = (np.prod(current_input_shape),)

            elif layer_type == 'dense':
                input_dim, output_dim, activation_name = current_input_shape[0], layer_config['neurons'], layer_config['activation']
                initializer = WeightInitializer(device=self.device, mode='auto', bias_init='auto')
                w, b, _ = initializer.initialize_dense_layer(input_dim, output_dim, activation_name)
                dense_layer = {'W': self.device.array(w), 'b': self.device.array(b.reshape(1, -1)), 'activation': activation_name}
                self.layers_list.append(dense_layer)
                current_input_shape = (output_dim,)
                self.trainable_params.extend([dense_layer['W'], dense_layer['b']])

        self.output_dim = Y_train.shape[1] if Y_train.ndim > 1 else 1
        self.output_activation = 'softmax' if self.output_dim > 1 else 'sigmoid'

        if optimizer == 'adam': self.optimizer = Adam()
        elif optimizer == 'rmsprop': self.optimizer = RMSprop()
        else: self.optimizer = None

    def forward_propagation(self, X):
        """
        Performs forward propagation by chaining through all layers (Conv -> Flatten -> Dense).
        Caches inputs and Z-values for backpropagation.
        Handles ConvLayer, Flatten, and Dense layers in correct order.
        """
        current_activation = X
        Z_values = []
        A_values = [X]  # Stores all activation outputs (A0 = Input, A1, A2...)

        for layer_idx, layer in enumerate(self.layers_list):
            if isinstance(layer, (ConvLayer, Flatten, MaxPooling2D, AveragePooling2D)):
                # --- ConvLayer forward ---
                current_activation = layer.forward(current_activation)
                A_values.append(current_activation)
            elif isinstance(layer, Flatten):
                # --- Flatten forward ---
                current_activation = layer.forward(current_activation)
                A_values.append(current_activation)
            elif isinstance(layer, dict) and 'W' in layer:
                # --- Dense forward ---
                W, b = layer['W'], layer['b']
                # Linear Transformation
                Z = self.device.dot(current_activation, W) + b
                Z_values.append(Z)
                # Activation
                if layer_idx == len(self.layers_list) - 1:
                    # Final output layer: use output_activation
                    A = activation(Z, self.output_activation, self.device)
                else:
                    A = activation(Z, layer['activation'], self.device)
                    A = self.regularization.apply_dropout(A, training=self.training)
                current_activation = A
                A_values.append(A)
        return A_values, Z_values

    def backpropagation(self, X, y, A_values, Z_values, learning_rate, clip_value=None):
        """
        Propagates gradients backward through Dense, Flatten, and Conv layers, updating all parameters.
        Handles ConvLayer and Flatten integration for CNNs.
        """
        N = X.shape[0]

        # 1. Output Layer Error (dOut = dLoss/dA * dA/dZ)
        output_error = A_values[-1] - y
        dOut = output_error * activation_derivative(A_values[-1], self.output_activation, self.device)

        # Gradients are accumulated in this list (must match self.trainable_params order)
        grads_to_update = []

        # Pointers for traversing Dense layers backwards
        Z_index = len(Z_values) - 1
        A_index = len(A_values) - 2  # Previous activation (current is A_values[-1])

        # --- Backward pass through all layers (Dense, Flatten, Conv) ---
        for i in reversed(range(len(self.layers_list))):
            layer = self.layers_list[i]

            if isinstance(layer, ConvLayer):
                # --- ConvLayer backward: computes dW, db, and dIn for previous layer ---
                dIn = layer.backward(dOut)  # Populates layer.grads
                # Store Conv gradients (W then b) - insert at beginning to maintain order
                grads_to_update.insert(0, layer.grads['db'])  # Insert bias first
                grads_to_update.insert(0, layer.grads['dW'])  # Insert weights second
                dOut = dIn  # Gradient for the previous layer (Flatten or another Conv)

            elif isinstance(layer, (Flatten, MaxPooling2D, AveragePooling2D)):
                dIn = layer.backward(dOut)
                dOut = dIn

            elif isinstance(layer, dict) and 'W' in layer:
                # --- Dense backward: computes dW, db, and dIn for previous layer ---
                W_param = layer['W']
                A_prev = A_values[A_index]

                # 1. Calculate delta (dZ)
                if i == len(self.layers_list) - 1:  # Output layer
                    delta = dOut
                else:
                    # Hidden Dense layer: Error from next layer already passed in dOut
                    Z_curr = Z_values[Z_index]
                    delta = dOut * activation_derivative(Z_curr, layer['activation'], self.device)

                # 2. Compute gradients dW and db
                dW = self.device.dot(A_prev.T, delta)
                db = self.device.sum(delta, axis=0, keepdims=True)

                # Store Dense gradients (W then b) - insert at beginning to maintain order
                grads_to_update.insert(0, db)
                grads_to_update.insert(0, dW)

                # 3. Calculate dOut for the previous layer (dIn)
                dOut = self.device.dot(delta, W_param.T)

                Z_index -= 1
                A_index -= 1

        # 4. Parameter Update (Optimization)
        # Apply L2 regularization to weights (W) only
        final_grads = []
        for i in range(0, len(self.trainable_params), 2):
            W_param = self.trainable_params[i]
            b_param = self.trainable_params[i + 1]
            grad_W = grads_to_update[i]
            grad_b = grads_to_update[i + 1]

            # L2 for W
            l2_grad_W = (self.regularization.l2_lambda * W_param) / N

            # Combine gradients
            final_grads.append(grad_W + l2_grad_W)
            final_grads.append(grad_b)  # Bias has no L2

            # Apply gradient clipping
            if clip_value is not None:
                final_grads[i] = self.regularization.clip_gradient(final_grads[i], clip_value)
                final_grads[i + 1] = self.regularization.clip_gradient(final_grads[i + 1], clip_value)

        if self.optimizer:
            # Optimizer updates ALL parameters in self.trainable_params list in place
            self.optimizer.update(self.trainable_params, final_grads)
        else:
            # Standard SGD update
            for i in range(len(self.trainable_params)):
                self.trainable_params[i] -= final_grads[i] * learning_rate


    # --- Methods copied from Multi_Layer_ANN and adapted for CNN structure ---

    def fit(self, epochs=50, learning_rate=0.01, generator=None, steps_per_epoch=None, lr_scheduler=None, early_stop=None, X_val=None, y_val=None,
            checkpoint=None, verbose=False, clipping_threshold=None):
        # Validate training hyperparameters
        validator = ModelValidator(device=None)
        batch_size = validator.validate_training_hyperparameters(
            learning_rate, epochs, 32, self.X_train  # Default batch_size=32 for CNN
        )

        if early_stop:
            assert X_val is not None and y_val is not None, "Validation set required for early stopping"

        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green',
                          disable=not verbose):
            start_time = time.time()

            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate

            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr,
                                 clip_value=clipping_threshold)
            self.training = False

            # metrics
            train_loss = self.loss_func(self.y_train, activations[-1], self.device)
            train_accuracy = np.mean((activations[-1] >= 0.5).astype(
                int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(
                np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            if train_loss is None or train_accuracy is None:
                print("Warning: train_loss or train_accuracy is None!")
                continue

            val_loss = val_accuracy = None
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(self.device.array(X_val))
                val_loss = self.loss_func(self.device.array(y_val), val_activations[-1], self.device)
                val_accuracy = np.mean((val_activations[-1] >= 0.5).astype(
                    int) == y_val) if self.output_activation == 'sigmoid' else np.mean(
                    np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            # NOTE: Checkpoint logic must be adapted if checkpoints save/load structure relies on self.weights/biases
            # For this MVP fix, we rely on external file saving.

            if verbose and (epoch % 10 == 0):
                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Accuracy: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.4f} | "
                    f"LR: {current_lr:.6f}  "
                )
                sys.stdout.flush()

            if early_stop:
                early_stop(val_loss)
                if early_stop.early_stop:
                    print('\n', "#" * 80)
                    print(
                        f"Early stop at epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
                    print('#' * 80)
                    break

        print("Training Completed!")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def evaluate(self, X, y, metrics=['loss', 'accuracy']):
        predictions = self.predict(X)
        results = {}

        if 'loss' in metrics:
            results['loss'] = self.loss_func(y, predictions, self.device)

        y_pred_classes = (predictions >= 0.5).astype(int) if self.output_activation == 'sigmoid' else np.argmax(predictions, axis=1)
        y_true_classes = y if self.output_activation == 'sigmoid' else np.argmax(y, axis=1)
        
        # NOTE: Metrics need full prediction chain, but we'll use simplified Dense metric call
        # Since Multi_Layer_CNN doesn't store self.layers like ANN, metrics must be calculated manually

        if 'accuracy' in metrics:
            results['accuracy'] = np.mean(y_pred_classes == y_true_classes)

        if 'precision' in metrics:
            results['precision'] = precision_score(y_true_classes, y_pred_classes)

        if 'recall' in metrics:
            results['recall'] = recall_score(y_true_classes, y_pred_classes)

        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_true_classes, y_pred_classes)

        if 'mean_absolute_error' in metrics:
            results['mean_absolute_error'] = mean_absolute_error(y, predictions)

        if 'mean_squared_error' in metrics:
            results['mean_squared_error'] = mean_squared_error(y, predictions)

        if 'r2_score' in metrics:
            results['r2_score'] = r2_score(y, predictions)    
        
        if 'root_mean_squared_error' in metrics:
            results['root_mean_squared_error'] = root_mean_squared_error(y, predictions)

        return results # Removed confusion_matrix for simplification/dependency reasons

    def save_model(self, file_path):
        # NOTE: Saving a complex CNN requires serializing the layers_list and trainable_params
        model_data = {
            'trainable_params': [p.tolist() for p in self.trainable_params],
            'layers_list_config': [layer.get_config() if hasattr(layer, 'get_config') else layer for layer in self.layers_list],
            'output_dim': self.output_dim,
            'output_activation': self.output_activation,
            'loss': self.loss,
            'l2_lambda': self.regularization.l2_lambda,
            'dropout_rate': self.regularization.dropout_rate
        }
        np.save(file_path, model_data)
        print(f"CNN Model saved to {file_path}")

    def load_model(self, file_path):
        # NOTE: Loading requires re-initializing the class and setting parameters
        # This is typically complex, but for Hacktoberfest MVP, we provide a placeholder:
        print(f"Placeholder: CNN Model loading is complex and requires explicit layer reconstruction.")
        print(f"Please re-initialize the model and use saved weights.")
        
        # Actual implementation requires re-creating layers and re-injecting params
        # This is outside the scope of the immediate fix.

    def summary(self):
        """
        Displays a summary of the CNN model architecture using the introspection module.
        
        This method provides a comprehensive overview of the CNN structure,
        showing convolutional layers, flatten operations, and dense layers with
        their respective parameters and output shapes.
        """
        introspector = create_introspector(self)
        layer_info = introspector.get_layer_info()
        param_counts = introspector.calculate_parameters()
        memory_usage = introspector.estimate_memory_usage()
        configuration = introspector.get_model_configuration()
        
        summary_text = ModelSummaryFormatter.format_summary(
            layer_info, param_counts, memory_usage, configuration, "Multi_Layer_CNN"
        )
        print(summary_text)

    def get_model_info(self):
        """
        Returns a dictionary containing detailed CNN model information using the introspection module.
        
        Returns:
            dict: A dictionary containing:
                - layer_info: List of dictionaries with layer details
                - total_params: Total number of parameters
                - memory_usage: Estimated memory usage in MB
                - configuration: Model configuration details
        """
        introspector = create_introspector(self)
        layer_info = introspector.get_layer_info()
        param_counts = introspector.calculate_parameters()
        memory_usage = introspector.estimate_memory_usage()
        configuration = introspector.get_model_configuration()
        
        return ModelSummaryFormatter.format_model_info(
            layer_info, param_counts, memory_usage, configuration
        )


# --- END Multi_Layer_CNN ---

# NOTE: The provided init.py is correct for exposing the new class:
# from.model import Multi_Layer_ANN, Plotting_Utils
# from.model import ConvLayer
# from.model import Flatten
# from.model import Multi_Layer_CNN # You must add this import