import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from pydeepflow.activations import activation, activation_derivative
from pydeepflow.losses import get_loss_function, get_loss_derivative
from pydeepflow.metrics import precision_score, recall_score, f1_score, confusion_matrix,mean_absolute_error, mean_squared_error, r2_score
from pydeepflow.device import Device
from pydeepflow.regularization import Regularization
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.cross_validator import CrossValidator
from pydeepflow.batch_normalization import BatchNormalization
from pydeepflow.weight_initialization import get_weight_initializer
from tqdm import tqdm
from pydeepflow.optimizers import Adam, RMSprop
from pydeepflow.validation import ModelValidator
import numpy as np
import sys
import time

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

    # Create patches using broadcasting
    X_col = np.zeros((N, H_out, W_out, Fh * Fw * C))
    
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            # This indexing relies on the previously created index structure, though implemented via loops here
            X_col[:, :, :, y * Fw + x::Fh * Fw] = X_padded[:, y:y_max:stride, x:x_max:stride, :]
    
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
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            if padding > 0:
                X_grad[:, y:y_max:stride, x:x_max:stride, :] += cols_reshaped[:, :, :, y * Fw + x::Fh * Fw]
            else:
                X_grad[:, y:y_max:stride, x:x_max:stride, :] += cols_reshaped[:, :, :, y * Fw + x::Fh * Fw]

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
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Fh = kernel_size
        self.Fw = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device if device is not None else Device(use_gpu=False)

        # He/Kaiming initialization
        scale = np.sqrt(2.0 / (self.Fh * self.Fw * self.in_channels))
        W_init = self.device.random().randn(self.Fh, self.Fw, self.in_channels, self.out_channels) * scale
        b_init = self.device.zeros((1, 1, 1, self.out_channels))

        # store params in a dict for clarity
        self.params = {'W': W_init, 'b': b_init}
        self.grads = {'dW': None, 'db': None}
        self.cache = None

    def forward(self, X):
        """ X shape: (N, H, W, C_prev) """
        N, H, W, C = X.shape
        Fh, Fw = self.Fh, self.Fw

        H_out = (H + 2 * self.padding - Fh) // self.stride + 1
        W_out = (W + 2 * self.padding - Fw) // self.stride + 1

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
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, use_batch_norm=False, optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32, initial_weights='auto'):
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
        validator.validate_initial_weights(initial_weights)
        
        # Validate and adjust batch_size
        self.batch_size = validator.validate_training_hyperparameters(
            learning_rate, epochs, batch_size, X_train
        )
        
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
        self.weights = []
        self.biases = []

        
        if len(self.activations) != len(hidden_layers):
            raise ValueError("The number of activation functions must match the number of hidden layers.")

        # Setup loss function
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # Move training data to the device (GPU or CPU)
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)

        # Initialize weights and biases with He initialization for better convergence
        for i in range(len(self.layers) - 1):
            weight_matrix = self.device.random().randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias_vector = self.device.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

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

    def fit(self, epochs, learning_rate=0.01, lr_scheduler=None, early_stop=None, X_val=None, y_val=None, checkpoint=None, verbose=False, clipping_threshold=None):
        """
        Trains the neural network model.

        Args:
            epochs (int): The number of epochs to train the model.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            lr_scheduler (object, optional): A learning rate scheduler. Defaults to None.
            early_stop (object, optional): An early stopping callback. Defaults to None.
            X_val (np.ndarray, optional): Validation features. Defaults to None.
            y_val (np.ndarray, optional): Validation labels. Defaults to None.
            checkpoint (object, optional): A model checkpointing callback. Defaults to None.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
            clipping_threshold (float, optional): The value for gradient clipping. Defaults to None.

        Raises:
            AssertionError: If early stopping is enabled but no validation set is provided.
        """
        if early_stop:
            assert X_val is not None and y_val is not None, "Validation set is required for early stopping"

        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green', disable=not verbose):
            start_time = time.time()

            # Adjust the learning rate using the scheduler if provided
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate

            # Forward and Backpropagation
            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr, clip_value=clipping_threshold)

            self.training = False

            # Compute training loss and accuracy
            train_loss = self.loss_func(self.y_train, activations[-1], self.device)
            train_accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            # # Debugging output
            # print(f"Computed Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

            if train_loss is None or train_accuracy is None:
                print("Warning: train_loss or train_accuracy is None!")
                continue  # Skip this epoch if values are not valid

            # Validation step
            val_loss = val_accuracy = None
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(self.device.array(X_val))
                val_loss = self.loss_func(self.device.array(y_val), val_activations[-1], self.device)
                val_accuracy = np.mean((val_activations[-1] >= 0.5).astype(int) == y_val) if self.output_activation == 'sigmoid' else np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

            # Store training history for plotting
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            # Checkpoint saving logic
            if checkpoint is not None and X_val is not None:
                if checkpoint.should_save(epoch, val_loss):
                    checkpoint.save_weights(epoch, self.weights, self.biases, val_loss)

            if verbose and (epoch % 10 == 0):
        # Display progress on the same line
                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Accuracy: {train_accuracy:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.2f}% | "
                    f"Learning Rate: {current_lr:.6f}   "
                )
                sys.stdout.flush()

            # Early stopping
            if early_stop: 
                early_stop(val_loss)
                if early_stop.early_stop:
                    print('\n', "#" * 150, '\n\n', "early stop at - "
                        f"Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}% "
                        f"Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy * 100:.2f}% "
                        f"Learning Rate: {current_lr:.6f}", '\n\n', "#" * 150)
                    break
                    
        print("Training Completed!")

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
                                      Available metrics: 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'.

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

# ====================================================================
# Plotting utilities (UNMODIFIED)
# ====================================================================

class Plotting_Utils:
    def plot_training_history(self, history, metrics=('loss', 'accuracy'), figure='history.png'):
        num_metrics = len(metrics)
        fig, ax = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        if num_metrics == 1:
            ax = [ax]

        for i, metric in enumerate(metrics):
            ax[i].plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
            if f'val_{metric}' in history:
                ax[i].plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
            ax[i].set_title(f"{metric.capitalize()} over Epochs")
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(metric.capitalize())
            ax[i].legend()

        plt.savefig(figure)
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, train_sizes, train_scores, val_scores, figure='learning_curve.png'):
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                             val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig(figure)
        plt.show()


# ====================================================================
# NEW CLASS: Multi_Layer_CNN (The Structural Fix)
# ====================================================================

class Multi_Layer_CNN:
    """
    A Sequential Model wrapper that chains Convolutional, Flatten, and Dense layers.
    It implements end-to-end forward propagation and backpropagation for CNN training.
    """
    def __init__(self, layers_list, X_train, Y_train, loss='categorical_crossentropy',
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, use_batch_norm=False, optimizer='sgd'):
        
        # Validate inputs before proceeding with initialization
        validator = ModelValidator(device=None)  # Device not needed for validation
        validator.validate_training_data(X_train, "X_train", max_dimensions=4)  # CNN supports up to 4D
        validator.validate_training_data(Y_train, "Y_train", max_dimensions=2)  # Labels are typically 1D or 2D
        validator.validate_data_compatibility(X_train, Y_train)
        validator.validate_cnn_layers(layers_list)
        validator.validate_cnn_input_data(X_train, layers_list)
        validator.validate_loss_function(loss)
        validator.validate_regularization_params(l2_lambda, dropout_rate)
        validator.validate_optimizer(optimizer)
        
        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)
        
        # Loss setup
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # Data and state setup
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)
        self.training = False
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        self.layers_list = []  # Stores ConvLayer/Flatten objects and Dense dicts
        self.trainable_params = []  # List of W/b arrays for optimization (Conv W, Conv b, Dense W, Dense b,...)
        
        # --- 1. Construct Layers and Initialize Weights ---
        
        current_input_shape = X_train.shape[1:]  # (H, W, C) or (Features,)
        
        for layer_config in layers_list:
            layer_type = layer_config['type'].lower()
            
            if layer_type == 'conv':
                if len(current_input_shape) != 3:
                    raise ValueError("ConvLayer requires 4D input (N, H, W, C). Check previous layer configuration.")
                    
                in_c = current_input_shape[-1]
                out_c = layer_config['out_channels']
                k_size = layer_config['kernel_size']
                stride = layer_config.get('stride', 1)
                padding = layer_config.get('padding', 0)
                
                conv_layer = ConvLayer(in_c, out_c, k_size, stride, padding, device=self.device)
                self.layers_list.append(conv_layer)
                
                # Update current shape for the next layer
                H, W = current_input_shape[0], current_input_shape[1]
                H_out = (H + 2 * padding - k_size) // stride + 1
                W_out = (W + 2 * padding - k_size) // stride + 1
                current_input_shape = (H_out, W_out, out_c) 
                
                # Add ConvLayer parameters to the trainable list
                self.trainable_params.extend([conv_layer.params['W'], conv_layer.params['b']])

            elif layer_type == 'flatten':
                if len(current_input_shape) < 3:
                    raise ValueError("Flatten layer expects 4D input (N, H, W, C).")
                
                flatten_layer = Flatten()
                self.layers_list.append(flatten_layer)
                
                current_input_dim = np.prod(current_input_shape)
                current_input_shape = (current_input_dim,) # New 2D shape

            elif layer_type == 'dense':
                if len(current_input_shape) != 1:
                    raise ValueError("DenseLayer expects 2D input (N, Features). Needs Flatten layer before.")

                input_dim = current_input_shape[0]
                output_dim = layer_config['neurons']
                activation_name = layer_config['activation']

                # Initialize Dense weights (W) and biases (b)
                scale = np.sqrt(2 / max(1, input_dim))
                w = self.device.random().randn(input_dim, output_dim) * scale
                b = self.device.zeros((1, output_dim))
                
                dense_layer = {'W': w, 'b': b, 'activation': activation_name}
                self.layers_list.append(dense_layer)
                
                current_input_shape = (output_dim,)
                
                # Add Dense layer parameters to the trainable list
                self.trainable_params.extend([dense_layer['W'], dense_layer['b']])

        # Final output settings (for loss function)
        self.output_dim = Y_train.shape[1] if Y_train.ndim > 1 else 1
        # Use the activation of the final dense layer config, but override the output layer if necessary
        final_layer_activation = layers_list[-1].get('activation', 'relu') if layers_list else 'relu'
        self.output_activation = 'softmax' if self.output_dim > 1 else 'sigmoid'

        # --- 2. Optimizer Setup ---
        if optimizer == 'adam':
            self.optimizer = Adam()
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop()
        else:
            self.optimizer = None # Default to SGD

    def forward_propagation(self, X):
        """
        Performs forward propagation by chaining through all layers (Conv -> Flatten -> Dense).
        Caches inputs and Z-values for backpropagation.
        """
        current_activation = X
        Z_values = []
        A_values = [X] # Stores all activation outputs (A0 = Input, A1, A2...)
        
        for layer_idx, layer in enumerate(self.layers_list):
            
            if isinstance(layer, ConvLayer) or isinstance(layer, Flatten):
                current_activation = layer.forward(current_activation)
                A_values.append(current_activation)

            elif isinstance(layer, dict) and 'W' in layer: # Dense layer
                W, b = layer['W'], layer['b']
                
                # Linear Transformation
                Z = self.device.dot(current_activation, W) + b
                Z_values.append(Z)

                # Activation
                if layer_idx == len(self.layers_list) - 1: # Final output layer
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
        
        # Accumulate gradients starting from the end of the layer list
        for i in reversed(range(len(self.layers_list))):
            layer = self.layers_list[i]

            if isinstance(layer, ConvLayer):
                # --- A. Conv Backward Pass ---
                
                dIn = layer.backward(dOut) # Populates layer.grads
                
                # Store Conv gradients (W then b) - insert at beginning to maintain order
                grads_to_update.insert(0, layer.grads['db']) # Insert bias first
                grads_to_update.insert(0, layer.grads['dW']) # Insert weights second
                    
                dOut = dIn # Gradient for the previous layer (Flatten or another Conv)

            elif isinstance(layer, Flatten):
                # --- B. Flatten Backward Pass ---
                dIn = layer.backward(dOut)
                dOut = dIn # Gradient for the previous layer (Conv)

            elif isinstance(layer, dict) and 'W' in layer: # Dense layer
                # --- C. Dense Backward Pass ---
                W_param = layer['W']
                A_prev = A_values[A_index] # Activation output of previous layer
                
                # 1. Calculate delta (dZ)
                if i == len(self.layers_list) - 1: # Output layer
                    delta = dOut 
                else:
                    # Hidden Dense layer: Error from next layer already passed in dOut
                    Z_curr = Z_values[Z_index]
                    delta = dOut * activation_derivative(Z_curr, layer['activation'], self.device)
                    
                # 2. Compute gradients dW and db
                dW = self.device.dot(A_prev.T, delta)
                db = self.device.sum(delta, axis=0, keepdims=True)
                
                # Store Dense gradients (W then b) - insert at beginning to maintain order
                grads_to_update.insert(0, db) # Insert bias first
                grads_to_update.insert(0, dW) # Insert weights second

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
            final_grads.append(grad_b) # Bias has no L2

            # Apply gradient clipping
            if clip_value is not None:
                final_grads[i] = self.regularization.clip_gradient(final_grads[i], clip_value)
                final_grads[i+1] = self.regularization.clip_gradient(final_grads[i+1], clip_value)

        if self.optimizer:
            # Optimizer updates ALL parameters in self.trainable_params list in place
            self.optimizer.update(self.trainable_params, final_grads)
        else:
            # Standard SGD update
            for i in range(len(self.trainable_params)):
                self.trainable_params[i] -= final_grads[i] * learning_rate


    # --- Methods copied from Multi_Layer_ANN and adapted for CNN structure ---
    
    def fit(self, epochs, learning_rate=0.01, lr_scheduler=None, early_stop=None, X_val=None, y_val=None, checkpoint=None, verbose=False, clipping_threshold=None):
        # Validate training hyperparameters
        validator = ModelValidator(device=None)
        batch_size = validator.validate_training_hyperparameters(
            learning_rate, epochs, 32, self.X_train  # Default batch_size=32 for CNN
        )
        
        if early_stop:
            assert X_val is not None and y_val is not None, "Validation set required for early stopping"

        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green', disable=not verbose):
            start_time = time.time()

            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate

            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr, clip_value=clipping_threshold)
            self.training = False

            # metrics
            train_loss = self.loss_func(self.y_train, activations[-1], self.device)
            train_accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            if train_loss is None or train_accuracy is None:
                print("Warning: train_loss or train_accuracy is None!")
                continue

            val_loss = val_accuracy = None
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(self.device.array(X_val))
                val_loss = self.loss_func(self.device.array(y_val), val_activations[-1], self.device)
                val_accuracy = np.mean((val_activations[-1] >= 0.5).astype(int) == y_val) if self.output_activation == 'sigmoid' else np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

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
                    print(f"Early stop at epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
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


# --- END Multi_Layer_CNN ---

# NOTE: The provided init.py is correct for exposing the new class:
# from.model import Multi_Layer_ANN, Plotting_Utils
# from.model import ConvLayer
# from.model import Flatten
# from.model import Multi_Layer_CNN # You must add this import