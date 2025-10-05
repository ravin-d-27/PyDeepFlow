import numpy as np
import matplotlib.pyplot as plt
from pydeepflow.activations import activation, activation_derivative
from pydeepflow.losses import get_loss_function, get_loss_derivative
from pydeepflow.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pydeepflow.device import Device
from pydeepflow.regularization import Regularization
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.cross_validator import CrossValidator
from pydeepflow.batch_normalization import BatchNormalization
from tqdm import tqdm
from pydeepflow.optimizers import Adam, RMSprop
import time
import sys

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
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, use_batch_norm=False, optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32):
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
        self._validate_inputs(X_train, Y_train, hidden_layers, activations, loss, 
                             l2_lambda, dropout_rate, optimizer, learning_rate, epochs, batch_size)
        
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

    def _validate_inputs(self, X_train, Y_train, hidden_layers, activations, loss, 
                        l2_lambda, dropout_rate, optimizer, learning_rate, epochs, batch_size):
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



class Plotting_Utils:
    """
    A utility class for plotting model training history.

    This class provides methods to visualize metrics such as loss and accuracy
    over the course of training, which is useful for diagnosing issues like
    overfitting or underfitting.
    """
    def plot_training_history(self, history, metrics=('loss', 'accuracy'), figure='history.png'):
        """
        Plots the training and validation history for specified metrics.

        Args:
            history (dict): A dictionary containing the training history.
                            Expected keys are 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'.
            metrics (tuple, optional): A tuple of metrics to plot, e.g., ('loss', 'accuracy').
                                       Defaults to ('loss', 'accuracy').
            figure (str, optional): The filename to save the plot to. Defaults to 'history.png'.
        """
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
        """
        Plots a learning curve for a model.

        A learning curve shows the validation and training score of an estimator for varying numbers of training samples.
        It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers
        more from a variance error or a bias error.

        Args:
            train_sizes (list or np.ndarray): Numbers of training examples that has been used to generate the learning curve.
            train_scores (np.ndarray): Scores on training sets.
            val_scores (np.ndarray): Scores on validation sets.
            figure (str, optional): The filename to save the plot to. Defaults to 'learning_curve.png'.
        """
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
