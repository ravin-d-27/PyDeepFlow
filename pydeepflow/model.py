# model.py
import numpy as np
from .activations import activation, activation_derivative  
from .losses import get_loss_function, get_loss_derivative  
from .device import Device                                  
from .regularization import Regularization
from tqdm import tqdm
import time


class Multi_Layer_ANN:
    """
    A Multi-Layer Artificial Neural Network (ANN) class for binary and multi-class classification tasks.
    Attributes:
        device: The device (CPU/GPU) where the computations will be performed.
        layers: List of the number of neurons in each layer of the network.
        activations: List of activation functions for each hidden layer.
        weights: List of weight matrices for each layer.
        biases: List of bias vectors for each layer.
        loss: The type of loss function being used.
        loss_func: The callable loss function used during training.
        loss_derivative: The callable derivative of the loss function for backpropagation.
        X_train: Training feature set moved to the specified device.
        y_train: Training label set moved to the specified device.
    """

    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy',
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0):
        """
        Initializes the ANN model with the provided architecture and configurations.
        Parameters:
            X_train (np.ndarray): The training feature set.
            Y_train (np.ndarray): The training label set (one-hot encoded for multi-class).
            hidden_layers (list): A list specifying the number of neurons in each hidden layer.
            activations (list): A list specifying the activation function for each hidden layer.
            loss (str): The type of loss function to use ('categorical_crossentropy' or
            'binary_crossentropy').
            use_gpu (bool): Whether to use GPU for computations. Defaults to False.
            l2_lambda (float): The regularization coefficient for L2 regularization.
            dropout_rate (float): Dropout rate to prevent overfitting.
        """
        
        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)

        # Determine the network architecture based on the classification task (binary or multi-class)
        if Y_train.ndim == 1 or Y_train.shape[1] == 1:  # Binary classification
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else:  # Multi-class classification
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'

        self.activations = activations
        self.weights = []
        self.biases = []

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

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.
        Parameters:
        X (np.ndarray): Input data.
        Returns:
        tuple: A tuple containing:
        - activations (list): List of activations for each layer.
        - Z_values (list): List of Z (pre-activation) values for each layer.
        """
        activations = [X]
        Z_values = []

        # Forward pass through hidden layers with dropout
        for i in range(len(self.weights) - 1):
            Z = self.device.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = activation(Z, self.activations[i], self.device)
            A = self.regularization.apply_dropout(A, training=self.training)  # Apply dropout during training
            activations.append(A)

        # Forward pass through the output layer
        Z_output = self.device.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, self.output_activation, self.device)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate):
        """
        Performs backpropagation through the network to compute weight updates.
        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.
        activations (list): List of activations from forward propagation.
        Z_values (list): List of Z (pre-activation) values from forward propagation.
        learning_rate (float): The learning rate for gradient updates.
        """
        # Calculate the error in the output layer
        output_error = y - activations[-1]
        d_output = output_error * activation_derivative(activations[-1], self.output_activation, self.device)

        # Backpropagate through the network
        deltas = [d_output]
        for i in reversed(range(len(self.weights) - 1)):
            error = self.device.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * activation_derivative(activations[i + 1], self.activations[i], self.device)
            deltas.append(delta)
        
        deltas.reverse()

        # Update weights and biases with L2 regularization
        for i in range(len(self.weights)):
            self.weights[i] += self.device.dot(activations[i].T, deltas[i]) * learning_rate
            self.biases[i] += self.device.sum(deltas[i], axis=0, keepdims=True) * learning_rate
        # Apply L2 regularization to the weights
        self.weights[i] -= learning_rate * self.regularization.apply_l2_regularization(self.weights[i], learning_rate, X.shape)

    def fit(self, epochs, learning_rate=0.01, lr_scheduler=None):
        """
        Trains the model for a given number of epochs with an optional learning rate scheduler.
        Parameters:
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        lr_scheduler (LearningRateScheduler optional): An instance of LearningRateScheduler for
        dynamic learning rate adjustment.
        """
        prev_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green'):
            start_time = time.time()

            # Adjust the learning rate using the scheduler if provided
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate
            # Forward and Backpropagation
            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr)

            self.training = False
            # Compute loss and accuracy
            loss = self.loss_func(self.y_train, activations[-1], self.device)
            accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation \
            == 'sigmoid' else \
            np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            # Log the loss change and update previous loss
            loss_change = prev_loss - loss if prev_loss != float('inf') else 0
            prev_loss = loss

            # Log training progress
            epoch_time = time.time() - start_time
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} Loss: {loss:.4f} Accuracy: {accuracy * 100:.2f}% Time: \
                {epoch_time:.2f}s Learning Rate: {current_lr:.6f}")
        print("Training Completed!")


    def predict(self, X):
        """
        Makes predictions based on input data using the trained model.
        Parameters:
        X (np.ndarray): Input data.
        Returns:
        np.ndarray: The predicted class labels for the input data.
        """
        activations, _ = self.forward_propagation(self.device.array(X))
        if self.output_activation == 'sigmoid':
            return self.device.asnumpy((activations[-1] >= 0.5).astype(int)).flatten()
        elif self.output_activation == 'softmax':
            return self.device.asnumpy(np.argmax(activations[-1], axis=1))
        
    def predict_prob(self, X):
        """
        Predicts the probability distribution for the input data.
        Parameters:
        X (np.ndarray): Input data.
        Returns:
        np.ndarray: The predicted probabilities for the input data.
        """
        activations, _ = self.forward_propagation(self.device.array(X))
        return self.device.asnumpy(activations[-1])