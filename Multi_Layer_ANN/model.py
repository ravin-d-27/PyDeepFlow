# model.py
import numpy as np
from activations import activation, activation_derivative
from losses import get_loss_function, get_loss_derivative
from device import Device
from tqdm import tqdm
import time

class Multi_Layer_ANN:
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy', use_gpu=False):
        self.device = Device(use_gpu=use_gpu)

        if Y_train.ndim == 1 or Y_train.shape[1] == 1:  # Binary classification
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else:  # Multi-class classification
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'

        self.activations = activations
        self.weights = []
        self.biases = []

        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # Move training data to the device (GPU or CPU)
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)

        # Initialize weights and biases using the device
        for i in range(len(self.layers) - 1):
            weight_matrix = self.device.random().randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias_vector = self.device.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, X):
        activations = [X]
        Z_values = []

        for i in range(len(self.weights) - 1):
            Z = self.device.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = activation(Z, self.activations[i], self.device)
            activations.append(A)

        Z_output = self.device.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, self.output_activation, self.device)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate):
        output_error = y - activations[-1]
        d_output = output_error * activation_derivative(activations[-1], self.output_activation, self.device)

        deltas = [d_output]
        for i in reversed(range(len(self.weights) - 1)):
            error = self.device.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * activation_derivative(activations[i + 1], self.activations[i], self.device)
            deltas.append(delta)
        
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += self.device.dot(activations[i].T, deltas[i]) * learning_rate
            self.biases[i] += self.device.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def fit(self, epochs, learning_rate):
        prev_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green'):
            start_time = time.time()

            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, learning_rate)

            loss = self.loss_func(self.y_train, activations[-1], self.device)
            accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            loss_change = prev_loss - loss if prev_loss != float('inf') else 0
            prev_loss = loss

            epoch_time = time.time() - start_time
            if epoch % 10 == 0:
                print(f"Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}% | Time: {epoch_time:.2f}s")

        print("Training Completed!")

    def predict(self, X):
        activations, _ = self.forward_propagation(self.device.array(X))
        if self.output_activation == 'sigmoid':
            return self.device.asnumpy((activations[-1] >= 0.5).astype(int)).flatten()
        elif self.output_activation == 'softmax':
            return self.device.asnumpy(np.argmax(activations[-1], axis=1))
        
    def predict_prob(self, X):
        activations, _ = self.forward_propagation(self.device.array(X))
        return self.device.asnumpy(activations[-1])
