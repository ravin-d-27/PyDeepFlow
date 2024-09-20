import numpy as np
from activations import activation, activation_derivative
from losses import get_loss_function, get_loss_derivative

class Multi_Layer_ANN:
    
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='binary_crossentropy'):
        self.layers = [X_train.shape[1]] + hidden_layers + [1]
        self.activations = activations
        self.weights = []
        self.biases = []
        
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, X):
        activations = [X]
        Z_values = []

        for i in range(len(self.weights) - 1):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = activation(Z, self.activations[i])
            activations.append(A)

        Z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, 'sigmoid')
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate):
        output_error = y.reshape(-1, 1) - activations[-1]
        d_output = output_error * activation_derivative(activations[-1], 'sigmoid')

        deltas = [d_output]
        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * activation_derivative(activations[i + 1], self.activations[i])
            deltas.append(delta)
        
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            
    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            activations, Z_values = self.forward_propagation(X_train)
            self.backpropagation(X_train, y_train, activations, Z_values, learning_rate)

            loss = self.loss_func(y_train, activations[-1])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return (activations[-1] >= 0.5).astype(int).flatten()
