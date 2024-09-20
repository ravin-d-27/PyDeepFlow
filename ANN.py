import numpy as np

class ANN:
    
    def __init__(self, X_train, Y_train):
        self.input_layer_neurons = X_train.shape[1]
        self.hidden_layer_neurons = 10
        self.output_neurons = 1
        
        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(self.input_layer_neurons, self.hidden_layer_neurons) - 0.5
        self.weights_hidden_output = np.random.rand(self.hidden_layer_neurons, self.output_neurons) - 0.5
        self.bias_hidden = np.random.rand(1, self.hidden_layer_neurons) - 0.5
        self.bias_output = np.random.rand(1, self.output_neurons) - 0.5
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward propagation
    def forward_propagation(self, X):
        z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # Corrected
        a_hidden = self.sigmoid(z_hidden)
        
        z_output = np.dot(a_hidden, self.weights_hidden_output) + self.bias_output  # Corrected
        a_output = self.sigmoid(z_output)
        
        return a_hidden, a_output

    # Backpropagation
    def backpropagation(self, X, y, a_hidden, a_output, learning_rate):
        # Use self to reference weights and biases
        output_error = y - a_output
        d_output = output_error * self.sigmoid_derivative(a_output)
        
        hidden_error = d_output.dot(self.weights_hidden_output.T)  # Corrected
        d_hidden = hidden_error * self.sigmoid_derivative(a_hidden)
        
        # Update weights and biases using self
        self.weights_hidden_output += a_hidden.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Predict function
    def predict(self, X):
        _, a_output = self.forward_propagation(X)
        return np.round(a_output)
