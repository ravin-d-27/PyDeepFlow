import numpy as np
from activations import activation, activation_derivative
from losses import get_loss_function, get_loss_derivative

from tqdm import tqdm
import time
from colorama import Fore, Style

# Dear users, please read the comments for better understanding of the code.  If you find any optimizations in my implementation, please don't hesitate to make a pull request and fix it
class Multi_Layer_ANN:
    
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy'):
        
        if Y_train.ndim == 1 or Y_train.shape[1] == 1:  # Binary classification
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else:  # Multi-class classification
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'
        
        self.activations = activations
        self.weights = []
        self.biases = []
        
        # Set the loss function based on binary or multi-class classification
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)
        
        self.X_train = X_train
        self.y_train = Y_train
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]) # This is called He Initialization to address the problem of vanishing/exploding gradients
            bias_vector = np.zeros((1, self.layers[i + 1])) # Initializing the bias vector for the neurons in layer i + 1 as zeros
            # shape of bias_vector = (1, number of units in the next layer)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, X):
        activations = [X] # This list will eventually store the activations of each layer in the network as we perform forward propagation.
        Z_values = [] #  linear combinations of inputs, weights, and biases before the activation function is applied at each layer.
        # It is nothing but Z=W⋅A+b

        for i in range(len(self.weights) - 1):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = activation(Z, self.activations[i])
            activations.append(A)

        # Adjust the output layer activation based on binary or multi-class classification
        Z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, self.output_activation)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate):
        # Adjust error calculation for binary or multi-class classification
        output_error = y - activations[-1] # error at the output layer
        d_output = output_error * activation_derivative(activations[-1], self.output_activation) # derivative of the loss with respect to the output activation, incorporating the derivative of the activation function 

        deltas = [d_output] # it will hold the error (or gradient) for each layer during backpropagation.
        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * activation_derivative(activations[i + 1], self.activations[i])
            deltas.append(delta)
        
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            
    def fit(self, epochs, learning_rate):
        prev_loss = float('inf')  
        
        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green'):
            start_time = time.time()  

            
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, learning_rate)

            
            loss = self.loss_func(self.y_train, activations[-1])
            accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))            
            loss_change = prev_loss - loss if prev_loss != float('inf') else 0
            prev_loss = loss  # Update previous loss

            # Time taken for this epoch
            epoch_time = time.time() - start_time

            # Print every 10th epoch with colors and formatted output
            if epoch % 10 == 0:
                
                print(f"Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL} | "
                    f"Accuracy: {Fore.GREEN}{accuracy * 100:.2f}%{Style.RESET_ALL} | "
                    f"Time: {epoch_time:.2f}s")
                
        print()
        print()
        print("Training is Completed Successfully !")
        print()
        print()

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        
        # Conditional logic based on the output activation function
        if self.output_activation == 'sigmoid':
            return (activations[-1] >= 0.5).astype(int).flatten()
        elif self.output_activation == 'softmax':
            return np.argmax(activations[-1], axis=1)  # Return class indices for multi-class
        else:
            raise ValueError(f"Unsupported output activation function: {self.output_activation}")

