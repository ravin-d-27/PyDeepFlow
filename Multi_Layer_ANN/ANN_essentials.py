import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Multi_Layer_ANN:
    
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='binary_crossentropy'):
        
        
        self.layers = [X_train.shape[1]] + hidden_layers + [1]  
        self.activations = activations 
        self.weights = []
        self.biases = []
        
        self.loss = loss
        self.loss_func = self.binary_crossentropy
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
    def choose_loss(self):
        if self.loss=='binary_crossentropy':
            self.loss_func = self.binary_crossentropy
            self.loss_derivative = self.binary_crossentropy_derivative
        elif self.loss == 'mse' or self.loss == 'MSE':
            self.loss_func = self.mse
            self.loss_derivative = self.mse_derivative
        else:
            raise ValueError("Unsupported Loss Function: {}".format(self.loss))
        
    def binary_crossentropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def binary_crossentropy_derivative(self, y_true, y_pred):
        return -(y_true / (y_pred + 1e-8)) + (1 - y_true) / (1 - y_pred + 1e-8)
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) 
    
    def mse_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size    
        
    def activation(self, x, func):
        
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported activation function: {func}")

    def activation_derivative(self, x, func):
        
        if func == 'relu':
            return np.where(x > 0, 1, 0)
        elif func == 'sigmoid':
            return x * (1 - x)
        else:
            raise ValueError(f"Unsupported activation function: {func}")

    def forward_propagation(self, X):
        
        activations = [X]
        Z_values = []  # Store Z values to use in backpropagation

        for i in range(len(self.weights) - 1):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = self.activation(Z, self.activations[i])
            activations.append(A)
        
        
        Z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = self.activation(Z_output, 'sigmoid')
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate):
        
        # Output layer error
        output_error = y.reshape(-1, 1) - activations[-1]
        d_output = output_error * self.activation_derivative(activations[-1], 'sigmoid')
        
        deltas = [d_output]
        
        # Backpropagate through the hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * self.activation_derivative(activations[i + 1], self.activations[i])
            deltas.append(delta)
        
        deltas.reverse()  # Now the deltas are in the correct order
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            
    def fit(self, epochs, learning_rate):
        
        
        for epoch in range(epochs):
            activations, Z_values = self.forward_propagation(X_train)
            self.backpropagation(X_train, y_train, activations, Z_values, learning_rate)
            
            # Compute and print loss
            loss = self.loss_func(y_train, activations[-1])
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        
        activations, _ = self.forward_propagation(X)
        return (activations[-1] >= 0.5).astype(int).flatten()
    
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Naive_Bayes/Naive-Bayes-Classification-Data.csv")
    
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.ravel()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    hidden_layers = [5, 5]  
    activations = ['relu', 'relu'] 

    ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations)
    
    ann.fit(epochs = 1000, learning_rate = 0.05)
    
    # Predict and evaluate
    y_pred = ann.predict(X_test)
    print(y_pred)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")