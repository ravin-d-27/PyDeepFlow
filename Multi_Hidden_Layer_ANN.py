import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ANN:
    
    def __init__(self, X_train, Y_train):
        self.input_layer_neurons = X_train.shape[1]
        self.hidden_layer1_neurons = 5
        self.hidden_layer2_neurons = 5  # Added another hidden layer with 5 neurons
        self.output_neurons = 1
        
        np.random.seed(42)
        # Weights and biases initialization for the first hidden layer
        self.weights_input_hidden1 = np.random.randn(self.input_layer_neurons, 
                                                    self.hidden_layer1_neurons) * np.sqrt(2 / self.input_layer_neurons)
        self.bias_hidden1 = np.zeros((1, self.hidden_layer1_neurons))
        
        # Weights and biases initialization for the second hidden layer
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_layer1_neurons, 
                                                     self.hidden_layer2_neurons) * np.sqrt(2 / self.hidden_layer1_neurons)
        self.bias_hidden2 = np.zeros((1, self.hidden_layer2_neurons))
        
        # Weights and biases for output layer
        self.weights_hidden2_output = np.random.randn(self.hidden_layer2_neurons, 
                                                      self.output_neurons) * np.sqrt(2 / self.hidden_layer2_neurons)
        self.bias_output = np.zeros((1, self.output_neurons))
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        # Forward propagation through the first hidden layer
        z_hidden1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        a_hidden1 = self.relu(z_hidden1)
        
        # Forward propagation through the second hidden layer
        z_hidden2 = np.dot(a_hidden1, self.weights_hidden1_hidden2) + self.bias_hidden2
        a_hidden2 = self.relu(z_hidden2)
        
        # Output layer
        z_output = np.dot(a_hidden2, self.weights_hidden2_output) + self.bias_output
        a_output = self.sigmoid(z_output)
        
        return a_hidden1, a_hidden2, a_output

    def backpropagation(self, X, y, a_hidden1, a_hidden2, a_output, learning_rate):
        # Output error
        output_error = y.reshape(-1, 1) - a_output
        d_output = output_error * self.sigmoid_derivative(a_output)
        
        # Hidden layer 2 error
        hidden2_error = d_output.dot(self.weights_hidden2_output.T)
        d_hidden2 = hidden2_error * self.relu_derivative(a_hidden2)
        
        # Hidden layer 1 error
        hidden1_error = d_hidden2.dot(self.weights_hidden1_hidden2.T)
        d_hidden1 = hidden1_error * self.relu_derivative(a_hidden1)
        
        # Update weights and biases for output layer
        self.weights_hidden2_output += a_hidden2.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        # Update weights and biases for second hidden layer
        self.weights_hidden1_hidden2 += a_hidden1.T.dot(d_hidden2) * learning_rate
        self.bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
        
        # Update weights and biases for first hidden layer
        self.weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate
        self.bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

    def predict(self, X):
        _, _, a_output = self.forward_propagation(X)
        return (a_output >=0.5).astype(int).flatten()
    

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


    # Initialize ANN
    ann = ANN(X_train, y_train)
    epochs = 1000
    learning_rate = 0.05
    
    for epoch in range(epochs):
        a_hidden1, a_hidden2, a_output = ann.forward_propagation(X_train)
        ann.backpropagation(X_train, y_train, a_hidden1, a_hidden2, a_output, learning_rate)
        
        # Compute and print loss
        loss = -np.mean(y_train * np.log(a_output + 1e-8) + (1 - y_train) * np.log(1 - a_output + 1e-8)) # Binary Cross Entropy
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    
    # Predict and evaluate
    y_pred = ann.predict(X_test)
    print(y_pred)
    # for i in range(len(y_pred)):
    #     print(y_pred[i],"-----",y_test[i])
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
