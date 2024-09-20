import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ANN:
    
    def __init__(self, X_train, Y_train):
        self.input_layer_neurons = X_train.shape[1]
        self.hidden_layer_neurons = 5
        self.output_neurons = 1
        
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(self.input_layer_neurons, 
                                                    self.hidden_layer_neurons) * np.sqrt(2 / self.input_layer_neurons)
        self.weights_hidden_output = np.random.randn(self.hidden_layer_neurons, 
                                                     self.output_neurons) * np.sqrt(2 / self.hidden_layer_neurons)
        self.bias_hidden = np.zeros((1, self.hidden_layer_neurons))
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
        
        z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        a_hidden = self.relu(z_hidden)  
        
        
        z_output = np.dot(a_hidden, self.weights_hidden_output) + self.bias_output
        a_output = self.sigmoid(z_output)  
        
        return a_hidden, a_output

    def backpropagation(self, X, y, a_hidden, a_output, learning_rate):
        # Output error
        output_error = y.reshape(-1, 1) - a_output
        d_output = output_error * self.sigmoid_derivative(a_output)
        
        # Hidden layer error
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * self.relu_derivative(a_hidden)  
        
        # # Debug: Check the gradients
        # print("Gradients for output layer:", d_output)
        # print("Gradients for hidden layer:", d_hidden)

        # Update weights and biases
        self.weights_hidden_output += a_hidden.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Debug: Check updated weights and biases
        # print("Updated weights from hidden to output:", self.weights_hidden_output)
        # print("Updated bias for output:", self.bias_output)
        # print("Updated weights from input to hidden:", self.weights_input_hidden)
        # print("Updated bias for hidden layer:", self.bias_hidden)

    def predict(self, X):
        _, a_output = self.forward_propagation(X)
        return (a_output >= 0.5).astype(int).flatten()
    

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
    epochs = 100
    learning_rate = 0.05
    
    for epoch in range(epochs):
        a_hidden, a_output = ann.forward_propagation(X_train)
        ann.backpropagation(X_train, y_train, a_hidden, a_output, learning_rate)
        
        # Compute and print loss
        #loss = np.mean(np.square(y_train - a_output))  # Mean Squared Error (MSE)
        
        loss = -np.mean(y_train * np.log(a_output + 1e-8) + (1 - y_train) * np.log(1 - a_output + 1e-8)) # Binary Cross Entropy

        
        # Debug: Check loss every 100 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    
    # Predict and evaluate
    y_pred = ann.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
