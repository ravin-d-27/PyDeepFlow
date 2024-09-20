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

    def forward_propagation(self, X):
        z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        a_hidden = self.sigmoid(z_hidden)
        
        z_output = np.dot(a_hidden, self.weights_hidden_output) + self.bias_output
        a_output = self.sigmoid(z_output)
        
        return a_hidden, a_output

    def backpropagation(self, X, y, a_hidden, a_output, learning_rate):
        output_error = y.reshape(-1, 1) - a_output 
        d_output = output_error * self.sigmoid_derivative(a_output)
        
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * self.sigmoid_derivative(a_hidden)
        
        self.weights_hidden_output += a_hidden.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def predict(self, X):
        _, a_output = self.forward_propagation(X)
        return np.round(a_output)
    

if __name__=="__main__":
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = pd.read_csv("Naive_Bayes/Naive-Bayes-Classification-Data.csv")
    
    print(df.head())
    
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.ravel()

    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    ann = ANN(X_train, y_train)
    epochs = 10
    learning_rate = 0.01
    
    for epoch in range(epochs):
        a_hidden, a_output = ann.forward_propagation(X_train)
        ann.backpropagation(X_train, y_train, a_hidden, a_output, learning_rate)
        
        loss = np.mean(np.square(y_train - a_output))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    
    y_pred = ann.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
            
    