import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Multi_Layer_ANN

if __name__ == "__main__":
    
    # Load Iris dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

    print(df.head())

    # Encode species labels to integers
    df['species'] = df['species'].astype('category').cat.codes

    # Split data into features (X) and labels (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert labels to one-hot encoding
    y_one_hot = np.eye(len(np.unique(y)))[y]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ask the user whether to use GPU
    use_gpu_input = input("Use GPU? (y/n): ").strip().lower()
    use_gpu = True if use_gpu_input == 'y' else False

    # Define the architecture of the network
    hidden_layers = [5, 5]
    activations = ['relu', 'relu']

    # Initialize the ANN with use_gpu option
    ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='categorical_crossentropy', use_gpu=use_gpu)

    # Train the model
    ann.fit(epochs=1000, learning_rate=0.01)

    # Make predictions on the test set
    y_pred = ann.predict(X_test)
    print(y_pred)

    # Convert one-hot encoded test labels back to integers
    y_test_labels = np.argmax(y_test, axis=1)

    # Calculate the accuracy of the model
    accuracy = np.mean(y_pred == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
