import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.cross_validator import CrossValidator  # Import CrossValidator

def load_and_preprocess_data(url):
    # Load the Iris dataset
    df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    print(df.head())

    # Encode species labels to integers
    df['species'] = df['species'].astype('category').cat.codes

    # Split data into features (X) and labels (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert labels to one-hot encoding
    y_one_hot = np.eye(len(np.unique(y)))[y]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_one_hot

if __name__ == "__main__":
    # Configuration
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    n_splits = 5  # Number of folds for cross-validation

    # Load and preprocess data
    X, y_one_hot = load_and_preprocess_data(url)

    # Ask the user whether to use GPU
    use_gpu_input = input("Use GPU? (y/n): ").strip().lower()
    use_gpu = True if use_gpu_input == 'y' else False

    # Define the architecture of the network
    hidden_layers = [5, 5]
    activations = ['relu', 'relu']

    # Initialize the ANN with use_gpu option
    ann = Multi_Layer_ANN(X, y_one_hot, hidden_layers, activations, loss='categorical_crossentropy', use_gpu=use_gpu)

    # Initialize CrossValidator and perform K-Fold Cross Validation
    cross_validator = CrossValidator(k=n_splits, metrics=["accuracy"])
    results = cross_validator.evaluate(ann, X, y_one_hot, epochs=1000, learning_rate=0.01, verbose=True)

    # Print cross-validation results
    print("Cross-Validation Results:", results)

    # Optionally train the model on the full dataset if needed
    # ann.fit(epochs=1000, learning_rate=0.01)

    # Example of making predictions on the entire dataset or a separate test set can be added here
