import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.cross_validator import CrossValidator  # Import CrossValidator

def load_and_preprocess_data(url):
    """
    Loads the Iris dataset from a URL, preprocesses it, and prepares it for training.

    This function performs the following steps:
    1. Loads the data from the provided URL using pandas.
    2. Encodes the categorical species labels into numerical format.
    3. Separates the features (X) from the labels (y).
    4. Converts the numerical labels to a one-hot encoded format.
    5. Standardizes the features using StandardScaler to have a mean of 0 and a variance of 1.

    Args:
        url (str): The URL to the Iris dataset (or a local file path).

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The standardized and preprocessed feature data.
            - y_one_hot (np.ndarray): The one-hot encoded labels.
    """
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
    """
    Main execution block to train and evaluate the neural network on the Iris dataset.

    This script serves as an example of how to use the Multi_Layer_ANN class. It includes:
    - Configuration for the dataset URL and cross-validation folds.
    - Loading and preprocessing of the Iris dataset.
    - User input to decide whether to use a GPU for computation.
    - Definition of the neural network architecture (hidden layers and activation functions).
    - Initialization of the Multi_Layer_ANN model.
    - K-fold cross-validation to evaluate the model's performance.
    - Printing the cross-validation results.
    """
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

    # Initialize CrossValidator
    cross_validator = CrossValidator(n_splits=n_splits)

    # Perform K-Fold Cross Validation and pass metrics to the evaluate method.
    results = cross_validator.evaluate(
        ann,
        X,
        y_one_hot,
        epochs=1000,
        learning_rate=0.01,
        metrics=["accuracy"],  # Metrics argument moved here
        verbose=True
    )

    # Print cross-validation results
    print("Cross-Validation Results:", results)

    # Optionally train the model on the full dataset if needed
    # ann.fit(epochs=1000, learning_rate=0.01)

    # Example of making predictions on the entire dataset or a separate test set can be added here
