import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Multi_Layer_ANN

if __name__ == "__main__":
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

    print(df.head())

    df['species'] = df['species'].astype('category').cat.codes


    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    y_one_hot = np.eye(len(np.unique(y)))[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Define the architecture
    hidden_layers = [5, 5]
    activations = ['relu', 'relu']

    ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='categorical_crossentropy')
    ann.fit(epochs=1000, learning_rate=0.01)

    # Make predictions
    y_pred = ann.predict(X_test)

    print(y_pred)

    # Convert predictions back to original labels
    y_test_labels = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


