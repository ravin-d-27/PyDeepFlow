# cnn_runner.py - Digits Image Classification Demo (using sklearn, no TensorFlow)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
# Import the new integrated class
from pydeepflow.model import Multi_Layer_ANN, ConvLayer, Flatten, Plotting_Utils, Multi_Layer_CNN
from pydeepflow.activations import activation

# --- Data Loading Function ---
def load_and_preprocess_digits():
    digits = load_digits()
    X = digits.images  # (1797, 8, 8)
    y = digits.target.reshape(-1, 1)

    # Normalize pixels 0-16 -> 0-1
    X = X.astype("float32") / 16.0

    # Reshape to CNN format (N, H, W, C)
    X = X.reshape((-1, 8, 8, 1)) 

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test


# --- CNN Model Demo (Uses Multi_Layer_CNN) ---
if __name__ == "__main__":

    print("Loading and preparing Digits dataset...")
    X_train, y_train, X_test, y_test = load_and_preprocess_digits()

    # Split for validation
    X_val = X_train[-100:]
    y_val = y_train[-100:]
    X_train = X_train[:-100]
    y_train = y_train[:-100]

    # --- DEFINE THE FULL SEQUENTIAL ARCHITECTURE ---
    # Conv -> Conv -> Flatten -> Dense -> Dense
    cnn_layers_config = [
        # First convolutional layer: 8x8x1 -> 6x6x16
        {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        
        # Second convolutional layer: 6x6x16 -> 4x4x32  
        {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        
        # Flatten: 4x4x32 -> 512
        {'type': 'flatten'},
        
        # First dense layer: 512 -> 64
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        
        # Output layer: 64 -> 10 (digits 0-9)
        {'type': 'dense', 'neurons': 10, 'activation': 'softmax'}
    ]

    print("\nInitializing Multi_Layer_CNN Model...")
    
    # Instantiate the new integrated model
    model = Multi_Layer_CNN(
        layers_list=cnn_layers_config,
        X_train=X_train,
        Y_train=y_train,
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    # Train the network end-to-end
    print("\nStarting integrated training for 50 epochs...")
    model.fit(
        epochs=50,
        learning_rate=0.01,
        verbose=True,
        X_val=X_val,
        y_val=y_val
    )

    # Final test set prediction (The model handles the Conv/Flatten/Dense chain internally)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100

    print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")

    # Optional: plot training history
    plot_util = Plotting_Utils()
    plot_util.plot_training_history(model.history, metrics=('loss', 'accuracy'), figure='training_history.png')
