# cnn_runner.py - Digits Image Classification Demo (using sklearn, no TensorFlow)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from pydeepflow.model import Multi_Layer_ANN, ConvLayer, Flatten, Plotting_Utils
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

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test


# --- CNN Model Demo ---
if __name__ == "__main__":

    print("Loading and preparing Digits dataset...")
    X_train, y_train, X_test, y_test = load_and_preprocess_digits()

    # Use all training samples
    X_train_sub = X_train
    y_train_sub = y_train

    # Small validation set (last 100 from training)
    X_val_raw = X_train_sub[-100:]
    y_val = y_train_sub[-100:]

    X_train_sub = X_train_sub[:-100]
    y_train_sub = y_train_sub[:-100]

    # --- MODEL ARCHITECTURE ---
    print("\nStarting ConvLayer and Flatten Test...")

    # Conv layers
    conv1 = ConvLayer(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    conv2 = ConvLayer(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)

    flatten = Flatten()

    # Forward pass for training data
    X_forward = conv1.forward(X_train_sub)
    X_forward = activation(X_forward, "relu", conv1.device)
    X_forward = conv2.forward(X_forward)
    X_forward = activation(X_forward, "relu", conv2.device)
    X_flat = flatten.forward(X_forward)

    # Forward pass for validation data
    X_val_forward = conv1.forward(X_val_raw)
    X_val_forward = activation(X_val_forward, "relu", conv1.device)
    X_val_forward = conv2.forward(X_val_forward)
    X_val_forward = activation(X_val_forward, "relu", conv2.device)
    X_val_flat = flatten.forward(X_val_forward)

    print(f"Flattened training shape: {X_flat.shape}")
    print(f"Flattened validation shape: {X_val_flat.shape}")

    # --- DENSE NETWORK ---
    model = Multi_Layer_ANN(
        X_flat, y_train_sub,
        hidden_layers=[128],
        activations=["relu"],
        loss='categorical_crossentropy',
        use_gpu=False,
        optimizer='adam'
    )

    # Train the network
    print("\nStarting training for 50 epochs...")
    model.fit(
        epochs=50,
        learning_rate=0.01,
        verbose=True,
        X_val=X_val_flat,
        y_val=y_val
    )

    # Final test set accuracy
    # Forward pass for test data
    X_test_forward = conv1.forward(X_test)
    X_test_forward = activation(X_test_forward, "relu", conv1.device)
    X_test_forward = conv2.forward(X_test_forward)
    X_test_forward = activation(X_test_forward, "relu", conv2.device)
    X_test_flat = flatten.forward(X_test_forward)

    y_pred = model.predict(X_test_flat)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100

    print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")

    # Optional: plot training history
    plot_util = Plotting_Utils()
    plot_util.plot_training_history(model.history, metrics=('loss', 'accuracy'), figure='training_history.png')
