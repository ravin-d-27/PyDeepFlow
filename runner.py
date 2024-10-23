import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.early_stopping import EarlyStopping
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.learning_rate_scheduler import LearningRateScheduler
from pydeepflow.model import Plotting_Utils  
from pydeepflow.cross_validator import CrossValidator  

if __name__ == "__main__":

    # Load Iris dataset from sklearn
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("First five rows of the dataset:")
    print(pd.DataFrame(X, columns=iris.feature_names).head())

    # Convert labels to one-hot encoding (for multiclass classification)
    y_one_hot = np.eye(len(np.unique(y)))[y]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Ask the user whether to use GPU (simulated as False for this example)
    use_gpu_input = False
    use_gpu = True if use_gpu_input == 'y' else False

    # Define the architecture of the network
    hidden_layers = [5, 5]  # Example: two hidden layers with 5 neurons each
    activations = ['relu', 'relu']  # ReLU activations for the hidden layers

    # Initialize the CrossValidator
    k_folds = 10 # Set the number of folds for cross-validation
    cross_validator = CrossValidator(n_splits=k_folds)

    # Perform k-fold cross-validation
    fold_accuracies = []  # To store accuracy for each fold
    for fold, (train_index, val_index) in enumerate(cross_validator.split(X, y_one_hot)):
        print(f"Training on fold {fold + 1}/{k_folds}")

        # Split data into training and validation sets for the current fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_one_hot[train_index], y_one_hot[val_index]

        # Initialize the ANN for each fold without batch normalization
        ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, 
                              loss='categorical_crossentropy', use_gpu=use_gpu)

        # Set up model checkpointing
        checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_loss', save_best_only=True, save_freq=5)

        # Callback functions
        lr_scheduler = LearningRateScheduler(initial_lr=0.01, strategy="cyclic")
        early_stop = EarlyStopping(patience=3)

        # Train the model and capture history
        ann.fit(epochs=1000, learning_rate=0.01, 
                lr_scheduler=lr_scheduler, 
                early_stop=early_stop, 
                X_val=X_val, 
                y_val=y_val, checkpoint=checkpoint)

        # Evaluate the model on the validation set
        y_pred_val = ann.predict(X_val)
        y_val_labels = np.argmax(y_val, axis=1)

        # Adjust prediction shape handling for accuracy calculation
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)  # Multi-class classification
        
        # Calculate and store the accuracy for this fold
        fold_accuracy = np.mean(y_pred_val_labels == y_val_labels)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy * 100:.2f}%")

    # Optionally plot training history of the last fold
    plot_utils = Plotting_Utils()  
    plot_utils.plot_training_history(ann.history)
