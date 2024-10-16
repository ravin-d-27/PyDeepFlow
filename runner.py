# runner.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.early_stopping import EarlyStopping
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.learning_rate_scheduler import LearningRateScheduler
from pydeepflow.model import Plotting_Utils  
from pydeepflow.cross_validator import CrossValidator  

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

    # Convert labels to one-hot encoding (for multiclass classification)
    y_one_hot = np.eye(len(np.unique(y)))[y]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Ask the user whether to use GPU
    use_gpu_input = input("Use GPU? (y/n): ").strip().lower()
    use_gpu = True if use_gpu_input == 'y' else False

    # Define the architecture of the network
    hidden_layers = [5, 5]  
    activations = ['relu', 'relu']  

    # Initialize the CrossValidator
    k_folds = 5  # Set the number of folds for cross-validation
    cross_validator = CrossValidator(n_splits=k_folds)  # Adjusted to use n_splits

    # Perform k-fold cross-validation
    fold_accuracies = []  # To store accuracy for each fold
    for fold, (train_index, val_index) in enumerate(cross_validator.split(X, y_one_hot)):
        print(f"Training on fold {fold + 1}/{k_folds}")

        # Split data into training and validation sets for the current fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_one_hot[train_index], y_one_hot[val_index]

        # Initialize the ANN for each fold
        ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='categorical_crossentropy', use_gpu=use_gpu)

        # Set up model checkpointing
        checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_loss', save_best_only=True, save_freq=5)

        # CallBack Functions 
        lr_scheduler = LearningRateScheduler(initial_lr=0.01, strategy="cyclic")
        early_stop = EarlyStopping(patience=3)

        # Train the model and capture history
        ann.fit(epochs=10000, learning_rate=0.01, lr_scheduler=lr_scheduler, early_stop=early_stop, 
                X_val=X_val, y_val=y_val, checkpoint=checkpoint)

        # Evaluate the model on the validation set
        y_pred_val = ann.predict(X_val)
        y_val_labels = np.argmax(y_val, axis=1)

        # Adjust prediction shape handling for accuracy calculation
        if y_pred_val.ndim == 2:  
            y_pred_val_labels = np.argmax(y_pred_val, axis=1)  # Multi-class classification
        else:
            y_pred_val_labels = (y_pred_val >= 0.5).astype(int)  # Binary classification (if applicable)

        # Calculate and store the accuracy for this fold
        fold_accuracy = np.mean(y_pred_val_labels == y_val_labels)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy * 100:.2f}%")

    # Print the average accuracy across all folds
    average_accuracy = np.mean(fold_accuracies)
    print(f"Average Accuracy across {k_folds} folds: {average_accuracy * 100:.2f}%")

    # Optionally plot training history of the last fold (if you need)
    plot_utils = Plotting_Utils()  
    plot_utils.plot_training_history(ann.history)  

    # Make predictions on the test set (optional)
    # Assuming you have a separate test set prepared
    # y_pred_test = ann.predict(X_test)

    # Calculate and print accuracy on the test set as needed
