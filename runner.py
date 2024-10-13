import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.early_stopping import EarlyStopping
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.learning_rate_scheduler import LearningRateScheduler
from pydeepflow.model import Plotting_Utils  # Correct import

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

    # Initialize the ANN with softmax output for multiclass classification
    ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='categorical_crossentropy', use_gpu=use_gpu)

    # Set up model checkpointing
    checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_loss', save_best_only=True, save_freq=5)

    # CallBack Functions 
    lr_scheduler = LearningRateScheduler(initial_lr=0.01, strategy="cyclic")
    early_stop = EarlyStopping(patience=3)

    # Train the model and capture history
    # increased num epochc to trigger early stopping
    ann.fit(epochs=10000, learning_rate=0.01, lr_scheduler=lr_scheduler,early_stop=early_stop, X_val=X_train, y_val=y_train, checkpoint=checkpoint)

    # Use Plotting_Utils to plot accuracy and loss
    plot_utils = Plotting_Utils()  
    plot_utils.plot_training_history(ann.history)  

    # Make predictions on the test set
    y_pred = ann.predict(X_test)

    # Convert one-hot encoded test labels back to integers
    y_test_labels = np.argmax(y_test, axis=1)

    # # Check the shape of y_pred
    # if y_pred.ndim == 1:  
    #     y_pred_labels = (y_pred >= 0.5).astype(int)  
    # elif y_pred.ndim == 2:  
    #     y_pred_labels = np.argmax(y_pred, axis=1)  

    # Calculate the accuracy of the model
    
    #accuracy = np.mean(y_pred_labels == y_test_labels)
    accuracy = np.mean(y_pred == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Load the best weights from the checkpoint file
    best_checkpoint_file = './checkpoints/checkpoint_epoch_995.npz'  
    
    data = np.load(best_checkpoint_file)
    print(data.files)  

    ann.load_weights(best_checkpoint_file)

    # Make predictions using the loaded weights
    y_pred_loaded = ann.predict(X_test)

    # Calculate accuracy of predictions using loaded weights
    # if y_pred_loaded.ndim == 1:  
    #     y_pred_loaded_labels = (y_pred_loaded >= 0.5).astype(int)
    # elif y_pred_loaded.ndim == 2:  
    #     y_pred_loaded_labels = np.argmax(y_pred_loaded, axis=1)

    accuracy_loaded = np.mean(y_pred_loaded == y_test_labels)
    
    print(f"Loaded Model Test Accuracy: {accuracy_loaded * 100:.2f}%")
