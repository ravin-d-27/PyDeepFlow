import numpy as np
import matplotlib.pyplot as plt  
from .activations import activation, activation_derivative  
from .losses import get_loss_function, get_loss_derivative  
from .device import Device                                  
from .regularization import Regularization
from .checkpoints import ModelCheckpoint
from tqdm import tqdm
import time

class Multi_Layer_ANN:
    """
    A Multi-Layer Artificial Neural Network (ANN) class for binary and multi-class classification tasks.
    """
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy',
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0):
        """
        Initializes the ANN model with the provided architecture and configurations.
        """
        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)

        # Determine the network architecture based on the classification task (binary or multi-class)
        if Y_train.ndim == 1 or Y_train.shape[1] == 1:
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else: 
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'

        self.activations = activations
        self.weights = []
        self.biases = []

        # Setup loss function
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # Move training data to the device (GPU or CPU)
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)

        # Initialize weights and biases with He initialization for better convergence
        for i in range(len(self.layers) - 1):
            weight_matrix = self.device.random().randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias_vector = self.device.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

        # Initialize training attribute
        self.training = False

        # Store metrics for plotting
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}


    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.
        """
        activations = [X]
        Z_values = []

        # Forward pass through hidden layers with dropout
        for i in range(len(self.weights) - 1):
            Z = self.device.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            A = activation(Z, self.activations[i], self.device)
            A = self.regularization.apply_dropout(A, training=self.training) 
            activations.append(A)

        # Forward pass through the output layer
        Z_output = self.device.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A_output = activation(Z_output, self.output_activation, self.device)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate, clip_value = None):
        """
        Performs backpropagation through the network to compute weight updates.
        """
        # Calculate the error in the output layer
        output_error = y - activations[-1]
        d_output = output_error * activation_derivative(activations[-1], self.output_activation, self.device)

        # Backpropagate through the network
        deltas = [d_output]
        for i in reversed(range(len(self.weights) - 1)):
            error = self.device.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * activation_derivative(activations[i + 1], self.activations[i], self.device)
            deltas.append(delta)
        
        deltas.reverse()

        #perform gradient clipping if clipping threshold is passed
        if clip_value:
            deltas_clipped = []
            for grad in deltas:
                # Compute the norm (magnitude) of the gradient
                grad_norm = self.device.norm(grad)
                # If the gradient norm exceeds the clip value, scale it down
                if grad_norm > clip_value:
                    clipped_grad = grad * (clip_value / grad_norm)
                else:
                    clipped_grad = grad      
                deltas_clipped.append(clipped_grad)
            deltas = deltas_clipped

        # Update weights and biases with L2 regularization
        for i in range(len(self.weights)):
            self.weights[i] += self.device.dot(activations[i].T, deltas[i]) * learning_rate
            self.biases[i] += self.device.sum(deltas[i], axis=0, keepdims=True) * learning_rate
        # Apply L2 regularization to the weights
        self.weights[i] -= learning_rate * self.regularization.apply_l2_regularization(self.weights[i], learning_rate, X.shape)

    def fit(self, epochs, learning_rate=0.01, lr_scheduler=None, early_stop = None, X_val=None, y_val=None, checkpoint=None, verbose=False, clipping_threshold = None):   
        """
        Trains the model for a given number of epochs with an optional learning rate scheduler.
        :param verbose (bool): toggle verbosity during training
        """
        if early_stop:
            assert X_val is not None and y_val is not None, "Validation set is required for early stopping"

        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green',disable=verbose):
            start_time = time.time()

            # Adjust the learning rate using the scheduler if provided
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate
            
            # Forward and Backpropagation
            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr,clip_value=clipping_threshold)

            self.training = False

            # Compute training loss and accuracy
            train_loss = self.loss_func(self.y_train, activations[-1], self.device)
            train_accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            # Validation step
            val_loss = val_accuracy = None
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(self.device.array(X_val))
                val_loss = self.loss_func(self.device.array(y_val), val_activations[-1], self.device)
                val_accuracy = np.mean((val_activations[-1] >= 0.5).astype(int) == y_val) if self.output_activation == 'sigmoid' else np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

            # Store training history for plotting
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            # Checkpoint saving logic
            if checkpoint is not None and X_val is not None:
                if checkpoint.should_save(epoch, val_loss):
                    checkpoint.save_weights(epoch, self.weights, self.biases, val_loss)

            # Log training progress
            if verbose:
                epoch_time = time.time() - start_time
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}% "
                        f"Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy * 100:.2f}% Time: {epoch_time:.2f}s "
                        f"Learning Rate: {current_lr:.6f}")
                
            # Early stopping 
            early_stop(val_loss)
            if early_stop.early_stop:
                print('\n',"#"*150,'\n\n', "early stop at - "
                      f"Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}% "
                      f"Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy * 100:.2f}% "
                      f"Learning Rate: {current_lr:.6f}",'\n\n', "#"*150)
                break
                
        print("Training Completed!")
        
        
    def load_weights(self, checkpoint_path):
        """
        Loads model weights from a checkpoint.
        """
        data = np.load(checkpoint_path)
        for i in range(len(self.weights)):
            try:
                self.weights[i] = data[f'weights_layer_{i}']
                self.biases[i] = data[f'biases_layer_{i}']
            except KeyError as e:
                print(f"Key error: {e}. Please check the checkpoint file.")

    def predict(self, X):
        """
        Makes predictions based on input data using the trained model.
        """
        activations, _ = self.forward_propagation(self.device.array(X))
        if self.output_activation == 'sigmoid':
            return self.device.asnumpy((activations[-1] >= 0.5).astype(int)).flatten()
        elif self.output_activation == 'softmax':
            return self.device.asnumpy(np.argmax(activations[-1], axis=1))
        
    def predict_prob(self, X):
        """
        Predicts the probability distribution for the input data.
        """
        activations, _ = self.forward_propagation(self.device.array(X))
        return self.device.asnumpy(activations[-1])


class Plotting_Utils:
    """
    Utility class for plotting training and validation metrics.
    """
    def plot_training_history(self, history, metrics=('loss', 'accuracy'), figure='history.png'):
        """
        Plots the training and validation loss/accuracy over epochs.
        Parameters:
            history (dict): A dictionary containing training history with keys 'train_loss', 'val_loss', 
                            'train_accuracy', and 'val_accuracy'.
            metrics (tuple): The metrics to plot ('loss' or 'accuracy').
        """
        epochs = len(history['train_loss'])
        fig, ax = plt.subplots(1, len(metrics), figsize=(12, 5))

        if 'loss' in metrics:
            ax[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='blue')
            if 'val_loss' in history:
                ax[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='orange')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Training vs Validation Loss')
            ax[0].legend()

        if 'accuracy' in metrics:
            ax[1].plot(range(1, epochs + 1), history['train_accuracy'], label='Train Accuracy', color='green')
            if 'val_accuracy' in history:
                ax[1].plot(range(1, epochs + 1), history['val_accuracy'], label='Val Accuracy', color='red')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('Training vs Validation Accuracy')
            ax[1].legend()

        plt.tight_layout()
        plt.savefig(figure)
        plt.show()
