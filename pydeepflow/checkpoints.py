import os
import numpy as np

class ModelCheckpoint:
    def __init__(self, save_dir, monitor="val_loss", save_best_only=True, save_freq=1):
        """
        Args:
            save_dir (str): Directory where the model weights will be saved.
            monitor (str): Metric to monitor for saving (e.g., 'val_loss', 'val_accuracy').
            save_best_only (bool): Save only the best weights based on the monitored metric.
            save_freq (int): Frequency of saving checkpoints (in epochs).
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.best_metric = np.inf if "loss" in monitor else -np.inf
        self.best_val_loss = float('inf')  # Initial best validation loss

    def save_weights(self, epoch, weights, biases, val_loss):
        
        """ Saves weights and biases to the specified directory.
        Args:
            epoch (int): The current epoch number.
            weights (numpy.ndarray): The weights of the model to be saved.
            biases (numpy.ndarray): The biases of the model to be saved.
            val_loss (float): The validation loss value to be logged.
        Returns:
            None
        """
        
        # Create the directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # Create the directory

        checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch}.npz"
        
        # Prepare data to save
        data = {}
        for i, (w, b) in enumerate(zip(weights, biases)):
            data[f'weights_layer_{i}'] = w  
            data[f'biases_layer_{i}'] = b    
        
        # Save as .npz file
        np.savez(checkpoint_path, **data)


    def should_save(self, epoch, metric):
        """
        Determines whether to save based on the best metric or save frequency.

        Args:
            epoch (int): The current epoch number.
            metric (float): The current value of the monitored metric (e.g., loss or accuracy).

        Returns:
            bool: True if the model should be saved, False otherwise.
        """
        if self.save_freq and epoch % self.save_freq == 0:
            if self.save_best_only:
                if ("loss" in self.monitor and metric < self.best_metric) or \
                   ("accuracy" in self.monitor and metric > self.best_metric):
                    self.best_metric = metric
                    return True
            else:
                return True
        return False

    