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
        os.makedirs(save_dir, exist_ok=True)

    def save_weights(self, epoch, weights, biases, metric):
        
        """ Saves weights and biases to the specified directory.
        Args:
            epoch (int): The current epoch number.
            weights (numpy.ndarray): The weights of the model to be saved.
            biases (numpy.ndarray): The biases of the model to be saved.
            metric (float): The performance metric value to be logged.
        Returns:
            None
        """
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.npz")
        np.savez(checkpoint_path, weights=weights, biases=biases)
        print(f"Saved checkpoint at epoch {epoch} with {self.monitor}: {metric:.4f}")

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

    def load_weights(self, model, checkpoint_path):
        """
        Loads weights and biases from a checkpoint file.

        Parameters:
        model (object): The model object to which the weights and biases will be loaded.
        checkpoint_path (str): The file path to the checkpoint file containing the weights and biases.

        Returns:
            None
        """
        
        data = np.load(checkpoint_path)
        model.weights = data['weights']
        model.biases = data['biases']
        print(f"Loaded weights from {checkpoint_path}")
