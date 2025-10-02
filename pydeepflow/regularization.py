import numpy as np
from .device import Device


class Regularization:
    """
    A class to apply L2 regularization and dropout to a neural network.

    This class provides methods to incorporate L2 regularization and dropout, which are
    techniques used to prevent overfitting in neural networks by penalizing large weights
    and randomly setting neuron activations to zero during training.

    Attributes:
        l2_lambda (float): The regularization parameter for L2 regularization.
        dropout_rate (float): The probability of setting a neuron's output to zero during dropout.
        device (Device): The device (CPU or GPU) on which to perform computations.
    """

    def __init__(self, l2_lambda=0.0, dropout_rate=0.0):
        """
        Initializes the Regularization instance.

        Args:
            l2_lambda (float, optional): The L2 regularization strength. Defaults to 0.0.
            dropout_rate (float, optional): The dropout rate, a value between 0 and 1. Defaults to 0.0.
        """
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.device = Device()

    def apply_l2_regularization(self, weights, learning_rate, X_shape):
        """
        Applies L2 regularization to the model's weights.

        This method updates the weights by subtracting a term proportional to the
        L2 regularization strength, effectively penalizing large weights.

        Args:
            weights (list): A list of weight matrices for each layer of the network.
            learning_rate (float): The learning rate used in the training process.
            X_shape (tuple): The shape of the input data, used for normalization.

        Returns:
            list: The updated list of weight matrices after applying L2 regularization.
        """
        for i in range(len(weights)):
            weights[i] -= (self.l2_lambda * weights[i]) / X_shape[0]
        return weights

    def apply_dropout(self, A, training=True):
        """
        Applies dropout to the activations of a layer.

        During training, this method randomly sets a fraction of the input units to 0
        at each update, which helps prevent overfitting. During inference, it scales
        the activations by the dropout rate to account for the increased number of active units.

        Args:
            A (np.ndarray): The activation matrix to apply dropout to.
            training (bool, optional): A flag indicating whether the model is in training mode.
                                       Defaults to True.

        Returns:
            np.ndarray: The activation matrix after applying dropout.
        """
        if training and self.dropout_rate > 0:
            dropout_mask = self.device.random().rand(*A.shape) > self.dropout_rate
            A *= dropout_mask
        else:
            A *= 1 - self.dropout_rate
        return A
