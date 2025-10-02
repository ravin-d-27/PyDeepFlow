import numpy as np


class BatchNormalization:
    """
    A class that implements Batch Normalization for a layer in a neural network.

    Batch Normalization helps stabilize the learning process and accelerate training
    by normalizing the inputs of each layer. This class can be used during training
    and inference.
    """

    def __init__(self, layer_size, epsilon=1e-5, momentum=0.9, device=np):
        """
        Initializes the BatchNormalization object.

        Parameters:
            layer_size (int): The size of the layer to which batch normalization is applied.
            epsilon (float): A small constant added to the variance for numerical stability.
            momentum (float): The momentum for updating the running mean and variance.
            device (module): The device module (e.g., numpy) to perform calculations on.
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.device = device

        self.gamma = self.device.ones((1, layer_size))
        self.beta = self.device.zeros((1, layer_size))

        self.running_mean = self.device.zeros((1, layer_size))
        self.running_variance = self.device.ones((1, layer_size))

    def normalize(self, Z, training=True):
        """
        Normalizes the input data Z.

        During training, it computes the batch mean and variance, and updates the
        running mean and variance. During inference, it uses the running statistics.

        Parameters:
            Z (ndarray): The input data of shape (batch_size, layer_size) to normalize.
            training (bool): A flag indicating whether the model is in training mode.
                             If True, updates running statistics; otherwise uses them.

        Returns:
            ndarray: The normalized and scaled output data.
        """
        if training:
            batch_mean = self.device.mean(Z, axis=0, keepdims=True)
            batch_variance = self.device.var(Z, axis=0, keepdims=True)

            Z_normalized = (Z - batch_mean) / self.device.sqrt(
                batch_variance + self.epsilon
            )

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_variance = (
                self.momentum * self.running_variance
                + (1 - self.momentum) * batch_variance
            )
        else:
            Z_normalized = (Z - self.running_mean) / self.device.sqrt(
                self.running_variance + self.epsilon
            )

        Z_scaled = self.gamma * Z_normalized + self.beta

        return Z_scaled

    def backprop(self, Z, dZ, learning_rate):
        """
        Computes the gradients for gamma and beta during backpropagation
        and updates their values.

        Parameters:
            Z (ndarray): The input data used for normalization, of shape (batch_size, layer_size).
            dZ (ndarray): The gradient of the loss with respect to the output of the layer,
                          of shape (batch_size, layer_size).
            learning_rate (float): The learning rate for updating gamma and beta.

        Returns:
            ndarray: The gradient of the loss with respect to the input data Z.
        """
        dgamma = self.device.sum(dZ * Z, axis=0, keepdims=True)
        dbeta = self.device.sum(dZ, axis=0, keepdims=True)

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dZ
