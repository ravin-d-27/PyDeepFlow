import numpy as np

class BatchNormalization:
    def __init__(self, layer_size, epsilon=1e-5, momentum=0.9, device=np):
        """
        Initializes the BatchNormalization class.

        Parameters:
        - layer_size: The number of units in the layer where batch normalization is applied.
        - epsilon: A small constant added to the denominator to prevent division by zero.
        - momentum: Momentum for updating the running mean and variance.
        - device: NumPy or CuPy (for GPU support). Default is NumPy.
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.device = device
        
        # Initialize gamma (scale) and beta (shift) parameters
        self.gamma = self.device.ones((1, layer_size))
        self.beta = self.device.zeros((1, layer_size))
        
        # Initialize running mean and variance (used during inference)
        self.running_mean = self.device.zeros((1, layer_size))
        self.running_variance = self.device.ones((1, layer_size))
    
    def normalize(self, Z, training=True):
        """
        Applies batch normalization to the input.

        Parameters:
        - Z: The input to normalize (output from the previous layer before activation).
        - training: Boolean flag to indicate whether it's in training or inference mode.

        Returns:
        - Z_scaled: The normalized and scaled input.
        """
        if training:
            # Compute batch mean and variance
            batch_mean = self.device.mean(Z, axis=0, keepdims=True)
            batch_variance = self.device.var(Z, axis=0, keepdims=True)
            
            # Normalize the batch
            Z_normalized = (Z - batch_mean) / self.device.sqrt(batch_variance + self.epsilon)
            
            # Update running mean and variance (for inference)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * batch_variance
        else:
            # During inference, use running mean and variance
            Z_normalized = (Z - self.running_mean) / self.device.sqrt(self.running_variance + self.epsilon)
        
        # Scale and shift the normalized values
        Z_scaled = self.gamma * Z_normalized + self.beta
        
        return Z_scaled
    
    def backprop(self, Z, dZ, learning_rate):
        """
        Performs the backpropagation step for batch normalization.

        Parameters:
        - Z: The input to the batch normalization layer (before normalization).
        - dZ: The gradient of the loss with respect to the output of this layer.
        - learning_rate: The learning rate for updating gamma and beta.

        Returns:
        - dZ: The modified gradient for further backpropagation.
        """
        # Compute gradients for gamma and beta
        dgamma = self.device.sum(dZ * Z, axis=0, keepdims=True)
        dbeta = self.device.sum(dZ, axis=0, keepdims=True)
        
        # Update gamma and beta using gradient descent
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dZ  # Pass on the gradient to the previous layer
