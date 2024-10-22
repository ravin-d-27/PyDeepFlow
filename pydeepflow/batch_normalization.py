import numpy as np

class BatchNormalization:
    def __init__(self, layer_size, epsilon=1e-5, momentum=0.9, device=np):
        self.epsilon = epsilon
        self.momentum = momentum
        self.device = device
        
        self.gamma = self.device.ones((1, layer_size))
        self.beta = self.device.zeros((1, layer_size))
        
        self.running_mean = self.device.zeros((1, layer_size))
        self.running_variance = self.device.ones((1, layer_size))
    
    def normalize(self, Z, training=True):
        if training:
            batch_mean = self.device.mean(Z, axis=0, keepdims=True)
            batch_variance = self.device.var(Z, axis=0, keepdims=True)
            
            Z_normalized = (Z - batch_mean) / self.device.sqrt(batch_variance + self.epsilon)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * batch_variance
        else:
            Z_normalized = (Z - self.running_mean) / self.device.sqrt(self.running_variance + self.epsilon)
        
        Z_scaled = self.gamma * Z_normalized + self.beta
        
        return Z_scaled
    
    def backprop(self, Z, dZ, learning_rate):
        dgamma = self.device.sum(dZ * Z, axis=0, keepdims=True)
        dbeta = self.device.sum(dZ, axis=0, keepdims=True)
        
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dZ
