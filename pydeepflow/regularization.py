import numpy as np 
from .device import Device

class Regularization:
    def __init__(self, l2_lambda=0.0, dropout_rate=0.0):
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.device = Device()

    def apply_l2_regularization(self, weights, learning_rate, X_shape):
        for i in range(len(weights)):
            weights[i] -= (self.l2_lambda * weights[i]) / X_shape[0]
        return weights

    def apply_dropout(self, A, training=True):
        if training and self.dropout_rate > 0:
            dropout_mask = self.device.random().rand(*A.shape) > self.dropout_rate
            A *= dropout_mask
        else:
            A *= (1 - self.dropout_rate)
        return A