import numpy as np

def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

def binary_crossentropy_derivative(y_true, y_pred):
    return -(y_true / (y_pred + 1e-8)) + (1 - y_true) / (1 - y_pred + 1e-8)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def get_loss_function(loss_name):
    if loss_name == 'binary_crossentropy':
        return binary_crossentropy
    elif loss_name == 'mse':
        return mse
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

def get_loss_derivative(loss_name):
    if loss_name == 'binary_crossentropy':
        return binary_crossentropy_derivative
    elif loss_name == 'mse':
        return mse_derivative
    else:
        raise ValueError(f"Unsupported loss derivative: {loss_name}")
