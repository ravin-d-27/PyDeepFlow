# losses.py

# Loss functions using the device abstraction for GPU/CPU support
def binary_crossentropy(y_true, y_pred, device):
    return -device.mean(y_true * device.log(y_pred + 1e-8) + (1 - y_true) * device.log(1 - y_pred + 1e-8))

def binary_crossentropy_derivative(y_true, y_pred, device):
    return -(y_true / (y_pred + 1e-8)) + (1 - y_true) / (1 - y_pred + 1e-8)

def mse(y_true, y_pred, device):
    return device.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred, device):
    return 2 * (y_pred - y_true) / y_true.size

def categorical_crossentropy(y_true, y_pred, device):
    return -device.sum(y_true * device.log(y_pred + 1e-8)) / y_true.shape[0]

def categorical_crossentropy_derivative(y_true, y_pred, device):
    return -y_true / (y_pred + 1e-8)

def hinge_loss(y_true, y_pred, device):
    return device.mean(device.maximum(0, 1 - y_true * y_pred))

def hinge_loss_derivative(y_true, y_pred, device):
    return device.where(y_true * y_pred < 1, -y_true, 0)

def huber_loss(y_true, y_pred, device, delta=1.0):
    error = y_true - y_pred
    is_small_error = device.abs(error) <= delta
    squared_loss = 0.5 * device.square(error)
    linear_loss = delta * (device.abs(error) - 0.5 * delta)
    return device.mean(device.where(is_small_error, squared_loss, linear_loss))

def huber_loss_derivative(y_true, y_pred, device, delta=1.0):
    error = y_pred - y_true
    is_small_error = device.abs(error) <= delta
    return device.where(is_small_error, error, delta * device.sign(error))

# Get the appropriate loss function
def get_loss_function(loss_name):
    if loss_name == 'binary_crossentropy':
        return binary_crossentropy
    elif loss_name == 'mse':
        return mse
    elif loss_name == 'categorical_crossentropy':
        return categorical_crossentropy
    elif loss_name == 'hinge':
        return hinge_loss
    elif loss_name == 'huber':
        return huber_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

# Get the appropriate loss derivative function
def get_loss_derivative(loss_name):
    if loss_name == 'binary_crossentropy':
        return binary_crossentropy_derivative
    elif loss_name == 'mse':
        return mse_derivative
    elif loss_name == 'categorical_crossentropy':
        return categorical_crossentropy_derivative
    elif loss_name == 'hinge':
        return hinge_loss_derivative
    elif loss_name == 'huber':
        return huber_loss_derivative
    else:
        raise ValueError(f"Unsupported loss derivative: {loss_name}")
