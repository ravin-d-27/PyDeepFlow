# losses.py

# Loss functions using the device abstraction for GPU/CPU support


def binary_crossentropy(y_true, y_pred, device):
    """
    Computes the binary crossentropy loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray or cp.ndarray
        Predicted probabilities for the positive class.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    float
        The binary crossentropy loss.
    """
    return -device.mean(
        y_true * device.log(y_pred + 1e-8)
        + (1 - y_true) * device.log(1 - y_pred + 1e-8)
    )


def binary_crossentropy_derivative(y_true, y_pred, device):
    """
    Computes the derivative of the binary crossentropy loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray or cp.ndarray
        Predicted probabilities for the positive class.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    np.ndarray or cp.ndarray
        The derivative of the binary crossentropy loss with respect to predictions.
    """
    return -(y_true / (y_pred + 1e-8)) + (1 - y_true) / (1 - y_pred + 1e-8)


def mse(y_true, y_pred, device):
    """
    Computes the Mean Squared Error (MSE) loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth values.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    float
        The Mean Squared Error loss.
    """
    return device.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred, device):
    """
    Computes the derivative of the Mean Squared Error (MSE) loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth values.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    np.ndarray or cp.ndarray
        The derivative of the Mean Squared Error loss with respect to predictions.
    """
    return 2 * (y_pred - y_true) / y_true.size


def categorical_crossentropy(y_true, y_pred, device):
    """
    Computes the categorical crossentropy loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth labels in one-hot encoded format.
    y_pred : np.ndarray or cp.ndarray
        Predicted probabilities for each class.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    float
        The categorical crossentropy loss.
    """
    return -device.sum(y_true * device.log(y_pred + 1e-8)) / y_true.shape[0]


def categorical_crossentropy_derivative(y_true, y_pred, device):
    """
    Computes the derivative of the categorical crossentropy loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth labels in one-hot encoded format.
    y_pred : np.ndarray or cp.ndarray
        Predicted probabilities for each class.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    np.ndarray or cp.ndarray
        The derivative of the categorical crossentropy loss with respect to predictions.
    """
    return -y_true / (y_pred + 1e-8)


def hinge_loss(y_true, y_pred, device):
    """
    Computes the hinge loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth labels, should be either -1 or 1.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    float
        The hinge loss.
    """
    return device.mean(device.maximum(0, 1 - y_true * y_pred))


def hinge_loss_derivative(y_true, y_pred, device):
    """
    Computes the derivative of the hinge loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth labels, should be either -1 or 1.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.

    Returns:
    --------
    np.ndarray or cp.ndarray
        The derivative of the hinge loss with respect to predictions.
    """
    return device.where(y_true * y_pred < 1, -y_true, 0)


def huber_loss(y_true, y_pred, device, delta=1.0):
    """
    Computes the Huber loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth values.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.
    delta : float, optional (default=1.0)
        The threshold for defining small and large errors.

    Returns:
    --------
    float
        The Huber loss.
    """
    error = y_true - y_pred
    is_small_error = device.abs(error) <= delta
    squared_loss = 0.5 * device.square(error)
    linear_loss = delta * (device.abs(error) - 0.5 * delta)
    return device.mean(device.where(is_small_error, squared_loss, linear_loss))


def huber_loss_derivative(y_true, y_pred, device, delta=1.0):
    """
    Computes the derivative of the Huber loss.

    Parameters:
    -----------
    y_true : np.ndarray or cp.ndarray
        Ground truth values.
    y_pred : np.ndarray or cp.ndarray
        Predicted values.
    device : Device
        The device instance (CPU or GPU) to perform calculations.
    delta : float, optional (default=1.0)
        The threshold for defining small and large errors.

    Returns:
    --------
    np.ndarray or cp.ndarray
        The derivative of the Huber loss with respect to predictions.
    """
    error = y_pred - y_true
    is_small_error = device.abs(error) <= delta
    return device.where(is_small_error, error, delta * device.sign(error))


# Get the appropriate loss function
def get_loss_function(loss_name):
    """
    Retrieves the specified loss function by name.

    Parameters:
    -----------
    loss_name : str
        The name of the loss function.

    Returns:
    --------
    function
        The corresponding loss function.

    Raises:
    -------
    ValueError
        If the specified loss function is unsupported.
    """
    if loss_name == "binary_crossentropy":
        return binary_crossentropy
    elif loss_name == "mse":
        return mse
    elif loss_name == "categorical_crossentropy":
        return categorical_crossentropy
    elif loss_name == "hinge":
        return hinge_loss
    elif loss_name == "huber":
        return huber_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


# Get the appropriate loss derivative function
def get_loss_derivative(loss_name):
    """
    Retrieves the specified loss derivative function by name.

    Parameters:
    -----------
    loss_name : str
        The name of the loss function.

    Returns:
    --------
    function
        The corresponding loss derivative function.

    Raises:
    -------
    ValueError
        If the specified loss derivative function is unsupported.
    """
    if loss_name == "binary_crossentropy":
        return binary_crossentropy_derivative
    elif loss_name == "mse":
        return mse_derivative
    elif loss_name == "categorical_crossentropy":
        return categorical_crossentropy_derivative
    elif loss_name == "hinge":
        return hinge_loss_derivative
    elif loss_name == "huber":
        return huber_loss_derivative
    else:
        raise ValueError(f"Unsupported loss derivative: {loss_name}")
