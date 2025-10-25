import numpy as np

def accuracy(y_true, y_pred, total_samples):
    """
    Calculates the accuracy for binary classsification.
    
    Accuracy is the ratio of correct predictions to the total predictions or observations.
    
    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels, as returned by a classifier.\
        total_samples (int): Total samples or length of the dataset used.

    Returns:
        float: The accuracy score.
    """
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    accuracy = (true_positives + true_negatives) / total_samples
    return accuracy

def precision_score(y_true, y_pred):
    """
    Calculates the precision for binary classification.

    Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
    High precision relates to a low false positive rate.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels, as returned by a classifier.

    Returns:
        float: The precision score.
    """
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall_score(y_true, y_pred):
    """
    Calculates the recall for binary classification.

    Recall is the ratio of correctly predicted positive observations to all observations in the actual class.
    High recall relates to a low false negative rate.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels, as returned by a classifier.

    Returns:
        float: The recall score.
    """
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / (actual_positives + 1e-7)
    return recall

def f1_score(y_true, y_pred):
    """
    Calculates the F1-score, which is the harmonic mean of precision and recall.

    This score takes both false positives and false negatives into account. It is a good way to show that a
    classifier has a good value for both recall and precision.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels, as returned by a classifier.

    Returns:
        float: The F1-score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.

    The matrix rows represent the true classes, while columns represent the predicted classes.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels, as returned by a classifier.
        num_classes (int): The total number of classes.

    Returns:
        np.ndarray: A confusion matrix of shape (num_classes, num_classes).
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    return matrix

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|y_true - y_pred|

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    float
        The MAE score.
    """
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).

    MSE = (1/n) * Σ(y_true - y_pred)^2

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    float
        The MSE score.
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Calculates the R-squared (coefficient of determination) regression score.

    R^2 = 1 - (Σ(y_true - y_pred)^2) / (Σ(y_true - y_mean)^2)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    float
        The R^2 score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).

    RMSE = sqrt((1/n) * Σ(y_true - y_pred)^2)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    
    Returns
    -------
    float
        The RMSE score.
    """
    return ((np.array(y_true) - np.array(y_pred)) ** 2).mean() ** 0.5
    
