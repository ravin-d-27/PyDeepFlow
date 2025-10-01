import numpy as np

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