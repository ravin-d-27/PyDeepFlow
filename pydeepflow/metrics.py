import numpy as np

def precision_score(y_true, y_pred):
    """
    Calculates precision.
    """
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall_score(y_true, y_pred):
    """
    Calculates recall.
    """
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / (actual_positives + 1e-7)
    return recall

def f1_score(y_true, y_pred):
    """
    Calculates the F1-score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Computes the confusion matrix.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    return matrix