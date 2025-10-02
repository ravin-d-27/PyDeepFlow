import unittest
import numpy as np
from pydeepflow.metrics import precision_score, recall_score, f1_score, confusion_matrix


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([1, 0, 1, 1, 0, 1])
        self.y_pred = np.array([1, 1, 1, 0, 0, 1])

    def test_precision_score(self):
        # TP = 3, FP = 1 => Precision = 3 / 4 = 0.75
        self.assertAlmostEqual(precision_score(self.y_true, self.y_pred), 0.75)

    def test_recall_score(self):
        # TP = 3, FN = 1 => Recall = 3 / 4 = 0.75
        self.assertAlmostEqual(recall_score(self.y_true, self.y_pred), 0.75)

    def test_f1_score(self):
        precision = 0.75
        recall = 0.75
        f1 = 2 * (precision * recall) / (precision + recall)
        self.assertAlmostEqual(f1_score(self.y_true, self.y_pred), f1, places=6)

    def test_confusion_matrix(self):
        # TN = 1, FP = 1
        # FN = 1, TP = 3
        expected_matrix = np.array([[1, 1], [1, 3]])
        np.testing.assert_array_equal(
            confusion_matrix(self.y_true, self.y_pred, num_classes=2), expected_matrix
        )


if __name__ == "__main__":
    unittest.main()
