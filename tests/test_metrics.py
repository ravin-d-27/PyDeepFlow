import unittest
import numpy as np
from pydeepflow.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error  

)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Setup for classification metrics
        self.y_true_clf = np.array([1, 0, 1, 1, 0, 1])
        self.y_pred_clf = np.array([1, 1, 1, 0, 0, 1])

        # Setup for regression metrics
        self.y_true_reg = np.array([3, -0.5, 2, 7])
        self.y_pred_reg = np.array([2.5, 0.0, 2, 8])

    def test_precision_score(self):
        # TP = 3, FP = 1 => Precision = 3 / 4 = 0.75
        self.assertAlmostEqual(precision_score(self.y_true_clf, self.y_pred_clf), 0.75)

    def test_recall_score(self):
        # TP = 3, FN = 1 => Recall = 3 / 4 = 0.75
        self.assertAlmostEqual(recall_score(self.y_true_clf, self.y_pred_clf), 0.75)

    def test_f1_score(self):
        precision = 0.75
        recall = 0.75
        f1 = 2 * (precision * recall) / (precision + recall)
        self.assertAlmostEqual(f1_score(self.y_true_clf, self.y_pred_clf), f1, places=6)

    def test_confusion_matrix(self):
        # TN = 1, FP = 1
        # FN = 1, TP = 3
        expected_matrix = np.array([[1, 1], [1, 3]])
        np.testing.assert_array_equal(confusion_matrix(self.y_true_clf, self.y_pred_clf, num_classes=2), expected_matrix)

    def test_mean_absolute_error(self):
        # MAE = ( |3-2.5| + |-0.5-0.0| + |2-2| + |7-8| ) / 4
        #     = ( 0.5 + 0.5 + 0 + 1 ) / 4 = 2 / 4 = 0.5
        self.assertAlmostEqual(mean_absolute_error(self.y_true_reg, self.y_pred_reg), 0.5)

    def test_mean_squared_error(self):
        # MSE = ( (3-2.5)**2 + (-0.5-0.0)**2 + (2-2)**2 + (7-8)**2 ) / 4
        #     = ( 0.25 + 0.25 + 0 + 1 ) / 4 = 1.5 / 4 = 0.375
        self.assertAlmostEqual(mean_squared_error(self.y_true_reg, self.y_pred_reg), 0.375)

    def test_r2_score(self):
        # ss_res = 1.5 (from MSE calculation)
        # y_mean = (3 - 0.5 + 2 + 7) / 4 = 11.5 / 4 = 2.875
        # ss_tot = (3-2.875)**2 + (-0.5-2.875)**2 + (2-2.875)**2 + (7-2.875)**2
        #        = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
        # R^2 = 1 - (1.5 / 29.1875) = 1 - 0.051389... approx 0.9486
        self.assertAlmostEqual(r2_score(self.y_true_reg, self.y_pred_reg), 0.94861051, places=5)


    def test_root_mean_squared_error(self):
        # Step 1: Differences
        # (3 - 2.5) = 0.5
        # (-0.5 - 0.0) = -0.5
        # (2 - 2) = 0
        # (7 - 8) = -1
        #
        # Step 2: Squared differences
        # [0.5², (-0.5)², 0², (-1)²] = [0.25, 0.25, 0, 1]
        #
        # Step 3: Mean Squared Error (MSE)
        # (0.25 + 0.25 + 0 + 1) / 4 = 0.375
        #
        # Step 4: Root Mean Squared Error (RMSE)
        # sqrt(0.375) = 0.6123724356957945
        self.assertAlmostEqual(root_mean_squared_error(self.y_true_reg, self.y_pred_reg), 0.6123724356957945, places=6)

if __name__ == '__main__':
    unittest.main()