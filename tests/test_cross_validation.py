import unittest
import numpy as np
from pydeepflow.cross_validator import CrossValidator

class TestCrossValidator(unittest.TestCase):
    def setUp(self):
        self.n_splits = 5
        self.cross_validator = CrossValidator(n_splits=self.n_splits)
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.y = np.array([0, 1, 0, 1, 0])

    def test_initialization(self):
        """Test that CrossValidator initializes with correct number of splits."""
        self.assertEqual(self.cross_validator.n_splits, self.n_splits)

    def test_split(self):
        """Test that split method generates correct number of folds and indices."""
        splits = list(self.cross_validator.split(self.X, self.y))
        self.assertEqual(len(splits), self.n_splits)

        # Check that each split has correct train and validation sizes
        for train_index, val_index in splits:
            self.assertEqual(len(train_index) + len(val_index), len(self.X))
            self.assertEqual(len(val_index), 1)  # 5 samples, 5 folds -> 1 sample per validation
            self.assertEqual(len(train_index), 4)  # 5 samples, 5 folds -> 4 samples per training
            # Ensure no overlap between train and validation indices
            self.assertTrue(len(np.intersect1d(train_index, val_index)) == 0)

    def test_metrics_accuracy(self):
        """Test accuracy metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        metrics = ['accuracy']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        expected_accuracy = np.mean(y_true == y_pred)  # 4/5 correct
        self.assertAlmostEqual(result['accuracy'], expected_accuracy)

    def test_metrics_precision(self):
        """Test precision metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        metrics = ['precision']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 2
        fp = np.sum((y_true == 0) & (y_pred == 1))  # 1
        expected_precision = tp / (tp + fp + 1e-12)  # 2 / (2 + 1)
        self.assertAlmostEqual(result['precision'], expected_precision)

    def test_metrics_recall(self):
        """Test recall metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        metrics = ['recall']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 2
        fn = np.sum((y_true == 1) & (y_pred == 0))  # 1
        expected_recall = tp / (tp + fn + 1e-12)  # 2 / (2 + 1)
        self.assertAlmostEqual(result['recall'], expected_recall)

    def test_metrics_f1(self):
        """Test F1 score metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        metrics = ['f1']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 2
        fp = np.sum((y_true == 0) & (y_pred == 1))  # 1
        fn = np.sum((y_true == 1) & (y_pred == 0))  # 1
        precision = tp / (tp + fp + 1e-7)  # 2 / (2 + 1)
        recall = tp / (tp + fn + 1e-7)  # 2 / (2 + 1)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        self.assertAlmostEqual(result['f1'], expected_f1)

    def test_metrics_multiple(self):
        """Test multiple metrics calculation together."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        
        # Expected values
        expected_accuracy = np.mean(y_true == y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 2
        fp = np.sum((y_true == 0) & (y_pred == 1))  # 1
        fn = np.sum((y_true == 1) & (y_pred == 0))  # 1
        expected_precision = tp / (tp + fp + 1e-7)
        expected_recall = tp / (tp + fn + 1e-7)
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        
        self.assertAlmostEqual(result['accuracy'], expected_accuracy)
        self.assertAlmostEqual(result['precision'], expected_precision)
        self.assertAlmostEqual(result['recall'], expected_recall)
        self.assertAlmostEqual(result['f1'], expected_f1)

    def test_metrics_empty(self):
        """Test metrics calculation with empty metrics list."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        metrics = []
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        self.assertEqual(result, {})

    def test_metrics_invalid_metric(self):
        """Test handling of invalid metric name."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        metrics = ['invalid_metric']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        self.assertEqual(result, {})

    def test_metrics_zero_division(self):
        """Test metrics calculation when TP, FP, FN are zero to avoid division by zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = ['precision', 'recall', 'f1']
        result = self.cross_validator.get_metrics(y_true, y_pred, metrics)
        self.assertAlmostEqual(result['precision'], 0.0)
        self.assertAlmostEqual(result['recall'], 0.0)
        self.assertTrue(np.isnan(result['f1']) or result['f1'] == 0.0)  # F1 is undefined when precision and recall are 0

if __name__ == "__main__":
    unittest.main()