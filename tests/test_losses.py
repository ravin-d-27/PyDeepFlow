import unittest
import numpy as np
from pydeepflow.losses import binary_crossentropy, mse, mse_derivative
from pydeepflow.device import Device

class TestLosses(unittest.TestCase):

    def setUp(self):
        self.device_cpu = Device(use_gpu=False)

    def test_binary_crossentropy(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8])
        result = binary_crossentropy(y_true, y_pred, self.device_cpu)
        expected = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        self.assertAlmostEqual(result, expected)

    def test_mse(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8])
        result = mse(y_true, y_pred, self.device_cpu)
        expected = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(result, expected)

    def test_mse_derivative(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8])
        result = mse_derivative(y_true, y_pred, self.device_cpu)
        expected = 2 * (y_pred - y_true) / y_true.size
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
