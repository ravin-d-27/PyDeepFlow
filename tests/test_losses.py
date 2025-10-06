
import unittest
import numpy as np
from pydeepflow.losses import (
    binary_crossentropy, mse, mse_derivative,
    categorical_crossentropy, categorical_crossentropy_derivative,
    hinge_loss, hinge_loss_derivative,
    huber_loss, huber_loss_derivative
)
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

    def test_categorical_crossentropy(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        result = categorical_crossentropy(y_true, y_pred, self.device_cpu)
        expected = -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
        self.assertAlmostEqual(result, expected)

    def test_categorical_crossentropy_derivative(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        result = categorical_crossentropy_derivative(y_true, y_pred, self.device_cpu)
        expected = -y_true / (y_pred + 1e-8)
        np.testing.assert_array_almost_equal(result, expected)

    def test_hinge_loss(self):
        y_true = np.array([1, -1, 1])
        y_pred = np.array([0.8, -0.5, 0.3])
        result = hinge_loss(y_true, y_pred, self.device_cpu)
        expected = np.mean(np.maximum(0, 1 - y_true * y_pred))
        self.assertAlmostEqual(result, expected)

    def test_hinge_loss_derivative(self):
        y_true = np.array([1, -1, 1])
        y_pred = np.array([0.8, -0.5, 0.3])
        result = hinge_loss_derivative(y_true, y_pred, self.device_cpu)
        expected = np.where(y_true * y_pred < 1, -y_true, 0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_huber_loss(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.7, 2.5])
        delta = 1.0
        error = y_true - y_pred
        is_small = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        expected = np.mean(np.where(is_small, squared_loss, linear_loss))
        result = huber_loss(y_true, y_pred, self.device_cpu, delta)
        self.assertAlmostEqual(result, expected)

    def test_huber_loss_derivative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.7, 2.5])
        delta = 1.0
        error = y_pred - y_true
        is_small = np.abs(error) <= delta
        expected = np.where(is_small, error, delta * np.sign(error))
        result = huber_loss_derivative(y_true, y_pred, self.device_cpu, delta)
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()

