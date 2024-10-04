import unittest
import numpy as np
from pydeepflow.activations import activation, activation_derivative
from pydeepflow.device import Device

class TestActivations(unittest.TestCase):

    def setUp(self):
        self.device_cpu = Device(use_gpu=False)

    def test_relu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'relu', self.device_cpu)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_leaky_relu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'leaky_relu', self.device_cpu)
        expected = np.array([-0.01, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_sigmoid(self):
        x = np.array([0])
        result = activation(x, 'sigmoid', self.device_cpu)
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh(self):
        x = np.array([0])
        result = activation(x, 'tanh', self.device_cpu)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_softmax(self):
        x = np.array([[1, 2, 3]])
        result = activation(x, 'softmax', self.device_cpu)
        expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            activation(np.array([1, 2, 3]), 'invalid', self.device_cpu)

    def test_relu_derivative(self):
        x = np.array([-1, 0, 1])
        result = activation_derivative(x, 'relu', self.device_cpu)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_sigmoid_derivative(self):
        x = np.array([0.5])
        result = activation_derivative(x, 'sigmoid', self.device_cpu)
        expected = np.array([0.25])
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
