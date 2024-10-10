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
        result = activation(x, 'leaky_relu', self.device_cpu, alpha=0.01)
        expected = np.array([-0.01, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_prelu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'prelu', self.device_cpu, alpha=0.01)
        expected = np.array([-0.01, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_gelu(self):
        x = np.array([0])
        result = activation(x, 'gelu', self.device_cpu)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_elu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'elu', self.device_cpu, alpha=1.0)
        expected = np.array([-0.6321, 0, 1])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_selu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'selu', self.device_cpu)
        expected = np.array([-1.1113, 0, 1.0507])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_mish(self):
        x = np.array([0])
        result = activation(x, 'mish', self.device_cpu)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_swish(self):
        x = np.array([0])
        result = activation(x, 'swish', self.device_cpu)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sigmoid(self):
        x = np.array([0])
        result = activation(x, 'sigmoid', self.device_cpu)
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_softsign(self):
        x = np.array([0, 1, -1])
        result = activation(x, 'softsign', self.device_cpu)
        expected = np.array([0, 0.5, -0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh(self):
        x = np.array([0])
        result = activation(x, 'tanh', self.device_cpu)
        expected = np.array([0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardtanh(self):
        x = np.array([-2, 0, 2])
        result = activation(x, 'hardtanh', self.device_cpu)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardswish(self):
        x = np.array([-3, 0, 3])
        result = activation(x, 'hardswish', self.device_cpu)
        expected = np.array([0, 0, 3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardsigmoid(self):
        x = np.array([-2, 0, 2])
        result = activation(x, 'hardsigmoid', self.device_cpu)
        expected = np.array([0, 0.5, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanhshrink(self):
        x = np.array([0, 1])
        result = activation(x, 'tanhshrink', self.device_cpu)
        expected = np.array([0, 1 - np.tanh(1)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_softshrink(self):
        x = np.array([-1.5, -0.5, 0.5, 1.5])
        result = activation(x, 'softshrink', self.device_cpu, alpha=1.0)
        expected = np.array([-0.5, 0, 0, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardshrink(self):
        x = np.array([-1.5, -0.5, 0.5, 1.5])
        result = activation(x, 'hardshrink', self.device_cpu, alpha=1.0)
        expected = np.array([-1.5, 0, 0, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_softplus(self):
        x = np.array([0])
        result = activation(x, 'softplus', self.device_cpu)
        expected = np.array([np.log(2)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_softmax(self):
        x = np.array([[1, 2, 3]])
        result = activation(x, 'softmax', self.device_cpu)
        expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rrelu(self):
        x = np.array([-1, 0, 1])
        result = activation(x, 'rrelu', self.device_cpu, alpha=0.01)
        expected = np.array([-0.01, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            activation(np.array([1, 2, 3]), 'invalid', self.device_cpu)

    # Derivative Tests
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

    def test_softsign_derivative(self):
        x = np.array([1])
        result = activation_derivative(x, 'softsign', self.device_cpu)
        expected = np.array([0.25])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh_derivative(self):
        x = np.array([0.5])
        result = activation_derivative(x, 'tanh', self.device_cpu)
        expected = np.array([0.78644773])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardtanh_derivative(self):
        x = np.array([-2, 0, 2])
        result = activation_derivative(x, 'hardtanh', self.device_cpu)
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardsigmoid_derivative(self):
        x = np.array([0])
        result = activation_derivative(x, 'hardsigmoid', self.device_cpu)
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanhshrink_derivative(self):
        x = np.array([1])
        result = activation_derivative(x, 'tanhshrink', self.device_cpu)
        expected = np.array([0.419974])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_softshrink_derivative(self):
        x = np.array([-1.5, -0.5, 0.5, 1.5])
        result = activation_derivative(x, 'softshrink', self.device_cpu, alpha=1.0)
        expected = np.array([1, 0, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_hardshrink_derivative(self):
        x = np.array([-1.5, -0.5, 0.5, 1.5])
        result = activation_derivative(x, 'hardshrink', self.device_cpu, alpha=1.0)
        expected = np.array([1, 0, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
