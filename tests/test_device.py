import unittest
import numpy as np
from pydeepflow.device import Device

class TestDevice(unittest.TestCase):

    def setUp(self):
        self.device_cpu = Device(use_gpu=False)

    def test_array(self):
        data = [1, 2, 3]
        result = self.device_cpu.array(data)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_zeros(self):
        shape = (2, 2)
        result = self.device_cpu.zeros(shape)
        expected = np.zeros(shape)
        np.testing.assert_array_equal(result, expected)

    def test_random(self):
        random_cpu = self.device_cpu.random().rand(3, 3)
        self.assertEqual(random_cpu.shape, (3, 3))

    def test_exp(self):
        x = np.array([0, 1, 2])
        result = self.device_cpu.exp(x)
        expected = np.exp(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dot(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = self.device_cpu.dot(a, b)
        expected = np.dot(a, b)
        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
