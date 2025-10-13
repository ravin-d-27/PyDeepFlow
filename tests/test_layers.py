import unittest
import numpy as np
from pydeepflow.model import MaxPooling2D, AveragePooling2D

class TestPoolingLayers(unittest.TestCase):

    def test_max_pooling_forward(self):
        pool = MaxPooling2D(pool_size=(2, 2), stride=2)
        X = np.arange(16).reshape(1, 4, 4, 1)
        out = pool.forward(X)
        expected_out = np.array([[[[ 5.], [ 7.]],[[13.], [15.]]]])
        self.assertEqual(out.shape, (1, 2, 2, 1))
        np.testing.assert_array_almost_equal(out, expected_out)

    def test_max_pooling_backward(self):
        pool = MaxPooling2D(pool_size=(2, 2), stride=2)
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32).reshape(1, 4, 4, 1)
        pool.forward(X)
        dOut = np.ones((1, 2, 2, 1))
        dX = pool.backward(dOut)
        expected_dX = np.array([[0,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,0,1]], dtype=np.float32).reshape(1, 4, 4, 1)
        np.testing.assert_array_almost_equal(dX, expected_dX)

    def test_avg_pooling_forward(self):
        pool = AveragePooling2D(pool_size=(2, 2), stride=2)
        X = np.arange(16).reshape(1, 4, 4, 1)
        out = pool.forward(X)
        expected_out = np.array([[[[ 2.5], [ 4.5]],[[10.5], [12.5]]]])
        self.assertEqual(out.shape, (1, 2, 2, 1))
        np.testing.assert_array_almost_equal(out, expected_out)

    def test_avg_pooling_backward(self):
        pool = AveragePooling2D(pool_size=(2, 2), stride=2)
        X = np.arange(16).reshape(1, 4, 4, 1)
        pool.forward(X)
        dOut = np.ones((1, 2, 2, 1))
        dX = pool.backward(dOut)
        expected_dX = np.ones((4, 4)) * 0.25
        expected_dX = expected_dX.reshape(1, 4, 4, 1)
        np.testing.assert_array_almost_equal(dX, expected_dX)

if __name__ == '__main__':
    unittest.main()
