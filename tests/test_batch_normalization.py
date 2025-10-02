import unittest
import numpy as np
from pydeepflow.batch_normalization import BatchNormalization
from pydeepflow.device import Device


class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        self.device = Device(use_gpu=False)
        self.bn = BatchNormalization(4, device=self.device)

    def test_normalize_training(self):
        Z = self.device.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
        normalized = self.bn.normalize(Z, training=True)
        self.assertEqual(normalized.shape, Z.shape)
        self.assertAlmostEqual(self.device.mean(normalized), 0, places=7)
        self.assertAlmostEqual(self.device.var(normalized), 1, places=2)

    def test_normalize_inference(self):
        Z = self.device.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
        self.bn.normalize(Z, training=True)  # Update running stats
        normalized = self.bn.normalize(Z, training=False)
        self.assertEqual(normalized.shape, Z.shape)

    def test_backprop(self):
        Z = self.device.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
        dZ = self.device.array(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8]]
        )
        self.bn.normalize(Z, training=True)
        output = self.bn.backprop(Z, dZ, learning_rate=0.01)
        self.assertEqual(output.shape, dZ.shape)


if __name__ == "__main__":
    unittest.main()
