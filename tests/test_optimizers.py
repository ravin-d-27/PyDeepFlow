import unittest
import numpy as np
from pydeepflow.optimizers import Adam, RMSprop

class TestOptimizers(unittest.TestCase):

    def test_adam_multiple_steps(self):
        optimizer = Adam(learning_rate=0.01)
        params = [np.array([1.0]), np.array([0.5])]
        grads = [np.array([0.1]), np.array([0.05])]

        # Run multiple updates
        for _ in range(10):
            optimizer.update(params, grads)

        # Params should decrease
        self.assertLess(params[0][0], 1.0, "Adam did not decrease param[0]")
        self.assertLess(params[1][0], 0.5, "Adam did not decrease param[1]")

        # Should remain positive
        self.assertGreater(params[0][0], 0.0, "Adam drove param[0] below zero")
        self.assertGreater(params[1][0], 0.0, "Adam drove param[1] below zero")

    def test_rmsprop_multiple_steps(self):
        optimizer = RMSprop(learning_rate=0.01)
        params = [np.array([1.0]), np.array([0.5])]
        grads = [np.array([0.1]), np.array([0.05])]

        # Run multiple updates
        for _ in range(10):
            optimizer.update(params, grads)

        # Params should decrease
        self.assertLess(params[0][0], 1.0, "RMSprop did not decrease param[0]")
        self.assertLess(params[1][0], 0.5, "RMSprop did not decrease param[1]")

        # Should remain positive
        self.assertGreater(params[0][0], 0.0, "RMSprop drove param[0] below zero")
        self.assertGreater(params[1][0], 0.0, "RMSprop drove param[1] below zero")
        print(f"\nRMSprop test passed with final params: {params}")
