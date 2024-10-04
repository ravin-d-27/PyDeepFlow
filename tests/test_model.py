import unittest
import numpy as np
from pydeepflow.model import Multi_Layer_ANN
from pydeepflow.device import Device

class TestMultiLayerANN(unittest.TestCase):

    def setUp(self):
        self.device_cpu = Device(use_gpu=False)

        # Small test dataset
        self.X_train = np.array([[0.5, 0.2], [0.9, 0.8], [0.1, 0.4]])
        self.y_train = np.array([[1, 0], [0, 1], [1, 0]])

        self.hidden_layers = [3]
        self.activations = ['relu']

        self.model = Multi_Layer_ANN(self.X_train, self.y_train, self.hidden_layers, self.activations, loss='categorical_crossentropy', use_gpu=False)

    def test_forward_propagation(self):
        activations, Z_values = self.model.forward_propagation(self.X_train)
        self.assertEqual(len(activations), 3)  # Input, Hidden, and Output layers
        self.assertEqual(activations[0].shape, self.X_train.shape)
        self.assertEqual(activations[-1].shape, self.y_train.shape)

    def test_fit(self):
        # Train the model for a small number of epochs
        self.model.fit(epochs=10, learning_rate=0.01)
        predictions = self.model.predict(self.X_train)
        self.assertEqual(predictions.shape[0], self.X_train.shape[0])

    def test_predict(self):
        self.model.fit(epochs=10, learning_rate=0.01)
        predictions = self.model.predict(self.X_train)
        self.assertEqual(predictions.shape, (self.X_train.shape[0],))

    def test_predict_prob(self):
        self.model.fit(epochs=10, learning_rate=0.01)
        prob_predictions = self.model.predict_prob(self.X_train)
        self.assertEqual(prob_predictions.shape, self.y_train.shape)

if __name__ == "__main__":
    unittest.main()
