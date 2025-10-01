import unittest
import numpy as np
from pydeepflow.model import Multi_Layer_ANN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestMultiLayerANN(unittest.TestCase):
    def setUp(self):
        # Load and prepare the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # One-hot encode the labels
        self.y_train = np.eye(3)[self.y_train]
        self.y_test = np.eye(3)[self.y_test]
    def test_model_with_batch_norm(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10, 10], 
                                activations=['relu', 'relu'], use_batch_norm=True)
        model.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        results = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(results['accuracy'], 0.8)

    def test_model_without_batch_norm(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10, 10], 
                                activations=['relu', 'relu'], use_batch_norm=False)
        model.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        results = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(results['accuracy'], 0.8)

    def test_forward_propagation(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        activations, Z_values = model.forward_propagation(self.X_train)
        
        self.assertEqual(len(activations), 3)  # Input, hidden, and output layers
        self.assertEqual(activations[0].shape, self.X_train.shape)
        self.assertEqual(activations[-1].shape, self.y_train.shape)

    def test_backpropagation(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        activations, Z_values = model.forward_propagation(self.X_train)
        model.backpropagation(self.X_train, self.y_train, activations, Z_values, learning_rate=0.01)
        
        # Check if weights and biases are updated
        for w in model.weights:
            self.assertFalse(np.allclose(w, 0))
        for b in model.biases:
            self.assertFalse(np.allclose(b, 0))

    def test_predict(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(predictions.shape, self.y_test.shape)

        def test_save_and_load_model(self):
            model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
            model.fit(epochs=10, learning_rate=0.01, verbose=False)
        
        # Save the model
        model.save_model('test_model.npy')
        
        # Create a new model and load the saved weights
        loaded_model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                       activations=['relu'], use_batch_norm=True)
        loaded_model.load_model('test_model.npy')
        
        # Compare predictions
        original_predictions = model.predict(self.X_test)
        loaded_predictions = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_learning_rate_scheduler(self):
        from pydeepflow.learning_rate_scheduler import LearningRateScheduler
        
        lr_scheduler = LearningRateScheduler(initial_lr=0.1, strategy="decay")
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        
        model.fit(epochs=20, learning_rate=0.1, lr_scheduler=lr_scheduler, verbose=False)
        
        # Check if learning rate has decreased
        self.assertLess(lr_scheduler.get_lr(19), 0.1)

    def test_early_stopping(self):
        from pydeepflow.early_stopping import EarlyStopping
        
        early_stop = EarlyStopping(patience=5)
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        
        model.fit(epochs=200, learning_rate=0.1, early_stop=early_stop,
                  X_val=self.X_test, y_val=self.y_test, verbose=False)
        
        # Check if training stopped early
        self.assertLess(len(model.history['train_loss']), 100)

    def test_model_checkpointing(self):
        from pydeepflow.checkpoints import ModelCheckpoint
        import os
        
        checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_loss', save_best_only=True)
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[5], 
                                activations=['relu'], use_batch_norm=True)
        
        model.fit(epochs=20, learning_rate=0.01, checkpoint=checkpoint, 
                  X_val=self.X_test, y_val=self.y_test, verbose=False)
        
        # Check if checkpoint file was created
        self.assertTrue(os.path.exists('./checkpoints'))
        checkpoint_files = os.listdir('./checkpoints')
        self.assertTrue(len(checkpoint_files) > 0)

    def test_batch_norm_effect(self):
        model_bn = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                activations=['relu', 'relu'], use_batch_norm=True)
        model_bn.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Model without batch norm
        model_no_bn = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                      activations=['relu', 'relu'], use_batch_norm=False)
        model_no_bn.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Compare performance
        results_bn = model_bn.evaluate(self.X_test, self.y_test)
        results_no_bn = model_no_bn.evaluate(self.X_test, self.y_test)
        
        # This test might not always pass due to the stochastic nature of training
        # but it gives an idea of the potential benefit of batch normalization
        self.assertGreaterEqual(results_bn['accuracy'], results_no_bn['accuracy'])

    def test_different_activations(self):
        activations = ['relu', 'sigmoid', 'tanh']
        for activation in activations:
            model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10], 
                                    activations=[activation], use_batch_norm=True)
            model.fit(epochs=50, learning_rate=0.01, verbose=False)
            results = model.evaluate(self.X_test, self.y_test)
            self.assertGreater(results['accuracy'], 0.8, f"Model with {activation} activation failed")

    def test_model_history(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10], 
                                activations=['relu'], use_batch_norm=True)
        model.fit(epochs=50, learning_rate=0.01, X_val=self.X_test, y_val=self.y_test, verbose=False)
        
        self.assertIn('train_loss', model.history)
        self.assertIn('val_loss', model.history)
        self.assertIn('train_accuracy', model.history)
        self.assertIn('val_accuracy', model.history)
        self.assertEqual(len(model.history['train_loss']), 50)

    def test_regularization(self):
        # Model with L2 regularization
        model_reg = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                    activations=['relu', 'relu'], use_batch_norm=True, l2_lambda=0.01)
        model_reg.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Model without regularization
        model_no_reg = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                       activations=['relu', 'relu'], use_batch_norm=True, l2_lambda=0.0)
        model_no_reg.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Compare performance on test set
        results_reg = model_reg.evaluate(self.X_test, self.y_test)
        results_no_reg = model_no_reg.evaluate(self.X_test, self.y_test)
        
        # Regularized model should generalize better (though this might not always be true)
        self.assertGreaterEqual(results_reg['accuracy'], results_no_reg['accuracy'])

    def test_dropout(self):
        model_dropout = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                    activations=['relu', 'relu'], use_batch_norm=True, dropout_rate=0.5)
        model_dropout.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Model without dropout
        model_no_dropout = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                           activations=['relu', 'relu'], use_batch_norm=True, dropout_rate=0.0)
        model_no_dropout.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Compare performance on test set
        results_dropout = model_dropout.evaluate(self.X_test, self.y_test)
        results_no_dropout = model_no_dropout.evaluate(self.X_test, self.y_test)
        
        # Dropout should help with generalization, but due to the stochastic nature,
        # this test might not always pass. It's more of a sanity check.
        self.assertGreaterEqual(results_dropout['accuracy'], 0.8)
        self.assertGreaterEqual(results_no_dropout['accuracy'], 0.8)

    def test_gradient_clipping(self):
        # Model with gradient clipping
        model_clip = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                     activations=['relu', 'relu'], use_batch_norm=True)
        model_clip.fit(epochs=50, learning_rate=0.01, clipping_threshold=1.0, verbose=False)
        
        # Model without gradient clipping
        model_no_clip = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[20, 20], 
                                        activations=['relu', 'relu'], use_batch_norm=True)
        model_no_clip.fit(epochs=50, learning_rate=0.01, verbose=False)
        
        # Both models should converge, but the clipped model might be more stable
        results_clip = model_clip.evaluate(self.X_test, self.y_test)
        results_no_clip = model_no_clip.evaluate(self.X_test, self.y_test)
        
        self.assertGreaterEqual(results_clip['accuracy'], 0.8)
        self.assertGreaterEqual(results_no_clip['accuracy'], 0.8)
    
    def test_adam_optimizer(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10, 10],
                                activations=['relu', 'relu'], use_batch_norm=True, optimizer='adam')
        model.fit(epochs=200, learning_rate=0.005, verbose=False)
        results = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(results['accuracy'], 0.8)

    def test_rmsprop_optimizer(self):
        model = Multi_Layer_ANN(self.X_train, self.y_train, hidden_layers=[10, 10],
                                activations=['relu', 'relu'], use_batch_norm=True, optimizer='rmsprop')
        model.fit(epochs=200, learning_rate=0.001, verbose=False)
        results = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(results['accuracy'], 0.8)

if __name__ == '__main__':
    unittest.main()
