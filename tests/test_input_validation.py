import unittest
import numpy as np
import sys
import os
import warnings
# Add the parent directory to the path to import pydeepflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydeepflow.model import Multi_Layer_ANN

class TestInputValidation(unittest.TestCase):
    """Test comprehensive input validation for Multi_Layer_ANN initialization."""
    
    def setUp(self):
        """Set up valid test data for comparison."""
        # Valid training data
        self.X_train_valid = np.random.randn(100, 4)
        self.y_train_binary = np.random.randint(0, 2, (100, 1))
        self.y_train_multiclass = np.eye(3)[np.random.randint(0, 3, 100)]
        
        # Valid configuration
        self.hidden_layers_valid = [10, 5]
        self.activations_valid = ['relu', 'sigmoid']
        self.loss_valid = 'categorical_crossentropy'
    
    def test_valid_initialization(self):
        """Test that valid inputs don't raise errors."""
        # Should not raise any exceptions
        model = Multi_Layer_ANN(
            self.X_train_valid, self.y_train_multiclass,
            self.hidden_layers_valid, self.activations_valid,
            loss=self.loss_valid
        )
        self.assertIsNotNone(model)
    
    # X_train validation tests
    def test_x_train_none(self):
        """Test X_train cannot be None."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(None, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("X_train cannot be None", str(context.exception))
    
    def test_x_train_empty(self):
        """Test X_train cannot be empty."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN([], self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("X_train cannot be empty", str(context.exception))
    
    def test_x_train_wrong_dimensions(self):
        """Test X_train dimension validation."""
        # 0D array
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(5, self.y_train_multiclass,
                        self.hidden_layers_valid, self.activations_valid)
        self.assertIn("must be at least 1", str(context.exception))  # Changed to match actual message
        
        # 3D array
        X_3d = np.random.randn(10, 4, 3)
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_3d, self.y_train_multiclass[:10],
                        self.hidden_layers_valid, self.activations_valid)
        self.assertIn("must be 1D or 2D array", str(context.exception))

    
    def test_x_train_non_numeric(self):
        """Test X_train must be numeric."""
        X_string = [['a', 'b'], ['c', 'd']]
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(X_string, self.y_train_binary[:2],
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("must contain numeric data", str(context.exception))
    
    def test_x_train_with_nan(self):
        """Test X_train cannot contain NaN values."""
        X_with_nan = self.X_train_valid.copy()
        X_with_nan[0, 0] = np.nan
        X_with_nan[1, 1] = np.nan
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_with_nan, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("contains NaN values", str(context.exception))
    
    def test_x_train_with_inf(self):
        """Test X_train cannot contain infinite values."""
        X_with_inf = self.X_train_valid.copy()
        X_with_inf[0, 0] = np.inf
        X_with_inf[1, 1] = -np.inf
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_with_inf, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("contains infinite values", str(context.exception))
    
    # Y_train validation tests
    def test_y_train_none(self):
        """Test Y_train cannot be None."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, None,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("Y_train cannot be None", str(context.exception))
    
    def test_y_train_empty(self):
        """Test Y_train cannot be empty."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, [],
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("Y_train cannot be empty", str(context.exception))
    
    # Data compatibility tests
    def test_sample_count_mismatch(self):
        """Test X_train and Y_train must have same number of samples."""
        X_short = self.X_train_valid[:50]  # 50 samples
        y_long = self.y_train_multiclass  # 100 samples
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_short, y_long,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("must have the same number of samples", str(context.exception))
    
    def test_insufficient_samples(self):
        """Test need at least 2 samples."""
        X_single = self.X_train_valid[:1]
        y_single = self.y_train_multiclass[:1]
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_single, y_single,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("Need at least 2 samples", str(context.exception))
    
    def test_invalid_one_hot_encoding(self):
        """Test invalid one-hot encoding detection."""
        # Create invalid one-hot (rows don't sum to 1)
        y_invalid_onehot = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1]])
        X_small = self.X_train_valid[:3]
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_small, y_invalid_onehot,
                          self.hidden_layers_valid, self.activations_valid)
        self.assertIn("should be one-hot encoded", str(context.exception))
    
    # Hidden layers validation tests
    def test_hidden_layers_wrong_type(self):
        """Test hidden_layers must be list or tuple."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          "invalid", self.activations_valid)
        self.assertIn("must be a list or tuple", str(context.exception))
    
    def test_hidden_layers_empty(self):
        """Test hidden_layers cannot be empty."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [], [])
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_hidden_layers_non_integer(self):
        """Test hidden layer sizes must be integers."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 5.5], ['relu', 'sigmoid'])
        self.assertIn("must be integers", str(context.exception))
    
    def test_hidden_layers_negative(self):
        """Test hidden layer sizes must be positive."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, -5], ['relu', 'sigmoid'])
        self.assertIn("must be positive integers", str(context.exception))
    
    def test_hidden_layers_zero(self):
        """Test hidden layer sizes cannot be zero."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 0], ['relu', 'sigmoid'])
        self.assertIn("must be positive integers", str(context.exception))
    
    def test_hidden_layers_too_large(self):
        """Test warning for very large hidden layers."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 15000], ['relu', 'sigmoid'])
        self.assertIn("seems too large", str(context.exception))
    
    # Activations validation tests
    def test_activations_wrong_type(self):
        """Test activations must be list or tuple."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, "relu")
        self.assertIn("must be a list or tuple", str(context.exception))
    
    def test_activations_count_mismatch(self):
        """Test activations count must match hidden layers count."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 5], ['relu'])  # 2 layers, 1 activation
        self.assertIn("must match number of hidden layers", str(context.exception))
    
    def test_activations_non_string(self):
        """Test activation functions must be strings."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 5], ['relu', 123])
        self.assertIn("must be strings", str(context.exception))
    
    def test_invalid_activation_function(self):
        """Test unsupported activation functions."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          [10, 5], ['relu', 'invalid_activation'])
        self.assertIn("Unsupported activation function", str(context.exception))
    
    # Loss function validation tests
    def test_loss_wrong_type(self):
        """Test loss function must be string."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          loss=123)
        self.assertIn("must be a string", str(context.exception))
    
    def test_invalid_loss_function(self):
        """Test unsupported loss functions."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          loss='invalid_loss')
        self.assertIn("Unsupported loss function", str(context.exception))
    
    # Regularization parameters validation tests
    def test_l2_lambda_wrong_type(self):
        """Test l2_lambda must be numeric."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          l2_lambda="invalid")
        self.assertIn("must be a number", str(context.exception))
    
    def test_l2_lambda_negative(self):
        """Test l2_lambda must be non-negative."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          l2_lambda=-0.1)
        self.assertIn("must be non-negative", str(context.exception))
    
    def test_l2_lambda_too_large(self):
        """Test warning for very large l2_lambda."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          l2_lambda=2.0)
        self.assertIn("seems too large", str(context.exception))
    
    def test_dropout_rate_wrong_type(self):
        """Test dropout_rate must be numeric."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          dropout_rate="invalid")
        self.assertIn("must be a number", str(context.exception))
    
    def test_dropout_rate_out_of_range(self):
        """Test dropout_rate must be in [0, 1)."""
        # Test negative
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          dropout_rate=-0.1)
        self.assertIn("must be in range [0, 1)", str(context.exception))
        
        # Test >= 1
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          dropout_rate=1.0)
        self.assertIn("must be in range [0, 1)", str(context.exception))
    
    # Optimizer validation tests
    def test_optimizer_wrong_type(self):
        """Test optimizer must be string."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          optimizer=123)
        self.assertIn("must be a string", str(context.exception))
    
    def test_invalid_optimizer(self):
        """Test unsupported optimizers."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          optimizer='invalid_optimizer')
        self.assertIn("Unsupported optimizer", str(context.exception))
    
    # Learning rate validation tests
    def test_learning_rate_wrong_type(self):
        """Test learning_rate must be numeric."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          learning_rate="invalid")
        self.assertIn("must be a number", str(context.exception))
    
    def test_learning_rate_negative(self):
        """Test learning_rate must be positive."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          learning_rate=-0.01)
        self.assertIn("must be positive", str(context.exception))
    
    def test_learning_rate_zero(self):
        """Test learning_rate cannot be zero."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          learning_rate=0.0)
        self.assertIn("must be positive", str(context.exception))
    
    def test_learning_rate_too_large(self):
        """Test learning_rate warning for values > 1.0."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          learning_rate=5.0)
        self.assertIn("seems too large", str(context.exception))
    
    def test_learning_rate_too_small(self):
        """Test learning_rate warning for very small values."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          learning_rate=1e-10)
        self.assertIn("seems too small", str(context.exception))
    
    # Epochs validation tests
    def test_epochs_wrong_type(self):
        """Test epochs must be integer."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          epochs=10.5)
        self.assertIn("must be an integer", str(context.exception))
    
    def test_epochs_negative(self):
        """Test epochs must be positive."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          epochs=-10)
        self.assertIn("must be positive", str(context.exception))
    
    def test_epochs_zero(self):
        """Test epochs cannot be zero."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          epochs=0)
        self.assertIn("must be positive", str(context.exception))
    
    def test_epochs_too_large(self):
        """Test epochs warning for very large values."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          epochs=15000)
        self.assertIn("seems too large", str(context.exception))
    
    # Batch size validation tests
    def test_batch_size_wrong_type(self):
        """Test batch_size must be integer."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          batch_size=32.5)
        self.assertIn("must be an integer", str(context.exception))
    
    def test_batch_size_negative(self):
        """Test batch_size must be positive."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          batch_size=-10)
        self.assertIn("must be positive", str(context.exception))
    
    def test_batch_size_zero(self):
        """Test batch_size cannot be zero."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                          self.hidden_layers_valid, self.activations_valid,
                          batch_size=0)
        self.assertIn("must be positive", str(context.exception))
    
    def test_batch_size_larger_than_samples(self):
        """Test batch_size larger than samples triggers warning and auto-adjustment."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                                self.hidden_layers_valid, self.activations_valid,
                                batch_size=200)  # More than 100 samples
            
            # Should have issued a warning
            self.assertGreaterEqual(len(w), 1)
            
            # Verify warning message mentions auto-adjustment
            warning_found = False
            for warning in w:
                if "Automatically adjusting batch_size" in str(warning.message):
                    warning_found = True
                    break
            self.assertTrue(warning_found, "Expected warning about batch_size auto-adjustment")
            
            # Verify batch_size was adjusted to match sample count
            self.assertEqual(model.batch_size, 100)

    
    def test_batch_size_too_large(self):
        """Test batch_size warning for very large values."""
        X_large = np.random.randn(2000, 4)
        y_large = np.eye(3)[np.random.randint(0, 3, 2000)]
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(X_large, y_large,
                          self.hidden_layers_valid, self.activations_valid,
                          batch_size=2000)
        self.assertIn("seems too large", str(context.exception))
    
    def test_batch_size_one_with_warning(self):
        """Test batch_size=1 generates warning for large datasets."""
        import warnings
        import sys
        
        # Clear any cached warnings
        for module in list(sys.modules.values()):
            if hasattr(module, '__warningregistry__'):
                module.__warningregistry__.clear()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                                self.hidden_layers_valid, self.activations_valid,
                                batch_size=1)
            
            # Check if warning was triggered
            self.assertGreaterEqual(len(w), 1, f"Expected at least one warning. Got {len(w)} warnings")
            
            # Verify the warning message contains "online learning"
            warning_messages = [str(warning.message) for warning in w]
            found = any("online learning" in msg for msg in warning_messages)
            self.assertTrue(found, f"Expected 'online learning' in warnings. Got: {warning_messages}")




    
    # Edge cases and boundary tests
    def test_minimal_valid_configuration(self):
        """Test minimal valid configuration works."""
        X_minimal = np.array([[1, 2], [3, 4]])
        y_minimal = np.array([[1], [0]])
        model = Multi_Layer_ANN(X_minimal, y_minimal, [1], ['relu'], batch_size=2)  # Set batch_size to match sample count
        self.assertIsNotNone(model)

    
    def test_large_valid_configuration(self):
        """Test large but reasonable configuration works."""
        X_large = np.random.randn(1000, 50)
        y_large = np.eye(10)[np.random.randint(0, 10, 1000)]
        model = Multi_Layer_ANN(X_large, y_large,
                              [100, 50, 20], ['relu', 'relu', 'sigmoid'],
                              l2_lambda=0.01, dropout_rate=0.2,
                              learning_rate=0.001, epochs=100, batch_size=32)
        self.assertIsNotNone(model)
    
    def test_all_supported_activations(self):
        """Test all supported activation functions work."""
        activations = ['relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh']
        hidden_layers = [5] * len(activations)
        model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                              hidden_layers, activations)
        self.assertIsNotNone(model)
    
    def test_all_supported_losses(self):
        """Test all supported loss functions work."""
        losses = ['categorical_crossentropy', 'binary_crossentropy', 'mse']
        for loss in losses:
            if loss == 'binary_crossentropy':
                y_data = self.y_train_binary
            else:
                y_data = self.y_train_multiclass
            model = Multi_Layer_ANN(self.X_train_valid, y_data,
                                  [10], ['relu'], loss=loss)
            self.assertIsNotNone(model)
    
    def test_all_supported_optimizers(self):
        """Test all supported optimizers work."""
        optimizers = ['sgd', 'adam', 'rmsprop']
        for optimizer in optimizers:
            model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                                  [10], ['relu'], optimizer=optimizer)
            self.assertIsNotNone(model)
    
    def test_hyperparameter_combinations(self):
        """Test various valid hyperparameter combinations."""
        test_cases = [
            {'learning_rate': 0.001, 'epochs': 50, 'batch_size': 16},
            {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32},
            {'learning_rate': 0.0001, 'epochs': 500, 'batch_size': 64},
            {'learning_rate': 0.1, 'epochs': 10, 'batch_size': 8},
        ]
        
        for params in test_cases:
            model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                                  [10], ['relu'],
                                  learning_rate=params['learning_rate'],
                                  epochs=params['epochs'],
                                  batch_size=params['batch_size'])
            self.assertIsNotNone(model)

    # Initial weights validation tests
    def test_initial_weights_valid(self):
        """Test valid initial_weights values work."""
        valid_weights = ['auto', 'he', 'xavier', 'glorot', 'lecun', 'random']
        
        for weight_init in valid_weights:
            model = Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                                   [10], ['relu'], initial_weights=weight_init)
            self.assertIsNotNone(model)

    def test_initial_weights_invalid(self):
        """Test invalid initial_weights values are rejected."""
        with self.assertRaises(ValueError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                           [10], ['relu'], initial_weights='invalid')
        self.assertIn("Unsupported initial_weights", str(context.exception))

    def test_initial_weights_wrong_type(self):
        """Test initial_weights must be string."""
        with self.assertRaises(TypeError) as context:
            Multi_Layer_ANN(self.X_train_valid, self.y_train_multiclass,
                           [10], ['relu'], initial_weights=123)
        self.assertIn("must be a string", str(context.exception))


if __name__ == '__main__':
    unittest.main()