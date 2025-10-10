import unittest
import numpy as np
import sys
import os
from io import StringIO

# Add the parent directory to the path to import pydeepflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydeepflow.model import Multi_Layer_ANN, Multi_Layer_CNN


class TestModelSummary(unittest.TestCase):
    def test_cnn_raises_on_non4d_input(self):
        """Test that Multi_Layer_CNN raises ValueError if input is not 4D."""
        X_bad = np.random.randn(100, 28, 28)  # 3D, should fail
        y = np.eye(10)[np.random.randint(0, 10, 100)]
        layers = [
            {'type': 'conv', 'out_channels': 8, 'kernel_size': 3},
            {'type': 'flatten'},
            {'type': 'dense', 'neurons': 10, 'activation': 'softmax'}
        ]
        with self.assertRaises(ValueError):
            Multi_Layer_CNN(layers, X_bad, y)

    def test_cnn_weight_initialization(self):
        """Test that ConvLayer and Dense layers use correct initializers in Multi_Layer_CNN."""
        # Use a simple CNN config
        X = np.random.randn(10, 8, 8, 3)
        y = np.eye(5)[np.random.randint(0, 5, 10)]
        layers = [
            {'type': 'conv', 'out_channels': 4, 'kernel_size': 3},
            {'type': 'flatten'},
            {'type': 'dense', 'neurons': 6, 'activation': 'relu'},
            {'type': 'dense', 'neurons': 5, 'activation': 'softmax'}
        ]
        model = Multi_Layer_CNN(layers, X, y)
        # Check ConvLayer weight shape and std (He)
        conv = model.layers_list[0]
        W = conv.params['W']
        fan_in = 3 * 3 * 3
        he_std = np.sqrt(2.0 / fan_in)
        self.assertAlmostEqual(W.std(), he_std, delta=he_std*0.5)
        # Check Dense layer weight std (He for relu, Xavier for softmax)
        dense1 = model.layers_list[2]
        dense2 = model.layers_list[3]
        W1 = dense1['W']
        W2 = dense2['W']
        he_dense_std = np.sqrt(2.0 / W1.shape[0])
        xavier_dense_std = np.sqrt(1.0 / W2.shape[0])
        self.assertAlmostEqual(W1.std(), he_dense_std, delta=he_dense_std*0.5)
        self.assertAlmostEqual(W2.std(), xavier_dense_std, delta=xavier_dense_std*0.5)
    """Test model.summary() method and get_model_info() functionality."""
    
    def setUp(self):
        """Set up test models with different architectures."""
        # Simple binary classification model
        self.X_binary = np.random.randn(100, 4)
        self.y_binary = np.random.randint(0, 2, (100, 1))
        
        self.model_binary = Multi_Layer_ANN(
            self.X_binary, self.y_binary,
            hidden_layers=[8, 4],
            activations=['relu', 'sigmoid'],
            loss='binary_crossentropy'
        )
        
        # Multi-class classification model
        self.X_multi = np.random.randn(200, 10)
        self.y_multi = np.eye(5)[np.random.randint(0, 5, 200)]
        
        self.model_multi = Multi_Layer_ANN(
            self.X_multi, self.y_multi,
            hidden_layers=[64, 32, 16],
            activations=['relu', 'relu', 'tanh'],
            loss='categorical_crossentropy',
            l2_lambda=0.01,
            dropout_rate=0.2,
            optimizer='adam'
        )
        
        # Minimal model
        self.X_minimal = np.random.randn(50, 2)
        self.y_minimal = np.random.randint(0, 2, (50, 1))
        
        self.model_minimal = Multi_Layer_ANN(
            self.X_minimal, self.y_minimal,
            hidden_layers=[3],
            activations=['sigmoid']
        )
        
        # CNN model for image classification
        self.X_image = np.random.randn(100, 28, 28, 1)  # MNIST-like data
        self.y_image = np.eye(10)[np.random.randint(0, 10, 100)]
        
        # CNN architecture: Conv -> Conv -> Flatten -> Dense
        self.cnn_layers = [
            {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'conv', 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'type': 'flatten'},
            {'type': 'dense', 'neurons': 128, 'activation': 'relu'},
            {'type': 'dense', 'neurons': 10, 'activation': 'softmax'}
        ]
        
        self.model_cnn = Multi_Layer_CNN(
            self.cnn_layers, self.X_image, self.y_image,
            loss='categorical_crossentropy',
            optimizer='adam'
        )

    def test_summary_method_exists(self):
        """Test that summary method exists and is callable."""
        self.assertTrue(hasattr(self.model_binary, 'summary'))
        self.assertTrue(callable(getattr(self.model_binary, 'summary')))

    def test_get_model_info_method_exists(self):
        """Test that get_model_info method exists and is callable."""
        self.assertTrue(hasattr(self.model_binary, 'get_model_info'))
        self.assertTrue(callable(getattr(self.model_binary, 'get_model_info')))

    def test_summary_output_format(self):
        """Test that summary produces properly formatted output."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.model_binary.summary()
            output = captured_output.getvalue()
            
            # Check for key components in output
            self.assertIn("Model: Multi_Layer_ANN", output)
            self.assertIn("Layer (type)", output)
            self.assertIn("Output Shape", output)
            self.assertIn("Param #", output)
            self.assertIn("Activation", output)
            self.assertIn("Total params:", output)
            self.assertIn("Trainable params:", output)
            self.assertIn("Memory usage:", output)
            self.assertIn("Model Configuration:", output)
            
        finally:
            sys.stdout = sys.__stdout__

    def test_parameter_calculation_binary(self):
        """Test parameter calculation for binary classification model."""
        info = self.model_binary.get_model_info()
        
        # Expected parameters:
        # Layer 1: (4 + 1) * 8 = 40
        # Layer 2: (8 + 1) * 4 = 36  
        # Layer 3: (4 + 1) * 1 = 5
        # Total: 40 + 36 + 5 = 81
        expected_params = 81
        
        self.assertEqual(info['total_params'], expected_params)
        self.assertEqual(info['trainable_params'], expected_params)
        self.assertEqual(info['non_trainable_params'], 0)

    def test_parameter_calculation_multiclass(self):
        """Test parameter calculation for multi-class model."""
        info = self.model_multi.get_model_info()
        
        # Expected parameters:
        # Layer 1: (10 + 1) * 64 = 704
        # Layer 2: (64 + 1) * 32 = 2080
        # Layer 3: (32 + 1) * 16 = 528
        # Layer 4: (16 + 1) * 5 = 85
        # Total: 704 + 2080 + 528 + 85 = 3397
        expected_params = 3397
        
        self.assertEqual(info['total_params'], expected_params)

    def test_layer_info_structure(self):
        """Test the structure of layer information."""
        info = self.model_binary.get_model_info()
        
        # Should have input + 2 hidden + 1 output = 4 layers
        self.assertEqual(len(info['layer_info']), 4)
        
        # Check input layer
        input_layer = info['layer_info'][0]
        self.assertEqual(input_layer['name'], 'Input')
        self.assertEqual(input_layer['type'], 'Input')
        self.assertEqual(input_layer['params'], 0)
        self.assertIsNone(input_layer['activation'])
        
        # Check hidden layers
        hidden1 = info['layer_info'][1]
        self.assertEqual(hidden1['name'], 'Dense_1')
        self.assertEqual(hidden1['type'], 'Dense')
        self.assertEqual(hidden1['activation'], 'relu')
        self.assertEqual(hidden1['params'], 40)  # (4+1)*8
        
        hidden2 = info['layer_info'][2]
        self.assertEqual(hidden2['name'], 'Dense_2')
        self.assertEqual(hidden2['activation'], 'sigmoid')
        self.assertEqual(hidden2['params'], 36)  # (8+1)*4
        
        # Check output layer
        output_layer = info['layer_info'][3]
        self.assertEqual(output_layer['name'], 'Dense_3')
        self.assertEqual(output_layer['type'], 'Dense (Output)')
        self.assertEqual(output_layer['activation'], 'sigmoid')  # Binary classification
        self.assertEqual(output_layer['params'], 5)  # (4+1)*1

    def test_output_shapes(self):
        """Test that output shapes are correctly calculated."""
        info = self.model_multi.get_model_info()
        
        expected_shapes = [
            (None, 10),  # Input
            (None, 64),  # Hidden 1
            (None, 32),  # Hidden 2
            (None, 16),  # Hidden 3
            (None, 5)    # Output
        ]
        
        for i, expected_shape in enumerate(expected_shapes):
            self.assertEqual(info['layer_info'][i]['output_shape'], expected_shape)

    def test_activation_functions(self):
        """Test that activation functions are correctly stored."""
        info = self.model_multi.get_model_info()
        
        expected_activations = [None, 'relu', 'relu', 'tanh', 'softmax']
        
        for i, expected_activation in enumerate(expected_activations):
            self.assertEqual(info['layer_info'][i]['activation'], expected_activation)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        info = self.model_binary.get_model_info()
        
        # Check that memory usage is calculated
        self.assertIn('memory_usage', info)
        memory = info['memory_usage']
        
        self.assertIn('parameters_mb', memory)
        self.assertIn('activations_mb', memory)
        self.assertIn('total_training_mb', memory)
        self.assertIn('total_inference_mb', memory)
        
        # Memory values should be positive
        self.assertGreater(memory['parameters_mb'], 0)
        self.assertGreater(memory['activations_mb'], 0)
        self.assertGreater(memory['total_training_mb'], memory['parameters_mb'])
        self.assertEqual(memory['total_inference_mb'], memory['parameters_mb'])

    def test_configuration_info(self):
        """Test that model configuration is correctly captured."""
        info = self.model_multi.get_model_info()
        config = info['configuration']
        
        self.assertEqual(config['loss_function'], 'categorical_crossentropy')
        self.assertEqual(config['l2_regularization'], 0.01)
        self.assertEqual(config['dropout_rate'], 0.2)
        self.assertEqual(config['optimizer'], 'Adam')
        self.assertEqual(config['device'], 'CPU')  # Default

    def test_minimal_model(self):
        """Test summary with minimal model architecture."""
        info = self.model_minimal.get_model_info()
        
        # Should have input + 1 hidden + 1 output = 3 layers
        self.assertEqual(len(info['layer_info']), 3)
        
        # Expected parameters: (2+1)*3 + (3+1)*1 = 9 + 4 = 13
        expected_params = 13
        self.assertEqual(info['total_params'], expected_params)

    def test_summary_with_different_optimizers(self):
        """Test summary display with different optimizers."""
        # Test with Adam optimizer
        info_adam = self.model_multi.get_model_info()
        self.assertEqual(info_adam['configuration']['optimizer'], 'Adam')
        
        # Test with SGD (default)
        info_sgd = self.model_binary.get_model_info()
        self.assertEqual(info_sgd['configuration']['optimizer'], 'SGD')

    def test_summary_no_crash_on_edge_cases(self):
        """Test that summary doesn't crash on edge cases."""
        # Test calling summary multiple times
        try:
            self.model_binary.summary()
            self.model_binary.summary()
            self.model_multi.summary()
        except Exception as e:
            self.fail(f"Summary method crashed: {e}")

    def test_get_model_info_return_type(self):
        """Test that get_model_info returns correct data types."""
        info = self.model_binary.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIsInstance(info['layer_info'], list)
        self.assertIsInstance(info['total_params'], int)
        self.assertIsInstance(info['memory_usage'], dict)
        self.assertIsInstance(info['configuration'], dict)

    def test_parameter_count_consistency(self):
        """Test that parameter counts are consistent between actual weights and calculation."""
        info = self.model_binary.get_model_info()
        calculated_params = info['total_params']
        
        # Count actual parameters from weights and biases
        actual_params = 0
        for i in range(len(self.model_binary.weights)):
            weight_params = np.prod(self.model_binary.weights[i].shape)
            bias_params = np.prod(self.model_binary.biases[i].shape)
            actual_params += weight_params + bias_params
        
        self.assertEqual(calculated_params, actual_params)

    def test_layer_names_uniqueness(self):
        """Test that layer names are unique and properly formatted."""
        info = self.model_multi.get_model_info()
        
        layer_names = [layer['name'] for layer in info['layer_info']]
        
        # Check uniqueness
        self.assertEqual(len(layer_names), len(set(layer_names)))
        
        # Check expected names
        expected_names = ['Input', 'Dense_1', 'Dense_2', 'Dense_3', 'Dense_4']
        self.assertEqual(layer_names, expected_names)

    def test_batch_size_in_config(self):
        """Test that batch_size is included in configuration when available."""
        # Model should have batch_size from validation
        info = self.model_binary.get_model_info()
        config = info['configuration']
        
        # batch_size should be set (from validation auto-adjustment or default)
        self.assertIn('batch_size', config)

    def test_summary_output_contains_all_layers(self):
        """Test that summary output contains information for all layers."""
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.model_multi.summary()
            output = captured_output.getvalue()
            
            # Should contain all layer names
            self.assertIn('Input', output)
            self.assertIn('Dense_1', output)
            self.assertIn('Dense_2', output)
            self.assertIn('Dense_3', output)
            self.assertIn('Dense_4', output)
            
            # Should contain activation functions
            self.assertIn('relu', output)
            self.assertIn('tanh', output)
            self.assertIn('softmax', output)
            
        finally:
            sys.stdout = sys.__stdout__


    # ========================================================================
    # CNN MODEL SUMMARY TESTS
    # ========================================================================

    def test_cnn_summary_method_exists(self):
        """Test that CNN summary method exists and is callable."""
        self.assertTrue(hasattr(self.model_cnn, 'summary'))
        self.assertTrue(callable(getattr(self.model_cnn, 'summary')))

    def test_cnn_get_model_info_method_exists(self):
        """Test that CNN get_model_info method exists and is callable."""
        self.assertTrue(hasattr(self.model_cnn, 'get_model_info'))
        self.assertTrue(callable(getattr(self.model_cnn, 'get_model_info')))

    def test_cnn_summary_output_format(self):
        """Test that CNN summary produces properly formatted output."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.model_cnn.summary()
            output = captured_output.getvalue()
            
            # Check for key components in output
            self.assertIn("Model: Multi_Layer_CNN", output)
            self.assertIn("Layer (type)", output)
            self.assertIn("Output Shape", output)
            self.assertIn("Param #", output)
            self.assertIn("Total params:", output)
            self.assertIn("Trainable params:", output)
            self.assertIn("Memory usage:", output)
            self.assertIn("Model Configuration:", output)
            
            # Check for CNN-specific layers
            self.assertIn("Conv2D", output)
            self.assertIn("Flatten", output)
            self.assertIn("Dense", output)
            
        finally:
            sys.stdout = sys.__stdout__

    def test_cnn_layer_info_structure(self):
        """Test the structure of CNN layer information."""
        info = self.model_cnn.get_model_info()
        
        # Should have input + conv + conv + flatten + dense + dense = 6 layers
        self.assertEqual(len(info['layer_info']), 6)
        
        # Check input layer
        input_layer = info['layer_info'][0]
        self.assertEqual(input_layer['name'], 'Input')
        self.assertEqual(input_layer['type'], 'Input')
        self.assertEqual(input_layer['params'], 0)
        
        # Check conv layers
        conv1 = info['layer_info'][1]
        self.assertEqual(conv1['name'], 'Conv2D_1')
        self.assertEqual(conv1['type'], 'Conv2D')
        self.assertGreater(conv1['params'], 0)
        
        conv2 = info['layer_info'][2]
        self.assertEqual(conv2['name'], 'Conv2D_2')
        self.assertEqual(conv2['type'], 'Conv2D')
        self.assertGreater(conv2['params'], 0)
        
        # Check flatten layer
        flatten = info['layer_info'][3]
        self.assertEqual(flatten['name'], 'Flatten_3')
        self.assertEqual(flatten['type'], 'Flatten')
        self.assertEqual(flatten['params'], 0)
        
        # Check dense layers
        dense1 = info['layer_info'][4]
        self.assertEqual(dense1['name'], 'Dense_4')
        self.assertEqual(dense1['type'], 'Dense')
        self.assertGreater(dense1['params'], 0)

    def test_cnn_parameter_calculation(self):
        """Test parameter calculation for CNN model."""
        info = self.model_cnn.get_model_info()
        
        # Should have parameters from conv and dense layers
        self.assertGreater(info['total_params'], 0)
        self.assertEqual(info['trainable_params'], info['total_params'])
        self.assertEqual(info['non_trainable_params'], 0)

    def test_cnn_memory_estimation(self):
        """Test memory usage estimation for CNN."""
        info = self.model_cnn.get_model_info()
        
        # Check that memory usage is calculated
        self.assertIn('memory_usage', info)
        memory = info['memory_usage']
        
        self.assertIn('parameters_mb', memory)
        self.assertIn('activations_mb', memory)
        self.assertIn('total_training_mb', memory)
        self.assertIn('total_inference_mb', memory)
        
        # Memory values should be positive
        self.assertGreater(memory['parameters_mb'], 0)
        self.assertGreaterEqual(memory['activations_mb'], 0)
        self.assertGreaterEqual(memory['total_training_mb'], memory['parameters_mb'])
        self.assertEqual(memory['total_inference_mb'], memory['parameters_mb'])

    def test_cnn_configuration_info(self):
        """Test that CNN model configuration is correctly captured."""
        info = self.model_cnn.get_model_info()
        config = info['configuration']
        
        self.assertEqual(config['loss_function'], 'categorical_crossentropy')
        self.assertEqual(config['l2_regularization'], 0.0)  # Default
        self.assertEqual(config['dropout_rate'], 0.0)  # Default
        self.assertEqual(config['optimizer'], 'Adam')
        self.assertEqual(config['device'], 'CPU')  # Default

    def test_cnn_output_shapes(self):
        """Test that CNN output shapes are correctly calculated."""
        info = self.model_cnn.get_model_info()
        
        # Input should be (None, 28, 28, 1)
        input_shape = info['layer_info'][0]['output_shape']
        self.assertEqual(input_shape, (None, 28, 28, 1))
        
        # After first conv (28x28x1 -> 28x28x32 with padding=1)
        conv1_shape = info['layer_info'][1]['output_shape']
        self.assertEqual(conv1_shape[1:3], (28, 28))  # Height and width
        self.assertEqual(conv1_shape[3], 32)  # Channels
        
        # After second conv with stride=2 (28x28x32 -> 14x14x64)
        conv2_shape = info['layer_info'][2]['output_shape']
        self.assertEqual(conv2_shape[1:3], (14, 14))  # Height and width
        self.assertEqual(conv2_shape[3], 64)  # Channels
        
        # After flatten (14*14*64 = 12544)
        flatten_shape = info['layer_info'][3]['output_shape']
        self.assertEqual(flatten_shape, (None, 12544))
        
        # After first dense (12544 -> 128)
        dense1_shape = info['layer_info'][4]['output_shape']
        self.assertEqual(dense1_shape, (None, 128))
        
        # After output dense (128 -> 10)
        output_shape = info['layer_info'][5]['output_shape']
        self.assertEqual(output_shape, (None, 10))

    def test_cnn_summary_no_crash(self):
        """Test that CNN summary doesn't crash on edge cases."""
        try:
            self.model_cnn.summary()
            self.model_cnn.summary()  # Call twice
        except Exception as e:
            self.fail(f"CNN Summary method crashed: {e}")

    def test_cnn_get_model_info_return_type(self):
        """Test that CNN get_model_info returns correct data types."""
        info = self.model_cnn.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIsInstance(info['layer_info'], list)
        self.assertIsInstance(info['total_params'], int)
        self.assertIsInstance(info['memory_usage'], dict)
        self.assertIsInstance(info['configuration'], dict)

    def test_cnn_layer_details(self):
        """Test that CNN layer details are properly formatted."""
        info = self.model_cnn.get_model_info()
        
        # Check conv layer details
        conv1 = info['layer_info'][1]
        if 'details' in conv1:
            self.assertIn('k=3', conv1['details'])
            self.assertIn('s=1', conv1['details'])
            self.assertIn('p=1', conv1['details'])
        
        conv2 = info['layer_info'][2]
        if 'details' in conv2:
            self.assertIn('k=3', conv2['details'])
            self.assertIn('s=2', conv2['details'])
        
        # Check flatten layer details
        flatten = info['layer_info'][3]
        if 'details' in flatten:
            self.assertIn('Flatten', flatten['details'])
        
        # Check dense layer details
        dense1 = info['layer_info'][4]
        if 'details' in dense1:
            self.assertIn('activation=relu', dense1['details'])

    def test_model_type_differentiation(self):
        """Test that ANN and CNN models are properly differentiated."""
        ann_info = self.model_binary.get_model_info()
        cnn_info = self.model_cnn.get_model_info()
        
        # ANN should have only Dense layers (plus Input)
        ann_layer_types = [layer['type'] for layer in ann_info['layer_info']]
        self.assertIn('Input', ann_layer_types)
        self.assertIn('Dense', ann_layer_types)
        self.assertNotIn('Conv2D', ann_layer_types)
        self.assertNotIn('Flatten', ann_layer_types)
        
        # CNN should have Conv2D, Flatten, and Dense layers
        cnn_layer_types = [layer['type'] for layer in cnn_info['layer_info']]
        self.assertIn('Input', cnn_layer_types)
        self.assertIn('Conv2D', cnn_layer_types)
        self.assertIn('Flatten', cnn_layer_types)
        self.assertIn('Dense', cnn_layer_types)

    def test_both_models_summary_compatibility(self):
        """Test that both ANN and CNN models have compatible summary interfaces."""
        # Both should have summary method
        self.assertTrue(hasattr(self.model_binary, 'summary'))
        self.assertTrue(hasattr(self.model_cnn, 'summary'))
        
        # Both should have get_model_info method
        self.assertTrue(hasattr(self.model_binary, 'get_model_info'))
        self.assertTrue(hasattr(self.model_cnn, 'get_model_info'))
        
        # Both should return similar info structure
        ann_info = self.model_binary.get_model_info()
        cnn_info = self.model_cnn.get_model_info()
        
        # Check common keys
        common_keys = ['layer_info', 'total_params', 'memory_usage', 'configuration']
        for key in common_keys:
            self.assertIn(key, ann_info)
            self.assertIn(key, cnn_info)


if __name__ == '__main__':
    unittest.main()