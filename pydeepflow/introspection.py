"""
Model Introspection Module for PyDeepFlow

This module provides comprehensive model analysis and visualization capabilities
for neural network architectures. It supports both ANN and CNN models with
unified interfaces for summary generation and model information extraction.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseModelIntrospector(ABC):
    """
    Abstract base class for model introspection functionality.
    
    This class defines the interface that all model introspectors must implement,
    ensuring consistent behavior across different model types (ANN, CNN, etc.).
    """
    
    @abstractmethod
    def get_layer_info(self):
        """
        Extract detailed information about each layer in the model.
        
        Returns:
            list: List of dictionaries containing layer information
        """
        pass
    
    @abstractmethod
    def calculate_parameters(self):
        """
        Calculate the total number of parameters in the model.
        
        Returns:
            dict: Dictionary containing parameter counts
        """
        pass
    
    @abstractmethod
    def estimate_memory_usage(self, batch_size=32):
        """
        Estimate memory usage for training and inference.
        
        Args:
            batch_size (int): Batch size for memory estimation
            
        Returns:
            dict: Dictionary containing memory usage estimates
        """
        pass
    
    @abstractmethod
    def get_model_configuration(self):
        """
        Extract model configuration information.
        
        Returns:
            dict: Dictionary containing model configuration
        """
        pass


class ANNIntrospector(BaseModelIntrospector):
    """
    Model introspector for Artificial Neural Networks (Dense/Fully-Connected layers).
    
    This class provides detailed analysis capabilities for ANN models including
    parameter counting, memory estimation, and architecture visualization.
    """
    
    def __init__(self, model):
        """
        Initialize the ANN introspector.
        
        Args:
            model: Multi_Layer_ANN instance to introspect
        """
        self.model = model
    
    def get_layer_info(self):
        """
        Extract detailed information about each layer in the ANN.
        
        Returns:
            list: List of dictionaries containing:
                - name: Layer name (e.g., 'Input', 'Dense_1')
                - type: Layer type (e.g., 'Input', 'Dense', 'Dense (Output)')
                - input_shape: Input tensor shape
                - output_shape: Output tensor shape
                - params: Number of parameters in the layer
                - activation: Activation function name
                - init_method: Weight initialization method (if available)
        """
        layer_info = []
        
        # Input layer
        layer_info.append({
            'name': 'Input',
            'type': 'Input',
            'input_shape': (None, self.model.layers[0]),
            'output_shape': (None, self.model.layers[0]),
            'params': 0,
            'activation': None,
            'init_method': None
        })
        
        # Hidden layers
        for i in range(len(self.model.layers) - 2):
            layer_params = (self.model.layers[i] + 1) * self.model.layers[i+1]
            
            # Get initialization method from metadata if available
            init_method = None
            if hasattr(self.model, 'init_metadata') and i < len(self.model.init_metadata):
                init_method = self.model.init_metadata[i].method
            
            layer_info.append({
                'name': f'Dense_{i+1}',
                'type': 'Dense',
                'input_shape': (None, self.model.layers[i]),
                'output_shape': (None, self.model.layers[i+1]),
                'params': layer_params,
                'activation': self.model.activations[i],
                'init_method': init_method
            })
        
        # Output layer
        output_params = (self.model.layers[-2] + 1) * self.model.layers[-1]
        output_layer_idx = len(self.model.layers) - 2
        
        # Get initialization method for output layer
        init_method = None
        if hasattr(self.model, 'init_metadata') and output_layer_idx < len(self.model.init_metadata):
            init_method = self.model.init_metadata[output_layer_idx].method
        
        layer_info.append({
            'name': f'Dense_{len(self.model.layers)-1}',
            'type': 'Dense (Output)',
            'input_shape': (None, self.model.layers[-2]),
            'output_shape': (None, self.model.layers[-1]),
            'params': output_params,
            'activation': self.model.output_activation,
            'init_method': init_method
        })
        
        return layer_info
    
    def calculate_parameters(self):
        """
        Calculate parameter counts for the ANN model.
        
        Returns:
            dict: Dictionary containing:
                - total_params: Total number of parameters
                - trainable_params: Number of trainable parameters
                - non_trainable_params: Number of non-trainable parameters
        """
        total_params = 0
        
        # Calculate parameters for each layer: (input_size + 1) * output_size
        for i in range(len(self.model.layers) - 1):
            layer_params = (self.model.layers[i] + 1) * self.model.layers[i+1]
            total_params += layer_params
        
        return {
            'total_params': int(total_params),
            'trainable_params': int(total_params),  # All params are trainable in basic ANN
            'non_trainable_params': 0
        }
    
    def estimate_memory_usage(self, batch_size=32):
        """
        Estimate memory usage for ANN training and inference.
        
        Args:
            batch_size (int): Batch size for estimation
            
        Returns:
            dict: Dictionary containing memory estimates in MB:
                - parameters_mb: Memory for model parameters
                - activations_mb: Memory for activations during forward pass
                - total_training_mb: Total memory for training
                - total_inference_mb: Total memory for inference
        """
        param_counts = self.calculate_parameters()
        
        # Memory for parameters (assuming float32 = 4 bytes per parameter)
        param_memory_mb = (param_counts['total_params'] * 4) / (1024 * 1024)
        
        # Memory for activations (estimate based on largest layer)
        # Use actual batch_size if available from model
        actual_batch_size = getattr(self.model, 'batch_size', batch_size)
        max_layer_size = max(self.model.layers)
        activation_memory_mb = (max_layer_size * actual_batch_size * 4) / (1024 * 1024)
        
        return {
            'parameters_mb': param_memory_mb,
            'activations_mb': activation_memory_mb,
            'total_training_mb': param_memory_mb + activation_memory_mb,
            'total_inference_mb': param_memory_mb
        }
    
    def get_model_configuration(self):
        """
        Extract ANN model configuration information.
        
        Returns:
            dict: Dictionary containing model configuration:
                - loss_function: Loss function name
                - l2_regularization: L2 regularization parameter
                - dropout_rate: Dropout rate
                - optimizer: Optimizer name
                - device: Device type (CPU/GPU)
                - batch_size: Batch size (if available)
                - initialization_metadata: List of initialization metadata (if available)
        """
        # Determine optimizer name
        optimizer_name = "SGD"  # Default
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_name = type(self.model.optimizer).__name__
        
        # Device information
        device_type = "GPU" if self.model.device.use_gpu else "CPU"
        
        config = {
            'loss_function': self.model.loss,
            'l2_regularization': self.model.regularization.l2_lambda,
            'dropout_rate': self.model.regularization.dropout_rate,
            'optimizer': optimizer_name,
            'device': device_type
        }
        
        # Add batch_size if available
        if hasattr(self.model, 'batch_size'):
            config['batch_size'] = self.model.batch_size
        else:
            config['batch_size'] = 'Not set'
        
        # Add initialization metadata if available
        if hasattr(self.model, 'init_metadata'):
            config['initialization_metadata'] = [
                {
                    'layer_index': meta.layer_index,
                    'layer_type': meta.layer_type,
                    'method': meta.method,
                    'activation': meta.activation,
                    'shape': meta.shape,
                    'bias_value': meta.bias_value,
                    'fan_in': meta.fan_in,
                    'fan_out': meta.fan_out,
                    'scale': meta.scale
                }
                for meta in self.model.init_metadata
            ]
        
        return config


class CNNIntrospector(BaseModelIntrospector):
    """
    Model introspector for Convolutional Neural Networks.
    
    This class provides detailed analysis capabilities for CNN models including
    convolutional layers, pooling layers, and mixed CNN+Dense architectures.
    """
    
    def __init__(self, model):
        """
        Initialize the CNN introspector.
        
        Args:
            model: Multi_Layer_CNN instance to introspect
        """
        self.model = model
    
    def get_layer_info(self):
        """
        Extract detailed information about each layer in the CNN.
        
        Returns:
            list: List of dictionaries containing layer information
        """
        layer_info = []
        current_shape = self.model.X_train.shape[1:]  # (H, W, C)
        
        # Input layer
        layer_info.append({
            'name': 'Input',
            'type': 'Input',
            'input_shape': (None,) + current_shape,
            'output_shape': (None,) + current_shape,
            'params': 0,
            'activation': None,
            'details': '-'
        })
        
        # Process each layer in the CNN
        for i, layer in enumerate(self.model.layers_list):
            layer_name = f"{self._get_layer_type_name(layer)}_{i+1}"
            
            if hasattr(layer, 'params') and isinstance(layer.params, dict) and 'W' in layer.params:  # Conv layer
                layer_params = np.prod(layer.params['W'].shape) + np.prod(layer.params['b'].shape)
                
                # Calculate output shape for conv layer
                if hasattr(layer, 'stride') and hasattr(layer, 'padding'):
                    H, W, C_in = current_shape
                    kernel_size = getattr(layer, 'Fh', getattr(layer, 'kernel_size', 3))
                    H_out = (H + 2 * layer.padding - kernel_size) // layer.stride + 1
                    W_out = (W + 2 * layer.padding - kernel_size) // layer.stride + 1
                    C_out = layer.out_channels
                    current_shape = (H_out, W_out, C_out)
                
                kernel_size = getattr(layer, 'Fh', getattr(layer, 'kernel_size', 3))
                details = f"k={kernel_size}, s={layer.stride}"
                if hasattr(layer, 'padding') and layer.padding > 0:
                    details += f", p={layer.padding}"
                
                layer_info.append({
                    'name': layer_name,
                    'type': 'Conv2D',
                    'input_shape': layer_info[-1]['output_shape'],
                    'output_shape': (None,) + current_shape,
                    'params': layer_params,
                    'activation': getattr(layer, 'activation', 'relu'),
                    'details': details
                })
                
            elif hasattr(layer, 'forward') and not (hasattr(layer, 'params') and 'W' in layer.params):  # Flatten layer
                # Calculate flattened size
                flattened_size = np.prod(current_shape)
                current_shape = (flattened_size,)
                
                layer_info.append({
                    'name': layer_name,
                    'type': 'Flatten',
                    'input_shape': layer_info[-1]['output_shape'],
                    'output_shape': (None,) + current_shape,
                    'params': 0,
                    'activation': None,
                    'details': 'Flatten operation'
                })
                
            elif isinstance(layer, dict) and 'W' in layer:  # Dense layer
                input_dim = current_shape[0]
                output_dim = layer['W'].shape[1]
                layer_params = np.prod(layer['W'].shape) + np.prod(layer['b'].shape)
                current_shape = (output_dim,)
                
                layer_type = 'Dense (Output)' if i == len(self.model.layers_list) - 1 else 'Dense'
                details = f"activation={layer['activation']}"
                
                layer_info.append({
                    'name': layer_name,
                    'type': layer_type,
                    'input_shape': layer_info[-1]['output_shape'],
                    'output_shape': (None,) + current_shape,
                    'params': layer_params,
                    'activation': layer['activation'],
                    'details': details
                })
        
        return layer_info
    
    def calculate_parameters(self):
        """
        Calculate parameter counts for the CNN model.
        
        Returns:
            dict: Dictionary containing parameter counts
        """
        total_params = 0
        
        # Count parameters from trainable_params list
        for param in self.model.trainable_params:
            total_params += np.prod(param.shape)
        
        return {
            'total_params': int(total_params),
            'trainable_params': int(total_params),
            'non_trainable_params': 0
        }
    
    def estimate_memory_usage(self, batch_size=32):
        """
        Estimate memory usage for CNN training and inference.
        
        Args:
            batch_size (int): Batch size for estimation
            
        Returns:
            dict: Dictionary containing memory estimates in MB
        """
        param_counts = self.calculate_parameters()
        
        # Memory for parameters (assuming float32 = 4 bytes per parameter)
        param_memory_mb = (param_counts['total_params'] * 4) / (1024 * 1024)
        
        # Memory for activations (estimate based on input size and largest feature maps)
        input_size = np.prod(self.model.X_train.shape[1:])
        activation_memory_mb = (input_size * batch_size * 4) / (1024 * 1024)
        
        return {
            'parameters_mb': param_memory_mb,
            'activations_mb': activation_memory_mb,
            'total_training_mb': param_memory_mb + activation_memory_mb,
            'total_inference_mb': param_memory_mb
        }
    
    def get_model_configuration(self):
        """
        Extract CNN model configuration information.
        
        Returns:
            dict: Dictionary containing model configuration
        """
        # Determine optimizer name
        optimizer_name = "SGD"  # Default
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_name = type(self.model.optimizer).__name__
        
        # Device information
        device_type = "GPU" if self.model.device.use_gpu else "CPU"
        
        return {
            'loss_function': self.model.loss,
            'l2_regularization': self.model.regularization.l2_lambda,
            'dropout_rate': self.model.regularization.dropout_rate,
            'optimizer': optimizer_name,
            'device': device_type,
            'batch_size': getattr(self.model, 'batch_size', 'Not set')
        }
    
    def _get_layer_type_name(self, layer):
        """
        Get the type name for a layer.
        
        Args:
            layer: Layer object
            
        Returns:
            str: Layer type name
        """
        if hasattr(layer, 'params') and isinstance(layer.params, dict) and 'W' in layer.params:  # Conv layer
            return 'Conv2D'
        elif hasattr(layer, 'forward') and not (hasattr(layer, 'params') and 'W' in layer.params):  # Flatten
            return 'Flatten'
        elif isinstance(layer, dict):  # Dense layer
            return 'Dense'
        else:
            return 'Unknown'


class ModelSummaryFormatter:
    """
    Utility class for formatting model summary output.
    
    This class provides methods to format model information into professional
    table layouts and summary displays, similar to Keras model.summary().
    """
    
    @staticmethod
    def format_summary(layer_info, param_counts, memory_usage, configuration, model_name="Multi_Layer_ANN"):
        """
        Format complete model summary for display.
        
        Args:
            layer_info (list): List of layer information dictionaries
            param_counts (dict): Parameter count information
            memory_usage (dict): Memory usage estimates
            configuration (dict): Model configuration
            model_name (str): Name of the model class
            
        Returns:
            str: Formatted summary string
        """
        lines = []
        
        # Header
        lines.append("=" * 100)
        lines.append(f"{f'Model: {model_name}':^100}")
        lines.append("=" * 100)
        
        # Table header
        if any('details' in layer for layer in layer_info):
            # CNN format with details column
            lines.append(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15} {'Details':<15}")
        else:
            # ANN format with activation and initialization columns
            lines.append(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<12} {'Activation':<15} {'Init Method':<20}")
        
        lines.append("=" * 100)
        
        # Layer information
        for layer in layer_info:
            layer_name = layer['name']
            output_shape = str(layer['output_shape'])
            param_count = f"{layer['params']:,}" if layer['params'] > 0 else "0"
            
            if 'details' in layer:
                # CNN format
                details = layer['details'] or '-'
                lines.append(f"{layer_name:<25} {output_shape:<20} {param_count:<15} {details:<15}")
            else:
                # ANN format with initialization method
                activation = layer['activation'] or '-'
                init_method = layer.get('init_method', None) or '-'
                lines.append(f"{layer_name:<25} {output_shape:<20} {param_count:<12} {activation:<15} {init_method:<20}")
        
        lines.append("=" * 100)
        
        # Parameter summary
        lines.append(f"Total params: {param_counts['total_params']:,}")
        lines.append(f"Trainable params: {param_counts['trainable_params']:,}")
        lines.append(f"Non-trainable params: {param_counts['non_trainable_params']:,}")
        
        # Memory usage
        lines.append("_" * 100)
        lines.append("Memory usage:")
        lines.append(f"  Parameters: ~{memory_usage['parameters_mb']:.2f} MB")
        lines.append(f"  Activations (est.): ~{memory_usage['activations_mb']:.2f} MB")
        lines.append(f"  Total (training): ~{memory_usage['total_training_mb']:.2f} MB")
        lines.append(f"  Total (inference): ~{memory_usage['total_inference_mb']:.2f} MB")
        
        # Model configuration
        lines.append("_" * 100)
        lines.append("Model Configuration:")
        lines.append(f"  Loss function: {configuration['loss_function']}")
        lines.append(f"  L2 regularization: {configuration['l2_regularization']}")
        lines.append(f"  Dropout rate: {configuration['dropout_rate']}")
        lines.append(f"  Optimizer: {configuration['optimizer']}")
        lines.append(f"  Device: {configuration['device']}")
        
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_model_info(layer_info, param_counts, memory_usage, configuration):
        """
        Format model information as a structured dictionary.
        
        Args:
            layer_info (list): List of layer information dictionaries
            param_counts (dict): Parameter count information
            memory_usage (dict): Memory usage estimates
            configuration (dict): Model configuration
            
        Returns:
            dict: Structured model information dictionary
        """
        return {
            'layer_info': layer_info,
            'total_params': param_counts['total_params'],
            'trainable_params': param_counts['trainable_params'],
            'non_trainable_params': param_counts['non_trainable_params'],
            'memory_usage': memory_usage,
            'configuration': configuration
        }


def create_introspector(model):
    """
    Factory function to create appropriate introspector for a model.
    
    Args:
        model: Model instance (Multi_Layer_ANN or Multi_Layer_CNN)
        
    Returns:
        BaseModelIntrospector: Appropriate introspector instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_class_name = type(model).__name__
    
    if model_class_name == 'Multi_Layer_ANN':
        return ANNIntrospector(model)
    elif model_class_name == 'Multi_Layer_CNN':
        return CNNIntrospector(model)
    else:
        raise ValueError(f"Unsupported model type: {model_class_name}")