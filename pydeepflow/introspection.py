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
    # This is the original, correct implementation for ANNs.
    def __init__(self, model):
        self.model = model

    def get_layer_info(self):
        layer_info = []
        layer_info.append({
            'name': 'Input', 'type': 'Input', 'input_shape': (None, self.model.layers[0]),
            'output_shape': (None, self.model.layers[0]), 'params': 0,
            'activation': None, 'init_method': None
        })
        for i in range(len(self.model.layers) - 2):
            layer_params = (self.model.layers[i] + 1) * self.model.layers[i+1]
            init_method = self.model.init_metadata[i].method if hasattr(self.model, 'init_metadata') and i < len(self.model.init_metadata) else None
            layer_info.append({
                'name': f'Dense_{i+1}', 'type': 'Dense', 'input_shape': (None, self.model.layers[i]),
                'output_shape': (None, self.model.layers[i+1]), 'params': layer_params,
                'activation': self.model.activations[i], 'init_method': init_method
            })
        output_params = (self.model.layers[-2] + 1) * self.model.layers[-1]
        output_layer_idx = len(self.model.layers) - 2
        init_method = self.model.init_metadata[output_layer_idx].method if hasattr(self.model, 'init_metadata') and output_layer_idx < len(self.model.init_metadata) else None
        layer_info.append({
            'name': f'Dense_{len(self.model.layers)-1}', 'type': 'Dense (Output)',
            'input_shape': (None, self.model.layers[-2]), 'output_shape': (None, self.model.layers[-1]),
            'params': output_params, 'activation': self.model.output_activation, 'init_method': init_method
        })
        return layer_info

    def calculate_parameters(self):
        total_params = sum((self.model.layers[i] + 1) * self.model.layers[i+1] for i in range(len(self.model.layers) - 1))
        return {'total_params': int(total_params), 'trainable_params': int(total_params), 'non_trainable_params': 0}

    def estimate_memory_usage(self, batch_size=32):
        param_counts = self.calculate_parameters()
        param_memory_mb = (param_counts['total_params'] * 4) / (1024 * 1024)
        actual_batch_size = getattr(self.model, 'batch_size', batch_size)
        max_layer_size = max(self.model.layers)
        activation_memory_mb = (max_layer_size * actual_batch_size * 4) / (1024 * 1024)
        return {'parameters_mb': param_memory_mb, 'activations_mb': activation_memory_mb, 'total_training_mb': param_memory_mb + activation_memory_mb, 'total_inference_mb': param_memory_mb}

    def get_model_configuration(self):
        optimizer_name = "SGD"
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_name = type(self.model.optimizer).__name__
        device_type = "GPU" if self.model.device.use_gpu else "CPU"
        config = {'loss_function': self.model.loss, 'l2_regularization': self.model.regularization.l2_lambda, 'dropout_rate': self.model.regularization.dropout_rate, 'optimizer': optimizer_name, 'device': device_type}
        config['batch_size'] = getattr(self.model, 'batch_size', 'Not set')
        if hasattr(self.model, 'init_metadata'):
            config['initialization_metadata'] = [{'layer_index': m.layer_index, 'layer_type': m.layer_type, 'method': m.method, 'activation': m.activation, 'shape': m.shape, 'bias_value': m.bias_value, 'fan_in': m.fan_in, 'fan_out': m.fan_out, 'scale': m.scale} for m in self.model.init_metadata]
        return config


class CNNIntrospector(BaseModelIntrospector):
    def __init__(self, model):
        self.model = model

    def get_layer_info(self):
        from pydeepflow.model import ConvLayer, Flatten, MaxPooling2D, AveragePooling2D
        layer_info = []
        current_shape = self.model.X_train.shape[1:]
        layer_info.append({'name': 'Input', 'type': 'Input', 'input_shape': (None,) + current_shape, 'output_shape': (None,) + current_shape, 'params': 0, 'activation': None, 'details': '-'})

        layer_counts = {}
        for i, layer in enumerate(self.model.layers_list):
            layer_class_name = layer.__class__.__name__ if not isinstance(layer, dict) else 'Dense'
            layer_counts[layer_class_name] = layer_counts.get(layer_class_name, 0) + 1
            layer_name = f"{layer_class_name.replace('2D', '')}_{layer_counts[layer_class_name]}"
            input_shape, params, activation, details = (None,) + current_shape, 0, None, '-'

            if isinstance(layer, ConvLayer):
                layer_type_name = 'Conv2D'
                params = np.prod(layer.params['W'].shape) + np.prod(layer.params['b'].shape)
                H, W, _ = current_shape
                k, s, p = layer.Fh, layer.stride, layer.padding
                out_h, out_w = (H + 2*p - k)//s + 1, (W + 2*p - k)//s + 1
                current_shape = (out_h, out_w, layer.out_channels)
                details = f"k={k}, s={s}, p={p}"
                activation = getattr(layer, 'activation', None)
            elif isinstance(layer, (MaxPooling2D, AveragePooling2D)):
                layer_type_name = layer.__class__.__name__
                H, W, C = current_shape
                pool_h, pool_w, s = layer.pool_height, layer.pool_width, layer.stride
                out_h, out_w = (H - pool_h)//s + 1, (W - pool_w)//s + 1
                current_shape = (out_h, out_w, C)
                details = f"pool=({pool_h},{pool_w}), s={s}"
            elif isinstance(layer, Flatten):
                layer_type_name = 'Flatten'
                current_shape = (int(np.prod(current_shape)),)
                details = "Flatten operation"
            elif isinstance(layer, dict):
                layer_type_name = 'Dense (Output)' if i == len(self.model.layers_list) - 1 else 'Dense'
                params = np.prod(layer['W'].shape) + np.prod(layer['b'].shape)
                current_shape = (layer['W'].shape[1],)
                activation = layer.get('activation')
                details = f"activation={activation}"
            else:
                layer_type_name = "Unknown"

            layer_info.append({'name': layer_name, 'type': layer_type_name, 'input_shape': input_shape, 'output_shape': (None,) + current_shape, 'params': int(params), 'activation': activation, 'details': details})
        return layer_info

    def calculate_parameters(self):
        total_params = sum(np.prod(p.shape) for p in self.model.trainable_params)
        return {'total_params': int(total_params), 'trainable_params': int(total_params), 'non_trainable_params': 0}

    def estimate_memory_usage(self, batch_size=32):
        param_counts = self.calculate_parameters()
        param_memory_mb = (param_counts['total_params'] * 4) / (1024 * 1024)
        activation_memory_mb = (np.prod(self.model.X_train.shape[1:]) * batch_size * 4) / (1024 * 1024)
        return {'parameters_mb': param_memory_mb, 'activations_mb': activation_memory_mb, 'total_training_mb': param_memory_mb + activation_memory_mb, 'total_inference_mb': param_memory_mb}

    def get_model_configuration(self):
        optimizer_name = type(self.model.optimizer).__name__ if hasattr(self.model, 'optimizer') and self.model.optimizer is not None else "SGD"
        device_type = "GPU" if self.model.device.use_gpu else "CPU"
        return {'loss_function': self.model.loss, 'l2_regularization': self.model.regularization.l2_lambda, 'dropout_rate': self.model.regularization.dropout_rate, 'optimizer': optimizer_name, 'device': device_type, 'batch_size': getattr(self.model, 'batch_size', 'Not set')}

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