"""
VGG16 Architecture Implementation for PyDeepFlow

This module implements the VGG16 deep convolutional neural network architecture
for transfer learning and feature extraction tasks.

VGG16 Architecture:
- Input: 224x224x3 RGB images
- 5 convolutional blocks with max pooling
- 3 fully connected layers
- Total: 13 conv layers + 3 FC layers = 16 layers with learnable parameters

Reference:
Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for 
Large-Scale Image Recognition. arXiv:1409.1556
"""

import numpy as np
from pydeepflow.model import ConvLayer, MaxPooling2D, Flatten, Multi_Layer_CNN
from pydeepflow.device import Device
from pydeepflow.weight_initialization import WeightInitializer
from pydeepflow.activations import activation
import warnings


class VGG16:
    """
    VGG16 Convolutional Neural Network for Transfer Learning.
    
    This class implements the VGG16 architecture, which consists of:
    - Block 1: 2 conv layers (64 filters) + max pool
    - Block 2: 2 conv layers (128 filters) + max pool
    - Block 3: 3 conv layers (256 filters) + max pool
    - Block 4: 3 conv layers (512 filters) + max pool
    - Block 5: 3 conv layers (512 filters) + max pool
    - Flatten layer
    - FC layer 1: 4096 neurons
    - FC layer 2: 4096 neurons
    - FC layer 3: num_classes neurons (output)
    
    All convolutional layers use:
    - 3x3 kernels
    - Stride of 1
    - Padding of 1 (to preserve spatial dimensions)
    - ReLU activation
    
    All max pooling layers use:
    - 2x2 window
    - Stride of 2
    
    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for classification. Default is 1000 (ImageNet).
    input_shape : tuple, optional
        Input image shape (height, width, channels). Default is (224, 224, 3).
    use_gpu : bool, optional
        Whether to use GPU acceleration. Default is False.
    include_top : bool, optional
        Whether to include the fully connected layers at the top. Default is True.
        Set to False for feature extraction.
    weights : str or None, optional
        Path to pretrained weights file or None for random initialization.
        Default is None.
    freeze_features : bool, optional
        If True, freeze convolutional layers for feature extraction mode.
        Default is False.
    
    Attributes
    ----------
    device : Device
        Device object for CPU/GPU operations.
    num_classes : int
        Number of output classes.
    input_shape : tuple
        Expected input shape.
    include_top : bool
        Whether fully connected layers are included.
    layers : list
        List of all layers in the network.
    feature_layers : list
        List of convolutional/pooling layers only.
    classifier_layers : list
        List of fully connected layers only.
    frozen_layers : set
        Set of layer indices that are frozen (not trainable).
    
    Examples
    --------
    Create VGG16 for transfer learning on a custom dataset:
    
    >>> from pydeepflow.pretrained import VGG16
    >>> 
    >>> # For feature extraction (freeze conv layers, train only FC layers)
    >>> vgg = VGG16(num_classes=10, freeze_features=True)
    >>> 
    >>> # For fine-tuning (train all layers)
    >>> vgg = VGG16(num_classes=10, freeze_features=False)
    >>> 
    >>> # For feature extraction without classification head
    >>> vgg_features = VGG16(include_top=False)
    >>> features = vgg_features.predict(X)  # Extract features
    """
    
    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), 
                 use_gpu=False, include_top=True, weights=None, 
                 freeze_features=False):
        """Initialize VGG16 architecture."""
        self.device = Device(use_gpu=use_gpu)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.include_top = include_top
        self.frozen_layers = set()
        
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D (H, W, C), got {input_shape}")
        
        if input_shape[2] != 3:
            warnings.warn(f"VGG16 was designed for RGB images (3 channels), "
                         f"but got {input_shape[2]} channels. This may affect performance.")
        
        # Build the architecture
        self.layers = []
        self.feature_layers = []
        self.classifier_layers = []
        
        self._build_architecture()
        
        # Load pretrained weights if provided
        if weights is not None:
            self.load_weights(weights)
        
        # Freeze feature layers if requested
        if freeze_features:
            self.freeze_feature_layers()
    
    def _build_architecture(self):
        """
        Build the complete VGG16 architecture.
        
        This method constructs all layers of VGG16 according to the original paper.
        """
        # Track current spatial dimensions
        H, W, C = self.input_shape
        
        # ====================================================================
        # BLOCK 1: 2x Conv(64) + MaxPool
        # ====================================================================
        # Conv1_1: 3x3x3 -> 3x3x64
        conv1_1 = ConvLayer(
            in_channels=C,
            out_channels=64, # number of filters initially
            kernel_size=3,  # filter of 3x3 will be iterated over image
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv1_1)
        self.feature_layers.append(conv1_1)
        C = 64  # Update channel count
        
        # Conv1_2: 3x3x64 -> 3x3x64
        conv1_2 = ConvLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv1_2)
        self.feature_layers.append(conv1_2)
        
        # MaxPool1: 2x2, stride 2
        pool1 = MaxPooling2D(pool_size=(2, 2), stride=2)
        self.layers.append(pool1)
        self.feature_layers.append(pool1)
        H, W = H // 2, W // 2  # Spatial dimensions halved
        
        # ====================================================================
        # BLOCK 2: 2x Conv(128) + MaxPool
        # ====================================================================
        # Conv2_1: 3x3x64 -> 3x3x128
        conv2_1 = ConvLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv2_1)
        self.feature_layers.append(conv2_1)
        C = 128
        
        # Conv2_2: 3x3x128 -> 3x3x128
        conv2_2 = ConvLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv2_2)
        self.feature_layers.append(conv2_2)
        
        # MaxPool2: 2x2, stride 2
        pool2 = MaxPooling2D(pool_size=(2, 2), stride=2)
        self.layers.append(pool2)
        self.feature_layers.append(pool2)
        H, W = H // 2, W // 2
        
        # ====================================================================
        # BLOCK 3: 3x Conv(256) + MaxPool
        # ====================================================================
        # Conv3_1: 3x3x128 -> 3x3x256
        conv3_1 = ConvLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv3_1)
        self.feature_layers.append(conv3_1)
        C = 256
        
        # Conv3_2: 3x3x256 -> 3x3x256
        conv3_2 = ConvLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv3_2)
        self.feature_layers.append(conv3_2)
        
        # Conv3_3: 3x3x256 -> 3x3x256
        conv3_3 = ConvLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv3_3)
        self.feature_layers.append(conv3_3)
        
        # MaxPool3: 2x2, stride 2
        pool3 = MaxPooling2D(pool_size=(2, 2), stride=2)
        self.layers.append(pool3)
        self.feature_layers.append(pool3)
        H, W = H // 2, W // 2
        
        # ====================================================================
        # BLOCK 4: 3x Conv(512) + MaxPool
        # ====================================================================
        # Conv4_1: 3x3x256 -> 3x3x512
        conv4_1 = ConvLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv4_1)
        self.feature_layers.append(conv4_1)
        C = 512
        
        # Conv4_2: 3x3x512 -> 3x3x512
        conv4_2 = ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv4_2)
        self.feature_layers.append(conv4_2)
        
        # Conv4_3: 3x3x512 -> 3x3x512
        conv4_3 = ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv4_3)
        self.feature_layers.append(conv4_3)
        
        # MaxPool4: 2x2, stride 2
        pool4 = MaxPooling2D(pool_size=(2, 2), stride=2)
        self.layers.append(pool4)
        self.feature_layers.append(pool4)
        H, W = H // 2, W // 2
        
        # ====================================================================
        # BLOCK 5: 3x Conv(512) + MaxPool
        # ====================================================================
        # Conv5_1: 3x3x512 -> 3x3x512
        conv5_1 = ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv5_1)
        self.feature_layers.append(conv5_1)
        
        # Conv5_2: 3x3x512 -> 3x3x512
        conv5_2 = ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv5_2)
        self.feature_layers.append(conv5_2)
        
        # Conv5_3: 3x3x512 -> 3x3x512
        conv5_3 = ConvLayer(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
            activation='relu',
            weight_init='he_normal'
        )
        self.layers.append(conv5_3)
        self.feature_layers.append(conv5_3)
        
        # MaxPool5: 2x2, stride 2
        pool5 = MaxPooling2D(pool_size=(2, 2), stride=2)
        self.layers.append(pool5)
        self.feature_layers.append(pool5)
        H, W = H // 2, W // 2
        
        # ====================================================================
        # FULLY CONNECTED LAYERS (Classifier)
        # ====================================================================
        if self.include_top:
            # Flatten layer
            flatten = Flatten()
            self.layers.append(flatten)
            
            # Calculate flattened size
            flattened_size = H * W * 512
            
            # FC1: 4096 neurons with ReLU
            initializer = WeightInitializer(
                device=self.device,
                mode='auto',
                bias_init='auto'
            )
            fc1_w, fc1_b, _ = initializer.initialize_dense_layer(
                input_dim=flattened_size,
                output_dim=4096,
                activation='relu'
            )
            fc1 = {
                'W': self.device.array(fc1_w),
                'b': self.device.array(fc1_b.reshape(1, -1)),
                'activation': 'relu'
            }
            self.layers.append(fc1)
            self.classifier_layers.append(fc1)
            
            # FC2: 4096 neurons with ReLU
            fc2_w, fc2_b, _ = initializer.initialize_dense_layer(
                input_dim=4096,
                output_dim=4096,
                activation='relu'
            )
            fc2 = {
                'W': self.device.array(fc2_w),
                'b': self.device.array(fc2_b.reshape(1, -1)),
                'activation': 'relu'
            }
            self.layers.append(fc2)
            self.classifier_layers.append(fc2)
            
            # FC3 (Output): num_classes neurons with softmax
            output_activation = 'softmax' if self.num_classes > 1 else 'sigmoid'
            fc3_w, fc3_b, _ = initializer.initialize_dense_layer(
                input_dim=4096,
                output_dim=self.num_classes,
                activation=output_activation
            )
            fc3 = {
                'W': self.device.array(fc3_w),
                'b': self.device.array(fc3_b.reshape(1, -1)),
                'activation': output_activation
            }
            self.layers.append(fc3)
            self.classifier_layers.append(fc3)
    
    def forward(self, X, training=False):
        """
        Perform forward pass through the network.
        
        Parameters
        ----------
        X : ndarray
            Input data with shape (N, H, W, C).
        training : bool, optional
            Whether in training mode (affects dropout, batch norm if added).
            Default is False.
        
        Returns
        -------
        ndarray
            Output activations. Shape depends on include_top:
            - If include_top=True: (N, num_classes)
            - If include_top=False: (N, H', W', 512) feature maps
        """
        # Validate input shape
        if X.ndim != 4:
            raise ValueError(f"Input must be 4D (N, H, W, C), got shape {X.shape}")
        
        if X.shape[1:] != self.input_shape:
            warnings.warn(f"Input shape {X.shape[1:]} differs from expected "
                         f"{self.input_shape}. This may affect performance.")
        
        current_output = X
        
        # Pass through all layers
        for i, layer in enumerate(self.layers):
            # Skip frozen layers during training (they still forward pass though)
            if isinstance(layer, (ConvLayer, MaxPooling2D, Flatten)):
                current_output = layer.forward(current_output)
            elif isinstance(layer, dict) and 'W' in layer:
                # Dense layer
                Z = self.device.dot(current_output, layer['W']) + layer['b']
                current_output = activation(Z, layer['activation'], self.device)
        
        return current_output
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Parameters
        ----------
        X : ndarray
            Input data with shape (N, H, W, C).
        
        Returns
        -------
        ndarray
            Predictions. For classification, returns class probabilities.
        """
        return self.forward(X, training=False)
    
    def freeze_feature_layers(self):
        """
        Freeze all convolutional layers for feature extraction.
        
        This is useful for transfer learning when you want to use VGG16
        as a fixed feature extractor and only train the classifier layers.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvLayer):
                self.frozen_layers.add(i)
        
        print(f"Frozen {len(self.frozen_layers)} convolutional layers for feature extraction.")
    
    def unfreeze_layers(self, layer_names=None, num_layers=None):
        """
        Unfreeze specific layers or the last N layers for fine-tuning.
        
        Parameters
        ----------
        layer_names : list of int, optional
            List of layer indices to unfreeze. If None, unfreezes based on num_layers.
        num_layers : int, optional
            Number of layers from the end to unfreeze. If None and layer_names is None,
            unfreezes all layers.
        
        Examples
        --------
        >>> # Unfreeze last 3 conv blocks for fine-tuning
        >>> vgg.unfreeze_layers(num_layers=10)
        >>> 
        >>> # Unfreeze specific layers
        >>> vgg.unfreeze_layers(layer_names=[10, 11, 12])
        >>> 
        >>> # Unfreeze all layers
        >>> vgg.unfreeze_layers()
        """
        if layer_names is not None:
            # Unfreeze specific layers
            for idx in layer_names:
                if idx in self.frozen_layers:
                    self.frozen_layers.remove(idx)
            print(f"Unfrozen layers: {layer_names}")
        elif num_layers is not None:
            # Unfreeze last N layers
            conv_indices = [i for i, layer in enumerate(self.layers) 
                           if isinstance(layer, ConvLayer)]
            to_unfreeze = conv_indices[-num_layers:] if num_layers <= len(conv_indices) else conv_indices
            for idx in to_unfreeze:
                if idx in self.frozen_layers:
                    self.frozen_layers.remove(idx)
            print(f"Unfrozen last {len(to_unfreeze)} convolutional layers.")
        else:
            # Unfreeze all layers
            self.frozen_layers.clear()
            print("Unfrozen all layers.")
    
    def get_trainable_params(self):
        """
        Get all trainable parameters (not frozen).
        
        Returns
        -------
        list
            List of parameter arrays that should be updated during training.
        """
        trainable = []
        for i, layer in enumerate(self.layers):
            if i not in self.frozen_layers:
                if isinstance(layer, ConvLayer):
                    trainable.extend([layer.params['W'], layer.params['b']])
                elif isinstance(layer, dict) and 'W' in layer:
                    trainable.extend([layer['W'], layer['b']])
        return trainable
    
    def summary(self):
        """
        Print a summary of the VGG16 architecture.
        
        Displays layer-by-layer breakdown with output shapes and parameter counts.
        """
        print("=" * 80)
        print("VGG16 Architecture Summary")
        print("=" * 80)
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Include Top (FC Layers): {self.include_top}")
        print(f"Frozen Layers: {len(self.frozen_layers)}")
        print("=" * 80)
        print(f"{'Layer':<30} {'Output Shape':<25} {'Params':<15}")
        print("-" * 80)
        
        # Track dimensions through network
        H, W, C = self.input_shape
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            frozen_mark = " [FROZEN]" if i in self.frozen_layers else ""
            
            if isinstance(layer, ConvLayer):
                # Conv layer
                out_c = layer.out_channels
                params = (layer.Fh * layer.Fw * layer.in_channels * out_c) + out_c
                output_shape = f"({H}, {W}, {out_c})"
                layer_name = f"Conv2D_{i}{frozen_mark}"
                print(f"{layer_name:<30} {output_shape:<25} {params:<15,}")
                total_params += params
                C = out_c
            elif isinstance(layer, MaxPooling2D):
                # Max pooling
                H, W = H // layer.stride, W // layer.stride
                output_shape = f"({H}, {W}, {C})"
                layer_name = f"MaxPooling2D_{i}"
                print(f"{layer_name:<30} {output_shape:<25} {'0':<15}")
            elif isinstance(layer, Flatten):
                # Flatten
                flat_size = H * W * C
                output_shape = f"({flat_size},)"
                layer_name = "Flatten"
                print(f"{layer_name:<30} {output_shape:<25} {'0':<15}")
            elif isinstance(layer, dict) and 'W' in layer:
                # Dense layer
                in_size = layer['W'].shape[0]
                out_size = layer['W'].shape[1]
                params = (in_size * out_size) + out_size
                output_shape = f"({out_size},)"
                layer_name = f"Dense_{i}{frozen_mark}"
                print(f"{layer_name:<30} {output_shape:<25} {params:<15,}")
                total_params += params
        
        print("=" * 80)
        print(f"Total Parameters: {total_params:,}")
        trainable_params = sum(
            np.prod(p.shape) for p in self.get_trainable_params()
        )
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print("=" * 80)
    
    def save_weights(self, filepath):
        """
        Save model weights to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the weights file (.npy format).
        """
        weights_dict = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvLayer):
                weights_dict[f'conv_{i}_W'] = self.device.asnumpy(layer.params['W'])
                weights_dict[f'conv_{i}_b'] = self.device.asnumpy(layer.params['b'])
            elif isinstance(layer, dict) and 'W' in layer:
                weights_dict[f'dense_{i}_W'] = self.device.asnumpy(layer['W'])
                weights_dict[f'dense_{i}_b'] = self.device.asnumpy(layer['b'])
        
        np.save(filepath, weights_dict)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the weights file (.npy format).
        """
        try:
            weights_dict = np.load(filepath, allow_pickle=True).item()
            
            for i, layer in enumerate(self.layers):
                if isinstance(layer, ConvLayer):
                    if f'conv_{i}_W' in weights_dict:
                        layer.params['W'] = self.device.array(weights_dict[f'conv_{i}_W'])
                        layer.params['b'] = self.device.array(weights_dict[f'conv_{i}_b'])
                elif isinstance(layer, dict) and 'W' in layer:
                    if f'dense_{i}_W' in weights_dict:
                        layer['W'] = self.device.array(weights_dict[f'dense_{i}_W'])
                        layer['b'] = self.device.array(weights_dict[f'dense_{i}_b'])
            
            print(f"Model weights loaded from {filepath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading weights: {str(e)}")
