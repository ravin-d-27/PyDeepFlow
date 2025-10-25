"""
Transfer Learning Utilities for PyDeepFlow

This module provides utility functions and classes for transfer learning workflows,
including layer freezing, feature extraction, and fine-tuning strategies.
"""

import numpy as np
from pydeepflow.model import ConvLayer
from pydeepflow.device import Device
import warnings


class TransferLearningManager:
    """
    Manager class for transfer learning operations.
    
    This class provides high-level APIs for common transfer learning tasks:
    - Feature extraction (frozen backbone)
    - Fine-tuning (selective unfreezing)
    - Progressive unfreezing strategies
    
    Parameters
    ----------
    model : object
        The pretrained model (e.g., VGG16) to manage.
    
    Attributes
    ----------
    model : object
        Reference to the model being managed.
    training_history : dict
        Records of training phases and their results.
    
    Examples
    --------
    >>> from pydeepflow.pretrained import VGG16
    >>> from pydeepflow.transfer_learning import TransferLearningManager
    >>> 
    >>> vgg = VGG16(num_classes=10)
    >>> manager = TransferLearningManager(vgg)
    >>> 
    >>> # Phase 1: Feature extraction (train only classifier)
    >>> manager.setup_feature_extraction()
    >>> # ... train model ...
    >>> 
    >>> # Phase 2: Fine-tuning (unfreeze last conv block)
    >>> manager.setup_fine_tuning(num_layers=3)
    >>> # ... continue training ...
    """
    
    def __init__(self, model):
        """Initialize the transfer learning manager."""
        self.model = model
        self.training_history = {
            'phases': [],
            'frozen_counts': [],
            'learning_rates': []
        }
    
    def setup_feature_extraction(self):
        """
        Set up the model for feature extraction mode.
        
        Freezes all convolutional layers, keeping only the classifier
        layers trainable. This is the recommended first phase of transfer
        learning when you have a small dataset.
        """
        self.model.freeze_feature_layers()
        self.training_history['phases'].append('feature_extraction')
        self.training_history['frozen_counts'].append(len(self.model.frozen_layers))
        
        print("\n" + "=" * 70)
        print("PHASE: Feature Extraction Mode")
        print("=" * 70)
        print("All convolutional layers are frozen.")
        print("Only classifier (FC) layers will be trained.")
        print("Recommended learning rate: 1e-3 to 1e-2")
        print("=" * 70 + "\n")
    
    def setup_fine_tuning(self, num_layers=None, layer_names=None, 
                         learning_rate_reduction=0.1):
        """
        Set up the model for fine-tuning mode.
        
        Unfreezes specified layers (or last N layers) for fine-tuning.
        This is typically done after initial training with frozen features.
        
        Parameters
        ----------
        num_layers : int, optional
            Number of layers from the end to unfreeze.
        layer_names : list of int, optional
            Specific layer indices to unfreeze.
        learning_rate_reduction : float, optional
            Factor to multiply previous learning rate by. Default is 0.1.
            Fine-tuning typically uses a lower learning rate than feature extraction.
        
        Notes
        -----
        Common fine-tuning strategies:
        - Unfreeze last conv block (num_layers=3 for VGG16)
        - Unfreeze last 2 blocks (num_layers=6 for VGG16)
        - Unfreeze all layers (num_layers=None and layer_names=None)
        """
        self.model.unfreeze_layers(layer_names=layer_names, num_layers=num_layers)
        self.training_history['phases'].append('fine_tuning')
        self.training_history['frozen_counts'].append(len(self.model.frozen_layers))
        
        print("\n" + "=" * 70)
        print("PHASE: Fine-Tuning Mode")
        print("=" * 70)
        print(f"Unfrozen layers for fine-tuning.")
        print(f"Current frozen layer count: {len(self.model.frozen_layers)}")
        print(f"Recommended LR reduction: {learning_rate_reduction}x")
        print(f"Example: If previous LR was 1e-2, use {learning_rate_reduction * 1e-2:.0e}")
        print("=" * 70 + "\n")
    
    def progressive_unfreeze(self, stages=3):
        """
        Implement progressive unfreezing strategy.
        
        This gradually unfreezes layers in stages, which can help prevent
        catastrophic forgetting of pretrained features.
        
        Parameters
        ----------
        stages : int, optional
            Number of unfreezing stages. Default is 3.
        
        Returns
        -------
        list of dict
            List of dictionaries defining each stage:
            - 'stage': Stage number
            - 'layers_to_unfreeze': Number of layers to unfreeze
            - 'recommended_lr': Suggested learning rate multiplier
        
        Examples
        --------
        >>> manager = TransferLearningManager(vgg16)
        >>> stages = manager.progressive_unfreeze(stages=3)
        >>> 
        >>> for stage_info in stages:
        ...     print(f"Stage {stage_info['stage']}: Unfreeze {stage_info['layers_to_unfreeze']} layers")
        ...     manager.model.unfreeze_layers(num_layers=stage_info['layers_to_unfreeze'])
        ...     # Train with recommended_lr
        ...     # ...
        """
        # Count convolutional layers
        conv_layers = [i for i, layer in enumerate(self.model.layers) 
                      if isinstance(layer, ConvLayer)]
        total_conv = len(conv_layers)
        
        if stages > total_conv:
            warnings.warn(f"Requested {stages} stages but only {total_conv} conv layers. "
                         f"Using {total_conv} stages instead.")
            stages = total_conv
        
        layers_per_stage = total_conv // stages
        
        unfreeze_schedule = []
        for stage in range(1, stages + 1):
            layers_to_unfreeze = layers_per_stage * stage
            lr_multiplier = 0.1 ** (stages - stage)  # Decrease LR as we unfreeze more
            
            unfreeze_schedule.append({
                'stage': stage,
                'layers_to_unfreeze': layers_to_unfreeze,
                'recommended_lr': lr_multiplier,
                'description': f"Unfreeze last {layers_to_unfreeze} conv layers"
            })
        
        print("\n" + "=" * 70)
        print("Progressive Unfreezing Schedule")
        print("=" * 70)
        for schedule in unfreeze_schedule:
            print(f"Stage {schedule['stage']}: {schedule['description']}")
            print(f"  └─ Recommended LR multiplier: {schedule['recommended_lr']:.2e}")
        print("=" * 70 + "\n")
        
        return unfreeze_schedule
    
    def get_training_summary(self):
        """
        Get a summary of the transfer learning training history.
        
        Returns
        -------
        dict
            Dictionary with training phase information.
        """
        return {
            'total_phases': len(self.training_history['phases']),
            'phases': self.training_history['phases'],
            'frozen_layer_progression': self.training_history['frozen_counts']
        }


def freeze_layers(model, layer_indices):
    """
    Freeze specific layers by their indices.
    
    Parameters
    ----------
    model : object
        Model with a frozen_layers attribute (set of frozen layer indices).
    layer_indices : list of int
        Indices of layers to freeze.
    
    Examples
    --------
    >>> from pydeepflow.transfer_learning import freeze_layers
    >>> freeze_layers(vgg16, [0, 1, 2, 3, 4])  # Freeze first conv block
    """
    if not hasattr(model, 'frozen_layers'):
        model.frozen_layers = set()
    
    for idx in layer_indices:
        model.frozen_layers.add(idx)
    
    print(f"Froze {len(layer_indices)} layers. Total frozen: {len(model.frozen_layers)}")


def unfreeze_layers(model, layer_indices=None):
    """
    Unfreeze specific layers or all layers.
    
    Parameters
    ----------
    model : object
        Model with a frozen_layers attribute.
    layer_indices : list of int or None, optional
        Indices of layers to unfreeze. If None, unfreezes all layers.
    
    Examples
    --------
    >>> from pydeepflow.transfer_learning import unfreeze_layers
    >>> unfreeze_layers(vgg16, [10, 11, 12])  # Unfreeze specific layers
    >>> unfreeze_layers(vgg16)  # Unfreeze all layers
    """
    if not hasattr(model, 'frozen_layers'):
        print("No frozen layers found.")
        return
    
    if layer_indices is None:
        # Unfreeze all
        model.frozen_layers.clear()
        print("Unfroze all layers.")
    else:
        for idx in layer_indices:
            if idx in model.frozen_layers:
                model.frozen_layers.remove(idx)
        print(f"Unfroze {len(layer_indices)} layers. Total frozen: {len(model.frozen_layers)}")


def get_layer_info(model):
    """
    Get information about all layers in the model.
    
    Parameters
    ----------
    model : object
        Model with layers attribute.
    
    Returns
    -------
    list of dict
        List of dictionaries containing layer information:
        - 'index': Layer index
        - 'type': Layer type (ConvLayer, Dense, MaxPooling2D, etc.)
        - 'trainable': Whether layer is trainable (not frozen)
        - 'params': Number of parameters
    
    Examples
    --------
    >>> from pydeepflow.transfer_learning import get_layer_info
    >>> layer_info = get_layer_info(vgg16)
    >>> for info in layer_info:
    ...     print(f"Layer {info['index']}: {info['type']} - Trainable: {info['trainable']}")
    """
    if not hasattr(model, 'layers'):
        raise AttributeError("Model does not have 'layers' attribute")
    
    frozen = getattr(model, 'frozen_layers', set())
    layer_info = []
    
    for i, layer in enumerate(model.layers):
        info = {
            'index': i,
            'trainable': i not in frozen
        }
        
        if isinstance(layer, ConvLayer):
            info['type'] = 'ConvLayer'
            params = (layer.Fh * layer.Fw * layer.in_channels * layer.out_channels) + layer.out_channels
            info['params'] = params
        elif isinstance(layer, dict) and 'W' in layer:
            info['type'] = 'Dense'
            params = np.prod(layer['W'].shape) + np.prod(layer['b'].shape)
            info['params'] = params
        else:
            info['type'] = type(layer).__name__
            info['params'] = 0
        
        layer_info.append(info)
    
    return layer_info


def calculate_trainable_params(model):
    """
    Calculate the number of trainable parameters in the model.
    
    Parameters
    ----------
    model : object
        Model to analyze.
    
    Returns
    -------
    dict
        Dictionary with:
        - 'total': Total number of parameters
        - 'trainable': Number of trainable parameters
        - 'frozen': Number of frozen parameters
        - 'trainable_ratio': Ratio of trainable to total parameters
    
    Examples
    --------
    >>> from pydeepflow.transfer_learning import calculate_trainable_params
    >>> params = calculate_trainable_params(vgg16)
    >>> print(f"Trainable: {params['trainable']:,} / {params['total']:,}")
    """
    layer_info = get_layer_info(model)
    
    total_params = sum(info['params'] for info in layer_info)
    trainable_params = sum(info['params'] for info in layer_info if info['trainable'])
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }


def print_transfer_learning_guide():
    """
    Print a comprehensive guide for transfer learning with PyDeepFlow.
    
    This function displays best practices and recommended strategies
    for successful transfer learning.
    """
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PyDeepFlow Transfer Learning Guide                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 RECOMMENDED WORKFLOW
─────────────────────────────────────────────────────────────────────────────────

Phase 1: Feature Extraction (Recommended duration: 10-20 epochs)
  • Freeze all convolutional layers
  • Train only the classifier (FC layers)
  • Use higher learning rate (1e-3 to 1e-2)
  • Goal: Adapt classifier to your dataset
  
  ```python
  vgg = VGG16(num_classes=10, freeze_features=True)
  # Train with LR = 1e-2
  ```

Phase 2: Fine-Tuning (Recommended duration: 10-30 epochs)
  • Unfreeze last conv block or last N layers
  • Train with lower learning rate (1e-4 to 1e-3)
  • Use learning rate decay
  • Goal: Adapt high-level features to your domain
  
  ```python
  vgg.unfreeze_layers(num_layers=3)  # Unfreeze last block
  # Train with LR = 1e-3
  ```

Phase 3: Full Fine-Tuning (Optional, 5-15 epochs)
  • Unfreeze all layers
  • Train with very low learning rate (1e-5 to 1e-4)
  • Monitor for overfitting
  • Goal: Full adaptation to your specific task
  
  ```python
  vgg.unfreeze_layers()  # Unfreeze all
  # Train with LR = 1e-4
  ```

🎯 KEY RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────────

1. Dataset Size Guidelines:
   • Small (<1000 samples): Feature extraction only
   • Medium (1k-10k): Feature extraction + fine-tune last block
   • Large (>10k): Feature extraction + full fine-tuning

2. Learning Rate Strategy:
   • Start high for frozen features (1e-2)
   • Reduce by 10x when unfreezing layers (1e-3)
   • Use learning rate decay/scheduling

3. Regularization:
   • Use dropout (0.3-0.5) in FC layers
   • Apply data augmentation aggressively
   • Consider L2 regularization (1e-4 to 1e-5)

4. Batch Size:
   • Larger batches (32-64) for stable training
   • Reduce if memory constrained

5. Monitoring:
   • Watch validation loss carefully
   • Stop if validation loss increases
   • Save checkpoints frequently

⚠️  COMMON PITFALLS
─────────────────────────────────────────────────────────────────────────────────

✗ Using same learning rate for all phases
✗ Unfreezing too early (before classifier converges)
✗ Not using data augmentation
✗ Training for too many epochs (overfitting)
✗ Forgetting to reduce learning rate when unfreezing

✓ Follow the phased approach above
✓ Monitor metrics closely
✓ Use early stopping
✓ Experiment with different unfreezing strategies

╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# Export all public functions
__all__ = [
    'TransferLearningManager',
    'freeze_layers',
    'unfreeze_layers',
    'get_layer_info',
    'calculate_trainable_params',
    'print_transfer_learning_guide'
]
