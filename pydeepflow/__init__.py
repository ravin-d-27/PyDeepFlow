from .activations import activation, activation_derivative
from .losses import get_loss_function, get_loss_derivative
from .device import Device
from .model import Multi_Layer_ANN, Plotting_Utils
from .learning_rate_scheduler import LearningRateScheduler
from .checkpoints import ModelCheckpoint
from .regularization import Regularization
from .early_stopping import EarlyStopping
from .cross_validator import CrossValidator
from .batch_normalization import BatchNormalization
from .gridSearch import GridSearchCV
from .validation import ModelValidator
from .introspection import ANNIntrospector, CNNIntrospector, ModelSummaryFormatter, create_introspector
from .preprocessing import ImageDataGenerator

# Try to import optional components
try:
    from .weight_initialization import get_weight_initializer
    _has_weight_init = True
except ImportError:
    _has_weight_init = False

try:
    from .model import ConvLayer, Flatten, Multi_Layer_CNN
    _has_cnn = True
except ImportError:
    _has_cnn = False

# Try to import pretrained models and transfer learning utilities
try:
    from .pretrained import VGG16
    from .transfer_learning import (
        TransferLearningManager,
        freeze_layers,
        unfreeze_layers,
        get_layer_info,
        calculate_trainable_params,
        print_transfer_learning_guide
    )
    _has_pretrained = True
except ImportError:
    _has_pretrained = False

__all__ = [
    "activation",
    "activation_derivative",
    "get_loss_function",
    "get_loss_derivative",
    "Device",
    "Multi_Layer_ANN",
    "Plotting_Utils",
    "LearningRateScheduler",
    "ModelCheckpoint",
    "Regularization",
    "EarlyStopping",
    "CrossValidator",
    "BatchNormalization",
    "GridSearchCV",
    "ModelValidator",
    "ANNIntrospector", 
    "CNNIntrospector",
    "ModelSummaryFormatter",
    "create_introspector",
    "ImageDataGenerator",
]

# Add optional components to __all__ if available
if _has_weight_init:
    __all__.append("get_weight_initializer")

if _has_cnn:
    __all__.extend(["ConvLayer", "Flatten", "Multi_Layer_CNN"])

if _has_pretrained:
    __all__.extend([
        "VGG16",
        "TransferLearningManager",
        "freeze_layers",
        "unfreeze_layers",
        "get_layer_info",
        "calculate_trainable_params",
        "print_transfer_learning_guide"
    ])
