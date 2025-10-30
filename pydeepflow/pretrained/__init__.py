"""
Pretrained Models Module for PyDeepFlow

This module provides pretrained deep learning architectures for transfer learning,
starting with VGG16 and expandable to other architectures like ResNet, VGG19, etc.
"""

from .vgg16 import VGG16
from .vgg19 import VGG19

__all__ = ['VGG16', 'VGG19']
