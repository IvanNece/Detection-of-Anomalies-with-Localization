"""
Models module for anomaly detection.

Contains implementations of PatchCore, PaDiM, and supporting components.
"""

from .backbones import ResNet50FeatureExtractor, get_resnet50_feature_extractor
from .memory_bank import MemoryBank, GreedyCoresetSubsampling
from .patchcore import PatchCore

__all__ = [
    'ResNet50FeatureExtractor',
    'get_resnet50_feature_extractor',
    'MemoryBank',
    'GreedyCoresetSubsampling',
    'PatchCore',
]
