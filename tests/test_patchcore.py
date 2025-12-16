"""
Unit tests for PatchCore implementation.

Tests feature extraction, memory bank, and PatchCore model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.backbones import ResNet50FeatureExtractor
from src.models.memory_bank import GreedyCoresetSubsampling, MemoryBank
from src.models.patchcore import PatchCore


class TestResNet50FeatureExtractor:
    """Test ResNet-50 feature extractor."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = ResNet50FeatureExtractor(
            layers=['layer2', 'layer3'],
            patch_size=3,
            pretrained=True
        )
        
        assert extractor.output_dim > 0
        assert len(extractor.feature_shapes) == 2
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        extractor = ResNet50FeatureExtractor()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        output = extractor(dummy_input)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == extractor.output_dim
        assert len(output.shape) == 4  # (B, C, H, W)
    
    def test_patch_features(self):
        """Test patch feature extraction."""
        extractor = ResNet50FeatureExtractor()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        patch_features, spatial_dims = extractor.get_patch_features(dummy_input)
        
        assert patch_features.ndim == 2  # (N, D)
        assert patch_features.shape[1] == extractor.output_dim
        assert len(spatial_dims) == 2  # (H, W)


class TestGreedyCoresetSubsampling:
    """Test coreset subsampling."""
    
    def test_sampling(self):
        """Test coreset sampling reduces features."""
        sampler = GreedyCoresetSubsampling(target_ratio=0.1)
        
        features = torch.randn(1000, 128)
        indices = sampler.sample(features)
        
        assert len(indices) == 100  # 10% of 1000
        assert len(set(indices)) == len(indices)  # No duplicates
    
    def test_random_projection(self):
        """Test random projection for speedup."""
        sampler = GreedyCoresetSubsampling(
            target_ratio=0.1,
            projection_dim=32
        )
        
        features = torch.randn(100, 128)
        projected = sampler._random_projection(features)
        
        assert projected.shape[1] == 32


class TestMemoryBank:
    """Test memory bank."""
    
    def test_initialization(self):
        """Test memory bank initialization."""
        features = np.random.randn(100, 128).astype(np.float32)
        bank = MemoryBank(features=features, use_faiss=False)
        
        assert len(bank) == 100
    
    def test_nearest_neighbor_search(self):
        """Test nearest neighbor search."""
        features = np.random.randn(100, 128).astype(np.float32)
        bank = MemoryBank(features=features, use_faiss=False)
        
        query = torch.randn(10, 128)
        distances, indices = bank.search_nearest(query, k=5)
        
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
    
    def test_anomaly_scoring(self):
        """Test anomaly score computation."""
        features = np.random.randn(100, 128).astype(np.float32)
        bank = MemoryBank(features=features, use_faiss=False, n_neighbors=9)
        
        query = torch.randn(10, 128)
        scores = bank.compute_anomaly_scores(query, apply_reweighting=True)
        
        assert scores.shape == (10,)
        assert np.all(scores >= 0)


class TestPatchCore:
    """Test PatchCore model."""
    
    def test_initialization(self):
        """Test PatchCore initialization."""
        model = PatchCore(
            backbone_layers=['layer2', 'layer3'],
            coreset_ratio=0.01,
            device='cpu'
        )
        
        assert model.backbone is not None
        assert model.memory_bank is None  # Not fitted yet
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict fails without fitting."""
        model = PatchCore(device='cpu')
        
        images = torch.randn(2, 3, 224, 224)
        
        with pytest.raises(AssertionError):
            model.predict(images)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
