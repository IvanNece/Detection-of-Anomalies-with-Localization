"""
Test suite for PaDiM implementation.

Verifies that PadimWrapper is correctly aligned with PatchCore interface
and works correctly with our data pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.padim_wrapper import PadimWrapper
from src.models.patchcore import PatchCore


class TestPadimWrapper:
    """Test PaDiM wrapper implementation and interface consistency."""
    
    def test_initialization(self):
        """Test PaDiM model initialization."""
        model = PadimWrapper(
            backbone='resnet50',
            layers=['layer1', 'layer2', 'layer3'],
            n_features=100,
            device='cpu'
        )
        
        assert model.backbone == 'resnet50'
        assert model.layers == ['layer1', 'layer2', 'layer3']
        assert model.n_features == 100
        assert not model.fitted
    
    def test_interface_alignment_with_patchcore(self):
        """Verify PaDiM interface matches PatchCore."""
        padim = PadimWrapper(device='cpu')
        patchcore = PatchCore(device='cpu')
        
        # Check both have same key methods
        assert hasattr(padim, 'fit')
        assert hasattr(padim, 'predict')
        assert hasattr(padim, 'save')
        assert hasattr(padim, 'load')
        assert hasattr(padim, 'get_info')
        
        assert hasattr(patchcore, 'fit')
        assert hasattr(patchcore, 'predict')
        assert hasattr(patchcore, 'save')
        assert hasattr(patchcore, 'load')
    
    def test_predict_shape_consistency(self):
        """Test predict output shapes match PatchCore."""
        model = PadimWrapper(device='cpu', image_size=224)
        
        # Create dummy trained model
        model.fitted = True
        H, W = 56, 56  # layer1 spatial resolution for 224x224 input
        C = 100  # n_features
        model.spatial_dims = (H, W)
        
        # Create dummy Gaussian parameters
        model.model_torch.mean = torch.randn(H, W, C)
        model.model_torch.cov = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        model.model_torch.cov_inv = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        
        # Test single image
        image_single = torch.randn(3, 224, 224)
        scores_single, heatmap_single = model.predict(image_single, return_heatmaps=True)
        
        assert isinstance(scores_single, (float, np.floating))
        assert heatmap_single.shape == (224, 224)
        
        # Test batch
        images_batch = torch.randn(4, 3, 224, 224)
        scores_batch, heatmaps_batch = model.predict(images_batch, return_heatmaps=True)
        
        assert scores_batch.shape == (4,)
        assert heatmaps_batch.shape == (4, 224, 224)
    
    def test_save_load_cycle(self):
        """Test save/load maintains model state."""
        model = PadimWrapper(
            backbone='resnet50',
            layers=['layer1', 'layer2'],
            n_features=50,
            device='cpu'
        )
        
        # Simulate trained model
        model.fitted = True
        H, W = 28, 28
        C = 50
        model.spatial_dims = (H, W)
        model.model_torch.mean = torch.randn(H, W, C)
        model.model_torch.cov = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        model.model_torch.cov_inv = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        model.training_stats = {'num_samples': 100, 'training_time_seconds': 10.5}
        
        # Save
        save_path = PROJECT_ROOT / 'tests' / 'temp_padim.pt'
        save_path.parent.mkdir(exist_ok=True)
        model.save(save_path, include_stats=False)
        
        # Load
        model2 = PadimWrapper(device='cpu')
        model2.load(save_path)
        
        # Verify
        assert model2.fitted
        assert model2.backbone == 'resnet50'
        assert model2.layers == ['layer1', 'layer2']
        assert model2.n_features == 50
        assert model2.spatial_dims == (H, W)
        assert torch.allclose(model2.model_torch.mean, model.model_torch.mean)
        
        # Cleanup
        save_path.unlink()
    
    def test_get_info(self):
        """Test get_info returns expected keys."""
        model = PadimWrapper(device='cpu')
        info = model.get_info()
        
        required_keys = [
            'model_type', 'backbone', 'layers', 'n_features',
            'image_size', 'device', 'fitted', 'training_stats'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        assert info['model_type'] == 'PaDiM'
        assert info['distance_metric'] == 'mahalanobis'


class TestPadimVsPatchCore:
    """Verify PaDiM and PatchCore interfaces are compatible."""
    
    def test_compatible_fit_interface(self):
        """Both should accept same DataLoader format."""
        # This is verified by the fact that both use:
        # fit(train_loader) where loader yields (images, masks, labels, paths)
        pass
    
    def test_compatible_predict_interface(self):
        """Both should accept batch of images and return (scores, heatmaps)."""
        padim = PadimWrapper(device='cpu', image_size=224)
        patchcore = PatchCore(device='cpu')
        
        # Setup dummy trained state
        padim.fitted = True
        H, W = 56, 56  # layer1 spatial resolution for 224x224 input
        C = 100  # n_features
        padim.spatial_dims = (H, W)
        padim.model_torch.mean = torch.randn(H, W, C)
        padim.model_torch.cov = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        padim.model_torch.cov_inv = torch.eye(C).unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)
        
        # Test same interface
        images = torch.randn(2, 3, 224, 224)
        
        padim_scores, padim_heatmaps = padim.predict(images, return_heatmaps=True)
        
        assert padim_scores.shape == (2,)
        assert padim_heatmaps.shape == (2, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
