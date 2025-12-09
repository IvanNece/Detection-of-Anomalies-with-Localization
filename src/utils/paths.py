"""
Path management utilities for the project.

This module provides utilities for managing file paths and directories
in a cross-platform manner.
"""

from pathlib import Path
from typing import Union


class ProjectPaths:
    """
    Centralized path management for the project.
    
    All paths are computed relative to the project root directory,
    ensuring portability across different environments.
    
    Example:
        >>> from src.utils.paths import ProjectPaths
        >>> paths = ProjectPaths()
        >>> print(paths.get_model_path('patchcore', 'hazelnut', 'clean'))
        outputs/models/patchcore_hazelnut_clean.pt
    """
    
    def __init__(self, root: Union[str, Path, None] = None):
        """
        Initialize ProjectPaths.
        
        Args:
            root: Project root directory. If None, auto-detected from this file.
        """
        if root is None:
            # Auto-detect: go up from src/utils/ to project root
            self.ROOT = Path(__file__).parent.parent.parent
        else:
            self.ROOT = Path(root)
        
        # Data directories
        self.DATA = self.ROOT / 'data'
        self.DATA_RAW = self.DATA / 'raw'
        self.DATA_PROCESSED = self.DATA / 'processed'
        self.DATA_SHIFTED = self.DATA / 'shifted'
        
        # Source code
        self.SRC = self.ROOT / 'src'
        
        # Outputs
        self.OUTPUTS = self.ROOT / 'outputs'
        self.MODELS = self.OUTPUTS / 'models'
        self.THRESHOLDS = self.OUTPUTS / 'thresholds'
        self.RESULTS = self.OUTPUTS / 'results'
        self.VISUALIZATIONS = self.OUTPUTS / 'visualizations'
        
        # Configs and scripts
        self.CONFIGS = self.ROOT / 'configs'
        self.SCRIPTS = self.ROOT / 'scripts'
        self.NOTEBOOKS = self.ROOT / 'notebooks'
    
    def get_model_path(
        self, 
        method: str, 
        class_name: str, 
        domain: str,
        extension: str = '.pt'
    ) -> Path:
        """
        Get path for a trained model file.
        
        Args:
            method: Method name ('patchcore', 'padim')
            class_name: Class name ('hazelnut', 'carpet', 'zipper')
            domain: Domain ('clean', 'shift')
            extension: File extension (default: '.pt')
        
        Returns:
            Path to model file
        
        Example:
            >>> paths.get_model_path('patchcore', 'hazelnut', 'clean')
            PosixPath('outputs/models/patchcore_hazelnut_clean.pt')
        """
        filename = f"{method}_{class_name}_{domain}{extension}"
        return self.MODELS / filename
    
    def get_split_path(self, domain: str = 'clean') -> Path:
        """
        Get path for split file.
        
        Args:
            domain: Domain ('clean', 'shift')
        
        Returns:
            Path to split JSON file
        """
        return self.DATA_PROCESSED / f"{domain}_splits.json"
    
    def get_results_path(self, experiment_name: str) -> Path:
        """
        Get path for results file.
        
        Args:
            experiment_name: Name of experiment
        
        Returns:
            Path to results JSON file
        """
        return self.RESULTS / f"{experiment_name}_results.json"
    
    def get_threshold_path(self, domain: str = 'clean') -> Path:
        """
        Get path for threshold file.
        
        Args:
            domain: Domain ('clean', 'shift')
        
        Returns:
            Path to threshold JSON file
        """
        return self.THRESHOLDS / f"{domain}_thresholds.json"
    
    def ensure_dirs(self) -> None:
        """
        Create all necessary directories if they don't exist.
        
        This should be called once at the beginning of experiments
        to ensure the directory structure is in place.
        """
        dirs = [
            self.DATA_RAW,
            self.DATA_PROCESSED,
            self.DATA_SHIFTED,
            self.MODELS,
            self.THRESHOLDS,
            self.RESULTS,
            self.VISUALIZATIONS,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("Ensured all output directories exist")
    
    def get_mvtec_class_path(self, class_name: str, domain: str = 'clean') -> Path:
        """
        Get path to MVTec AD class directory.
        
        Args:
            class_name: Class name ('hazelnut', 'carpet', 'zipper')
            domain: 'clean' for original or 'shift' for transformed
        
        Returns:
            Path to class directory
        """
        if domain == 'clean':
            return self.DATA_RAW / 'mvtec_ad' / class_name
        else:
            return self.DATA_SHIFTED / class_name
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProjectPaths(root={self.ROOT})"


# Global instance for convenience
paths = ProjectPaths()
