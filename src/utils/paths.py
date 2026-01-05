"""
Path management utilities for the project.

"""

from pathlib import Path
from typing import Union


class ProjectPaths:
    """
    Centralized path management for the project.
    
    All paths are computed relative to the project root directory,
    ensuring portability across different environments.
    
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
        
    def get_split_path(self, domain: str = 'clean') -> Path:
        """
        Get path for split file.
        
        Args:
            domain: Domain ('clean', 'shifted')
        
        Returns:
            Path to split JSON file
        """
        return self.DATA_PROCESSED / f"{domain}_splits.json"
                    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProjectPaths(root={self.ROOT})"


# Global instance for convenience
paths = ProjectPaths()
