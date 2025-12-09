"""
Configuration management utilities.

This module provides utilities for loading and managing configuration files,
with support for YAML and nested dictionary access.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration manager with nested attribute access.
    
    Allows accessing nested configuration keys using dot notation:
    config.dataset.classes instead of config['dataset']['classes']
    
    Example:
        >>> config = Config.load('configs/experiment_config.yaml')
        >>> print(config.seed)
        42
        >>> print(config.dataset.classes)
        ['hazelnut', 'carpet', 'zipper']
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        
        # Convert nested dicts to Config objects for dot access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Config object
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"Loaded configuration from {config_path}")
        return cls(config_dict)
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            save_path: Path where to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        
        print(f"Saved configuration to {save_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default fallback.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get('dataset.image_size', 224)
            224
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config back to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self._config
    
    def __repr__(self) -> str:
        """String representation of Config."""
        return f"Config({self._config})"
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config object
    """
    return Config.load(config_path)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries recursively.
    
    Values in override_config take precedence over base_config.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with override values
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
