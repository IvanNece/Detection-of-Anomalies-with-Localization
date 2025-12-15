"""
Models package initialization.

This module provides wrappers for anomaly detection models used in the project.

NOTE: When PatchCore implementation is merged from the PatchCore branch,
      add PatchCoreWrapper to imports below.
"""

from src.models.padim_wrapper import PadimWrapper

__all__ = ['PadimWrapper']
