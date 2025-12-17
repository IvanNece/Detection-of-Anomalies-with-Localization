"""
Evaluation package for anomaly detection.

This package provides evaluation tools including:
    - Evaluator: Single-class evaluation
    - MultiClassEvaluator: Multi-class evaluation with aggregation
    - evaluate_model_on_dataloader: Convenience function for model evaluation
"""

from .evaluator import (
    Evaluator,
    MultiClassEvaluator,
    evaluate_model_on_dataloader
)

__all__ = [
    'Evaluator',
    'MultiClassEvaluator',  
    'evaluate_model_on_dataloader'
]
