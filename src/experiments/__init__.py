"""Experiment-specific analysis frameworks for different research contexts."""

from .base import ExperimentFramework
from .kidney_healing import KidneyHealingExperiment

__all__ = [
    'ExperimentFramework',
    'KidneyHealingExperiment'
]