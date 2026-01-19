"""Experiment framework for systematic evaluation."""
from .config import (
    ExperimentConfig,
    generate_experiment_matrix,
    TrainingParadigm,
    MATCHUPS_VS_CPU,
    MATCHUPS_2P,
    PAPER_EXPERIMENTS,
    SEEDS,
    TRAINING_CONFIGS,
    RL_METRICS,
    INTERP_METRICS,
    GROUND_TRUTH_FEATURES,
)

__all__ = [
    'ExperimentConfig',
    'generate_experiment_matrix',
    'TrainingParadigm',
    'MATCHUPS_VS_CPU',
    'MATCHUPS_2P',
    'PAPER_EXPERIMENTS',
    'SEEDS',
    'TRAINING_CONFIGS',
    'RL_METRICS',
    'INTERP_METRICS',
    'GROUND_TRUTH_FEATURES',
]

