"""
Ground Truth Feature Extraction

Extracts raw game state features from RAM without any heuristic discretization.
These are the ONLY features used for interpretability analysis.

Design Principle:
- NO arbitrary thresholds (no "close < 60 pixels")
- NO hand-crafted categories (no "spacing = close/mid/far")
- ONLY direct RAM values and simple derived quantities
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


# =============================================================================
# GROUND TRUTH FEATURE DEFINITIONS
# =============================================================================
# Each feature is a pure function: info dict → numeric value
# No discretization, no heuristics

GROUND_TRUTH_FEATURES: Dict[str, Callable[[Dict], float]] = {
    # Spatial features (raw positions)
    'agent_x': lambda info: float(info.get('agent_x', 128)),
    'enemy_x': lambda info: float(info.get('enemy_x', 128)),

    # Derived spatial (still continuous, no discretization)
    'distance': lambda info: float(abs(info.get('agent_x', 128) - info.get('enemy_x', 128))),
    'relative_position': lambda info: float(info.get('agent_x', 128) - info.get('enemy_x', 128)),  # Signed

    # Health features (raw values)
    'agent_hp': lambda info: float(info.get('agent_hp', 176)),
    'enemy_hp': lambda info: float(info.get('enemy_hp', 176)),

    # Derived health (still continuous)
    'hp_diff': lambda info: float(info.get('agent_hp', 176) - info.get('enemy_hp', 176)),
    'hp_ratio': lambda info: float(info.get('agent_hp', 176)) / max(float(info.get('agent_hp', 176) + info.get('enemy_hp', 176)), 1),
    'total_hp': lambda info: float(info.get('agent_hp', 176) + info.get('enemy_hp', 176)),

    # Temporal
    'timer': lambda info: float(info.get('round_countdown', 99)),
}

# Feature metadata for analysis
FEATURE_RANGES = {
    'agent_x': (0, 256),
    'enemy_x': (0, 256),
    'distance': (0, 256),
    'relative_position': (-256, 256),
    'agent_hp': (0, 176),
    'enemy_hp': (0, 176),
    'hp_diff': (-176, 176),
    'hp_ratio': (0, 1),
    'total_hp': (0, 352),
    'timer': (0, 99),
}


@dataclass
class GroundTruthState:
    """Container for extracted ground truth features."""
    features: Dict[str, float]

    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Convert to numpy array for analysis."""
        if feature_names is None:
            feature_names = list(GROUND_TRUTH_FEATURES.keys())
        return np.array([self.features.get(name, 0.0) for name in feature_names], dtype=np.float32)

    def __getitem__(self, key: str) -> float:
        return self.features.get(key, 0.0)


class GroundTruthExtractor:
    """
    Extracts ground truth features from game state.

    This class does NOT discretize or categorize—it only extracts raw values.
    Discretization for MI computation happens in the analysis modules.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: Which features to extract. None = all features.
        """
        self.feature_names = feature_names or list(GROUND_TRUTH_FEATURES.keys())
        self._validate_features()

    def _validate_features(self):
        """Ensure all requested features exist."""
        for name in self.feature_names:
            if name not in GROUND_TRUTH_FEATURES:
                raise ValueError(f"Unknown feature: {name}. Available: {list(GROUND_TRUTH_FEATURES.keys())}")

    def extract(self, info: Dict) -> GroundTruthState:
        """
        Extract ground truth features from game info dict.

        Args:
            info: Game state info dict from environment step

        Returns:
            GroundTruthState with all requested features
        """
        features = {}
        for name in self.feature_names:
            try:
                features[name] = GROUND_TRUTH_FEATURES[name](info)
            except (KeyError, TypeError, ZeroDivisionError):
                features[name] = 0.0
        return GroundTruthState(features=features)

    def extract_from_info(self, info: Dict) -> Dict[str, float]:
        """
        Extract ground truth features as a simple dict.

        Convenience method for runtime callbacks that need raw feature values.

        Args:
            info: Game state info dict from environment step

        Returns:
            Dict mapping feature names to float values
        """
        return self.extract(info).features

    def extract_batch(self, infos: List[Dict]) -> np.ndarray:
        """
        Extract features from a batch of info dicts.

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        return np.array([self.extract(info).to_array(self.feature_names) for info in infos])


def discretize_feature(values: np.ndarray, n_bins: int = 10, method: str = 'uniform') -> np.ndarray:
    """
    Discretize continuous feature values into bins for MI computation.

    This is the ONLY place where discretization happens, and it's done
    uniformly without domain-specific thresholds.

    Args:
        values: Continuous feature values
        n_bins: Number of bins
        method: 'uniform' (equal width) or 'quantile' (equal frequency)

    Returns:
        Integer bin indices
    """
    if method == 'uniform':
        # Equal-width bins
        min_val, max_val = values.min(), values.max()
        if min_val == max_val:
            return np.zeros(len(values), dtype=np.int32)
        bins = np.linspace(min_val, max_val, n_bins + 1)
        return np.clip(np.digitize(values, bins[1:-1]), 0, n_bins - 1).astype(np.int32)

    elif method == 'quantile':
        # Equal-frequency bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(values, percentiles)
        # Handle duplicate bin edges
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.zeros(len(values), dtype=np.int32)
        return np.clip(np.digitize(values, bins[1:-1]), 0, len(bins) - 2).astype(np.int32)

    else:
        raise ValueError(f"Unknown discretization method: {method}")


# =============================================================================
# DATA COLLECTION UTILITIES
# =============================================================================

@dataclass
class RolloutData:
    """Container for collected rollout data."""
    activations: np.ndarray      # (n_frames, activation_dim)
    features: np.ndarray         # (n_frames, n_features)
    actions: np.ndarray          # (n_frames,)
    rewards: np.ndarray          # (n_frames,)
    episode_ids: np.ndarray      # (n_frames,) - which episode each frame belongs to
    feature_names: List[str]     # Names of features in order

    def __len__(self):
        return len(self.actions)

    def get_feature(self, name: str) -> np.ndarray:
        """Get a specific feature column by name."""
        if name not in self.feature_names:
            raise ValueError(f"Feature {name} not in data. Available: {self.feature_names}")
        idx = self.feature_names.index(name)
        return self.features[:, idx]


class RolloutCollector:
    """
    Collects activations and ground truth features during evaluation.

    Usage:
        collector = RolloutCollector()
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                activation = model.policy.features_extractor(obs_tensor)
                action = model.predict(obs)
                collector.add_frame(activation, info, action, reward, episode)
                obs, reward, done, _, info = env.step(action)

        data = collector.get_data()
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.extractor = GroundTruthExtractor(feature_names)
        self.feature_names = self.extractor.feature_names

        self._activations: List[np.ndarray] = []
        self._features: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._episode_ids: List[int] = []

    def add_frame(
        self,
        activation: np.ndarray,
        info: Dict,
        action: int,
        reward: float = 0.0,
        episode_id: int = 0
    ):
        """Add a single frame of data."""
        self._activations.append(activation.flatten())
        self._features.append(self.extractor.extract(info).to_array(self.feature_names))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._episode_ids.append(episode_id)

    def get_data(self) -> RolloutData:
        """Get collected data as RolloutData object."""
        return RolloutData(
            activations=np.array(self._activations),
            features=np.array(self._features),
            actions=np.array(self._actions),
            rewards=np.array(self._rewards),
            episode_ids=np.array(self._episode_ids),
            feature_names=self.feature_names,
        )

    def clear(self):
        """Clear all collected data."""
        self._activations.clear()
        self._features.clear()
        self._actions.clear()
        self._rewards.clear()
        self._episode_ids.clear()

    def __len__(self):
        return len(self._actions)
