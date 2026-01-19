"""
Module A: Information-Theoretic Policy Analysis

Computes mutual information I(action; feature) to determine which game state
features the policy depends on.

Theory:
    I(A; F) = H(A) - H(A|F)
            = Σ p(a,f) log[p(a,f) / (p(a)p(f))]

    High MI → policy strongly depends on that feature
    Low MI → feature has little influence on action selection

Key principle: We discretize features uniformly (no domain-specific bins).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from .ground_truth import RolloutData, discretize_feature


@dataclass
class MIResult:
    """Result of mutual information analysis for one feature."""
    feature_name: str
    mi_bits: float                    # I(action; feature) in bits
    normalized_mi: float              # MI / H(action), range [0, 1]
    action_entropy: float             # H(action) in bits
    feature_entropy: float            # H(feature) in bits
    conditional_entropy: float        # H(action | feature) in bits
    n_samples: int
    n_bins: int


@dataclass
class MIAnalysisResult:
    """Complete MI analysis results."""
    results: Dict[str, MIResult]      # feature_name → MIResult
    action_entropy: float             # H(action) for the policy
    n_samples: int
    n_action_classes: int

    def to_dataframe(self):
        """Convert to pandas DataFrame for easy viewing."""
        import pandas as pd
        rows = []
        for name, result in self.results.items():
            rows.append({
                'feature': name,
                'MI (bits)': result.mi_bits,
                'normalized_MI': result.normalized_mi,
                'H(feature)': result.feature_entropy,
            })
        df = pd.DataFrame(rows)
        return df.sort_values('MI (bits)', ascending=False)

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by MI (descending)."""
        return sorted(
            [(name, r.mi_bits) for name, r in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )


def compute_entropy(labels: np.ndarray) -> float:
    """
    Compute entropy H(X) in bits.

    H(X) = -Σ p(x) log2(p(x))
    """
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def compute_conditional_entropy(labels: np.ndarray, conditions: np.ndarray) -> float:
    """
    Compute conditional entropy H(X|Y) in bits.

    H(X|Y) = Σ p(y) H(X|Y=y)
           = -Σ p(x,y) log2(p(x|y))
    """
    unique_conditions = np.unique(conditions)
    total = len(labels)
    if total == 0:
        return 0.0

    cond_entropy = 0.0
    for cond in unique_conditions:
        mask = conditions == cond
        p_cond = np.sum(mask) / total
        labels_given_cond = labels[mask]
        h_given_cond = compute_entropy(labels_given_cond)
        cond_entropy += p_cond * h_given_cond

    return cond_entropy


def compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information I(X; Y) in bits.

    I(X; Y) = H(X) - H(X|Y)
    """
    h_x = compute_entropy(x)
    h_x_given_y = compute_conditional_entropy(x, y)
    return max(0.0, h_x - h_x_given_y)  # Clamp to handle numerical issues


class MutualInformationAnalyzer:
    """
    Analyzes which game state features the policy depends on using mutual information.

    Usage:
        analyzer = MutualInformationAnalyzer(n_bins=10)
        result = analyzer.analyze(rollout_data)
        print(result.get_ranking())
    """

    def __init__(
        self,
        n_bins: int = 10,
        discretization_method: str = 'uniform',
        min_samples: int = 100
    ):
        """
        Args:
            n_bins: Number of bins for discretizing continuous features
            discretization_method: 'uniform' or 'quantile'
            min_samples: Minimum samples required for reliable MI estimation
        """
        self.n_bins = n_bins
        self.discretization_method = discretization_method
        self.min_samples = min_samples

    def compute_mi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        discrete_x: bool = True,
        discrete_y: bool = False
    ) -> float:
        """
        Compute mutual information I(X; Y) between two arrays.
        
        This is a convenience method for runtime analysis.
        
        Args:
            x: First variable (e.g., actions)
            y: Second variable (e.g., feature values)
            discrete_x: If True, treat x as already discrete
            discrete_y: If True, treat y as already discrete
            
        Returns:
            MI value in bits
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        if len(x) < 10:
            return 0.0
            
        # Discretize if needed
        if not discrete_x:
            x = discretize_feature(x, n_bins=self.n_bins, method=self.discretization_method)
        if not discrete_y:
            y = discretize_feature(y, n_bins=self.n_bins, method=self.discretization_method)
        
        return compute_mutual_information(x, y)

    def analyze(self, data: RolloutData) -> MIAnalysisResult:
        """
        Compute MI between actions and each game state feature.

        Args:
            data: RolloutData containing activations, features, and actions

        Returns:
            MIAnalysisResult with MI for each feature
        """
        if len(data) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples, got {len(data)}")

        actions = data.actions
        n_action_classes = len(np.unique(actions))
        action_entropy = compute_entropy(actions)

        results = {}
        for feature_name in data.feature_names:
            feature_values = data.get_feature(feature_name)

            # Discretize continuous feature
            feature_bins = discretize_feature(
                feature_values,
                n_bins=self.n_bins,
                method=self.discretization_method
            )

            # Compute MI
            mi = compute_mutual_information(actions, feature_bins)
            feature_entropy = compute_entropy(feature_bins)
            conditional_entropy = compute_conditional_entropy(actions, feature_bins)

            # Normalized MI (as fraction of action entropy)
            normalized_mi = mi / action_entropy if action_entropy > 0 else 0.0

            results[feature_name] = MIResult(
                feature_name=feature_name,
                mi_bits=mi,
                normalized_mi=normalized_mi,
                action_entropy=action_entropy,
                feature_entropy=feature_entropy,
                conditional_entropy=conditional_entropy,
                n_samples=len(data),
                n_bins=self.n_bins,
            )

        return MIAnalysisResult(
            results=results,
            action_entropy=action_entropy,
            n_samples=len(data),
            n_action_classes=n_action_classes,
        )

    def analyze_with_sensitivity(
        self,
        data: RolloutData,
        bin_range: List[int] = [5, 10, 15, 20]
    ) -> Dict[int, MIAnalysisResult]:
        """
        Run analysis with different bin counts to check sensitivity.

        Args:
            data: RolloutData
            bin_range: List of bin counts to try

        Returns:
            Dict mapping n_bins → MIAnalysisResult
        """
        results = {}
        for n_bins in bin_range:
            self.n_bins = n_bins
            results[n_bins] = self.analyze(data)
        return results


def plot_mi_results(result: MIAnalysisResult, save_path: Optional[str] = None):
    """
    Create bar chart of MI values.

    Args:
        result: MIAnalysisResult to visualize
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    ranking = result.get_ranking()
    features = [r[0] for r in ranking]
    mi_values = [r[1] for r in ranking]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, mi_values, color='steelblue')

    ax.set_xlabel('Mutual Information (bits)')
    ax.set_title(f'I(action; feature) - Policy Feature Dependence\n'
                 f'H(action) = {result.action_entropy:.2f} bits, n = {result.n_samples}')
    ax.invert_yaxis()  # Highest MI at top

    # Add value labels
    for bar, val in zip(bars, mi_values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MI plot to {save_path}")

    return fig


def print_mi_summary(result: MIAnalysisResult):
    """Print formatted MI analysis summary."""
    print("\n" + "="*60)
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*60)
    print(f"Samples: {result.n_samples}")
    print(f"Action classes: {result.n_action_classes}")
    print(f"H(action): {result.action_entropy:.3f} bits")
    print("\nFeature Importance (by MI):")
    print("-"*40)

    for feature, mi in result.get_ranking():
        r = result.results[feature]
        print(f"  {feature:20s}: {mi:.3f} bits (normalized: {r.normalized_mi:.2%})")

    print("-"*40)
