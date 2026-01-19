"""
Module B: Unsupervised Concept Discovery

Discovers what the network naturally encodes by clustering activations
and characterizing clusters by game state statistics.

Key principle: We do NOT define concepts a priori. We cluster first,
then ask "what game states map to each cluster?"

Theory:
    If the network has learned to distinguish game states, activations
    from different states should cluster separately—even without labels.

Literature:
    - ACE: Automatic Concept-based Explanations (Ghorbani et al., ICML 2019)
    - CRAFT: Concept Recursive Activation FacTorization (Fel et al., CVPR 2023)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from .ground_truth import RolloutData


@dataclass
class ClusterProfile:
    """Statistical profile of a single cluster."""
    cluster_id: int
    n_samples: int
    fraction: float                          # Fraction of total samples

    # Game state statistics (mean ± std for each feature)
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]

    # Action distribution within cluster
    action_distribution: Dict[int, float]    # action → probability
    dominant_action: int
    action_entropy: float                    # Diversity of actions in cluster

    # Outcome statistics (if available)
    mean_reward: float

    def get_summary(self) -> str:
        """Get human-readable cluster summary."""
        lines = [f"Cluster {self.cluster_id} ({self.n_samples} samples, {self.fraction:.1%})"]
        lines.append("  Game State:")
        for feat, mean in sorted(self.feature_means.items()):
            std = self.feature_stds[feat]
            lines.append(f"    {feat}: {mean:.1f} ± {std:.1f}")
        lines.append(f"  Dominant action: {self.dominant_action} ({self.action_distribution.get(self.dominant_action, 0):.1%})")
        lines.append(f"  Action entropy: {self.action_entropy:.2f} bits")
        return "\n".join(lines)


@dataclass
class ConceptDiscoveryResult:
    """Complete concept discovery results."""
    cluster_profiles: List[ClusterProfile]
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_labels: np.ndarray               # Cluster assignment for each sample
    cluster_centers: Optional[np.ndarray]    # Cluster centers (if k-means)

    # Feature importance for clustering
    feature_cluster_correlation: Dict[str, float]  # How well each feature separates clusters

    def get_profile(self, cluster_id: int) -> ClusterProfile:
        """Get profile for a specific cluster."""
        for profile in self.cluster_profiles:
            if profile.cluster_id == cluster_id:
                return profile
        raise ValueError(f"Cluster {cluster_id} not found")

    def print_summary(self):
        """Print formatted summary of all clusters."""
        print("\n" + "="*60)
        print("UNSUPERVISED CONCEPT DISCOVERY")
        print("="*60)
        print(f"Clusters: {self.n_clusters}")
        print(f"Silhouette Score: {self.silhouette_score:.3f}")
        print(f"Calinski-Harabasz Score: {self.calinski_harabasz_score:.1f}")

        print("\nFeature-Cluster Correlation (how well feature separates clusters):")
        for feat, corr in sorted(self.feature_cluster_correlation.items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {corr:.3f}")

        print("\n" + "-"*60)
        for profile in self.cluster_profiles:
            print(profile.get_summary())
            print("-"*60)


def compute_action_entropy(action_counts: Dict[int, int]) -> float:
    """Compute entropy of action distribution."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def compute_feature_cluster_correlation(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Compute how well each feature separates clusters using eta-squared.

    eta² = SS_between / SS_total
    High eta² → feature values differ significantly between clusters
    """
    correlations = {}
    n_clusters = len(np.unique(cluster_labels))

    for i, name in enumerate(feature_names):
        feature_values = features[:, i]

        # Total sum of squares
        grand_mean = np.mean(feature_values)
        ss_total = np.sum((feature_values - grand_mean) ** 2)

        if ss_total == 0:
            correlations[name] = 0.0
            continue

        # Between-group sum of squares
        ss_between = 0.0
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 0:
                cluster_mean = np.mean(feature_values[mask])
                ss_between += np.sum(mask) * (cluster_mean - grand_mean) ** 2

        eta_squared = ss_between / ss_total
        correlations[name] = eta_squared

    return correlations


class ConceptDiscoverer:
    """
    Discovers emergent concepts in network representations via clustering.

    Usage:
        discoverer = ConceptDiscoverer(n_clusters=8)
        result = discoverer.discover(rollout_data)
        result.print_summary()
    """

    def __init__(
        self,
        n_clusters: int = 8,
        method: str = 'kmeans',
        standardize: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_clusters: Number of clusters to find
            method: 'kmeans' or 'hierarchical'
            standardize: Whether to standardize activations before clustering
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.method = method
        self.standardize = standardize
        self.random_state = random_state

        self.scaler = StandardScaler() if standardize else None
        self.cluster_model = None

    def discover(self, data: RolloutData) -> ConceptDiscoveryResult:
        """
        Discover concepts by clustering activations.

        Args:
            data: RolloutData containing activations and features

        Returns:
            ConceptDiscoveryResult with cluster profiles
        """
        activations = data.activations

        # Standardize activations
        if self.standardize:
            activations_scaled = self.scaler.fit_transform(activations)
        else:
            activations_scaled = activations

        # Cluster
        if self.method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == 'hierarchical':
            self.cluster_model = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        cluster_labels = self.cluster_model.fit_predict(activations_scaled)

        # Compute cluster quality metrics
        sil_score = silhouette_score(activations_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(activations_scaled, cluster_labels)

        # Build cluster profiles
        profiles = self._build_profiles(data, cluster_labels)

        # Compute feature-cluster correlations
        feature_correlations = compute_feature_cluster_correlation(
            data.features, cluster_labels, data.feature_names
        )

        # Get cluster centers if k-means
        centers = None
        if hasattr(self.cluster_model, 'cluster_centers_'):
            centers = self.cluster_model.cluster_centers_

        return ConceptDiscoveryResult(
            cluster_profiles=profiles,
            n_clusters=self.n_clusters,
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            cluster_labels=cluster_labels,
            cluster_centers=centers,
            feature_cluster_correlation=feature_correlations,
        )

    def _build_profiles(
        self,
        data: RolloutData,
        cluster_labels: np.ndarray
    ) -> List[ClusterProfile]:
        """Build statistical profiles for each cluster."""
        profiles = []
        total_samples = len(data)

        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            n_samples = np.sum(mask)

            if n_samples == 0:
                continue

            # Feature statistics
            cluster_features = data.features[mask]
            feature_means = {}
            feature_stds = {}
            for i, name in enumerate(data.feature_names):
                feature_means[name] = float(np.mean(cluster_features[:, i]))
                feature_stds[name] = float(np.std(cluster_features[:, i]))

            # Action distribution
            cluster_actions = data.actions[mask]
            action_counts = Counter(cluster_actions)
            total_actions = sum(action_counts.values())
            action_dist = {a: c / total_actions for a, c in action_counts.items()}
            dominant_action = max(action_counts, key=action_counts.get)
            action_entropy = compute_action_entropy(action_counts)

            # Reward statistics
            cluster_rewards = data.rewards[mask]
            mean_reward = float(np.mean(cluster_rewards))

            profiles.append(ClusterProfile(
                cluster_id=cluster_id,
                n_samples=n_samples,
                fraction=n_samples / total_samples,
                feature_means=feature_means,
                feature_stds=feature_stds,
                action_distribution=action_dist,
                dominant_action=dominant_action,
                action_entropy=action_entropy,
                mean_reward=mean_reward,
            ))

        # Sort by size (largest first)
        profiles.sort(key=lambda p: p.n_samples, reverse=True)
        return profiles

    def find_optimal_k(
        self,
        data: RolloutData,
        k_range: List[int] = [3, 5, 8, 10, 15, 20],
        metric: str = 'silhouette'
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters using elbow/silhouette method.

        Args:
            data: RolloutData
            k_range: Candidate cluster counts
            metric: 'silhouette' or 'calinski_harabasz'

        Returns:
            (optimal_k, scores_dict)
        """
        activations = data.activations
        if self.standardize:
            activations = self.scaler.fit_transform(activations)

        scores = {}
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(activations)

            if metric == 'silhouette':
                scores[k] = silhouette_score(activations, labels)
            elif metric == 'calinski_harabasz':
                scores[k] = calinski_harabasz_score(activations, labels)

        optimal_k = max(scores, key=scores.get)
        return optimal_k, scores


def plot_cluster_profiles(
    result: ConceptDiscoveryResult,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize cluster profiles as heatmap.

    Args:
        result: ConceptDiscoveryResult
        feature_names: Features to include in visualization
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    n_clusters = len(result.cluster_profiles)
    n_features = len(feature_names)

    # Build matrix of normalized feature means
    matrix = np.zeros((n_clusters, n_features))
    for i, profile in enumerate(result.cluster_profiles):
        for j, feat in enumerate(feature_names):
            matrix[i, j] = profile.feature_means.get(feat, 0)

    # Normalize columns for visualization
    matrix_norm = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix_norm, aspect='auto', cmap='RdBu_r')

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {p.cluster_id} ({p.fraction:.0%})"
                        for p in result.cluster_profiles])

    plt.colorbar(im, label='Normalized feature mean')
    ax.set_title(f'Cluster Feature Profiles (silhouette={result.silhouette_score:.2f})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster profiles to {save_path}")

    return fig


def plot_cluster_action_distributions(
    result: ConceptDiscoveryResult,
    top_k_actions: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize action distributions per cluster.

    Args:
        result: ConceptDiscoveryResult
        top_k_actions: Show only top K most common actions
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    # Find most common actions across all clusters
    all_actions = Counter()
    for profile in result.cluster_profiles:
        for action, prob in profile.action_distribution.items():
            all_actions[action] += prob

    top_actions = [a for a, _ in all_actions.most_common(top_k_actions)]

    # Build matrix
    n_clusters = len(result.cluster_profiles)
    matrix = np.zeros((n_clusters, len(top_actions)))
    for i, profile in enumerate(result.cluster_profiles):
        for j, action in enumerate(top_actions):
            matrix[i, j] = profile.action_distribution.get(action, 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='Blues')

    ax.set_xticks(range(len(top_actions)))
    ax.set_xticklabels([f"A{a}" for a in top_actions])
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{p.cluster_id}" for p in result.cluster_profiles])

    ax.set_xlabel('Action')
    ax.set_ylabel('Cluster')
    ax.set_title('Action Distribution per Cluster')

    plt.colorbar(im, label='P(action | cluster)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved action distributions to {save_path}")

    return fig
