"""
Module D: Representation Geometry Analysis

Analyzes the geometric structure of the activation space to understand
how game-relevant features are encoded.

Key questions:
1. Are states with similar features close in activation space?
2. Is there a "distance direction" in the representation?
3. How cleanly separable are different game states?

Methods:
- PCA: Find principal axes and correlate with features
- Silhouette: Measure cluster quality for feature-based groupings
- Linear separability: Can a linear classifier separate feature bins?

Literature:
- CKA: Centered Kernel Alignment (Kornblith et al., 2019)
- Representation Similarity Analysis (Kriegeskorte et al., 2008)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .ground_truth import RolloutData, discretize_feature


@dataclass
class PCAFeatureCorrelation:
    """Correlation between a PC and game features."""
    pc_index: int
    variance_explained: float
    feature_correlations: Dict[str, float]     # feature_name → Pearson r
    feature_pvalues: Dict[str, float]          # feature_name → p-value
    top_correlated_feature: str
    top_correlation: float


@dataclass
class SeparabilityResult:
    """Separability metrics for one feature."""
    feature_name: str
    n_bins: int
    silhouette_score: float                    # -1 to 1, higher = better separation
    linear_probe_accuracy: float               # Cross-validated accuracy
    linear_probe_std: float                    # Std across folds


@dataclass
class GeometryAnalysisResult:
    """Complete geometry analysis results."""
    # PCA results
    pca_correlations: List[PCAFeatureCorrelation]
    total_variance_explained: float            # By all analyzed PCs
    n_components_90_variance: int              # PCs needed for 90% variance

    # Separability results
    separability: Dict[str, SeparabilityResult]

    # Raw data for visualization
    pca_embeddings: np.ndarray                 # (n_samples, n_components)
    feature_values: Dict[str, np.ndarray]      # For coloring visualizations

    def get_pc_feature_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get PC-feature correlation matrix for heatmap visualization.

        Returns:
            (matrix, pc_labels, feature_names)
        """
        n_pcs = len(self.pca_correlations)
        feature_names = list(self.pca_correlations[0].feature_correlations.keys())
        n_features = len(feature_names)

        matrix = np.zeros((n_pcs, n_features))
        pc_labels = []

        for i, pc_corr in enumerate(self.pca_correlations):
            pc_labels.append(f"PC{pc_corr.pc_index+1} ({pc_corr.variance_explained:.1%})")
            for j, feat in enumerate(feature_names):
                matrix[i, j] = pc_corr.feature_correlations.get(feat, 0)

        return matrix, pc_labels, feature_names

    def print_summary(self):
        """Print formatted geometry analysis summary."""
        print("\n" + "="*60)
        print("REPRESENTATION GEOMETRY ANALYSIS")
        print("="*60)

        print(f"\nPCA Summary:")
        print(f"  Components for 90% variance: {self.n_components_90_variance}")
        print(f"  Variance explained (top {len(self.pca_correlations)} PCs): {self.total_variance_explained:.1%}")

        print("\nPC-Feature Correlations:")
        print("-"*50)
        for pc in self.pca_correlations:
            print(f"  PC{pc.pc_index+1} ({pc.variance_explained:.1%} var):")
            print(f"    Top correlation: {pc.top_correlated_feature} (r={pc.top_correlation:.3f})")

        print("\nFeature Separability:")
        print("-"*50)
        for feat, sep in sorted(self.separability.items(),
                                 key=lambda x: x[1].silhouette_score, reverse=True):
            print(f"  {feat}:")
            print(f"    Silhouette: {sep.silhouette_score:.3f}")
            print(f"    Linear probe: {sep.linear_probe_accuracy:.1%} ± {sep.linear_probe_std:.1%}")


class GeometryAnalyzer:
    """
    Analyzes geometric structure of representation space.

    Usage:
        analyzer = GeometryAnalyzer(n_components=10)
        result = analyzer.analyze(rollout_data)
        result.print_summary()
    """

    def __init__(
        self,
        n_components: int = 10,
        n_bins_separability: int = 3,
        standardize: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_components: Number of PCA components to analyze
            n_bins_separability: Number of bins for separability analysis
            standardize: Whether to standardize activations
            random_state: Random seed
        """
        self.n_components = n_components
        self.n_bins_separability = n_bins_separability
        self.standardize = standardize
        self.random_state = random_state

        self.scaler = StandardScaler() if standardize else None
        self.pca = None

    def analyze(self, data: RolloutData) -> GeometryAnalysisResult:
        """
        Perform complete geometry analysis.

        Args:
            data: RolloutData containing activations and features

        Returns:
            GeometryAnalysisResult
        """
        activations = data.activations

        # Standardize
        if self.standardize:
            activations_scaled = self.scaler.fit_transform(activations)
        else:
            activations_scaled = activations

        # PCA
        self.pca = PCA(n_components=min(self.n_components, activations_scaled.shape[1]))
        pca_embeddings = self.pca.fit_transform(activations_scaled)

        # PC-feature correlations
        pca_correlations = self._compute_pc_correlations(
            pca_embeddings, data.features, data.feature_names
        )

        # Variance explained
        total_var = sum(self.pca.explained_variance_ratio_[:self.n_components])

        # Find components needed for 90% variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        n_90 = int(np.searchsorted(cumsum, 0.9) + 1)

        # Separability analysis
        separability = self._compute_separability(
            activations_scaled, data.features, data.feature_names
        )

        # Store feature values for visualization
        feature_values = {
            name: data.get_feature(name)
            for name in data.feature_names
        }

        return GeometryAnalysisResult(
            pca_correlations=pca_correlations,
            total_variance_explained=total_var,
            n_components_90_variance=n_90,
            separability=separability,
            pca_embeddings=pca_embeddings,
            feature_values=feature_values,
        )

    def _compute_pc_correlations(
        self,
        pca_embeddings: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[PCAFeatureCorrelation]:
        """Compute correlation between each PC and game features."""
        correlations = []

        for pc_idx in range(pca_embeddings.shape[1]):
            pc_values = pca_embeddings[:, pc_idx]
            var_explained = self.pca.explained_variance_ratio_[pc_idx]

            feature_corrs = {}
            feature_pvals = {}

            for feat_idx, feat_name in enumerate(feature_names):
                feat_values = features[:, feat_idx]
                r, p = stats.pearsonr(pc_values, feat_values)
                feature_corrs[feat_name] = r
                feature_pvals[feat_name] = p

            # Find top correlated feature
            top_feat = max(feature_corrs, key=lambda k: abs(feature_corrs[k]))
            top_corr = feature_corrs[top_feat]

            correlations.append(PCAFeatureCorrelation(
                pc_index=pc_idx,
                variance_explained=var_explained,
                feature_correlations=feature_corrs,
                feature_pvalues=feature_pvals,
                top_correlated_feature=top_feat,
                top_correlation=top_corr,
            ))

        return correlations

    def _compute_separability(
        self,
        activations: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, SeparabilityResult]:
        """Compute separability metrics for each feature."""
        results = {}

        for feat_idx, feat_name in enumerate(feature_names):
            feat_values = features[:, feat_idx]

            # Discretize for silhouette and linear probe
            feat_bins = discretize_feature(
                feat_values,
                n_bins=self.n_bins_separability,
                method='quantile'
            )

            # Skip if only one bin (constant feature)
            if len(np.unique(feat_bins)) < 2:
                results[feat_name] = SeparabilityResult(
                    feature_name=feat_name,
                    n_bins=self.n_bins_separability,
                    silhouette_score=0.0,
                    linear_probe_accuracy=0.0,
                    linear_probe_std=0.0,
                )
                continue

            # Silhouette score
            sil = silhouette_score(activations, feat_bins)

            # Linear probe accuracy (cross-validated)
            probe = LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
                solver='lbfgs',
                multi_class='multinomial'
            )
            try:
                scores = cross_val_score(probe, activations, feat_bins, cv=5)
                probe_acc = scores.mean()
                probe_std = scores.std()
            except Exception:
                probe_acc = 0.0
                probe_std = 0.0

            results[feat_name] = SeparabilityResult(
                feature_name=feat_name,
                n_bins=self.n_bins_separability,
                silhouette_score=sil,
                linear_probe_accuracy=probe_acc,
                linear_probe_std=probe_std,
            )

        return results


def plot_pca_correlation_heatmap(
    result: GeometryAnalysisResult,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of PC-feature correlations.

    Args:
        result: GeometryAnalysisResult
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    matrix, pc_labels, feature_names = result.get_pc_feature_matrix()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(len(pc_labels)))
    ax.set_yticklabels(pc_labels)

    # Add correlation values as text
    for i in range(len(pc_labels)):
        for j in range(len(feature_names)):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, label='Pearson r')
    ax.set_title('PC-Feature Correlations')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PC correlation heatmap to {save_path}")

    return fig


def plot_separability_comparison(
    result: GeometryAnalysisResult,
    save_path: Optional[str] = None
):
    """
    Plot separability metrics comparison across features.

    Args:
        result: GeometryAnalysisResult
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    features = list(result.separability.keys())
    silhouettes = [result.separability[f].silhouette_score for f in features]
    probe_accs = [result.separability[f].linear_probe_accuracy for f in features]
    probe_stds = [result.separability[f].linear_probe_std for f in features]

    # Sort by silhouette
    sorted_indices = np.argsort(silhouettes)[::-1]
    features = [features[i] for i in sorted_indices]
    silhouettes = [silhouettes[i] for i in sorted_indices]
    probe_accs = [probe_accs[i] for i in sorted_indices]
    probe_stds = [probe_stds[i] for i in sorted_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Silhouette scores
    ax1.barh(features, silhouettes, color='steelblue')
    ax1.set_xlabel('Silhouette Score')
    ax1.set_title('Geometric Separability')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.invert_yaxis()

    # Linear probe accuracy
    ax2.barh(features, probe_accs, xerr=probe_stds, color='coral', capsize=3)
    ax2.set_xlabel('Linear Probe Accuracy')
    ax2.set_title('Linear Separability')
    ax2.axvline(x=1/result.separability[features[0]].n_bins,
                color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved separability comparison to {save_path}")

    return fig


def plot_tsne_by_feature(
    result: GeometryAnalysisResult,
    feature_name: str,
    perplexity: int = 30,
    save_path: Optional[str] = None
):
    """
    Plot t-SNE visualization colored by a feature.

    Args:
        result: GeometryAnalysisResult
        feature_name: Feature to color by
        perplexity: t-SNE perplexity
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Use PCA embeddings as input (faster than raw activations)
    embeddings = result.pca_embeddings[:, :50]  # Use top 50 PCs

    # Subsample if too many points
    max_points = 5000
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        feature_values = result.feature_values[feature_name][idx]
    else:
        feature_values = result.feature_values[feature_name]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=feature_values,
        cmap='viridis',
        alpha=0.6,
        s=5
    )

    plt.colorbar(scatter, label=feature_name)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE colored by {feature_name}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    return fig
