"""
Representation Metrics Module

Provides metrics for analyzing neural network representations:
- KL divergence between representation distributions
- Jensen-Shannon divergence for symmetric distance
- Centered Kernel Alignment (CKA) for layer comparison
- Representation similarity analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import warnings


class RepresentationMetrics:
    """
    Collection of metrics for analyzing neural network representations.
    """

    @staticmethod
    def kl_divergence_gmm(
        samples1: np.ndarray,
        samples2: np.ndarray,
        n_components: int = 3,
        n_samples: int = 1000
    ) -> float:
        """
        Estimate KL divergence between two distributions using GMM approximation.

        Args:
            samples1: Samples from first distribution (n1, d)
            samples2: Samples from second distribution (n2, d)
            n_components: Number of GMM components
            n_samples: Samples for Monte Carlo estimation

        Returns:
            Estimated KL(P1 || P2)
        """
        if len(samples1) < 10 or len(samples2) < 10:
            return float('inf')

        # Fit GMMs
        gmm1 = GaussianMixture(n_components=min(n_components, len(samples1) // 5), random_state=42)
        gmm2 = GaussianMixture(n_components=min(n_components, len(samples2) // 5), random_state=42)

        gmm1.fit(samples1)
        gmm2.fit(samples2)

        # Sample from first distribution
        samples, _ = gmm1.sample(n_samples)

        # Compute log probabilities
        log_p1 = gmm1.score_samples(samples)
        log_p2 = gmm2.score_samples(samples)

        # KL divergence estimate
        kl = np.mean(log_p1 - log_p2)

        return max(0, kl)  # KL should be non-negative

    @staticmethod
    def jensen_shannon_divergence(
        samples1: np.ndarray,
        samples2: np.ndarray,
        n_components: int = 3
    ) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric KL).

        Args:
            samples1: Samples from first distribution
            samples2: Samples from second distribution
            n_components: Number of GMM components

        Returns:
            JS divergence value
        """
        kl_1_2 = RepresentationMetrics.kl_divergence_gmm(samples1, samples2, n_components)
        kl_2_1 = RepresentationMetrics.kl_divergence_gmm(samples2, samples1, n_components)

        return (kl_1_2 + kl_2_1) / 2

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Linear Centered Kernel Alignment between representations.

        CKA measures similarity between representation spaces independent of
        orthogonal transformations and isotropic scaling.

        Args:
            X: First representation matrix (n_samples, d1)
            Y: Second representation matrix (n_samples, d2)

        Returns:
            CKA similarity value (0-1, higher = more similar)
        """
        # Center the matrices
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Compute linear kernels
        XXT = X @ X.T
        YYT = Y @ Y.T

        # HSIC (Hilbert-Schmidt Independence Criterion)
        def hsic(K, L):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

        hsic_xy = hsic(XXT, YYT)
        hsic_xx = hsic(XXT, XXT)
        hsic_yy = hsic(YYT, YYT)

        if hsic_xx * hsic_yy <= 0:
            return 0.0

        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

    @staticmethod
    def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
        """
        Compute CKA with RBF (Gaussian) kernel.

        More sensitive to fine-grained representation structure than linear CKA.

        Args:
            X: First representation matrix (n_samples, d1)
            Y: Second representation matrix (n_samples, d2)
            sigma: RBF bandwidth (auto-computed if None)

        Returns:
            CKA similarity value (0-1)
        """
        def rbf_kernel(Z, sigma):
            dist_sq = cdist(Z, Z, 'sqeuclidean')
            return np.exp(-dist_sq / (2 * sigma ** 2))

        # Auto bandwidth using median heuristic
        if sigma is None:
            sigma_x = np.median(cdist(X, X, 'euclidean'))
            sigma_y = np.median(cdist(Y, Y, 'euclidean'))
            sigma = (sigma_x + sigma_y) / 2

        K = rbf_kernel(X, sigma)
        L = rbf_kernel(Y, sigma)

        # Center kernels
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H
        L = H @ L @ H

        # HSIC
        hsic_xy = np.trace(K @ L) / ((n - 1) ** 2)
        hsic_xx = np.trace(K @ K) / ((n - 1) ** 2)
        hsic_yy = np.trace(L @ L) / ((n - 1) ** 2)

        if hsic_xx * hsic_yy <= 0:
            return 0.0

        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

    @staticmethod
    def representation_similarity_matrix(
        layer_activations: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise CKA similarities between all layers.

        Args:
            layer_activations: Dict mapping layer names to activation matrices

        Returns:
            Similarity matrix (n_layers, n_layers)
        """
        layer_names = list(layer_activations.keys())
        n_layers = len(layer_names)
        similarity_matrix = np.zeros((n_layers, n_layers))

        for i, name1 in enumerate(layer_names):
            for j, name2 in enumerate(layer_names):
                if i <= j:
                    cka = RepresentationMetrics.linear_cka(
                        layer_activations[name1],
                        layer_activations[name2]
                    )
                    similarity_matrix[i, j] = cka
                    similarity_matrix[j, i] = cka

        return similarity_matrix

    @staticmethod
    def intrinsic_dimensionality(X: np.ndarray, method: str = 'pca') -> float:
        """
        Estimate intrinsic dimensionality of representation space.

        Args:
            X: Activation matrix (n_samples, d)
            method: 'pca' for PCA-based or 'mle' for MLE estimate

        Returns:
            Estimated intrinsic dimensionality
        """
        if method == 'pca':
            # PCA-based: count dimensions explaining 95% variance
            X_centered = X - X.mean(axis=0)
            _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
            explained_variance_ratio = (s ** 2) / (s ** 2).sum()
            cumsum = np.cumsum(explained_variance_ratio)
            return np.searchsorted(cumsum, 0.95) + 1

        elif method == 'mle':
            # MLE estimate (Two-NN method simplified)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=3)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            r1 = distances[:, 1]
            r2 = distances[:, 2]
            mu = r2 / (r1 + 1e-8)
            mu = mu[mu > 1]
            if len(mu) == 0:
                return X.shape[1]
            return len(mu) / np.sum(np.log(mu))

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def cluster_purity(
        activations: np.ndarray,
        labels: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> float:
        """
        Measure how well clusters in activation space align with class labels.

        Args:
            activations: Activation matrix
            labels: True class labels
            n_clusters: Number of clusters (defaults to unique labels)

        Returns:
            Purity score (0-1, higher = better alignment)
        """
        from sklearn.cluster import KMeans

        n_clusters = n_clusters or len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(activations)

        purity = 0
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() == 0:
                continue
            cluster_true_labels = labels[cluster_mask]
            most_common = stats.mode(cluster_true_labels, keepdims=False)[0]
            purity += np.sum(cluster_true_labels == most_common)

        return purity / len(labels)

    @staticmethod
    def feature_importance_variance(
        activations: np.ndarray,
        method: str = 'variance'
    ) -> np.ndarray:
        """
        Compute importance scores for each feature dimension.

        Args:
            activations: Activation matrix (n_samples, d)
            method: 'variance' or 'mad' (median absolute deviation)

        Returns:
            Importance scores for each dimension (d,)
        """
        if method == 'variance':
            return np.var(activations, axis=0)
        elif method == 'mad':
            median = np.median(activations, axis=0)
            return np.median(np.abs(activations - median), axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def representation_drift(
        activations_before: np.ndarray,
        activations_after: np.ndarray,
        method: str = 'centroid'
    ) -> float:
        """
        Measure drift between two sets of activations.

        Useful for tracking how representations change during training or
        when facing different opponents.

        Args:
            activations_before: Earlier activations
            activations_after: Later activations
            method: 'centroid' for centroid distance, 'wasserstein' for EMD

        Returns:
            Drift magnitude
        """
        if method == 'centroid':
            centroid_before = np.mean(activations_before, axis=0)
            centroid_after = np.mean(activations_after, axis=0)
            return np.linalg.norm(centroid_after - centroid_before)

        elif method == 'wasserstein':
            # 1D Wasserstein per dimension, then average
            drifts = []
            for dim in range(activations_before.shape[1]):
                drift_1d = stats.wasserstein_distance(
                    activations_before[:, dim],
                    activations_after[:, dim]
                )
                drifts.append(drift_1d)
            return np.mean(drifts)

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def concept_alignment(
        activations: np.ndarray,
        concept_vectors: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Measure alignment of activations with known concept directions.

        Args:
            activations: Activation matrix (n_samples, d)
            concept_vectors: Dict mapping concept names to direction vectors

        Returns:
            Dict mapping concept names to alignment scores
        """
        alignments = {}

        for concept_name, direction in concept_vectors.items():
            # Normalize direction
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Project activations onto direction
            projections = activations @ direction

            # Alignment is variance of projections (high variance = strong encoding)
            alignments[concept_name] = np.var(projections)

        # Normalize by total variance
        total_var = np.var(activations)
        if total_var > 0:
            alignments = {k: v / total_var for k, v in alignments.items()}

        return alignments


def compute_all_metrics(
    activations1: np.ndarray,
    activations2: np.ndarray,
    labels1: Optional[np.ndarray] = None,
    labels2: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all available metrics between two sets of activations.

    Args:
        activations1: First activation set
        activations2: Second activation set
        labels1: Optional labels for first set
        labels2: Optional labels for second set

    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}

    # Distance metrics
    metrics['js_divergence'] = RepresentationMetrics.jensen_shannon_divergence(
        activations1, activations2
    )
    metrics['linear_cka'] = RepresentationMetrics.linear_cka(activations1, activations2)
    metrics['rbf_cka'] = RepresentationMetrics.rbf_cka(activations1, activations2)
    metrics['centroid_drift'] = RepresentationMetrics.representation_drift(
        activations1, activations2, method='centroid'
    )

    # Individual set metrics
    metrics['intrinsic_dim_1'] = RepresentationMetrics.intrinsic_dimensionality(activations1)
    metrics['intrinsic_dim_2'] = RepresentationMetrics.intrinsic_dimensionality(activations2)

    # Label-dependent metrics
    if labels1 is not None:
        metrics['cluster_purity_1'] = RepresentationMetrics.cluster_purity(activations1, labels1)
    if labels2 is not None:
        metrics['cluster_purity_2'] = RepresentationMetrics.cluster_purity(activations2, labels2)

    return metrics
