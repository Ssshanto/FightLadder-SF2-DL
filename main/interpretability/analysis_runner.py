#!/usr/bin/env python3
"""
Unified Interpretability Analysis Runner

Runs all three analysis modules (MI, Clustering, Geometry) on a trained model
and produces a comprehensive report.

Usage:
    python analysis_runner.py --model-path path/to/model.zip --output results/
    python analysis_runner.py --model-path path/to/model.zip --n-episodes 20 --output results/

Outputs:
    - MI analysis table and bar chart
    - Cluster profiles and heatmaps
    - PCA correlation matrix and t-SNE visualizations
    - Summary report
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ground_truth import RolloutCollector, RolloutData, GROUND_TRUTH_FEATURES
from .mi_analysis import MutualInformationAnalyzer, MIAnalysisResult, plot_mi_results, print_mi_summary
from .concept_discovery import ConceptDiscoverer, ConceptDiscoveryResult, plot_cluster_profiles, plot_cluster_action_distributions
from .geometry_analysis import GeometryAnalyzer, GeometryAnalysisResult, plot_pca_correlation_heatmap, plot_separability_comparison, plot_tsne_by_feature


def collect_rollout_data(
    model,
    env,
    n_episodes: int = 10,
    feature_names: Optional[list] = None,
    verbose: bool = True
) -> RolloutData:
    """
    Collect activation and feature data from model rollouts.

    Args:
        model: Trained PPO model
        env: Gym environment
        n_episodes: Number of episodes to collect
        feature_names: Which features to extract (None = all)
        verbose: Print progress

    Returns:
        RolloutData containing activations, features, actions
    """
    collector = RolloutCollector(feature_names)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get activation from feature extractor
            obs_tensor = torch.tensor(obs).unsqueeze(0).float()
            with torch.no_grad():
                features = model.policy.features_extractor(obs_tensor.to(model.device))
                activation = features.cpu().numpy().squeeze()

            # Get action
            action, _ = model.predict(obs, deterministic=True)

            # Collect data
            collector.add_frame(
                activation=activation,
                info=info,
                action=int(action),
                reward=0.0,  # Will be filled after step
                episode_id=episode
            )

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward

            # Update last frame's reward
            collector._rewards[-1] = float(reward)

        if verbose:
            outcome = "win" if info.get('enemy_hp', 176) < info.get('agent_hp', 0) else "loss"
            print(f"Episode {episode + 1}/{n_episodes}: {outcome}, reward={episode_reward:.1f}")

    return collector.get_data()


def run_full_analysis(
    data: RolloutData,
    output_dir: str,
    n_clusters: int = 8,
    n_pca_components: int = 10,
    verbose: bool = True
) -> dict:
    """
    Run all three analysis modules.

    Args:
        data: RolloutData from collect_rollout_data
        output_dir: Directory to save results
        n_clusters: Number of clusters for concept discovery
        n_pca_components: Number of PCA components to analyze
        verbose: Print results

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # =========================================================================
    # Module A: Mutual Information Analysis
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("Running Module A: Mutual Information Analysis")
        print("="*60)

    mi_analyzer = MutualInformationAnalyzer(n_bins=10)
    mi_result = mi_analyzer.analyze(data)
    results['mi'] = mi_result

    if verbose:
        print_mi_summary(mi_result)

    # Save MI plot
    try:
        plot_mi_results(mi_result, save_path=os.path.join(output_dir, 'mi_analysis.png'))
    except Exception as e:
        print(f"Could not save MI plot: {e}")

    # =========================================================================
    # Module B: Unsupervised Concept Discovery
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("Running Module B: Unsupervised Concept Discovery")
        print("="*60)

    discoverer = ConceptDiscoverer(n_clusters=n_clusters)
    cluster_result = discoverer.discover(data)
    results['clusters'] = cluster_result

    if verbose:
        cluster_result.print_summary()

    # Save cluster plots
    try:
        plot_cluster_profiles(
            cluster_result,
            data.feature_names,
            save_path=os.path.join(output_dir, 'cluster_profiles.png')
        )
        plot_cluster_action_distributions(
            cluster_result,
            save_path=os.path.join(output_dir, 'cluster_actions.png')
        )
    except Exception as e:
        print(f"Could not save cluster plots: {e}")

    # =========================================================================
    # Module D: Representation Geometry Analysis
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("Running Module D: Representation Geometry Analysis")
        print("="*60)

    geometry_analyzer = GeometryAnalyzer(n_components=n_pca_components)
    geometry_result = geometry_analyzer.analyze(data)
    results['geometry'] = geometry_result

    if verbose:
        geometry_result.print_summary()

    # Save geometry plots
    try:
        plot_pca_correlation_heatmap(
            geometry_result,
            save_path=os.path.join(output_dir, 'pca_correlations.png')
        )
        plot_separability_comparison(
            geometry_result,
            save_path=os.path.join(output_dir, 'separability.png')
        )
        # t-SNE for top feature
        if mi_result.get_ranking():
            top_feature = mi_result.get_ranking()[0][0]
            plot_tsne_by_feature(
                geometry_result,
                top_feature,
                save_path=os.path.join(output_dir, f'tsne_{top_feature}.png')
            )
    except Exception as e:
        print(f"Could not save geometry plots: {e}")

    # =========================================================================
    # Save Results
    # =========================================================================
    # Save raw data
    with open(os.path.join(output_dir, 'rollout_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    # Save summary as JSON
    summary = generate_summary(mi_result, cluster_result, geometry_result)
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save text report
    report = generate_text_report(mi_result, cluster_result, geometry_result)
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report)

    if verbose:
        print(f"\nResults saved to {output_dir}/")

    return results


def generate_summary(
    mi_result: MIAnalysisResult,
    cluster_result: ConceptDiscoveryResult,
    geometry_result: GeometryAnalysisResult
) -> dict:
    """Generate JSON-serializable summary."""
    return {
        'timestamp': datetime.now().isoformat(),
        'n_samples': mi_result.n_samples,
        'mi_analysis': {
            'action_entropy': mi_result.action_entropy,
            'feature_importance': {
                name: result.mi_bits
                for name, result in mi_result.results.items()
            }
        },
        'cluster_analysis': {
            'n_clusters': cluster_result.n_clusters,
            'silhouette_score': cluster_result.silhouette_score,
            'feature_cluster_correlation': cluster_result.feature_cluster_correlation,
        },
        'geometry_analysis': {
            'variance_explained': geometry_result.total_variance_explained,
            'n_components_90_var': geometry_result.n_components_90_variance,
            'separability': {
                name: {
                    'silhouette': result.silhouette_score,
                    'linear_probe_acc': result.linear_probe_accuracy,
                }
                for name, result in geometry_result.separability.items()
            },
            'pc_top_correlations': [
                {
                    'pc': pc.pc_index + 1,
                    'var_explained': pc.variance_explained,
                    'top_feature': pc.top_correlated_feature,
                    'correlation': pc.top_correlation,
                }
                for pc in geometry_result.pca_correlations[:5]
            ]
        }
    }


def generate_text_report(
    mi_result: MIAnalysisResult,
    cluster_result: ConceptDiscoveryResult,
    geometry_result: GeometryAnalysisResult
) -> str:
    """Generate human-readable text report."""
    lines = [
        "=" * 70,
        "INTERPRETABILITY ANALYSIS REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Samples: {mi_result.n_samples}",
        "",
        "=" * 70,
        "EXECUTIVE SUMMARY",
        "=" * 70,
        "",
    ]

    # Find key insights
    top_mi_features = mi_result.get_ranking()[:3]
    top_sep_features = sorted(
        geometry_result.separability.items(),
        key=lambda x: x[1].silhouette_score,
        reverse=True
    )[:3]

    lines.append("Key Findings:")
    lines.append("")
    lines.append(f"1. POLICY DEPENDENCE (MI Analysis):")
    for i, (feat, mi) in enumerate(top_mi_features):
        lines.append(f"   - {feat}: {mi:.3f} bits")
    lines.append("")

    lines.append(f"2. REPRESENTATION STRUCTURE (Geometry Analysis):")
    lines.append(f"   - PC1 top correlation: {geometry_result.pca_correlations[0].top_correlated_feature} "
                f"(r={geometry_result.pca_correlations[0].top_correlation:.2f})")
    lines.append(f"   - Best separated feature: {top_sep_features[0][0]} "
                f"(silhouette={top_sep_features[0][1].silhouette_score:.2f})")
    lines.append("")

    lines.append(f"3. DISCOVERED CONCEPTS (Clustering):")
    lines.append(f"   - Found {cluster_result.n_clusters} distinct clusters "
                f"(silhouette={cluster_result.silhouette_score:.2f})")
    lines.append(f"   - Best cluster-separating feature: "
                f"{max(cluster_result.feature_cluster_correlation.items(), key=lambda x: x[1])[0]}")
    lines.append("")

    # Detailed sections
    lines.extend([
        "=" * 70,
        "DETAILED RESULTS",
        "=" * 70,
        "",
        "--- Module A: Mutual Information ---",
        f"H(action) = {mi_result.action_entropy:.3f} bits",
        "",
        "Feature MI (bits):",
    ])

    for feat, mi in mi_result.get_ranking():
        r = mi_result.results[feat]
        lines.append(f"  {feat:20s}: {mi:.3f} (normalized: {r.normalized_mi:.1%})")

    lines.extend([
        "",
        "--- Module B: Clustering ---",
        f"Clusters: {cluster_result.n_clusters}",
        f"Silhouette: {cluster_result.silhouette_score:.3f}",
        "",
        "Cluster Profiles:",
    ])

    for profile in cluster_result.cluster_profiles[:5]:  # Top 5
        lines.append(f"  Cluster {profile.cluster_id} ({profile.fraction:.1%} of data):")
        for feat, mean in list(profile.feature_means.items())[:3]:
            lines.append(f"    {feat}: {mean:.1f}")

    lines.extend([
        "",
        "--- Module D: Geometry ---",
        f"Variance explained (top 10 PCs): {geometry_result.total_variance_explained:.1%}",
        f"PCs for 90% variance: {geometry_result.n_components_90_variance}",
        "",
        "Top PC-Feature Correlations:",
    ])

    for pc in geometry_result.pca_correlations[:5]:
        lines.append(f"  PC{pc.pc_index+1} ({pc.variance_explained:.1%} var): "
                    f"{pc.top_correlated_feature} (r={pc.top_correlation:.2f})")

    lines.extend([
        "",
        "Separability Scores:",
    ])

    for feat, sep in sorted(geometry_result.separability.items(),
                            key=lambda x: x[1].silhouette_score, reverse=True):
        lines.append(f"  {feat:20s}: sil={sep.silhouette_score:.2f}, "
                    f"probe={sep.linear_probe_accuracy:.1%}")

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run interpretability analysis')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--state', default='Champion.Level1.RyuVsGuile', help='Game state')
    parser.add_argument('--output', default='interpretability_results', help='Output directory')
    parser.add_argument('--n-episodes', type=int, default=10, help='Episodes to collect')
    parser.add_argument('--n-clusters', type=int, default=8, help='Number of clusters')
    parser.add_argument('--n-pca', type=int, default=10, help='PCA components')

    args = parser.parse_args()

    # Import here to avoid circular imports
    import stable_retro as retro
    from stable_baselines3 import PPO
    from common.const import sf_game
    from common.retro_wrappers import SFWrapper

    # Create environment
    print(f"Loading model from {args.model_path}")
    env = retro.make(
        game=sf_game,
        state=args.state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        players=2,
    )
    env = SFWrapper(env, side='left', rendering=False, reset_type='round')

    # Load model
    model = PPO.load(args.model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Collect data
    print(f"\nCollecting data from {args.n_episodes} episodes...")
    data = collect_rollout_data(model, env, n_episodes=args.n_episodes)
    print(f"Collected {len(data)} frames")

    # Run analysis
    results = run_full_analysis(
        data,
        output_dir=args.output,
        n_clusters=args.n_clusters,
        n_pca_components=args.n_pca,
        verbose=True
    )

    env.close()
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
