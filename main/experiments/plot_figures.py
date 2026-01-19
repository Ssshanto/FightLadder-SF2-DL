"""
Plotting Module for Research Paper Figures

Generates publication-quality figures:
- Learning curves with confidence intervals
- MI evolution over training
- Feature importance bar charts
- Correlation heatmaps

Uses matplotlib with publication-ready styling.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import matplotlib with Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Publication-quality style settings
STYLE_SETTINGS = {
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update(STYLE_SETTINGS)


def plot_learning_curves(
    curves: Dict[str, Dict],
    output_file: str,
    title: str = "Learning Curves",
    ylabel: str = "Win Rate",
    ylim: Tuple[float, float] = None,
    show_confidence: bool = True,
    colors: List[str] = None,
):
    """
    Plot learning curves with confidence intervals.
    
    Args:
        curves: Dict mapping experiment name to {timesteps, mean, std}
        output_file: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        ylim: Optional y-axis limits
        show_confidence: Whether to show confidence intervals
        colors: Optional list of colors
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if colors is None:
        colors = plt.cm.tab10.colors
    
    for i, (name, data) in enumerate(curves.items()):
        if not data.get('timesteps'):
            continue
        
        timesteps = np.array(data['timesteps'])
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        
        color = colors[i % len(colors)]
        
        # Plot mean
        ax.plot(timesteps, mean, label=name, color=color)
        
        # Plot confidence interval (1 std)
        if show_confidence and len(std) > 0:
            ax.fill_between(timesteps, mean - std, mean + std, 
                          color=color, alpha=0.2)
    
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    
    if ylim:
        ax.set_ylim(ylim)
    
    # Format x-axis with K/M notation
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
    )
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved learning curves to {output_file}")


def plot_mi_evolution(
    mi_curves: Dict[str, Dict[str, Dict]],
    output_file: str,
    features_to_plot: List[str] = None,
    title: str = "Mutual Information Evolution During Training",
):
    """
    Plot MI evolution for multiple features over training.
    
    Args:
        mi_curves: Dict mapping feature name to {experiment: {timesteps, mean, std}}
        output_file: Path to save figure
        features_to_plot: List of features to include (None = all)
        title: Plot title
    """
    set_publication_style()
    
    if features_to_plot is None:
        features_to_plot = list(mi_curves.keys())
    
    # Create subplots for each feature
    n_features = len(features_to_plot)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]
    
    colors = plt.cm.tab10.colors
    
    for idx, feature in enumerate(features_to_plot):
        if feature not in mi_curves:
            continue
        
        ax = axes[idx]
        feature_data = mi_curves[feature]
        
        for i, (exp_name, data) in enumerate(feature_data.items()):
            if not data.get('timesteps'):
                continue
            
            timesteps = np.array(data['timesteps'])
            mean = np.array(data['mean'])
            std = np.array(data['std'])
            
            color = colors[i % len(colors)]
            
            ax.plot(timesteps, mean, label=exp_name, color=color)
            ax.fill_between(timesteps, mean - std, mean + std, 
                          color=color, alpha=0.2)
        
        # Clean up feature name for display
        display_name = feature.replace('mi_action_', '').replace('_', ' ').title()
        ax.set_title(f'I(action; {display_name})')
        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel('Mutual Information (nats)')
        ax.legend(loc='best')
        
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
        )
    
    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved MI evolution plot to {output_file}")


def plot_feature_importance(
    mi_stats: Dict,
    output_file: str,
    experiment_name: str = None,
    top_k: int = 10,
    title: str = "Feature Importance (Mutual Information)",
):
    """
    Plot bar chart of feature importance based on MI.
    
    Args:
        mi_stats: MI statistics from aggregate_results
        output_file: Path to save figure
        experiment_name: Specific experiment to plot (None = first)
        top_k: Number of top features to show
        title: Plot title
    """
    set_publication_style()
    
    if experiment_name is None:
        experiment_name = list(mi_stats.keys())[0] if mi_stats else None
    
    if experiment_name not in mi_stats:
        print(f"Experiment {experiment_name} not found")
        return
    
    stats = mi_stats[experiment_name]['mi_statistics']
    
    # Sort by mean MI
    sorted_features = sorted(
        stats.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )[:top_k]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_names = [f[0].replace('mi_action_', '').replace('_', ' ').title() 
                   for f in sorted_features]
    means = [f[1]['mean'] for f in sorted_features]
    stds = [f[1]['std'] for f in sorted_features]
    
    y_pos = np.arange(len(feature_names))
    
    bars = ax.barh(y_pos, means, xerr=stds, color='steelblue', 
                   capsize=3, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Mutual Information I(action; feature)')
    ax.set_title(title)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.0005, i, f'{mean:.4f}', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved feature importance plot to {output_file}")


def plot_correlation_heatmap(
    correlations: Dict,
    output_file: str,
    experiment_name: str = None,
    title: str = "MI-Performance Correlation",
):
    """
    Plot heatmap of correlations between MI values and performance.
    
    Args:
        correlations: Correlation data from aggregate_results
        output_file: Path to save figure
        experiment_name: Specific experiment to plot
        title: Plot title
    """
    set_publication_style()
    
    if experiment_name is None:
        experiment_name = list(correlations.keys())[0] if correlations else None
    
    if experiment_name not in correlations:
        print(f"Experiment {experiment_name} not found")
        return
    
    corr_data = correlations[experiment_name]
    
    if not corr_data:
        print("No correlation data available")
        return
    
    # Sort by absolute correlation
    sorted_features = sorted(
        corr_data.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    feature_names = [f[0].replace('mi_action_', '').replace('_', ' ').title() 
                   for f in sorted_features]
    corr_values = [f[1]['correlation'] for f in sorted_features]
    
    # Color by sign of correlation
    colors = ['forestgreen' if c > 0 else 'indianred' for c in corr_values]
    
    y_pos = np.arange(len(feature_names))
    
    bars = ax.barh(y_pos, corr_values, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Correlation with Win Rate')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-1, 1)
    
    # Add value labels
    for i, corr in enumerate(corr_values):
        offset = 0.05 if corr >= 0 else -0.15
        ax.text(corr + offset, i, f'{corr:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved correlation heatmap to {output_file}")


def plot_combined_summary(
    report: Dict,
    output_dir: str,
    prefix: str = "paper",
):
    """
    Generate all paper figures from a summary report.
    
    Args:
        report: Summary report from aggregate_results
        output_dir: Directory to save figures
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Win rate learning curves
    if 'win_rate_curves' in report:
        plot_learning_curves(
            report['win_rate_curves'],
            os.path.join(output_dir, f"{prefix}_win_rate_curves.pdf"),
            title="Win Rate During Training",
            ylabel="Win Rate",
            ylim=(0, 1),
        )
        # Also save PNG for quick viewing
        plot_learning_curves(
            report['win_rate_curves'],
            os.path.join(output_dir, f"{prefix}_win_rate_curves.png"),
            title="Win Rate During Training",
            ylabel="Win Rate",
            ylim=(0, 1),
        )
    
    # 2. MI evolution
    if 'mi_learning_curves' in report:
        # Get all features from MI curves
        features = list(report['mi_learning_curves'].keys())
        plot_mi_evolution(
            report['mi_learning_curves'],
            os.path.join(output_dir, f"{prefix}_mi_evolution.pdf"),
            features_to_plot=features[:4],  # Top 4
        )
        plot_mi_evolution(
            report['mi_learning_curves'],
            os.path.join(output_dir, f"{prefix}_mi_evolution.png"),
            features_to_plot=features[:4],
        )
    
    # 3. Feature importance
    if 'mi_statistics' in report:
        for exp_name in report['mi_statistics']:
            safe_name = exp_name.replace(' ', '_')
            plot_feature_importance(
                report['mi_statistics'],
                os.path.join(output_dir, f"{prefix}_feature_importance_{safe_name}.pdf"),
                experiment_name=exp_name,
            )
            plot_feature_importance(
                report['mi_statistics'],
                os.path.join(output_dir, f"{prefix}_feature_importance_{safe_name}.png"),
                experiment_name=exp_name,
            )
    
    # 4. Correlation heatmap
    if 'mi_correlations' in report:
        for exp_name in report['mi_correlations']:
            safe_name = exp_name.replace(' ', '_')
            plot_correlation_heatmap(
                report['mi_correlations'],
                os.path.join(output_dir, f"{prefix}_correlations_{safe_name}.pdf"),
                experiment_name=exp_name,
            )
            plot_correlation_heatmap(
                report['mi_correlations'],
                os.path.join(output_dir, f"{prefix}_correlations_{safe_name}.png"),
                experiment_name=exp_name,
            )
    
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--report', required=True,
                        help='Path to experiment_report.json')
    parser.add_argument('--output-dir', default='figures',
                        help='Directory to save figures')
    parser.add_argument('--prefix', default='paper',
                        help='Filename prefix')
    
    args = parser.parse_args()
    
    # Load report
    with open(args.report) as f:
        report = json.load(f)
    
    # Generate figures
    plot_combined_summary(report, args.output_dir, args.prefix)
