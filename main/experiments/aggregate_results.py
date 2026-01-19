"""
Results Aggregation Module

Aggregates results from multiple experiment runs to compute:
- Mean/std statistics across seeds
- Learning curves with confidence intervals
- MI evolution analysis
- Correlation between MI and performance

Outputs data suitable for paper tables and figures.
"""

import os
import json
import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExperimentResults:
    """Container for a single experiment's results."""
    name: str
    seed: int
    config: Dict
    mi_history: List[Dict]
    final_metrics: Dict
    
    @classmethod
    def load(cls, experiment_dir: str) -> Optional['ExperimentResults']:
        """Load results from an experiment directory."""
        try:
            # Load config
            config_file = os.path.join(experiment_dir, "models", "config.json")
            if not os.path.exists(config_file):
                return None
            with open(config_file) as f:
                config = json.load(f)
            
            # Load MI history
            history_file = os.path.join(experiment_dir, "interpretability", "mi_history.json")
            if os.path.exists(history_file):
                with open(history_file) as f:
                    mi_data = json.load(f)
                mi_history = mi_data.get('history', [])
                final_metrics = {
                    'total_timesteps': mi_data.get('total_timesteps', 0),
                    'total_episodes': mi_data.get('total_episodes', 0),
                    'total_wins': mi_data.get('total_wins', 0),
                    'final_win_rate': mi_data.get('final_win_rate', 0),
                }
            else:
                mi_history = []
                final_metrics = {}
            
            return cls(
                name=config.get('name', 'unknown'),
                seed=config.get('seed', 0),
                config=config,
                mi_history=mi_history,
                final_metrics=final_metrics,
            )
        except Exception as e:
            print(f"Error loading {experiment_dir}: {e}")
            return None


def load_all_experiments(base_dir: str) -> Dict[str, List[ExperimentResults]]:
    """
    Load all experiments from a base directory.
    
    Returns:
        Dictionary mapping experiment name to list of results (one per seed)
    """
    experiments = {}
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(base_dir, "*_seed*")):
        if not os.path.isdir(exp_dir):
            continue
        
        result = ExperimentResults.load(exp_dir)
        if result is None:
            continue
        
        if result.name not in experiments:
            experiments[result.name] = []
        experiments[result.name].append(result)
    
    # Sort by seed
    for name in experiments:
        experiments[name].sort(key=lambda x: x.seed)
    
    return experiments


def compute_mi_statistics(experiments: Dict[str, List[ExperimentResults]]) -> Dict:
    """
    Compute MI statistics across seeds for each experiment configuration.
    
    Returns:
        Dictionary with MI statistics suitable for paper tables
    """
    results = {}
    
    for name, exp_list in experiments.items():
        if not exp_list:
            continue
        
        # Get all MI feature names
        all_features = set()
        for exp in exp_list:
            if exp.mi_history:
                for entry in exp.mi_history:
                    all_features.update(entry.get('mi_results', {}).keys())
        
        # Compute final MI values across seeds
        final_mi = {feat: [] for feat in all_features}
        final_win_rates = []
        
        for exp in exp_list:
            if exp.mi_history:
                # Get last analysis
                last_entry = exp.mi_history[-1]
                for feat, value in last_entry.get('mi_results', {}).items():
                    final_mi[feat].append(value)
                final_win_rates.append(last_entry.get('win_rate_overall', 0))
            
            if exp.final_metrics:
                if 'final_win_rate' in exp.final_metrics:
                    if not final_win_rates or final_win_rates[-1] != exp.final_metrics['final_win_rate']:
                        pass  # Already added from mi_history
        
        # Compute mean and std for each feature
        mi_stats = {}
        for feat, values in final_mi.items():
            if values:
                mi_stats[feat] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n_seeds': len(values),
                }
        
        results[name] = {
            'n_seeds': len(exp_list),
            'mi_statistics': mi_stats,
            'win_rate': {
                'mean': float(np.mean(final_win_rates)) if final_win_rates else 0,
                'std': float(np.std(final_win_rates)) if final_win_rates else 0,
                'values': final_win_rates,
            },
            'config': exp_list[0].config,
        }
    
    return results


def compute_learning_curves(
    experiments: Dict[str, List[ExperimentResults]],
    metric: str = 'win_rate_overall'
) -> Dict:
    """
    Compute learning curves with confidence intervals across seeds.
    
    Args:
        experiments: Dictionary of experiment results
        metric: Metric to plot (win_rate_overall, mi_action_distance, etc.)
    
    Returns:
        Dictionary with timesteps, mean, std for each experiment
    """
    curves = {}
    
    for name, exp_list in experiments.items():
        if not exp_list:
            continue
        
        # Collect all (timestep, value) pairs from each seed
        seed_data = []
        for exp in exp_list:
            if not exp.mi_history:
                continue
            
            timesteps = []
            values = []
            for entry in exp.mi_history:
                ts = entry.get('timestep', 0)
                if metric in entry:
                    val = entry[metric]
                elif metric in entry.get('mi_results', {}):
                    val = entry['mi_results'][metric]
                else:
                    continue
                timesteps.append(ts)
                values.append(val)
            
            if timesteps:
                seed_data.append((np.array(timesteps), np.array(values)))
        
        if not seed_data:
            continue
        
        # Find common timesteps (use union and interpolate)
        all_timesteps = sorted(set().union(*[set(ts) for ts, _ in seed_data]))
        
        if not all_timesteps:
            continue
        
        # Interpolate each seed to common timesteps
        interpolated = []
        for ts, vals in seed_data:
            interp_vals = np.interp(all_timesteps, ts, vals)
            interpolated.append(interp_vals)
        
        interpolated = np.array(interpolated)
        
        curves[name] = {
            'timesteps': all_timesteps,
            'mean': interpolated.mean(axis=0).tolist(),
            'std': interpolated.std(axis=0).tolist(),
            'n_seeds': len(seed_data),
        }
    
    return curves


def compute_mi_correlation_with_performance(
    experiments: Dict[str, List[ExperimentResults]]
) -> Dict:
    """
    Compute correlation between MI values and performance (win rate).
    
    This helps identify which features the policy learns to depend on
    as it improves at the task.
    """
    correlations = {}
    
    for name, exp_list in experiments.items():
        if not exp_list:
            continue
        
        # Collect all (MI, win_rate) pairs across all seeds and timesteps
        feature_data = {}  # feature -> list of (mi_value, win_rate)
        
        for exp in exp_list:
            for entry in exp.mi_history:
                win_rate = entry.get('win_rate_overall', 0)
                for feat, mi_value in entry.get('mi_results', {}).items():
                    if feat not in feature_data:
                        feature_data[feat] = []
                    feature_data[feat].append((mi_value, win_rate))
        
        # Compute correlation for each feature
        feature_correlations = {}
        for feat, data in feature_data.items():
            if len(data) < 3:
                continue
            mi_vals = np.array([d[0] for d in data])
            win_rates = np.array([d[1] for d in data])
            
            # Pearson correlation
            if np.std(mi_vals) > 0 and np.std(win_rates) > 0:
                corr = np.corrcoef(mi_vals, win_rates)[0, 1]
                feature_correlations[feat] = {
                    'correlation': float(corr),
                    'n_samples': len(data),
                }
        
        correlations[name] = feature_correlations
    
    return correlations


def generate_latex_table(mi_stats: Dict, top_k: int = 5) -> str:
    """
    Generate LaTeX table for MI statistics.
    
    Args:
        mi_stats: MI statistics from compute_mi_statistics
        top_k: Number of top features to include
    
    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mutual Information Between Policy Actions and Game State Features}",
        r"\label{tab:mi_statistics}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Feature & I(action; feature) & Correlation with Win Rate \\",
        r"\midrule",
    ]
    
    for exp_name, stats in mi_stats.items():
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{exp_name}}}}} \\\\")
        
        # Sort features by mean MI
        mi_items = sorted(
            stats['mi_statistics'].items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )[:top_k]
        
        for feat, feat_stats in mi_items:
            feat_display = feat.replace('mi_action_', '').replace('_', ' ').title()
            mean = feat_stats['mean']
            std = feat_stats['std']
            lines.append(f"{feat_display} & ${mean:.4f} \\pm {std:.4f}$ & -- \\\\")
        
        lines.append(r"\midrule")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_summary_report(base_dir: str, output_file: str = None) -> Dict:
    """
    Generate a comprehensive summary report of all experiments.
    
    Args:
        base_dir: Base directory containing experiment outputs
        output_file: Optional file to save JSON report
    
    Returns:
        Dictionary containing all statistics
    """
    print(f"Loading experiments from {base_dir}...")
    experiments = load_all_experiments(base_dir)
    
    if not experiments:
        print("No experiments found!")
        return {}
    
    print(f"Found {len(experiments)} experiment configurations:")
    for name, exp_list in experiments.items():
        print(f"  {name}: {len(exp_list)} seeds")
    
    print("\nComputing statistics...")
    
    # Compute all statistics
    mi_stats = compute_mi_statistics(experiments)
    win_rate_curves = compute_learning_curves(experiments, 'win_rate_overall')
    mi_correlations = compute_mi_correlation_with_performance(experiments)
    
    # Compute MI learning curves for key features
    mi_curves = {}
    key_features = ['mi_action_distance', 'mi_action_agent_x', 'mi_action_agent_hp', 'mi_action_hp_diff']
    for feat in key_features:
        mi_curves[feat] = compute_learning_curves(experiments, feat)
    
    report = {
        'n_experiments': len(experiments),
        'experiment_names': list(experiments.keys()),
        'mi_statistics': mi_stats,
        'win_rate_curves': win_rate_curves,
        'mi_correlations': mi_correlations,
        'mi_learning_curves': mi_curves,
    }
    
    # Generate summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for exp_name, stats in mi_stats.items():
        print(f"\n{exp_name}:")
        print(f"  Seeds: {stats['n_seeds']}")
        print(f"  Win Rate: {stats['win_rate']['mean']:.2%} ± {stats['win_rate']['std']:.2%}")
        
        # Top MI features
        mi_items = sorted(
            stats['mi_statistics'].items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )[:3]
        print("  Top MI features:")
        for feat, feat_stats in mi_items:
            feat_display = feat.replace('mi_action_', '')
            print(f"    {feat_display}: {feat_stats['mean']:.4f} ± {feat_stats['std']:.4f}")
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--input-dir', default='experiments_output',
                        help='Directory containing experiment outputs')
    parser.add_argument('--output', default='experiment_report.json',
                        help='Output file for JSON report')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    
    args = parser.parse_args()
    
    report = generate_summary_report(args.input_dir, args.output)
    
    if args.latex and report:
        latex_table = generate_latex_table(report['mi_statistics'])
        latex_file = args.output.replace('.json', '_table.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {latex_file}")
