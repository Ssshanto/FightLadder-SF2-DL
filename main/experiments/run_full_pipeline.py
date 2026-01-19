#!/usr/bin/env python3
"""
Full Experiment Pipeline for Research Paper

This script orchestrates the complete experiment workflow:
1. Training with runtime interpretability logging
2. Post-training full interpretability analysis (MI, Clustering, Geometry)
3. Video/screenshot generation
4. Results aggregation across seeds
5. Paper figure and table generation

Usage:
    # Run everything
    python run_full_pipeline.py --config short --seeds 42 123 456
    
    # Just analysis on existing runs
    python run_full_pipeline.py --analyze-only --input-dir results
    
    # Generate paper assets from aggregated results
    python run_full_pipeline.py --figures-only --input-dir results
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pickle

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_training_experiment(config_dict: Dict, output_dir: str, dry_run: bool = False) -> bool:
    """Run a single training experiment."""
    
    cmd = [
        "python", "train.py",
        f"--state={config_dict['matchup']}",
        f"--side={config_dict['side']}",
        f"--total-steps={config_dict['total_steps']}",
        f"--num-env={config_dict['num_env']}",
        f"--save-dir={os.path.join(output_dir, 'models')}",
        f"--log-dir={os.path.join(output_dir, 'logs')}",
        f"--video-dir={os.path.join(output_dir, 'videos')}",
        f"--model-name-prefix={config_dict['name']}_seed{config_dict['seed']}",
        "--enable-interpretability",
        f"--interp-log-dir={os.path.join(output_dir, 'interpretability')}",
        f"--interp-probe-freq={config_dict['interp_probe_freq']}",
    ]
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    print(f"\n{'='*60}")
    print(f"Starting training: {config_dict['name']}_seed{config_dict['seed']}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        return result.returncode == 0
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def run_post_training_analysis(
    model_path: str,
    output_dir: str,
    state: str = "Champion.Level1.RyuVsGuile",
    n_episodes: int = 20,
    n_clusters: int = 8,
) -> bool:
    """
    Run full interpretability analysis on a trained model.
    
    This runs all three modules:
    - Module A: Mutual Information Analysis
    - Module B: Unsupervised Concept Discovery
    - Module D: Representation Geometry Analysis
    """
    
    analysis_dir = os.path.join(output_dir, 'post_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "interpretability.analysis_runner",
        f"--model-path={model_path}",
        f"--state={state}",
        f"--output={analysis_dir}",
        f"--n-episodes={n_episodes}",
        f"--n-clusters={n_clusters}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running post-training analysis")
    print(f"Model: {model_path}")
    print(f"Output: {analysis_dir}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        return result.returncode == 0
    except Exception as e:
        print(f"Analysis failed: {e}")
        return False


def record_evaluation_videos(
    model_path: str,
    output_dir: str,
    state: str = "Champion.Level1.RyuVsGuile",
    n_episodes: int = 5,
) -> bool:
    """Record high-quality evaluation videos."""
    
    video_dir = os.path.join(output_dir, 'videos', 'evaluation')
    os.makedirs(video_dir, exist_ok=True)
    
    cmd = [
        "python", "train.py",
        "--eval-only",
        f"--model-file={model_path}",
        f"--state={state}",
        f"--video-dir={video_dir}",
        f"--num-episodes={n_episodes}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Recording evaluation videos")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        return result.returncode == 0
    except Exception as e:
        print(f"Video recording failed: {e}")
        return False


def generate_experiment_config(
    name: str,
    seed: int,
    config_preset: str = 'short',
    matchup: str = "Champion.Level1.RyuVsGuile",
) -> Dict:
    """Generate experiment configuration."""
    
    presets = {
        'short': {
            'total_steps': 500_000,
            'num_env': 8,
            'interp_probe_freq': 25_000,
        },
        'medium': {
            'total_steps': 2_000_000,
            'num_env': 8,
            'interp_probe_freq': 100_000,
        },
        'full': {
            'total_steps': 10_000_000,
            'num_env': 16,
            'interp_probe_freq': 500_000,
        }
    }
    
    preset = presets[config_preset]
    
    return {
        'name': name,
        'seed': seed,
        'matchup': matchup,
        'side': 'left',
        **preset,
    }


def find_final_model(experiment_dir: str) -> Optional[str]:
    """Find the final trained model in an experiment directory."""
    models_dir = os.path.join(experiment_dir, 'models')
    if not os.path.exists(models_dir):
        return None
    
    # Look for final model first
    for f in os.listdir(models_dir):
        if 'final' in f and f.endswith('.zip'):
            return os.path.join(models_dir, f)
    
    # Otherwise find latest checkpoint
    checkpoints = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not checkpoints:
        return None
    
    # Sort by step number
    def get_step(name):
        try:
            return int(name.split('_')[-2])
        except:
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return os.path.join(models_dir, checkpoints[0])


def aggregate_results(results_dir: str, output_dir: str) -> Dict:
    """Aggregate results from all experiments."""
    
    from experiments.aggregate_results import (
        load_all_experiments,
        compute_mi_statistics,
        compute_learning_curves,
        compute_mi_correlation_with_performance,
        generate_summary_report,
    )
    
    print(f"\n{'='*60}")
    print("Aggregating results across all experiments")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    report = generate_summary_report(
        results_dir,
        output_file=os.path.join(output_dir, 'experiment_report.json')
    )
    
    return report


def generate_paper_figures(report: Dict, output_dir: str):
    """Generate all figures for the paper."""
    
    from experiments.plot_figures import plot_combined_summary
    
    print(f"\n{'='*60}")
    print("Generating paper figures")
    print(f"{'='*60}")
    
    figures_dir = os.path.join(output_dir, 'paper_figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_combined_summary(report, figures_dir, prefix='paper')


def generate_latex_tables(report: Dict, output_dir: str):
    """Generate LaTeX tables for the paper."""
    
    print(f"\n{'='*60}")
    print("Generating LaTeX tables")
    print(f"{'='*60}")
    
    tables_dir = os.path.join(output_dir, 'latex_tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # MI Statistics Table
    mi_table = generate_mi_latex_table(report.get('mi_statistics', {}))
    with open(os.path.join(tables_dir, 'mi_statistics.tex'), 'w') as f:
        f.write(mi_table)
    
    # Win Rate Table
    wr_table = generate_winrate_latex_table(report.get('mi_statistics', {}))
    with open(os.path.join(tables_dir, 'win_rates.tex'), 'w') as f:
        f.write(wr_table)
    
    print(f"LaTeX tables saved to {tables_dir}")


def generate_mi_latex_table(mi_stats: Dict) -> str:
    """Generate LaTeX table for MI statistics."""
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mutual Information Between Policy Actions and Game State Features. "
        r"Values represent I(action; feature) in bits, averaged across seeds with standard deviation.}",
        r"\label{tab:mi_statistics}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Feature & MI (bits) & Normalized MI \\",
        r"\midrule",
    ]
    
    for exp_name, stats in mi_stats.items():
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{exp_name.replace('_', ' ').title()}}}}} \\\\")
        
        mi_items = sorted(
            stats['mi_statistics'].items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )[:8]  # Top 8
        
        for feat, feat_stats in mi_items:
            feat_display = feat.replace('mi_action_', '').replace('_', ' ').title()
            mean = feat_stats['mean']
            std = feat_stats['std']
            # Assume normalized is roughly mean/max_entropy
            lines.append(f"{feat_display} & ${mean:.4f} \\pm {std:.4f}$ & -- \\\\")
        
        lines.append(r"\midrule")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_winrate_latex_table(mi_stats: Dict) -> str:
    """Generate LaTeX table for win rates."""
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training Performance Across Random Seeds}",
        r"\label{tab:win_rates}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & Seeds & Win Rate & Std \\",
        r"\midrule",
    ]
    
    for exp_name, stats in mi_stats.items():
        exp_display = exp_name.replace('_', ' ').title()
        n_seeds = stats['n_seeds']
        wr_mean = stats['win_rate']['mean']
        wr_std = stats['win_rate']['std']
        lines.append(f"{exp_display} & {n_seeds} & {wr_mean:.1%} & {wr_std:.1%} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Full experiment pipeline for research paper')
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--analyze', action='store_true', help='Run post-training analysis')
    parser.add_argument('--videos', action='store_true', help='Record evaluation videos')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results')
    parser.add_argument('--figures', action='store_true', help='Generate paper figures')
    parser.add_argument('--tables', action='store_true', help='Generate LaTeX tables')
    parser.add_argument('--all', action='store_true', help='Run everything')
    
    # Shortcuts
    parser.add_argument('--analyze-only', action='store_true', 
                        help='Only run analysis on existing experiments')
    parser.add_argument('--figures-only', action='store_true',
                        help='Only generate figures from existing results')
    
    # Configuration
    parser.add_argument('--config', choices=['short', 'medium', 'full'], default='short',
                        help='Training configuration preset')
    parser.add_argument('--matchup', default='Champion.Level1.RyuVsGuile',
                        help='Game state/matchup')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--output-dir', default='../results',
                        help='Base output directory')
    parser.add_argument('--input-dir', default=None,
                        help='Input directory for analysis (defaults to output-dir)')
    
    # Analysis parameters
    parser.add_argument('--analysis-episodes', type=int, default=20,
                        help='Episodes for post-training analysis')
    parser.add_argument('--video-episodes', type=int, default=5,
                        help='Episodes to record')
    parser.add_argument('--n-clusters', type=int, default=8,
                        help='Number of clusters for concept discovery')
    
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.all:
        args.train = args.analyze = args.videos = args.aggregate = args.figures = args.tables = True
    if args.analyze_only:
        args.analyze = args.aggregate = args.figures = args.tables = True
    if args.figures_only:
        args.aggregate = args.figures = args.tables = True
    
    # Default: if nothing specified, run everything
    if not any([args.train, args.analyze, args.videos, args.aggregate, args.figures, args.tables]):
        args.train = args.analyze = args.videos = args.aggregate = args.figures = args.tables = True
    
    input_dir = args.input_dir or args.output_dir
    output_dir = args.output_dir
    
    # Make paths absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(str(Path(__file__).parent.parent), output_dir)
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(str(Path(__file__).parent.parent), input_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract experiment name from matchup
    matchup_name = args.matchup.split('.')[-1].lower().replace('vs', '_vs_')
    
    print(f"\n{'#'*60}")
    print(f"# FULL EXPERIMENT PIPELINE")
    print(f"# Config: {args.config}")
    print(f"# Matchup: {args.matchup}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}\n")
    
    experiment_dirs = []
    
    # =========================================================================
    # PHASE 1: Training
    # =========================================================================
    if args.train:
        print("\n" + "="*60)
        print("PHASE 1: TRAINING")
        print("="*60)
        
        for seed in args.seeds:
            config = generate_experiment_config(
                name=matchup_name,
                seed=seed,
                config_preset=args.config,
                matchup=args.matchup,
            )
            
            exp_dir = os.path.join(output_dir, f"{matchup_name}_seed{seed}")
            experiment_dirs.append(exp_dir)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save config
            config_path = os.path.join(exp_dir, 'models', 'config.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            success = run_training_experiment(config, exp_dir, dry_run=args.dry_run)
            
            if not success and not args.dry_run:
                print(f"[WARNING] Training failed for seed {seed}")
    else:
        # Find existing experiment directories
        for seed in args.seeds:
            exp_dir = os.path.join(input_dir, f"{matchup_name}_seed{seed}")
            if os.path.exists(exp_dir):
                experiment_dirs.append(exp_dir)
    
    # =========================================================================
    # PHASE 2: Post-Training Analysis
    # =========================================================================
    if args.analyze and not args.dry_run:
        print("\n" + "="*60)
        print("PHASE 2: POST-TRAINING ANALYSIS")
        print("="*60)
        
        for exp_dir in experiment_dirs:
            model_path = find_final_model(exp_dir)
            if model_path:
                run_post_training_analysis(
                    model_path=model_path,
                    output_dir=exp_dir,
                    state=args.matchup,
                    n_episodes=args.analysis_episodes,
                    n_clusters=args.n_clusters,
                )
            else:
                print(f"[WARNING] No model found in {exp_dir}")
    
    # =========================================================================
    # PHASE 3: Video Recording
    # =========================================================================
    if args.videos and not args.dry_run:
        print("\n" + "="*60)
        print("PHASE 3: VIDEO RECORDING")
        print("="*60)
        
        for exp_dir in experiment_dirs:
            model_path = find_final_model(exp_dir)
            if model_path:
                record_evaluation_videos(
                    model_path=model_path,
                    output_dir=exp_dir,
                    state=args.matchup,
                    n_episodes=args.video_episodes,
                )
    
    # =========================================================================
    # PHASE 4: Results Aggregation
    # =========================================================================
    aggregated_dir = os.path.join(output_dir, 'aggregated')
    report = {}
    
    if args.aggregate and not args.dry_run:
        print("\n" + "="*60)
        print("PHASE 4: RESULTS AGGREGATION")
        print("="*60)
        
        report = aggregate_results(input_dir, aggregated_dir)
    elif os.path.exists(os.path.join(aggregated_dir, 'experiment_report.json')):
        with open(os.path.join(aggregated_dir, 'experiment_report.json')) as f:
            report = json.load(f)
    
    # =========================================================================
    # PHASE 5: Paper Figures
    # =========================================================================
    if args.figures and report and not args.dry_run:
        print("\n" + "="*60)
        print("PHASE 5: PAPER FIGURES")
        print("="*60)
        
        generate_paper_figures(report, aggregated_dir)
    
    # =========================================================================
    # PHASE 6: LaTeX Tables
    # =========================================================================
    if args.tables and report and not args.dry_run:
        print("\n" + "="*60)
        print("PHASE 6: LATEX TABLES")
        print("="*60)
        
        generate_latex_tables(report, aggregated_dir)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    print(f"\nResults saved to: {output_dir}")
    if experiment_dirs:
        print(f"Experiments: {len(experiment_dirs)}")
    if os.path.exists(aggregated_dir):
        print(f"Aggregated results: {aggregated_dir}")
        if os.path.exists(os.path.join(aggregated_dir, 'paper_figures')):
            print(f"Paper figures: {os.path.join(aggregated_dir, 'paper_figures')}")
        if os.path.exists(os.path.join(aggregated_dir, 'latex_tables')):
            print(f"LaTeX tables: {os.path.join(aggregated_dir, 'latex_tables')}")


if __name__ == "__main__":
    main()
