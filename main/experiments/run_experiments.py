#!/usr/bin/env python3
"""
Experiment Runner

Runs multiple training experiments with systematic logging for research paper.
Supports parallel execution, checkpointing, and result aggregation.

Usage:
    python run_experiments.py --config short --seeds 42 123 456
    python run_experiments.py --config full --parallel 2
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    ExperimentConfig,
    generate_experiment_matrix,
    MATCHUPS_VS_CPU,
    SEEDS,
    TRAINING_CONFIGS,
)


def run_single_experiment(config: ExperimentConfig, dry_run: bool = False) -> Dict:
    """
    Run a single training experiment.
    
    Args:
        config: Experiment configuration
        dry_run: If True, only print command without executing
    
    Returns:
        Dictionary with experiment results
    """
    # Create directories
    for dir_path in [config.save_dir, config.log_dir, config.interp_log_dir, config.video_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save config
    config_file = os.path.join(config.save_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Build command
    cmd = ["python", "train.py"] + config.to_args()
    cmd_str = " ".join(cmd)
    
    result = {
        'name': f"{config.name}_seed{config.seed}",
        'config': config.to_dict(),
        'command': cmd_str,
        'start_time': datetime.now().isoformat(),
        'status': 'pending',
    }
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}_seed{config.seed}")
    print(f"Matchup: {config.matchup}")
    print(f"Total steps: {config.total_steps:,}")
    print(f"Command: {cmd_str}")
    print(f"{'='*60}\n")
    
    if dry_run:
        result['status'] = 'dry_run'
        return result
    
    # Run training
    try:
        start_time = time.time()
        
        process = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),  # Run from main/ directory
            capture_output=False,  # Stream output to console
            text=True,
        )
        
        elapsed_time = time.time() - start_time
        
        result['end_time'] = datetime.now().isoformat()
        result['elapsed_seconds'] = elapsed_time
        result['return_code'] = process.returncode
        
        if process.returncode == 0:
            result['status'] = 'completed'
            print(f"\n[SUCCESS] Experiment {config.name}_seed{config.seed} completed in {elapsed_time/3600:.2f} hours")
        else:
            result['status'] = 'failed'
            print(f"\n[FAILED] Experiment {config.name}_seed{config.seed} failed with return code {process.returncode}")
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"\n[ERROR] Experiment {config.name}_seed{config.seed}: {e}")
    
    # Save result
    result_file = os.path.join(config.save_dir, "experiment_result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def run_experiment_wrapper(args):
    """Wrapper for parallel execution."""
    config, dry_run = args
    return run_single_experiment(config, dry_run)


def run_all_experiments(
    experiments: List[ExperimentConfig],
    parallel: int = 1,
    dry_run: bool = False,
    resume: bool = False,
) -> List[Dict]:
    """
    Run all experiments in the list.
    
    Args:
        experiments: List of experiment configurations
        parallel: Number of parallel experiments (1 = sequential)
        dry_run: If True, only print commands
        resume: If True, skip completed experiments
    
    Returns:
        List of experiment results
    """
    results = []
    
    if resume:
        # Filter out completed experiments
        remaining = []
        for config in experiments:
            result_file = os.path.join(config.save_dir, "experiment_result.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    result = json.load(f)
                    if result.get('status') == 'completed':
                        print(f"[SKIP] {config.name}_seed{config.seed} already completed")
                        results.append(result)
                        continue
            remaining.append(config)
        experiments = remaining
    
    if not experiments:
        print("No experiments to run.")
        return results
    
    print(f"\nRunning {len(experiments)} experiments...")
    print(f"Parallel workers: {parallel}")
    print(f"Dry run: {dry_run}")
    
    if parallel == 1:
        # Sequential execution
        for config in experiments:
            result = run_single_experiment(config, dry_run)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_experiment_wrapper, (config, dry_run)): config
                for config in experiments
            }
            
            for future in as_completed(futures):
                config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] Experiment {config.name}_seed{config.seed} raised: {e}")
                    results.append({
                        'name': f"{config.name}_seed{config.seed}",
                        'status': 'error',
                        'error': str(e),
                    })
    
    return results


def save_experiment_summary(results: List[Dict], output_dir: str):
    """Save summary of all experiment results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'completed': sum(1 for r in results if r.get('status') == 'completed'),
        'failed': sum(1 for r in results if r.get('status') == 'failed'),
        'errors': sum(1 for r in results if r.get('status') == 'error'),
        'experiments': results,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {summary['total_experiments']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Summary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run training experiments for research paper')
    parser.add_argument('--config', choices=['short', 'medium', 'full'], default='short',
                        help='Training configuration preset')
    parser.add_argument('--matchups', nargs='+', default=None,
                        help='Specific matchups to run (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Specific seeds to run (default: all)')
    parser.add_argument('--output-dir', default='experiments_output',
                        help='Base output directory')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel experiments (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run, skipping completed experiments')
    parser.add_argument('--list', action='store_true',
                        help='List experiment configurations and exit')
    
    args = parser.parse_args()
    
    # Generate experiment matrix
    experiments = generate_experiment_matrix(
        config_name=args.config,
        matchups=args.matchups if args.matchups else None,
        seeds=args.seeds if args.seeds else None,
        base_dir=args.output_dir,
    )
    
    if args.list:
        print(f"\nExperiment Matrix ({args.config} config):")
        print(f"{'='*60}")
        for i, config in enumerate(experiments, 1):
            print(f"{i:3d}. {config.name}_seed{config.seed}")
            print(f"     Matchup: {config.matchup}")
            print(f"     Steps: {config.total_steps:,}, Envs: {config.num_env}")
        print(f"\nTotal: {len(experiments)} experiments")
        return
    
    # Run experiments
    results = run_all_experiments(
        experiments,
        parallel=args.parallel,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    
    # Save summary
    save_experiment_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
