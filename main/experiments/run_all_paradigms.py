#!/usr/bin/env python3
"""
Unified Experiment Runner for ALL Training Paradigms

This script runs experiments across multiple training paradigms:
1. PPO vs CPU (train.py) - Single agent vs game AI
2. Best Response (best_response.py) - Train against frozen opponent  
3. IPPO (ippo.py) - Independent PPO self-play
4. League Training (train_ma.py) - FSP/PSRO league

Each experiment includes:
- Training with interpretability logging
- Post-training analysis (MI, Clustering, Geometry)
- Video recording
- Results aggregation

Usage:
    # Run all priority 1 experiments (core)
    python run_all_paradigms.py --priority 1 --config medium
    
    # Run specific paradigm
    python run_all_paradigms.py --paradigm ppo_vs_cpu --config medium --seeds 42 123
    
    # Dry run to see what would be executed
    python run_all_paradigms.py --priority 1 --dry-run
    
    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=0 python run_all_paradigms.py --paradigm ppo_vs_cpu
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    TrainingParadigm, TRAINING_CONFIGS, PAPER_EXPERIMENTS,
    MATCHUPS_VS_CPU, MATCHUPS_2P, SEEDS
)


def get_script_for_paradigm(paradigm: TrainingParadigm) -> str:
    """Get the training script for a paradigm."""
    mapping = {
        TrainingParadigm.PPO_VS_CPU: "train.py",
        TrainingParadigm.BEST_RESPONSE: "best_response.py",
        TrainingParadigm.IPPO: "ippo.py",
        TrainingParadigm.LEAGUE_FSP: "train_ma.py",
        TrainingParadigm.LEAGUE_PSRO: "train_ma.py",
    }
    return mapping[paradigm]


def build_command(
    paradigm: TrainingParadigm,
    matchup: str,
    name: str,
    seed: int,
    config_name: str,
    output_dir: str,
) -> List[str]:
    """Build command line arguments for a training run."""
    
    config = TRAINING_CONFIGS[config_name]
    script = get_script_for_paradigm(paradigm)
    exp_dir = os.path.join(output_dir, f"{name}_seed{seed}")
    
    # Base command
    cmd = ["python", script]
    
    # Common arguments
    cmd.extend([
        f"--total-steps={config['total_steps']}",
        f"--num-env={config['num_env']}",
        f"--save-dir={os.path.join(exp_dir, 'models')}",
        f"--log-dir={os.path.join(exp_dir, 'logs')}",
        f"--video-dir={os.path.join(exp_dir, 'videos')}",
        f"--model-name-prefix={name}_seed{seed}",
    ])
    
    # Paradigm-specific arguments
    if paradigm == TrainingParadigm.PPO_VS_CPU:
        # Single agent vs CPU
        cmd.extend([
            f"--state={matchup}",
            "--side=left",
            "--enable-interpretability",
            f"--interp-log-dir={os.path.join(exp_dir, 'interpretability')}",
            f"--interp-probe-freq={config['interp_probe_freq']}",
        ])
        
    elif paradigm == TrainingParadigm.BEST_RESPONSE:
        # Best response (2 player, only update left)
        cmd.extend([
            f"--seed={seed}",
            "--side=both",
            "--update-left=1",
            "--update-right=0",  # Freeze right player
            "--enable-interpretability",
            f"--interp-log-dir={os.path.join(exp_dir, 'interpretability')}",
            f"--interp-probe-freq={config['interp_probe_freq']}",
        ])
        
    elif paradigm == TrainingParadigm.IPPO:
        # Independent PPO self-play
        cmd.extend([
            f"--seed={seed}",
            "--side=both",
            "--update-left=1",
            "--update-right=1",
            "--enable-interpretability",
            f"--interp-log-dir={os.path.join(exp_dir, 'interpretability')}",
            f"--interp-probe-freq={config['interp_probe_freq']}",
        ])
        
    elif paradigm == TrainingParadigm.LEAGUE_FSP:
        # FSP League
        cmd.extend([
            f"--seed={seed}",
            "--side=both",
            "--fsp-league",
        ])
        
    elif paradigm == TrainingParadigm.LEAGUE_PSRO:
        # PSRO League
        cmd.extend([
            f"--seed={seed}",
            "--side=both",
            "--psro-league",
        ])
    
    return cmd


def run_experiment(
    paradigm: TrainingParadigm,
    matchup: str,
    name: str,
    seed: int,
    config_name: str,
    output_dir: str,
    dry_run: bool = False,
) -> bool:
    """Run a single experiment."""
    
    cmd = build_command(paradigm, matchup, name, seed, config_name, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Experiment: {name}_seed{seed}")
    print(f"Paradigm: {paradigm.value}")
    print(f"Matchup: {matchup}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")
    
    if dry_run:
        print(f"[DRY RUN] Would execute:\n  {' '.join(cmd)}")
        return True
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            # Inherit stdout/stderr for real-time output
        )
        
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] Completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            return True
        else:
            print(f"\n[FAILED] Return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False


def run_post_analysis(exp_dir: str, matchup: str, dry_run: bool = False) -> bool:
    """Run post-training interpretability analysis."""
    
    # Find the model file
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        print(f"[SKIP] No models directory: {models_dir}")
        return False
    
    # Find latest model
    model_files = sorted(Path(models_dir).glob("*.zip"))
    if not model_files:
        print(f"[SKIP] No model files in: {models_dir}")
        return False
    
    model_path = str(model_files[-1])
    analysis_dir = os.path.join(exp_dir, 'post_analysis')
    
    cmd = [
        "python", "-m", "interpretability.analysis_runner",
        f"--model-path={model_path}",
        f"--state={matchup}",
        f"--output={analysis_dir}",
        "--n-episodes=20",
        "--n-clusters=8",
    ]
    
    print(f"\n[ANALYSIS] Running post-training analysis...")
    
    if dry_run:
        print(f"[DRY RUN] Would execute:\n  {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run experiments across all paradigms')
    
    # Experiment selection
    parser.add_argument('--priority', type=int, choices=[1, 2, 3],
                       help='Run experiments by priority level (1=core, 2=multi-agent, 3=if time)')
    parser.add_argument('--paradigm', type=str, 
                       choices=['ppo_vs_cpu', 'best_response', 'ippo', 'league_fsp', 'league_psro'],
                       help='Run specific paradigm only')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments (priority 1, 2, and 3)')
    
    # Configuration
    parser.add_argument('--config', type=str, default='medium',
                       choices=['short', 'medium', 'full'],
                       help='Training configuration')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Override seeds (default: from config)')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory for results')
    
    # Control
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip post-training analysis')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    experiments_to_run = []
    
    if args.paradigm:
        # Run specific paradigm
        paradigm_map = {
            'ppo_vs_cpu': TrainingParadigm.PPO_VS_CPU,
            'best_response': TrainingParadigm.BEST_RESPONSE,
            'ippo': TrainingParadigm.IPPO,
            'league_fsp': TrainingParadigm.LEAGUE_FSP,
            'league_psro': TrainingParadigm.LEAGUE_PSRO,
        }
        paradigm = paradigm_map[args.paradigm]
        
        # Use appropriate matchup
        if paradigm == TrainingParadigm.PPO_VS_CPU:
            matchups = MATCHUPS_VS_CPU
        else:
            matchups = MATCHUPS_2P
        
        seeds = args.seeds if args.seeds else [42, 123]
        
        for matchup in matchups:
            for seed in seeds:
                name = f"{args.paradigm}_{matchup.split('.')[-1].lower()}"
                experiments_to_run.append({
                    'paradigm': paradigm,
                    'matchup': matchup,
                    'name': name,
                    'seed': seed,
                })
    
    elif args.priority or args.all:
        # Run by priority
        priorities = [args.priority] if args.priority else [1, 2, 3]
        
        for p in priorities:
            key = f'priority_{p}'
            if key in PAPER_EXPERIMENTS:
                for exp in PAPER_EXPERIMENTS[key]:
                    seeds = args.seeds if args.seeds else exp['seeds']
                    for seed in seeds:
                        experiments_to_run.append({
                            'paradigm': exp['paradigm'],
                            'matchup': exp['matchup'],
                            'name': exp['name'],
                            'seed': seed,
                        })
    
    else:
        parser.print_help()
        print("\n[ERROR] Specify --priority, --paradigm, or --all")
        return
    
    # Print experiment plan
    print("\n" + "="*70)
    print("EXPERIMENT PLAN")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Total steps: {TRAINING_CONFIGS[args.config]['total_steps']:,}")
    print(f"Output dir: {args.output_dir}")
    print(f"Experiments: {len(experiments_to_run)}")
    print("-"*70)
    
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"  {i}. {exp['name']}_seed{exp['seed']} ({exp['paradigm'].value})")
    
    print("="*70 + "\n")
    
    if not args.dry_run:
        input("Press Enter to start, or Ctrl+C to cancel...")
    
    # Run experiments
    results = []
    output_dir = os.path.abspath(os.path.join(
        Path(__file__).parent.parent, args.output_dir
    ))
    
    for exp in experiments_to_run:
        # Training
        success = run_experiment(
            paradigm=exp['paradigm'],
            matchup=exp['matchup'],
            name=exp['name'],
            seed=exp['seed'],
            config_name=args.config,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        
        # Post-analysis (if training succeeded)
        if success and not args.skip_analysis and not args.dry_run:
            exp_dir = os.path.join(output_dir, f"{exp['name']}_seed{exp['seed']}")
            run_post_analysis(exp_dir, exp['matchup'], dry_run=args.dry_run)
        
        results.append({
            'experiment': f"{exp['name']}_seed{exp['seed']}",
            'paradigm': exp['paradigm'].value,
            'success': success,
        })
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Completed: {success_count}/{len(results)}")
    
    for r in results:
        status = "[OK]" if r['success'] else "[FAIL]"
        print(f"  {status} {r['experiment']} ({r['paradigm']})")
    
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
