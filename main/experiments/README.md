# Experiment Framework for Research Paper

This module provides a systematic framework for running RL training experiments with comprehensive interpretability logging. It is designed to generate reproducible results with statistical rigor for peer-reviewed publication.

## Quick Start

### 1. Run Experiments

```bash
# List available experiments
python experiments/run_experiments.py --list --config short

# Dry run (show commands without executing)
python experiments/run_experiments.py --config short --dry-run

# Run 2 seeds with short config (500K steps each)
python experiments/run_experiments.py --config short --seeds 42 123

# Run all 5 seeds (full experiment)
python experiments/run_experiments.py --config short

# Resume interrupted experiments
python experiments/run_experiments.py --config short --resume
```

### 2. Training Configurations

| Config | Steps | Envs | MI Analysis Freq | Use Case |
|--------|-------|------|------------------|----------|
| `short` | 500K | 8 | 25K | Testing/debugging |
| `medium` | 2M | 8 | 100K | Preliminary results |
| `full` | 10M | 16 | 500K | Final paper results |

### 3. Aggregate Results

After experiments complete:

```bash
# Generate summary statistics
python experiments/aggregate_results.py \
    --input-dir experiments_output \
    --output experiment_report.json \
    --latex
```

### 4. Generate Figures

```bash
python experiments/plot_figures.py \
    --report experiment_report.json \
    --output-dir figures
```

## Output Structure

```
experiments_output/
├── experiment_summary.json          # Overall summary
├── ryu_vs_guile_seed42/
│   ├── models/
│   │   ├── config.json              # Experiment configuration
│   │   ├── ppo_ryu_*.zip           # Model checkpoints
│   │   └── experiment_result.json   # Run result
│   ├── logs/
│   │   └── PPO_*/events.out.*      # TensorBoard logs
│   ├── interpretability/
│   │   ├── mi_analysis_*.json      # Per-analysis MI results
│   │   ├── mi_history.json         # Full MI evolution
│   │   └── eval/
│   │       ├── eval_*.json         # Per-evaluation results
│   │       └── eval_history.json   # Full evaluation history
│   └── videos/
│       └── episode_*.mp4           # Evaluation videos
└── ryu_vs_guile_seed123/
    └── ...
```

## Metrics Collected

### RL Performance Metrics
- **Win Rate**: Fraction of episodes where agent_hp > enemy_hp at termination
- **Episode Reward**: Mean ± std reward per episode
- **Episode Length**: Mean ± std steps per episode
- **HP Differential**: Mean health advantage at episode end

### Interpretability Metrics
- **MI(action; feature)**: Mutual information between policy actions and game state features
- **Activation Statistics**: Mean, std, sparsity, dead neurons
- **Action Entropy**: Diversity of action selection
- **Feature Distributions**: Statistics of ground truth features

### Ground Truth Features (from RAM)
| Feature | Description |
|---------|-------------|
| `agent_x` | Agent horizontal position (0-256) |
| `enemy_x` | Enemy horizontal position (0-256) |
| `distance` | Absolute distance between agents |
| `relative_position` | Signed distance (agent_x - enemy_x) |
| `agent_hp` | Agent health points (0-176) |
| `enemy_hp` | Enemy health points (0-176) |
| `hp_diff` | Health differential |
| `hp_ratio` | Health ratio |
| `timer` | Round timer (0-99) |

## Running on cvpc

```bash
# SSH to cvpc
ssh cvpc

# Attach to tmux session
tmux attach -t fl

# Navigate to project
cd /mnt/code/Reaz/FightLadder-SF2-DL/main

# Pull latest changes
git pull

# Activate environment
conda activate fightladder

# Run experiments (NO --render flag on headless server)
python experiments/run_experiments.py --config short --seeds 42 123
```

## Single Training Run

For manual control over a single training run:

```bash
python train.py \
    --state Champion.Level1.RyuVsGuile \
    --total-steps 500000 \
    --num-env 8 \
    --enable-interpretability \
    --interp-log-dir interpretability_logs \
    --interp-probe-freq 25000
```

## Viewing Results

### TensorBoard
```bash
tensorboard --logdir=experiments_output
# Then open http://localhost:6006
```

### JSON Reports
```bash
# View MI history
cat experiments_output/ryu_vs_guile_seed42/interpretability/mi_history.json | python -m json.tool

# View evaluation history
cat experiments_output/ryu_vs_guile_seed42/interpretability/eval/eval_history.json | python -m json.tool
```

## Paper Statistics

The aggregate_results.py script computes:

1. **MI Statistics Table**: Mean ± std MI for each feature across seeds
2. **Learning Curves**: Win rate and MI evolution with confidence intervals
3. **MI-Performance Correlation**: How MI values correlate with win rate
4. **LaTeX Tables**: Ready-to-use tables for the paper

Example LaTeX output:
```latex
\begin{table}[t]
\centering
\caption{Mutual Information Between Policy Actions and Game State Features}
\begin{tabular}{lcc}
\toprule
Feature & I(action; feature) & Correlation \\
\midrule
Distance & $0.0042 \pm 0.0008$ & 0.73 \\
Agent X & $0.0038 \pm 0.0005$ & 0.68 \\
...
\bottomrule
\end{tabular}
\end{table}
```
