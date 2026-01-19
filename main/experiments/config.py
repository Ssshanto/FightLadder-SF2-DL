"""
Experiment Configuration for Research Paper

Defines the experiment matrix for systematic evaluation of:
1. RL training performance (win rate, rewards, episode length)
2. Interpretability metrics (MI evolution, feature dependencies)

Supports multiple training paradigms:
- Single-agent PPO vs CPU (train.py)
- Best Response vs frozen opponent (best_response.py)
- Independent PPO self-play (ippo.py)
- League training with FSP/PSRO (train_ma.py)

Designed for reproducibility and statistical rigor.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import os


# =============================================================================
# TRAINING PARADIGMS
# =============================================================================

class TrainingParadigm(Enum):
    """Available training paradigms."""
    PPO_VS_CPU = "ppo_vs_cpu"           # Single-agent vs game AI (train.py)
    BEST_RESPONSE = "best_response"      # Best response vs frozen opponent (best_response.py)
    IPPO = "ippo"                        # Independent PPO self-play (ippo.py)
    LEAGUE_FSP = "league_fsp"            # Fictitious Self-Play league (train_ma.py)
    LEAGUE_PSRO = "league_psro"          # PSRO league (train_ma.py)


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Available matchups - single player vs CPU
MATCHUPS_VS_CPU = [
    "Champion.Level1.RyuVsGuile",       # Projectile vs projectile
    "Champion.Level1.RyuVsRyu",         # Mirror match (if available)
]

# Available matchups - 2 player mode
MATCHUPS_2P = [
    "Champion.RyuVsRyu.2Player.align",  # Mirror match self-play
]

# Characters available (for reference)
CHARACTERS = [
    'Ryu', 'Ken', 'Honda', 'Blanka', 'Guile', 'Balrog',
    'Vega', 'Chunli', 'Zangief', 'Dhalsim', 'Sagat', 'Bison'
]

# Random seeds for reproducibility (multiple runs per config)
SEEDS = [42, 123, 456, 789, 1024]

# Training configurations
TRAINING_CONFIGS = {
    'minimal': {
        'total_steps': 10_000,       # ~1-2 minutes - just for validation
        'num_env': 2,
        'interp_probe_freq': 5_000,  # 2 MI analyses per run
        'checkpoint_freq': 5_000,
    },
    'short': {
        'total_steps': 500_000,
        'num_env': 8,
        'interp_probe_freq': 25_000,  # 20 MI analyses per run
        'checkpoint_freq': 50_000,
    },
    'medium': {
        'total_steps': 2_000_000,
        'num_env': 8,
        'interp_probe_freq': 100_000,  # 20 MI analyses per run
        'checkpoint_freq': 200_000,
    },
    'full': {
        'total_steps': 10_000_000,
        'num_env': 16,
        'interp_probe_freq': 500_000,  # 20 MI analyses per run
        'checkpoint_freq': 1_000_000,
    }
}

# =============================================================================
# PAPER EXPERIMENT MATRIX (8-10 hour deadline with 2 GPUs)
# =============================================================================
# Priority 1 (Core - ~3hr each, run in parallel on 2 GPUs):
#   - PPO vs Guile: 2M steps × 2 seeds
#   - PPO vs Ryu (mirror): 2M steps × 2 seeds
#
# Priority 2 (Multi-agent - ~1.5hr each):
#   - IPPO Self-Play: 2M steps × 1 seed
#   - Best Response: 2M steps × 1 seed
#
# Total estimated time: ~5-6 hours with 2 GPUs in parallel

PAPER_EXPERIMENTS = {
    'priority_1': [
        # Core single-agent experiments (must have)
        {
            'paradigm': TrainingParadigm.PPO_VS_CPU,
            'matchup': 'Champion.Level1.RyuVsGuile',
            'name': 'ppo_ryu_vs_guile',
            'seeds': [42, 123],
            'config': 'medium',
        },
        {
            'paradigm': TrainingParadigm.PPO_VS_CPU,
            'matchup': 'Champion.Level1.RyuVsRyu',
            'name': 'ppo_ryu_vs_ryu_cpu',
            'seeds': [42, 123],
            'config': 'medium',
        },
    ],
    'priority_2': [
        # Multi-agent experiments (important for paper)
        {
            'paradigm': TrainingParadigm.IPPO,
            'matchup': 'Champion.RyuVsRyu.2Player.align',
            'name': 'ippo_self_play',
            'seeds': [42],
            'config': 'medium',
        },
        {
            'paradigm': TrainingParadigm.BEST_RESPONSE,
            'matchup': 'Champion.RyuVsRyu.2Player.align',
            'name': 'best_response',
            'seeds': [42],
            'config': 'medium',
        },
    ],
    'priority_3': [
        # League training (if time permits)
        {
            'paradigm': TrainingParadigm.LEAGUE_FSP,
            'matchup': 'Champion.RyuVsRyu.2Player.align',
            'name': 'fsp_league',
            'seeds': [42],
            'config': 'short',  # Shorter for time constraints
        },
    ],
}

# Default config for experiments
DEFAULT_CONFIG = 'short'  # Use 'short' for testing, 'full' for paper


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Identification
    name: str
    seed: int
    matchup: str = "Champion.Level1.RyuVsGuile"
    
    # Training parameters
    total_steps: int = 500_000
    num_env: int = 8
    side: str = 'left'
    
    # Model architecture (PPO defaults)
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 4
    gamma: float = 0.94
    
    # Frame stacking
    num_stack: int = 12
    num_step_frames: int = 8
    
    # Action space
    enable_combo: bool = False
    null_combo: bool = False
    transform_action: bool = False
    
    # Interpretability
    enable_interpretability: bool = True
    interp_probe_freq: int = 25_000
    interp_log_freq: int = 1000
    
    # Logging/saving
    checkpoint_freq: int = 50_000
    eval_episodes: int = 20
    eval_freq: int = 100_000  # Evaluate every N steps
    
    # Directories (will be set based on experiment name)
    base_dir: str = "experiments_output"
    save_dir: str = field(default="", init=False)
    log_dir: str = field(default="", init=False)
    interp_log_dir: str = field(default="", init=False)
    video_dir: str = field(default="", init=False)
    
    def __post_init__(self):
        """Set up directory paths after initialization."""
        exp_name = f"{self.name}_seed{self.seed}"
        self.save_dir = os.path.join(self.base_dir, exp_name, "models")
        self.log_dir = os.path.join(self.base_dir, exp_name, "logs")
        self.interp_log_dir = os.path.join(self.base_dir, exp_name, "interpretability")
        self.video_dir = os.path.join(self.base_dir, exp_name, "videos")
    
    def to_args(self) -> List[str]:
        """Convert config to command line arguments for train.py."""
        args = [
            f"--state={self.matchup}",
            f"--side={self.side}",
            f"--total-steps={self.total_steps}",
            f"--num-env={self.num_env}",
            f"--num-stack={self.num_stack}",
            f"--num-step-frames={self.num_step_frames}",
            f"--save-dir={self.save_dir}",
            f"--log-dir={self.log_dir}",
            f"--video-dir={self.video_dir}",
            f"--model-name-prefix={self.name}_seed{self.seed}",
            f"--num-episodes={self.eval_episodes}",
        ]
        
        if self.enable_combo:
            args.append("--enable-combo")
        if self.null_combo:
            args.append("--null-combo")
        if self.transform_action:
            args.append("--transform-action")
            
        if self.enable_interpretability:
            args.extend([
                "--enable-interpretability",
                f"--interp-log-dir={self.interp_log_dir}",
                f"--interp-probe-freq={self.interp_probe_freq}",
                f"--interp-log-freq={self.interp_log_freq}",
            ])
        
        return args
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'seed': self.seed,
            'matchup': self.matchup,
            'total_steps': self.total_steps,
            'num_env': self.num_env,
            'side': self.side,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'num_stack': self.num_stack,
            'num_step_frames': self.num_step_frames,
            'enable_combo': self.enable_combo,
            'enable_interpretability': self.enable_interpretability,
            'interp_probe_freq': self.interp_probe_freq,
            'checkpoint_freq': self.checkpoint_freq,
            'eval_episodes': self.eval_episodes,
        }


def generate_experiment_matrix(
    config_name: str = DEFAULT_CONFIG,
    matchups: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    base_dir: str = "experiments_output"
) -> List[ExperimentConfig]:
    """
    Generate the full experiment matrix.
    
    Args:
        config_name: Name of training config ('short', 'medium', 'full')
        matchups: List of matchup states to test
        seeds: List of random seeds for multiple runs
        base_dir: Base directory for all experiment outputs
    
    Returns:
        List of ExperimentConfig objects
    """
    if matchups is None:
        matchups = MATCHUPS_VS_CPU
    if seeds is None:
        seeds = SEEDS
    
    config_params = TRAINING_CONFIGS[config_name]
    experiments = []
    
    for matchup in matchups:
        # Extract opponent name for experiment naming
        # e.g., "Champion.Level1.RyuVsGuile" -> "ryu_vs_guile"
        parts = matchup.split('.')[-1]  # "RyuVsGuile"
        exp_base_name = parts.lower().replace('vs', '_vs_')
        
        for seed in seeds:
            config = ExperimentConfig(
                name=exp_base_name,
                seed=seed,
                matchup=matchup,
                total_steps=config_params['total_steps'],
                num_env=config_params['num_env'],
                interp_probe_freq=config_params['interp_probe_freq'],
                checkpoint_freq=config_params['checkpoint_freq'],
                base_dir=base_dir,
            )
            experiments.append(config)
    
    return experiments


# =============================================================================
# METRICS TO COLLECT
# =============================================================================

# RL Performance Metrics
RL_METRICS = [
    'episode_reward_mean',
    'episode_reward_std', 
    'episode_length_mean',
    'win_rate',
    'loss/policy_loss',
    'loss/value_loss',
    'loss/entropy_loss',
]

# Interpretability Metrics (from MI analysis)
INTERP_METRICS = [
    'mi_action_agent_x',
    'mi_action_enemy_x',
    'mi_action_distance',
    'mi_action_relative_position',
    'mi_action_agent_hp',
    'mi_action_enemy_hp',
    'mi_action_hp_diff',
    'mi_action_hp_ratio',
    'mi_action_timer',
]

# Ground truth features for analysis
GROUND_TRUTH_FEATURES = {
    'agent_x': 'Agent horizontal position (0-256)',
    'enemy_x': 'Enemy horizontal position (0-256)',
    'distance': 'Absolute distance between agents',
    'relative_position': 'Signed distance (agent_x - enemy_x)',
    'agent_hp': 'Agent health points (0-176)',
    'enemy_hp': 'Enemy health points (0-176)',
    'hp_diff': 'Health differential (agent_hp - enemy_hp)',
    'hp_ratio': 'Health ratio (agent_hp / total_hp)',
    'total_hp': 'Combined health points',
    'timer': 'Round timer (0-99)',
}
