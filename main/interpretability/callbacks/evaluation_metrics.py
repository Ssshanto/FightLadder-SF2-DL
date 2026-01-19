"""
Evaluation Callback for Comprehensive RL Metrics

Periodically evaluates the policy against the training environment and logs:
- Win rate (computed from episode outcomes)
- Episode reward statistics
- Episode length statistics
- Health differentials

These metrics are essential for quantifying training progress in the paper.
"""

import os
import json
import time
from typing import Dict, List, Optional, Callable
from collections import deque

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class EvaluationMetricsCallback(BaseCallback):
    """
    Callback for tracking comprehensive RL performance metrics.
    
    Logs to both TensorBoard and JSON files for paper statistics.
    
    Args:
        log_dir: Directory to save evaluation logs
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Number of episodes per evaluation
        deterministic: Whether to use deterministic actions during eval
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        log_dir: str = "eval_logs",
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        # Tracking across training
        self.eval_count = 0
        self.eval_history: List[Dict] = []
        
        # Rolling window for training metrics
        self.train_episode_rewards = deque(maxlen=100)
        self.train_episode_lengths = deque(maxlen=100)
        self.train_episode_wins = deque(maxlen=100)
        self.train_episode_count = 0
        
        # Current episode tracking (per env)
        self._episode_rewards = None
        self._episode_lengths = None
        
    def _on_training_start(self):
        """Called at start of training."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize per-env tracking
        n_envs = self.training_env.num_envs
        self._episode_rewards = np.zeros(n_envs)
        self._episode_lengths = np.zeros(n_envs)
        
        if self.verbose >= 1:
            print(f"[EvalMetrics] Started. Evaluation every {self.eval_freq} steps.")
            print(f"[EvalMetrics] {self.n_eval_episodes} episodes per evaluation")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Track episode metrics from training
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        # Accumulate rewards and lengths
        if len(rewards) > 0:
            self._episode_rewards += np.array(rewards)
            self._episode_lengths += 1
        
        # Check for episode completions
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                # Record episode stats
                self.train_episode_rewards.append(self._episode_rewards[i])
                self.train_episode_lengths.append(self._episode_lengths[i])
                self.train_episode_count += 1
                
                # Check win condition
                if isinstance(info, dict):
                    agent_hp = info.get('agent_hp', 0)
                    enemy_hp = info.get('enemy_hp', 0)
                    win = 1 if agent_hp > enemy_hp else 0
                    self.train_episode_wins.append(win)
                
                # Reset tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0
        
        # Log training metrics periodically
        if self.num_timesteps > 0 and self.num_timesteps % 5000 == 0:
            self._log_training_metrics()
        
        # Run evaluation
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            self._run_evaluation()
        
        return True
    
    def _log_training_metrics(self):
        """Log rolling training metrics to TensorBoard."""
        if not self.train_episode_rewards:
            return
        
        metrics = {
            'train/episode_reward_mean': np.mean(self.train_episode_rewards),
            'train/episode_reward_std': np.std(self.train_episode_rewards),
            'train/episode_length_mean': np.mean(self.train_episode_lengths),
            'train/win_rate': np.mean(self.train_episode_wins) if self.train_episode_wins else 0,
            'train/episode_count': self.train_episode_count,
        }
        
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.record(key, value)
    
    def _run_evaluation(self):
        """Run evaluation episodes and log results."""
        self.eval_count += 1
        
        if self.verbose >= 1:
            print(f"\n[EvalMetrics] Running evaluation #{self.eval_count} at step {self.num_timesteps}...")
        
        # Collect evaluation metrics
        episode_rewards = []
        episode_lengths = []
        episode_wins = []
        episode_hp_diffs = []
        
        # Use first env for evaluation (simpler than creating new env)
        # Note: In production, you'd want a separate eval env
        obs = self.training_env.reset()
        
        current_rewards = np.zeros(self.training_env.num_envs)
        current_lengths = np.zeros(self.training_env.num_envs)
        completed = 0
        
        # Run until we have enough episodes
        max_steps = 10000  # Safety limit
        steps = 0
        
        while completed < self.n_eval_episodes and steps < max_steps:
            # Get action
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # Step environment
            obs, rewards, dones, infos = self.training_env.step(action)
            
            current_rewards += rewards
            current_lengths += 1
            steps += 1
            
            # Check for episode completions
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and completed < self.n_eval_episodes:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    
                    if isinstance(info, dict):
                        agent_hp = info.get('agent_hp', 0)
                        enemy_hp = info.get('enemy_hp', 0)
                        win = 1 if agent_hp > enemy_hp else 0
                        episode_wins.append(win)
                        episode_hp_diffs.append(agent_hp - enemy_hp)
                    
                    completed += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
        
        # Compute statistics
        eval_results = {
            'timestep': int(self.num_timesteps),
            'eval_count': self.eval_count,
            'n_episodes': len(episode_rewards),
            'reward_mean': float(np.mean(episode_rewards)) if episode_rewards else 0,
            'reward_std': float(np.std(episode_rewards)) if episode_rewards else 0,
            'reward_min': float(np.min(episode_rewards)) if episode_rewards else 0,
            'reward_max': float(np.max(episode_rewards)) if episode_rewards else 0,
            'length_mean': float(np.mean(episode_lengths)) if episode_lengths else 0,
            'length_std': float(np.std(episode_lengths)) if episode_lengths else 0,
            'win_rate': float(np.mean(episode_wins)) if episode_wins else 0,
            'hp_diff_mean': float(np.mean(episode_hp_diffs)) if episode_hp_diffs else 0,
            'hp_diff_std': float(np.std(episode_hp_diffs)) if episode_hp_diffs else 0,
        }
        
        # Log to TensorBoard
        if self.logger is not None:
            self.logger.record("eval/reward_mean", eval_results['reward_mean'])
            self.logger.record("eval/reward_std", eval_results['reward_std'])
            self.logger.record("eval/length_mean", eval_results['length_mean'])
            self.logger.record("eval/win_rate", eval_results['win_rate'])
            self.logger.record("eval/hp_diff_mean", eval_results['hp_diff_mean'])
        
        # Store in history
        self.eval_history.append(eval_results)
        
        # Save to file
        results_file = os.path.join(self.log_dir, f"eval_{self.eval_count}.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        if self.verbose >= 1:
            print(f"[EvalMetrics] Evaluation Results (step {self.num_timesteps}):")
            print(f"  Win Rate: {eval_results['win_rate']:.2%}")
            print(f"  Reward: {eval_results['reward_mean']:.1f} ± {eval_results['reward_std']:.1f}")
            print(f"  Episode Length: {eval_results['length_mean']:.0f} ± {eval_results['length_std']:.0f}")
            print(f"  HP Differential: {eval_results['hp_diff_mean']:.1f} ± {eval_results['hp_diff_std']:.1f}")
    
    def _on_training_end(self):
        """Called at end of training."""
        # Save complete history
        history_file = os.path.join(self.log_dir, "eval_history.json")
        with open(history_file, 'w') as f:
            json.dump({
                'total_timesteps': int(self.num_timesteps),
                'total_evaluations': self.eval_count,
                'total_train_episodes': self.train_episode_count,
                'final_train_win_rate': float(np.mean(self.train_episode_wins)) if self.train_episode_wins else 0,
                'history': self.eval_history,
            }, f, indent=2)
        
        if self.verbose >= 1:
            print(f"[EvalMetrics] Training ended. Ran {self.eval_count} evaluations.")
            print(f"[EvalMetrics] History saved to {history_file}")
    
    def get_eval_history(self) -> List[Dict]:
        """Get the full evaluation history."""
        return self.eval_history
