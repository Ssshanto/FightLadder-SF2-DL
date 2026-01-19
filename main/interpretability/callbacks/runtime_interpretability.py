"""
Runtime Interpretability Callback

Performs rigorous interpretability analysis during training:
- Collects activations and ground truth features from rollouts
- Computes mutual information I(action; feature) to measure policy dependencies
- Tracks activation statistics (mean, std, sparsity)
- Logs results to TensorBoard for tracking learning dynamics

This module uses ONLY raw RAM values as ground truth, avoiding hand-crafted
concepts that would introduce circular reasoning in probing analysis.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback

# Import rigorous interpretability modules
from ..ground_truth import GroundTruthExtractor, GROUND_TRUTH_FEATURES
from ..mi_analysis import MutualInformationAnalyzer


@dataclass
class ActivationBuffer:
    """Buffer for collecting activations and features during training."""
    
    max_size: int = 10000
    activations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    features: Dict[str, List[float]] = field(default_factory=lambda: {k: [] for k in GROUND_TRUTH_FEATURES})
    
    def add(self, activation: np.ndarray, action: int, feature_dict: Dict[str, float], reward: float = 0.0):
        """Add a single timestep's data to the buffer."""
        if len(self.activations) >= self.max_size:
            # Remove oldest entries
            self.activations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            for k in self.features:
                self.features[k].pop(0)
        
        self.activations.append(activation)
        self.actions.append(action)
        self.rewards.append(reward)
        for k, v in feature_dict.items():
            if k in self.features:
                self.features[k].append(v)
    
    def __len__(self):
        return len(self.activations)
    
    def clear(self):
        """Clear all data from buffer."""
        self.activations.clear()
        self.actions.clear()
        self.rewards.clear()
        for k in self.features:
            self.features[k].clear()
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert buffer to numpy arrays for analysis."""
        return {
            'activations': np.array(self.activations) if self.activations else np.array([]),
            'actions': np.array(self.actions) if self.actions else np.array([]),
            'rewards': np.array(self.rewards) if self.rewards else np.array([]),
            'features': {k: np.array(v) for k, v in self.features.items()}
        }


class RuntimeInterpretabilityCallback(BaseCallback):
    """
    Callback for runtime interpretability analysis during training.
    
    Collects policy activations and game state features during training,
    then periodically computes mutual information to measure what the
    policy has learned to encode.
    
    Args:
        log_dir: Directory to save analysis results
        analysis_frequency: How often to run MI analysis (in timesteps)
        log_frequency: How often to log summary stats (in timesteps)
        buffer_size: Maximum number of samples to keep in buffer
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
    """
    
    def __init__(
        self,
        log_dir: str = "interpretability_logs",
        analysis_frequency: int = 50000,
        log_frequency: int = 1000,
        buffer_size: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.analysis_frequency = analysis_frequency
        self.log_frequency = log_frequency
        
        # Data collection
        self.buffer = ActivationBuffer(max_size=buffer_size)
        self.ground_truth_extractor = GroundTruthExtractor()
        self.mi_analyzer = MutualInformationAnalyzer()
        
        # Tracking
        self.analysis_count = 0
        self.last_mi_results: Optional[Dict[str, float]] = None
        
        # MI evolution tracking (for plotting learning dynamics)
        self.mi_history: List[Dict] = []
        self.activation_stats_history: List[Dict] = []
        
        # Episode tracking for win rate
        self.episode_wins = 0
        self.episode_count = 0
        self.recent_wins = deque(maxlen=100)  # Rolling window
        
        # Forward hook handle
        self._hook_handle = None
        self._last_activation = None
        
    def _on_training_start(self):
        """Called at the start of training."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Register forward hook to capture activations
        self._register_activation_hook()
        
        if self.verbose >= 1:
            print(f"[RuntimeInterp] Started. Analysis every {self.analysis_frequency} steps.")
            print(f"[RuntimeInterp] Logging to {self.log_dir}")
    
    def _register_activation_hook(self):
        """Register a forward hook on the policy network to capture activations."""
        try:
            # Access the policy network
            policy = self.model.policy
            
            # Find the last layer before action head (typically features_extractor or mlp_extractor)
            # For CNN policies, we want the output of the CNN + flatten
            if hasattr(policy, 'features_extractor'):
                target_module = policy.features_extractor
                
                def hook_fn(module, input, output):
                    # Flatten and store the activation
                    if isinstance(output, torch.Tensor):
                        self._last_activation = output.detach().cpu().numpy()
                
                self._hook_handle = target_module.register_forward_hook(hook_fn)
                
                if self.verbose >= 1:
                    print(f"[RuntimeInterp] Registered hook on features_extractor")
            else:
                if self.verbose >= 1:
                    print(f"[RuntimeInterp] Warning: Could not find features_extractor")
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"[RuntimeInterp] Warning: Could not register hook: {e}")
    
    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Extract ground truth features from info dict
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', [])
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        
        # Track episode outcomes for win rate
        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and isinstance(info, dict):
                self.episode_count += 1
                # Check if agent won (agent_hp > enemy_hp at end)
                agent_hp = info.get('agent_hp', 0)
                enemy_hp = info.get('enemy_hp', 0)
                win = 1 if agent_hp > enemy_hp else 0
                self.episode_wins += win
                self.recent_wins.append(win)
        
        # Debug: print buffer size periodically
        if self.num_timesteps % 5000 == 0 and self.verbose >= 1:
            win_rate = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else 0
            print(f"[RuntimeInterp] Step {self.num_timesteps}, buffer: {len(self.buffer)}, "
                  f"win_rate (last 100): {win_rate:.2%}")
        
        if infos and self._last_activation is not None:
            for i, info in enumerate(infos):
                # Skip if info doesn't have required fields
                if not isinstance(info, dict) or 'agent_hp' not in info:
                    continue
                    
                # Extract features from the info dict (RAM values)
                features = self.ground_truth_extractor.extract_from_info(info)
                
                # Get activation for this env (handle batch dimension)
                if len(self._last_activation.shape) > 1 and i < len(self._last_activation):
                    activation = self._last_activation[i].flatten()
                else:
                    activation = self._last_activation.flatten()
                
                # Get action
                if i < len(actions):
                    action = actions[i]
                    if isinstance(action, np.ndarray):
                        action = int(action.flatten()[0]) if action.size > 0 else 0
                    else:
                        action = int(action)
                else:
                    action = 0
                
                # Get reward
                reward = float(rewards[i]) if i < len(rewards) else 0.0
                
                # Add to buffer
                self.buffer.add(activation, action, features, reward)
        
        # Periodic analysis based on num_timesteps
        if self.num_timesteps > 0 and self.num_timesteps % self.analysis_frequency == 0:
            if len(self.buffer) >= 100:
                self._run_analysis()
            elif self.verbose >= 1:
                print(f"[RuntimeInterp] Step {self.num_timesteps}: buffer too small ({len(self.buffer)}), skipping analysis")
        
        return True
    
    def _compute_activation_stats(self, activations: np.ndarray) -> Dict[str, float]:
        """Compute statistics about the activation patterns."""
        if len(activations) == 0:
            return {}
        
        stats = {
            'activation_mean': float(np.mean(activations)),
            'activation_std': float(np.std(activations)),
            'activation_max': float(np.max(activations)),
            'activation_min': float(np.min(activations)),
            'activation_sparsity': float(np.mean(activations == 0)),  # Fraction of zeros
            'activation_dim': int(activations.shape[1]) if len(activations.shape) > 1 else 1,
        }
        
        # Per-neuron statistics (variance across samples)
        if len(activations.shape) > 1:
            neuron_vars = np.var(activations, axis=0)
            stats['neuron_var_mean'] = float(np.mean(neuron_vars))
            stats['neuron_var_std'] = float(np.std(neuron_vars))
            stats['dead_neurons'] = int(np.sum(neuron_vars < 1e-6))  # Nearly constant neurons
            stats['active_neurons'] = int(np.sum(neuron_vars >= 1e-6))
        
        return stats
    
    def _compute_feature_stats(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each ground truth feature."""
        stats = {}
        for name, values in features.items():
            if len(values) == 0:
                continue
            stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        return stats
    
    def _run_analysis(self):
        """Run mutual information analysis on collected data."""
        self.analysis_count += 1
        
        if self.verbose >= 1:
            print(f"\n[RuntimeInterp] Running MI analysis #{self.analysis_count} at step {self.num_timesteps}...")
        
        # Get data from buffer
        data = self.buffer.to_arrays()
        activations = data['activations']
        actions = data['actions']
        rewards = data['rewards']
        features = data['features']
        
        if len(activations) < 100:
            if self.verbose >= 1:
                print(f"[RuntimeInterp] Not enough data ({len(activations)} samples), skipping analysis")
            return
        
        # Compute MI for each feature
        mi_results = {}
        for feature_name, feature_values in features.items():
            if len(feature_values) == 0:
                continue
            
            try:
                # MI between action and feature: I(action; feature)
                mi = self.mi_analyzer.compute_mi(
                    actions,
                    np.array(feature_values),
                    discrete_x=True,
                    discrete_y=False
                )
                mi_results[f'mi_action_{feature_name}'] = float(mi)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"[RuntimeInterp] Warning: MI computation failed for {feature_name}: {e}")
        
        self.last_mi_results = mi_results
        
        # Compute activation statistics
        activation_stats = self._compute_activation_stats(activations)
        
        # Compute feature statistics
        feature_stats = self._compute_feature_stats(features)
        
        # Compute action distribution
        unique_actions, action_counts = np.unique(actions, return_counts=True)
        action_entropy = -np.sum((action_counts / len(actions)) * np.log2(action_counts / len(actions) + 1e-10))
        
        # Compute reward statistics
        reward_stats = {
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_min': float(np.min(rewards)),
            'reward_max': float(np.max(rewards)),
        }
        
        # Win rate
        win_rate = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else 0.0
        overall_win_rate = self.episode_wins / self.episode_count if self.episode_count > 0 else 0.0
        
        # Log to TensorBoard
        if self.logger is not None:
            # MI results
            for key, value in mi_results.items():
                self.logger.record(f"interpretability/{key}", value)
            
            # Activation stats
            for key, value in activation_stats.items():
                self.logger.record(f"interpretability/activation/{key}", value)
            
            # Reward stats
            for key, value in reward_stats.items():
                self.logger.record(f"interpretability/{key}", value)
            
            # Action entropy
            self.logger.record("interpretability/action_entropy", action_entropy)
            self.logger.record("interpretability/num_unique_actions", len(unique_actions))
            
            # Win rates
            self.logger.record("interpretability/win_rate_recent", win_rate)
            self.logger.record("interpretability/win_rate_overall", overall_win_rate)
            self.logger.record("interpretability/episode_count", self.episode_count)
        
        # Store in history for plotting
        history_entry = {
            'timestep': int(self.num_timesteps),
            'analysis_count': self.analysis_count,
            'n_samples': len(activations),
            'mi_results': mi_results,
            'activation_stats': activation_stats,
            'feature_stats': feature_stats,
            'reward_stats': reward_stats,
            'action_entropy': float(action_entropy),
            'num_unique_actions': int(len(unique_actions)),
            'win_rate_recent': float(win_rate),
            'win_rate_overall': float(overall_win_rate),
            'episode_count': int(self.episode_count),
        }
        self.mi_history.append(history_entry)
        
        # Save results to file
        results_file = os.path.join(self.log_dir, f"mi_analysis_{self.analysis_count}.json")
        with open(results_file, 'w') as f:
            json.dump(history_entry, f, indent=2)
        
        if self.verbose >= 1:
            # Print summary
            print(f"[RuntimeInterp] MI Analysis Results (step {self.num_timesteps}):")
            sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_mi[:5]:  # Top 5
                feature = key.replace('mi_action_', '')
                print(f"  I(action; {feature}) = {value:.4f}")
            print(f"  Action entropy: {action_entropy:.3f} ({len(unique_actions)} unique actions)")
            print(f"  Win rate (recent/overall): {win_rate:.2%} / {overall_win_rate:.2%}")
            print(f"  Activation dim: {activation_stats.get('activation_dim', 'N/A')}, "
                  f"dead neurons: {activation_stats.get('dead_neurons', 'N/A')}")
    
    def _on_training_end(self):
        """Called at the end of training."""
        # Remove hook
        if self._hook_handle is not None:
            self._hook_handle.remove()
        
        # Final analysis
        if len(self.buffer) >= 100:
            self._run_analysis()
        
        # Save complete MI history
        history_file = os.path.join(self.log_dir, "mi_history.json")
        with open(history_file, 'w') as f:
            json.dump({
                'total_timesteps': int(self.num_timesteps),
                'total_analyses': self.analysis_count,
                'total_episodes': self.episode_count,
                'total_wins': self.episode_wins,
                'final_win_rate': self.episode_wins / self.episode_count if self.episode_count > 0 else 0,
                'history': self.mi_history,
            }, f, indent=2)
        
        if self.verbose >= 1:
            print(f"[RuntimeInterp] Training ended. Ran {self.analysis_count} analyses.")
            print(f"[RuntimeInterp] Final win rate: {self.episode_wins}/{self.episode_count} "
                  f"= {self.episode_wins/self.episode_count:.2%}" if self.episode_count > 0 else "")
            print(f"[RuntimeInterp] MI history saved to {history_file}")
    
    def get_last_results(self) -> Optional[Dict[str, float]]:
        """Get the results from the last MI analysis."""
        return self.last_mi_results
    
    def get_mi_history(self) -> List[Dict]:
        """Get the full MI analysis history."""
        return self.mi_history
