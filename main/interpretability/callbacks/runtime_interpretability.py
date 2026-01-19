"""
Runtime Interpretability Callback

Performs rigorous interpretability analysis during training:
- Collects activations and ground truth features from rollouts
- Computes mutual information I(action; feature) to measure policy dependencies
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
    features: Dict[str, List[float]] = field(default_factory=lambda: {k: [] for k in GROUND_TRUTH_FEATURES})
    
    def add(self, activation: np.ndarray, action: int, feature_dict: Dict[str, float]):
        """Add a single timestep's data to the buffer."""
        if len(self.activations) >= self.max_size:
            # Remove oldest entries
            self.activations.pop(0)
            self.actions.pop(0)
            for k in self.features:
                self.features[k].pop(0)
        
        self.activations.append(activation)
        self.actions.append(action)
        for k, v in feature_dict.items():
            if k in self.features:
                self.features[k].append(v)
    
    def __len__(self):
        return len(self.activations)
    
    def clear(self):
        """Clear all data from buffer."""
        self.activations.clear()
        self.actions.clear()
        for k in self.features:
            self.features[k].clear()
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert buffer to numpy arrays for analysis."""
        return {
            'activations': np.array(self.activations) if self.activations else np.array([]),
            'actions': np.array(self.actions) if self.actions else np.array([]),
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
        
        # Debug: print buffer size periodically
        if self.num_timesteps % 5000 == 0 and self.verbose >= 1:
            print(f"[RuntimeInterp] Step {self.num_timesteps}, buffer size: {len(self.buffer)}")
        
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
                
                # Add to buffer
                self.buffer.add(activation, action, features)
        
        # Periodic analysis based on num_timesteps
        if self.num_timesteps > 0 and self.num_timesteps % self.analysis_frequency == 0:
            if len(self.buffer) >= 100:
                self._run_analysis()
            elif self.verbose >= 1:
                print(f"[RuntimeInterp] Step {self.num_timesteps}: buffer too small ({len(self.buffer)}), skipping analysis")
        
        return True
    
    def _run_analysis(self):
        """Run mutual information analysis on collected data."""
        self.analysis_count += 1
        
        if self.verbose >= 1:
            print(f"\n[RuntimeInterp] Running MI analysis #{self.analysis_count} at step {self.num_timesteps}...")
        
        # Get data from buffer
        data = self.buffer.to_arrays()
        activations = data['activations']
        actions = data['actions']
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
        
        # Log to TensorBoard
        if self.logger is not None:
            for key, value in mi_results.items():
                self.logger.record(f"interpretability/{key}", value)
        
        # Save results to file
        results_file = os.path.join(self.log_dir, f"mi_analysis_{self.analysis_count}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestep': int(self.num_timesteps),
                'analysis_count': self.analysis_count,
                'n_samples': len(activations),
                'mi_results': mi_results
            }, f, indent=2)
        
        if self.verbose >= 1:
            # Print summary
            print(f"[RuntimeInterp] MI Analysis Results (step {self.num_timesteps}):")
            sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_mi[:5]:  # Top 5
                feature = key.replace('mi_action_', '')
                print(f"  I(action; {feature}) = {value:.4f}")
    
    def _on_training_end(self):
        """Called at the end of training."""
        # Remove hook
        if self._hook_handle is not None:
            self._hook_handle.remove()
        
        # Final analysis
        if len(self.buffer) >= 100:
            self._run_analysis()
        
        if self.verbose >= 1:
            print(f"[RuntimeInterp] Training ended. Ran {self.analysis_count} analyses.")
    
    def get_last_results(self) -> Optional[Dict[str, float]]:
        """Get the results from the last MI analysis."""
        return self.last_mi_results
