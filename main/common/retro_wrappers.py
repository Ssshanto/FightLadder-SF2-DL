""" Adapter for Retro: https://github.com/Farama-Foundation/stable-retro. Game dynamics implementation is inspired from https://github.com/linyiLYi/street-fighter-ai. """
import os
import math
import time
import gzip
import cv2
import gymnasium as gym
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Union
from gymnasium.wrappers import FrameStackObservation
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from .const import *
from common.utils import linear_schedule


class GridRenderer:
    """Renders multiple environment frames in a grid layout using opencv"""
    
    def __init__(self, num_envs, frame_shape=(200, 256, 3)):
        self.num_envs = num_envs
        self.frame_shape = frame_shape
        
        # Calculate grid dimensions (as square as possible)
        self.grid_cols = int(np.ceil(np.sqrt(num_envs)))
        self.grid_rows = int(np.ceil(num_envs / self.grid_cols))
        
        self.window_name = 'FightLadder Training Grid'
        
    def render_grid(self, frames):
        """
        Arrange frames in a grid and display with opencv
        
        Args:
            frames: list of RGB frames from environments
        """
        if not frames or frames[0] is None:
            return
        
        # Pad frames list to fill grid
        frames = list(frames)
        while len(frames) < self.grid_rows * self.grid_cols:
            frames.append(np.zeros_like(frames[0]))
        
        # Arrange frames in grid
        rows = []
        for i in range(self.grid_rows):
            start_idx = i * self.grid_cols
            end_idx = start_idx + self.grid_cols
            row_frames = frames[start_idx:end_idx]
            rows.append(np.hstack(row_frames))
        
        grid = np.vstack(rows)
        
        # opencv expects BGR
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        
        cv2.imshow(self.window_name, grid_bgr)
        cv2.waitKey(1)
        
    def close(self):
        cv2.destroyWindow(self.window_name)


class SFWrapper(gym.Wrapper):

    def __init__(self, env, side, reset_type="round", init_level=1, rendering=False, num_stack=12, num_step_frames=8, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False):
        super(SFWrapper, self).__init__(env)
        self.env = FrameStackObservation(env, stack_size=num_stack)

        assert side in ['left', 'right', 'both'], "side should be 'left', 'right' or 'both'"
        self.side = side

        self.num_stack = num_stack
        self.num_step_frames = num_step_frames

        self.aggresive_coeff = 3.0
        self.dense_coeff = 1.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        # Observation space: downsampled frames (200->100, 256->128) with sampled frames along last axis
        # num_sampled_frames = len(range(0, num_stack, num_step_frames // 2)) = len(range(0, 12, 4)) = 3
        num_sampled_frames = len(range(0, self.num_stack, self.num_step_frames // 2))
        self.observation_space = Box(low=0, high=255, shape=(100, 128, num_sampled_frames), dtype=np.uint8)
        self.action_dim = 12 + 3 if (enable_combo or null_combo) else 12 # 3 bits for combos
        if transform_action:
            # self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(COMBOS) for _ in range(self.players)])
            self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(SF_COMBOS)]) if (enable_combo or null_combo) else MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS)])
            def action_transformer(action):
                players_action = []
                for player_action in action:
                    if player_action >= len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS):
                        # if self.null_combo:
                        #     print(f"player_action = {player_action}, invalid for null combo", flush=True)
                        button_bits = [0 for _ in range(12)]
                        combo_bits = [int(i) for i in np.binary_repr(player_action - len(DIRECTIONS_BUTTONS) - len(ATTACKS_BUTTONS)).zfill(3)]
                    elif player_action >= len(DIRECTIONS_BUTTONS):
                        direction_buttons = []
                        attack_buttons = ATTACKS_BUTTONS[player_action - len(DIRECTIONS_BUTTONS)]
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]
                    else:
                        direction_buttons = DIRECTIONS_BUTTONS[player_action]
                        attack_buttons = []
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]
                    players_action.append(np.array(button_bits + combo_bits))
                return np.hstack(players_action)
            self.action_transformer = action_transformer
        else:
            # self.action_space = MultiBinary(self.players * self.action_dim)
            self.action_space = MultiBinary(self.action_dim)
            self.action_transformer = None
        
        self.reset_type = reset_type
        self.rendering = rendering

        self.init_level = init_level
        self.state_dir = state_dir
        self.verbose = verbose
        self.enable_combo = enable_combo
        self.null_combo = null_combo

    def save_state_to_file(self, name="test.state"):
        content = self.env.em.get_state()
        print(f"Save state to {os.path.join(self.state_dir, name)}")
        with gzip.open(os.path.join(self.state_dir, name), 'wb') as f:
            f.write(content)

    def _get_obs(self, obs):
        # gymnasium's FrameStackObservation returns shape (stack_size, H, W, C) = (12, 200, 256, 3)
        # We sample every (num_step_frames // 2) = 4 frames, downsample spatially by 2x,
        # and extract one channel per frame (cycling through R, G, B)
        # Result shape: (100, 128, num_sampled_frames) where num_sampled_frames = 12 // 4 = 3
        if isinstance(obs, dict):
            # Handle dict observation space (if used with action history)
            frames = obs['observations']  # shape: (stack_size, H, W, C)
            sampled_indices = range(0, self.num_stack, self.num_step_frames // 2)
            processed = np.stack([frames[i, ::2, ::2, i % 3] for i in sampled_indices], axis=-1)
            return {
                'observations': processed,
                'actions': np.squeeze(obs['actions'][list(sampled_indices)], axis=-1),
            }
        else:
            # Standard observation: obs shape is (stack_size, H, W, C) = (12, 200, 256, 3)
            sampled_indices = range(0, self.num_stack, self.num_step_frames // 2)
            # For each sampled frame, downsample spatially and take one color channel
            return np.stack([obs[i, ::2, ::2, i % 3] for i in sampled_indices], axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp
        
        self.level = self.init_level # NOTE: only valid for some states
        self.save_state = False
        self.match_status = START_STATUS # NOTE: only valid for some states
        self.round_status = END_STATUS # NOTE: only valid for some states
        self.during_transation = True
        self.round_num = 0
        self.extra_round = False

        self.total_timesteps = 0
    
        return self._get_obs(obs)

    def update_status(self, info, bonus=False):
        # info['level'] = self.level
        # info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        # info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        # print(info, flush=True)
        max_round = 1 if bonus else 3

        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        if self.match_status == END_STATUS and (agent_victories == 0 and enemy_victories == 0):
            self.match_status = START_STATUS
            self.save_state = True
            self.level += 1
            self.round_num = 0
            self.extra_round = False
        elif self.match_status == START_STATUS and (self.round_num == max_round or agent_victories == 2 or enemy_victories == 2):
            self.match_status = END_STATUS
            if self.verbose:
                if agent_victories < enemy_victories:
                    print(f"Level {self.level} is over and player loses")
                elif agent_victories > enemy_victories:
                    print(f"Level {self.level} is over and player wins")
                else:
                    print(f"Draw level {self.level}")
                    if (not bonus) and (not self.extra_round):
                        self.match_status = START_STATUS
                        self.round_num -= 1
                        self.extra_round = True
                        print(f"One more round for draw level {self.level}")
        if self.round_status == END_STATUS and (agent_hp == self.full_hp and enemy_hp == self.full_hp and round_countdown > 0):
            self.round_status = START_STATUS
            self.prev_agent_hp = self.full_hp
            self.prev_enemy_hp = self.full_hp
        elif self.round_status == START_STATUS and ((agent_hp < 0 or enemy_hp < 0) or timesup):
            self.round_status = END_STATUS
            self.round_num += 1
            if self.verbose:
                if agent_hp < enemy_hp:
                    print(f"The round is over and player loses")
                elif agent_hp > enemy_hp:
                    print(f"The round is over and player wins")
                else:
                    print(f"Draw round")
        if self.match_status == END_STATUS:
            info['match'] = 'start' if self.match_status == START_STATUS else 'end'
            info['round'] = 'start' if self.round_status == START_STATUS else 'end'
            assert self.round_status == END_STATUS, info
        if self.round_status == START_STATUS:
            info['match'] = 'start' if self.match_status == START_STATUS else 'end'
            info['round'] = 'start' if self.round_status == START_STATUS else 'end'
            assert self.match_status == START_STATUS, info

    def step(self, action):
        if self.action_transformer is not None:
            action = self.action_transformer(action)
        if self.side == 'both':
            assert action.shape[-1] == 2 * self.action_dim, f"action.shape[-1]={action.shape[-1]}, 2 * self.action_dim={2 * self.action_dim}"
        else:
            assert action.shape[-1] == self.action_dim, f"action.shape[-1]={action.shape[-1]}, self.action_dim={self.action_dim}"

        if self.level in SF_BONUS_LEVEL: # skip bonus level
            skip_level = self.level
            no_op = np.zeros_like(action[:24])
            while self.level == skip_level:
                obs, _reward, _terminated, _truncated, info = self.env.step(no_op)
                self.update_status(info, bonus=True)
            if self.verbose:
                print(f"Skip bonus level {skip_level}")
        
        custom_done = False

        if self.side == 'left':
            action[3] = 0 # Filter out the "START/PAUSE" button
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(SF_COMBOS)
            if combo_id >= len(SF_COMBOS):
                action_seq = [np.hstack([action[:12], np.zeros_like(action[:12])]) for _ in range(self.num_step_frames)]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([combo[t], np.zeros_like(combo[t])]) for t in range(self.num_step_frames)]
        elif self.side == 'right':
            action[3] = 0 # Filter out the "START/PAUSE" button
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(SF_COMBOS)
            if combo_id >= len(SF_COMBOS):
                action_seq = [np.hstack([np.zeros_like(action[:12]), action[:12]]) for _ in range(self.num_step_frames)]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([np.zeros_like(combo[t]), combo[t]]) for t in range(self.num_step_frames)]
        else:
            action[3] = 0 # Filter out the "START/PAUSE" button
            action[self.action_dim + 3] = 0
            if self.enable_combo:
                combo_ids = [int(4 * action[self.action_dim - 3] + 2 * action[self.action_dim - 2] + action[self.action_dim - 1]), int(4 * action[-3] + 2 * action[-2] + action[-1])]
            else:
                combo_ids = [len(SF_COMBOS), len(SF_COMBOS)]
            action_seqs = []
            for player_id, combo_id in enumerate(combo_ids):
                if combo_id >= len(SF_COMBOS):
                    action_seq = [action[player_id * self.action_dim : player_id * self.action_dim + 12] for _ in range(self.num_step_frames)]
                else:
                    combo = SF_COMBOS[combo_id]
                    assert self.num_step_frames == len(combo)
                    action_seq = combo
                action_seqs.append(action_seq)
            action_seq = [np.hstack([action_1, action_2]) for action_1, action_2 in zip(action_seqs[0], action_seqs[1])]
        
        for i in range(self.num_step_frames):            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _terminated, _truncated, info = self.env.step(action_seq[i])
            self.update_status(info)
            # Note: Rendering is now handled by GridRenderer via RenderCallback in training loop
            # Individual env rendering in subprocesses would create multiple windows and conflict with grid display

        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        self.total_timesteps += self.num_step_frames
        
        if self.during_transation and (self.match_status == END_STATUS or self.round_status == END_STATUS):
            # During transation between episodes, do nothing
            custom_done = False
            custom_reward = 0
            custom_reward_inverse = 0
            if (enemy_victories == 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                # Player loses the game
                custom_done = not self.reset_type == "never"
            if (agent_victories == 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                # Player wins the match
                custom_done = self.reset_type == "match"
                if self.level == 15:
                    print(f"Player wins the game")
                    custom_done = True
                    self.save_state = True
                    self.level += 1
        else:
            self.during_transation = False
            # if self.save_state and self.state_dir is not None:
            #     self.save_state = False
            #     self.save_state_to_file(f"Level{self.level}.{self.total_timesteps}.state")

            if (agent_hp < 0 and enemy_hp < 0) or (timesup and agent_hp == enemy_hp):
                custom_reward = 1
                custom_reward_inverse = 1
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transation = True
            elif agent_hp < 0 or (timesup and agent_hp < enemy_hp):
                custom_reward = -math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1))     
                custom_reward_inverse = math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (enemy_victories >= 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                        # Player loses the game
                        # if self.verbose:
                        #     print("Player loses the game")
                        custom_done = not self.reset_type == "never"
            elif enemy_hp < 0 or (timesup and agent_hp > enemy_hp):
                custom_reward = math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                custom_reward_inverse = -math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))

                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (agent_victories >= 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                        # Player wins the match
                        # if self.verbose:
                        #     print("Player wins the match")
                        custom_done = self.reset_type == "match"
                        if self.level == 15:
                            print(f"Player wins the game")
                            custom_done = True
                            self.save_state = True
                            self.level += 1
            # While the fighting is still going on
            else:
                custom_reward = self.dense_coeff * (self.aggresive_coeff * (self.prev_enemy_hp - enemy_hp) - (self.prev_agent_hp - agent_hp))
                custom_reward_inverse = self.dense_coeff * (self.aggresive_coeff * (self.prev_agent_hp - agent_hp) - (self.prev_enemy_hp - enemy_hp))
                self.prev_agent_hp = agent_hp
                self.prev_enemy_hp = enemy_hp
                custom_done = False

        # if custom_reward != 0:
        #     print("reward:{}".format(custom_reward))

        info['level'] = self.level
        info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        if custom_done:
            info['outcome'] = 'win' if (agent_hp > enemy_hp) else ('lose' if (agent_hp < enemy_hp) else 'draw')

        # truncated is always False - we use terminated (custom_done) for episode boundaries
        truncated = False
        
        if self.side == 'left':
            return self._get_obs(obs), 0.001 * custom_reward, custom_done, truncated, info 
        elif self.side == 'right':
            return self._get_obs(obs), 0.001 * custom_reward_inverse, custom_done, truncated, info 
        else:
            # 2-player mode: return both rewards for self-play training
            return self._get_obs(obs), 0.001 * custom_reward, 0.001 * custom_reward_inverse, custom_done, truncated, info 


class Monitor2P(Monitor):
    
    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)
        
        self.rewards_other = None
        self.episode_returns_other = []
    
    def reset(self, **kwargs) -> GymObs:
        self.rewards_other = []
        return super().reset(**kwargs)
    
    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        # 2P mode returns: (obs, reward, reward_other, done, truncated, info)
        observation, reward, reward_other, done, truncated, info = self.env.step(action)
        # Combine terminated and truncated for done signal
        done = done or truncated
        self.rewards.append(reward)
        self.rewards_other.append(reward_other)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_rew_other = sum(self.rewards_other)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "ro": round(ep_rew_other, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_returns_other.append(ep_rew_other)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        # Return 6 values for 2P mode compatibility with _worker2p
        return observation, reward, reward_other, done, truncated, info