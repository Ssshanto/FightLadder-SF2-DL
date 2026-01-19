import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy
from tqdm import tqdm

import stable_retro as retro
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from common.const import *
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from common.game import get_next_level
from common.algorithms import IPPO
from common.retro_wrappers import SFWrapper, Monitor2P


STATE = "Champion.RyuVsRyu.2Player.align"


def constructor(args, side, log_name=None, single_env=False):
    pass


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, seed=0, render_mode=None):
    def _init():
        players = 2
        # When rendering is enabled, set render_mode to 'rgb_array'
        actual_render_mode = render_mode
        if actual_render_mode is None and rendering:
            actual_render_mode = 'rgb_array'
            
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players,
            render_mode=actual_render_mode,
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action)
        env = Monitor2P(env)
        # Note: gymnasium uses reset(seed=seed) instead of env.seed()
        return env
    return _init


@torch.no_grad()
def evaluate(args, model, greedy=0.99, record=True, render_fps=60):
    win_cnt = 0
    
    # Create grid renderer for live display during evaluation
    grid_renderer = None
    if args.render:
        from common.retro_wrappers import GridRenderer
        import time
        grid_renderer = GridRenderer(num_envs=1)
        frame_delay = 1.0 / render_fps if render_fps > 0 else 0
    
    for i in range(1, args.num_episodes + 1):
        # Use render_mode="rgb_array" for recording or live display
        render_mode = "rgb_array" if (record or args.render) else None
        env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=0, render_mode=render_mode)().env

        done = False
        
        obs, info = env.reset()
        if record:
            # Access the unwrapped retro env for rendering
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            video_log = [Image.fromarray(base_env.render())]

        while not done:
            if np.random.uniform() > greedy:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=False)
            else:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=True)

            obs, reward, reward_other, done, truncated, info = env.step(np.hstack([action, action_other]))
            
            # Get frame for display/recording
            if record or args.render:
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                frame = base_env.render()
                
                if record:
                    video_log.append(Image.fromarray(frame))
                
                if args.render and grid_renderer:
                    grid_renderer.render_grid([frame])
                    if frame_delay > 0:
                        time.sleep(frame_delay)
            
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

            if done:
                if record:
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/episode_{i}.mp4", mode='w')
                    stream = container.add_stream('h264', rate=10)
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = 'yuv420p'
                    for img in video_log:
                        frame = av.VideoFrame.from_image(img)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    remain_packets = stream.encode(None)
                    container.mux(remain_packets)
                    container.close()

        if info['enemy_hp'] < info['agent_hp']:
            print("Victory!")
            win_cnt += 1

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward
    
        env.close()
    
    # Cleanup grid renderer
    if grid_renderer:
        grid_renderer.close()
    
    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate


def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e7))
    parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument('--render-fps', type=int, help='FPS for rendering during evaluation', default=60)
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, skip training')
    parser.add_argument('--enable-interpretability', action='store_true', help='Enable interpretability logging during training')
    parser.add_argument('--interp-log-dir', type=str, default='interpretability_logs', help='Directory for interpretability logs')
    parser.add_argument('--interp-log-freq', type=int, default=1000, help='Interpretability logging frequency (steps)')
    parser.add_argument('--interp-probe-freq', type=int, default=50000, help='Concept probe training frequency (steps)')
    
    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.finetune_dir, exist_ok=True)
    
    # If eval_only mode, load model and run evaluation without creating training environments
    if args.eval_only:
        if not args.model_file:
            raise ValueError("--eval-only requires --model-file to specify which model to evaluate")
        print(f"\n[INFO] Evaluation-only mode, loading model from {args.model_file}...")
        
        # Create a single dummy environment just to initialize model structure
        dummy_env = VecTransposeImage2P(SubprocVecEnv2P([
            make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=False, 
                    enable_combo=args.enable_combo, null_combo=args.null_combo, 
                    transform_action=args.transform_action, seed=0, render_mode=None)
        ]))
        
        model = IPPO.load(args.model_file, env=dummy_env, device="cuda")
        dummy_env.close()
        
        results = evaluate(args, model, record=True, render_fps=args.render_fps)
        print(results)
        return
                                 
    # Set up the environment and model
    def env_generator():
        # Pass render_mode when --render flag is set
        render_mode = 'rgb_array' if args.render else None
        env = [make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=i, render_mode=render_mode) for i in range(args.num_env)]
        return VecTransposeImage2P(SubprocVecEnv2P(env))
        # return SubprocVecEnv2P(env)

    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint

    def finetune_model_generator(model_file=None, lr_schedule=linear_schedule(5.0e-5, 2.5e-6), other_lr_schedule=linear_schedule(5.0e-5, 2.5e-6), clip_range_schedule=linear_schedule(0.075, 0.025)):
        finetune_env = env_generator()
        finetune_model = IPPO(
            "CnnPolicy", 
            finetune_env,
            device="cuda", 
            verbose=1,
            n_steps=512,
            batch_size=1024, # 512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
            other_learning_rate=other_lr_schedule,
        )
        if model_file:
            print("load model from " + model_file)
            if model_file.endswith(".pt"):
                model_file = torch.load(model_file, map_location=torch.device('cpu'))["kwargs"]["agent_dict"]
            finetune_model.set_parameters(model_file)
        return finetune_model

    finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_final_steps")
    lr_schedule = 1e-4 # if args.async_update else linear_schedule(2.5e-4, 2.5e-6)
    other_lr_schedule = 1e-4 # if args.async_update else linear_schedule(2.5e-4/args.other_timescale, 2.5e-6/args.other_timescale)
    clip_range_schedule = 0.1 # if args.async_update else linear_schedule(0.15, 0.025)
    model = finetune_model_generator(args.model_file, lr_schedule=lr_schedule, other_lr_schedule=other_lr_schedule, clip_range_schedule=clip_range_schedule)

    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        model.set_parameters_2p(args.left_model_file, args.right_model_file)
    
    model.save(os.path.join(args.save_dir, args.model_name_prefix + f"_0_steps"))

    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=f"{args.model_name_prefix}")
    
    # Create progress bar callback
    from stable_baselines3.common.callbacks import BaseCallback
    
    class TqdmCallback(BaseCallback):
        def __init__(self, total_timesteps):
            super().__init__()
            self.pbar = None
            self.total_timesteps = total_timesteps
            
        def _on_training_start(self):
            self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")
            
        def _on_step(self):
            if self.pbar:
                self.pbar.update(self.num_timesteps - self.pbar.n)
            return True
            
        def _on_training_end(self):
            if self.pbar:
                self.pbar.close()
    
    # Create grid renderer and callback if rendering is enabled
    callbacks = [checkpoint_callback, TqdmCallback(args.total_steps)]
    
    # Add interpretability callback if enabled
    if args.enable_interpretability:
        try:
            from interpretability.callbacks import RuntimeInterpretabilityCallback
            os.makedirs(args.interp_log_dir, exist_ok=True)
            interp_callback = RuntimeInterpretabilityCallback(
                log_dir=args.interp_log_dir,
                log_frequency=args.interp_log_freq,
                analysis_frequency=args.interp_probe_freq,
                verbose=1,
            )
            callbacks.append(interp_callback)
            print(f"[INFO] Interpretability logging enabled, saving to {args.interp_log_dir}")
        except ImportError as e:
            print(f"[WARNING] Interpretability not available: {e}")
    
    if args.render:
        from common.retro_wrappers import GridRenderer
        from stable_baselines3.common.callbacks import BaseCallback
        
        grid_renderer = GridRenderer(num_envs=args.num_env)
        
        class RenderCallback(BaseCallback):
            def __init__(self, grid_renderer, render_freq=8):
                super().__init__()
                self.grid_renderer = grid_renderer
                self.render_freq = render_freq
                
            def _on_step(self) -> bool:
                if self.n_calls % self.render_freq == 0:
                    frames = self.training_env.get_images()
                    self.grid_renderer.render_grid(frames)
                return True
                
            def _on_training_end(self) -> None:
                self.grid_renderer.close()
        
        render_callback = RenderCallback(grid_renderer, render_freq=8)
        callbacks.append(render_callback)
    
    if args.async_update:
        model.async_learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback],
            fsp=args.fsp,
            fsp_threshold=args.fsp_threshold,
        )
    else:
        model.learn( 
            total_timesteps=args.total_steps*args.other_timescale,
            callback=callbacks
        )
    model.save(finetune_epoch_model_path)
    results = evaluate(args, model, record=True, render_fps=args.render_fps)
    print(results)
    with open(f"{args.finetune_dir}/{args.model_name_prefix}_start_results.txt", 'w') as f:
        f.write(str(results))


if __name__ == "__main__":
    main()
