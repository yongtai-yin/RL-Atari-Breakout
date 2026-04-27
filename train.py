#!/usr/bin/env python3
"""
Unified training script for PPO and DQN on Atari Breakout.

Usage:
    python train.py --algo ppo --seed 0 --steps 20000000
    python train.py --algo dqn --seed 100 --steps 20000000
"""

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import argparse
import math
from tqdm import tqdm

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from utils import create_train_env, ensure_dirs, get_device


class ProgressCallback(BaseCallback):
    """Custom progress bar callback for training visualization."""

    def __init__(self, total_timesteps, log_freq=10000):
        super().__init__()
        self.total = total_timesteps
        self.log_freq = log_freq
        self.pbar = None
        self._start_timesteps = 0
        self._last_update_timesteps = 0

    def _on_training_start(self):
        self._start_timesteps = self.num_timesteps
        self._last_update_timesteps = self.num_timesteps
        self.pbar = tqdm(total=self.total, desc="Training Progress")

    def _on_step(self):
        current_progress = self.num_timesteps - self._start_timesteps
        delta = self.num_timesteps - self._last_update_timesteps

        if self.pbar and (delta >= self.log_freq or current_progress >= self.total):
            remaining = self.total - self.pbar.n
            self.pbar.update(min(delta, remaining))
            self._last_update_timesteps = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.pbar:
            delta = self.num_timesteps - self._last_update_timesteps
            remaining = self.total - self.pbar.n
            if delta > 0 and remaining > 0:
                self.pbar.update(min(delta, remaining))
            self.pbar.close()
            self.pbar = None


def create_ppo_model(env, seed=0, device="auto"):
    """
    Create PPO model with reference hyperparameters.

    Parameters:
    -----------
    env : VecEnv
        Training environment
    seed : int
        Random seed
    device : str
        Device to use for training

    Returns:
    --------
    PPO : PPO model instance
    """
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/ppo/",
        device=device,
        seed=seed
    )
    return model


def create_dqn_model(env, seed=0, device="auto"):
    """
    Create DQN model with SB3 default + Atari optimization.

    Parameters:
    -----------
    env : VecEnv
        Training environment
    seed : int
        Random seed
    device : str
        Device to use for training

    Returns:
    --------
    DQN : DQN model instance
    """
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/dqn/",
        device=device,
        seed=seed
    )
    return model


def train(algo, seed=0, total_timesteps=20_000_000, n_envs=8, resume_path=None):
    """
    Train PPO or DQN model.

    Parameters:
    -----------
    algo : str
        Algorithm name ('ppo' or 'dqn')
    seed : int
        Random seed
    total_timesteps : int
        Total training steps
    n_envs : int
        Number of parallel environments
    resume_path : str or None
        Path to resume from checkpoint
    """
    algo = algo.lower()
    if algo not in {"ppo", "dqn"}:
        raise ValueError("algo must be 'ppo' or 'dqn'")
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    if resume_path and not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    ensure_dirs()

    # Explicitly show device info at training start
    device = get_device()

    print(f"Setting up training environment (seed={seed}, n_envs={n_envs})...")
    env = create_train_env(n_envs=n_envs, seed=seed)
    resume = resume_path is not None

    try:
        # Create or load model
        if resume:
            print(f"Resuming from checkpoint: {resume_path}")
            model_cls = PPO if algo == "ppo" else DQN
            model = model_cls.load(resume_path, env=env, device=device)
        else:
            print(f"Creating new {algo.upper()} model...")
            if algo == "ppo":
                model = create_ppo_model(env, seed=seed, device=device)
            else:
                model = create_dqn_model(env, seed=seed, device=device)

        # CheckpointCallback counts vectorized env.step() calls, so divide by
        # n_envs to checkpoint every 5M actual environment timesteps.
        checkpoint_freq = max(math.ceil(5_000_000 / n_envs), 1)
        progress_callback = ProgressCallback(total_timesteps=total_timesteps)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=f"./models/{algo}/",
            name_prefix=f"{algo}_breakout_seed{seed}"
        )

        print(f"\nStarting {algo.upper()} training for {total_timesteps} steps...")
        print(f"Training device confirmed: {device}")

        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback],
            reset_num_timesteps=not resume,
            tb_log_name=f"{algo.upper()}_seed{seed}",
        )

        final_path = f"models/{algo}/{algo}_breakout_seed{seed}_final"
        model.save(final_path)
        print(f"Saved final model to {final_path}")

        return model
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO or DQN on Breakout")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "dqn"],
        required=True,
        help="Algorithm to train (ppo or dqn)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (0, 100, 200 recommended)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000_000,
        help="Total training steps"
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    print(f"Arguments: algo={args.algo}, seed={args.seed}, steps={args.steps}, envs={args.envs}")

    train(
        algo=args.algo,
        seed=args.seed,
        total_timesteps=args.steps,
        n_envs=args.envs,
        resume_path=args.resume
    )
