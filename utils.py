#!/usr/bin/env python3
"""
Environment utility functions for PPO vs DQN comparison project.

Provides unified environment creation and preprocessing for both algorithms.
"""

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

from pathlib import Path

import torch
import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv

# Register ALE environments
gym.register_envs(ale_py)

ENV_ID = "BreakoutNoFrameskip-v4"
ALGORITHMS = ("ppo", "dqn")
DEFAULT_DQN_EVAL_EPSILON = 0.05
DEFAULT_DETERMINISTIC_BY_ALGO = {
    "ppo": False,
    "dqn": False,
}
PROJECT_DIRS = (
    Path("models/ppo"),
    Path("models/dqn"),
    Path("logs/ppo"),
    Path("logs/dqn"),
    Path("videos"),
    Path("results"),
)


def infer_algorithm(model_path, valid_algorithms=ALGORITHMS):
    """Infer algorithm name from a model path such as models/ppo/..."""
    path = Path(model_path)
    path_parts = {part.lower() for part in path.parts}
    filename = path.name.lower()

    matches = [
        algo
        for algo in valid_algorithms
        if algo in path_parts or filename.startswith(f"{algo}_")
    ]
    if len(matches) != 1:
        valid = ", ".join(valid_algorithms)
        raise ValueError(f"Could not infer algorithm from model path. Use one of: {valid}.")
    return matches[0]


def resolve_policy_determinism(algo, policy="auto"):
    """
    Resolve policy prediction mode for evaluation/playback.

    PPO needs sampled actions by default to match the reference implementation.
    DQN uses epsilon-greedy by default to match the original DQN evaluation
    protocol; callers must set the model exploration rate explicitly.
    """
    if algo not in DEFAULT_DETERMINISTIC_BY_ALGO:
        raise ValueError("algo must be 'ppo' or 'dqn'")
    if policy == "auto":
        return DEFAULT_DETERMINISTIC_BY_ALGO[algo]
    if policy == "deterministic":
        return True
    if policy == "sampled":
        return False
    if policy == "epsilon-greedy":
        return False
    raise ValueError("policy must be 'auto', 'deterministic', 'sampled', or 'epsilon-greedy'")


def get_device():
    """
    Get the device being used (cuda or cpu) with explicit logging.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")
    if device == "cpu":
        print("[WARNING] No GPU detected! Training will be significantly slower.")
    return device


def create_train_env(n_envs=8, seed=0):
    """
    Create vectorized training environment for PPO/DQN.

    Parameters:
    -----------
    n_envs : int
        Number of parallel environments
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    VecEnv : Vectorized environment with preprocessing
    """
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")

    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def create_eval_env(render_mode=None):
    """
    Create single evaluation environment with Atari preprocessing.

    Parameters:
    -----------
    render_mode : str or None
        'human' for visual rendering, 'rgb_array' for video recording, None for headless

    Returns:
    --------
    gym.Env : Single environment with preprocessing wrappers
    """
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False
    )
    env = FireResetEnv(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def ensure_dirs():
    """
    Create necessary directories for models and logs.
    """
    for directory in PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
