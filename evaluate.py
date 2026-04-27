#!/usr/bin/env python3
"""
Model evaluation script for PPO and DQN agents.

Usage:
    python evaluate.py --model models/ppo/ppo_breakout_seed0_final.zip --episodes 100
    python evaluate.py --model models/dqn/dqn_breakout_seed0_final.zip --episodes 100 --record-video
    python evaluate.py --model models/ppo/ppo_breakout_seed0_final.zip --policy sampled
    python evaluate.py --model models/dqn/dqn_breakout_seed0_final.zip --policy deterministic
"""

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed

from utils import (
    DEFAULT_DQN_EVAL_EPSILON,
    create_eval_env,
    ensure_dirs,
    infer_algorithm,
    resolve_policy_determinism,
)


MODEL_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
}
MODEL_SEED_RE = re.compile(r"_seed(?P<seed>-?\d+)(?:_|\.|$)")


def infer_model_seed(model_path):
    """Infer training seed from model filename such as ppo_breakout_seed100_final.zip."""
    match = MODEL_SEED_RE.search(Path(model_path).name)
    return int(match.group("seed")) if match else None


def build_eval_output_path(stats):
    """
    Build a non-colliding default path for evaluation JSON.

    The model seed identifies the trained model. The eval seed identifies the
    stochastic evaluation run. Keeping both prevents seed0/100/200 model
    evaluations from overwriting each other when the evaluation seed is shared.
    """
    algo = stats["algorithm"]
    model_seed = stats.get("model_seed")
    eval_seed = stats.get("eval_seed")
    model_tag = f"seed{model_seed}" if model_seed is not None else Path(stats["model_path"]).stem
    eval_tag = f"evalseed{eval_seed}" if eval_seed is not None else "evalseedNone"
    policy_tag = format_policy_tag(stats)
    return Path("results") / f"{algo}_{model_tag}_{eval_tag}_{policy_tag}_evaluation.json"


def format_policy_tag(stats):
    """Build a filesystem-safe policy tag for evaluation outputs."""
    policy = stats.get("policy", "auto")
    if policy == "epsilon-greedy":
        epsilon = stats.get("dqn_eval_epsilon")
        if epsilon is not None:
            epsilon_tag = f"{epsilon:g}".replace(".", "p")
            return f"{policy}-eps{epsilon_tag}"
    return policy


def evaluate_model(
    model_path,
    num_episodes=100,
    record_video=False,
    verbose=False,
    algo=None,
    policy="auto",
    deterministic=None,
    dqn_eval_epsilon=None,
    seed=0,
):
    """
    Evaluate a trained model on Breakout.

    Parameters:
    -----------
    model_path : str
        Path to saved model file
    num_episodes : int
        Number of episodes to evaluate
    record_video : bool
        Whether to record video of best episodes
    verbose : bool
        Print detailed per-episode results
    algo : str or None
        Algorithm name ('ppo' or 'dqn'), inferred from model path if None
    policy : str
        Policy prediction mode: 'auto', 'deterministic', 'sampled', or
        'epsilon-greedy'. The epsilon-greedy mode is DQN-only.
    deterministic : bool or None
        Backward-compatible override for policy. Prefer policy for new code.
    dqn_eval_epsilon : float or None
        Epsilon used by DQN epsilon-greedy evaluation. Defaults to 0.05 for
        DQN auto or explicit epsilon-greedy evaluation.
    seed : int or None
        Evaluation seed. Use None for unseeded episodes.

    Returns:
    --------
    dict : Evaluation statistics
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    ensure_dirs()

    algo = (algo or infer_algorithm(model_path, MODEL_CLASSES.keys())).lower()
    if algo not in MODEL_CLASSES:
        raise ValueError("algo must be 'ppo' or 'dqn'")
    if deterministic is not None:
        policy = "deterministic" if deterministic else "sampled"
    if policy == "auto" and algo == "dqn":
        policy = "epsilon-greedy"
    if policy == "epsilon-greedy":
        if algo != "dqn":
            raise ValueError("--policy epsilon-greedy is only supported for DQN")
        if dqn_eval_epsilon is None:
            dqn_eval_epsilon = DEFAULT_DQN_EVAL_EPSILON
    elif dqn_eval_epsilon is not None:
        raise ValueError("--dqn-eval-eps requires --policy epsilon-greedy")
    if dqn_eval_epsilon is not None and not 0.0 <= dqn_eval_epsilon <= 1.0:
        raise ValueError("--dqn-eval-eps must be in [0, 1]")
    deterministic = resolve_policy_determinism(algo, policy)

    if seed is not None:
        set_random_seed(seed)

    model_seed = infer_model_seed(model_path)
    model_tag = f"seed{model_seed}" if model_seed is not None else Path(model_path).stem
    eval_tag = f"evalseed{seed}" if seed is not None else "evalseedNone"
    policy_tag = format_policy_tag({
        "policy": policy,
        "dqn_eval_epsilon": dqn_eval_epsilon,
    })

    print(f"Loading model from {model_path}")
    model = MODEL_CLASSES[algo].load(model_path, device="auto")
    if policy == "epsilon-greedy":
        model.exploration_rate = dqn_eval_epsilon

    cv2 = None
    best_video_path = None
    best_video_score = float("-inf")
    best_video_episode = None
    if record_video:
        try:
            import cv2 as cv2_module
            cv2 = cv2_module
        except ImportError:
            print("Warning: cv2 not installed, video recording disabled")

    # When recording, stream each episode to a temporary file and retain only
    # the highest-scoring episode. This avoids keeping all frames in memory and
    # avoids saving every episode to disk.
    env = create_eval_env(render_mode="rgb_array" if cv2 else None)
    if seed is not None:
        env.action_space.seed(seed)

    scores = []
    steps_list = []

    print(f"Evaluating {algo.upper()} model for {num_episodes} episodes...")
    if policy == "epsilon-greedy":
        print(f"Policy prediction: epsilon-greedy (epsilon={dqn_eval_epsilon:g})")
    else:
        print(f"Policy prediction: {'deterministic' if deterministic else 'sampled'}")
    if cv2:
        print("Video recording: keeping only the highest-scoring episode")

    try:
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            reset_seed = None if seed is None else seed + episode
            obs, _ = env.reset(seed=reset_seed)
            done = False
            episode_reward = 0
            steps = 0
            writer = None
            temp_video_path = None

            if cv2:
                temp_video_path = Path("videos") / (
                    f".tmp_{algo}_{model_tag}_{eval_tag}_{policy_tag}_episode{episode + 1}.mp4"
                )

            try:
                while not done:
                    if cv2:
                        frame = env.render()
                        if writer is None:
                            height, width = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(temp_video_path), fourcc, 15.0, (width, height))
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    steps += 1
            finally:
                if writer is not None:
                    writer.release()

            if cv2 and temp_video_path and temp_video_path.exists():
                if episode_reward > best_video_score:
                    if best_video_path and best_video_path.exists():
                        best_video_path.unlink()

                    best_video_score = float(episode_reward)
                    best_video_episode = episode + 1
                    best_video_path = Path("videos") / (
                        f"{algo}_{model_tag}_{eval_tag}_{policy_tag}_best_episode{best_video_episode}_"
                        f"score{int(best_video_score)}.mp4"
                    )
                    if best_video_path.exists():
                        best_video_path.unlink()
                    temp_video_path.replace(best_video_path)
                else:
                    temp_video_path.unlink()

            scores.append(episode_reward)
            steps_list.append(steps)

            if verbose:
                print(f"Episode {episode+1}: Score={episode_reward}, Steps={steps}")
    finally:
        env.close()

    # Calculate statistics
    stats = {
        "model_path": model_path,
        "algorithm": algo,
        "model_seed": model_seed,
        "eval_seed": seed,
        "num_episodes": num_episodes,
        "policy": policy,
        "deterministic": deterministic,
        "dqn_eval_epsilon": dqn_eval_epsilon,
        "seed": seed,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_steps": float(np.mean(steps_list)),
        "scores": [float(s) for s in scores],
        "steps": [int(s) for s in steps_list],
        "best_video_path": str(best_video_path) if best_video_path else None,
        "best_video_episode": best_video_episode,
    }

    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results for {algo.upper()}")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"Median Score: {stats['median_score']:.2f}")
    print(f"Min/Max Score: {stats['min_score']:.0f}/{stats['max_score']:.0f}")
    print(f"Average Steps per Episode: {stats['mean_steps']:.2f}")

    if best_video_path:
        print(f"Saved best video: {best_video_path}")

    return stats


def save_results(stats, output_path=None):
    """
    Save evaluation results to JSON file.

    Parameters:
    -----------
    stats : dict
        Evaluation statistics
    output_path : str or None
        Path to save results, auto-generated if None
    """
    if output_path is None:
        output_path = build_eval_output_path(stats)
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Breakout agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "dqn"],
        default=None,
        help="Algorithm type. Inferred from model path if omitted"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of best episodes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode results"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["auto", "deterministic", "sampled", "epsilon-greedy"],
        default="auto",
        help=(
            "Policy prediction mode. auto uses sampled PPO and DQN epsilon-greedy "
            "with epsilon=0.05; epsilon-greedy is DQN-only."
        )
    )
    parser.add_argument(
        "--dqn-eval-eps",
        type=float,
        default=None,
        help="DQN epsilon for --policy epsilon-greedy. Defaults to 0.05."
    )
    parser.add_argument(
        "--deterministic",
        action="store_const",
        const="deterministic",
        dest="policy_override",
        default=None,
        help="Shortcut for --policy deterministic"
    )
    parser.add_argument(
        "--sampled",
        action="store_const",
        const="sampled",
        dest="policy_override",
        help="Shortcut for --policy sampled"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Evaluation seed. Use -1 for unseeded evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON"
    )

    args = parser.parse_args()

    policy = args.policy_override or args.policy

    stats = evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        record_video=args.record_video,
        verbose=args.verbose,
        algo=args.algo,
        policy=policy,
        dqn_eval_epsilon=args.dqn_eval_eps,
        seed=None if args.seed < 0 else args.seed,
    )

    save_results(stats, args.output)
