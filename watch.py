#!/usr/bin/env python3
"""
Watch trained agent play Breakout in real-time.

Usage:
    python watch.py --model models/ppo/ppo_breakout_seed0_final.zip --episodes 5
    python watch.py --model models/dqn/dqn_breakout_seed0_final.zip --policy deterministic
"""

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import argparse
import time

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed

from utils import DEFAULT_DQN_EVAL_EPSILON, create_eval_env, infer_algorithm, resolve_policy_determinism


MODEL_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
}


def watch_agent(
    model_path,
    num_episodes=5,
    algo=None,
    policy="auto",
    deterministic=None,
    dqn_eval_epsilon=None,
    seed=0,
    delay=0.02,
):
    """
    Watch a trained agent play Breakout with visual rendering.

    Parameters:
    -----------
    model_path : str
        Path to saved model file
    num_episodes : int
        Number of episodes to watch
    algo : str or None
        Algorithm name ('ppo' or 'dqn'), inferred from model path if None
    policy : str
        Policy prediction mode: 'auto', 'deterministic', 'sampled', or
        'epsilon-greedy'. The epsilon-greedy mode is DQN-only.
    deterministic : bool or None
        Backward-compatible override for policy. Prefer policy for new code.
    dqn_eval_epsilon : float or None
        Epsilon used by DQN epsilon-greedy playback. Defaults to 0.05 for
        DQN auto or explicit epsilon-greedy playback.
    seed : int or None
        Base seed for playback. Use None for unseeded episodes.
    delay : float
        Sleep duration after each environment step
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")

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

    print(f"Loading {algo.upper()} model from {model_path}")
    model = MODEL_CLASSES[algo].load(model_path, device="auto")
    if policy == "epsilon-greedy":
        model.exploration_rate = dqn_eval_epsilon

    # Create environment with visual rendering
    env = create_eval_env(render_mode="human")
    if seed is not None:
        env.action_space.seed(seed)

    print(f"\nWatching {algo.upper()} agent play {num_episodes} episodes...")
    if policy == "epsilon-greedy":
        print(f"Policy prediction: epsilon-greedy (epsilon={dqn_eval_epsilon:g})")
    else:
        print(f"Policy prediction: {'deterministic' if deterministic else 'sampled'}")
    print("Press Ctrl+C to stop\n")

    total_reward = 0

    try:
        for episode in range(num_episodes):
            reset_seed = None if seed is None else seed + episode
            obs, _ = env.reset(seed=reset_seed)
            done = False
            episode_reward = 0
            step = 0

            print(f"=== Episode {episode+1}/{num_episodes} ===")

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step += 1

                # Slow down for smooth viewing
                if delay:
                    time.sleep(delay)

                # Progress update
                if step % 100 == 0:
                    print(f"  Step {step}, Score: {episode_reward}")

            total_reward += episode_reward
            print(f"  Episode finished. Final Score: {episode_reward}\n")

        avg_reward = total_reward / num_episodes
        print(f"Average score over {num_episodes} episodes: {avg_reward:.2f}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained agent play Breakout")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "dqn"],
        default=None,
        help="Algorithm type. Inferred from model path if omitted"
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
        help="Playback seed. Use -1 for unseeded playback"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="Seconds to sleep after each environment step"
    )

    args = parser.parse_args()

    policy = args.policy_override or args.policy

    watch_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        algo=args.algo,
        policy=policy,
        dqn_eval_epsilon=args.dqn_eval_eps,
        seed=None if args.seed < 0 else args.seed,
        delay=args.delay,
    )
