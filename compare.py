#!/usr/bin/env python3
"""
Performance comparison script for PPO vs DQN on Breakout.

Generates comparison charts and exports analysis data.

Usage:
    python compare.py --seeds 0 100 200
"""

import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import argparse
import json
import re
from pathlib import Path

import numpy as np

from utils import DEFAULT_DQN_EVAL_EPSILON

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib not installed, charts will be skipped")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("Warning: tensorboard not installed, log parsing will be skipped")


ALGORITHMS = ("ppo", "dqn")
DEFAULT_MILESTONES = (1, 5, 10, 15, 20)
DEFAULT_MIN_LOG_STEPS = 1_000_000
FIG_SIZE = (3.5, 2.625)
FIG_DPI = 300
COLORS = {
    "ppo": "#0072B2",
    "dqn": "#D55E00",
}
MARKERS = {
    "ppo": "o",
    "dqn": "s",
}
EVAL_FILE_RE = re.compile(
    r"^(?P<algo>ppo|dqn)_seed(?P<model_seed>-?\d+)"
    r"(?:_evalseed(?P<eval_seed>-?\d+|None))?"
    r"(?:_(?P<policy>auto|sampled|deterministic|epsilon-greedy(?:-eps[0-9p]+)?))?"
    r"_evaluation\.json$"
)
RUN_SEED_RE = re.compile(r"^(?:PPO|DQN)_seed(?P<seed>-?\d+)", re.IGNORECASE)


def normalize_policy_name(stats, filename_policy=None):
    """Return the effective policy name recorded by an evaluation result."""
    policy = stats.get("policy") or filename_policy or "legacy"
    if policy.startswith("epsilon-greedy"):
        return "epsilon-greedy"
    return policy


def evaluation_result_rank(algo, stats, filename_policy=None):
    """
    Rank multiple evaluation JSON files for the same algorithm/seed.

    This keeps old diagnostic files from overriding the final protocol when
    pure-greedy and epsilon-greedy DQN results coexist in results/.
    """
    policy = normalize_policy_name(stats, filename_policy)
    if algo == "dqn":
        epsilon = stats.get("dqn_eval_epsilon")
        if policy == "epsilon-greedy" and epsilon == DEFAULT_DQN_EVAL_EPSILON:
            return 40
        if policy == "epsilon-greedy":
            return 30
        if policy in {"deterministic", "auto"}:
            return 10
        return 0
    if policy == "auto":
        return 30
    if policy == "sampled":
        return 20
    return 10


def configure_plot_style():
    """Use a compact, publication-oriented style for single-column figures."""
    if not HAS_PLOT:
        return

    plt.rcParams.update({
        "figure.figsize": FIG_SIZE,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(output_path):
    """Save current Matplotlib figure with consistent export parameters."""
    plt.tight_layout(pad=0.35)
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {output_path}")


def format_run_label(algo, run_id):
    """Create concise legend labels for TensorBoard run directories."""
    seed_match = RUN_SEED_RE.match(run_id)
    if seed_match:
        return f"{algo.upper()} s{seed_match.group('seed')}"
    return f"{algo.upper()} {run_id}"


def read_tensorboard_run(run_dir, metric="rollout/ep_rew_mean"):
    """
    Read one TensorBoard run directory and extract scalar values.

    Returns:
    --------
    tuple : (steps, values) arrays, or (None, None) if the metric is missing.
    """
    if not HAS_TB:
        return None, None

    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None, None

    try:
        accumulator = EventAccumulator(str(run_dir))
        accumulator.Reload()
    except Exception as exc:
        print(f"Warning: Could not read {run_dir}: {exc}")
        return None, None

    if metric not in accumulator.Tags().get("scalars", []):
        return None, None

    events = accumulator.Scalars(metric)
    if not events:
        return None, None

    pairs = sorted((event.step, event.value) for event in events)
    steps, values = zip(*pairs)
    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_tensorboard_logs(log_dir="logs", min_steps=DEFAULT_MIN_LOG_STEPS):
    """
    Read TensorBoard runs for all algorithms.

    SB3 log directories are auto-numbered (for example PPO_1, DQN_1) and do
    not contain the seed. To avoid false seed/run associations, training
    curves are keyed by run directory name instead of seed.
    """
    logs = {algo: {} for algo in ALGORITHMS}
    log_dir = Path(log_dir)

    if not HAS_TB:
        return logs

    for algo in ALGORITHMS:
        algo_dir = log_dir / algo
        prefix = algo.upper()

        if not algo_dir.exists():
            print(f"Warning: log directory not found: {algo_dir}")
            continue

        for run_dir in sorted(path for path in algo_dir.iterdir() if path.is_dir()):
            if not run_dir.name.startswith(prefix):
                continue

            steps, rewards = read_tensorboard_run(run_dir)
            if steps is None:
                print(f"  Skipped {prefix} {run_dir.name}: no rollout/ep_rew_mean")
                continue

            max_step = int(steps[-1])
            if max_step < min_steps:
                print(f"  Skipped {prefix} {run_dir.name}: only {max_step} steps")
                continue

            logs[algo][run_dir.name] = (steps, rewards)
            print(f"  {prefix} {run_dir.name}: {len(steps)} points, max_step={max_step}")

    return logs


def read_evaluation_results(results_dir="results", seeds=None):
    """
    Read evaluation JSON files.

    Returns:
    --------
    dict : {algo: {seed: stats_dict}}
    """
    results = {algo: {} for algo in ALGORITHMS}
    selected_ranks = {algo: {} for algo in ALGORITHMS}
    seed_filter = None if seeds is None else set(seeds)

    for json_file in sorted(Path(results_dir).glob("*_seed*_evaluation.json")):
        match = EVAL_FILE_RE.match(json_file.name)
        if not match:
            continue

        algo = match.group("algo")
        filename_seed = int(match.group("model_seed"))
        if seed_filter is not None and filename_seed not in seed_filter:
            continue

        try:
            with json_file.open(encoding="utf-8") as file:
                stats = json.load(file)
        except Exception as exc:
            print(f"Warning: Could not read {json_file}: {exc}")
            continue

        stat_algo = stats.get("algorithm", algo)
        stat_seed = stats.get("model_seed", stats.get("training_seed", filename_seed))

        if stat_algo != algo:
            print(f"Warning: Ignoring {json_file}, algorithm mismatch: {stat_algo} != {algo}")
            continue
        if stat_seed is not None and int(stat_seed) != filename_seed:
            print(f"Warning: Ignoring {json_file}, model seed mismatch: {stat_seed} != {filename_seed}")
            continue

        rank = evaluation_result_rank(algo, stats, match.group("policy"))
        previous_rank = selected_ranks[algo].get(filename_seed, -1)
        if filename_seed in results[algo] and rank <= previous_rank:
            previous_policy = results[algo][filename_seed].get("policy", "legacy")
            current_policy = stats.get("policy", match.group("policy") or "legacy")
            print(
                f"  Skipped {json_file.name}: policy {current_policy} has lower priority "
                f"than selected {algo} seed{filename_seed} result ({previous_policy})"
            )
            continue

        if filename_seed in results[algo]:
            previous_policy = results[algo][filename_seed].get("policy", "legacy")
            current_policy = stats.get("policy", match.group("policy") or "legacy")
            print(
                f"  Selected {json_file.name}: policy {current_policy} overrides "
                f"{previous_policy} for {algo} seed{filename_seed}"
            )

        results[algo][filename_seed] = stats
        selected_ranks[algo][filename_seed] = rank

    for seed, rank in selected_ranks["dqn"].items():
        if rank < 40:
            policy = results["dqn"][seed].get("policy", "legacy")
            print(
                f"Warning: DQN seed{seed} selected policy is {policy}, not the final "
                f"epsilon-greedy epsilon={DEFAULT_DQN_EVAL_EPSILON:g} protocol"
            )

    return results


def compute_sample_efficiency(steps, rewards, milestones=DEFAULT_MILESTONES, tolerance=0.1):
    """
    Compute reward near milestone steps.

    Parameters:
    -----------
    steps : np.ndarray
        Training steps
    rewards : np.ndarray
        Corresponding rewards
    milestones : iterable
        Milestone steps in millions
    tolerance : float
        Maximum relative distance from each milestone
    """
    if steps is None or rewards is None or len(steps) == 0:
        return {}

    efficiency = {}
    for milestone in milestones:
        target_step = milestone * 1_000_000
        idx = int(np.argmin(np.abs(steps - target_step)))
        relative_error = abs(int(steps[idx]) - target_step) / target_step
        if relative_error <= tolerance:
            efficiency[f"{milestone}M"] = float(rewards[idx])
    return efficiency


def compute_stability(rewards, window_size=100):
    """
    Compute training stability metrics from a reward series.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.size == 0:
        return {
            "reward_variance": 0.0,
            "reward_std": 0.0,
            "mean_rolling_std": 0.0,
            "max_rolling_std": 0.0,
            "smoothness": 0.0,
        }

    window_size = min(window_size, rewards.size)
    if window_size < 2:
        rolling_mean = rewards
        rolling_std = np.array([0.0])
    else:
        kernel = np.ones(window_size) / window_size
        rolling_mean = np.convolve(rewards, kernel, mode="valid")
        rolling_std = np.array([
            np.std(rewards[start:start + window_size])
            for start in range(len(rolling_mean))
        ])

    return {
        "reward_variance": float(np.var(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_rolling_std": float(np.mean(rolling_std)) if rolling_std.size else 0.0,
        "max_rolling_std": float(np.max(rolling_std)) if rolling_std.size else 0.0,
        "smoothness": float(np.mean(np.abs(np.diff(rolling_mean)))) if rolling_mean.size > 1 else 0.0,
    }


def aggregate_eval_scores(eval_results):
    """Compute aggregate final-performance metrics by algorithm."""
    summary = {}
    for algo in ALGORITHMS:
        scores = [
            float(stats["mean_score"])
            for stats in eval_results[algo].values()
            if "mean_score" in stats
        ]
        summary[algo] = {
            "num_seeds": len(scores),
            "mean_score": float(np.mean(scores)) if scores else None,
            "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "seeds": sorted(eval_results[algo].keys()),
        }
    return summary


def plot_learning_curves(training_logs, output_dir="results"):
    """Plot learning curves for all discovered training runs."""
    if not HAS_PLOT:
        return

    has_data = any(training_logs[algo] for algo in ALGORITHMS)
    if not has_data:
        print("No TensorBoard reward data found; skipping learning curve plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    plt.figure(figsize=FIG_SIZE)

    for algo in ALGORITHMS:
        for run_id, (steps, rewards) in training_logs[algo].items():
            plt.plot(
                steps / 1_000_000,
                rewards,
                label=format_run_label(algo, run_id),
                alpha=0.75,
                color=COLORS[algo],
            )

    plt.xlabel("Training steps (millions)")
    plt.ylabel("Mean episode reward")
    plt.legend(ncol=2, loc="lower right", handlelength=1.5, columnspacing=0.8)
    plt.grid(True, alpha=0.3)
    output_path = output_dir / "learning_curves.png"
    save_figure(output_path)


def plot_final_performance(eval_results, output_dir="results"):
    """Plot final performance comparison from evaluation JSON files."""
    if not HAS_PLOT:
        return

    summary = aggregate_eval_scores(eval_results)
    if not any(summary[algo]["num_seeds"] for algo in ALGORITHMS):
        print("No evaluation results found; skipping final performance plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    means = [
        summary[algo]["mean_score"] if summary[algo]["mean_score"] is not None else 0
        for algo in ALGORITHMS
    ]
    stds = [summary[algo]["std_score"] for algo in ALGORITHMS]

    configure_plot_style()
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(ALGORITHMS))
    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[COLORS[algo] for algo in ALGORITHMS],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
    )

    plt.xticks(x, [algo.upper() for algo in ALGORITHMS])
    plt.ylabel("Mean evaluation score")

    for bar, mean, std in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 8,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.grid(True, axis="y")
    output_path = output_dir / "final_performance.png"
    save_figure(output_path)


def plot_sample_efficiency(training_logs, output_dir="results"):
    """Plot mean reward at sample-efficiency milestones."""
    if not HAS_PLOT:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    milestone_labels = [f"{milestone}M" for milestone in DEFAULT_MILESTONES]
    efficiency_by_algo = {algo: {label: [] for label in milestone_labels} for algo in ALGORITHMS}

    for algo in ALGORITHMS:
        for steps, rewards in training_logs[algo].values():
            efficiency = compute_sample_efficiency(steps, rewards)
            for label, reward in efficiency.items():
                efficiency_by_algo[algo][label].append(reward)

    valid_labels = [
        label
        for label in milestone_labels
        if any(efficiency_by_algo[algo][label] for algo in ALGORITHMS)
    ]
    if not valid_labels:
        print("No milestone data found; skipping sample efficiency plot")
        return

    configure_plot_style()
    x = np.arange(len(valid_labels))
    plt.figure(figsize=FIG_SIZE)

    for algo in ALGORITHMS:
        means = [
            float(np.mean(efficiency_by_algo[algo][label]))
            if efficiency_by_algo[algo][label]
            else np.nan
            for label in valid_labels
        ]
        plt.plot(
            x,
            means,
            marker=MARKERS[algo],
            label=algo.upper(),
            color=COLORS[algo],
        )

    plt.xticks(x, valid_labels)
    plt.xlabel("Training progress")
    plt.ylabel("Average reward")
    plt.legend(loc="lower right", handlelength=1.5)
    plt.grid(True, alpha=0.3)
    output_path = output_dir / "sample_efficiency.png"
    save_figure(output_path)


def summarize_training_logs(training_logs):
    """Build JSON-serializable training-run summaries."""
    summary = {algo: {} for algo in ALGORITHMS}

    for algo in ALGORITHMS:
        for run_id, (steps, rewards) in training_logs[algo].items():
            seed_match = RUN_SEED_RE.match(run_id)
            summary[algo][run_id] = {
                "seed": int(seed_match.group("seed")) if seed_match else None,
                "max_step": int(steps[-1]),
                "num_points": int(len(steps)),
                "sample_efficiency": compute_sample_efficiency(steps, rewards),
                "training_stability": compute_stability(rewards),
            }

    return summary


def generate_report(training_logs, eval_results, output_dir="results"):
    """
    Generate comparison report as JSON and print a concise summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "training_runs": summarize_training_logs(training_logs),
        "final_performance": aggregate_eval_scores(eval_results),
        "evaluation_results": eval_results,
    }

    report_path = output_dir / "comparison_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved: {report_path}")

    print("\n" + "=" * 60)
    print("PPO vs DQN Comparison Summary")
    print("=" * 60)

    for algo in ALGORITHMS:
        final = report["final_performance"][algo]
        print(f"\n{algo.upper()} Evaluation Results:")
        if final["num_seeds"]:
            print(f"  Mean: {final['mean_score']:.2f} ± {final['std_score']:.2f}")
            print(f"  Seeds: {final['seeds']}")
        else:
            print("  No evaluation JSON found")

        run_ids = list(report["training_runs"][algo])
        print(f"  Training runs: {run_ids if run_ids else 'none'}")

    ppo_mean = report["final_performance"]["ppo"]["mean_score"]
    dqn_mean = report["final_performance"]["dqn"]["mean_score"]
    if ppo_mean is not None and dqn_mean is not None:
        diff = ppo_mean - dqn_mean
        winner = "PPO" if diff > 0 else "DQN"
        print(f"\nWinner by evaluation mean: {winner} (+{abs(diff):.2f} score)")

    print("=" * 60)

    return report


def compare_algorithms(
    seeds=(0, 100, 200),
    log_dir="logs",
    results_dir="results",
    min_log_steps=DEFAULT_MIN_LOG_STEPS,
):
    """
    Main comparison function.
    """
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading TensorBoard logs...")
    training_logs = read_tensorboard_logs(log_dir, min_steps=min_log_steps)

    print("Reading evaluation results...")
    eval_results = read_evaluation_results(results_dir, seeds=seeds)

    print("\nGenerating comparison charts...")
    plot_learning_curves(training_logs, output_dir)
    plot_final_performance(eval_results, output_dir)
    plot_sample_efficiency(training_logs, output_dir)

    return generate_report(training_logs, eval_results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO and DQN performance")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 100, 200],
        help="Evaluation seeds to include in comparison"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Evaluation results and output directory"
    )
    parser.add_argument(
        "--min-log-steps",
        type=int,
        default=DEFAULT_MIN_LOG_STEPS,
        help="Ignore TensorBoard runs shorter than this many timesteps"
    )

    args = parser.parse_args()

    compare_algorithms(
        seeds=args.seeds,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        min_log_steps=args.min_log_steps,
    )
