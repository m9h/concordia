"""Automated research loop for SustainHub experiments.

Runs the experiment ladder across multiple seeds and hyperparameter
configurations, aggregates results with confidence intervals, and
saves a unified leaderboard.

Usage:
    # Quick sanity check (3 seeds, 5 sprints)
    python -m examples.games.sustain_hub.research_loop --mode=quick

    # Full research run (20 seeds, 10 sprints)
    python -m examples.games.sustain_hub.research_loop --mode=full

    # Hyperparameter sweep
    python -m examples.games.sustain_hub.research_loop --mode=sweep
"""

import json
import os
import time
from typing import Any

import numpy as np

from absl import app
from absl import flags
from examples.games.sustain_hub import experiments
from examples.games.sustain_hub import active_inference as aif

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'quick', 'Run mode: quick, full, or sweep.')
flags.DEFINE_string('research_output', '/tmp/sustain_hub_research',
                    'Output directory for research results.')


def run_ladder(num_sprints: int, seed: int, num_agents: int = 8) -> list[dict]:
    """Run the full L0-L7 ladder with a given seed."""
    results = []
    for level in experiments.EXPERIMENT_LEVELS:
        runner = experiments.ExperimentRunner(
            level=level,
            num_agents=num_agents,
            num_sprints=num_sprints,
            seed=seed,
        )
        result = runner.run()
        results.append(result)
    return results


def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    """Aggregate results across seeds into mean +/- std."""
    num_levels = len(all_runs[0])
    aggregated = []

    for level_idx in range(num_levels):
        level_results = [run[level_idx] for run in all_runs]

        # Collect scalar metrics across seeds
        mean_his = [r['mean_hi'] for r in level_results]
        final_his = [r['final_hi'] for r in level_results]
        rqs = [r['resilience_quotient'] for r in level_results]
        coverages = [r['mean_coverage'] for r in level_results]
        diversities = [r['mean_diversity'] for r in level_results]
        strat_divs = [r['strategy_diversity'] for r in level_results]

        # Average HI trajectories (pad shorter ones)
        max_len = max(len(r['hi_trajectory']) for r in level_results)
        padded_trajs = []
        for r in level_results:
            traj = r['hi_trajectory']
            padded = traj + [traj[-1]] * (max_len - len(traj))
            padded_trajs.append(padded)
        mean_traj = np.mean(padded_trajs, axis=0).tolist()
        std_traj = np.std(padded_trajs, axis=0).tolist()

        agg = {
            'level': level_results[0]['level'],
            'level_name': level_results[0]['level_name'],
            'description': level_results[0].get('description', ''),
            'new_concept': level_results[0].get('new_concept', ''),
            'rl_analog': level_results[0].get('rl_analog', ''),
            'aif_mechanism': level_results[0].get('aif_mechanism', ''),
            'num_seeds': len(all_runs),
            'num_sprints': level_results[0]['num_sprints'],
            # Aggregated metrics
            'mean_hi': float(np.mean(mean_his)),
            'mean_hi_std': float(np.std(mean_his)),
            'final_hi': float(np.mean(final_his)),
            'final_hi_std': float(np.std(final_his)),
            'resilience_quotient': float(np.mean(rqs)),
            'rq_std': float(np.std(rqs)),
            'mean_coverage': float(np.mean(coverages)),
            'mean_diversity': float(np.mean(diversities)),
            'strategy_diversity': float(np.mean(strat_divs)),
            # Trajectories
            'hi_trajectory': mean_traj,
            'hi_trajectory_std': std_traj,
        }
        aggregated.append(agg)

    return aggregated


def run_sweep(base_seed: int, num_sprints: int, num_seeds: int) -> list[dict]:
    """Sweep hyperparameters for the key levels (L0, L4, L5, L7)."""
    sweep_configs = [
        {'name': 'alpha_low', 'level': 4, 'alpha': 2.0},
        {'name': 'alpha_high', 'level': 4, 'alpha': 8.0},
        {'name': 'gamma_low', 'level': 6, 'gamma': 0.5},
        {'name': 'gamma_high', 'level': 6, 'gamma': 2.0},
        {'name': 'lr_low', 'level': 5, 'lr': 0.01},
        {'name': 'lr_high', 'level': 5, 'lr': 0.3},
        {'name': 'agents_4', 'level': 7, 'num_agents': 4},
        {'name': 'agents_12', 'level': 7, 'num_agents': 12},
        {'name': 'sprints_20', 'level': 7, 'num_sprints': 20},
    ]

    sweep_results = []
    for config in sweep_configs:
        level_idx = config['level']
        level = experiments.EXPERIMENT_LEVELS[level_idx]
        n_agents = config.get('num_agents', 8)
        n_sprints = config.get('num_sprints', num_sprints)

        runs = []
        for s in range(num_seeds):
            runner = experiments.ExperimentRunner(
                level=level,
                num_agents=n_agents,
                num_sprints=n_sprints,
                seed=base_seed + s,
            )
            # Override hyperparameters
            if 'alpha' in config:
                for agent in runner.agents:
                    agent.alpha = config['alpha']
            if 'gamma' in config:
                for agent in runner.agents:
                    agent.gamma = config['gamma']
            if 'lr' in config:
                for agent in runner.agents:
                    agent.learning_rate = config['lr']

            result = runner.run()
            runs.append(result)

        # Aggregate
        mean_his = [r['mean_hi'] for r in runs]
        rqs = [r['resilience_quotient'] for r in runs]
        sweep_results.append({
            'config': config['name'],
            'level': level_idx,
            'level_name': level.name,
            'param': {k: v for k, v in config.items() if k not in ('name', 'level')},
            'mean_hi': float(np.mean(mean_his)),
            'mean_hi_std': float(np.std(mean_his)),
            'rq': float(np.mean(rqs)),
            'rq_std': float(np.std(rqs)),
            'num_seeds': num_seeds,
        })

    return sweep_results


def print_leaderboard(results: list[dict]) -> None:
    """Print the aggregated leaderboard."""
    print("\n" + "=" * 110)
    print("RESEARCH LEADERBOARD: RL → Active Inference (aggregated across seeds)")
    print("=" * 110)
    print(f"{'Lvl':<4} {'Name':<32} {'Mean HI':>10} {'Final HI':>10} "
          f"{'RQ':>10} {'Coverage':>10} {'Seeds':>6} {'Sprints':>8}")
    print("-" * 110)

    for r in results:
        std_tag = f" ±{r['mean_hi_std']:.3f}" if r.get('mean_hi_std', 0) > 0 else ""
        rq_std = f" ±{r['rq_std']:.3f}" if r.get('rq_std', 0) > 0 else ""
        print(f"{r['level']:<4} {r['level_name']:<32} "
              f"{r['mean_hi']:>6.3f}{std_tag:>7s} "
              f"{r['final_hi']:>6.3f}      "
              f"{r['resilience_quotient']:>6.3f}{rq_std:>7s} "
              f"{r['mean_coverage']:>6.3f}     "
              f"{r['num_seeds']:>4}   "
              f"{r['num_sprints']:>6}")

    print("=" * 110)


def main(argv):
    del argv

    mode = FLAGS.mode
    output_dir = FLAGS.research_output
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    if mode == 'quick':
        num_seeds, num_sprints = 5, 5
    elif mode == 'full':
        num_seeds, num_sprints = 20, 10
    elif mode == 'sweep':
        num_seeds, num_sprints = 10, 10
    else:
        print(f"Unknown mode: {mode}. Use quick, full, or sweep.")
        return

    print(f"Mode: {mode} | Seeds: {num_seeds} | Sprints: {num_sprints}")
    print(f"Output: {output_dir}")

    if mode in ('quick', 'full'):
        # Run ladder across multiple seeds
        all_runs = []
        for seed_offset in range(num_seeds):
            seed = 42 + seed_offset
            print(f"\n--- Seed {seed} ({seed_offset + 1}/{num_seeds}) ---")
            ladder = run_ladder(num_sprints, seed)
            all_runs.append(ladder)

        # Aggregate
        aggregated = aggregate_runs(all_runs)
        print_leaderboard(aggregated)

        # Save
        out_file = os.path.join(output_dir, 'leaderboard.json')
        with open(out_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nLeaderboard saved to {out_file}")

        # Also save as precomputed results for the dashboard
        with open('precomputed_ladder_results.json', 'w') as f:
            json.dump(aggregated, f, indent=2)
        print("Updated precomputed_ladder_results.json")

    elif mode == 'sweep':
        sweep_results = run_sweep(42, num_sprints, num_seeds)

        print("\n" + "=" * 90)
        print("HYPERPARAMETER SWEEP RESULTS")
        print("=" * 90)
        print(f"{'Config':<20} {'Level':<32} {'Mean HI':>12} {'RQ':>12}")
        print("-" * 90)
        for r in sweep_results:
            print(f"{r['config']:<20} L{r['level']}: {r['level_name']:<28} "
                  f"{r['mean_hi']:>6.3f} ±{r['mean_hi_std']:.3f}  "
                  f"{r['rq']:>6.3f} ±{r['rq_std']:.3f}")
        print("=" * 90)

        out_file = os.path.join(output_dir, 'sweep_results.json')
        with open(out_file, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep results saved to {out_file}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    app.run(main)
