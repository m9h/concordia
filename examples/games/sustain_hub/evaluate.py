#!/usr/bin/env python3
"""Evaluation helper for SustainHub autoresearch experiments.

Runs simulations, computes the composite SustainScore, and logs results.

Usage:
  # Run a single fast experiment:
  python -m examples.games.sustain_hub.evaluate

  # Run N experiments and average (handles stochasticity):
  python -m examples.games.sustain_hub.evaluate --runs=3

  # Append result to results.tsv:
  python -m examples.games.sustain_hub.evaluate --log --description="baseline"

  # Full-scale run (16 agents, 5 sprints, stress):
  python -m examples.games.sustain_hub.evaluate --tier=3 --log
"""

import json
import math
import os
import subprocess
import sys
import time

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('runs', 1, 'Number of runs to average.')
flags.DEFINE_bool('log', False, 'Append result to results.tsv.')
flags.DEFINE_string('description', '', 'Experiment description for results.tsv.')
flags.DEFINE_string('output_dir', '/tmp/sustain_hub_autoresearch', 'Output directory.')
flags.DEFINE_integer('tier', 1, 'Experiment tier: 1=fast (4 agents, 1 sprint), '
                     '2=medium (8 agents, 3 sprints), 3=full (16 agents, 5 sprints).')

TIER_CONFIGS = {
    1: {'num_sprints': 1, 'community_size': 4, 'enable_stress': False},
    2: {'num_sprints': 3, 'community_size': 8, 'enable_stress': True},
    3: {'num_sprints': 5, 'community_size': 16, 'enable_stress': True},
}


def compute_fairness(scores: dict) -> float:
    """Compute fairness as 1 - normalized Gini coefficient of agent scores."""
    if not scores:
        return 1.0
    values = sorted(scores.values())
    n = len(values)
    if n <= 1:
        return 1.0
    # Shift scores to be non-negative for Gini calculation
    min_val = min(values)
    shifted = [v - min_val for v in values]
    total = sum(shifted)
    if total == 0:
        return 1.0  # All equal → perfectly fair
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(shifted):
        cumulative += v
        gini_sum += cumulative
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    return max(0.0, 1.0 - gini)


def compute_strategy_diversity(sprint_history: list, player_roles: dict) -> float:
    """Fraction of agents that changed task type across sprints."""
    if len(sprint_history) < 2:
        return 1.0  # Can't measure with 1 sprint; assume full diversity
    agents = set()
    for sprint in sprint_history:
        agents.update(sprint.get('joint_action', {}).keys())
    if not agents:
        return 0.0
    changers = 0
    for agent in agents:
        tasks = []
        for sprint in sprint_history:
            task = sprint.get('joint_action', {}).get(agent)
            if task:
                tasks.append(task)
        if len(tasks) >= 2 and len(set(tasks)) > 1:
            changers += 1
    return changers / len(agents) if agents else 0.0


def compute_sustain_score(data: dict) -> dict:
    """Compute composite SustainScore from results.json data."""
    hi = data.get('harmony_index', 0.0)
    rq = data.get('resilience_quotient', 0.0)
    scores = data.get('scores', {})
    sprint_history = data.get('sprint_history', [])
    player_roles = data.get('player_roles', {})

    fairness = compute_fairness(scores)
    strategy_div = compute_strategy_diversity(sprint_history, player_roles)

    # StressValidity: placeholder (would need stress vs no-stress comparison)
    # For single runs, assume 1.0 if stress was enabled, 0.5 if not
    stress_validity = 1.0 if data.get('dropout_name') else 0.5

    sustain_score = hi * (1 + rq) * fairness * strategy_div * stress_validity

    return {
        'sustain_score': sustain_score,
        'harmony_index': hi,
        'resilience_quotient': rq,
        'fairness': fairness,
        'strategy_diversity': strategy_div,
        'stress_validity': stress_validity,
    }


def run_once(run_id: int) -> dict:
    """Run one simulation and return results dict with SustainScore."""
    tier = TIER_CONFIGS[FLAGS.tier]
    out_dir = os.path.join(FLAGS.output_dir, f'run_{run_id}')
    cmd = [
        sys.executable, '-m', 'examples.games.sustain_hub.run',
        f'--num_sprints={tier["num_sprints"]}',
        f'--community_size={tier["community_size"]}',
        '--skip_backstory',
        f'--output_dir={out_dir}',
    ]
    if not tier['enable_stress']:
        cmd.append('--noenable_stress')

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    duration = time.time() - t0

    if result.returncode != 0:
        print(f'Run {run_id} CRASHED ({duration:.0f}s)', file=sys.stderr)
        stderr_tail = result.stderr[-500:] if result.stderr else ''
        stdout_tail = result.stdout[-500:] if result.stdout else ''
        print(stderr_tail or stdout_tail or 'no output', file=sys.stderr)
        return {
            'sustain_score': 0.0, 'harmony_index': 0.0,
            'resilience_quotient': 0.0, 'fairness': 0.0,
            'strategy_diversity': 0.0, 'duration': duration, 'status': 'crash',
        }

    results_path = os.path.join(out_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f'Run {run_id}: no results.json found', file=sys.stderr)
        return {
            'sustain_score': 0.0, 'harmony_index': 0.0,
            'resilience_quotient': 0.0, 'fairness': 0.0,
            'strategy_diversity': 0.0, 'duration': duration, 'status': 'crash',
        }

    with open(results_path) as f:
        data = json.load(f)

    metrics = compute_sustain_score(data)
    metrics['duration'] = duration
    metrics['status'] = 'ok'
    return metrics


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def main(argv):
    del argv

    tier = TIER_CONFIGS[FLAGS.tier]
    print(f'Tier {FLAGS.tier}: {tier["community_size"]} agents, '
          f'{tier["num_sprints"]} sprints, '
          f'stress={"on" if tier["enable_stress"] else "off"}')

    results = []
    for i in range(FLAGS.runs):
        print(f'\n--- Run {i+1}/{FLAGS.runs} ---')
        r = run_once(i)
        results.append(r)
        print(f'  SustainScore={r["sustain_score"]:.4f}  '
              f'HI={r["harmony_index"]:.4f}  RQ={r["resilience_quotient"]:.4f}  '
              f'Fair={r["fairness"]:.2f}  Div={r["strategy_diversity"]:.2f}  '
              f'({r["duration"]:.0f}s)  [{r["status"]}]')

    ok_results = [r for r in results if r['status'] == 'ok']
    if not ok_results:
        print('\nAll runs crashed.')
        return

    def avg(key):
        return sum(r[key] for r in ok_results) / len(ok_results)

    avg_ss = avg('sustain_score')
    avg_hi = avg('harmony_index')
    avg_rq = avg('resilience_quotient')
    avg_fair = avg('fairness')
    avg_div = avg('strategy_diversity')
    total_duration = sum(r['duration'] for r in results)

    print(f'\n{"="*60}')
    print(f'RESULT (avg of {len(ok_results)}/{FLAGS.runs} successful runs)')
    print(f'{"="*60}')
    print(f'  SustainScore:        {avg_ss:.4f}')
    print(f'  Harmony Index:       {avg_hi:.4f}')
    print(f'  Resilience Quotient: {avg_rq:.4f}')
    print(f'  Fairness:            {avg_fair:.4f}')
    print(f'  Strategy Diversity:  {avg_div:.4f}')
    print(f'  Total time:          {total_duration:.0f}s')

    if FLAGS.log:
        commit = get_git_hash()
        desc = FLAGS.description or 'no description'
        tsv_path = os.path.join(os.path.dirname(__file__), 'results.tsv')
        if not os.path.exists(tsv_path):
            with open(tsv_path, 'w') as f:
                f.write('commit\tsustain_score\tharmony_index\tresilience_quotient'
                        '\tfairness\tstrategy_div\tstatus\tdescription\n')
        with open(tsv_path, 'a') as f:
            status = 'ok' if ok_results else 'crash'
            f.write(f'{commit}\t{avg_ss:.4f}\t{avg_hi:.4f}\t{avg_rq:.4f}'
                    f'\t{avg_fair:.4f}\t{avg_div:.4f}\t{status}\t{desc}\n')
        print(f'\nLogged to {tsv_path}')


if __name__ == '__main__':
    app.run(main)
