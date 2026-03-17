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
from typing import Any

import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('runs', 1, 'Number of runs to average.')
flags.DEFINE_bool('log', False, 'Append result to results.tsv.')
flags.DEFINE_string('description', '', 'Experiment description for results.tsv.')
flags.DEFINE_string('output_dir', '/tmp/sustain_hub_autoresearch', 'Output directory.')
flags.DEFINE_integer('tier', 1, 'Experiment tier: 1=fast (4 agents, 1 sprint), '
                     '2=medium (8 agents, 3 sprints), 3=full (16 agents, 5 sprints).')
flags.DEFINE_string('vllm_url', None, 'vLLM API base URL (e.g. http://localhost:8000/v1).')
flags.DEFINE_string('model_name', None, 'Model name to use with vLLM or Vertex AI.')
flags.DEFINE_bool('use_mock', False, 'Use mock model instead of a real LLM.')

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


def compute_brs(sprint_history: list, player_roles: dict) -> dict:
    """Burnout Risk Score per agent.

    BRS = max(consecutive non-preferred task sprints) / total sprints.
    Lower is better. High BRS indicates burnout risk.
    """
    role_to_preferred = {
        'Contributor': 'bug_fix',
        'Innovator': 'feature',
        'Knowledge Curator': 'documentation',
        'Maintainer': 'code_review',
    }

    # Build per-agent task type sequence across sprints
    agent_tasks = {}  # name -> list of task_type_or_skip
    for sprint in sprint_history:
        joint_action = sprint.get('joint_action', {})
        for name, task in joint_action.items():
            if name not in agent_tasks:
                agent_tasks[name] = []
            # Determine task type from the task label
            # Task labels start with "Fix:", "Feature:", "Docs:", "Review:"
            if not task or task.lower().startswith('skip'):
                agent_tasks[name].append('skip')
            elif task.startswith('Fix:'):
                agent_tasks[name].append('bug_fix')
            elif task.startswith('Feature:'):
                agent_tasks[name].append('feature')
            elif task.startswith('Docs:') or task.startswith('Doc:'):
                agent_tasks[name].append('documentation')
            elif task.startswith('Review:'):
                agent_tasks[name].append('code_review')
            else:
                agent_tasks[name].append('unknown')

    brs = {}
    for name, tasks in agent_tasks.items():
        if not tasks:
            brs[name] = 0.0
            continue

        role_str = player_roles.get(name, 'Contributor')
        preferred = role_to_preferred.get(role_str, 'bug_fix')

        max_consecutive = 0
        current = 0
        for t in tasks:
            if t != preferred and t != 'skip':
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        brs[name] = max_consecutive / len(tasks) if tasks else 0.0

    return brs


def compute_sue(sprint_history: list, player_roles: dict) -> float:
    """Skill Utilization Efficiency.

    SUE = mean skill-match score. 1.0 = preferred, 0.5 = adjacent, 0.0 = unrelated.
    """
    ADJACENCY = {
        'bug_fix': {'feature'},
        'feature': {'bug_fix', 'code_review'},
        'documentation': {'code_review'},
        'code_review': {'documentation', 'feature'},
    }

    role_to_preferred = {
        'Contributor': 'bug_fix',
        'Innovator': 'feature',
        'Knowledge Curator': 'documentation',
        'Maintainer': 'code_review',
    }

    scores = []
    for sprint in sprint_history:
        joint_action = sprint.get('joint_action', {})
        for name, task in joint_action.items():
            if not task or task.lower().startswith('skip'):
                continue
            # Parse task type
            if task.startswith('Fix:'):
                task_type = 'bug_fix'
            elif task.startswith('Feature:'):
                task_type = 'feature'
            elif task.startswith('Docs:') or task.startswith('Doc:'):
                task_type = 'documentation'
            elif task.startswith('Review:'):
                task_type = 'code_review'
            else:
                continue

            role_str = player_roles.get(name, 'Contributor')
            preferred = role_to_preferred.get(role_str, 'bug_fix')

            if task_type == preferred:
                scores.append(1.0)
            elif task_type in ADJACENCY.get(preferred, set()):
                scores.append(0.5)
            else:
                scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_chs(hi: float, mean_brs: float, sue: float, rq: float,
                weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
    """Community Health Score. CHS = w1*HI + w2*(1-mean_BRS) + w3*SUE + w4*RQ."""
    w1, w2, w3, w4 = weights
    return w1 * hi + w2 * (1.0 - mean_brs) + w3 * sue + w4 * rq


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Ensure valid distributions
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # KL(p||m) + KL(q||m), each clamped to avoid log(0)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def compute_belief_alignment(sprint_history: list) -> dict:
    """Compute Belief Alignment Index (BAI) per sprint.

    BAI = 1 - mean_pairwise_JSD over agents' health belief vectors.
    Range [0, 1]: 1 = perfect alignment, 0 = maximum divergence.

    Args:
        sprint_history: List of sprint dicts, each potentially containing
            'belief_snapshots' with per-agent belief state summaries.

    Returns:
        Dict with 'bai_per_sprint' (list of floats) and 'mean_bai' (float).
    """
    bai_per_sprint = []
    for sprint in sprint_history:
        snapshots = sprint.get('belief_snapshots', {})
        if len(snapshots) < 2:
            bai_per_sprint.append(1.0)
            continue

        # Extract health belief vectors
        health_beliefs = []
        for agent_state in snapshots.values():
            beliefs_health = agent_state.get('beliefs_health', {})
            if beliefs_health:
                vec = np.array(list(beliefs_health.values()), dtype=np.float64)
                health_beliefs.append(vec)

        if len(health_beliefs) < 2:
            bai_per_sprint.append(1.0)
            continue

        # Compute mean pairwise JSD
        n = len(health_beliefs)
        total_jsd = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_jsd += _jsd(health_beliefs[i], health_beliefs[j])
                count += 1
        mean_jsd = total_jsd / count if count > 0 else 0.0
        # JSD is bounded in [0, ln(2)] ≈ 0.693; normalize to [0, 1]
        normalized_jsd = mean_jsd / math.log(2)
        bai = max(0.0, 1.0 - normalized_jsd)
        bai_per_sprint.append(bai)

    mean_bai = sum(bai_per_sprint) / len(bai_per_sprint) if bai_per_sprint else 1.0
    return {
        'bai_per_sprint': bai_per_sprint,
        'mean_bai': mean_bai,
    }


DIALOGUE_ACT_CATEGORIES = [
    'propose_task',
    'request_help',
    'offer_mentoring',
    'express_concern',
    'free_ride_justify',
    'coordinate',
    'criticize',
    'encourage',
]

# Keyword patterns for rule-based dialogue act classification.
# Each category maps to a list of patterns (case-insensitive substring match).
_DIALOGUE_ACT_PATTERNS: dict[str, list[str]] = {
    'propose_task': [
        'should work on', 'i\'ll take', 'i will handle', 'let me do',
        'we need someone on', 'i\'ll focus on', 'plan to work on',
        'going to tackle', 'pick up the', 'volunteer for',
    ],
    'request_help': [
        'could use help', 'need assistance', 'anyone available',
        'struggling with', 'can someone', 'help me with',
        'not sure how to', 'would appreciate',
    ],
    'offer_mentoring': [
        'i can show you', 'let me help', 'i\'ll mentor', 'teach you',
        'walk you through', 'pair with', 'guide you', 'show you how',
    ],
    'express_concern': [
        'worried about', 'concerned that', 'risk of', 'falling behind',
        'burnout', 'unsustainable', 'declining', 'neglected',
        'no one is covering', 'project health',
    ],
    'free_ride_justify': [
        'prefer to stick with', 'best at', 'most productive when',
        'my expertise is in', 'not my area', 'someone else should',
        'i\'ll skip', 'rather focus on what i know',
    ],
    'coordinate': [
        'let\'s divide', 'coordinate', 'make sure we cover',
        'who wants to', 'split the work', 'balance', 'distribute',
        'between us', 'take turns', 'collectively',
    ],
    'criticize': [
        'not pulling weight', 'always picks', 'free riding',
        'shirking', 'should have', 'disappointed', 'unfair',
        'why didn\'t', 'never helps with',
    ],
    'encourage': [
        'great job', 'well done', 'appreciate', 'thank you',
        'keep it up', 'good work', 'proud of', 'nice effort',
        'team is doing', 'progress',
    ],
}


def classify_dialogue_acts(
    narrative_history: list[dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Classify dialogue acts from agent conversation logs.

    Uses rule-based keyword matching for fast, reproducible classification.
    For richer analysis, an LLM classifier could be substituted.

    Args:
        narrative_history: List of per-sprint dicts, each mapping
            agent_name -> {Pragmatic, Epistemic, Uncertainty, Strategy}.

    Returns:
        Dict with:
          'act_counts': {category: count} total across all sprints
          'per_sprint': list of {category: count} per sprint
          'per_agent': {agent: {category: count}}
          'coordination_index': fraction of utterances that are coordination acts
          'social_pressure_index': ratio of (criticize + express_concern) to
              (encourage + offer_mentoring)
    """
    total_counts: dict[str, int] = {cat: 0 for cat in DIALOGUE_ACT_CATEGORIES}
    per_sprint: list[dict[str, int]] = []
    per_agent: dict[str, dict[str, int]] = {}
    total_utterances = 0

    for sprint_reasoning in narrative_history:
        sprint_counts: dict[str, int] = {cat: 0 for cat in DIALOGUE_ACT_CATEGORIES}
        for agent_name, reasoning in sprint_reasoning.items():
            if agent_name not in per_agent:
                per_agent[agent_name] = {cat: 0 for cat in DIALOGUE_ACT_CATEGORIES}

            # Combine all reasoning text for this agent-sprint
            text = ' '.join(str(v) for v in reasoning.values()).lower()
            if not text.strip():
                continue
            total_utterances += 1

            for category, patterns in _DIALOGUE_ACT_PATTERNS.items():
                if any(p in text for p in patterns):
                    total_counts[category] += 1
                    sprint_counts[category] += 1
                    per_agent[agent_name][category] += 1

        per_sprint.append(sprint_counts)

    # Coordination Index: fraction of utterances with coordination acts
    coordination_index = (
        total_counts['coordinate'] / total_utterances
        if total_utterances > 0 else 0.0
    )

    # Social Pressure Index: negative pressure / positive support
    negative = total_counts['criticize'] + total_counts['express_concern']
    positive = total_counts['encourage'] + total_counts['offer_mentoring']
    social_pressure_index = negative / max(positive, 1)

    return {
        'act_counts': total_counts,
        'per_sprint': per_sprint,
        'per_agent': per_agent,
        'coordination_index': coordination_index,
        'social_pressure_index': social_pressure_index,
    }


def compute_trust_metrics(data: dict) -> dict:
    """Compute trust dynamics metrics from simulation results.

    Args:
        data: Results dict containing trust_final, trust_reciprocity,
            trust_centralization, and trust_history.

    Returns:
        Dict with trust_reciprocity, trust_centralization,
        mean_trust, trust_evolution (delta from initial to final).
    """
    trust_final = data.get('trust_final', {})
    reciprocity = data.get('trust_reciprocity', 1.0)
    centralization = data.get('trust_centralization', 0.0)
    trust_history = data.get('trust_history', [])

    # Mean trust across all pairs
    trust_values = []
    for row in trust_final.values():
        trust_values.extend(row.values())
    mean_trust = sum(trust_values) / len(trust_values) if trust_values else 0.0

    # Trust evolution: compare first and last snapshots
    trust_delta = 0.0
    if len(trust_history) >= 2:
        first = trust_history[0]
        last = trust_history[-1]
        first_vals = [v for row in first.values() for v in row.values()]
        last_vals = [v for row in last.values() for v in row.values()]
        if first_vals and last_vals:
            trust_delta = (
                sum(last_vals) / len(last_vals)
                - sum(first_vals) / len(first_vals)
            )

    return {
        'trust_reciprocity': reciprocity,
        'trust_centralization': centralization,
        'mean_trust': mean_trust,
        'trust_delta': trust_delta,
    }


def detect_coalitions(sprint_history: list) -> dict:
    """Detect coalitions via belief clustering per sprint.

    Groups agents by Jensen-Shannon divergence of their health beliefs.
    Uses simple agglomerative clustering: agents within JSD < threshold
    form a coalition.

    Args:
        sprint_history: List of sprint dicts with 'belief_snapshots'.

    Returns:
        Dict with:
          'coalitions_per_sprint': list of list-of-lists (agent name groups)
          'stable_coalitions': coalitions that persist across >50% of sprints
          'num_coalitions_per_sprint': list of int
    """
    threshold = 0.15  # JSD threshold for same coalition
    coalitions_per_sprint: list[list[list[str]]] = []

    for sprint in sprint_history:
        snapshots = sprint.get('belief_snapshots', {})
        if len(snapshots) < 2:
            coalitions_per_sprint.append([list(snapshots.keys())])
            continue

        # Extract health belief vectors
        agents = []
        beliefs = []
        for name, state in snapshots.items():
            bh = state.get('beliefs_health', {})
            if bh:
                agents.append(name)
                beliefs.append(np.array(list(bh.values()), dtype=np.float64))

        if len(agents) < 2:
            coalitions_per_sprint.append([agents])
            continue

        # Simple agglomerative clustering
        n = len(agents)
        # Compute pairwise JSD matrix
        jsd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _jsd(beliefs[i], beliefs[j])
                jsd_matrix[i, j] = d
                jsd_matrix[j, i] = d

        # Greedy single-linkage: if JSD(i,j) < threshold, same cluster
        cluster_ids = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if jsd_matrix[i, j] < threshold:
                    # Merge clusters
                    old_id = cluster_ids[j]
                    new_id = cluster_ids[i]
                    for k in range(n):
                        if cluster_ids[k] == old_id:
                            cluster_ids[k] = new_id

        # Group agents by cluster
        clusters: dict[int, list[str]] = {}
        for idx, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(agents[idx])
        coalitions_per_sprint.append(list(clusters.values()))

    # Find stable coalitions (appear in >50% of sprints)
    # Represent coalitions as frozensets and count
    coalition_counts: dict[frozenset, int] = {}
    for sprint_coalitions in coalitions_per_sprint:
        for group in sprint_coalitions:
            if len(group) >= 2:  # Solo agents aren't coalitions
                key = frozenset(group)
                coalition_counts[key] = coalition_counts.get(key, 0) + 1

    num_sprints = len(coalitions_per_sprint)
    stable = [
        sorted(list(members))
        for members, count in coalition_counts.items()
        if count > num_sprints * 0.5
    ]

    return {
        'coalitions_per_sprint': [
            [sorted(g) for g in sprint]
            for sprint in coalitions_per_sprint
        ],
        'stable_coalitions': stable,
        'num_coalitions_per_sprint': [
            len(sprint) for sprint in coalitions_per_sprint
        ],
    }


def compute_burnout_cascade_metrics(data: dict) -> dict:
    """Extract burnout cascade metrics from simulation results."""
    return {
        'cascade_count': data.get('burnout_cascade_count', 0),
        'mean_burnout': data.get('mean_burnout', 0.0),
        'peak_burnout': data.get('peak_burnout', 0.0),
    }


def compute_norm_metrics(data: dict) -> dict:
    """Extract norm emergence metrics from simulation results."""
    return {
        'norm_emergence_rate': data.get('norm_emergence_rate', 0.0),
        'norm_stability_index': data.get('norm_stability_index', 1.0),
        'norm_compliance_rate': data.get('norm_compliance_rate', 1.0),
        'active_norms': data.get('active_norms', {}),
    }


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

    brs = compute_brs(sprint_history, player_roles)
    mean_brs = sum(brs.values()) / len(brs) if brs else 0.0
    sue = compute_sue(sprint_history, player_roles)
    chs = compute_chs(hi, mean_brs, sue, rq)

    # Belief Alignment Index (requires AIF loop to be closed)
    bai_result = compute_belief_alignment(sprint_history)

    # Dialogue act analysis
    narrative_history = data.get('narrative_history', [])
    dialogue_result = classify_dialogue_acts(narrative_history)

    # Trust dynamics
    trust_result = compute_trust_metrics(data)

    # Norm emergence
    norm_result = compute_norm_metrics(data)

    # Burnout cascades
    burnout_cascade_result = compute_burnout_cascade_metrics(data)

    # Coalition detection
    coalition_result = detect_coalitions(sprint_history)

    return {
        'sustain_score': sustain_score,
        'harmony_index': hi,
        'resilience_quotient': rq,
        'fairness': fairness,
        'strategy_diversity': strategy_div,
        'stress_validity': stress_validity,
        'brs_per_agent': brs,
        'mean_brs': mean_brs,
        'sue': sue,
        'chs': chs,
        'belief_alignment': bai_result,
        'mean_bai': bai_result['mean_bai'],
        'dialogue_acts': dialogue_result,
        'coordination_index': dialogue_result['coordination_index'],
        'social_pressure_index': dialogue_result['social_pressure_index'],
        'trust_metrics': trust_result,
        'trust_reciprocity': trust_result['trust_reciprocity'],
        'mean_trust': trust_result['mean_trust'],
        'norm_metrics': norm_result,
        'norm_emergence_rate': norm_result['norm_emergence_rate'],
        'norm_compliance_rate': norm_result['norm_compliance_rate'],
        'burnout_cascade': burnout_cascade_result,
        'burnout_cascade_count': burnout_cascade_result['cascade_count'],
        'mean_burnout': burnout_cascade_result['mean_burnout'],
        'coalitions': coalition_result,
        'num_stable_coalitions': len(coalition_result['stable_coalitions']),
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
    if FLAGS.vllm_url:
        cmd.append(f'--vllm_url={FLAGS.vllm_url}')
    if FLAGS.model_name:
        cmd.append(f'--model_name={FLAGS.model_name}')
    if FLAGS.use_mock:
        cmd.append('--use_mock')

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
            'strategy_diversity': 0.0, 'mean_brs': 0.0,
            'sue': 0.0, 'chs': 0.0,
            'duration': duration, 'status': 'crash',
        }

    results_path = os.path.join(out_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f'Run {run_id}: no results.json found', file=sys.stderr)
        return {
            'sustain_score': 0.0, 'harmony_index': 0.0,
            'resilience_quotient': 0.0, 'fairness': 0.0,
            'strategy_diversity': 0.0, 'mean_brs': 0.0,
            'sue': 0.0, 'chs': 0.0,
            'duration': duration, 'status': 'crash',
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
              f'BRS={r["mean_brs"]:.4f}  SUE={r["sue"]:.4f}  CHS={r["chs"]:.4f}  '
              f'BAI={r.get("mean_bai", 0):.4f}  '
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
    avg_brs = avg('mean_brs')
    avg_sue = avg('sue')
    avg_chs = avg('chs')
    avg_bai = avg('mean_bai')
    total_duration = sum(r['duration'] for r in results)

    print(f'\n{"="*60}')
    print(f'RESULT (avg of {len(ok_results)}/{FLAGS.runs} successful runs)')
    print(f'{"="*60}')
    print(f'  SustainScore:        {avg_ss:.4f}')
    print(f'  Harmony Index:       {avg_hi:.4f}')
    print(f'  Resilience Quotient: {avg_rq:.4f}')
    print(f'  Fairness:            {avg_fair:.4f}')
    print(f'  Strategy Diversity:  {avg_div:.4f}')
    print(f'  Mean BRS:            {avg_brs:.4f}')
    print(f'  SUE:                 {avg_sue:.4f}')
    print(f'  CHS:                 {avg_chs:.4f}')
    print(f'  Belief Alignment:    {avg_bai:.4f}')
    print(f'  Total time:          {total_duration:.0f}s')

    if FLAGS.log:
        commit = get_git_hash()
        desc = FLAGS.description or 'no description'
        tsv_path = os.path.join(os.path.dirname(__file__), 'results.tsv')
        if not os.path.exists(tsv_path):
            with open(tsv_path, 'w') as f:
                f.write('commit\tsustain_score\tharmony_index\tresilience_quotient'
                        '\tfairness\tstrategy_div\tmean_brs\tsue\tchs'
                        '\tstatus\tdescription\n')
        with open(tsv_path, 'a') as f:
            status = 'ok' if ok_results else 'crash'
            f.write(f'{commit}\t{avg_ss:.4f}\t{avg_hi:.4f}\t{avg_rq:.4f}'
                    f'\t{avg_fair:.4f}\t{avg_div:.4f}\t{avg_brs:.4f}'
                    f'\t{avg_sue:.4f}\t{avg_chs:.4f}\t{status}\t{desc}\n')
        print(f'\nLogged to {tsv_path}')


if __name__ == '__main__':
    app.run(main)
