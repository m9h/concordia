"""Metric bridge: compute Concordia SustainHub metrics from LLAMOSC output.

This module maps LLAMOSC simulation results into the Concordia metric
framework so that the two systems can be compared on the same scale.

The authoritative Concordia formulas live in:
  - examples/games/sustain_hub/simulation.py  (harmony_index, resilience_quotient)
  - examples/games/sustain_hub/evaluate.py    (compute_fairness, compute_sustain_score)
  - examples/games/sustain_hub/social_data.py (reward constants)

Every formula below is replicated *exactly* from those files with comments
indicating which source it mirrors.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Concordia reward constants (social_data.py)
# ---------------------------------------------------------------------------
REWARD_PREFERRED_SUCCESS = 3.0
REWARD_PREFERRED_FAILURE = -1.0
REWARD_NONPREFERRED_SUCCESS = 1.0
REWARD_NONPREFERRED_FAILURE = -1.0
REWARD_SKIP = 0.0

# Harmony Index weighting (simulation.py default)
ALPHA = 0.6

# Concordia task types (social_data.py)
TASK_TYPES = ["bug_fix", "feature", "documentation", "code_review"]


# ============================================================================
# Concept Mapping
# ============================================================================

def map_code_quality_to_reward(code_quality: int | float) -> float:
    """Map LLAMOSC code_quality (1-5) to Concordia reward scale [-1, 3].

    Concordia uses:
        REWARD_PREFERRED_SUCCESS = 3.0  (best outcome)
        REWARD_FAILURE           = -1.0 (worst outcome)

    We linearly interpolate:
        code_quality 1 -> -1.0
        code_quality 5 ->  3.0
    """
    # Linear: reward = -1 + (code_quality - 1) * (3 - (-1)) / (5 - 1)
    #                = -1 + (code_quality - 1) * 1.0
    return -1.0 + (code_quality - 1) * (REWARD_PREFERRED_SUCCESS - REWARD_PREFERRED_FAILURE) / (5 - 1)


def map_difficulty_to_task_type(difficulty: int | float) -> str:
    """Map LLAMOSC issue_difficulty (1-5) to a Concordia task type.

    Mapping (designed to spread across all four Concordia task types):
        difficulty 1     -> "documentation"
        difficulty 2     -> "bug_fix"
        difficulty 3     -> "feature"
        difficulty 4, 5  -> "code_review"
    """
    if difficulty <= 1:
        return "documentation"
    elif difficulty <= 2:
        return "bug_fix"
    elif difficulty <= 3:
        return "feature"
    else:
        return "code_review"


def group_into_sprints(
    results: list[dict[str, Any]],
    issues_per_sprint: int = 1,
) -> list[list[dict[str, Any]]]:
    """Group consecutive LLAMOSC timesteps into sprints.

    Each sprint contains *issues_per_sprint* consecutive timestep results.
    The last sprint may contain fewer if the total is not evenly divisible.
    """
    sprints: list[list[dict[str, Any]]] = []
    for i in range(0, len(results), issues_per_sprint):
        sprints.append(results[i : i + issues_per_sprint])
    return sprints


# ============================================================================
# Gini coefficient helpers
# ============================================================================

def gini_coefficient_simulation(counts: list[int | float]) -> float:
    """Gini coefficient as implemented in simulation.py harmony_index().

    Uses the absolute-difference formula:
        gini = sum(|x_i - x_j| for all i,j) / (2 * n * sum(x))
    """
    counts = sorted(counts)
    n = len(counts)
    if n == 0 or sum(counts) == 0:
        return 0.0
    numerator = sum(
        abs(counts[i] - counts[j])
        for i in range(n)
        for j in range(n)
    )
    denominator = 2 * n * sum(counts)
    return numerator / denominator if denominator > 0 else 0.0


def gini_coefficient_evaluate(values: list[float]) -> float:
    """Gini coefficient as implemented in evaluate.py compute_fairness().

    Uses the cumulative-sum formula with non-negative shift:
        shifted = [v - min_val for v in sorted(values)]
        gini = (2 * sum_cumulative) / (n * total) - (n + 1) / n
    Returns the raw gini, caller computes 1 - gini for fairness.
    """
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    if n <= 1:
        return 0.0
    min_val = min(values)
    shifted = [v - min_val for v in values]
    total = sum(shifted)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for v in shifted:
        cumulative += v
        gini_sum += cumulative
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    return max(0.0, gini)


# ============================================================================
# Harmony Index  (mirrors simulation.py harmony_index())
# ============================================================================

def compute_harmony_index(
    task_counts: dict[str, int],
    cumulative_scores: dict[str, float],
    alpha: float = ALPHA,
) -> float:
    """HI = alpha * avg_success + (1 - alpha) * fairness.

    Exactly replicates simulation.py lines 381-406.

    Args:
        task_counts: number of tasks completed per agent.
        cumulative_scores: sum of Concordia-scale rewards per agent.
        alpha: weighting factor (default 0.6).

    Returns:
        The Harmony Index in [0, 1] (may exceed 1 if scores are very high
        relative to task counts).
    """
    if not task_counts or all(v == 0 for v in task_counts.values()):
        return 0.0

    total_tasks = sum(task_counts.values())
    total_score = sum(cumulative_scores.values())
    max_possible = total_tasks * REWARD_PREFERRED_SUCCESS
    avg_success = total_score / max_possible if max_possible > 0 else 0.0

    counts = sorted(task_counts.values())
    gini = gini_coefficient_simulation(counts)
    fairness = 1.0 - gini

    return alpha * avg_success + (1.0 - alpha) * fairness


def compute_harmony_index_from_results(
    results: list[dict[str, Any]],
    agents: list[str] | None = None,
    alpha: float = ALPHA,
) -> float:
    """Compute HI directly from a list of LLAMOSC timestep results.

    Builds task_counts and cumulative_scores by mapping each accepted PR's
    code_quality to the Concordia reward scale.

    If *agents* is provided, only those agents are counted (used for
    stress-test dropout simulation).
    """
    task_counts: dict[str, int] = defaultdict(int)
    cumulative_scores: dict[str, float] = defaultdict(float)

    for step in results:
        contributor = step["selected_contributor"]
        if agents is not None and contributor not in agents:
            continue
        if step.get("pr_accepted"):
            task_counts[contributor] += 1
            reward = map_code_quality_to_reward(step["code_quality"])
            cumulative_scores[contributor] += reward

    # Ensure all requested agents appear in task_counts (even with 0)
    if agents is not None:
        for a in agents:
            task_counts.setdefault(a, 0)

    return compute_harmony_index(dict(task_counts), dict(cumulative_scores), alpha)


# ============================================================================
# Resilience Quotient  (mirrors simulation.py resilience_quotient property)
# ============================================================================

def compute_resilience_quotient(
    results: list[dict[str, Any]],
    agents: list[str],
    issues_per_sprint: int = 1,
    stress_sprint_fraction: float = 0.4,
    dropout_fraction: float = 0.2,
    alpha: float = ALPHA,
) -> float:
    """Compute RQ by simulating contributor dropout stress.

    Exactly mirrors simulation.py lines 338-379:
        RQ = mean(HI during/after stress) / mean(HI before stress)

    Stress is injected at the sprint located at *stress_sprint_fraction* of
    the simulation.  The stress consists of removing *dropout_fraction* of
    agents from the post-stress HI calculations (simulating contributor
    dropout).

    Returns 1.0 if there is insufficient data for a meaningful calculation.
    """
    sprints = group_into_sprints(results, issues_per_sprint)
    if len(sprints) < 2:
        return 1.0

    # Determine the stress sprint index (0-indexed)
    stress_idx = max(1, int(len(sprints) * stress_sprint_fraction))
    if stress_idx >= len(sprints):
        return 1.0

    # Determine which agents "drop out" during stress
    n_dropout = max(1, int(len(agents) * dropout_fraction))
    # Drop the last N agents alphabetically (deterministic)
    sorted_agents = sorted(agents)
    dropout_agents = set(sorted_agents[-n_dropout:])
    remaining_agents = [a for a in sorted_agents if a not in dropout_agents]

    # Pre-stress HI: compute per-sprint HI using all agents
    pre_stress_his: list[float] = []
    cumulative_results: list[dict[str, Any]] = []
    for sprint_results in sprints[:stress_idx]:
        cumulative_results.extend(sprint_results)
        hi = compute_harmony_index_from_results(cumulative_results, agents=None, alpha=alpha)
        pre_stress_his.append(hi)

    # Stress + post-stress HI: compute per-sprint HI excluding dropout agents
    stress_his: list[float] = []
    cumulative_results_post: list[dict[str, Any]] = []
    for sprint_results in sprints[stress_idx:]:
        cumulative_results_post.extend(sprint_results)
        hi = compute_harmony_index_from_results(
            cumulative_results_post, agents=remaining_agents, alpha=alpha
        )
        stress_his.append(hi)

    if not pre_stress_his or not stress_his:
        return 1.0

    hi_before = sum(pre_stress_his) / len(pre_stress_his)
    hi_during = sum(stress_his) / len(stress_his)

    if hi_before <= 0:
        return 0.0

    return round(hi_during / hi_before, 4)


# ============================================================================
# Fairness  (mirrors evaluate.py compute_fairness())
# ============================================================================

def compute_fairness(scores: dict[str, float]) -> float:
    """1 - Gini coefficient of cumulative scores per agent.

    Exactly replicates evaluate.py lines 49-69.
    Uses the cumulative-sum Gini variant with non-negative shifting.
    """
    if not scores:
        return 1.0
    values = sorted(scores.values())
    n = len(values)
    if n <= 1:
        return 1.0
    min_val = min(values)
    shifted = [v - min_val for v in values]
    total = sum(shifted)
    if total == 0:
        return 1.0  # all equal -> perfectly fair
    cumulative = 0.0
    gini_sum = 0.0
    for v in shifted:
        cumulative += v
        gini_sum += cumulative
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    return max(0.0, 1.0 - gini)


# ============================================================================
# Strategy Diversity  (mirrors evaluate.py compute_strategy_diversity())
# ============================================================================

def compute_strategy_diversity(
    results: list[dict[str, Any]],
    issues_per_sprint: int = 1,
) -> float:
    """Fraction of agents who worked on different difficulty levels across sprints.

    Mirrors evaluate.py lines 72-90 but adapted for LLAMOSC:
    instead of tracking Concordia task-type strings across sprints, we track
    the *mapped task type* (via difficulty) that each agent worked on.
    An agent counts as a "changer" if they worked on at least two distinct
    task types across all sprints.
    """
    sprints = group_into_sprints(results, issues_per_sprint)
    if len(sprints) < 2:
        return 1.0  # matches Concordia: can't measure with 1 sprint

    # Collect per-agent task types across sprints
    agent_tasks: dict[str, list[str]] = defaultdict(list)
    for sprint_results in sprints:
        for step in sprint_results:
            if step.get("pr_accepted"):
                contributor = step["selected_contributor"]
                task_type = map_difficulty_to_task_type(step["issue_difficulty"])
                agent_tasks[contributor].append(task_type)

    if not agent_tasks:
        return 0.0

    changers = 0
    for _agent, tasks in agent_tasks.items():
        if len(tasks) >= 2 and len(set(tasks)) > 1:
            changers += 1

    return changers / len(agent_tasks)


# ============================================================================
# SustainScore  (mirrors evaluate.py compute_sustain_score())
# ============================================================================

def compute_sustain_score(
    hi: float,
    rq: float,
    fairness: float,
    strategy_diversity: float,
    has_stress: bool = True,
) -> float:
    """SustainScore = HI * (1 + RQ) * Fairness * StrategyDiversity * StressValidity.

    Exactly replicates evaluate.py line 108.

    StressValidity is 1.0 if stress events are present, 0.5 otherwise
    (evaluate.py line 106).
    """
    stress_validity = 1.0 if has_stress else 0.5
    return hi * (1.0 + rq) * fairness * strategy_diversity * stress_validity


# ============================================================================
# GSoC 2025 proposed metrics
# ============================================================================

def compute_brs(
    results: list[dict[str, Any]],
    agent_experience: dict[str, float],
) -> dict[str, float]:
    """Burnout Risk Score per agent.

    BRS = (consecutive non-preferred assignments) / (total assignments).

    A "non-preferred" assignment is one where the mapped task type does NOT
    match the agent's best-fit type based on experience level:
        experience 1-2 -> preferred "documentation" (low-skill)
        experience 2-3 -> preferred "bug_fix"
        experience 3-4 -> preferred "feature"
        experience 4-5 -> preferred "code_review" (high-skill)
    """
    def preferred_type(exp: float) -> str:
        if exp <= 2:
            return "documentation"
        elif exp <= 3:
            return "bug_fix"
        elif exp <= 4:
            return "feature"
        else:
            return "code_review"

    # Build per-agent assignment sequence
    agent_assignments: dict[str, list[str]] = defaultdict(list)
    for step in results:
        if step.get("pr_accepted"):
            contributor = step["selected_contributor"]
            task_type = map_difficulty_to_task_type(step["issue_difficulty"])
            agent_assignments[contributor].append(task_type)

    brs_scores: dict[str, float] = {}
    for agent, exp in agent_experience.items():
        pref = preferred_type(exp)
        assignments = agent_assignments.get(agent, [])
        if not assignments:
            brs_scores[agent] = 0.0
            continue

        # Count longest consecutive run of non-preferred assignments
        max_consecutive = 0
        current_consecutive = 0
        for task in assignments:
            if task != pref:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        brs_scores[agent] = max_consecutive / len(assignments) if assignments else 0.0

    return brs_scores


def compute_sue(
    results: list[dict[str, Any]],
    agent_experience: dict[str, float],
) -> float:
    """Skill Utilization Efficiency.

    SUE = mean skill-match score across all assignments.
        1.0 if difficulty matches experience level
        0.5 if adjacent (off by one tier)
        0.0 otherwise

    Experience tiers:
        1-2 -> tier 1  (matches difficulty 1-2)
        2-3 -> tier 2  (matches difficulty 2-3)
        3-4 -> tier 3  (matches difficulty 3-4)
        4-5 -> tier 4  (matches difficulty 4-5)
    """
    def exp_tier(exp: float) -> int:
        if exp <= 2:
            return 1
        elif exp <= 3:
            return 2
        elif exp <= 4:
            return 3
        else:
            return 4

    def diff_tier(diff: float) -> int:
        if diff <= 2:
            return 1
        elif diff <= 3:
            return 2
        elif diff <= 4:
            return 3
        else:
            return 4

    scores: list[float] = []
    for step in results:
        if not step.get("pr_accepted"):
            continue
        contributor = step["selected_contributor"]
        exp = agent_experience.get(contributor, 1.0)
        e_tier = exp_tier(exp)
        d_tier = diff_tier(step["issue_difficulty"])
        gap = abs(e_tier - d_tier)
        if gap == 0:
            scores.append(1.0)
        elif gap == 1:
            scores.append(0.5)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_chs(
    hi: float,
    mean_brs: float,
    sue: float,
    rq: float,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """Community Health Score.

    CHS = w1*HI + w2*(1 - mean_BRS) + w3*SUE + w4*RQ
    """
    w1, w2, w3, w4 = weights
    return w1 * hi + w2 * (1.0 - mean_brs) + w3 * sue + w4 * rq


# ============================================================================
# Main bridge: LLAMOSC JSON -> Concordia-compatible metrics dict
# ============================================================================

def compute_all_metrics(
    llamosc_data: dict[str, Any],
    issues_per_sprint: int = 1,
    stress_sprint_fraction: float = 0.4,
    dropout_fraction: float = 0.2,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Compute all Concordia + GSoC metrics from raw LLAMOSC output.

    Args:
        llamosc_data: parsed JSON from llamosc_results.json.
        issues_per_sprint: how many LLAMOSC issues constitute one Concordia sprint.
        stress_sprint_fraction: at what fraction of the sim to inject stress.
        dropout_fraction: fraction of agents to "drop out" during stress.
        alpha: HI weighting (default 0.6).

    Returns:
        A dict matching the vidhi_baseline_results.json schema, suitable for
        direct comparison with the Concordia experiment ladder.
    """
    results = llamosc_data["results"]
    agent_state = llamosc_data.get("agent_final_state", {})
    config = llamosc_data.get("config", {})

    agents = list(agent_state.keys())
    agent_experience = {a: s["experience"] for a, s in agent_state.items()}

    # --- Core Concordia metrics ---
    # Per-agent cumulative scores (Concordia-scale)
    cumulative_scores: dict[str, float] = defaultdict(float)
    task_counts: dict[str, int] = defaultdict(int)
    for step in results:
        if step.get("pr_accepted"):
            contributor = step["selected_contributor"]
            task_counts[contributor] += 1
            cumulative_scores[contributor] += map_code_quality_to_reward(step["code_quality"])

    # Ensure all agents present
    for a in agents:
        task_counts.setdefault(a, 0)
        cumulative_scores.setdefault(a, 0.0)

    # HI per sprint (for trajectory)
    sprints = group_into_sprints(results, issues_per_sprint)
    hi_trajectory: list[float] = []
    cumulative_tc: dict[str, int] = defaultdict(int)
    cumulative_cs: dict[str, float] = defaultdict(float)
    for a in agents:
        cumulative_tc[a] = 0
        cumulative_cs[a] = 0.0
    for sprint_results in sprints:
        for step in sprint_results:
            if step.get("pr_accepted"):
                c = step["selected_contributor"]
                cumulative_tc[c] += 1
                cumulative_cs[c] += map_code_quality_to_reward(step["code_quality"])
        hi = compute_harmony_index(dict(cumulative_tc), dict(cumulative_cs), alpha)
        hi_trajectory.append(hi)

    final_hi = hi_trajectory[-1] if hi_trajectory else 0.0
    mean_hi = sum(hi_trajectory) / len(hi_trajectory) if hi_trajectory else 0.0

    # RQ
    rq = compute_resilience_quotient(
        results, agents, issues_per_sprint, stress_sprint_fraction, dropout_fraction, alpha
    )

    # Fairness (evaluate.py style — on cumulative scores)
    fairness = compute_fairness(dict(cumulative_scores))

    # Strategy diversity
    strategy_div = compute_strategy_diversity(results, issues_per_sprint)

    # SustainScore
    has_stress = True  # we always inject synthetic stress for RQ
    sustain_score = compute_sustain_score(mean_hi, rq, fairness, strategy_div, has_stress)

    # Task coverage (how many of the 4 Concordia task types were attempted)
    attempted_types: set[str] = set()
    for step in results:
        if step.get("pr_accepted"):
            attempted_types.add(map_difficulty_to_task_type(step["issue_difficulty"]))
    coverage = len(attempted_types) / len(TASK_TYPES)

    # Diversity = fraction of task types with at least one completed task
    diversity = coverage  # same measure in single-sprint context

    # --- GSoC 2025 metrics ---
    brs_scores = compute_brs(results, agent_experience)
    mean_brs = sum(brs_scores.values()) / len(brs_scores) if brs_scores else 0.0
    sue = compute_sue(results, agent_experience)
    chs = compute_chs(mean_hi, mean_brs, sue, rq)

    # --- Build sprint_history (matching vidhi_baseline_results.json format) ---
    sprint_history: list[dict[str, Any]] = []
    for idx, sprint_results in enumerate(sprints):
        actions: dict[str, str] = {}
        rewards: dict[str, float] = {}
        for step in sprint_results:
            contributor = step["selected_contributor"]
            task_type = map_difficulty_to_task_type(step["issue_difficulty"])
            actions[contributor] = task_type
            if step.get("pr_accepted"):
                rewards[contributor] = map_code_quality_to_reward(step["code_quality"])
            else:
                rewards[contributor] = REWARD_PREFERRED_FAILURE

        sprint_entry = {
            "sprint": idx,
            "actions": actions,
            "rewards": rewards,
            "harmony_index": hi_trajectory[idx] if idx < len(hi_trajectory) else 0.0,
            "diversity": len(set(actions.values())) / len(TASK_TYPES) if actions else 0.0,
            "coverage": len(set(actions.values())) / len(TASK_TYPES) if actions else 0.0,
        }
        sprint_history.append(sprint_entry)

    # --- Assemble output matching vidhi_baseline_results.json schema ---
    output = {
        "level": "llamosc",
        "level_name": "LLAMOSC Baseline",
        "description": (
            "LLAMOSC agent-based simulation mapped to Concordia metric space. "
            f"Algorithm: {config.get('algorithm', 'unknown')}, "
            f"test_mode: {config.get('test_mode', False)}, "
            f"seed: {config.get('seed', None)}."
        ),
        "new_concept": "LLM-agent task allocation",
        "rl_analog": "LLAMOSC simulation",
        "aif_mechanism": "N/A",
        "num_sprints": len(sprints),
        "duration_s": 0.0,  # filled in by caller if timing
        "mean_hi": round(mean_hi, 6),
        "final_hi": round(final_hi, 6),
        "resilience_quotient": round(rq, 6),
        "mean_coverage": round(coverage, 4),
        "mean_diversity": round(diversity, 4),
        "strategy_diversity": round(strategy_div, 4),
        "hi_trajectory": [round(h, 6) for h in hi_trajectory],
        "sprint_history": sprint_history,
        # SustainScore components
        "sustain_score": round(sustain_score, 6),
        "fairness": round(fairness, 6),
        "stress_validity": 1.0,
        # GSoC 2025 metrics
        "gsoc_metrics": {
            "brs_per_agent": {a: round(v, 4) for a, v in brs_scores.items()},
            "mean_brs": round(mean_brs, 4),
            "sue": round(sue, 4),
            "chs": round(chs, 4),
        },
        # LLAMOSC-native metrics (for reference)
        "llamosc_native": {
            "solve_rate": llamosc_data.get("final_metrics", {}).get("solve_rate", 0.0),
            "avg_code_quality": llamosc_data.get("final_metrics", {}).get("avg_code_quality", 0.0),
            "total_issues": config.get("n_issues", 0),
            "n_contributors": config.get("n_contributors", 0),
        },
    }

    return output


def load_and_compute(
    llamosc_path: str,
    issues_per_sprint: int = 1,
    stress_sprint_fraction: float = 0.4,
    dropout_fraction: float = 0.2,
) -> dict[str, Any]:
    """Load a LLAMOSC results JSON file and compute all metrics."""
    with open(llamosc_path) as f:
        data = json.load(f)
    return compute_all_metrics(
        data,
        issues_per_sprint=issues_per_sprint,
        stress_sprint_fraction=stress_sprint_fraction,
        dropout_fraction=dropout_fraction,
    )
