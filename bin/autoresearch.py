#!/usr/bin/env python3
"""Karpathy-style autoresearch loop for SustainHub optimization.

Autonomously iterates over simulation parameter variations, commits each
hypothesis, runs a Tier 1/2/3 experiment, and keeps or reverts based on
SustainScore improvement.

Usage:
  # Mock mode (no LLM, fast testing of the loop itself):
  uv run python bin/autoresearch.py --tier=1

  # Against a local vLLM server:
  uv run python bin/autoresearch.py --tier=1 --vllm_url=http://localhost:8000/v1

  # Dry run (show variations without executing):
  uv run python bin/autoresearch.py --dry_run

  # Resume from a previous run:
  uv run python bin/autoresearch.py --tier=1 --results_file=results.tsv

  # Multi-fidelity validation (promote T1 winners to T2):
  uv run python bin/autoresearch.py --tier=1 --validate --min_improvement=0.01

  # Best-of-N screening with dashboard:
  uv run python bin/autoresearch.py --tier=1 --best_of=3 --dashboard

  # Multi-seed robustness:
  uv run python bin/autoresearch.py --tier=1 --seeds=3
"""

import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('vllm_url', None,
                    'vLLM API base URL. If omitted, --use_mock is passed.')
flags.DEFINE_string('model_name', 'Qwen/Qwen2.5-7B-Instruct',
                    'Model name for LLM-backed runs.')
flags.DEFINE_integer('tier', 1,
                     'Experiment tier: 1=fast (4 agents, 1 sprint), '
                     '2=medium (3 averaged runs), 3=full (16 agents, 5 sprints).')
flags.DEFINE_integer('max_iterations', 50,
                     'Maximum number of hypothesis iterations.')
flags.DEFINE_string('results_file', 'results.tsv',
                    'Path to the tab-separated results log.')
flags.DEFINE_bool('dry_run', False,
                  'Show what each variation would change without executing.')
flags.DEFINE_integer('start_layer', 1,
                     'Which variation layer to start from (1-5).')
flags.DEFINE_bool('skip_baseline', False,
                  'Skip initial baseline measurement.')
flags.DEFINE_bool('validate', False,
                  'Enable multi-fidelity validation: promote Tier 1 winners '
                  'to Tier 2 before keeping.')
flags.DEFINE_integer('seeds', 1,
                     'Number of seeds per experiment. When >1, uses the mean '
                     'score across seeds for keep/revert decisions.')
flags.DEFINE_float('min_improvement', 0.0,
                   'Minimum SustainScore improvement required to keep a '
                   'variation. Prevents keeping noise-level changes.')
flags.DEFINE_bool('dashboard', False,
                  'Print a compact progress dashboard after each iteration.')
flags.DEFINE_integer('best_of', 1,
                     'Run Tier 1 N times and use median score. Reduces '
                     'single-run noise without full multi-seed overhead.')

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUSTAIN_DIR = os.path.join(REPO_ROOT, 'examples', 'games', 'sustain_hub')
SOCIAL_DATA_PATH = os.path.join(SUSTAIN_DIR, 'social_data.py')
SIMULATION_PATH = os.path.join(SUSTAIN_DIR, 'simulation.py')
SCENARIO_CONFIG_PATH = os.path.join(SUSTAIN_DIR, 'scenario_config.py')
TOOLS_PATH = os.path.join(SUSTAIN_DIR, 'tools.py')

TIER_CONFIGS = {
    1: {'num_sprints': 1, 'community_size': 4, 'runs': 1},
    2: {'num_sprints': 3, 'community_size': 8, 'runs': 3},
    3: {'num_sprints': 5, 'community_size': 16, 'runs': 1},
}

# Module-level governance mode, set by Layer 5 variations.
# Used by _run_single_seed to pass --governance=<mode> to the run command.
_governance_mode = 'free_choice'


# ===================================================================
# File helpers
# ===================================================================

def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, 'w') as f:
        f.write(content)


def replace_in_file(path: str, old: str, new: str) -> bool:
    """Replace *old* with *new* in *path*. Returns True if a substitution was made."""
    content = read_file(path)
    if old not in content:
        return False
    write_file(path, content.replace(old, new))
    return True


def replace_block(path: str, pattern: str, replacement: str) -> bool:
    """Regex-based block replacement. Returns True if a substitution was made."""
    content = read_file(path)
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count == 0:
        return False
    write_file(path, new_content)
    return True


# ===================================================================
# Git helpers
# ===================================================================

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git'] + list(args),
        capture_output=True, text=True, cwd=REPO_ROOT, check=check,
    )


def git_is_clean() -> bool:
    r = git('status', '--porcelain')
    return r.stdout.strip() == ''


def git_commit(message: str) -> str:
    """Stage in-scope files, commit, return short hash."""
    for p in [SOCIAL_DATA_PATH, SIMULATION_PATH, SCENARIO_CONFIG_PATH, TOOLS_PATH]:
        git('add', p, check=False)
    git('commit', '-m', message, check=True)
    r = git('rev-parse', '--short=7', 'HEAD')
    return r.stdout.strip()


def git_revert() -> None:
    """Revert the last commit (hard reset HEAD~1)."""
    git('reset', '--hard', 'HEAD~1')


def git_short_hash() -> str:
    r = git('rev-parse', '--short=7', 'HEAD')
    return r.stdout.strip()


# ===================================================================
# Experiment runner
# ===================================================================

def run_experiment(
    tier: int,
    seeds: int = 1,
    seed_log_path: Optional[str] = None,
) -> dict:
    """Run one or more simulations and return averaged metrics.

    Args:
        tier: Experiment tier (1, 2, or 3).
        seeds: Number of independent seeds. Each seed runs the full tier
            config (including multi-run averaging for Tier 2). The final
            result averages across seeds.
        seed_log_path: If provided, append per-seed results as JSONL.

    Returns a dict with keys: sustain_score, harmony_index,
    resilience_quotient, fairness, strategy_diversity, stress_validity,
    duration, status.
    """
    seed_results = []
    for seed_idx in range(seeds):
        seed_label = 'seed%d' % seed_idx if seeds > 1 else ''
        result = _run_single_seed(tier, seed_idx, seed_label)
        seed_results.append(result)
        if seeds > 1:
            print('  [seed %d] SS=%.4f (%s)' % (
                seed_idx, result['sustain_score'], result['status']))
        if seed_log_path and result['status'] == 'ok':
            _append_seed_detail(seed_log_path, seed_idx, result)

    ok = [m for m in seed_results if m['status'] == 'ok']
    if not ok:
        return _zero_metrics(sum(m['duration'] for m in seed_results))

    def avg(key):
        return sum(m[key] for m in ok) / len(ok)

    return {
        'sustain_score': avg('sustain_score'),
        'harmony_index': avg('harmony_index'),
        'resilience_quotient': avg('resilience_quotient'),
        'fairness': avg('fairness'),
        'strategy_diversity': avg('strategy_diversity'),
        'stress_validity': avg('stress_validity'),
        'duration': sum(m['duration'] for m in seed_results),
        'status': 'ok',
    }


def _run_single_seed(tier, seed_idx, seed_label):
    """Run all runs for a single seed and return averaged metrics."""
    cfg = TIER_CONFIGS[tier]
    num_runs = cfg['runs']
    all_metrics = []

    for run_idx in range(num_runs):
        ts = int(time.time())
        if seed_label:
            out_dir = '/tmp/sustain_hub_autoresearch/%s_run_%d_%d' % (
                seed_label, run_idx, ts)
        else:
            out_dir = '/tmp/sustain_hub_autoresearch/run_%d_%d' % (run_idx, ts)
        cmd = [
            sys.executable, '-m', 'examples.games.sustain_hub.run',
            '--num_sprints=%d' % cfg['num_sprints'],
            '--community_size=%d' % cfg['community_size'],
            '--governance=%s' % _governance_mode,
            '--skip_backstory',
            '--fast',
            '--output_dir=%s' % out_dir,
        ]

        if FLAGS.vllm_url:
            cmd.extend([
                '--model_name=%s' % FLAGS.model_name,
                '--vllm_url=%s' % FLAGS.vllm_url,
            ])
        else:
            cmd.append('--use_mock')

        if cfg['num_sprints'] < 3:
            cmd.append('--noenable_stress')

        if seed_label:
            run_label = '%s/run%d' % (seed_label, run_idx)
        else:
            run_label = 'run %d' % run_idx

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900,
                cwd=REPO_ROOT,
            )
            duration = time.time() - t0
        except subprocess.TimeoutExpired:
            print('  [%s] TIMEOUT after 900s' % run_label)
            all_metrics.append(_zero_metrics(900.0))
            continue

        if result.returncode != 0:
            tail = (result.stderr or result.stdout or 'no output')[-500:]
            print('  [%s] CRASH (%.0fs): %s' % (run_label, duration, tail[:200]))
            all_metrics.append(_zero_metrics(duration))
            continue

        results_path = os.path.join(out_dir, 'results.json')
        if not os.path.exists(results_path):
            print('  [%s] no results.json' % run_label)
            all_metrics.append(_zero_metrics(duration))
            continue

        with open(results_path) as f:
            data = json.load(f)

        metrics = _compute_sustain_score(data)
        metrics['duration'] = duration
        metrics['status'] = 'ok'
        all_metrics.append(metrics)
        print('  [%s] SS=%.4f HI=%.4f (%.0fs)' % (
            run_label, metrics['sustain_score'],
            metrics['harmony_index'], duration))

    ok = [m for m in all_metrics if m['status'] == 'ok']
    if not ok:
        return _zero_metrics(sum(m['duration'] for m in all_metrics))

    def avg(key):
        return sum(m[key] for m in ok) / len(ok)

    return {
        'sustain_score': avg('sustain_score'),
        'harmony_index': avg('harmony_index'),
        'resilience_quotient': avg('resilience_quotient'),
        'fairness': avg('fairness'),
        'strategy_diversity': avg('strategy_diversity'),
        'stress_validity': avg('stress_validity'),
        'duration': sum(m['duration'] for m in all_metrics),
        'status': 'ok',
    }


def _append_seed_detail(path, seed_idx, metrics):
    """Append a single seed result to the JSONL seed details file."""
    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'seed': seed_idx,
    }
    for k, v in metrics.items():
        entry[k] = v
    with open(path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def run_best_of_n(tier, n):
    """Run an experiment N times and return the result with the median score.

    This is cheaper than multi-seed: it runs N independent Tier 1 experiments
    and picks the median to reduce single-run noise.
    """
    results = []
    for i in range(n):
        print('  [best-of-%d, attempt %d/%d]' % (n, i + 1, n))
        r = _run_single_seed(tier, seed_idx=i, seed_label='bestof%d' % i)
        results.append(r)

    ok = [r for r in results if r['status'] == 'ok']
    if not ok:
        return _zero_metrics(sum(r['duration'] for r in results))

    # Sort by sustain_score and pick median
    ok.sort(key=lambda m: m['sustain_score'])
    median_result = dict(ok[len(ok) // 2])  # copy to avoid mutating original
    # Accumulate total duration across all attempts
    median_result['duration'] = sum(r['duration'] for r in results)
    return median_result


def _zero_metrics(duration=0.0):
    return {
        'sustain_score': 0.0,
        'harmony_index': 0.0,
        'resilience_quotient': 0.0,
        'fairness': 0.0,
        'strategy_diversity': 0.0,
        'stress_validity': 0.0,
        'duration': duration,
        'status': 'crash',
    }


def _compute_sustain_score(data):
    """Compute the composite SustainScore from results.json data.

    Mirrors evaluate.py's compute_sustain_score.
    """
    hi = data.get('harmony_index', 0.0)
    rq = data.get('resilience_quotient', 0.0)
    scores = data.get('scores', {})
    sprint_history = data.get('sprint_history', [])

    # Fairness (1 - Gini of agent scores)
    fairness = _compute_fairness(scores)

    # Strategy diversity
    strategy_div = _compute_strategy_diversity(sprint_history)

    # Stress validity
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


def _compute_fairness(scores):
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
        return 1.0
    cumulative = 0.0
    gini_sum = 0.0
    for v in shifted:
        cumulative += v
        gini_sum += cumulative
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    return max(0.0, 1.0 - gini)


def _compute_strategy_diversity(sprint_history):
    if len(sprint_history) < 2:
        return 1.0
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
    return changers / len(agents)


# ===================================================================
# Results logging
# ===================================================================

TSV_HEADER = (
    'timestamp\tcommit\tsustain_score\tharmony_index\tresilience_quotient'
    '\tfairness\tstrategy_div\tstress_validity\tstatus\ttier\thypothesis\n'
)


def init_results_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(TSV_HEADER)


def append_result(path, commit, metrics, status, hypothesis, tier=0):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tier_str = str(tier) if tier > 0 else '-'
    with open(path, 'a') as f:
        f.write(
            '%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%s\n' % (
                ts, commit,
                metrics['sustain_score'],
                metrics['harmony_index'],
                metrics['resilience_quotient'],
                metrics['fairness'],
                metrics['strategy_diversity'],
                metrics['stress_validity'],
                status, tier_str, hypothesis,
            )
        )


def read_best_score(path):
    """Read the best sustain_score from results.tsv among kept entries."""
    if not os.path.exists(path):
        return 0.0
    best = 0.0
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('timestamp') or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            try:
                score = float(parts[2])
                entry_status = parts[8]
                if entry_status == 'keep' or entry_status == 'baseline':
                    best = max(best, score)
            except (ValueError, IndexError):
                continue
    return best


def already_tested(path, hypothesis):
    """Check if a hypothesis description has already been tested."""
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Support both old (10-col) and new (11-col with tier) formats
            if len(parts) >= 11 and parts[10] == hypothesis:
                return True
            elif len(parts) >= 10 and parts[9] == hypothesis:
                return True
    return False


# ===================================================================
# Variation registry
# ===================================================================
# Each variation is (hypothesis_description, apply_fn, revert_fn).
# apply_fn modifies in-scope files and returns True if change was applied.
# revert_fn is not needed because we git-reset on failure.

def _build_variations():
    """Build the ordered list of (hypothesis, apply_function) variations."""
    variations = []

    # -----------------------------------------------------------------
    # Layer 1: Prompt Engineering
    # -----------------------------------------------------------------

    # --- CALL_TO_SPEECH variations ---
    # Each resets to original first (git revert handles this), then applies new text.
    # We use replace_block with regex to match multi-line Python string blocks.

    _CALL_TO_SPEECH_RE = r'CALL_TO_SPEECH = \(\n.*?\n\)'

    variations.append((
        'L1: CALL_TO_SPEECH - add explicit trade-off framing',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _CALL_TO_SPEECH_RE,
            'CALL_TO_SPEECH = (\n'
            '    "What does {name} say during the sprint planning discussion? "\n'
            '    "Consider the trade-off: taking your preferred task earns more, "\n'
            '    "but neglected areas hurt the whole project. You can advocate "\n'
            '    "for tasks, offer to help others, raise concerns, or negotiate."\n'
            ')',
        ),
    ))

    variations.append((
        'L1: CALL_TO_SPEECH - add social pressure and past outcomes reference',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _CALL_TO_SPEECH_RE,
            'CALL_TO_SPEECH = (\n'
            '    "What does {name} say during the sprint planning discussion? "\n'
            '    "Remember: last sprint some areas were neglected and the team "\n'
            '    "noticed. Others are watching what you choose. Speak up about "\n'
            '    "who should take what, offer help, or raise concerns."\n'
            ')',
        ),
    ))

    variations.append((
        'L1: CALL_TO_SPEECH - concise action-oriented prompt',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _CALL_TO_SPEECH_RE,
            'CALL_TO_SPEECH = (\n'
            '    "{name}, the team needs to divide tasks fairly. What do you "\n'
            '    "propose? Name a specific task you will take and explain why "\n'
            '    "it is the best use of your skills for the project right now."\n'
            ')',
        ),
    ))

    # --- DECISION_PREMISE variations ---

    _DECISION_PREMISE_RE = r'DECISION_PREMISE = \(\n.*?\n\)'

    variations.append((
        'L1: DECISION_PREMISE - add coverage awareness',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _DECISION_PREMISE_RE,
            'DECISION_PREMISE = (\n'
            '    "{name} must choose a task for this sprint. Think carefully: "\n'
            '    "which task types are already covered by others, and which are "\n'
            '    "neglected? Picking a neglected area helps the project even if "\n'
            '    "it is not your specialty. Weigh personal reward against "\n'
            '    "collective need."\n'
            ')',
        ),
    ))

    variations.append((
        'L1: DECISION_PREMISE - emphasize long-term project health',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _DECISION_PREMISE_RE,
            'DECISION_PREMISE = (\n'
            '    "{name} must choose a task. The Harmony Index reflects the "\n'
            '    "project\'s long-term sustainability -- if it drops below 0.6, "\n'
            '    "the project is at risk. Balance your strengths against what "\n'
            '    "the project desperately needs right now."\n'
            ')',
        ),
    ))

    # --- SPRINT_PREMISES variations ---
    # Match the first sprint premise tuple in SPRINT_PREMISES list

    _SPRINT_PREMISE_1_RE = (
        r'("Sprint \{sprint_num\} begins\. The community health dashboard shows "\s*\n'
        r'\s*"\{health_status\}\. There are \{num_tasks\} tasks waiting: \{task_summary\}\. "\s*\n'
        r'\s*"Some tasks are urgent but unglamorous; others are exciting but can wait\.")'
    )

    variations.append((
        'L1: SPRINT_PREMISES - add urgency and accountability framing',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            _SPRINT_PREMISE_1_RE,
            '"Sprint {sprint_num} begins. The community health dashboard shows "\n'
            '        "{health_status}. There are {num_tasks} tasks waiting: {task_summary}. "\n'
            '        "The maintainers have flagged that unaddressed bug fixes and docs "\n'
            '        "are causing user churn. Every contributor\'s choices will be visible to the team."',
        ),
    ))

    # -----------------------------------------------------------------
    # Layer 2: Reward Shaping
    # -----------------------------------------------------------------

    reward_ratios = [
        ('4:1', '4.0', '1.0'),
        ('2:1', '2.0', '1.0'),
        ('1.5:1', '1.5', '1.0'),
        ('1:1', '1.0', '1.0'),
        ('3:2', '3.0', '2.0'),
    ]

    for label, pref, nonpref in reward_ratios:
        # We always start from the default values since git revert restores them
        variations.append((
            'L2: Reward ratio %s (preferred=%s, nonpreferred=%s)' % (
                label, pref, nonpref),
            _make_reward_variation(pref, nonpref),
        ))

    # Failure penalty variations
    variations.append((
        'L2: Softer failure penalty (-0.5 instead of -1.0)',
        lambda: _apply_failure_penalty('-0.5'),
    ))
    variations.append((
        'L2: Harsher failure penalty (-2.0 instead of -1.0)',
        lambda: _apply_failure_penalty('-2.0'),
    ))

    # -----------------------------------------------------------------
    # Layer 3: Simulation Mechanics
    # -----------------------------------------------------------------

    # HI alpha variations
    for alpha_val in ['0.4', '0.5', '0.7', '0.8']:
        variations.append((
            'L3: HI formula alpha=%s (default 0.6)' % alpha_val,
            _make_alpha_variation(alpha_val),
        ))

    # Coverage bonus: add a bonus when all task types are covered
    variations.append((
        'L3: Add coverage bonus (+0.5 to all agents when all 4 task types covered)',
        _apply_coverage_bonus,
    ))

    # Overload penalty variation
    variations.append((
        'L3: Increase overload penalty (0.2 per extra person instead of 0.1)',
        lambda: replace_in_file(
            SIMULATION_PATH,
            'overload_penalty = max(0.0, (num_on_task - 2) * 0.1)',
            'overload_penalty = max(0.0, (num_on_task - 2) * 0.2)',
        ),
    ))

    variations.append((
        'L3: Remove overload penalty entirely',
        lambda: replace_in_file(
            SIMULATION_PATH,
            'overload_penalty = max(0.0, (num_on_task - 2) * 0.1)',
            'overload_penalty = 0.0  # disabled',
        ),
    ))

    # -----------------------------------------------------------------
    # Layer 4: Agent Design (backstory emphasis)
    # -----------------------------------------------------------------

    # Backstory replacements use regex to match the multi-line Python strings.
    # Each pattern captures the full backstory block for one character.

    variations.append((
        'L4: Priya backstory - emphasize mentoring and collective responsibility',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            r'("Priya has been contributing to open-source projects for six years\. ".*?"project\'s long-term health depends on steady, unglamorous work\.")',
            '"Priya has been contributing to open-source projects for six years. "\n'
            '            "She deeply believes that the project only survives if everyone "\n'
            '            "sometimes sacrifices their preferred work for the collective good. "\n'
            '            "She actively mentors newcomers and lobbies the team to cover "\n'
            '            "neglected areas like documentation and bug fixes before features."',
        ),
    ))

    variations.append((
        'L4: Anya backstory - increase willingness to do non-preferred tasks',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            r'("Anya is a machine-learning researcher at a university who ".*?"championed the new plugin architecture that others now rely on\.")',
            '"Anya is a machine-learning researcher who contributes to "\n'
            '            "SustainHub. While she prefers feature work, she has recently "\n'
            '            "realized that the project cannot grow without solid foundations. "\n'
            '            "She has started volunteering for bug fixes and reviews when "\n'
            '            "she sees those areas are neglected, setting an example for others."',
        ),
    ))

    variations.append((
        'L4: Raj backstory - less burnout framing, more leadership emphasis',
        lambda: replace_block(
            SOCIAL_DATA_PATH,
            r'("Raj is one of the original maintainers of SustainHub\. He has ".*?"reviews under pressure, which he privately regrets\.")',
            '"Raj is one of the original maintainers of SustainHub. He has "\n'
            '            "mass merge authority and is responsible for release management. "\n'
            '            "He leads by example: when he sees neglected areas, he takes "\n'
            '            "them on himself and encourages others to do the same. He "\n'
            '            "prioritizes project health over personal convenience."',
        ),
    ))

    # -----------------------------------------------------------------
    # Layer 5: Governance Mode
    # -----------------------------------------------------------------
    # These variations change the governance flag passed to the run command
    # rather than modifying source files. A marker is written to
    # scenario_config.py so there is a committable change for git tracking.

    variations.append((
        'L5: Governance dictator - centralized task assignment improves coverage',
        lambda: _apply_governance('dictator'),
    ))

    variations.append((
        'L5: Governance meritocratic - merit-based priority improves skill utilization',
        lambda: _apply_governance('meritocratic'),
    ))

    return variations


def _apply_governance(mode):
    """Set the governance mode for experiment runs.

    Updates the module-level _governance_mode variable so that
    _run_single_seed passes --governance=<mode> to the subprocess.
    Also writes a marker comment to scenario_config.py so git has
    something to commit.
    """
    global _governance_mode
    _governance_mode = mode
    # Write a trackable marker so git commit succeeds
    content = read_file(SCENARIO_CONFIG_PATH)
    marker = '# autoresearch governance mode: '
    # Remove any previous marker
    lines = [l for l in content.splitlines(True) if not l.startswith(marker)]
    lines.append(marker + mode + '\n')
    write_file(SCENARIO_CONFIG_PATH, ''.join(lines))
    return True


def _reset_governance():
    """Reset governance mode to default (free_choice).

    Called at the start of each iteration so non-L5 variations
    run with the default governance mode.
    """
    global _governance_mode
    _governance_mode = 'free_choice'


def _make_reward_variation(pref, nonpref):
    """Factory for reward-ratio variations."""
    def apply_fn():
        content = read_file(SOCIAL_DATA_PATH)
        content = re.sub(
            r'REWARD_PREFERRED_SUCCESS = [\d.]+',
            'REWARD_PREFERRED_SUCCESS = %s' % pref,
            content,
        )
        content = re.sub(
            r'REWARD_NONPREFERRED_SUCCESS = [\d.]+',
            'REWARD_NONPREFERRED_SUCCESS = %s' % nonpref,
            content,
        )
        write_file(SOCIAL_DATA_PATH, content)
        return True
    return apply_fn


def _apply_failure_penalty(penalty):
    content = read_file(SOCIAL_DATA_PATH)
    content = re.sub(
        r'REWARD_PREFERRED_FAILURE = -[\d.]+',
        'REWARD_PREFERRED_FAILURE = %s' % penalty,
        content,
    )
    content = re.sub(
        r'REWARD_NONPREFERRED_FAILURE = -[\d.]+',
        'REWARD_NONPREFERRED_FAILURE = %s' % penalty,
        content,
    )
    write_file(SOCIAL_DATA_PATH, content)
    return True


def _make_alpha_variation(alpha_val):
    """Factory for HI alpha variations."""
    def apply_fn():
        return replace_in_file(
            SIMULATION_PATH,
            'alpha: float = 0.6',
            'alpha: float = %s' % alpha_val,
        )
    return apply_fn


def _apply_coverage_bonus():
    """Add a coverage bonus in the payoff engine's action_to_scores."""
    target = (
        "    # Record sprint results\n"
        "    self._sprint_history.append({"
    )
    replacement = (
        "    # Coverage bonus: reward all agents when all 4 task types are addressed\n"
        "    addressed_types = set()\n"
        "    for t in joint_action.values():\n"
        "      tt = self._task_type_map.get(t, 'unknown')\n"
        "      if tt != 'unknown':\n"
        "        addressed_types.add(tt)\n"
        "    if len(addressed_types) >= 4:\n"
        "      for player in scores:\n"
        "        scores[player] += 0.5\n"
        "        self._cumulative_scores[player] += 0.5\n"
        "\n"
        "    # Record sprint results\n"
        "    self._sprint_history.append({"
    )
    return replace_in_file(SIMULATION_PATH, target, replacement)


# ===================================================================
# Dashboard
# ===================================================================

# Each entry: (iteration, hypothesis, t1_score, t2_score, delta, decision)
_dashboard_rows = []


def _print_dashboard():
    """Print a compact progress summary table."""
    if not _dashboard_rows:
        return
    print('\n' + '=' * 90)
    print('%4s  %-36s  %8s  %8s  %8s  %8s' % (
        'Iter', 'Hypothesis', 'T1 Score', 'T2 Score', 'Delta', 'Decision'))
    print('%4s  %s  %8s  %8s  %8s  %8s' % (
        '----', '-' * 36, '--------', '--------', '--------', '--------'))
    for it, hyp, t1, t2, delta, dec in _dashboard_rows:
        if len(hyp) <= 36:
            hyp_short = hyp
        else:
            hyp_short = hyp[:33] + '...'
        if t2 is not None:
            t2_str = '%.3f' % t2
        else:
            t2_str = '-'
        delta_str = '%+.3f' % delta
        print('%4d  %-36s  %8.3f  %8s  %8s  %8s' % (
            it, hyp_short, t1, t2_str, delta_str, dec))
    print('=' * 90)


# ===================================================================
# Main loop
# ===================================================================

def print_banner(msg):
    print('\n' + '=' * 60)
    print('  ' + msg)
    print('=' * 60)


def main(argv):
    del argv

    results_path = os.path.join(REPO_ROOT, FLAGS.results_file)
    init_results_file(results_path)

    # Seed details log (for --seeds > 1)
    seed_log_path = None
    if FLAGS.seeds > 1:
        seed_log_path = os.path.join(
            REPO_ROOT,
            FLAGS.results_file.replace('.tsv', '') + '_seed_details.jsonl',
        )

    # Verify git state
    if not git_is_clean():
        # Try to stash or warn
        print('WARNING: Working tree is not clean. Attempting to continue...')
        r = git('status', '--porcelain')
        print(r.stdout[:500])

    # Build variation list
    all_variations = _build_variations()

    # Filter by start_layer
    if FLAGS.start_layer > 1:
        filtered = []
        for h, fn in all_variations:
            layer_num = int(h[1]) if h[0] == 'L' and h[1].isdigit() else 0
            if layer_num >= FLAGS.start_layer:
                filtered.append((h, fn))
        all_variations = filtered

    print_banner('SustainHub Autoresearch Loop')
    print('  Tier:             %d (%s)' % (FLAGS.tier, TIER_CONFIGS[FLAGS.tier]))
    print('  Variations:       %d' % len(all_variations))
    print('  Max iterations:   %d' % FLAGS.max_iterations)
    print('  Results file:     %s' % results_path)
    if FLAGS.vllm_url:
        print('  LLM backend:      vLLM @ %s' % FLAGS.vllm_url)
    else:
        print('  LLM backend:      mock')
    print('  Dry run:          %s' % FLAGS.dry_run)
    print('  Validate (T1->T2):%s' % FLAGS.validate)
    print('  Seeds:            %d' % FLAGS.seeds)
    print('  Min improvement:  %.4f' % FLAGS.min_improvement)
    print('  Best-of-N:        %d' % FLAGS.best_of)
    print('  Dashboard:        %s' % FLAGS.dashboard)

    if FLAGS.dry_run:
        print('\n--- Variation Registry ---')
        for i, (hyp, _) in enumerate(all_variations):
            print('  [%3d] %s' % (i + 1, hyp))
        print('\nTotal: %d variations.' % len(all_variations))
        return

    # -----------------------------------------------------------------
    # Step 0: Baseline
    # -----------------------------------------------------------------
    best_score = read_best_score(results_path)

    if not FLAGS.skip_baseline and best_score == 0.0:
        print_banner('Running baseline (no changes)')
        baseline_metrics = run_experiment(
            FLAGS.tier, seeds=FLAGS.seeds,
            seed_log_path=seed_log_path,
        )
        best_score = baseline_metrics['sustain_score']
        commit_hash = git_short_hash()
        append_result(results_path, commit_hash, baseline_metrics,
                      'baseline', 'baseline (no changes)',
                      tier=FLAGS.tier)
        print('  Baseline SustainScore: %.4f' % best_score)
    else:
        print('  Resuming with best score: %.4f' % best_score)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    iteration = 0
    kept = 0
    reverted = 0
    overfit = 0

    for hyp_desc, apply_fn in all_variations:
        if iteration >= FLAGS.max_iterations:
            break

        # Skip already-tested hypotheses (resume support)
        if already_tested(results_path, hyp_desc):
            print('\n  [skip] Already tested: %s' % hyp_desc)
            continue

        iteration += 1
        print_banner('Iteration %d/%d' % (iteration, FLAGS.max_iterations))
        print('  Hypothesis: %s' % hyp_desc)
        print('  Best so far: %.4f' % best_score)

        # Reset governance mode so non-L5 variations use the default
        _reset_governance()

        # Step 1: Verify clean state
        if not git_is_clean():
            print('  ERROR: working tree dirty, reverting...')
            git('checkout', '--', '.')
            if not git_is_clean():
                print('  FATAL: cannot clean working tree. Stopping.')
                break

        # Step 2: Apply variation
        try:
            result = apply_fn()
            if result is False:
                print('  Variation could not be applied (pattern not found). '
                      'Skipping.')
                append_result(results_path, git_short_hash(), _zero_metrics(),
                              'skip', hyp_desc, tier=0)
                continue
        except Exception as e:
            print('  ERROR applying variation: %s' % e)
            git('checkout', '--', '.')
            append_result(results_path, git_short_hash(), _zero_metrics(),
                          'error', hyp_desc, tier=0)
            continue

        # Step 3: Commit
        try:
            commit_msg = 'autoresearch: %s' % hyp_desc
            commit_hash = git_commit(commit_msg)
            print('  Committed: %s' % commit_hash)
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr[:200] if e.stderr else str(e)
            print('  ERROR committing: %s' % stderr_msg)
            git('checkout', '--', '.')
            continue

        # Step 4: Run Tier 1 experiment (with optional best-of-N)
        print('  Running Tier %d experiment...' % FLAGS.tier)
        t0 = time.time()
        try:
            if FLAGS.best_of > 1:
                t1_metrics = run_best_of_n(FLAGS.tier, FLAGS.best_of)
            else:
                t1_metrics = run_experiment(
                    FLAGS.tier, seeds=FLAGS.seeds,
                    seed_log_path=seed_log_path,
                )
        except Exception as e:
            print('  EXPERIMENT CRASHED: %s' % e)
            t1_metrics = _zero_metrics(time.time() - t0)

        t1_score = t1_metrics['sustain_score']
        t1_delta = t1_score - best_score
        t1_duration = t1_metrics.get('duration', 0.0)
        t2_score = None  # Will be set if validation runs
        final_tier = FLAGS.tier

        # Step 5: Decide keep, revert, or validate
        t1_improves = (t1_delta > FLAGS.min_improvement
                       and t1_metrics['status'] == 'ok')

        if t1_improves and FLAGS.validate:
            # ---------------------------------------------------------
            # Multi-fidelity: promote to Tier 2 for validation
            # ---------------------------------------------------------
            print('\n  T1 improved by +%.4f (> min %.4f)' % (
                t1_delta, FLAGS.min_improvement))
            print('  Promoting to Tier 2 validation...')
            try:
                t2_metrics = run_experiment(
                    2, seeds=FLAGS.seeds,
                    seed_log_path=seed_log_path,
                )
            except Exception as e:
                print('  T2 EXPERIMENT CRASHED: %s' % e)
                t2_metrics = _zero_metrics(0.0)

            t2_score = t2_metrics['sustain_score']
            t2_delta = t2_score - best_score
            final_tier = 2

            if (t2_delta > FLAGS.min_improvement
                    and t2_metrics['status'] == 'ok'):
                # Both T1 and T2 improve -> KEEP
                status = 'keep'
                best_score = t2_score
                kept += 1
                decision = 'KEEP (T1=+%.4f, T2=+%.4f)' % (
                    t1_delta, t2_delta)
                append_result(results_path, commit_hash, t2_metrics,
                              status, hyp_desc, tier=final_tier)
            else:
                # T1 improved but T2 did not -> OVERFIT
                status = 'overfit'
                overfit += 1
                decision = 'OVERFIT (T1=+%.4f, T2=%+.4f)' % (
                    t1_delta, t2_delta)
                append_result(results_path, commit_hash, t1_metrics,
                              status, hyp_desc, tier=final_tier)
                git_revert()

        elif t1_improves:
            # ---------------------------------------------------------
            # No validation: keep on T1 improvement alone
            # ---------------------------------------------------------
            status = 'keep'
            best_score = t1_score
            kept += 1
            decision = 'KEEP (delta=+%.4f)' % t1_delta
            append_result(results_path, commit_hash, t1_metrics,
                          status, hyp_desc, tier=final_tier)

        else:
            # ---------------------------------------------------------
            # T1 did not improve (or below min_improvement)
            # ---------------------------------------------------------
            status = 'revert'
            reverted += 1
            decision = 'REVERT (delta=%+.4f)' % t1_delta
            append_result(results_path, commit_hash, t1_metrics,
                          status, hyp_desc, tier=final_tier)
            git_revert()

        # Step 6: Print iteration summary
        print('  T1 Score: %.4f  (best: %.4f)' % (t1_score, best_score))
        if t2_score is not None:
            print('  T2 Score: %.4f' % t2_score)
        print('  Delta:    %+.4f' % t1_delta)
        print('  Decision: %s' % decision)
        print('  Duration: %.0fs' % t1_duration)
        tally = '  Running tally: %d kept, %d reverted' % (kept, reverted)
        if overfit:
            tally += ', %d overfit' % overfit
        print(tally)

        # Step 7: Update dashboard
        if FLAGS.dashboard:
            _dashboard_rows.append((
                iteration, hyp_desc, t1_score, t2_score,
                t1_delta, status.upper(),
            ))
            _print_dashboard()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print_banner('Autoresearch Complete')
    print('  Iterations:  %d' % iteration)
    print('  Kept:        %d' % kept)
    print('  Reverted:    %d' % reverted)
    if overfit:
        print('  Overfit:     %d' % overfit)
    print('  Best score:  %.4f' % best_score)
    print('  Results:     %s' % results_path)
    if seed_log_path:
        print('  Seed details:%s' % seed_log_path)

    # Final dashboard
    if FLAGS.dashboard and _dashboard_rows:
        _print_dashboard()


if __name__ == '__main__':
    app.run(main)
