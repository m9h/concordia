"""Code execution scoring engine for SustainHub agents.

Connects LLM agents to the toy calculator project by:
  1. Loading issues from toy_project/issues/
  2. Generating patches via an LLM
  3. Applying patches to a temp copy and running pytest
  4. Mapping test pass rates to SustainHub reward constants

This module replaces the stochastic success/failure in the abstract
simulation with real pass/fail test scoring.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOY_PROJECT_DIR = os.path.join(_THIS_DIR, 'toy_project')
ISSUE_DIR = os.path.join(TOY_PROJECT_DIR, 'issues')

# ---------------------------------------------------------------------------
# Reward constants (duplicated here for standalone use; prefer importing
# from social_data when running inside the full framework).
# ---------------------------------------------------------------------------

try:
    from examples.games.sustain_hub import social_data
    REWARD_PREFERRED_SUCCESS = social_data.REWARD_PREFERRED_SUCCESS
    REWARD_PREFERRED_FAILURE = social_data.REWARD_PREFERRED_FAILURE
    REWARD_NONPREFERRED_SUCCESS = social_data.REWARD_NONPREFERRED_SUCCESS
    REWARD_NONPREFERRED_FAILURE = social_data.REWARD_NONPREFERRED_FAILURE
    REWARD_SKIP = social_data.REWARD_SKIP
    TASK_TYPES = social_data.TASK_TYPES
except ImportError:
    # Standalone fallback -- same numeric values as social_data.py
    REWARD_PREFERRED_SUCCESS = 3.0
    REWARD_PREFERRED_FAILURE = -1.0
    REWARD_NONPREFERRED_SUCCESS = 1.0
    REWARD_NONPREFERRED_FAILURE = -1.0
    REWARD_SKIP = 0.0
    TASK_TYPES = ['bug_fix', 'feature', 'documentation', 'code_review']

# Timeout for pytest subprocess (seconds).
_PYTEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# 1. Load issues from the toy project
# ---------------------------------------------------------------------------

# Map from issue id -> test class name (derived from test_calculator.py).
# This is the ground truth; issue markdown metadata is parsed best-effort.
_TEST_CLASS_MAP: dict[int, str] = {
    1: 'TestIssue1_DivideByZero',
    2: 'TestIssue2_NegativeSqrt',
    3: 'TestIssue3_FactorialValidation',
    4: 'TestIssue4_Percentage',
    5: 'TestIssue5_AbsoluteValue',
    6: 'TestIssue6_Logarithm',
    7: 'TestIssue7_Mean',
    8: 'TestIssue8_IsPrime',
    9: 'TestIssue9_Docstrings',
    10: 'TestIssue10_HistoryErrors',
}


def _parse_issue_metadata(text: str) -> dict[str, str]:
    """Extract **Key**: value or **Key:** value pairs from issue markdown."""
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        # Matches both  **Type**: bug_fix  and  **Type:** bug_fix
        m = re.match(r'\*\*(\w[\w\s]*?)\*\*:?\s*:?\s*(.+)', line)
        if m:
            key = m.group(1).strip().lower().replace(' ', '_')
            metadata[key] = m.group(2).strip()
    return metadata


def _extract_title(text: str) -> str:
    """Return the first markdown heading as the issue title."""
    for line in text.splitlines():
        if line.startswith('# '):
            # Strip leading "# Issue N: " prefix if present.
            title = re.sub(r'^#\s*Issue\s*\d+:\s*', '', line).strip()
            return title
    return '(untitled)'


def load_issues() -> dict[int, dict[str, Any]]:
    """Read all issue_*.md files and return structured metadata.

    Returns:
        Dict mapping issue_id -> {
            'title': str,
            'type': str,       # bug_fix | feature | documentation | code_review
            'difficulty': str,  # easy | medium | hard
            'description': str, # full markdown body
            'test_class': str,  # e.g. TestIssue1_DivideByZero
            'file': str,        # relative path inside toy_project
        }
    """
    issues: dict[int, dict[str, Any]] = {}
    if not os.path.isdir(ISSUE_DIR):
        return issues

    for fname in sorted(os.listdir(ISSUE_DIR)):
        m = re.match(r'issue_(\d+)\.md$', fname)
        if not m:
            continue
        issue_id = int(m.group(1))
        fpath = os.path.join(ISSUE_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()

        meta = _parse_issue_metadata(text)
        title = _extract_title(text)
        test_class = _TEST_CLASS_MAP.get(issue_id, f'TestIssue{issue_id}')

        issues[issue_id] = {
            'title': title,
            'type': meta.get('type', 'bug_fix'),
            'difficulty': meta.get('difficulty', 'medium'),
            'description': text,
            'test_class': test_class,
            'file': meta.get('file', 'calculator/calculator.py'),
        }

    return issues


# ---------------------------------------------------------------------------
# 2. Generate patches via LLM
# ---------------------------------------------------------------------------

def generate_patch(
    issue: dict[str, Any],
    model: Any,
    agent_expertise: str = 'software',
) -> str:
    """Ask the LLM to produce a unified diff that fixes *issue*.

    Args:
        issue: An entry from load_issues().
        model: A concordia LanguageModel (or any object with sample_text).
        agent_expertise: Free-text description of the agent's expertise.

    Returns:
        Raw diff string from the model (may need cleanup before applying).
    """
    source_path = os.path.join(TOY_PROJECT_DIR, issue['file'])
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        source_code = '(file not found)'

    prompt = (
        f'You are a {agent_expertise} developer. '
        f'Here is the issue:\n\n{issue["description"]}\n\n'
        f'Here is the current source code of `{issue["file"]}`:\n'
        f'```python\n{source_code}\n```\n\n'
        f'Generate a unified diff patch that fixes this issue. '
        f'Output ONLY the diff, nothing else.'
    )

    response = model.sample_text(prompt, max_tokens=2048, temperature=0.2)
    return response


# ---------------------------------------------------------------------------
# 3. Apply and test patches
# ---------------------------------------------------------------------------

def _has_patch_command() -> bool:
    """Return True if the `patch` CLI utility is available."""
    try:
        subprocess.run(
            ['patch', '--version'],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _apply_patch_cli(tmp_dir: str, patch_text: str) -> tuple[bool, str]:
    """Apply *patch_text* using the `patch` command. Returns (ok, msg)."""
    patch_file = os.path.join(tmp_dir, '_agent.patch')
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_text)
    result = subprocess.run(
        ['patch', '-p1', '--batch', '--fuzz=3', '-i', patch_file],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
        timeout=10,
    )
    if result.returncode == 0:
        return True, result.stdout
    return False, result.stderr or result.stdout


def _apply_patch_fallback(tmp_dir: str, patch_text: str) -> tuple[bool, str]:
    """Best-effort string-replacement patch application.

    Handles the most common case: a single-file unified diff that replaces
    a contiguous block of old lines with new lines.
    """
    # Extract target file from diff header (--- a/path or --- path)
    file_match = re.search(r'^---\s+(?:a/)?(.*?)(?:\t.*)?$', patch_text, re.M)
    if not file_match:
        # No diff header -- maybe the model just returned the new file content.
        # Try to detect a python code block and overwrite calculator.py.
        code_match = re.search(
            r'```python\n(.*?)```', patch_text, re.S
        )
        if code_match:
            target = os.path.join(tmp_dir, 'calculator', 'calculator.py')
            with open(target, 'w', encoding='utf-8') as f:
                f.write(code_match.group(1))
            return True, 'Applied as full-file replacement from code block.'
        return False, 'Could not parse diff header or code block.'

    rel_path = file_match.group(1).strip()
    target = os.path.join(tmp_dir, rel_path)
    if not os.path.isfile(target):
        return False, f'Target file not found: {rel_path}'

    with open(target, 'r', encoding='utf-8') as f:
        original = f.read()

    # Collect all hunks: extract removed (-) and added (+) blocks.
    modified = original
    hunks = re.findall(
        r'^@@.*?@@.*?\n((?:[+ \-].*\n)*)',
        patch_text,
        re.M,
    )
    for hunk in hunks:
        old_lines: list[str] = []
        new_lines: list[str] = []
        for line in hunk.splitlines(keepends=True):
            if line.startswith('-'):
                old_lines.append(line[1:])
            elif line.startswith('+'):
                new_lines.append(line[1:])
            elif line.startswith(' '):
                old_lines.append(line[1:])
                new_lines.append(line[1:])
        old_block = ''.join(old_lines)
        new_block = ''.join(new_lines)
        if old_block and old_block in modified:
            modified = modified.replace(old_block, new_block, 1)

    if modified == original:
        return False, 'Patch produced no changes (old block not found in file).'

    with open(target, 'w', encoding='utf-8') as f:
        f.write(modified)
    return True, 'Applied via string replacement.'


def _apply_patch(tmp_dir: str, patch_text: str) -> tuple[bool, str]:
    """Apply a patch to *tmp_dir*, trying CLI first then fallback."""
    if _has_patch_command():
        ok, msg = _apply_patch_cli(tmp_dir, patch_text)
        if ok:
            return ok, msg
        # CLI failed -- try fallback
    return _apply_patch_fallback(tmp_dir, patch_text)


def _run_pytest(tmp_dir: str, test_class: str) -> dict[str, Any]:
    """Run pytest on a specific test class inside *tmp_dir*.

    Returns dict with keys: passed, failed, total, error, raw_output.
    """
    test_spec = f'tests/test_calculator.py::{test_class}'
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_spec, '-v', '--tb=short'],
            capture_output=True,
            text=True,
            cwd=tmp_dir,
            timeout=_PYTEST_TIMEOUT,
        )
        output = result.stdout + '\n' + result.stderr
    except subprocess.TimeoutExpired:
        return {
            'passed': 0,
            'failed': 0,
            'total': 0,
            'error': 'pytest timed out',
            'raw_output': '',
        }

    # Parse the summary line, e.g. "3 passed, 1 failed" or "4 passed"
    passed = 0
    failed = 0
    m_passed = re.search(r'(\d+)\s+passed', output)
    m_failed = re.search(r'(\d+)\s+failed', output)
    m_error = re.search(r'(\d+)\s+error', output)
    if m_passed:
        passed = int(m_passed.group(1))
    if m_failed:
        failed = int(m_failed.group(1))
    if m_error:
        failed += int(m_error.group(1))
    total = passed + failed

    error_msg = None
    if total == 0:
        # No tests collected -- likely import error or bad class name
        error_msg = 'No tests collected. Possible import error.'
        # Try to extract a useful snippet
        for marker in ('ERRORS', 'ERROR', 'ImportError', 'ModuleNotFoundError'):
            if marker in output:
                idx = output.index(marker)
                error_msg = output[idx:idx + 500].strip()
                break

    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'error': error_msg,
        'raw_output': output,
    }


def score_patch(issue_id: int, patch: str) -> dict[str, Any]:
    """Apply *patch* to a temp copy of the toy project and run tests.

    Args:
        issue_id: The issue number (1-10).
        patch: A unified diff string (or raw replacement code).

    Returns:
        {
            'issue_id': int,
            'passed': int,
            'failed': int,
            'total': int,
            'score': float,   # passed / total, 0.0 - 1.0
            'error': str | None,
        }
    """
    issues = load_issues()
    issue = issues.get(issue_id)
    if issue is None:
        return {
            'issue_id': issue_id,
            'passed': 0,
            'failed': 0,
            'total': 0,
            'score': 0.0,
            'error': f'Unknown issue id: {issue_id}',
        }

    test_class = issue['test_class']
    tmp_dir = tempfile.mkdtemp(prefix=f'sustain_issue{issue_id}_')

    try:
        # Copy the toy project into the temp directory.
        shutil.copytree(
            TOY_PROJECT_DIR,
            tmp_dir,
            dirs_exist_ok=True,
        )

        # Apply the patch.
        ok, apply_msg = _apply_patch(tmp_dir, patch)
        if not ok:
            return {
                'issue_id': issue_id,
                'passed': 0,
                'failed': 0,
                'total': 0,
                'score': 0.0,
                'error': f'Patch application failed: {apply_msg}',
            }

        # Run pytest.
        result = _run_pytest(tmp_dir, test_class)
        total = result['total']
        passed = result['passed']
        score = passed / total if total > 0 else 0.0

        return {
            'issue_id': issue_id,
            'passed': passed,
            'failed': result['failed'],
            'total': total,
            'score': score,
            'error': result['error'],
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Map issues to SustainHub task types
# ---------------------------------------------------------------------------

# Display-label prefixes per task type.
_LABEL_PREFIX: dict[str, str] = {
    'bug_fix': 'Fix',
    'feature': 'Feature',
    'documentation': 'Docs',
    'code_review': 'Review',
}


def get_task_pool(
    num_tasks: int = 8,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Select a balanced set of code-task issues.

    Tries to include a mix of bug_fix, feature, and documentation issues.
    Falls back gracefully if fewer issues are available than requested.

    Args:
        num_tasks: Target number of tasks to return.
        rng: Seeded random.Random instance (for reproducibility).

    Returns:
        List of dicts with keys: label, issue_id, task_type, difficulty.
    """
    if rng is None:
        rng = random.Random()

    issues = load_issues()
    if not issues:
        return []

    # Group by type.
    by_type: dict[str, list[int]] = {}
    for iid, meta in issues.items():
        by_type.setdefault(meta['type'], []).append(iid)

    # Build a balanced selection: cycle through available types.
    selected_ids: list[int] = []
    type_order = [t for t in ['bug_fix', 'feature', 'documentation', 'code_review']
                  if t in by_type]

    # Shuffle within each type for variety.
    for t in type_order:
        rng.shuffle(by_type[t])

    idx = 0
    while len(selected_ids) < num_tasks and type_order:
        t = type_order[idx % len(type_order)]
        if by_type[t]:
            selected_ids.append(by_type[t].pop(0))
        else:
            type_order.remove(t)
            if not type_order:
                break
            continue
        idx += 1

    # If we still haven't reached num_tasks, add any remaining issues.
    remaining = [iid for iid in issues if iid not in selected_ids]
    rng.shuffle(remaining)
    for iid in remaining:
        if len(selected_ids) >= num_tasks:
            break
        selected_ids.append(iid)

    # Build task pool entries.
    pool: list[dict[str, Any]] = []
    for iid in selected_ids:
        meta = issues[iid]
        prefix = _LABEL_PREFIX.get(meta['type'], meta['type'].title())
        label = f'{prefix}: {meta["title"]}'
        pool.append({
            'label': label,
            'issue_id': iid,
            'task_type': meta['type'],
            'difficulty': meta['difficulty'],
        })

    return pool


# ---------------------------------------------------------------------------
# 5. Scoring integration
# ---------------------------------------------------------------------------

def score_code_task(
    issue_id: int,
    patch: str,
    *,
    is_preferred: bool = False,
) -> float:
    """Score a code task and return a SustainHub-compatible reward.

    This replaces the stochastic success/failure roll in the abstract
    simulation with real pytest pass/fail scoring.

    Args:
        issue_id: The issue number.
        patch: Unified diff (or replacement code) from the agent.
        is_preferred: Whether this task type is the agent's preferred type.

    Returns:
        A float reward on the SustainHub scale:
            All tests pass  -> REWARD_PREFERRED_SUCCESS (3.0) or
                               REWARD_NONPREFERRED_SUCCESS (1.0)
            Some tests pass -> proportional interpolation
            No tests pass   -> REWARD_PREFERRED_FAILURE (-1.0) or
                               REWARD_NONPREFERRED_FAILURE (-1.0)
    """
    result = score_patch(issue_id, patch)
    score = result['score']  # 0.0 to 1.0

    if is_preferred:
        success_reward = REWARD_PREFERRED_SUCCESS
        failure_reward = REWARD_PREFERRED_FAILURE
    else:
        success_reward = REWARD_NONPREFERRED_SUCCESS
        failure_reward = REWARD_NONPREFERRED_FAILURE

    if score >= 1.0:
        return success_reward
    elif score <= 0.0:
        return failure_reward
    else:
        # Linear interpolation between failure and success rewards.
        return failure_reward + score * (success_reward - failure_reward)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    issues = load_issues()
    if not issues:
        print('No issues found. Check ISSUE_DIR:', ISSUE_DIR)
        sys.exit(1)

    print(f'Loaded {len(issues)} issues from {ISSUE_DIR}\n')
    for iid, issue in sorted(issues.items()):
        print(
            f'  Issue {iid:>2}: [{issue["type"]:<15}] '
            f'[{issue["difficulty"]:<6}] {issue["title"]}'
        )

    print(f'\nTask pool (default 8):')
    pool = get_task_pool(num_tasks=8, rng=random.Random(42))
    for task in pool:
        print(f'  {task["label"]}  (issue {task["issue_id"]}, {task["difficulty"]})')
