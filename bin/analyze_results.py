#!/usr/bin/env python3
"""Results analysis and visualization for SustainHub autoresearch experiments.

Analyzes autoresearch results.tsv, experiment ladder ladder_results.json,
and overnight runner manifest.json files. Produces summary tables, component
breakdowns, and trajectory analysis in multiple output formats.

Usage:
  # Analyze autoresearch results (auto-detects results.tsv in cwd):
  uv run python bin/analyze_results.py --mode=autoresearch

  # Analyze with explicit input and top-N:
  uv run python bin/analyze_results.py --mode=autoresearch --input=results.tsv --top_n=5

  # Analyze experiment ladder:
  uv run python bin/analyze_results.py --mode=ladder

  # Analyze overnight model sweep:
  uv run python bin/analyze_results.py --mode=sweep --input=/tmp/sustainhub_overnight/manifest.json

  # Export as TSV for piping:
  uv run python bin/analyze_results.py --mode=autoresearch --format=tsv

  # Compare two result files side-by-side:
  uv run python bin/analyze_results.py --mode=autoresearch --compare=results_old.tsv --input=results.tsv

  # JSON output for programmatic consumption:
  uv run python bin/analyze_results.py --mode=ladder --format=json
"""

import csv
import io
import json
import math
import os
import statistics
import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'mode', None, ['autoresearch', 'ladder', 'sweep'],
    'Analysis mode: autoresearch (results.tsv), ladder (ladder_results.json), '
    'or sweep (manifest.json).')
flags.DEFINE_string(
    'input', None,
    'Path to results file or directory. Auto-detected if omitted.')
flags.DEFINE_enum(
    'format', 'table', ['table', 'tsv', 'json', 'csv'],
    'Output format.')
flags.DEFINE_integer(
    'top_n', 0,
    'Show only top N results (0 = show all).')
flags.DEFINE_string(
    'sort_by', 'sustain_score',
    'Column to sort by (default: sustain_score).')
flags.DEFINE_string(
    'compare', None,
    'Path to a second results file for side-by-side comparison.')

flags.mark_flag_as_required('mode')

# ---------------------------------------------------------------------------
# ASCII table helpers (no external deps)
# ---------------------------------------------------------------------------


def _format_table(headers: list[str], rows: list[list[str]],
                  min_width: int = 8) -> str:
    """Render an ASCII table with fixed-width columns."""
    if not headers:
        return ''
    col_widths = [max(min_width, len(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    def _row_str(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else min_width
            parts.append(str(cell).ljust(w))
        return '  '.join(parts)

    lines = [_row_str(headers)]
    lines.append('  '.join('-' * w for w in col_widths))
    for row in rows:
        lines.append(_row_str(row))
    return '\n'.join(lines)


def _format_kv(pairs: list[tuple[str, str]], indent: int = 2) -> str:
    """Render key-value pairs as aligned text."""
    if not pairs:
        return ''
    max_key = max(len(k) for k, _ in pairs)
    prefix = ' ' * indent
    return '\n'.join(f'{prefix}{k:<{max_key}}  {v}' for k, v in pairs)


def _safe_float(val, default=0.0):
    """Parse a float, returning default on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fmt(val, decimals=4):
    """Format a float for display."""
    if isinstance(val, float):
        return f'{val:.{decimals}f}'
    return str(val)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def emit_output(fmt: str, *, table_text: str = '', data: object = None,
                headers: list[str] = None, rows: list[list] = None):
    """Emit results in the requested format.

    For 'table' format, prints table_text directly.
    For 'json', serializes data.
    For 'tsv'/'csv', uses headers+rows.
    """
    if fmt == 'table':
        print(table_text)
    elif fmt == 'json':
        print(json.dumps(data, indent=2, default=str))
    elif fmt in ('tsv', 'csv'):
        if headers and rows is not None:
            sep = '\t' if fmt == 'tsv' else ','
            buf = io.StringIO()
            if fmt == 'csv':
                writer = csv.writer(buf)
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)
            else:
                buf.write(sep.join(headers) + '\n')
                for row in rows:
                    buf.write(sep.join(str(c) for c in row) + '\n')
            print(buf.getvalue(), end='')


# ===================================================================
# Autoresearch analysis
# ===================================================================


def _parse_results_tsv(path: str) -> list[dict]:
    """Parse autoresearch results.tsv into a list of row dicts."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r') as f:
        header_line = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header_line is None:
                header_line = line.split('\t')
                continue
            parts = line.split('\t')
            row = {}
            for i, key in enumerate(header_line):
                row[key] = parts[i] if i < len(parts) else ''
            # Convert numeric fields
            for key in ('sustain_score', 'harmony_index',
                        'resilience_quotient', 'fairness',
                        'strategy_div', 'stress_validity'):
                row[key] = _safe_float(row.get(key))
            rows.append(row)
    return rows


def _extract_layer(hypothesis: str) -> str:
    """Extract layer label (e.g. 'L1') from a hypothesis string."""
    h = hypothesis.strip()
    if h.startswith('L') and len(h) > 1 and h[1].isdigit():
        return h[:2]
    return ''


def analyze_autoresearch(path: str):
    """Analyze autoresearch results.tsv."""
    rows = _parse_results_tsv(path)
    if not rows:
        print(f'No data found in {path}')
        return

    # --- Summary statistics ---
    total = len(rows)
    statuses = {}
    for r in rows:
        s = r.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1

    kept = statuses.get('keep', 0)
    reverted = statuses.get('revert', 0)
    skipped = statuses.get('skip', 0)
    errors = statuses.get('error', 0)
    baselines = statuses.get('baseline', 0)
    crash = statuses.get('crash', 0)
    error_rate = (errors + crash) / total if total > 0 else 0.0

    summary_pairs = [
        ('Total iterations:', str(total)),
        ('Baselines:', str(baselines)),
        ('Kept:', str(kept)),
        ('Reverted:', str(reverted)),
        ('Skipped:', str(skipped)),
        ('Errors/crashes:', str(errors + crash)),
        ('Error rate:', f'{error_rate:.1%}'),
    ]

    # --- Best variation ---
    valid = [r for r in rows if r.get('status') in ('keep', 'baseline')]
    if valid:
        best = max(valid, key=lambda r: r['sustain_score'])
    else:
        best = max(rows, key=lambda r: r['sustain_score']) if rows else None

    # --- Layer analysis ---
    layer_data: dict[str, list[float]] = {}
    for r in rows:
        layer = _extract_layer(r.get('hypothesis', ''))
        if layer:
            layer_data.setdefault(layer, []).append(r['sustain_score'])
    layer_summary = []
    for layer in sorted(layer_data.keys()):
        scores = layer_data[layer]
        layer_summary.append({
            'layer': layer,
            'count': len(scores),
            'mean_score': statistics.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
        })

    # --- Score trajectory (best_score over time) ---
    trajectory = []
    best_so_far = 0.0
    for r in rows:
        if r.get('status') in ('keep', 'baseline'):
            best_so_far = max(best_so_far, r['sustain_score'])
        trajectory.append({
            'iteration': len(trajectory) + 1,
            'score': r['sustain_score'],
            'best_so_far': best_so_far,
            'status': r.get('status', ''),
            'hypothesis': r.get('hypothesis', ''),
        })

    # --- Component breakdown ---
    components = ['harmony_index', 'resilience_quotient', 'fairness',
                  'strategy_div', 'stress_validity']
    comp_stats = {}
    for comp in components:
        vals = [r[comp] for r in rows if r.get('status') != 'skip']
        if vals:
            comp_stats[comp] = {
                'mean': statistics.mean(vals),
                'max': max(vals),
                'min': min(vals),
                'stdev': statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

    # --- Sort and top_n ---
    sort_key = FLAGS.sort_by
    display_rows = list(rows)
    if sort_key in ('sustain_score', 'harmony_index', 'resilience_quotient',
                     'fairness', 'strategy_div', 'stress_validity'):
        display_rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)
    if FLAGS.top_n > 0:
        display_rows = display_rows[:FLAGS.top_n]

    # --- Build output ---
    fmt = FLAGS.format

    if fmt == 'json':
        data = {
            'summary': {
                'total': total, 'kept': kept, 'reverted': reverted,
                'skipped': skipped, 'errors': errors + crash,
                'error_rate': error_rate,
            },
            'best_variation': {
                'hypothesis': best.get('hypothesis', '') if best else '',
                'sustain_score': best['sustain_score'] if best else 0.0,
                'commit': best.get('commit', '') if best else '',
            } if best else None,
            'layer_analysis': layer_summary,
            'trajectory': trajectory,
            'component_breakdown': comp_stats,
            'rows': display_rows,
        }
        emit_output('json', data=data)
        return

    if fmt in ('tsv', 'csv'):
        headers = ['timestamp', 'commit', 'sustain_score', 'harmony_index',
                    'resilience_quotient', 'fairness', 'strategy_div',
                    'stress_validity', 'status', 'hypothesis']
        out_rows = []
        for r in display_rows:
            out_rows.append([
                r.get('timestamp', ''),
                r.get('commit', ''),
                _fmt(r['sustain_score']),
                _fmt(r['harmony_index']),
                _fmt(r['resilience_quotient']),
                _fmt(r['fairness']),
                _fmt(r['strategy_div']),
                _fmt(r['stress_validity']),
                r.get('status', ''),
                r.get('hypothesis', ''),
            ])
        emit_output(fmt, headers=headers, rows=out_rows)
        return

    # --- Table format ---
    lines = []
    lines.append('=' * 72)
    lines.append('  AUTORESEARCH RESULTS ANALYSIS')
    lines.append(f'  Source: {path}')
    lines.append('=' * 72)

    # Summary
    lines.append('')
    lines.append('--- Summary ---')
    lines.append(_format_kv(summary_pairs))

    # Best variation
    if best:
        lines.append('')
        lines.append('--- Best Variation ---')
        lines.append(_format_kv([
            ('Hypothesis:', best.get('hypothesis', '')),
            ('SustainScore:', _fmt(best['sustain_score'])),
            ('HI:', _fmt(best['harmony_index'])),
            ('RQ:', _fmt(best['resilience_quotient'])),
            ('Fairness:', _fmt(best['fairness'])),
            ('Strategy Div:', _fmt(best['strategy_div'])),
            ('Stress Valid:', _fmt(best['stress_validity'])),
            ('Commit:', best.get('commit', '')),
        ]))

    # Layer analysis
    if layer_summary:
        lines.append('')
        lines.append('--- Layer Analysis ---')
        lheaders = ['Layer', 'Count', 'Mean SS', 'Max SS', 'Min SS']
        lrows = []
        for ls in layer_summary:
            lrows.append([
                ls['layer'],
                str(ls['count']),
                _fmt(ls['mean_score']),
                _fmt(ls['max_score']),
                _fmt(ls['min_score']),
            ])
        lines.append(_format_table(lheaders, lrows))

    # Score trajectory
    lines.append('')
    lines.append('--- Score Trajectory ---')
    theaders = ['Iter', 'Score', 'Best', 'Status', 'Hypothesis']
    trows = []
    for t in trajectory:
        hyp = t['hypothesis']
        if len(hyp) > 50:
            hyp = hyp[:47] + '...'
        trows.append([
            str(t['iteration']),
            _fmt(t['score']),
            _fmt(t['best_so_far']),
            t['status'],
            hyp,
        ])
    lines.append(_format_table(theaders, trows))

    # Component breakdown
    if comp_stats:
        lines.append('')
        lines.append('--- Component Breakdown ---')
        cheaders = ['Component', 'Mean', 'Max', 'Min', 'StdDev']
        crows = []
        for comp in components:
            if comp in comp_stats:
                cs = comp_stats[comp]
                crows.append([
                    comp,
                    _fmt(cs['mean']),
                    _fmt(cs['max']),
                    _fmt(cs['min']),
                    _fmt(cs['stdev']),
                ])
        lines.append(_format_table(cheaders, crows))

    # All results table (sorted, top_n applied)
    if display_rows:
        lines.append('')
        top_label = f' (top {FLAGS.top_n})' if FLAGS.top_n > 0 else ''
        lines.append(f'--- All Results{top_label} (sorted by {FLAGS.sort_by}) ---')
        aheaders = ['#', 'SS', 'HI', 'RQ', 'Fair', 'StrDiv', 'Status',
                     'Hypothesis']
        arows = []
        for i, r in enumerate(display_rows, 1):
            hyp = r.get('hypothesis', '')
            if len(hyp) > 45:
                hyp = hyp[:42] + '...'
            arows.append([
                str(i),
                _fmt(r['sustain_score']),
                _fmt(r['harmony_index']),
                _fmt(r['resilience_quotient']),
                _fmt(r['fairness']),
                _fmt(r['strategy_div']),
                r.get('status', ''),
                hyp,
            ])
        lines.append(_format_table(aheaders, arows))

    lines.append('')
    emit_output('table', table_text='\n'.join(lines))


# ===================================================================
# Autoresearch comparison
# ===================================================================


def compare_autoresearch(path_a: str, path_b: str):
    """Compare two autoresearch result files side-by-side."""
    rows_a = _parse_results_tsv(path_a)
    rows_b = _parse_results_tsv(path_b)

    if not rows_a:
        print(f'No data in {path_a}')
        return
    if not rows_b:
        print(f'No data in {path_b}')
        return

    def _summary(rows, label):
        total = len(rows)
        kept = sum(1 for r in rows if r.get('status') == 'keep')
        valid = [r for r in rows if r.get('status') in ('keep', 'baseline')]
        best_ss = max((r['sustain_score'] for r in valid), default=0.0)
        mean_ss = statistics.mean(r['sustain_score'] for r in rows) if rows else 0.0
        mean_hi = statistics.mean(r['harmony_index'] for r in rows) if rows else 0.0
        return {
            'label': label,
            'total': total,
            'kept': kept,
            'best_ss': best_ss,
            'mean_ss': mean_ss,
            'mean_hi': mean_hi,
        }

    sa = _summary(rows_a, os.path.basename(path_a))
    sb = _summary(rows_b, os.path.basename(path_b))

    fmt = FLAGS.format

    if fmt == 'json':
        emit_output('json', data={'file_a': sa, 'file_b': sb})
        return

    if fmt in ('tsv', 'csv'):
        headers = ['Metric', sa['label'], sb['label'], 'Delta']
        out_rows = [
            ['Total', str(sa['total']), str(sb['total']),
             str(sb['total'] - sa['total'])],
            ['Kept', str(sa['kept']), str(sb['kept']),
             str(sb['kept'] - sa['kept'])],
            ['Best SS', _fmt(sa['best_ss']), _fmt(sb['best_ss']),
             _fmt(sb['best_ss'] - sa['best_ss'])],
            ['Mean SS', _fmt(sa['mean_ss']), _fmt(sb['mean_ss']),
             _fmt(sb['mean_ss'] - sa['mean_ss'])],
            ['Mean HI', _fmt(sa['mean_hi']), _fmt(sb['mean_hi']),
             _fmt(sb['mean_hi'] - sa['mean_hi'])],
        ]
        emit_output(fmt, headers=headers, rows=out_rows)
        return

    lines = []
    lines.append('=' * 72)
    lines.append('  AUTORESEARCH COMPARISON')
    lines.append(f'  A: {path_a}')
    lines.append(f'  B: {path_b}')
    lines.append('=' * 72)
    lines.append('')

    headers = ['Metric', sa['label'], sb['label'], 'Delta']
    comp_rows = [
        ['Total iterations', str(sa['total']), str(sb['total']),
         f'{sb["total"] - sa["total"]:+d}'],
        ['Kept', str(sa['kept']), str(sb['kept']),
         f'{sb["kept"] - sa["kept"]:+d}'],
        ['Best SustainScore', _fmt(sa['best_ss']), _fmt(sb['best_ss']),
         f'{sb["best_ss"] - sa["best_ss"]:+.4f}'],
        ['Mean SustainScore', _fmt(sa['mean_ss']), _fmt(sb['mean_ss']),
         f'{sb["mean_ss"] - sa["mean_ss"]:+.4f}'],
        ['Mean HI', _fmt(sa['mean_hi']), _fmt(sb['mean_hi']),
         f'{sb["mean_hi"] - sa["mean_hi"]:+.4f}'],
    ]
    lines.append(_format_table(headers, comp_rows))
    lines.append('')

    emit_output('table', table_text='\n'.join(lines))


# ===================================================================
# Ladder analysis
# ===================================================================


def _load_ladder_results(path: str) -> list[dict]:
    """Load ladder_results.json."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def analyze_ladder(path: str):
    """Analyze experiment ladder results."""
    results = _load_ladder_results(path)
    if not results:
        print(f'No ladder data found in {path}')
        return

    # Sort by level
    results.sort(key=lambda r: r.get('level', 0))

    # --- Level comparison table ---
    headers = ['Level', 'Name', 'Mean HI', 'Coverage', 'Diversity',
               'Strat Div', 'Duration(s)']
    rows = []
    for r in results:
        rows.append([
            str(r.get('level', '?')),
            r.get('level_name', ''),
            _fmt(r.get('mean_hi', 0.0)),
            _fmt(r.get('mean_coverage', 0.0)),
            _fmt(r.get('mean_diversity', 0.0)),
            _fmt(r.get('strategy_diversity', 0.0)),
            _fmt(r.get('duration_s', 0.0), 2),
        ])

    # --- Concept impact (deltas) ---
    deltas = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        deltas.append({
            'level': curr.get('level', '?'),
            'new_concept': curr.get('new_concept', ''),
            'delta_hi': curr.get('mean_hi', 0.0) - prev.get('mean_hi', 0.0),
            'delta_coverage': (curr.get('mean_coverage', 0.0)
                               - prev.get('mean_coverage', 0.0)),
            'delta_diversity': (curr.get('mean_diversity', 0.0)
                                - prev.get('mean_diversity', 0.0)),
            'delta_strat_div': (curr.get('strategy_diversity', 0.0)
                                - prev.get('strategy_diversity', 0.0)),
        })

    # --- Statistical summary per level (when multiple seeds via
    #     sprint_history or hi_trajectory) ---
    level_stats = []
    for r in results:
        hi_traj = r.get('hi_trajectory', [])
        if hi_traj:
            level_stats.append({
                'level': r.get('level', '?'),
                'n_sprints': len(hi_traj),
                'hi_mean': statistics.mean(hi_traj),
                'hi_stdev': (statistics.stdev(hi_traj)
                             if len(hi_traj) > 1 else 0.0),
                'hi_min': min(hi_traj),
                'hi_max': max(hi_traj),
            })

    # --- Sort and top_n ---
    sort_key = FLAGS.sort_by
    sort_map = {
        'sustain_score': 'mean_hi',
        'mean_hi': 'mean_hi',
        'coverage': 'mean_coverage',
        'diversity': 'mean_diversity',
        'strategy_diversity': 'strategy_diversity',
    }
    actual_key = sort_map.get(sort_key, 'mean_hi')
    display_results = sorted(results,
                             key=lambda r: r.get(actual_key, 0.0),
                             reverse=True)
    if FLAGS.top_n > 0:
        display_results = display_results[:FLAGS.top_n]

    fmt = FLAGS.format

    if fmt == 'json':
        data = {
            'levels': results,
            'concept_impact': deltas,
            'sprint_statistics': level_stats,
        }
        emit_output('json', data=data)
        return

    if fmt in ('tsv', 'csv'):
        out_rows = []
        for r in display_results:
            out_rows.append([
                str(r.get('level', '?')),
                r.get('level_name', ''),
                _fmt(r.get('mean_hi', 0.0)),
                _fmt(r.get('mean_coverage', 0.0)),
                _fmt(r.get('mean_diversity', 0.0)),
                _fmt(r.get('strategy_diversity', 0.0)),
                _fmt(r.get('duration_s', 0.0), 2),
            ])
        emit_output(fmt, headers=headers, rows=out_rows)
        return

    # --- Table format ---
    lines = []
    lines.append('=' * 90)
    lines.append('  EXPERIMENT LADDER ANALYSIS')
    lines.append(f'  Source: {path}')
    lines.append('=' * 90)

    lines.append('')
    lines.append('--- Level Comparison ---')
    lines.append(_format_table(headers, rows))

    # Concept impact
    if deltas:
        lines.append('')
        lines.append('--- Concept Impact (delta from previous level) ---')
        dheaders = ['Level', 'New Concept', 'd(HI)', 'd(Cov)', 'd(Div)',
                     'd(StratDiv)']
        drows = []
        for d in deltas:
            drows.append([
                str(d['level']),
                d['new_concept'][:35] if len(d['new_concept']) > 35
                else d['new_concept'],
                f'{d["delta_hi"]:+.4f}',
                f'{d["delta_coverage"]:+.4f}',
                f'{d["delta_diversity"]:+.4f}',
                f'{d["delta_strat_div"]:+.4f}',
            ])
        lines.append(_format_table(dheaders, drows))

    # Sprint-level statistics
    if level_stats:
        lines.append('')
        lines.append('--- HI Trajectory Statistics (per level) ---')
        sheaders = ['Level', 'Sprints', 'HI Mean', 'HI StdDev', 'HI Min',
                     'HI Max']
        srows = []
        for s in level_stats:
            srows.append([
                str(s['level']),
                str(s['n_sprints']),
                _fmt(s['hi_mean']),
                _fmt(s['hi_stdev']),
                _fmt(s['hi_min']),
                _fmt(s['hi_max']),
            ])
        lines.append(_format_table(sheaders, srows))

    # HI trajectories (compact sparkline-style)
    lines.append('')
    lines.append('--- HI Trajectories ---')
    for r in results:
        hi_traj = r.get('hi_trajectory', [])
        if hi_traj:
            traj_str = ' -> '.join(_fmt(h) for h in hi_traj)
            lines.append(f'  L{r.get("level", "?")}: {traj_str}')

    lines.append('')
    emit_output('table', table_text='\n'.join(lines))


# ===================================================================
# Sweep (model comparison) analysis
# ===================================================================


def _load_manifest(path: str) -> dict:
    """Load a manifest.json from the overnight runner."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def _load_results_json(output_dir: str) -> dict | None:
    """Load results.json from an experiment output directory."""
    results_path = os.path.join(output_dir, 'results.json')
    if not os.path.exists(results_path):
        return None
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _compute_sustain_score_from_data(data: dict) -> dict:
    """Compute SustainScore components from a results.json dict.

    Mirrors the formula: SS = HI * (1 + RQ) * fairness * strategy_div * stress_validity
    """
    hi = data.get('harmony_index', 0.0)
    rq = data.get('resilience_quotient', 0.0)

    # Fairness (1 - Gini)
    scores = data.get('scores', {})
    fairness = _compute_fairness(scores)

    # Strategy diversity
    sprint_history = data.get('sprint_history', [])
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


def _compute_fairness(scores: dict) -> float:
    """Fairness = 1 - normalized Gini coefficient."""
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


def _compute_strategy_diversity(sprint_history: list) -> float:
    """Fraction of agents that changed task across sprints."""
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


def analyze_sweep(path: str):
    """Analyze overnight runner manifest.json for model sweep comparison."""
    manifest = _load_manifest(path)
    if not manifest:
        print(f'No manifest data found in {path}')
        return

    all_runs = manifest.get('results', [])
    if not all_runs:
        print('Manifest contains no results.')
        return

    # Group concordia runs by model
    model_runs: dict[str, list[dict]] = {}
    ladder_runs: list[dict] = []
    for run in all_runs:
        if run.get('type') == 'concordia':
            model = run.get('model_name', 'unknown')
            model_runs.setdefault(model, []).append(run)
        elif run.get('type') == 'ladder':
            ladder_runs.append(run)

    # For each model, load results.json from output_dir and compute metrics
    model_metrics: dict[str, list[dict]] = {}
    for model, runs in model_runs.items():
        metrics_list = []
        for run in runs:
            out_dir = run.get('output_dir', '')
            data = _load_results_json(out_dir) if out_dir else None
            if data:
                metrics = _compute_sustain_score_from_data(data)
                metrics['duration'] = run.get('duration_s', 0.0)
                metrics['returncode'] = run.get('returncode', -1)
                metrics_list.append(metrics)
            else:
                # Run failed or missing
                metrics_list.append({
                    'sustain_score': 0.0,
                    'harmony_index': 0.0,
                    'resilience_quotient': 0.0,
                    'fairness': 0.0,
                    'strategy_diversity': 0.0,
                    'stress_validity': 0.0,
                    'duration': run.get('duration_s', 0.0),
                    'returncode': run.get('returncode', -1),
                })
        model_metrics[model] = metrics_list

    # Build model comparison summary
    model_summary = []
    for model, metrics_list in model_metrics.items():
        ok_runs = [m for m in metrics_list if m.get('returncode') == 0]
        total_runs = len(metrics_list)
        success_rate = len(ok_runs) / total_runs if total_runs > 0 else 0.0

        if ok_runs:
            mean_hi = statistics.mean(m['harmony_index'] for m in ok_runs)
            mean_ss = statistics.mean(m['sustain_score'] for m in ok_runs)
            mean_fair = statistics.mean(m['fairness'] for m in ok_runs)
            mean_strat = statistics.mean(
                m['strategy_diversity'] for m in ok_runs)
        else:
            mean_hi = mean_ss = mean_fair = mean_strat = 0.0

        total_duration = sum(m.get('duration', 0.0) for m in metrics_list)

        model_summary.append({
            'model': model,
            'total_runs': total_runs,
            'success_rate': success_rate,
            'mean_hi': mean_hi,
            'mean_ss': mean_ss,
            'mean_fairness': mean_fair,
            'mean_strat_div': mean_strat,
            'total_duration': total_duration,
        })

    # Sort by mean_ss by default
    sort_key = FLAGS.sort_by
    sort_map = {
        'sustain_score': 'mean_ss',
        'mean_hi': 'mean_hi',
        'harmony_index': 'mean_hi',
        'fairness': 'mean_fairness',
        'strategy_diversity': 'mean_strat_div',
        'duration': 'total_duration',
        'success_rate': 'success_rate',
    }
    actual_key = sort_map.get(sort_key, 'mean_ss')
    model_summary.sort(key=lambda r: r.get(actual_key, 0.0), reverse=True)

    if FLAGS.top_n > 0:
        model_summary = model_summary[:FLAGS.top_n]

    # --- Best model per metric ---
    best_per_metric = {}
    if model_summary:
        for metric_key, metric_label in [
            ('mean_ss', 'SustainScore'),
            ('mean_hi', 'Harmony Index'),
            ('mean_fairness', 'Fairness'),
            ('mean_strat_div', 'Strategy Diversity'),
            ('success_rate', 'Success Rate'),
        ]:
            best_model = max(model_summary,
                             key=lambda r: r.get(metric_key, 0.0))
            best_per_metric[metric_label] = (
                best_model['model'],
                best_model.get(metric_key, 0.0),
            )

    fmt = FLAGS.format

    if fmt == 'json':
        data = {
            'manifest_summary': {
                'total_duration_s': manifest.get('total_duration_s', 0),
                'total_runs': manifest.get('total_runs', 0),
                'successes': manifest.get('successes', 0),
                'failures': manifest.get('failures', 0),
                'cached': manifest.get('cached', 0),
            },
            'model_comparison': model_summary,
            'best_per_metric': {
                k: {'model': v[0], 'value': v[1]}
                for k, v in best_per_metric.items()
            },
            'ladder_runs': len(ladder_runs),
        }
        emit_output('json', data=data)
        return

    headers = ['Model', 'Runs', 'Success%', 'Mean HI', 'Mean SS',
               'Fairness', 'Strat Div', 'Duration(s)']
    out_rows = []
    for ms in model_summary:
        out_rows.append([
            ms['model'],
            str(ms['total_runs']),
            f'{ms["success_rate"]:.0%}',
            _fmt(ms['mean_hi']),
            _fmt(ms['mean_ss']),
            _fmt(ms['mean_fairness']),
            _fmt(ms['mean_strat_div']),
            _fmt(ms['total_duration'], 1),
        ])

    if fmt in ('tsv', 'csv'):
        emit_output(fmt, headers=headers, rows=out_rows)
        return

    # --- Table format ---
    lines = []
    lines.append('=' * 90)
    lines.append('  MODEL SWEEP ANALYSIS')
    lines.append(f'  Source: {path}')
    lines.append(f'  Total duration: '
                 f'{manifest.get("total_duration_s", 0):.1f}s')
    lines.append(f'  Total runs: {manifest.get("total_runs", 0)} '
                 f'(successes: {manifest.get("successes", 0)}, '
                 f'failures: {manifest.get("failures", 0)}, '
                 f'cached: {manifest.get("cached", 0)})')
    lines.append('=' * 90)

    if model_summary:
        lines.append('')
        lines.append('--- Model Comparison ---')
        lines.append(_format_table(headers, out_rows))

    if best_per_metric:
        lines.append('')
        lines.append('--- Best Model Per Metric ---')
        bheaders = ['Metric', 'Best Model', 'Value']
        brows = []
        for metric_label, (model_name, value) in best_per_metric.items():
            brows.append([metric_label, model_name, _fmt(value)])
        lines.append(_format_table(bheaders, brows))

    if ladder_runs:
        lines.append('')
        lines.append(f'  (Also contains {len(ladder_runs)} ladder experiment '
                     f'runs; use --mode=ladder to analyze those.)')

    lines.append('')
    emit_output('table', table_text='\n'.join(lines))


# ===================================================================
# Auto-detection
# ===================================================================


def _auto_detect_input(mode: str) -> str:
    """Auto-detect the input path based on mode."""
    if mode == 'autoresearch':
        candidates = [
            'results.tsv',
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'results.tsv'),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return 'results.tsv'  # Will show "no data" message

    elif mode == 'ladder':
        return '/tmp/sustain_hub_experiments/ladder_results.json'

    elif mode == 'sweep':
        return '/tmp/sustainhub_overnight/manifest.json'

    return ''


# ===================================================================
# Main
# ===================================================================


def main(argv):
    del argv

    mode = FLAGS.mode
    input_path = FLAGS.input or _auto_detect_input(mode)

    # Handle comparison mode
    if FLAGS.compare and mode == 'autoresearch':
        compare_autoresearch(FLAGS.compare, input_path)
        return

    if mode == 'autoresearch':
        analyze_autoresearch(input_path)
    elif mode == 'ladder':
        analyze_ladder(input_path)
    elif mode == 'sweep':
        analyze_sweep(input_path)


if __name__ == '__main__':
    app.run(main)
