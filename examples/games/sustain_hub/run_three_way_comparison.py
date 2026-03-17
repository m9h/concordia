#!/usr/bin/env python3
"""Three-Way Comparison Harness: Rohira × LLAMOSC × Concordia SustainHub.

Orchestrates running all three open-source community simulations and
produces a unified comparison table with normalized metrics.

Usage:
    # Show results from all collected data
    python run_three_way_comparison.py --results

    # Run all experiments that have runners set up
    python run_three_way_comparison.py --run=all

    # Run specific experiments
    python run_three_way_comparison.py --run=rohira_baseline,llamosc_auth

    # List available experiments
    python run_three_way_comparison.py --list

    # Export comparison CSV
    python run_three_way_comparison.py --results --csv=comparison.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / 'results'
ROHIRA_DIR = Path.home() / 'dev' / 'rohira-sustainhub'
LLAMOSC_DIR = Path.home() / 'dev' / 'llamosc'
CONCORDIA_DIR = THIS_DIR.parent.parent.parent  # repo root

# ── Experiment Definitions ────────────────────────────────────────────

EXPERIMENTS = {
    # Rohira baseline
    'rohira_baseline': {
        'system': 'rohira',
        'description': 'Rohira RL (SARSA+MAB), 10 agents, 10 steps, no dropout',
        'runner': str(ROHIRA_DIR / 'run_comparison.py'),
        'args': ['--agents=10', '--steps=10', '--seeds=5'],
        'result_file': 'rohira_baseline.json',
    },
    'rohira_dropout': {
        'system': 'rohira',
        'description': 'Rohira RL with 2 dropouts/step',
        'runner': str(ROHIRA_DIR / 'run_comparison.py'),
        'args': ['--agents=10', '--steps=10', '--seeds=5', '--dropouts_per_step=2'],
        'result_file': 'rohira_dropout.json',
    },

    # LLAMOSC
    'llamosc_auth': {
        'system': 'llamosc',
        'description': 'LLAMOSC authoritarian governance (testing mode)',
        'runner': str(LLAMOSC_DIR / 'run_comparison.py'),
        'args': ['--algorithm=a', '--seeds=5', '--mode=testing'],
        'result_file': 'llamosc_authoritarian.json',
    },
    'llamosc_decentral': {
        'system': 'llamosc',
        'description': 'LLAMOSC decentralized governance (testing mode)',
        'runner': str(LLAMOSC_DIR / 'run_comparison.py'),
        'args': ['--algorithm=d', '--seeds=5', '--mode=testing'],
        'result_file': 'llamosc_decentralized.json',
    },
    'llamosc_auth_nim': {
        'system': 'llamosc',
        'description': 'LLAMOSC authoritarian + NIM cloud LLM',
        'runner': str(LLAMOSC_DIR / 'run_comparison.py'),
        'args': ['--algorithm=a', '--seeds=5', '--mode=nim'],
        'result_file': 'llamosc_auth_nim.json',
    },

    # Concordia (pre-collected results)
    'concordia_free': {
        'system': 'concordia',
        'description': 'Concordia LLM+AIF, free choice governance',
        'runner': None,  # Results already collected
        'result_file': 'concordia_B_governance.json',
        'result_key': 'B1',
    },
    'concordia_dictator': {
        'system': 'concordia',
        'description': 'Concordia LLM+AIF, dictator governance',
        'runner': None,
        'result_file': 'concordia_B_governance.json',
        'result_key': 'B2',
    },
    'concordia_meritocratic': {
        'system': 'concordia',
        'description': 'Concordia LLM+AIF, meritocratic governance',
        'runner': None,
        'result_file': 'concordia_B_governance.json',
        'result_key': 'B3',
    },
}


# ── Runner Functions ──────────────────────────────────────────────────

def run_experiment(name, exp):
    """Run a single experiment via subprocess."""
    runner = exp.get('runner')
    if not runner or not os.path.exists(runner):
        print(f'  [{name}] Skipped (runner not found: {runner})')
        return None

    result_file = RESULTS_DIR / exp['result_file']
    args = exp.get('args', [])

    # Determine venv python
    system = exp['system']
    if system == 'rohira':
        python = str(ROHIRA_DIR / '.venv' / 'bin' / 'python')
        cwd = str(ROHIRA_DIR)
    elif system == 'llamosc':
        python = str(LLAMOSC_DIR / '.venv' / 'bin' / 'python')
        cwd = str(LLAMOSC_DIR)
    else:
        python = sys.executable
        cwd = str(CONCORDIA_DIR)

    if not os.path.exists(python):
        print(f'  [{name}] Skipped (venv not found: {python})')
        return None

    cmd = [python, runner] + args + [f'--output={result_file}']
    print(f'  [{name}] Running: {" ".join(cmd[:3])}...')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=cwd,
        )
        if result.returncode != 0:
            print(f'  [{name}] FAILED (exit {result.returncode})')
            if result.stderr:
                print(f'    {result.stderr[:200]}')
            return None
        print(f'  [{name}] OK')
        return result_file
    except subprocess.TimeoutExpired:
        print(f'  [{name}] TIMEOUT')
        return None


# ── Result Loading ────────────────────────────────────────────────────

def load_result(name, exp):
    """Load results for an experiment, returning a normalized dict."""
    result_file = RESULTS_DIR / exp['result_file']
    if not result_file.exists():
        return None

    with open(result_file) as f:
        data = json.load(f)

    # Handle Concordia B-series (multiple experiments in one file)
    result_key = exp.get('result_key')
    if result_key and isinstance(data, dict) and result_key in data:
        d = data[result_key]
        return {
            'name': name,
            'system': 'concordia',
            'description': exp['description'],
            'hi': d.get('harmony_index_mean', d.get('chs_mean')),
            'hi_std': d.get('harmony_index_std', d.get('chs_std')),
            'rq': d.get('resilience_quotient_mean'),
            'brs': d.get('mean_brs_mean'),
            'sue': d.get('sue_mean'),
            'chs': d.get('chs_mean'),
            'bai': d.get('mean_bai_mean', d.get('mean_bai')),
            'trust_recip': d.get('trust_reciprocity_mean', d.get('trust_reciprocity')),
            'norm_emerg': d.get('norm_emergence_rate_mean', d.get('norm_emergence_rate')),
            'coord_idx': d.get('coordination_index_mean', d.get('coordination_index')),
            'burnout': d.get('mean_burnout_mean', d.get('mean_burnout')),
            'seeds': d.get('num_ok', d.get('num_seeds')),
            'raw': d,
        }

    system = data.get('system', exp['system'])
    summary = data.get('summary', {})

    if system == 'rohira':
        return {
            'name': name,
            'system': 'rohira',
            'description': exp['description'],
            'hi': summary.get('hi_mean'),
            'hi_std': summary.get('hi_std'),
            'rq': summary.get('rq_mean'),
            'brs': None,
            'sue': None,
            'chs': None,
            'ro': summary.get('ro_mean'),
            'seeds': data.get('config', {}).get('seeds'),
            'raw': summary,
        }
    elif system == 'llamosc':
        return {
            'name': name,
            'system': 'llamosc',
            'description': exp['description'],
            'hi': summary.get('hi_equivalent_mean'),
            'hi_std': summary.get('hi_equivalent_std'),
            'rq': None,
            'brs': None,
            'sue': None,
            'chs': None,
            'experience': summary.get('experience_mean'),
            'motivation': summary.get('motivation_mean'),
            'code_quality': summary.get('code_quality_mean'),
            'seeds': data.get('config', {}).get('seeds'),
            'raw': summary,
        }
    else:
        return {
            'name': name,
            'system': system,
            'description': exp['description'],
            'raw': summary,
            'seeds': data.get('config', {}).get('seeds'),
        }


# ── Display Functions ─────────────────────────────────────────────────

def print_comparison_table(results):
    """Print a formatted comparison table."""
    print()
    print('=' * 100)
    print('  THREE-WAY COMPARISON: Rohira (RL) × LLAMOSC (LLM) × Concordia (LLM+AIF)')
    print('=' * 100)
    print()

    # Header
    fmt = '{:<25s} {:>8s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}'
    print(fmt.format('Experiment', 'System', 'HI(±std)', 'RQ', 'BRS', 'SUE', 'CHS', 'Seeds'))
    print('-' * 100)

    for r in results:
        if r is None:
            continue

        hi_str = f"{r.get('hi', 0):.3f}" if r.get('hi') is not None else '—'
        if r.get('hi_std') is not None:
            hi_str += f"±{r['hi_std']:.3f}"

        rq_str = f"{r['rq']:.3f}" if r.get('rq') is not None else '—'
        brs_str = f"{r['brs']:.3f}" if r.get('brs') is not None else '—'
        sue_str = f"{r['sue']:.3f}" if r.get('sue') is not None else '—'
        chs_str = f"{r['chs']:.3f}" if r.get('chs') is not None else '—'
        seeds_str = str(r.get('seeds', '—'))

        print(fmt.format(r['name'], r['system'], hi_str, rq_str, brs_str, sue_str, chs_str, seeds_str))

        # Show LLAMOSC-specific metrics on second line
        if r['system'] == 'llamosc':
            exp_str = f"Exp={r.get('experience', 0):.2f}" if r.get('experience') else ''
            mot_str = f"Mot={r.get('motivation', 0):.2f}" if r.get('motivation') else ''
            cq_str = f"CQ={r.get('code_quality', 0):.2f}" if r.get('code_quality') else ''
            print(f"{'':25s} {'':>8s} {exp_str:>10s} {mot_str:>8s} {cq_str:>8s}")

        # Show Rohira RO metric
        if r['system'] == 'rohira' and r.get('ro') is not None:
            ro_val = r['ro']
            print(f"{'':25s} {'':>8s} {'RO='+f'{ro_val:.3f}':>10s}")

        # Show Concordia-specific enriched metrics
        if r['system'] == 'concordia':
            extras = []
            if r.get('bai') is not None:
                extras.append(f"BAI={r['bai']:.3f}")
            if r.get('trust_recip') is not None:
                extras.append(f"TrustRecip={r['trust_recip']:.3f}")
            if r.get('norm_emerg') is not None:
                extras.append(f"NormEmerg={r['norm_emerg']:.2f}")
            if r.get('coord_idx') is not None:
                extras.append(f"CoordIdx={r['coord_idx']:.3f}")
            if r.get('burnout') is not None:
                extras.append(f"Burnout={r['burnout']:.3f}")
            if extras:
                print(f"{'':25s} {'':>8s} {' '.join(extras)}")

    print('-' * 100)
    print()


def export_csv(results, csv_path):
    """Export comparison results to CSV."""
    fieldnames = ['name', 'system', 'description', 'hi', 'hi_std', 'rq',
                  'brs', 'sue', 'chs', 'bai', 'trust_recip', 'norm_emerg',
                  'coord_idx', 'burnout', 'seeds']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            if r is not None:
                writer.writerow(r)
    print(f'  CSV exported to: {csv_path}')


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Three-way comparison harness')
    parser.add_argument('--run', type=str, default=None,
                        help='Experiments to run (comma-separated, or "all")')
    parser.add_argument('--results', action='store_true',
                        help='Show results from collected data')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    parser.add_argument('--csv', type=str, default=None,
                        help='Export comparison table to CSV')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.list:
        print('\nAvailable experiments:')
        for name, exp in EXPERIMENTS.items():
            runner_status = 'ready' if exp.get('runner') and os.path.exists(exp['runner']) else 'pre-collected' if not exp.get('runner') else 'not set up'
            result_exists = (RESULTS_DIR / exp['result_file']).exists()
            result_status = 'has results' if result_exists else 'no results'
            print(f'  {name:<25s} [{exp["system"]:>10s}] [{runner_status:>14s}] [{result_status}]')
            print(f'    {exp["description"]}')
        return

    if args.run:
        names = list(EXPERIMENTS.keys()) if args.run == 'all' else args.run.split(',')
        print(f'\nRunning {len(names)} experiments...\n')
        for name in names:
            exp = EXPERIMENTS.get(name)
            if not exp:
                print(f'  [{name}] Unknown experiment, skipping')
                continue
            if not exp.get('runner'):
                print(f'  [{name}] Pre-collected results, skipping run')
                continue
            run_experiment(name, exp)
        print()

    if args.results or args.csv or args.run:
        results = []
        for name, exp in EXPERIMENTS.items():
            r = load_result(name, exp)
            if r is not None:
                results.append(r)

        if not results:
            print('No results found. Run experiments first or check results/ directory.')
            return

        print_comparison_table(results)

        if args.csv:
            export_csv(results, args.csv)

    if not (args.run or args.results or args.csv or args.list):
        parser.print_help()


if __name__ == '__main__':
    main()
