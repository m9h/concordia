"""Side-by-side comparison of LLAMOSC and Concordia SustainHub metrics.

Computes Concordia-compatible metrics from LLAMOSC output and prints them
next to the Concordia experiment ladder results.

Usage:
    uv run llamosc-compare --llamosc-results output/llamosc_results.json
    uv run llamosc-compare --llamosc-results output/llamosc_results.json \
        --concordia-results ../vidhi_baseline_results.json
    uv run llamosc-compare --run-first --test --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


def load_concordia_results(path: str) -> dict[str, Any] | None:
    """Load Concordia experiment ladder results from JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison_table(
    llamosc_metrics: dict[str, Any],
    concordia_data: dict[str, Any] | None,
) -> None:
    """Print a formatted side-by-side comparison table."""

    sep = "=" * 78
    header_sep = "-" * 78

    print(f"\n{sep}")
    print("  LLAMOSC vs Concordia SustainHub  --  Metric Comparison")
    print(sep)

    # Determine which Concordia levels to show
    levels: dict[str, dict[str, Any]] = {}
    if concordia_data and "ladder_results" in concordia_data:
        levels = concordia_data["ladder_results"]

    # Core metrics row format
    metrics_to_compare = [
        ("Harmony Index (mean)", "mean_hi"),
        ("Harmony Index (final)", "final_hi"),
        ("Resilience Quotient", "resilience_quotient"),
        ("Fairness", "fairness"),
        ("Strategy Diversity", "strategy_diversity"),
        ("Coverage", "mean_coverage"),
        ("SustainScore", "sustain_score"),
    ]

    # Build column headers
    col_width = 12
    label_width = 24
    columns = ["LLAMOSC"]
    col_values: list[dict[str, float]] = [llamosc_metrics]

    for level_key in sorted(levels.keys()):
        level_data = levels[level_key]
        columns.append(level_data.get("level_name", level_key)[:col_width])
        col_values.append(level_data)

    # Print header
    header = f"{'Metric':<{label_width}}"
    for col in columns:
        header += f"  {col:>{col_width}}"
    print(f"\n{header}")
    print(header_sep)

    # Print each metric row
    for label, key in metrics_to_compare:
        row = f"{label:<{label_width}}"
        for col_data in col_values:
            val = col_data.get(key)
            if val is not None:
                row += f"  {val:>{col_width}.4f}"
            else:
                row += f"  {'N/A':>{col_width}}"
        print(row)

    print(header_sep)

    # Print LLAMOSC-only extras
    print(f"\n{'LLAMOSC-specific metrics':}")
    print(header_sep)

    native = llamosc_metrics.get("llamosc_native", {})
    print(f"  {'Solve rate:':<22} {native.get('solve_rate', 0):.2%}")
    print(f"  {'Avg code quality:':<22} {native.get('avg_code_quality', 0):.2f} / 5")
    print(f"  {'Total issues:':<22} {native.get('total_issues', 0)}")
    print(f"  {'Contributors:':<22} {native.get('n_contributors', 0)}")

    # GSoC metrics
    gsoc = llamosc_metrics.get("gsoc_metrics", {})
    if gsoc:
        print(f"\n{'GSoC 2025 proposed metrics':}")
        print(header_sep)
        print(f"  {'Mean BRS:':<22} {gsoc.get('mean_brs', 0):.4f}  (lower is better)")
        print(f"  {'SUE:':<22} {gsoc.get('sue', 0):.4f}  (higher is better)")
        print(f"  {'CHS:':<22} {gsoc.get('chs', 0):.4f}  (higher is better)")
        brs_per_agent = gsoc.get("brs_per_agent", {})
        if brs_per_agent:
            print(f"  {'BRS per agent:'}")
            for agent, brs in sorted(brs_per_agent.items()):
                print(f"    {agent:<20} {brs:.4f}")

    # HI trajectory
    hi_traj = llamosc_metrics.get("hi_trajectory", [])
    if hi_traj:
        print(f"\n{'HI trajectory (LLAMOSC)':}")
        print(header_sep)
        for i, hi in enumerate(hi_traj):
            bar = "#" * int(hi * 40)
            print(f"  Sprint {i:<3} | {hi:.4f} |{bar}")

    print(f"\n{sep}\n")


def run_simulation(args: argparse.Namespace, output_dir: str) -> str:
    """Run LLAMOSC simulation and return path to results JSON."""
    cmd = ["uv", "run", "llamosc-headless"]
    if args.test:
        cmd.append("--test")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.contributors:
        cmd.extend(["--contributors", str(args.contributors)])
    if args.issues:
        cmd.extend(["--issues", str(args.issues)])
    cmd.extend(["--output-dir", output_dir])

    print(f"Running LLAMOSC: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    duration = time.time() - t0

    if result.returncode != 0:
        print(f"LLAMOSC failed ({duration:.1f}s):", file=sys.stderr)
        print(result.stderr or result.stdout, file=sys.stderr)
        sys.exit(1)

    print(f"LLAMOSC completed in {duration:.1f}s")
    print(result.stdout)

    results_path = os.path.join(output_dir, "llamosc_results.json")
    if not os.path.exists(results_path):
        print(f"Expected output not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    return results_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLAMOSC and Concordia SustainHub metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run llamosc-compare --llamosc-results output/llamosc_results.json
  uv run llamosc-compare --run-first --test --seed 42
  uv run llamosc-compare --llamosc-results output/llamosc_results.json \\
      --concordia-results ../vidhi_baseline_results.json
        """,
    )
    parser.add_argument(
        "--llamosc-results", type=str, default=None,
        help="Path to LLAMOSC results JSON (skip if --run-first is set)",
    )
    parser.add_argument(
        "--concordia-results", type=str, default=None,
        help="Path to Concordia experiment ladder results JSON "
             "(default: auto-detect vidhi_baseline_results.json or precomputed_ladder_results.json)",
    )
    parser.add_argument(
        "--run-first", action="store_true",
        help="Run the LLAMOSC simulation before comparing",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Use test mode when running simulation (--run-first only)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for simulation (--run-first only)",
    )
    parser.add_argument(
        "--contributors", type=int, default=None,
        help="Number of contributors (--run-first only)",
    )
    parser.add_argument(
        "--issues", type=int, default=None,
        help="Number of issues (--run-first only)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for simulation results (--run-first only)",
    )
    parser.add_argument(
        "--issues-per-sprint", type=int, default=1,
        help="Number of LLAMOSC issues per Concordia sprint (default: 1)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write LLAMOSC Concordia metrics to this JSON file",
    )

    args = parser.parse_args()

    # Resolve base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Step 1: Get LLAMOSC results
    if args.run_first:
        output_dir = args.output_dir or os.path.join(base_dir, "output_compare")
        llamosc_path = run_simulation(args, output_dir)
    elif args.llamosc_results:
        llamosc_path = args.llamosc_results
        if not os.path.isabs(llamosc_path):
            llamosc_path = os.path.join(os.getcwd(), llamosc_path)
    else:
        # Try default location
        llamosc_path = os.path.join(base_dir, "output", "llamosc_results.json")
        if not os.path.exists(llamosc_path):
            print("No LLAMOSC results found. Use --llamosc-results or --run-first.", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(llamosc_path):
        print(f"LLAMOSC results not found: {llamosc_path}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Compute Concordia metrics from LLAMOSC output
    from llamosc.concordia_metrics import load_and_compute
    print(f"Computing Concordia metrics from: {llamosc_path}")
    llamosc_metrics = load_and_compute(llamosc_path, issues_per_sprint=args.issues_per_sprint)

    # Step 3: Load Concordia results for comparison
    concordia_data = None
    if args.concordia_results:
        concordia_path = args.concordia_results
        if not os.path.isabs(concordia_path):
            concordia_path = os.path.join(os.getcwd(), concordia_path)
    else:
        # Auto-detect
        parent_dir = os.path.dirname(base_dir)
        candidates = [
            os.path.join(parent_dir, "vidhi_baseline_results.json"),
            os.path.join(parent_dir, "precomputed_ladder_results.json"),
            os.path.join(base_dir, "vidhi_baseline_results.json"),
        ]
        concordia_path = None
        for c in candidates:
            if os.path.exists(c):
                concordia_path = c
                break

    if concordia_path and os.path.exists(concordia_path):
        print(f"Loading Concordia results from: {concordia_path}")
        concordia_data = load_concordia_results(concordia_path)
    else:
        print("No Concordia results found for comparison (LLAMOSC metrics only).")

    # Step 4: Print comparison
    print_comparison_table(llamosc_metrics, concordia_data)

    # Step 5: Optionally save LLAMOSC metrics
    output_json = args.output_json
    if output_json:
        if not os.path.isabs(output_json):
            output_json = os.path.join(os.getcwd(), output_json)
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(llamosc_metrics, f, indent=2)
        print(f"LLAMOSC Concordia metrics saved to: {output_json}")


if __name__ == "__main__":
    main()
