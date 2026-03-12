#!/usr/bin/env python3
"""
Aggregate results from multiple simulation runs.

This script scans a directory for results.json files, extracts metrics
(like harmony_index, resilience_quotient, scores, and sprint_history),
and computes comparison statistics across different models.
"""

import argparse
import glob
import json
import os
import statistics
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/tmp/sustainhub_overnight",
        help="Directory containing simulation results to aggregate.",
    )
    return parser.parse_args()

def extract_model_name(dir_name):
    """
    Extracts the model name from a directory name formatted as {model}_seed_{N}.
    If the pattern doesn't match perfectly, it tries to drop the _seed_{N} part.
    """
    parts = dir_name.split("_seed_")
    if len(parts) >= 2:
        return "_seed_".join(parts[:-1])
    return dir_name

def main():
    args = parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        sys.exit(1)

    pattern = os.path.join(results_dir, "**", "results.json")
    results_files = glob.glob(pattern, recursive=True)

    if not results_files:
        print(f"No results.json files found in {results_dir}")
        sys.exit(0)

    # Dictionary to hold all aggregated data
    # Format: { model_name: [ { run_data }, ... ] }
    model_data = {}

    for file_path in results_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue

        # Parent directory name
        parent_dir = os.path.basename(os.path.dirname(file_path))
        model_name = extract_model_name(parent_dir)

        # Extract required metrics
        # Default to None if not present in the JSON
        harmony_index = data.get("harmony_index")
        resilience_quotient = data.get("resilience_quotient")
        scores = data.get("scores", {})
        sprint_history = data.get("sprint_history", [])

        if model_name not in model_data:
            model_data[model_name] = []

        run_info = {
            "harmony_index": harmony_index,
            "resilience_quotient": resilience_quotient,
            "scores": scores,
            "sprint_history": sprint_history,
            "file_path": file_path,
            "parent_dir": parent_dir
        }
        model_data[model_name].append(run_info)

    # Compute statistics and print table
    print(f"{'Model':<30} | {'Runs':<5} | {'Mean HI':<8} | {'Std HI':<8} | {'Mean Resilience':<15} | {'Min HI':<8} | {'Max HI':<8}")
    print("-" * 100)

    # Data for plotting
    plot_data = [] # List of tuples: (model_name, [harmony_indices])

    # Structure for the combined JSON
    combined_json = {
        "models": model_data,
        "summary": {}
    }

    for model, runs in model_data.items():
        hi_values = [r["harmony_index"] for r in runs if r.get("harmony_index") is not None]
        rq_values = [r["resilience_quotient"] for r in runs if r.get("resilience_quotient") is not None]

        num_runs = len(runs)

        if hi_values:
            mean_hi = statistics.mean(hi_values)
            min_hi = min(hi_values)
            max_hi = max(hi_values)
            std_hi = statistics.stdev(hi_values) if len(hi_values) > 1 else 0.0
        else:
            mean_hi = min_hi = max_hi = std_hi = float('nan')

        if rq_values:
            mean_rq = statistics.mean(rq_values)
        else:
            mean_rq = float('nan')

        plot_data.append((model, hi_values))

        summary_stats = {
            "runs": num_runs,
            "mean_hi": mean_hi,
            "std_hi": std_hi,
            "mean_resilience": mean_rq,
            "min_hi": min_hi,
            "max_hi": max_hi
        }
        combined_json["summary"][model] = summary_stats

        # Print row
        def fmt(val):
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return str(val)

        print(f"{model:<30} | {num_runs:<5} | {fmt(mean_hi):<8} | {fmt(std_hi):<8} | {fmt(mean_rq):<15} | {fmt(min_hi):<8} | {fmt(max_hi):<8}")

    # Save combined JSON
    out_json_path = os.path.join(results_dir, "comparison.json")
    try:
        with open(out_json_path, "w") as f:
            json.dump(combined_json, f, indent=2)
        print(f"\nSaved combined data to {out_json_path}")
    except Exception as e:
        print(f"\nFailed to save {out_json_path}: {e}")

    # Optionally save a box plot
    try:
        import matplotlib.pyplot as plt
        if plot_data:
            # Sort by model name for consistent plotting
            plot_data.sort(key=lambda x: x[0])
            labels = [x[0] for x in plot_data]
            data_to_plot = [x[1] for x in plot_data]

            plt.figure(figsize=(10, 6))
            plt.boxplot(data_to_plot, labels=labels)
            plt.title('Harmony Index by Model')
            plt.ylabel('Harmony Index (HI)')
            plt.xlabel('Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            out_plot_path = os.path.join(results_dir, "model_comparison.png")
            plt.savefig(out_plot_path)
            print(f"Saved model comparison plot to {out_plot_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping box plot generation.")
    except Exception as e:
        print(f"\nFailed to generate plot: {e}")

if __name__ == "__main__":
    main()
