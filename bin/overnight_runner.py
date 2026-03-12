#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overnight batch runner for DGX Spark or any machine with local vLLM.

Runs two types of experiments:
  1. Standalone experiment ladder (no LLM, pure AIF math) - fast, many seeds
  2. Full Concordia simulations (needs LLM via vLLM or Vertex AI) - slower

Usage:
  # Run just the ladder (no GPU/LLM needed):
  python bin/overnight_runner.py --mode=ladder

  # Run full Concordia sims against local vLLM:
  python bin/overnight_runner.py --mode=concordia --vllm_url=http://localhost:8000/v1

  # Run both:
  python bin/overnight_runner.py --mode=all --vllm_url=http://localhost:8000/v1

  # Use Vertex AI instead of vLLM:
  python bin/overnight_runner.py --mode=concordia --project=dlab-host-24819-1188
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time


def run_ladder_experiment(level: int, seed: int, num_sprints: int,
                          output_dir: str) -> dict:
    """Run a single standalone experiment ladder level."""
    level_dir = os.path.join(output_dir, f"level_{level}_seed_{seed}")
    os.makedirs(level_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "examples.games.sustain_hub.experiments",
        f"--level={level}",
        f"--num_sprints={num_sprints}",
        f"--seed={seed}",
        f"--output_dir={level_dir}",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0

    return {
        "type": "ladder",
        "level": level,
        "seed": seed,
        "duration_s": round(duration, 2),
        "returncode": result.returncode,
        "output_dir": level_dir,
        "stderr": result.stderr[-500:] if result.returncode != 0 else "",
    }


def run_concordia_sim(seed: int, num_sprints: int, community_size: int,
                      output_dir: str, vllm_url: str | None = None,
                      project: str | None = None,
                      model_name: str = "gemini-2.0-flash") -> dict:
    """Run a full Concordia simulation."""
    sim_dir = os.path.join(output_dir, f"concordia_seed_{seed}")
    os.makedirs(sim_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "examples.games.sustain_hub.run",
        f"--num_sprints={num_sprints}",
        f"--community_size={community_size}",
        f"--output_dir={sim_dir}",
        "--fast",
        "--skip_backstory",
    ]

    if vllm_url:
        cmd.extend([
            f"--model_name={model_name}",
            f"--vllm_url={vllm_url}",
        ])
    elif project:
        cmd.extend([
            f"--model_name={model_name}",
            f"--project={project}",
        ])
    else:
        cmd.append("--use_mock")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    duration = time.time() - t0

    return {
        "type": "concordia",
        "seed": seed,
        "duration_s": round(duration, 2),
        "returncode": result.returncode,
        "output_dir": sim_dir,
        "stderr": result.stderr[-500:] if result.returncode != 0 else "",
    }


def run_ladder_batch(args):
    """Run the full experiment ladder across multiple seeds."""
    print(f"=== Ladder Batch: {8} levels x {args.num_seeds} seeds x "
          f"{args.num_sprints} sprints ===")

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.ladder_workers
    ) as executor:
        futures = {}
        for level in range(8):
            for i in range(args.num_seeds):
                seed = args.base_seed + i
                future = executor.submit(
                    run_ladder_experiment,
                    level=level,
                    seed=seed,
                    num_sprints=args.num_sprints,
                    output_dir=os.path.join(args.output_dir, "ladder"),
                )
                futures[future] = (level, seed)

        for future in concurrent.futures.as_completed(futures):
            level, seed = futures[future]
            try:
                result = future.result()
                status = "OK" if result["returncode"] == 0 else "FAIL"
                print(f"  L{level} seed={seed}: {status} "
                      f"({result['duration_s']}s)")
                results.append(result)
            except Exception as e:
                print(f"  L{level} seed={seed}: ERROR {e}")
                results.append({
                    "type": "ladder", "level": level, "seed": seed,
                    "error": str(e),
                })

    return results


def run_concordia_batch(args):
    """Run multiple Concordia simulations."""
    print(f"=== Concordia Batch: {args.num_concordia_runs} runs, "
          f"{args.community_size} agents, {args.num_sprints} sprints ===")

    if args.vllm_url:
        print(f"  Backend: vLLM at {args.vllm_url}")
    elif args.project:
        print(f"  Backend: Vertex AI ({args.project})")
    else:
        print("  Backend: Mock model (no LLM)")

    results = []
    # Run Concordia sims sequentially (each is already LLM-bound)
    for i in range(args.num_concordia_runs):
        seed = args.base_seed + i
        print(f"  Starting sim {i+1}/{args.num_concordia_runs} (seed={seed})...")
        try:
            result = run_concordia_sim(
                seed=seed,
                num_sprints=args.num_sprints,
                community_size=args.community_size,
                output_dir=os.path.join(args.output_dir, "concordia"),
                vllm_url=args.vllm_url,
                project=args.project,
                model_name=args.model_name,
            )
            status = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  Sim {i+1}: {status} ({result['duration_s']}s)")
            results.append(result)
        except subprocess.TimeoutExpired:
            print(f"  Sim {i+1}: TIMEOUT (>30min)")
            results.append({
                "type": "concordia", "seed": seed, "error": "timeout",
            })
        except Exception as e:
            print(f"  Sim {i+1}: ERROR {e}")
            results.append({
                "type": "concordia", "seed": seed, "error": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Overnight batch runner for SustainHub experiments"
    )
    parser.add_argument("--mode", choices=["ladder", "concordia", "all"],
                        default="all")
    parser.add_argument("--output_dir", default="/tmp/sustainhub_overnight")
    parser.add_argument("--num_sprints", type=int, default=10)
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Seeds for ladder experiments")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--ladder_workers", type=int, default=8,
                        help="Parallel workers for ladder (CPU-bound)")
    parser.add_argument("--num_concordia_runs", type=int, default=5)
    parser.add_argument("--community_size", type=int, default=8)
    parser.add_argument("--model_name", default="gemini-2.0-flash")

    # Backend selection (mutually exclusive in practice)
    parser.add_argument("--vllm_url", default=None,
                        help="vLLM API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--project", default=None,
                        help="GCP project for Vertex AI")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SustainHub Overnight Batch Runner")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    all_results = []
    t0 = time.time()

    if args.mode in ("ladder", "all"):
        ladder_results = run_ladder_batch(args)
        all_results.extend(ladder_results)

    if args.mode in ("concordia", "all"):
        concordia_results = run_concordia_batch(args)
        all_results.extend(concordia_results)

    total_time = time.time() - t0

    # Save manifest
    manifest = {
        "total_duration_s": round(total_time, 1),
        "total_runs": len(all_results),
        "successes": sum(1 for r in all_results if r.get("returncode") == 0),
        "failures": sum(1 for r in all_results if r.get("returncode", -1) != 0),
        "results": all_results,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time/60:.1f} minutes")
    print(f"  Successes: {manifest['successes']}/{manifest['total_runs']}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
