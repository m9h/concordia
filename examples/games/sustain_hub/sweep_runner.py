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

"""Batch runner for SustainHub parameter sweeps."""

import json
import os
import subprocess
import concurrent.futures
import sys
from typing import Any, Dict, List

def run_simulation(
    run_id: int,
    api_key: str,
    model_name: str,
    num_sprints: int,
    seed: int,
    community_size: int = 8,
    use_mock: bool = False
) -> Dict[str, Any]:
    output_dir = f"/tmp/sustain_hub_sweep/run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "./.venv/bin/python3", "-m", "examples.games.sustain_hub.run",
        f"--model_name={model_name}",
        f"--num_sprints={num_sprints}",
        f"--community_size={community_size}",
        "--verbose",
        f"--output_dir={output_dir}",
    ]

    if use_mock:
        cmd.append("--use_mock")

    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:."

    print(f"Starting Run {run_id} (Seed: {seed})...", flush=True)

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    output_lines = []
    for line in process.stdout:
        print(f"[Run {run_id}] {line}", end="", flush=True)
        output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        print(f"Run {run_id} failed with exit code {process.returncode}", flush=True)
        return {"run_id": run_id, "status": "failed"}

    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
            data["run_id"] = run_id
            data["status"] = "success"
            return data

    return {"run_id": run_id, "status": "not_found"}

def main():
    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    MODEL = "gemini-2.5-flash"
    NUM_SPRINTS = 2
    NUM_RUNS = 1
    COMMUNITY_SIZE = 6


    print("="*60)
    print(f"SustainHub Real Test: {COMMUNITY_SIZE} agents, {NUM_SPRINTS} sprints")
    print("="*60, flush=True)


    sweep_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_RUNS) as executor:
        futures = [
            executor.submit(run_simulation, i, API_KEY, MODEL, NUM_SPRINTS, 42 + i, COMMUNITY_SIZE)
            for i in range(NUM_RUNS)
        ]

        for future in concurrent.futures.as_completed(futures):
            sweep_results.append(future.result())

    print("\n" + "="*60)
    print("SWEEP SUMMARY")
    print("="*60)
    print(f"{'Run ID':<8} | {'HI':<10} | {'RQ':<10} | {'Policy':<20}")
    print("-" * 60)

    successful_runs = [r for r in sweep_results if r["status"] == "success"]
    for r in successful_runs:
        hi = r.get("harmony_index", 0)
        rq = r.get("resilience_quotient", 0)
        policy = r.get("final_policy", "None")
        print(f"{r['run_id']:<8} | {hi:<10.3f} | {rq:<10.3f} | {policy:<20}")

    if successful_runs:
        avg_hi = sum(r["harmony_index"] for r in successful_runs) / len(successful_runs)
        avg_rq = sum(r["resilience_quotient"] for r in successful_runs) / len(successful_runs)
        print("-" * 60)
        print(f"{'AVERAGE':<8} | {avg_hi:<10.3f} | {avg_rq:<10.3f}")

    print("="*60)

if __name__ == "__main__":
    main()
