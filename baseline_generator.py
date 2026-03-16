import json
import os
import subprocess
import numpy as np
from typing import Any, Dict, List

def run_cmd(cmd: List[str]) -> bool:
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:."
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        return False
    return True

def main():
    # Use NVIDIA NIM for speed and reliability
    os.environ["NVIDIA_API_KEY"] = "nvapi-▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄"
    
    baseline_results = []
    seeds = [42, 43, 44]
    
    print("--- 1. Running Baseline Simulations (use_active_inference=False) ---")
    for seed in seeds:
        output_dir = f"/tmp/vidhi_baseline/run_seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "./.venv/bin/python3", "-m", "examples.games.sustain_hub.run",
            "--nvidia_nim",
            "--model_name=meta/llama-3.1-70b-instruct",
            "--use_active_inference=False",
            "--num_sprints=5",
            "--community_size=8",
            f"--seed={seed}",
            f"--output_dir={output_dir}",
            "--skip_backstory",
            "--fast",
            "--verbose"
        ]
        if run_cmd(cmd):
            res_path = os.path.join(output_dir, "results.json")
            if os.path.exists(res_path):
                with open(res_path, "r") as f:
                    baseline_results.append(json.load(f))

    print("\n--- 2. Running Experiment Ladder (Levels 0, 1, 2, 3) ---")
    ladder_output_dir = "/tmp/vidhi_baseline/ladder"
    os.makedirs(ladder_output_dir, exist_ok=True)
    cmd = [
        "./.venv/bin/python3", "-m", "examples.games.sustain_hub.experiments",
        "--level=0,1,2,3",
        "--num_sprints=10",
        f"--output_dir={ladder_output_dir}"
    ]
    ladder_data = {}
    if run_cmd(cmd):
        ladder_path = os.path.join(ladder_output_dir, "ladder_results.json")
        if os.path.exists(ladder_path):
            with open(ladder_path, "r") as f:
                raw_ladder = json.load(f)
                for item in raw_ladder:
                    ladder_data[f"L{item['level']}"] = item

    # Summary calculations
    hi_list = [r["harmony_index"] for r in baseline_results]
    rq_list = [r["resilience_quotient"] for r in baseline_results]
    
    summary = {
        "mean_hi": float(np.mean(hi_list)) if hi_list else 0,
        "std_hi": float(np.std(hi_list)) if hi_list else 0,
        "mean_rq": float(np.mean(rq_list)) if rq_list else 0,
        "std_rq": float(np.std(rq_list)) if rq_list else 0,
        "sample_size": len(baseline_results)
    }

    final_output = {
        "baseline_runs": baseline_results,
        "ladder_results": ladder_data,
        "summary": summary
    }

    output_path = "vidhi_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n✅ All baseline tasks complete. Results saved to {output_path}")
    print(f"Summary: HI = {summary['mean_hi']:.3f} (±{summary['std_hi']:.3f}), RQ = {summary['mean_rq']:.3f}")

if __name__ == "__main__":
    main()
