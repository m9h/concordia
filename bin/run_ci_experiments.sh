#!/bin/bash
# Batch runner for Collective Innovation experiments.
#
# Phase 1: Connectivity sweep (3 modes × 3 seeds = 9 runs)
# Phase 2: Prompt mode sweep  (4 modes × 3 seeds = 12 runs)
#
# Usage: bash bin/run_ci_experiments.sh
set -euo pipefail

PYTHON=".venv/bin/python"
MODEL="qwen/qwen2.5-7b-instruct"
BASE_DIR="/tmp/ci_experiments_$(date +%Y%m%d_%H%M%S)"
SEEDS="0 1 2"
NUM_STEPS=50
NUM_AGENTS=6

# Verify NGC_API_KEY is set
if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "ERROR: NGC_API_KEY is not set. Export it before running this script."
    echo "  export NGC_API_KEY=nvapi-..."
    exit 1
fi

mkdir -p "$BASE_DIR"

# Tracking arrays for the summary table
declare -a RUN_LABELS=()
declare -a RUN_STATUSES=()

echo "========================================================================"
echo "Collective Innovation — Batch Experiment Runner"
echo "========================================================================"
echo "Model:      $MODEL"
echo "Steps:      $NUM_STEPS"
echo "Agents:     $NUM_AGENTS"
echo "Seeds:      $SEEDS"
echo "Output:     $BASE_DIR"
echo "Started:    $(date)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# run_one: Execute a single simulation and verify results.json was produced.
# ---------------------------------------------------------------------------
run_one() {
    local label="$1"
    local seed="$2"
    local connectivity="$3"
    local prompt_mode="$4"
    local out_dir="$BASE_DIR/${label}_seed${seed}"
    local run_tag="${label}_seed${seed}"

    mkdir -p "$out_dir"
    echo ""
    echo "[$(date +%H:%M:%S)] START  $run_tag  (connectivity=$connectivity  prompt=$prompt_mode  seed=$seed)"

    # Run the simulation; capture exit code without letting set -e kill us
    local rc=0
    $PYTHON -m examples.games.collective_innovation.run \
        --nvidia_nim \
        --model_name="$MODEL" \
        --num_steps="$NUM_STEPS" \
        --num_agents="$NUM_AGENTS" \
        --connectivity="$connectivity" \
        --prompt_mode="$prompt_mode" \
        --seed="$seed" \
        --output_dir="$out_dir" \
        2>&1 | tee "$out_dir/run.log" | tail -20 \
        || rc=$?

    # Verify results.json was written
    if [[ -f "$out_dir/results.json" ]]; then
        echo "[$(date +%H:%M:%S)] OK     $run_tag  (exit=$rc, results.json exists)"
        RUN_LABELS+=("$run_tag")
        RUN_STATUSES+=("OK")
    else
        echo "[$(date +%H:%M:%S)] FAIL   $run_tag  (exit=$rc, results.json MISSING)"
        RUN_LABELS+=("$run_tag")
        RUN_STATUSES+=("FAIL")
    fi
}

# ===========================  PHASE 1  =====================================
echo ""
echo "========================================================================"
echo "PHASE 1: Connectivity Sweep  (3 modes × 3 seeds = 9 runs)"
echo "  Prompt mode fixed: openended_multi"
echo "========================================================================"

for conn in fully_connected dynamic isolated; do
    for seed in $SEEDS; do
        run_one "P1_${conn}" "$seed" "$conn" "openended_multi"
    done
done

# ===========================  PHASE 2  =====================================
echo ""
echo "========================================================================"
echo "PHASE 2: Prompt Mode Sweep  (4 modes × 3 seeds = 12 runs)"
echo "  Connectivity fixed: dynamic"
echo "========================================================================"

for pmode in openended_single openended_multi targeted_single targeted_multi; do
    for seed in $SEEDS; do
        run_one "P2_${pmode}" "$seed" "dynamic" "$pmode"
    done
done

# ===========================  SUMMARY  =====================================
echo ""
echo "========================================================================"
echo "BATCH COMPLETE — $(date)"
echo "========================================================================"

# Print per-run status table
printf "\n%-40s  %6s\n" "RUN" "STATUS"
printf "%-40s  %6s\n" "----------------------------------------" "------"
for i in "${!RUN_LABELS[@]}"; do
    printf "%-40s  %6s\n" "${RUN_LABELS[$i]}" "${RUN_STATUSES[$i]}"
done

# Aggregate metrics from results.json files
echo ""
echo "========================================================================"
echo "METRICS SUMMARY"
echo "========================================================================"

$PYTHON -c "
import json, os, glob

base = '$BASE_DIR'
rows = []
for path in sorted(glob.glob(os.path.join(base, '*/results.json'))):
    label = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        data = json.load(f)
    rows.append({
        'label': label,
        'unique_discoveries': data.get('unique_discoveries', 'N/A'),
        'innovation_score': data.get('innovation_score', 'N/A'),
        'discovery_rate': data.get('discovery_rate', 'N/A'),
        'knowledge_diversity': data.get('knowledge_diversity', 'N/A'),
    })

if not rows:
    print('No results.json files found — all runs may have failed.')
else:
    hdr = f\"{'Config':<40s}  {'Uniq':>6s}  {'Innov':>8s}  {'Rate':>8s}  {'Divers':>8s}\"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        ud = r['unique_discoveries']
        isc = r['innovation_score']
        dr = r['discovery_rate']
        kd = r['knowledge_diversity']
        ud_s = f'{ud:>6d}' if isinstance(ud, (int, float)) else f'{ud:>6s}'
        isc_s = f'{isc:>8.3f}' if isinstance(isc, (int, float)) else f'{isc:>8s}'
        dr_s = f'{dr:>8.3f}' if isinstance(dr, (int, float)) else f'{dr:>8s}'
        kd_s = f'{kd:>8.3f}' if isinstance(kd, (int, float)) else f'{kd:>8s}'
        print(f'{r[\"label\"]:<40s}  {ud_s}  {isc_s}  {dr_s}  {kd_s}')

    # Save machine-readable summary
    with open(os.path.join(base, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nSummary saved to {base}/summary.json')
"

echo ""
echo "All outputs in: $BASE_DIR"
