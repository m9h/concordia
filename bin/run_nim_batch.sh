#!/bin/bash
# Run SustainHub experiments on NIM cloud.
# Usage: bash bin/run_nim_batch.sh
set -euo pipefail

PYTHON=".venv/bin/python"
NIM_URL="https://integrate.api.nvidia.com/v1"
MODEL="qwen/qwen2.5-7b-instruct"

# Ensure NVIDIA_API_KEY is set (fall back to NGC_API_KEY)
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-${NGC_API_KEY:-}}"
if [[ -z "$NVIDIA_API_KEY" ]]; then
    echo "ERROR: Neither NVIDIA_API_KEY nor NGC_API_KEY is set." >&2
    exit 1
fi
BASE_DIR="/tmp/sustainhub_nim_$(date +%Y%m%d_%H%M%S)"
SEEDS="0 1 2 3 4"
FAIL_COUNT=0
PASS_COUNT=0

mkdir -p "$BASE_DIR"
echo "Output: $BASE_DIR"
echo "Started: $(date)"

run_sim() {
    local label="$1" seed="$2" sprints="$3" size="$4" gov="$5"
    local out_dir="$BASE_DIR/${label}_seed${seed}"
    local log_file="$out_dir/run.log"
    mkdir -p "$out_dir"
    echo "[$(date +%H:%M:%S)] $label seed=$seed starting..."

    local exit_code=0
    $PYTHON -m examples.games.sustain_hub.run \
        --vllm_url="$NIM_URL" \
        --model_name="$MODEL" \
        --num_sprints="$sprints" \
        --community_size="$size" \
        --governance="$gov" \
        --fast \
        --output_dir="$out_dir" \
        --seed="$seed" \
        > "$log_file" 2>&1 || exit_code=$?

    # Show last few lines of output
    tail -5 "$log_file"

    if [[ $exit_code -ne 0 ]]; then
        echo "  FAIL: $label seed=$seed exited with code $exit_code" >&2
        echo "  Log: $log_file" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    if [[ ! -f "$out_dir/results.json" ]]; then
        echo "  FAIL: $label seed=$seed exited 0 but no results.json produced" >&2
        echo "  Log: $log_file" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    echo "  OK: $label seed=$seed -> $out_dir/results.json"
    PASS_COUNT=$((PASS_COUNT + 1))
    return 0
}

echo ""
echo "=== B-series: Governance with enriched metrics (5 seeds × 3 modes) ==="
for seed in $SEEDS; do
    run_sim "B1_free" "$seed" 5 8 "free_choice" || true
done
for seed in $SEEDS; do
    run_sim "B2_dictator" "$seed" 5 8 "dictator" || true
done
for seed in $SEEDS; do
    run_sim "B3_meritocratic" "$seed" 5 8 "meritocratic" || true
done

echo ""
echo "=== A-series: Seeds 3-4 (Rohira comparison, 10 agents, 10 sprints) ==="
for seed in 3 4; do
    run_sim "A1_rohira" "$seed" 10 10 "free_choice" || true
done

echo ""
echo "=== A2: LLM+AIF variant (5 seeds) ==="
for seed in $SEEDS; do
    run_sim "A2_aif" "$seed" 10 10 "free_choice" || true
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Results in: $BASE_DIR"
echo "Passed: $PASS_COUNT  Failed: $FAIL_COUNT"
echo "========================================"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo ""
    echo "WARNING: $FAIL_COUNT run(s) failed. Check logs above for details."
fi

# Aggregate results (only from successful runs)
$PYTHON -c "
import json, os, glob
base = '$BASE_DIR'
results = {}
found = sorted(glob.glob(os.path.join(base, '*/results.json')))
if not found:
    print('No results.json files found to aggregate.')
else:
    for d in found:
        label = os.path.basename(os.path.dirname(d))
        with open(d) as f:
            data = json.load(f)
        results[label] = {
            'hi': data.get('harmony_index'),
            'rq': data.get('resilience_quotient'),
            'mean_bai': data.get('aif_belief_trajectories') and len(data.get('aif_belief_trajectories', {})) > 0,
            'trust_recip': data.get('trust_reciprocity'),
            'norm_emergence': data.get('norm_emergence_rate'),
            'burnout': data.get('mean_burnout'),
        }
        print(f'{label}: HI={data.get(\"harmony_index\",0):.3f} BAI-agents={len(data.get(\"aif_belief_trajectories\",{}))} Trust={data.get(\"trust_reciprocity\",0):.3f}')

    with open(os.path.join(base, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSummary saved to {base}/summary.json ({len(results)} runs)')
"

# Exit with failure if any runs failed
if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
