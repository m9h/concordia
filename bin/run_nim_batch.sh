#!/bin/bash
# Run SustainHub experiments on NIM cloud.
# Usage: bash bin/run_nim_batch.sh
set -e

PYTHON=".venv/bin/python"
NIM_URL="https://integrate.api.nvidia.com/v1"
MODEL="qwen/qwen2.5-7b-instruct"
BASE_DIR="/tmp/sustainhub_nim_$(date +%Y%m%d_%H%M%S)"
SEEDS="0 1 2 3 4"

mkdir -p "$BASE_DIR"
echo "Output: $BASE_DIR"
echo "Started: $(date)"

run_sim() {
    local label="$1" seed="$2" sprints="$3" size="$4" gov="$5"
    local out_dir="$BASE_DIR/${label}_seed${seed}"
    mkdir -p "$out_dir"
    echo "[$(date +%H:%M:%S)] $label seed=$seed starting..."
    $PYTHON -m examples.games.sustain_hub.run \
        --vllm_url="$NIM_URL" \
        --model_name="$MODEL" \
        --num_sprints="$sprints" \
        --community_size="$size" \
        --governance="$gov" \
        --fast \
        --output_dir="$out_dir" \
        --seed="$seed" \
        2>&1 | tail -5
    echo "[$(date +%H:%M:%S)] $label seed=$seed done (exit=$?)"
}

echo ""
echo "=== B-series: Governance with enriched metrics (5 seeds × 3 modes) ==="
for seed in $SEEDS; do
    run_sim "B1_free" "$seed" 5 8 "free_choice"
done
for seed in $SEEDS; do
    run_sim "B2_dictator" "$seed" 5 8 "dictator"
done
for seed in $SEEDS; do
    run_sim "B3_meritocratic" "$seed" 5 8 "meritocratic"
done

echo ""
echo "=== A-series: Seeds 3-4 (Rohira comparison, 10 agents, 10 sprints) ==="
for seed in 3 4; do
    run_sim "A1_rohira" "$seed" 10 10 "free_choice"
done

echo ""
echo "=== A2: LLM+AIF variant (5 seeds) ==="
for seed in $SEEDS; do
    run_sim "A2_aif" "$seed" 10 10 "free_choice"
done

echo ""
echo "Finished: $(date)"
echo "Results in: $BASE_DIR"

# Aggregate results
$PYTHON -c "
import json, os, glob
base = '$BASE_DIR'
results = {}
for d in sorted(glob.glob(os.path.join(base, '*/results.json'))):
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
print(f'\nSummary saved to {base}/summary.json')
"
