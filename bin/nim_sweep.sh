#!/usr/bin/env bash
# nim_sweep.sh — Run SustainHub simulation across multiple NVIDIA NIM models
# Usage: bash bin/nim_sweep.sh
#
# Requires: NVIDIA_API_KEY env var set

set -euo pipefail

MODELS=(
  "meta/llama-3.1-8b-instruct"
  "meta/llama-3.1-70b-instruct"
  "meta/llama-3.3-70b-instruct"
  "qwen/qwen2.5-7b-instruct"
  "qwen/qwq-32b"
  "qwen/qwen3.5-122b-a10b"
  "deepseek-ai/deepseek-r1-distill-qwen-7b"
  "deepseek-ai/deepseek-r1-distill-qwen-32b"
  "nvidia/llama-3.1-nemotron-70b-instruct"
  "nvidia/llama-3.3-nemotron-super-49b-v1"
  "nvidia/nemotron-3-super-120b-a12b"
  "google/gemma-3-27b-it"
)

SPRINTS=3
COMMUNITY=8
OUTPUT_BASE="/tmp/sustain_hub_sweep_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "SustainHub Model Sweep — NVIDIA NIM"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "Sprints: $SPRINTS | Community: $COMMUNITY"
echo "Output: $OUTPUT_BASE"
echo "========================================"

PIDS=()
for model in "${MODELS[@]}"; do
  safe_name=$(echo "$model" | tr '/' '_')
  output_dir="${OUTPUT_BASE}/${safe_name}_${TIMESTAMP}"
  log_file="${OUTPUT_BASE}/${safe_name}.log"
  mkdir -p "$output_dir"

  echo "Starting: $model → $log_file"
  python -m examples.games.sustain_hub.run \
    --nvidia_nim \
    --model_name="$model" \
    --num_sprints=$SPRINTS \
    --community_size=$COMMUNITY \
    --fast \
    --output_dir="$output_dir" \
    > "$log_file" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All ${#MODELS[@]} simulations launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

FAILED=0
SUCCEEDED=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  model=${MODELS[$i]}
  safe_name=$(echo "$model" | tr '/' '_')
  if wait "$pid"; then
    echo "  DONE: $model"
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    echo "  FAIL: $model (see ${OUTPUT_BASE}/${safe_name}.log)"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================"
echo "Sweep complete: $SUCCEEDED succeeded, $FAILED failed"
echo "========================================"

# Aggregate results
echo ""
echo "Results summary:"
for model in "${MODELS[@]}"; do
  safe_name=$(echo "$model" | tr '/' '_')
  results_file="${OUTPUT_BASE}/${safe_name}_${TIMESTAMP}/results.json"
  if [ -f "$results_file" ]; then
    hi=$(python -c "import json; r=json.load(open('$results_file')); print(f'{r[\"harmony_index\"]:.3f}')" 2>/dev/null || echo "ERR")
    rq=$(python -c "import json; r=json.load(open('$results_file')); print(f'{r[\"resilience_quotient\"]:.3f}')" 2>/dev/null || echo "ERR")
    printf "  %-45s HI=%-6s RQ=%s\n" "$model" "$hi" "$rq"
  else
    printf "  %-45s NO RESULTS\n" "$model"
  fi
done
