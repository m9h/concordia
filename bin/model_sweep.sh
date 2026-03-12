#!/bin/bash
# model_sweep.sh — Compare LLM backbones on SustainHub social dilemma
#
# Research question: Does the LLM backbone affect emergent cooperation?
# NeurIPS 2025 showed s=0.426 average, s<0.2 on social dilemmas.
# Different models may negotiate/cooperate differently.
#
# Usage:
#   # Start vLLM with a specific model first:
#   MODEL=meta-llama/Llama-3.1-70B-Instruct bash bin/setup_vllm.sh --docker
#
#   # Then run the sweep (one model at a time — swap vLLM between runs):
#   bash bin/model_sweep.sh
#
# Models sized for DGX Spark (128GB unified memory):
#
# TIER 1: Small (4-8GB) — fast, many seeds for statistical power
#   meta-llama/Llama-3.1-8B-Instruct          ~4GB   Speed baseline
#   Qwen/Qwen2.5-7B-Instruct                  ~4GB   Strong reasoning at 7B
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   ~4GB   R1 reasoning at 7B
#
# TIER 2: Medium (35-40GB) — best quality/speed tradeoff
#   meta-llama/Llama-3.1-70B-Instruct         ~35GB   Well-studied baseline
#   Qwen/Qwen2.5-72B-Instruct                 ~36GB   Top reasoning benchmarks
#   deepseek-ai/DeepSeek-R1-Distill-Llama-70B ~35GB   Chain-of-thought baked in
#
# TIER 3: Large (60-70GB) — maximum quality, fits in 128GB
#   mistralai/Mistral-Large-Instruct-2411     ~62GB   Largest that fits

set -e

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sustainhub_model_sweep}"
NUM_SPRINTS="${NUM_SPRINTS:-5}"
NUM_RUNS="${NUM_RUNS:-3}"
COMMUNITY_SIZE="${COMMUNITY_SIZE:-8}"

# Small models — run more seeds since they're fast
SMALL_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# Large models — fewer seeds, richer results
LARGE_MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

echo "============================================"
echo "SustainHub Model Sweep"
echo "  Output: ${OUTPUT_DIR}"
echo "  vLLM:   ${VLLM_URL}"
echo "============================================"
echo ""
echo "NOTE: You must restart vLLM with each model."
echo "For each model below:"
echo "  1. docker rm -f sustainhub-vllm"
echo "  2. MODEL=<model> bash bin/setup_vllm.sh --docker"
echo "  3. Wait for model to load (docker logs -f sustainhub-vllm)"
echo "  4. Run the overnight_runner command shown"
echo ""

for model in "${SMALL_MODELS[@]}"; do
    model_short=$(basename "$model")
    echo "--- ${model_short} (small, 5 seeds) ---"
    echo "  MODEL=${model} bash bin/setup_vllm.sh --docker"
    echo "  python bin/overnight_runner.py \\"
    echo "    --mode=concordia \\"
    echo "    --vllm_url=${VLLM_URL} \\"
    echo "    --model_name=${model} \\"
    echo "    --num_concordia_runs=5 \\"
    echo "    --num_sprints=${NUM_SPRINTS} \\"
    echo "    --community_size=${COMMUNITY_SIZE} \\"
    echo "    --output_dir=${OUTPUT_DIR}/${model_short}"
    echo ""
done

for model in "${LARGE_MODELS[@]}"; do
    model_short=$(basename "$model")
    echo "--- ${model_short} (large, 3 seeds) ---"
    echo "  MODEL=${model} bash bin/setup_vllm.sh --docker"
    echo "  python bin/overnight_runner.py \\"
    echo "    --mode=concordia \\"
    echo "    --vllm_url=${VLLM_URL} \\"
    echo "    --model_name=${model} \\"
    echo "    --num_concordia_runs=3 \\"
    echo "    --num_sprints=${NUM_SPRINTS} \\"
    echo "    --community_size=${COMMUNITY_SIZE} \\"
    echo "    --output_dir=${OUTPUT_DIR}/${model_short}"
    echo ""
done

echo "After all runs, compare results:"
echo "  python -c \"import json,glob; [print(f.split('/')[-2], json.load(open(f))['harmony_index']) for f in sorted(glob.glob('${OUTPUT_DIR}/*/concordia/*/results.json'))]\""
