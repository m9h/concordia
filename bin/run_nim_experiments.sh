#!/usr/bin/env bash
# run_nim_experiments.sh — Run all cross-system comparison experiments via NIM API
#
# Usage:
#   export NVIDIA_API_KEY="nvapi-..."
#   bash bin/run_nim_experiments.sh [model]
#
# Default model: meta/llama-3.1-8b-instruct
# Other good options:
#   meta/llama-3.1-70b-instruct
#   meta/llama-3.3-70b-instruct
#   nvidia/llama-3.1-nemotron-70b-instruct
#   qwen/qwen2.5-7b-instruct

set -euo pipefail

MODEL="${1:-meta/llama-3.1-8b-instruct}"

if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "Error: NVIDIA_API_KEY not set."
  echo "Get a free key at https://build.nvidia.com"
  echo ""
  echo "Usage:"
  echo "  export NVIDIA_API_KEY='nvapi-...'"
  echo "  bash bin/run_nim_experiments.sh [model]"
  exit 1
fi

echo "========================================"
echo "Cross-System Comparison via NIM API"
echo "========================================"
echo "Model: $MODEL"
echo "API key: ${NVIDIA_API_KEY:0:10}..."
echo "========================================"

# Step 1: Quick smoke test (1 sprint, 4 agents)
echo ""
echo "--- Smoke test ---"
python -m examples.games.sustain_hub.run \
  --nvidia_nim \
  --model_name="$MODEL" \
  --num_sprints=1 \
  --community_size=4 \
  --skip_backstory \
  --fast \
  --output_dir=/tmp/nim_smoke_test 2>&1 | tail -10

if [ $? -ne 0 ]; then
  echo "Smoke test failed! Check your API key and model."
  exit 1
fi
echo "Smoke test passed."

# Step 2: Run cross-system experiments (A1, A2, B1-B3, C1)
# A3 is CPU-only (already running from previous session)
echo ""
echo "--- Running cross-system comparison suite ---"
python -m examples.games.sustain_hub.experiments \
  --cross_system=A1,A2,B1,B2,B3,C1 \
  --nvidia_nim \
  --model_name="$MODEL" \
  --output_dir=/tmp/sustain_hub_cross_system

echo ""
echo "--- Running collective innovation game ---"
python -m examples.games.collective_innovation.run \
  --nvidia_nim \
  --model_name="$MODEL" \
  --num_steps=20 \
  --num_agents=6 \
  --connectivity=fully_connected \
  --prompt_mode=openended_multi \
  --output_dir=/tmp/collective_innovation_nim

echo ""
echo "========================================"
echo "All NIM experiments complete!"
echo "Results in:"
echo "  /tmp/sustain_hub_cross_system/"
echo "  /tmp/collective_innovation_nim/"
echo "========================================"
