#!/bin/bash
# setup_vllm.sh - Start vLLM server on DGX Spark
#
# Usage:
#   # Direct (if vLLM is installed):
#   bash bin/setup_vllm.sh
#
#   # Via Docker (recommended for DGX Spark):
#   bash bin/setup_vllm.sh --docker
#
# The server exposes an OpenAI-compatible API on port 8000.
# Point overnight_runner.py at it with --vllm_url=http://<dgx-ip>:8000/v1

set -e

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================"
echo "vLLM Server Setup"
echo "  Model: ${MODEL}"
echo "  Port: ${PORT}"
echo "  GPU Memory: ${GPU_MEM}"
echo "  Max Context: ${MAX_MODEL_LEN}"
echo "============================================"

if [ "$1" = "--docker" ]; then
    echo "Starting via Docker..."
    docker run -d \
        --name sustainhub-vllm \
        --gpus all \
        --shm-size=16g \
        -p ${PORT}:${PORT} \
        -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        vllm/vllm-openai:latest \
        --model "${MODEL}" \
        --dtype auto \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM} \
        --port ${PORT} \
        --tensor-parallel-size 1

    echo "Container started. Waiting for model to load..."
    echo "Check status: docker logs -f sustainhub-vllm"
    echo "Test: curl http://localhost:${PORT}/v1/models"
else
    echo "Starting vLLM directly..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --dtype auto \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM} \
        --port ${PORT} \
        --tensor-parallel-size 1
fi
