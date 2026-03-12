#!/bin/bash
# setup_vllm.sh - Start vLLM server on DGX Spark (Grace Blackwell, aarch64)
#
# Uses NVIDIA NGC v26 PyTorch container with vLLM installed on top.
# The NGC container has native aarch64 support for Grace Blackwell.
#
# Usage:
#   # Via Docker (recommended for DGX Spark):
#   bash bin/setup_vllm.sh --docker
#
#   # Direct (if vLLM is already installed in your environment):
#   bash bin/setup_vllm.sh
#
# The server exposes an OpenAI-compatible API on port 8000.
# Point overnight_runner.py at it with --vllm_url=http://<dgx-ip>:8000/v1

set -e

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NGC_TAG="${NGC_TAG:-26.01-py3}"

echo "============================================"
echo "vLLM Server Setup (DGX Spark)"
echo "  Model:       ${MODEL}"
echo "  Port:        ${PORT}"
echo "  GPU Memory:  ${GPU_MEM}"
echo "  Max Context: ${MAX_MODEL_LEN}"
echo "  NGC Tag:     ${NGC_TAG}"
echo "============================================"

if [ "$1" = "--docker" ]; then
    # Stop existing container if running
    docker rm -f sustainhub-vllm 2>/dev/null || true

    echo "Building vLLM image from NGC v26 base..."
    docker build -t sustainhub-vllm-server -f - . <<'DOCKERFILE'
FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN pip install --no-cache-dir vllm
DOCKERFILE

    echo "Starting vLLM server container..."
    docker run -d \
        --name sustainhub-vllm \
        --runtime=nvidia \
        --gpus all \
        --shm-size=16g \
        -p ${PORT}:${PORT} \
        -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        sustainhub-vllm-server \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --dtype auto \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEM} \
            --port ${PORT} \
            --tensor-parallel-size 1 \
            --trust-remote-code

    echo ""
    echo "Container started. Model is loading..."
    echo "  Monitor:  docker logs -f sustainhub-vllm"
    echo "  Test:     curl http://localhost:${PORT}/v1/models"
    echo "  Stop:     docker rm -f sustainhub-vllm"

elif [ "$1" = "--nim" ]; then
    # Alternative: use NVIDIA NIM (pre-optimized inference)
    # Requires NGC API key
    echo "Starting NVIDIA NIM container..."
    docker rm -f sustainhub-nim 2>/dev/null || true

    docker run -d \
        --name sustainhub-nim \
        --runtime=nvidia \
        --gpus all \
        --shm-size=16g \
        -p ${PORT}:8000 \
        -e NGC_API_KEY="${NGC_API_KEY}" \
        nvcr.io/nim/meta/llama-3.1-70b-instruct:latest

    echo "NIM container started."
    echo "  Monitor:  docker logs -f sustainhub-nim"
    echo "  Test:     curl http://localhost:${PORT}/v1/models"

else
    echo "Starting vLLM directly (no container)..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --dtype auto \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM} \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        --trust-remote-code
fi
