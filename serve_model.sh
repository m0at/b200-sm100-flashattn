#!/bin/bash
# Serve a trained model via vLLM on a cheap GPU (RTX 4090 / A100 40GB)
# Usage: bash serve_model.sh --model m0at/devstral-agent-v1 [--port 8000] [--no-quant]
set -e

MODEL=""
PORT=8000
QUANT="--quantization gptq"
MAX_LEN=131072

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --no-quant) QUANT=""; shift;;
        --max-len) MAX_LEN="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Usage: bash serve_model.sh --model HF_MODEL_ID [--port 8000] [--no-quant]"
    exit 1
fi

echo "=== Installing vLLM ==="
pip install vllm 2>&1 | tail -3

echo "=== Serving $MODEL on port $PORT ==="
echo "Quantization: ${QUANT:-none}"
echo "Max context: $MAX_LEN"

vllm serve "$MODEL" \
    --max-model-len $MAX_LEN \
    --port $PORT \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.92 \
    $QUANT
