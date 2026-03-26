#!/bin/bash
# B200 Training Environment Setup for VAST.ai
# Handles: PyTorch cu128 nightly, CUDA 12.8 toolkit, flash-attn sm_100, uv
# Run once after renting a B200 instance.
set -e

echo "============================================="
echo "  B200 Training Suite - Environment Setup"
echo "============================================="
echo ""

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"B200"* ]]; then
    echo "WARNING: This script is designed for B200 GPUs (sm_100)."
    echo "Detected: $GPU_NAME"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# Step 1: Install uv
echo ""
echo "=== Step 1/6: Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -2
export PATH="$HOME/.local/bin:$PATH"

# Step 2: Create venv with PyTorch cu128 nightly
echo ""
echo "=== Step 2/6: Creating venv with PyTorch cu128 ==="
uv venv /workspace/venv --python 3.11 2>&1 | tail -1
PY=/workspace/venv/bin/python

uv pip install --python $PY \
    "torch==2.12.0.dev20260325+cu128" \
    --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -3

# Verify CUDA works
$PY -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.randn(10, 10, device='cuda', dtype=torch.bfloat16)
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU OK')
"

# Step 3: Install training stack
echo ""
echo "=== Step 3/6: Installing training stack ==="
uv pip install --python $PY \
    transformers peft trl accelerate datasets huggingface_hub \
    numpy psutil pip 2>&1 | tail -3

# Step 4: Install CUDA 12.8 toolkit (needed for flash-attn compilation)
echo ""
echo "=== Step 4/6: Installing CUDA 12.8 toolkit ==="
if [ ! -f /usr/local/cuda-12.8/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb 2>&1 | tail -1
    apt-get update -qq 2>&1 | tail -1
    apt-get install -y -qq cuda-toolkit-12-8 2>&1 | tail -3
    rm -f cuda-keyring_1.1-1_all.deb
fi
echo "nvcc: $(/usr/local/cuda-12.8/bin/nvcc --version | grep release)"

# Step 5: Build or download flash-attn
echo ""
echo "=== Step 5/6: Setting up flash-attn for sm_100 ==="

# Try downloading pre-built first
PRE_BUILT=0
$PY -c "
from huggingface_hub import hf_hub_download
try:
    f = hf_hub_download('and-y/build-artifacts', 'flash_attn_sm100_cu128_torch2.12.tar.gz', local_dir='/workspace')
    print(f'Downloaded pre-built flash-attn: {f}')
except:
    print('NO_PREBUILT')
" 2>&1 | tee /tmp/flash_dl.log

if grep -q "NO_PREBUILT" /tmp/flash_dl.log; then
    echo "Pre-built not available. Building from source (~15 min)..."
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export FLASH_ATTN_CUDA_ARCHS="100"
    export MAX_JOBS=$(nproc)

    cd /workspace
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>&1 | tail -1
    cd flash-attention
    $PY -m pip install --no-build-isolation . 2>&1 | tail -5
    cd /workspace
    rm -rf flash-attention
else
    echo "Installing pre-built flash-attn..."
    cd /workspace/venv/lib/python3.11/site-packages
    tar xzf /workspace/flash_attn_sm100_cu128_torch2.12.tar.gz
    echo "Installed from pre-built tarball"
fi

# Verify
$PY -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print(f'flash-attn sm_100: OK ({out.shape})')
"

# Step 6: Final verification
echo ""
echo "=== Step 6/6: Final verification ==="
$PY -c "
import torch; print(f'torch {torch.__version__} CUDA {torch.version.cuda}')
import flash_attn; print(f'flash-attn {flash_attn.__version__}')
from transformers import AutoTokenizer; print('transformers OK')
from peft import LoraConfig; print('peft OK')
from trl import SFTTrainer; print('trl OK')
print()
print('============================================')
print('  B200 Training Environment: READY')
print('  Python: /workspace/venv/bin/python')
print('============================================')
"
