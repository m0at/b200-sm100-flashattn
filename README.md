# B200 Training Suite for VAST.ai

Ready-to-run LoRA fine-tuning on NVIDIA B200 (Blackwell) GPUs via VAST.ai. Supports Llama 3.x, Mistral 3.x, and Devstral models. Handles the CUDA 12.8 / PyTorch nightly / flash-attn sm_100 setup that currently requires manual intervention.

## What this solves

B200 GPUs (sm_100 / compute capability 10.0) are not yet supported by stable PyTorch or pre-built flash-attn wheels. This suite:

- Installs PyTorch nightly cu128 (the only version with sm_100 kernels)
- Installs CUDA 12.8 toolkit for native sm_100 compilation
- Builds flash-attn from source with `FLASH_ATTN_CUDA_ARCHS=100`
- Provides a pre-built flash-attn tarball to skip the 15-min compile
- Handles FP8 model dequantization to BF16 for training (Devstral ships FP8-only)
- Sets up a uv-managed venv to avoid pip dependency hell

## Quick Start

```bash
# 1. Rent a B200 on VAST.ai
vast search offers 'gpu_name=B200 num_gpus=1 reliability>0.92 disk_space>200' -o 'dph_total'
vast create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 500

# 2. SSH in and run setup
scp -P <PORT> setup_b200.sh root@<HOST>:/workspace/
ssh -p <PORT> root@<HOST> "bash /workspace/setup_b200.sh"

# 3. Train
ssh -p <PORT> root@<HOST> "/workspace/venv/bin/python train_lora.py \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
  --data your-dataset.jsonl \
  --output /workspace/output"
```

## Files

| File | Purpose |
|------|---------|
| `setup_b200.sh` | One-shot B200 environment setup (PyTorch, CUDA 12.8, flash-attn, uv) |
| `train_lora.py` | Universal LoRA training script (Llama, Mistral, Devstral) |
| `train_dpo.py` | DPO preference training on top of SFT adapter |
| `dequant_fp8.py` | Convert FP8 models to BF16 for training |
| `serve_model.sh` | Serve trained model via vLLM on cheap GPU |
| `push_to_hf.py` | Push trained model + artifacts to HuggingFace |

## Supported Models

| Model | Params | Format | Notes |
|-------|--------|--------|-------|
| Llama 3.1 8B/70B | 8B/70B | BF16 | Works out of the box |
| Llama 3.2 1B/3B | 1B/3B | BF16 | Works out of the box |
| Mistral Small 3.1/3.2 24B | 24B | FP8 | Auto-dequantized to BF16 |
| Devstral Small 2 24B | 24B | FP8 | Auto-dequantized to BF16 |
| Qwen 3.5 | 7B-72B | BF16 | Works out of the box |

## Data Format

Training data is JSONL with a `messages` array per line. Roles: `system`, `user`, `assistant`, `tool`. See `example_data.jsonl` for complete examples.

```json
{"messages": [
  {"role": "system", "content": "You are a terminal assistant..."},
  {"role": "user", "content": "Task description"},
  {"role": "assistant", "content": "Reasoning.\n<tool_call>command</tool_call>"},
  {"role": "tool", "content": "command output"},
  {"role": "assistant", "content": "<tool_call>next command</tool_call>"},
  {"role": "tool", "content": "output"},
  {"role": "assistant", "content": "Done."}
]}
```

Key conventions:
- Assistant wraps commands in `<tool_call>...</tool_call>` tags
- Tool messages contain raw stdout/stderr
- Brief reasoning before tool calls is fine, not after
- Conversations end with an assistant message ("Done.")
- The `train_lora.py` script auto-formats messages into the model's chat template, or falls back to manual `[INST]`/`[/INST]` formatting if the template rejects tool messages

## Performance (measured on 1x B200 179GB)

| Config | Step Time | Notes |
|--------|-----------|-------|
| 24B model, SDPA attention, batch 2, seq 16K | ~195s/step | No flash-attn |
| 24B model, flash_attention_2, batch 2, seq 16K | **~46s/step** | **4.2x faster** |
| 8B model, flash_attention_2, batch 8, seq 8K | ~8s/step | Estimated |

## Cost

| Task | GPU | Time | Cost |
|------|-----|------|------|
| 24B LoRA (25K examples, 3 epochs) | 1x B200 | ~2.5 hrs | ~$7.50 |
| 24B DPO (200 pairs, 1 epoch) | 1x B200 | ~15 min | ~$0.75 |
| Inference serving (INT4) | 1x RTX 4090 | continuous | $0.27/hr |

## Pre-built Artifacts

Available at [and-y/build-artifacts](https://huggingface.co/and-y/build-artifacts):

- `flash_attn_sm100_cu128_torch2.12.tar.gz` — Pre-compiled flash-attn for B200
- `devstral-small-2-24b-bf16/` — Devstral Small 2 in BF16 (converted from official FP8)

## License

Apache 2.0
