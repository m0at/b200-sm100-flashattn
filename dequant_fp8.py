#!/usr/bin/env python3
"""Convert FP8 HuggingFace models to BF16 and push to HuggingFace.

Usage:
    python dequant_fp8.py \
        --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
        --output /workspace/devstral-bf16 \
        --push-to and-y/Devstral-Small-2-24B-bf16
"""

import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser(description="Dequantize FP8 model to BF16")
    p.add_argument("--model", required=True, help="HuggingFace model ID (FP8)")
    p.add_argument("--output", required=True, help="Local output directory")
    p.add_argument("--push-to", default=None, help="HuggingFace repo to push to")
    p.add_argument("--private", action="store_true", help="Make HF repo private")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")

    # Load model
    print(f"Loading {args.model} (FP8)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cpu")

    try:
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            args.model, token=token, **load_kwargs
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, token=token, trust_remote_code=True, **load_kwargs
        )

    # Strip quantization
    if hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")
    if hasattr(model, "hf_quantizer"):
        model.hf_quantizer = None

    # Cast all tensors to BF16
    print("Casting to BF16...")
    fp8_count = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
            fp8_count += 1
    for name, buf in model.named_buffers():
        if buf.dtype not in (torch.bfloat16, torch.int64, torch.int32, torch.bool):
            buf.data = buf.data.to(torch.bfloat16)
    print(f"Cast {fp8_count} parameters from FP8 to BF16")

    # Save
    print(f"Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    tokenizer.save_pretrained(args.output)

    size_gb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
    ) / 1024**3
    print(f"Saved: {size_gb:.1f} GB")

    # Push to HF
    if args.push_to:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        try:
            create_repo(args.push_to, private=args.private, token=token)
        except Exception:
            pass
        print(f"Pushing to {args.push_to}...")
        api.upload_folder(
            folder_path=args.output,
            repo_id=args.push_to,
            token=token,
        )
        print(f"Pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
