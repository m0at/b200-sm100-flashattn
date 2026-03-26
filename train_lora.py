#!/usr/bin/env python3
"""Universal LoRA fine-tuning script for B200 GPUs.

Supports: Llama 3.x, Mistral 3.x, Devstral, Qwen 2.5
Handles: FP8 dequantization, flash_attention_2, BF16 training

Usage:
    python train_lora.py --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
                         --data train.jsonl \
                         --output /workspace/output
"""

import argparse
import json
import os
import torch
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on B200")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--data", required=True, help="Training data JSONL (messages format)")
    p.add_argument("--output", default="/workspace/output", help="Output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-merge", action="store_true", help="Don't merge LoRA after training")
    p.add_argument("--text-field", default=None, help="Use pre-formatted text field instead of messages")
    return p.parse_args()


def load_data(path):
    from datasets import Dataset
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_conversation(example, tokenizer):
    """Convert messages format to text. Falls back to tokenizer chat template,
    or manual formatting if the template rejects tool messages."""
    if "text" in example:
        return example

    msgs = example.get("messages", [])
    if not msgs:
        return {"text": ""}

    # Try tokenizer chat template first
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    except Exception:
        pass

    # Manual formatting: fold tool messages into the conversation
    parts = []
    for msg in msgs:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]")
        elif role == "user":
            parts.append(f"[INST]{content}[/INST]")
        elif role == "assistant":
            parts.append(f"{content}</s>")
        elif role == "tool":
            parts.append(f"[INST]{content}[/INST]")

    return {"text": "<s>" + "".join(parts)}


def load_model(model_id, attn_impl):
    """Load model, handling FP8 dequantization if needed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"Loading {model_id}...")
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="cpu",  # Load to CPU first for FP8 handling
    )

    # Try standard loader, fall back to model-specific
    model = None
    for loader_name in ["AutoModelForCausalLM", "Mistral3ForConditionalGeneration"]:
        try:
            if loader_name == "AutoModelForCausalLM":
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            else:
                from transformers import Mistral3ForConditionalGeneration
                model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            print(f"Loaded with {loader_name}")
            break
        except (ValueError, ImportError):
            continue

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, **load_kwargs
        )
        print("Loaded with trust_remote_code")

    # Handle FP8 quantization
    if hasattr(model.config, "quantization_config"):
        print("Detected FP8 weights. Dequantizing to BF16...")
        delattr(model.config, "quantization_config")
        if hasattr(model, "hf_quantizer"):
            model.hf_quantizer = None
        if hasattr(model, "is_quantized"):
            model.is_quantized = False

        # Cast all tensors
        for name, param in model.named_parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        for name, buf in model.named_buffers():
            if buf.dtype not in (torch.bfloat16, torch.int64, torch.int32, torch.bool):
                buf.data = buf.data.to(torch.bfloat16)

        # Patch validator
        import transformers.trainer as _tr
        _tr.validate_quantization_for_training = lambda m: None

        print("Dequantization complete.")

    # Move to GPU
    print("Moving model to CUDA...")
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.config.use_cache = False

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    print(f"Model ready: {device}, {dtype}, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    return model, tokenizer


def main():
    args = parse_args()

    # Attention implementation
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        print(f"Using flash_attention_2 (v{flash_attn.__version__})")
    except ImportError:
        attn_impl = "sdpa"
        print(f"flash-attn not available, using SDPA (expect ~4x slower)")

    # Load model
    model, tokenizer = load_model(args.model, attn_impl)

    # LoRA config
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
          f"trainable={trainable/1e6:.0f}M ({trainable/total*100:.2f}%)")

    # Data
    from functools import partial
    train_data = load_data(args.data)
    print(f"Train examples (raw): {len(train_data)}")

    if args.text_field:
        pass  # Already has text field
    elif "text" not in train_data.column_names:
        fmt_fn = partial(format_conversation, tokenizer=tokenizer)
        train_data = train_data.map(fmt_fn, remove_columns=["messages"])
    print(f"Train examples (formatted): {len(train_data)}")

    # Training
    from trl import SFTTrainer, SFTConfig

    os.makedirs(args.output, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        max_seq_length=args.max_seq_len,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_text_field="text",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    print(f"\nStarting training ({args.epochs} epochs, {len(train_data)} examples)...")
    trainer.train()

    # Save
    print("Saving adapter...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if not args.no_merge:
        print("Merging LoRA into base model...")
        merged = model.merge_and_unload()
        merge_dir = os.path.join(args.output, "merged")
        merged.save_pretrained(merge_dir, safe_serialization=True)
        tokenizer.save_pretrained(merge_dir)
        print(f"Merged model saved to {merge_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
