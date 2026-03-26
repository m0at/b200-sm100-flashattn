#!/usr/bin/env python3
"""Push trained model and artifacts to HuggingFace.

Usage:
    python push_to_hf.py --model-dir /workspace/output/merged \
                         --repo m0at/devstral-agent-v1 \
                         --private
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Directory with model files")
    p.add_argument("--repo", required=True, help="HuggingFace repo ID")
    p.add_argument("--private", action="store_true")
    p.add_argument("--adapter-only", action="store_true", help="Push only the LoRA adapter, not merged")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN environment variable")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    try:
        create_repo(args.repo, private=args.private, token=token)
        print(f"Created {'private' if args.private else 'public'} repo: {args.repo}")
    except Exception:
        print(f"Repo {args.repo} exists")

    print(f"Uploading {args.model_dir} to {args.repo}...")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo,
        token=token,
    )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
