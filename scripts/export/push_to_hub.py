"""
scripts/export/push_to_hub.py
==============================
Push Lokneeti-3B model, tokenizer, and dataset to HuggingFace Hub.

Usage:
  # Push model adapter:
  python scripts/export/push_to_hub.py --type model \\
      --local-path outputs/lokneeti-3b-qlora/final_adapter \\
      --repo-id vikhram-labs/Lokneeti-3B

  # Push dataset:
  python scripts/export/push_to_hub.py --type dataset \\
      --local-path data/final \\
      --repo-id vikhram-labs/lokneeti-governance-dataset
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.utils.hub import hub_login, push_model_to_hub, push_dataset_to_hub
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

MODEL_CARD_PATH = Path("docs/MODEL_CARD.md")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Push Lokneeti to HuggingFace Hub")
    ap.add_argument("--type",       choices=["model", "dataset"], required=True)
    ap.add_argument("--local-path", type=str, required=True)
    ap.add_argument("--repo-id",    type=str, required=True)
    ap.add_argument("--message",    type=str, default="Upload Lokneeti-3B")
    ap.add_argument("--private",    action="store_true")
    args = ap.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        log.error(f"Local path not found: {local_path}")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set. Add it to .env or export HF_TOKEN=hf_...")
        sys.exit(1)

    hub_login(token=token)

    if args.type == "model":
        url = push_model_to_hub(
            local_model_dir=local_path,
            repo_id=args.repo_id,
            commit_message=args.message,
            private=args.private,
            token=token,
        )
        log.info(f"✅ Model available at: {url}")

        # Push model card if it exists
        if MODEL_CARD_PATH.exists():
            from lokneeti.utils.hub import push_model_card
            push_model_card(MODEL_CARD_PATH, args.repo_id, token=token)

    elif args.type == "dataset":
        url = push_dataset_to_hub(
            local_dataset_dir=local_path,
            repo_id=args.repo_id,
            commit_message=args.message,
            private=args.private,
            token=token,
        )
        log.info(f"✅ Dataset available at: {url}")

    log.info("🎉 HuggingFace Hub push complete!")


if __name__ == "__main__":
    main()
