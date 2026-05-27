"""
scripts/train/train_qlora.py
=============================
Main QLoRA fine-tuning entry point for Lokneeti-3B.

Usage:
  # Full training:
  python scripts/train/train_qlora.py --config configs/training_config.yaml \\
      --train data/final/train.jsonl --eval data/final/val.jsonl

  # Quick smoke test (100 samples):
  python scripts/train/train_qlora.py --config configs/training_config.yaml \\
      --train data/final/train.jsonl --max-samples 100 --epochs 1

  # Resume from checkpoint:
  python scripts/train/train_qlora.py --config configs/training_config.yaml \\
      --train data/final/train.jsonl --resume checkpoints/checkpoint-200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.training.config import TrainingConfig
from lokneeti.training.trainer import LoknetiTrainer
from lokneeti.utils.logging import configure_root_logger, get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lokneeti-3B QLoRA Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config",      type=str, default="configs/training_config.yaml")
    ap.add_argument("--train",       type=str, default="data/final/train.jsonl")
    ap.add_argument("--eval",        type=str, default=None)
    ap.add_argument("--resume",      type=str, default=None,
                    help="Path to checkpoint directory to resume training from")
    ap.add_argument("--base-model",  type=str, default=None,
                    help="Override base model (e.g. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--output-dir",  type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit dataset to N samples (debug mode)")
    ap.add_argument("--epochs",      type=int, default=None)
    ap.add_argument("--lr",          type=float, default=None)
    ap.add_argument("--lora-rank",   type=int, default=None)
    ap.add_argument("--no-wandb",    action="store_true",
                    help="Disable W&B logging")
    ap.add_argument("--merge",       action="store_true",
                    help="Merge LoRA weights after training")
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    configure_root_logger()
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        log.info(f"Loading config from {config_path}")
        config = TrainingConfig.from_yaml(config_path)
    else:
        log.warning(f"Config file not found: {config_path} — using defaults")
        config = TrainingConfig()

    # ── Apply CLI overrides ──────────────────────────────────────────────
    if args.base_model:
        config.base_model = args.base_model
        log.info(f"Base model overridden: {args.base_model}")
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
        log.info(f"⚠️  Debug mode: max_samples={args.max_samples}")
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.lora_rank is not None:
        config.lora.r = args.lora_rank
        config.lora.lora_alpha = args.lora_rank * 2
    if args.no_wandb:
        config.report_to = "none"

    # ── Validate data files ──────────────────────────────────────────────
    train_path = Path(args.train)
    if not train_path.exists():
        log.error(
            f"Training file not found: {train_path}\n"
            "Run: python scripts/data/policy_dataset_builder.py first."
        )
        sys.exit(1)

    eval_path = Path(args.eval) if args.eval else None

    # ── Log configuration ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Lokneeti-3B QLoRA Training")
    log.info("=" * 60)
    log.info(f"  Base model:  {config.base_model}")
    log.info(f"  Train data:  {train_path}")
    log.info(f"  Eval data:   {eval_path or 'None'}")
    log.info(f"  Output dir:  {config.output_dir}")
    log.info(f"  LoRA rank:   {config.lora.r} (alpha={config.lora.lora_alpha})")
    log.info(f"  Epochs:      {config.num_train_epochs}")
    log.info(f"  Batch size:  {config.per_device_train_batch_size} × "
             f"{config.gradient_accumulation_steps} grad_accum = "
             f"{config.per_device_train_batch_size * config.gradient_accumulation_steps} eff.")
    log.info(f"  LR:          {config.learning_rate}")
    log.info(f"  4-bit quant: {config.quantization.load_in_4bit}")
    log.info(f"  Max seq len: {config.max_seq_length}")
    log.info(f"  W&B:         {config.report_to}")
    log.info("=" * 60)

    # ── Train ────────────────────────────────────────────────────────────
    trainer = LoknetiTrainer(config=config)
    trainer.train(
        train_data_path=train_path,
        eval_data_path=eval_path,
        resume_from_checkpoint=args.resume,
    )

    # ── Optionally merge LoRA weights ────────────────────────────────────
    if args.merge:
        merged_dir = Path(config.output_dir) / "merged_model"
        log.info(f"Merging LoRA weights → {merged_dir}")
        trainer.save_merged(merged_dir)

    log.info("🎉 Training complete! Adapter saved to: " + config.output_dir)


if __name__ == "__main__":
    main()
