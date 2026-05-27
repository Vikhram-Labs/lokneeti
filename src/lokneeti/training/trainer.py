"""
lokneeti.training.trainer
=========================
QLoRA fine-tuning trainer for Lokneeti-3B.

Wraps TRL SFTTrainer with:
  - Unsloth acceleration (2x speedup on T4)
  - bitsandbytes 4-bit quantization
  - PEFT LoRA adapters
  - Automatic checkpoint resume
  - VRAM-safe configuration
  - Gradient checkpointing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from lokneeti.training.config import TrainingConfig
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# System prompt injected into every training example
SYSTEM_PROMPT = (
    "You are Lokneeti, a constitutional governance reasoning system developed by Vikhram Labs. "
    "Your purpose is to analyze Indian public policy, detect constitutional risks, and reason "
    "about democratic governance using structured Constitutional Chain Compression methodology. "
    "You do not engage in casual conversation. You produce precise, structured governance analysis."
)


class LoknetiTrainer:
    """
    Lokneeti-3B QLoRA Training Orchestrator.

    Handles the complete training lifecycle:
      1. Load base model with 4-bit quantization
      2. Apply LoRA adapters
      3. Prepare dataset
      4. Run SFTTrainer
      5. Save adapter and optionally merge

    Unsloth is used when available for 2x speedup.
    Falls back to standard PEFT + Transformers if Unsloth is not installed.

    Usage::

        config = TrainingConfig.from_yaml("configs/training_config.yaml")
        trainer = LoknetiTrainer(config)
        trainer.train("data/final/train.jsonl")
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self._use_unsloth = self._check_unsloth()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_data_path: str | Path,
        eval_data_path: Optional[str | Path] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Run the complete QLoRA fine-tuning pipeline.

        Args:
            train_data_path:        Path to training JSONL file.
            eval_data_path:         Optional evaluation JSONL file.
            resume_from_checkpoint: Path to checkpoint directory to resume from.
        """
        log.info(f"🚀 Starting Lokneeti-3B training — base: {self.config.base_model}")
        log.info(f"   Backend: {'Unsloth' if self._use_unsloth else 'Standard PEFT'}")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Load datasets
        train_dataset = self._load_dataset(train_data_path)
        eval_dataset = self._load_dataset(eval_data_path) if eval_data_path else None

        # Build training arguments
        training_args = self._build_training_arguments()

        # Build SFTTrainer
        sft_trainer = self._build_sft_trainer(
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            log.info(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")

        # Train
        log.info("⚡ Training started...")
        sft_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save
        self._save(sft_trainer)
        log.info("✅ Training complete!")

    def save_merged(self, output_dir: str | Path) -> None:
        """
        Merge LoRA adapter weights into the base model and save.

        Use this to create a standalone model for GGUF export.
        WARNING: Requires ~12GB RAM. Not recommended on Colab.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Merging LoRA adapters → {output_dir}")
        merged = self.model.merge_and_unload()
        merged.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        log.info(f"✅ Merged model saved to {output_dir}")

    # ------------------------------------------------------------------ #
    # Private: Model Loading                                               #
    # ------------------------------------------------------------------ #
    def _load_model_and_tokenizer(self) -> None:
        cfg = self.config

        if self._use_unsloth:
            self._load_with_unsloth()
        else:
            self._load_with_peft()

    def _load_with_unsloth(self) -> None:
        """Load model using Unsloth for accelerated training."""
        from unsloth import FastLanguageModel

        cfg = self.config
        log.info("Loading model with Unsloth (2× speedup)...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            dtype=None,  # Auto-detect (bfloat16 on Ampere, float16 on T4)
            load_in_4bit=cfg.quantization.load_in_4bit,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=cfg.lora.r,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC
            random_state=cfg.seed,
        )
        log.info(f"✅ Unsloth model loaded — LoRA r={cfg.lora.r}")

    def _load_with_peft(self) -> None:
        """Standard PEFT + bitsandbytes loading fallback."""
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType

        cfg = self.config
        log.info("Loading model with standard PEFT...")

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()

        # LoRA
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("✅ PEFT model loaded")

    # ------------------------------------------------------------------ #
    # Private: Dataset                                                      #
    # ------------------------------------------------------------------ #
    def _load_dataset(self, path: str | Path) -> Dataset:
        """Load a JSONL dataset file into a HuggingFace Dataset."""
        path = Path(path)
        log.info(f"Loading dataset: {path}")
        dataset = load_dataset("json", data_files=str(path), split="train")

        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

        log.info(f"Dataset loaded: {len(dataset)} examples")
        return dataset

    # ------------------------------------------------------------------ #
    # Private: Training Arguments                                           #
    # ------------------------------------------------------------------ #
    def _build_training_arguments(self) -> TrainingArguments:
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            optim=cfg.optim,
            gradient_checkpointing=cfg.gradient_checkpointing,
            evaluation_strategy=cfg.evaluation_strategy,
            eval_steps=cfg.eval_steps,
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            logging_dir=cfg.logging_dir,
            logging_steps=cfg.logging_steps,
            report_to=cfg.report_to,
            run_name=cfg.run_name,
            seed=cfg.seed,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
        )

    # ------------------------------------------------------------------ #
    # Private: SFTTrainer                                                   #
    # ------------------------------------------------------------------ #
    def _build_sft_trainer(
        self,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> SFTTrainer:
        cfg = self.config
        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=cfg.dataset_text_field,
            max_seq_length=cfg.max_seq_length,
            packing=cfg.packing,
            args=training_args,
        )

    # ------------------------------------------------------------------ #
    # Private: Save                                                         #
    # ------------------------------------------------------------------ #
    def _save(self, trainer: SFTTrainer) -> None:
        cfg = self.config
        adapter_path = Path(cfg.output_dir) / "final_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        log.info(f"✅ Adapter saved to: {adapter_path}")

    @staticmethod
    def _check_unsloth() -> bool:
        try:
            import unsloth  # noqa: F401
            return True
        except ImportError:
            log.warning("Unsloth not found — using standard PEFT (slower)")
            return False
