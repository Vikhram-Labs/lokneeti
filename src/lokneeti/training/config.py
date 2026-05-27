"""
lokneeti.training.config
========================
Dataclass-based training configuration with YAML loading support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration for Lokneeti-3B QLoRA."""

    # Paths
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "./outputs/lokneeti-3b-qlora"
    checkpoint_dir: str = "./checkpoints"
    run_name: str = "lokneeti-3b-v1"

    # Hyperparameters
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Memory
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_8bit"
    dataloader_num_workers: int = 2

    # Sequence
    max_seq_length: int = 2048
    packing: bool = False

    # Evaluation & Saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Logging
    logging_dir: str = "./logs"
    logging_steps: int = 10
    report_to: str = "wandb"
    seed: int = 42

    # Dataset
    dataset_text_field: str = "text"
    max_samples: Optional[int] = None

    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        cfg = cls()
        train_data = data.get("training", {})
        for k, v in train_data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        if "lora" in data:
            for k, v in data["lora"].items():
                if hasattr(cfg.lora, k):
                    setattr(cfg.lora, k, v)

        if "quantization" in data:
            for k, v in data["quantization"].items():
                if hasattr(cfg.quantization, k):
                    setattr(cfg.quantization, k, v)

        return cfg

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
