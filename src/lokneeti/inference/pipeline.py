"""
lokneeti.inference.pipeline
============================
Lokneeti-3B inference pipeline.

Supports:
  - HuggingFace model (full or LoRA adapter)
  - CPU fallback for demo environments
  - Structured Constitutional Chain Compression output
  - Streaming generation
  - Batch inference

Usage::

    pipeline = LoknetiPipeline.from_pretrained("vikhram-labs/Lokneeti-3B")
    result = pipeline.analyze("A welfare scheme excludes biometric-failure citizens.")
    print(result.output)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch

from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Lokneeti, a constitutional governance reasoning system developed by Vikhram Labs. "
    "Your purpose is to analyze Indian public policy, detect constitutional risks, and reason "
    "about democratic governance using structured Constitutional Chain Compression methodology. "
    "You do not engage in casual conversation. You produce precise, structured governance analysis."
)


@dataclass
class GovernanceResponse:
    """Structured response from the Lokneeti inference pipeline."""
    input_text: str
    output: str
    model_id: str
    latency_seconds: float
    tokens_generated: int = 0
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)


class LoknetiPipeline:
    """
    Lokneeti-3B inference pipeline.

    Wraps a HuggingFace CausalLM model with governance-specific
    prompt formatting and structured output parsing.

    Supports both full model and LoRA adapter inference.
    CPU fallback is automatic when CUDA is unavailable.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_id: str = "lokneeti-3b",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"LoknetiPipeline ready — device: {self.device}, model: {model_id}")

    # ------------------------------------------------------------------ #
    # Factory Methods                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        adapter_path: Optional[str | Path] = None,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> "LoknetiPipeline":
        """
        Load the pipeline from a HuggingFace Hub model ID or local path.

        Args:
            model_id:     Hub repo (e.g. 'vikhram-labs/Lokneeti-3B') or local path.
            adapter_path: Optional path to a LoRA adapter directory.
            load_in_4bit: Load in 4-bit quantization (requires bitsandbytes).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info(f"Loading Lokneeti model: {model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }

        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel
            log.info(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, model_id=model_id, **kwargs)

    # ------------------------------------------------------------------ #
    # Public Inference API                                                  #
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        policy_text: str,
        instruction: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> GovernanceResponse:
        """
        Analyse a governance scenario or policy text.

        Args:
            policy_text:   The policy document or scenario to analyse.
            instruction:   Optional task instruction (defaults to generic analysis).
            system_prompt: System prompt (defaults to Lokneeti system prompt).

        Returns:
            GovernanceResponse with structured analysis output.
        """
        if instruction is None:
            instruction = (
                "Analyse the following governance scenario using Constitutional Chain "
                "Compression. Identify constitutional risks, enumerate the reasoning "
                "chain, and provide a structured governance conclusion."
            )

        prompt = self._format_prompt(instruction, policy_text, system_prompt)
        start = time.time()

        output_text, tokens = self._generate(prompt)
        latency = time.time() - start

        return GovernanceResponse(
            input_text=policy_text,
            output=output_text,
            model_id=self.model_id,
            latency_seconds=round(latency, 3),
            tokens_generated=tokens,
            device=self.device,
        )

    def detect_constitutional_risk(self, policy_text: str) -> GovernanceResponse:
        """Shortcut for constitutional risk detection."""
        return self.analyze(
            policy_text=policy_text,
            instruction=(
                "Detect any constitutional risks in the following policy text. "
                "Identify implicated constitutional articles and assess risk level."
            ),
        )

    def abstract_grievance(self, grievance_text: str) -> GovernanceResponse:
        """Shortcut for citizen grievance abstraction."""
        return self.analyze(
            policy_text=grievance_text,
            instruction=(
                "Compress the following citizen grievance into a structured constitutional "
                "rights violation report with recommended remedies."
            ),
        )

    def batch_analyze(self, texts: List[str]) -> List[GovernanceResponse]:
        """Analyze multiple governance scenarios."""
        return [self.analyze(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Private: Prompt & Generation                                          #
    # ------------------------------------------------------------------ #
    def _format_prompt(
        self, instruction: str, input_text: str, system_prompt: str
    ) -> str:
        """Format a ChatML-style prompt for Qwen2.5/Phi-3."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{input_text}".strip()},
        ]
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Manual fallback
        parts = [f"<|im_start|>system\n{system_prompt}\n<|im_end|>"]
        parts.append(f"<|im_start|>user\n{instruction}\n\n{input_text}\n<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> tuple[str, int]:
        """Run generation and return (output_text, tokens_generated)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800,
        )
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text.strip(), len(generated_ids)
