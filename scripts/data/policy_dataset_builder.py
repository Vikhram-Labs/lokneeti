"""
scripts/data/policy_dataset_builder.py
=======================================
Master dataset builder — assembles, cleans, deduplicates, and
splits the full Lokneeti training corpus from all sources.

Pipeline:
  1. Load all raw JSONL sources from data/raw/
  2. Clean and normalise text
  3. Deduplicate using MinHash
  4. Generate synthetic instruction-tuning data
  5. Merge real + synthetic examples
  6. Split into train/val/test
  7. Save formatted JSONL files to data/final/
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import jsonlines
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.data.cleaner import TextCleaner
from lokneeti.data.deduplicator import MinHashDeduplicator
from lokneeti.reasoning.synthetic_generator import SyntheticDataGenerator, GeneratorConfig
from lokneeti.schemas.datasets import GovernanceExample, SyntheticInstruction
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

RAW_DIR    = Path("./data/raw")
SYNTH_DIR  = Path("./data/synthetic")
FINAL_DIR  = Path("./data/final")

SYSTEM_PROMPT = (
    "You are Lokneeti, a constitutional governance reasoning system developed by Vikhram Labs. "
    "Your purpose is to analyze Indian public policy, detect constitutional risks, and reason "
    "about democratic governance using structured Constitutional Chain Compression methodology. "
    "You do not engage in casual conversation. You produce precise, structured governance analysis."
)

SPLIT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}


def load_raw_examples(raw_dir: Path) -> List[GovernanceExample]:
    """Load all GovernanceExample objects from raw JSONL files."""
    examples: List[GovernanceExample] = []
    jsonl_files = list(raw_dir.glob("*.jsonl"))

    if not jsonl_files:
        log.warning(f"No JSONL files found in {raw_dir} — run scrapers first")
        return []

    for path in tqdm(jsonl_files, desc="Loading raw data"):
        try:
            with jsonlines.open(str(path)) as reader:
                for row in reader:
                    try:
                        examples.append(GovernanceExample(**row))
                    except Exception as e:
                        log.debug(f"Skipping malformed row in {path.name}: {e}")
        except Exception as e:
            log.error(f"Failed to read {path.name}: {e}")

    log.info(f"Loaded {len(examples)} raw governance examples")
    return examples


def convert_to_instructions(
    examples: List[GovernanceExample],
) -> List[SyntheticInstruction]:
    """Convert raw GovernanceExample objects to Alpaca-style instructions."""
    instructions: List[SyntheticInstruction] = []
    for ex in tqdm(examples, desc="Converting to instructions"):
        instr = ex.to_alpaca()
        # Only include if there's enough text for a meaningful output
        if len(ex.text) > 100:
            instr.output = (
                f"Source: {ex.source}\n"
                f"Domain: {ex.domain.value}\n"
                f"Articles Referenced: {', '.join(ex.article_refs) or 'None'}\n\n"
                f"Summary:\n{ex.text[:800]}"
            )
            instructions.append(instr)
    return instructions


def split_dataset(
    examples: List[SyntheticInstruction],
    ratios: dict = SPLIT_RATIOS,
    seed: int = 42,
) -> dict[str, List[SyntheticInstruction]]:
    """Split examples into train/val/test."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def save_split(
    examples: List[SyntheticInstruction],
    output_path: Path,
    system_prompt: str = SYSTEM_PROMPT,
) -> None:
    """Save a dataset split as JSONL with formatted 'text' field."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for ex in examples:
            writer.write({
                "instruction": ex.instruction,
                "input":       ex.input,
                "output":      ex.output,
                "text":        ex.to_text(system_prompt),
                "task_type":   ex.task_type.value,
                "domain":      ex.domain.value,
                "language":    ex.language.value,
                "article_refs": ex.article_refs,
                "is_synthetic": ex.is_synthetic,
            })
    log.info(f"  Saved {len(examples)} examples → {output_path.name}")


def build(
    n_synthetic_per_category: int = 50,
    n_c3_chains: int = 100,
    skip_real: bool = False,
) -> None:
    """
    Run the full dataset build pipeline.

    Args:
        n_synthetic_per_category: Templates per synthetic task category.
        n_c3_chains:              Number of C³ chain examples to generate.
        skip_real:                Skip loading real corpus (synthetic only — fast debug mode).
    """
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    all_instructions: List[SyntheticInstruction] = []

    # ── 1. Load and clean real corpus ────────────────────────────────────
    if not skip_real:
        log.info("📂 Loading real governance corpus...")
        raw_examples = load_raw_examples(RAW_DIR)

        if raw_examples:
            cleaner  = TextCleaner()
            deduper  = MinHashDeduplicator()
            texts    = [ex.text for ex in raw_examples]
            unique_texts = deduper.deduplicate(texts)
            unique_set   = set(unique_texts)
            unique_examples = [ex for ex in raw_examples if ex.text in unique_set]
            log.info(f"After dedup: {len(unique_examples)} examples")
            real_instructions = convert_to_instructions(unique_examples)
            all_instructions.extend(real_instructions)
            log.info(f"Real corpus instructions: {len(real_instructions)}")

    # ── 2. Generate synthetic data ────────────────────────────────────────
    log.info("🤖 Generating synthetic governance instructions...")
    gen = SyntheticDataGenerator(
        config=GeneratorConfig(
            templates_per_category=n_synthetic_per_category,
            c3_scenarios_count=n_c3_chains,
        )
    )
    synthetic = gen.generate_all()

    # Save synthetic separately
    synth_path = SYNTH_DIR / "synthetic_instructions.jsonl"
    gen.save(synthetic, synth_path, system_prompt=SYSTEM_PROMPT)
    all_instructions.extend(synthetic)

    log.info(f"🎯 Total instructions before split: {len(all_instructions)}")

    # ── 3. Split ──────────────────────────────────────────────────────────
    splits = split_dataset(all_instructions)
    for split_name, split_data in splits.items():
        save_split(split_data, FINAL_DIR / f"{split_name}.jsonl")

    # ── 4. Summary ────────────────────────────────────────────────────────
    log.info("✅ Dataset build complete!")
    log.info(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    log.info(f"   Output directory: {FINAL_DIR.resolve()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build Lokneeti training dataset")
    ap.add_argument("--n-synthetic", type=int, default=50)
    ap.add_argument("--n-c3",        type=int, default=100)
    ap.add_argument("--synthetic-only", action="store_true")
    args = ap.parse_args()

    build(
        n_synthetic_per_category=args.n_synthetic,
        n_c3_chains=args.n_c3,
        skip_real=args.synthetic_only,
    )
