"""
scripts/eval/evaluate.py
=========================
Governance reasoning evaluation script for Lokneeti-3B.

Usage:
  python scripts/eval/evaluate.py \\
      --model vikhram-labs/Lokneeti-3B \\
      --adapter outputs/lokneeti-3b-qlora/final_adapter \\
      --test data/final/test.jsonl \\
      --output outputs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.evaluation.evaluator import GovernanceEvaluator
from lokneeti.inference.pipeline import LoknetiPipeline
from lokneeti.schemas.datasets import Language, TaskType
from lokneeti.utils.logging import configure_root_logger, get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Lokneeti-3B governance reasoning")
    ap.add_argument("--model",   type=str, default="vikhram-labs/Lokneeti-3B")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--test",    type=str, default="data/final/test.jsonl")
    ap.add_argument("--output",  type=str, default="outputs/eval_report.json")
    ap.add_argument("--max",     type=int, default=None, help="Limit evaluation examples")
    ap.add_argument("--no-inference", action="store_true",
                    help="Skip model inference — score existing outputs in test file")
    return ap.parse_args()


def load_test_examples(test_path: Path, max_n: int = None) -> list:
    examples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    if max_n:
        examples = examples[:max_n]
    log.info(f"Loaded {len(examples)} test examples from {test_path}")
    return examples


def main() -> None:
    configure_root_logger()
    args = parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        log.error(f"Test file not found: {test_path}")
        sys.exit(1)

    examples = load_test_examples(test_path, max_n=args.max)

    # ── Run model inference (unless skipped) ─────────────────────────────
    if not args.no_inference:
        log.info(f"Loading model: {args.model}")
        pipeline = LoknetiPipeline.from_pretrained(
            model_id=args.model,
            adapter_path=args.adapter,
        )

        log.info("Running inference on test examples...")
        for ex in tqdm(examples, desc="Generating outputs"):
            instruction = ex.get("instruction", "Analyse this governance scenario.")
            inp         = ex.get("input", "")
            response    = pipeline.analyze(
                policy_text=inp or instruction,
                instruction=instruction if inp else None,
            )
            ex["model_output"] = response.output
            ex["id"] = ex.get("id", ex.get("source_id", "unknown"))
    else:
        # Assume model_output field already exists (e.g. from prior run)
        for i, ex in enumerate(examples):
            ex["model_output"] = ex.get("model_output", ex.get("output", ""))
            ex["id"] = str(i)

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("Running evaluation metrics...")
    evaluator = GovernanceEvaluator()

    eval_examples = []
    for ex in examples:
        eval_examples.append({
            "id":               ex.get("id", "unknown"),
            "model_output":     ex.get("model_output", ""),
            "reference_output": ex.get("output", ""),
            "task_type":        ex.get("task_type", "constitutional_qa"),
            "language":         ex.get("language", "en"),
            "article_refs":     ex.get("article_refs", []),
        })

    results = evaluator.evaluate_batch(eval_examples)
    metrics = evaluator.compute_aggregate(results)

    # ── Report ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("  LOKNEETI-3B EVALUATION REPORT")
    log.info("=" * 50)
    log.info(f"  Examples evaluated:         {metrics.num_examples}")
    log.info(f"  Avg ROUGE-L:                {metrics.avg_rouge_l:.4f}")
    log.info(f"  Constitutional consistency: {metrics.avg_constitutional_consistency:.4f}")
    log.info(f"  Article recall:             {metrics.avg_article_recall:.4f}")
    log.info(f"  Hallucination rate:         {metrics.hallucination_rate:.4f}")
    log.info("-" * 50)
    log.info("  Per-task ROUGE-L:")
    for task, score in metrics.task_breakdown.items():
        log.info(f"    {task:35s}: {score:.4f}")
    log.info("=" * 50)

    # Save report
    evaluator.save_report(metrics, results, Path(args.output))
    log.info(f"✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
