"""
lokneeti.evaluation.evaluator
==============================
Governance reasoning evaluation suite for Lokneeti-3B.

Evaluates:
  1. Constitutional reasoning accuracy (article recall)
  2. Governance consistency (output structure compliance)
  3. Hallucination rate (fabricated article references)
  4. Multilingual robustness (across Hindi, Bengali, Tamil, etc.)
  5. Policy contradiction detection accuracy
  6. ROUGE-L overlap with reference outputs

Outputs per-example and aggregate JSON reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from lokneeti.schemas.datasets import EvaluationResult, Language, TaskType
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# Known constitutional articles for hallucination detection
VALID_ARTICLES = {f"Article {i}" for i in range(1, 400)} | {
    "Article 21A", "Article 300A", "Article 370",
    "First Schedule", "Second Schedule", "Third Schedule",
    "Fourth Schedule", "Fifth Schedule", "Sixth Schedule",
    "Seventh Schedule", "Eighth Schedule", "Ninth Schedule",
    "Tenth Schedule", "Eleventh Schedule", "Twelfth Schedule",
}


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics across all examples."""
    num_examples: int = 0
    avg_rouge_l: float = 0.0
    avg_constitutional_consistency: float = 0.0
    avg_article_recall: float = 0.0
    hallucination_rate: float = 0.0
    task_breakdown: Dict[str, float] = field(default_factory=dict)
    language_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class GovernanceEvaluator:
    """
    Benchmark evaluator for Lokneeti-3B governance reasoning.

    Usage::

        evaluator = GovernanceEvaluator()
        results = evaluator.evaluate_batch(model_outputs, references)
        metrics = evaluator.compute_aggregate(results)
        evaluator.save_report(metrics, results, "outputs/eval_report.json")
    """

    def __init__(self) -> None:
        self._rouge = self._load_rouge()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #
    def evaluate_single(
        self,
        example_id: str,
        model_output: str,
        reference_output: str,
        task_type: TaskType = TaskType.CONSTITUTIONAL_QA,
        language: Language = Language.ENGLISH,
        reference_articles: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate a single model output against a reference."""

        # 1. ROUGE-L
        rouge_l = self._compute_rouge_l(model_output, reference_output)

        # 2. Constitutional consistency (output structure check)
        consistency = self._compute_constitutional_consistency(model_output, task_type)

        # 3. Hallucination detection
        hallucination = self._detect_hallucination(model_output)

        # 4. Article recall
        article_recall = self._compute_article_recall(
            model_output, reference_articles or []
        )

        return EvaluationResult(
            example_id=example_id,
            task_type=task_type,
            model_output=model_output,
            reference_output=reference_output,
            rouge_l=rouge_l,
            constitutional_consistency=consistency,
            hallucination_flag=hallucination,
            article_recall=article_recall,
            language=language,
        )

    def evaluate_batch(
        self,
        examples: List[Dict],
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of examples.

        Each example dict should have:
          - id, model_output, reference_output
          - task_type (optional), language (optional), article_refs (optional)
        """
        results: List[EvaluationResult] = []
        iterator = tqdm(examples, desc="Evaluating", unit="ex") if show_progress else examples

        for ex in iterator:
            result = self.evaluate_single(
                example_id=ex.get("id", "unknown"),
                model_output=ex["model_output"],
                reference_output=ex["reference_output"],
                task_type=TaskType(ex.get("task_type", "constitutional_qa")),
                language=Language(ex.get("language", "en")),
                reference_articles=ex.get("article_refs", []),
            )
            results.append(result)

        return results

    def compute_aggregate(self, results: List[EvaluationResult]) -> AggregateMetrics:
        """Compute aggregate metrics from a list of evaluation results."""
        if not results:
            return AggregateMetrics()

        n = len(results)
        metrics = AggregateMetrics(
            num_examples=n,
            avg_rouge_l=sum(r.rouge_l for r in results) / n,
            avg_constitutional_consistency=sum(
                r.constitutional_consistency for r in results
            ) / n,
            avg_article_recall=sum(r.article_recall for r in results) / n,
            hallucination_rate=sum(1 for r in results if r.hallucination_flag) / n,
        )

        # Task breakdown
        task_groups: Dict[str, List[float]] = {}
        for r in results:
            task_groups.setdefault(r.task_type.value, []).append(r.rouge_l)
        metrics.task_breakdown = {k: sum(v) / len(v) for k, v in task_groups.items()}

        # Language breakdown
        lang_groups: Dict[str, List[float]] = {}
        for r in results:
            lang_groups.setdefault(r.language.value, []).append(r.rouge_l)
        metrics.language_breakdown = {k: sum(v) / len(v) for k, v in lang_groups.items()}

        log.info(
            f"📊 Evaluation complete — "
            f"ROUGE-L: {metrics.avg_rouge_l:.3f}, "
            f"Consistency: {metrics.avg_constitutional_consistency:.3f}, "
            f"Hallucination rate: {metrics.hallucination_rate:.3f}"
        )
        return metrics

    def save_report(
        self,
        metrics: AggregateMetrics,
        results: List[EvaluationResult],
        output_path: str | Path,
    ) -> None:
        """Save a full evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "aggregate": metrics.to_dict(),
            "examples": [
                {
                    "id": r.example_id,
                    "task_type": r.task_type.value,
                    "language": r.language.value,
                    "rouge_l": round(r.rouge_l, 4),
                    "constitutional_consistency": round(r.constitutional_consistency, 4),
                    "hallucination_flag": r.hallucination_flag,
                    "article_recall": round(r.article_recall, 4),
                    "model_output": r.model_output[:300],
                    "reference_output": r.reference_output[:300],
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        log.info(f"✅ Evaluation report saved to {output_path}")

    # ------------------------------------------------------------------ #
    # Private Metrics                                                       #
    # ------------------------------------------------------------------ #
    def _compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        """Compute ROUGE-L F1 score."""
        if self._rouge is None:
            return self._rouge_l_fallback(hypothesis, reference)
        try:
            scores = self._rouge.compute(
                predictions=[hypothesis],
                references=[reference],
                rouge_types=["rougeL"],
            )
            return float(scores["rougeL"])
        except Exception:
            return self._rouge_l_fallback(hypothesis, reference)

    @staticmethod
    def _rouge_l_fallback(hypothesis: str, reference: str) -> float:
        """Pure-Python LCS-based ROUGE-L fallback."""
        h_tokens = hypothesis.lower().split()
        r_tokens = reference.lower().split()
        if not h_tokens or not r_tokens:
            return 0.0

        # LCS length
        m, n = len(r_tokens), len(h_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if r_tokens[i - 1] == h_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / n if n else 0
        recall = lcs / m if m else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _compute_constitutional_consistency(output: str, task_type: TaskType) -> float:
        """
        Check whether the model output contains expected structural elements
        for the given task type. Returns a score in [0, 1].
        """
        output_lower = output.lower()
        checks: List[bool] = []

        if task_type == TaskType.CONSTITUTIONAL_QA:
            checks = [
                "article" in output_lower,
                any(k in output_lower for k in ["right", "fundamental", "guarantee"]),
            ]
        elif task_type == TaskType.POLICY_CONTRADICTION:
            checks = [
                any(k in output_lower for k in ["conflict", "contradiction", "inconsistent"]),
                "article" in output_lower,
                any(k in output_lower for k in ["recommend", "risk", "violation"]),
            ]
        elif task_type == TaskType.CONSTITUTIONAL_CHAIN:
            checks = [
                "chain" in output_lower or "->" in output,
                "risk level" in output_lower,
                "conclusion" in output_lower,
                "article" in output_lower,
            ]
        elif task_type == TaskType.WELFARE_RISK_ANALYSIS:
            checks = [
                any(k in output_lower for k in ["risk", "exclusion", "vulnerable"]),
                "article" in output_lower,
                any(k in output_lower for k in ["recommend", "safeguard", "remedy"]),
            ]
        elif task_type == TaskType.GRIEVANCE_ABSTRACTION:
            checks = [
                any(k in output_lower for k in ["grievance", "violation", "right"]),
                any(k in output_lower for k in ["remedy", "rti", "petition", "authority"]),
            ]
        else:
            checks = ["article" in output_lower]

        return sum(checks) / len(checks) if checks else 0.5

    @staticmethod
    def _detect_hallucination(output: str) -> bool:
        """
        Detect hallucinated constitutional article references.

        Flags the output if it contains article references that do not
        exist in the Indian Constitution.
        """
        pattern = re.compile(r"\bArticle\s+(\d+[A-Z]?)\b", re.IGNORECASE)
        found = pattern.findall(output)
        for num in found:
            ref = f"Article {num}"
            if ref not in VALID_ARTICLES:
                log.debug(f"Hallucination detected: {ref!r} not in valid articles")
                return True
        return False

    @staticmethod
    def _compute_article_recall(output: str, reference_articles: List[str]) -> float:
        """
        Compute recall of expected constitutional articles in the output.

        Returns 1.0 if no reference articles exist (nothing to recall).
        """
        if not reference_articles:
            return 1.0

        output_lower = output.lower()
        recalled = sum(
            1 for art in reference_articles if art.lower() in output_lower
        )
        return recalled / len(reference_articles)

    @staticmethod
    def _load_rouge():
        """Load HuggingFace evaluate ROUGE metric — graceful fallback."""
        try:
            import evaluate
            return evaluate.load("rouge")
        except Exception:
            log.warning("evaluate/rouge not available — using Python LCS fallback")
            return None
