"""
lokneeti.schemas.datasets
=========================
Pydantic v2 dataclasses and models for all Lokneeti data structures.

This module defines the canonical schemas used throughout:
  - Data collection & cleaning
  - Synthetic instruction generation
  - Constitutional Chain Compression (C³) intermediate format
  - Training dataset format (Alpaca-style)
  - Evaluation results

All schemas are JSON-serialisable and HuggingFace datasets-compatible.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enums                                                                         #
# --------------------------------------------------------------------------- #
class GovernanceDomain(str, Enum):
    """Thematic domain tags for governance corpus items."""
    CONSTITUTIONAL       = "constitutional"
    WELFARE              = "welfare"
    FEDERAL              = "federal"
    PARLIAMENTARY        = "parliamentary"
    JUDICIARY            = "judiciary"
    GRIEVANCE            = "grievance"
    POLICY_ANALYSIS      = "policy_analysis"
    BUDGET_FINANCE       = "budget_finance"
    RIGHTS_LIBERTIES     = "rights_liberties"
    MULTILINGUAL         = "multilingual"
    RTI                  = "rti"


class TaskType(str, Enum):
    """Instruction-tuning task categories."""
    CONSTITUTIONAL_QA       = "constitutional_qa"
    POLICY_CONTRADICTION    = "policy_contradiction"
    WELFARE_RISK_ANALYSIS   = "welfare_risk_analysis"
    FEDERAL_CONFLICT        = "federal_conflict"
    INCLUSION_ANALYSIS      = "inclusion_analysis"
    GRIEVANCE_ABSTRACTION   = "grievance_abstraction"
    IMPLEMENTATION_GAP      = "implementation_gap"
    INSTITUTIONAL_REASONING = "institutional_reasoning"
    CONSTITUTIONAL_CHAIN    = "constitutional_chain"


class Language(str, Enum):
    """Supported language codes (ISO 639-1)."""
    ENGLISH    = "en"
    HINDI      = "hi"
    BENGALI    = "bn"
    TELUGU     = "te"
    TAMIL      = "ta"
    GUJARATI   = "gu"
    KANNADA    = "kn"
    MALAYALAM  = "ml"
    MARATHI    = "mr"
    PUNJABI    = "pa"
    URDU       = "ur"
    ODIA       = "or"


# --------------------------------------------------------------------------- #
# Raw Corpus Items                                                               #
# --------------------------------------------------------------------------- #
class GovernanceExample(BaseModel):
    """
    A single raw document/chunk from any governance corpus source.

    This is the universal intermediate format before instruction-tuning.
    All scrapers and parsers emit this schema.
    """

    id: str = Field(..., description="Unique deterministic ID (e.g. sha256 prefix)")
    source: str = Field(..., description="Data source name (e.g. 'constitution', 'parliament')")
    url: Optional[str] = Field(None, description="Original URL of the document")
    title: Optional[str] = Field(None, description="Document or section title")
    text: str = Field(..., min_length=10, description="Raw document text")
    language: Language = Field(Language.ENGLISH, description="Primary language")
    domain: GovernanceDomain = Field(..., description="Governance domain tag")
    article_refs: List[str] = Field(
        default_factory=list,
        description="Constitutional article references (e.g. ['Article 21', 'Article 14'])",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary source-specific metadata (year, ministry, etc.)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text field must not be blank")
        return v.strip()

    def to_alpaca(self) -> "SyntheticInstruction":
        """Convert a raw governance example to a minimal Alpaca instruction."""
        return SyntheticInstruction(
            instruction=f"Summarise the following governance document from source '{self.source}'.",
            input=self.text[:2000],
            output="",
            task_type=TaskType.CONSTITUTIONAL_QA,
            domain=self.domain,
            language=self.language,
            article_refs=self.article_refs,
            source_id=self.id,
        )


# --------------------------------------------------------------------------- #
# Instruction-Tuning Format (Alpaca-style)                                      #
# --------------------------------------------------------------------------- #
class SyntheticInstruction(BaseModel):
    """
    Alpaca-style instruction-tuning example for Lokneeti fine-tuning.

    Follows the standard format:
        { instruction, input, output }
    with Lokneeti-specific governance metadata fields.
    """

    instruction: str = Field(..., description="Task instruction for the model")
    input: str = Field(default="", description="Optional input context or policy text")
    output: str = Field(..., description="Target governance reasoning output")
    task_type: TaskType = Field(..., description="Task category")
    domain: GovernanceDomain = Field(GovernanceDomain.CONSTITUTIONAL)
    language: Language = Field(Language.ENGLISH)
    article_refs: List[str] = Field(default_factory=list)
    source_id: Optional[str] = Field(None, description="Traceable source document ID")
    quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence/quality score assigned during generation",
    )
    is_synthetic: bool = Field(default=True, description="True for template-generated examples")

    @field_validator("instruction", "output")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    def to_text(self, system_prompt: str = "") -> str:
        """
        Render as a single training string using the ChatML template
        compatible with Qwen2.5 and Phi-3.

        Example output::

            <|im_start|>system
            You are Lokneeti...
            <|im_end|>
            <|im_start|>user
            {instruction}

            {input}
            <|im_end|>
            <|im_start|>assistant
            {output}
            <|im_end|>
        """
        parts: List[str] = []

        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}\n<|im_end|>")

        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"

        parts.append(f"<|im_start|>user\n{user_content}\n<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{self.output}\n<|im_end|>")

        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Constitutional Chain Compression (C³) Schema                                  #
# --------------------------------------------------------------------------- #
class ChainNode(BaseModel):
    """A single reasoning hop in the Constitutional Chain Compression graph."""

    concept: str = Field(..., description="Constitutional concept or right (e.g. 'Article_21')")
    relation: str = Field(..., description="Reasoning relation (e.g. '->', 'excludes', 'enables')")
    target: str = Field(..., description="Target concept or policy element")
    weight: float = Field(default=1.0, description="Salience weight of this reasoning hop")


class ConstitutionalChain(BaseModel):
    """
    Constitutional Chain Compression (C³) intermediate reasoning structure.

    This is Lokneeti's novel reasoning schema — a compact symbolic graph
    that bridges raw policy input and structured governance output.

    The chain encodes:
        1. Input policy scenario
        2. Symbolic intermediate reasoning hops
        3. Constitutional conclusion

    Example::

        input_scenario: "Welfare scheme excludes biometric-failure citizens"
        chain_nodes:
          - Right_to_food -> exclusion_risk
          - Article_21 -> welfare_access
          - Implementation_gap -> vulnerable_groups
        conclusion: "Constitutional vulnerability under Article 21..."
    """

    input_scenario: str = Field(..., description="Raw policy scenario or grievance text")
    chain_nodes: List[ChainNode] = Field(
        ...,
        min_length=1,
        description="Ordered list of constitutional reasoning hops",
    )
    conclusion: str = Field(..., description="Final governance reasoning conclusion")
    risk_level: str = Field(
        default="medium",
        description="Constitutional risk level: low | medium | high | critical",
    )
    articles_implicated: List[str] = Field(
        default_factory=list,
        description="List of implicated constitutional articles",
    )

    @model_validator(mode="after")
    def validate_risk_level(self) -> "ConstitutionalChain":
        valid = {"low", "medium", "high", "critical"}
        if self.risk_level not in valid:
            raise ValueError(f"risk_level must be one of {valid}")
        return self

    def to_instruction(self, task_type: TaskType = TaskType.CONSTITUTIONAL_CHAIN) -> SyntheticInstruction:
        """Convert a C³ chain into a training instruction-output pair."""
        # Format the symbolic chain as compact text
        chain_text = "\n".join(
            f"  {node.concept} {node.relation} {node.target}"
            for node in self.chain_nodes
        )

        output = (
            f"[Constitutional Chain Compression]\n"
            f"CHAIN:\n{chain_text}\n\n"
            f"RISK LEVEL: {self.risk_level.upper()}\n"
            f"ARTICLES: {', '.join(self.articles_implicated) or 'None explicitly cited'}\n\n"
            f"CONCLUSION:\n{self.conclusion}"
        )

        return SyntheticInstruction(
            instruction=(
                "Analyse the following governance scenario using Constitutional Chain "
                "Compression. Identify constitutional risks, enumerate the reasoning "
                "chain, and provide a structured governance conclusion."
            ),
            input=self.input_scenario,
            output=output,
            task_type=task_type,
            domain=GovernanceDomain.CONSTITUTIONAL,
            article_refs=self.articles_implicated,
            is_synthetic=True,
        )


# --------------------------------------------------------------------------- #
# Dataset Configuration                                                         #
# --------------------------------------------------------------------------- #
class DatasetConfig(BaseModel):
    """Runtime dataset configuration (loaded from data_config.yaml)."""

    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    synthetic_dir: str = "./data/synthetic"
    final_dir: str = "./data/final"
    max_seq_length: int = 2048
    train_split: float = 0.90
    val_split: float = 0.05
    test_split: float = 0.05
    seed: int = 42

    @model_validator(mode="after")
    def validate_splits(self) -> "DatasetConfig":
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Dataset splits must sum to 1.0, got {total}")
        return self


# --------------------------------------------------------------------------- #
# Evaluation                                                                    #
# --------------------------------------------------------------------------- #
class EvaluationResult(BaseModel):
    """Single benchmark evaluation result for one model response."""

    example_id: str
    task_type: TaskType
    model_output: str
    reference_output: str
    rouge_l: float = Field(0.0, ge=0.0, le=1.0)
    constitutional_consistency: float = Field(0.0, ge=0.0, le=1.0)
    hallucination_flag: bool = False
    article_recall: float = Field(0.0, ge=0.0, le=1.0)
    language: Language = Language.ENGLISH
    metadata: Dict[str, Any] = Field(default_factory=dict)
