"""
lokneeti.reasoning.constitutional_chain
========================================
Constitutional Chain Compression (C³) Engine

This module implements Lokneeti's core novel reasoning paradigm:

    **Constitutional Chain Compression (C³)**

C³ is a compact symbolic reasoning schema for governance AI that encodes
the pathway from a raw policy scenario to a constitutional conclusion via
an ordered sequence of conceptual reasoning hops.

Each hop is a triplet:
    (Constitutional_Concept) → (Relation) → (Policy_Element)

The chain is then compressed into a structured governance conclusion.

This schema enables:
  - Reproducible constitutional risk detection
  - Explainable governance reasoning
  - Structured synthetic data generation
  - Hallucination-resistant policy analysis

Reference:
    Lokneeti-3B Technical Report, Vikhram Labs, 2024.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from lokneeti.schemas.datasets import (
    ChainNode,
    ConstitutionalChain,
    GovernanceDomain,
    Language,
    SyntheticInstruction,
    TaskType,
)
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Constitutional Knowledge Base                                                 #
# --------------------------------------------------------------------------- #

# Core constitutional rights mapped to their implications
CONSTITUTIONAL_RIGHTS: Dict[str, Dict[str, str | List[str]]] = {
    "Article_14": {
        "right": "Equality before law",
        "risk_markers": ["discrimination", "differential", "exclusion", "arbitrary"],
        "domain": "rights_liberties",
    },
    "Article_15": {
        "right": "Prohibition of discrimination on grounds of religion, race, caste, sex",
        "risk_markers": ["caste", "religion", "sex", "race", "discrimination"],
        "domain": "rights_liberties",
    },
    "Article_16": {
        "right": "Equality of opportunity in public employment",
        "risk_markers": ["employment", "appointment", "reservation", "opportunity"],
        "domain": "rights_liberties",
    },
    "Article_19": {
        "right": "Freedom of speech, assembly, movement",
        "risk_markers": ["speech", "press", "assembly", "movement", "restriction"],
        "domain": "rights_liberties",
    },
    "Article_21": {
        "right": "Right to life and personal liberty",
        "risk_markers": [
            "life", "liberty", "dignity", "health", "food", "shelter",
            "education", "livelihood", "privacy", "exclusion",
        ],
        "domain": "rights_liberties",
    },
    "Article_21A": {
        "right": "Right to education",
        "risk_markers": ["education", "school", "child", "learning", "dropout"],
        "domain": "rights_liberties",
    },
    "Article_22": {
        "right": "Protection against arbitrary arrest",
        "risk_markers": ["arrest", "detention", "custody", "preventive"],
        "domain": "rights_liberties",
    },
    "Article_32": {
        "right": "Right to constitutional remedies",
        "risk_markers": ["remedy", "writ", "petition", "enforcement", "access_to_justice"],
        "domain": "judiciary",
    },
    "Article_39": {
        "right": "Directive Principles — equal pay, welfare",
        "risk_markers": ["pay", "wage", "livelihood", "welfare", "equal"],
        "domain": "welfare",
    },
    "Article_41": {
        "right": "Right to work, education and public assistance",
        "risk_markers": ["work", "unemployment", "disability", "sickness", "assistance"],
        "domain": "welfare",
    },
    "Article_246": {
        "right": "Distribution of legislative powers (Union/State lists)",
        "risk_markers": ["central", "state", "concurrent", "jurisdiction", "federal"],
        "domain": "federal",
    },
    "Article_300A": {
        "right": "Right to property",
        "risk_markers": ["property", "land", "acquisition", "compensation"],
        "domain": "rights_liberties",
    },
    "Seventh_Schedule": {
        "right": "Union, State and Concurrent legislative lists",
        "risk_markers": ["list", "schedule", "legislative", "subject", "power"],
        "domain": "federal",
    },
}

# Governance risk relations used in chain nodes
GOVERNANCE_RELATIONS = [
    "->",           # Implies / leads to
    "excludes",     # Causes exclusion of
    "enables",      # Enables access to
    "violates",     # Directly violates
    "risks",        # Creates constitutional risk for
    "contradicts",  # Policy contradicts
    "protects",     # Constitutional protection for
    "limits",       # Legal limitation on
    "amplifies",    # Amplifies risk for
]

# Risk level thresholds
class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# --------------------------------------------------------------------------- #
# Chain Result                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ChainResult:
    """
    Result of running the Constitutional Chain Compression engine.
    """
    scenario: str
    chain: ConstitutionalChain
    instruction: SyntheticInstruction
    triggered_articles: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence: float = 0.9


# --------------------------------------------------------------------------- #
# Constitutional Chain Engine                                                   #
# --------------------------------------------------------------------------- #
class ConstitutionalChainEngine:
    """
    Constitutional Chain Compression (C³) Engine.

    Converts a raw governance scenario into a structured constitutional
    reasoning chain using a rule-based symbolic knowledge graph.

    This engine powers Lokneeti's synthetic data generation pipeline.
    For inference, the trained model reproduces this reasoning style
    through learned generalization.

    Usage::

        engine = ConstitutionalChainEngine()
        result = engine.analyze(
            "A welfare scheme excludes biometric-failure citizens."
        )
        print(result.chain.conclusion)
        print(result.instruction.to_text(system_prompt))
    """

    def __init__(self) -> None:
        self.kb = CONSTITUTIONAL_RIGHTS
        log.info("ConstitutionalChainEngine initialized with knowledge base "
                 f"({len(self.kb)} constitutional provisions)")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #
    def analyze(self, scenario: str) -> ChainResult:
        """
        Analyze a governance scenario through Constitutional Chain Compression.

        Args:
            scenario: Raw policy scenario text (English or Indic).

        Returns:
            ChainResult with full chain, instruction, and risk assessment.
        """
        # Step 1: Detect implicated constitutional provisions
        triggered = self._detect_constitutional_triggers(scenario)
        if not triggered:
            triggered = [("Article_21", self.kb["Article_21"])]  # Default

        # Step 2: Build reasoning chain nodes
        chain_nodes = self._build_chain_nodes(scenario, triggered)

        # Step 3: Assess risk level
        risk_level = self._assess_risk_level(triggered, chain_nodes)

        # Step 4: Generate constitutional conclusion
        conclusion = self._generate_conclusion(scenario, triggered, chain_nodes, risk_level)

        # Step 5: Assemble ConstitutionalChain
        articles = [art.replace("_", " ") for art, _ in triggered]
        chain = ConstitutionalChain(
            input_scenario=scenario,
            chain_nodes=chain_nodes,
            conclusion=conclusion,
            risk_level=risk_level.value,
            articles_implicated=articles,
        )

        # Step 6: Convert to training instruction
        instruction = chain.to_instruction()

        log.debug(
            f"C³ analysis complete — "
            f"{len(chain_nodes)} hops, risk={risk_level.value}, "
            f"articles={articles}"
        )

        return ChainResult(
            scenario=scenario,
            chain=chain,
            instruction=instruction,
            triggered_articles=articles,
            risk_level=risk_level,
            confidence=min(0.5 + 0.1 * len(triggered), 1.0),
        )

    # ------------------------------------------------------------------ #
    # Private Methods                                                       #
    # ------------------------------------------------------------------ #
    def _detect_constitutional_triggers(
        self, scenario: str
    ) -> List[Tuple[str, dict]]:
        """
        Identify which constitutional provisions are implicated by the scenario.

        Returns a list of (article_key, provision_info) tuples ordered by
        relevance (number of matching risk markers).
        """
        scenario_lower = scenario.lower()
        matches: List[Tuple[str, dict, int]] = []

        for article, info in self.kb.items():
            markers = info.get("risk_markers", [])
            score = sum(1 for m in markers if m in scenario_lower)
            if score > 0:
                matches.append((article, info, score))

        # Sort by score descending, take top 4 articles
        matches.sort(key=lambda x: x[2], reverse=True)
        return [(art, info) for art, info, _ in matches[:4]]

    def _build_chain_nodes(
        self,
        scenario: str,
        triggered: List[Tuple[str, dict]],
    ) -> List[ChainNode]:
        """
        Build the symbolic reasoning chain from triggered constitutional provisions.
        """
        nodes: List[ChainNode] = []
        scenario_lower = scenario.lower()

        # Policy element extraction (simple keyword-based)
        policy_elements = self._extract_policy_elements(scenario)

        for i, (article, info) in enumerate(triggered):
            right = str(info["right"])
            right_slug = right.split(",")[0].replace(" ", "_")[:30]

            if i == 0 and policy_elements:
                # First node: Right → exclusion/risk element
                nodes.append(ChainNode(
                    concept=article,
                    relation="->",
                    target=f"{policy_elements[0]}_risk",
                    weight=1.0,
                ))
            elif i == 1:
                # Second node: Article → welfare/access implication
                nodes.append(ChainNode(
                    concept=article,
                    relation="->",
                    target="constitutional_protection",
                    weight=0.9,
                ))
            else:
                # Subsequent nodes: domain-specific chain
                domain = str(info.get("domain", "governance"))
                nodes.append(ChainNode(
                    concept=article,
                    relation="risks",
                    target=f"{domain}_vulnerability",
                    weight=0.8,
                ))

        # Final node: Implementation gap → vulnerable groups
        if "exclude" in scenario_lower or "deny" in scenario_lower or "fail" in scenario_lower:
            nodes.append(ChainNode(
                concept="Implementation_gap",
                relation="->",
                target="vulnerable_groups",
                weight=1.0,
            ))

        return nodes

    @staticmethod
    def _extract_policy_elements(scenario: str) -> List[str]:
        """Extract simple policy element keywords from a scenario."""
        keywords = [
            "biometric", "aadhaar", "exclusion", "denial", "access",
            "ration", "pension", "healthcare", "education", "subsidy",
            "benefit", "scheme", "grievance", "complaint", "allocation",
            "fund", "corruption", "delay", "audit",
        ]
        found = [k.replace("-", "_") for k in keywords if k in scenario.lower()]
        return found if found else ["policy_element"]

    @staticmethod
    def _assess_risk_level(
        triggered: List[Tuple[str, dict]],
        nodes: List[ChainNode],
    ) -> RiskLevel:
        """Assess the overall constitutional risk level."""
        # Critical articles that elevate risk immediately
        critical_articles = {"Article_21", "Article_14", "Article_32"}
        high_articles = {"Article_15", "Article_16", "Article_21A"}

        triggered_keys = {art for art, _ in triggered}

        if triggered_keys & critical_articles and len(triggered) >= 2:
            return RiskLevel.CRITICAL
        elif triggered_keys & critical_articles:
            return RiskLevel.HIGH
        elif triggered_keys & high_articles:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    @staticmethod
    def _generate_conclusion(
        scenario: str,
        triggered: List[Tuple[str, dict]],
        nodes: List[ChainNode],
        risk_level: RiskLevel,
    ) -> str:
        """Generate a structured governance conclusion from the chain."""
        if not triggered:
            return (
                "Insufficient constitutional triggers detected. Further "
                "legal analysis is required to assess governance risk."
            )

        primary_article = triggered[0][0].replace("_", " ")
        primary_right = str(triggered[0][1]["right"]).split(",")[0]

        risk_phrases = {
            RiskLevel.CRITICAL: (
                f"This policy creates a CRITICAL constitutional vulnerability under "
                f"{primary_article} (Right to {primary_right}). Immediate legislative "
                f"or administrative remedy is required to prevent fundamental rights violation."
            ),
            RiskLevel.HIGH: (
                f"This policy presents HIGH constitutional risk under {primary_article}. "
                f"The implementation gap may cause substantive violation of the right to "
                f"{primary_right.lower()} for affected citizens, warranting urgent review."
            ),
            RiskLevel.MEDIUM: (
                f"This policy exhibits MEDIUM constitutional tension under {primary_article}. "
                f"The right to {primary_right.lower()} may be compromised for marginalized "
                f"groups. Policy amendment with inclusive safeguards is recommended."
            ),
            RiskLevel.LOW: (
                f"LOW constitutional risk detected. The policy touches on {primary_article} "
                f"but current evidence does not indicate systematic fundamental rights violation. "
                f"Monitoring is advisable."
            ),
        }

        conclusion = risk_phrases[risk_level]

        if len(triggered) > 1:
            secondary_articles = [a.replace("_", " ") for a, _ in triggered[1:]]
            conclusion += (
                f" Secondary constitutional implications detected under: "
                f"{', '.join(secondary_articles)}."
            )

        return conclusion
