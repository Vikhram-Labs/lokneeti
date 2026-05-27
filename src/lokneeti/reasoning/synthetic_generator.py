"""
lokneeti.reasoning.synthetic_generator
=======================================
Synthetic Instruction Dataset Generator for Lokneeti-3B.

Generates ~10,000+ instruction-tuning examples covering:
  1. Constitutional QA
  2. Policy Contradiction Detection
  3. Welfare Risk Analysis
  4. Federal Conflict Reasoning
  5. Inclusion Analysis
  6. Grievance Abstraction
  7. Implementation Gap Detection
  8. Constitutional Chain Compression (C³)

Uses template-based generation for low-compute reproducibility.
All outputs conform to the SyntheticInstruction schema.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import jsonlines
from tqdm.auto import tqdm

from lokneeti.reasoning.constitutional_chain import ConstitutionalChainEngine
from lokneeti.schemas.datasets import (
    GovernanceDomain,
    Language,
    SyntheticInstruction,
    TaskType,
)
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Template Banks
# ─────────────────────────────────────────────────────────────────────────────

CONSTITUTIONAL_QA_TEMPLATES = [
    {
        "instruction": "What does {article} of the Indian Constitution guarantee to citizens?",
        "input": "",
        "output": "{article} guarantees the right to {right}. This is a {type} right enforceable under Article 32.",
    },
    {
        "instruction": "Explain the significance of {article} in the context of Indian welfare policy.",
        "input": "The government is implementing a new scheme for {group}.",
        "output": "Under {article}, the state has a constitutional obligation to ensure {right} for all citizens including {group}. Any scheme that systematically excludes {group} from this right may be challenged as unconstitutional.",
    },
    {
        "instruction": "How does {article} interact with the Directive Principles of State Policy?",
        "input": "",
        "output": "{article} represents a Fundamental Right that is justiciable, while the Directive Principles provide the policy framework. The Supreme Court has held that Fundamental Rights and Directive Principles are complementary, not contradictory, and must be read harmoniously.",
    },
    {
        "instruction": "Identify constitutional protections available to {group} under the Indian Constitution.",
        "input": "{group} are facing exclusion from {scheme} due to {reason}.",
        "output": "The following constitutional protections apply: (1) {article} — Right to {right}, (2) Article 14 — Equality before law, (3) Article 21 — Right to life and dignity. The exclusion may constitute a violation of these fundamental rights and is subject to judicial review.",
    },
]

POLICY_CONTRADICTION_TEMPLATES = [
    {
        "instruction": "Detect any constitutional contradictions in the following policy excerpt.",
        "input": "{policy_text}",
        "output": "Constitutional Analysis:\n\n1. IDENTIFIED CONTRADICTION: {contradiction}\n\n2. IMPLICATED PROVISION: {article} — {right}\n\n3. LEGAL REASONING: {reasoning}\n\n4. RECOMMENDATION: {recommendation}",
    },
    {
        "instruction": "Does the following policy conflict with any provisions of the Indian Constitution?",
        "input": "{policy_text}",
        "output": "Policy Contradiction Assessment:\n\nStatus: CONFLICT DETECTED\n\nPrimary Conflict: The policy's {provision} clause creates tension with {article} which mandates {right}.\n\nSecondary Concern: The implementation mechanism may violate Article 14's equality guarantee by creating differential treatment of {group}.\n\nConstitutional Risk Level: {risk_level}",
    },
    {
        "instruction": "Analyse whether this policy directive is consistent with the constitutional framework.",
        "input": "{policy_text}",
        "output": "Constitutional Consistency Analysis:\n\nThe directive under review {verdict}. Key findings:\n\n(a) The {element} provision {assessment} with {article}.\n(b) Implementation guidelines {compliance} standard constitutional safeguards.\n(c) The policy's {scope} may require legislative backing under Article 246 to avoid federal conflict.",
    },
]

WELFARE_RISK_TEMPLATES = [
    {
        "instruction": "Assess the constitutional risk to welfare beneficiaries in the following scenario.",
        "input": "{scenario}",
        "output": "Welfare Risk Assessment:\n\nBeneficiary Group: {group}\nExclusion Mechanism: {mechanism}\n\nConstitutional Risk:\n• {article} (Right to {right}) — {risk_assessment}\n• Article 14 — {equality_risk}\n\nRisk Level: {risk_level}\n\nRecommendation: {recommendation}",
    },
    {
        "instruction": "Identify marginalized groups at constitutional risk from this welfare scheme design.",
        "input": "{scheme_description}",
        "output": "Inclusion Risk Analysis:\n\nAt-Risk Groups Identified:\n1. {group_1} — Exclusion mechanism: {reason_1}\n2. {group_2} — Exclusion mechanism: {reason_2}\n\nConstitutional Basis for Inclusion:\n• Article 21: Right to {right} cannot be denied due to administrative barriers\n• Article 15(3)/(4): Special provisions for women and socially/educationally backward classes\n\nRecommended Safeguards: {safeguards}",
    },
]

GRIEVANCE_ABSTRACTION_TEMPLATES = [
    {
        "instruction": "Compress the following citizen grievance into a structured constitutional rights violation report.",
        "input": "{grievance_text}",
        "output": "Grievance Abstraction Report:\n\nGrievance Summary: {summary}\nAffected Right: {article} — {right}\nGovernment Entity Responsible: {entity}\nNature of Violation: {violation_type}\nConstitutional Remedy Available: {remedy}\nPriority Level: {priority}",
    },
    {
        "instruction": "Convert this citizen complaint into a formal RTI-compatible governance intelligence report.",
        "input": "{complaint}",
        "output": "RTI Governance Intelligence Report:\n\n1. MATTER: {matter}\n2. CONSTITUTIONAL FOUNDATION: {article}\n3. DEPARTMENT CONCERNED: {department}\n4. INFORMATION SOUGHT: {information}\n5. LEGAL BASIS: Section 3 of RTI Act 2005 read with {article}\n6. PUBLIC INTEREST: {public_interest}",
    },
]

FEDERAL_CONFLICT_TEMPLATES = [
    {
        "instruction": "Analyse the federal dimension of this policy implementation conflict.",
        "input": "{conflict_scenario}",
        "output": "Federal Conflict Analysis:\n\nConflict Type: {conflict_type}\nUnion Position: {union_position}\nState Position: {state_position}\n\nConstitutional Framework:\n• Seventh Schedule — {list_classification}\n• Article 246 — {legislative_power}\n• Article 254 — {repugnancy_analysis}\n\nConclusion: {conclusion}",
    },
    {
        "instruction": "Does the Central Government have the constitutional authority to mandate {policy} across all States?",
        "input": "{context}",
        "output": "Federal Authority Analysis:\n\nThe Central Government's authority to mandate {policy} depends on the Seventh Schedule classification:\n\n• If {subject} falls under the Union List (List I): Centre has exclusive legislative competence\n• If {subject} falls under State List (List II): Centralized mandate requires constitutional amendment\n• If {subject} falls under Concurrent List (List III): Centre's law prevails under Article 254\n\nCurrent Assessment: {assessment}\n\nFederal Risk Level: {risk_level}",
    },
]

IMPLEMENTATION_GAP_TEMPLATES = [
    {
        "instruction": "Identify implementation gaps in the following scheme that may create constitutional risk.",
        "input": "{scheme_text}",
        "output": "Implementation Gap Analysis:\n\nIdentified Gaps:\n1. {gap_1} — Constitutional implication: {implication_1}\n2. {gap_2} — Constitutional implication: {implication_2}\n3. {gap_3} — Constitutional implication: {implication_3}\n\nSystemic Risk: {systemic_risk}\n\nRecommended Remediation:\n• Short-term: {short_term}\n• Long-term: {long_term}",
    },
]

# Substitution banks for template filling
ARTICLES_BANK = [
    ("Article 14", "equality before law", "Fundamental"),
    ("Article 15", "non-discrimination on grounds of religion, race, caste, sex or place of birth", "Fundamental"),
    ("Article 16", "equality of opportunity in public employment", "Fundamental"),
    ("Article 19", "freedom of speech, expression, assembly and movement", "Fundamental"),
    ("Article 21", "life, personal liberty and dignity", "Fundamental"),
    ("Article 21A", "free and compulsory education for children aged 6-14", "Fundamental"),
    ("Article 32", "constitutional remedies and access to justice", "Fundamental"),
    ("Article 39", "adequate livelihood, equal pay and protection of health", "Directive"),
    ("Article 41", "work, education and public assistance in cases of need", "Directive"),
    ("Article 300A", "protection against arbitrary deprivation of property", "Legal"),
]

GROUPS_BANK = [
    "marginalised farmers", "Aadhaar-excluded citizens", "biometric-failure beneficiaries",
    "transgender persons", "persons with disabilities", "scheduled tribe communities",
    "migrant labourers", "single women heads of household", "elderly citizens",
    "undocumented homeless persons", "informal sector workers",
]

POLICY_TEXTS_BANK = [
    "All beneficiaries of the Public Distribution System must authenticate using biometric verification within 30 days or their ration cards shall be automatically cancelled.",
    "The new healthcare scheme shall be available only to citizens who possess an Aadhaar-linked bank account and a valid PAN card.",
    "State governments may impose any additional eligibility criteria for Central scheme beneficiaries at their discretion without prior approval from the Ministry.",
    "Beneficiaries who fail digital literacy assessment shall be categorised as 'low-priority' and served last in queue allocation.",
    "The scheme excludes persons who have migrated across state boundaries in the preceding two years.",
    "Pension disbursement shall be conditional on annual physical verification at district offices — exemptions are not available for persons with disabilities.",
    "The welfare fund shall only be accessible to citizens residing in urban areas with a population above 50,000.",
    "Grievances submitted in languages other than Hindi and English will not be entertained at the national portal level.",
]

GRIEVANCE_TEXTS_BANK = [
    "I am a tribal woman from a remote village. My MNREGA wages have not been paid for 6 months. The Block Development Officer refuses to meet me and the online portal does not work in our area.",
    "My father is 80 years old and cannot walk. The pension office requires him to physically appear for identity verification every year. He cannot travel and his pension has been stopped.",
    "Our village was displaced for a dam project 10 years ago. We were promised rehabilitation but have received nothing. Our children cannot attend school as we have no fixed address.",
    "I applied for disability certificate 2 years ago. The civil surgeon office keeps asking for bribes. Without the certificate I cannot access any welfare scheme.",
    "The school in our village has been non-functional for 8 months. The teacher draws a salary but never comes. Our children have no access to midday meals or education.",
]

SCHEMES_BANK = [
    "PM Kisan Samman Nidhi",
    "Pradhan Mantri Jan Dhan Yojana",
    "Ayushman Bharat PM-JAY",
    "National Food Security Act PDS",
    "MNREGA",
    "PM Awas Yojana",
    "National Pension Scheme",
    "Beti Bachao Beti Padhao",
]

REASONS_BANK = [
    "lack of Aadhaar-linked bank account",
    "failure in biometric authentication",
    "absence of permanent residential address",
    "inability to complete digital literacy assessment",
    "migration across state boundaries",
    "lack of valid caste or income certificate",
    "non-possession of a ration card",
    "exclusion from Below Poverty Line (BPL) list",
    "physical inability to attend mandatory verification",
    "language barriers in accessing online portals",
    "administrative delays in document processing",
    "arbitrary reclassification of eligibility criteria",
]

CONFLICT_SCENARIOS_BANK = [
    "The Central Government has issued a directive mandating all States to implement a uniform land acquisition policy. Several States with State List powers over land are contesting this directive.",
    "The Centre has enacted legislation on agricultural markets (APMC). States that have their own APMC Acts are claiming federal encroachment.",
    "A Central scheme for police modernisation requires States to adopt a uniform FIR format. States contend that police is a State subject under List II.",
    "The Union Government has issued guidelines on groundwater regulation but water is listed in both Union and State lists.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Generator Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GeneratorConfig:
    """Configuration for SyntheticDataGenerator."""
    templates_per_category: int = 50
    c3_scenarios_count: int = 100
    seed: int = 42
    output_file: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticDataGenerator:
    """
    Synthetic instruction dataset generator for Lokneeti-3B.

    Generates thousands of governance reasoning examples using template
    substitution + the Constitutional Chain Compression engine.

    Usage::

        gen = SyntheticDataGenerator()
        examples = gen.generate_all(n_per_category=50)
        gen.save(examples, "data/synthetic/lokneeti_sft_train.jsonl")
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        chain_engine: Optional[ConstitutionalChainEngine] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.chain_engine = chain_engine or ConstitutionalChainEngine()
        random.seed(self.config.seed)
        log.info("SyntheticDataGenerator ready")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #
    def generate_all(
        self,
        n_per_category: Optional[int] = None,
    ) -> List[SyntheticInstruction]:
        """
        Generate the complete synthetic dataset.

        Args:
            n_per_category: Override templates_per_category from config.

        Returns:
            List of SyntheticInstruction objects ready for training.
        """
        n = n_per_category or self.config.templates_per_category
        all_examples: List[SyntheticInstruction] = []

        categories = [
            ("Constitutional QA",    self._generate_constitutional_qa, n),
            ("Policy Contradiction", self._generate_policy_contradiction, n),
            ("Welfare Risk",         self._generate_welfare_risk, n),
            ("Grievance Abstraction",self._generate_grievance_abstraction, n),
            ("Federal Conflict",     self._generate_federal_conflict, n),
            ("Implementation Gap",   self._generate_implementation_gap, n),
            ("C³ Chain Compression", self._generate_c3_chains, self.config.c3_scenarios_count),
        ]

        for name, generator_fn, count in tqdm(categories, desc="Generating categories"):
            examples = list(generator_fn(count))
            all_examples.extend(examples)
            log.info(f"✅ {name}: {len(examples)} examples generated")

        # Shuffle
        random.shuffle(all_examples)
        log.info(f"🎯 Total synthetic examples: {len(all_examples)}")
        return all_examples

    def save(
        self,
        examples: List[SyntheticInstruction],
        output_path: str | Path,
        system_prompt: str = "",
    ) -> None:
        """
        Save synthetic examples to a JSONL file.

        Saves both the raw schema format AND the formatted 'text' field
        for direct use in SFTTrainer.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(str(output_path), mode="w") as writer:
            for ex in tqdm(examples, desc=f"Saving → {output_path.name}"):
                writer.write({
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "text": ex.to_text(system_prompt),
                    "task_type": ex.task_type.value,
                    "domain": ex.domain.value,
                    "language": ex.language.value,
                    "article_refs": ex.article_refs,
                    "is_synthetic": ex.is_synthetic,
                })

        log.info(f"✅ Saved {len(examples)} examples to {output_path}")

    # ------------------------------------------------------------------ #
    # Category Generators                                                   #
    # ------------------------------------------------------------------ #
    def _generate_constitutional_qa(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            art, right, atype = random.choice(ARTICLES_BANK)
            group = random.choice(GROUPS_BANK)
            scheme = random.choice(SCHEMES_BANK)
            reason = random.choice(REASONS_BANK)
            template = random.choice(CONSTITUTIONAL_QA_TEMPLATES)

            fmt = dict(
                article=art, right=right, group=group,
                scheme=scheme, type=atype, reason=reason,
            )
            instruction = template["instruction"].format(**fmt)
            inp = template["input"].format(**fmt) if template["input"] else ""
            output = template["output"].format(**fmt)

            yield SyntheticInstruction(
                instruction=instruction,
                input=inp,
                output=output,
                task_type=TaskType.CONSTITUTIONAL_QA,
                domain=GovernanceDomain.CONSTITUTIONAL,
                article_refs=[art],
            )

    def _generate_policy_contradiction(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            policy_text = random.choice(POLICY_TEXTS_BANK)
            art, right, _ = random.choice(ARTICLES_BANK)
            group = random.choice(GROUPS_BANK)
            template = random.choice(POLICY_CONTRADICTION_TEMPLATES)

            output = template["output"].format(
                contradiction=f"Mandatory exclusion clause conflicts with {art}",
                article=art, right=right, group=group,
                provision="eligibility restriction",
                reasoning=f"The policy creates arbitrary classification without rational basis, violating {art}.",
                recommendation="Introduce exemption mechanisms for marginalized groups.",
                risk_level="HIGH",
                element="exclusion", assessment="conflicts", compliance="violates",
                scope="mandatory requirements", verdict="has constitutional contradictions",
            )

            yield SyntheticInstruction(
                instruction=template["instruction"],
                input=template["input"].format(policy_text=policy_text),
                output=output,
                task_type=TaskType.POLICY_CONTRADICTION,
                domain=GovernanceDomain.CONSTITUTIONAL,
                article_refs=[art, "Article 14"],
            )

    def _generate_welfare_risk(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            art, right, _ = random.choice(ARTICLES_BANK)
            group1 = random.choice(GROUPS_BANK)
            group2 = random.choice([g for g in GROUPS_BANK if g != group1])
            scheme = random.choice(SCHEMES_BANK)
            template = random.choice(WELFARE_RISK_TEMPLATES)
            scenario = f"{scheme} excludes {group1} due to lack of biometric authentication."

            output = template["output"].format(
                group=group1, group_1=group1, group_2=group2,
                mechanism="biometric exclusion",
                reason_1="no smartphone access", reason_2="lack of Aadhaar linkage",
                article=art, right=right,
                risk_assessment=f"Systematic exclusion creates constitutional violation",
                equality_risk="Differential treatment without reasonable classification",
                risk_level="HIGH",
                scheme_description=f"{scheme} design",
                recommendation=f"Introduce alternative authentication for {group1}",
                safeguards="offline authentication, ombudsman, grievance portal",
            )

            yield SyntheticInstruction(
                instruction=template["instruction"],
                input=template.get("input", scenario).format(
                    scenario=scenario,
                    scheme_description=f"{scheme} requires biometric for all beneficiaries",
                ),
                output=output,
                task_type=TaskType.WELFARE_RISK_ANALYSIS,
                domain=GovernanceDomain.WELFARE,
                article_refs=[art, "Article 21"],
            )

    def _generate_grievance_abstraction(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            grievance = random.choice(GRIEVANCE_TEXTS_BANK)
            art, right, _ = random.choice(ARTICLES_BANK)
            template = random.choice(GRIEVANCE_ABSTRACTION_TEMPLATES)

            output = template["output"].format(
                summary=grievance[:100] + "...",
                article=art, right=right,
                entity="District/Block Level Authority",
                violation_type="Administrative denial of constitutional entitlement",
                remedy=f"Writ petition under Article 32/226 or RTI application",
                priority="HIGH",
                matter="Denial of constitutional entitlement",
                department="Ministry of Rural Development / State Welfare Department",
                information="Status of application, reason for denial, grievance redressal timeline",
                public_interest="Affects fundamental rights of marginalized citizens",
                complaint=grievance,
            )

            yield SyntheticInstruction(
                instruction=template["instruction"],
                input=grievance,
                output=output,
                task_type=TaskType.GRIEVANCE_ABSTRACTION,
                domain=GovernanceDomain.GRIEVANCE,
                article_refs=[art, "Article 21"],
            )

    def _generate_federal_conflict(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            scenario = random.choice(CONFLICT_SCENARIOS_BANK)
            art, right, _ = random.choice(ARTICLES_BANK)
            template = random.choice(FEDERAL_CONFLICT_TEMPLATES)

            output = template["output"].format(
                conflict_type="Centre-State legislative conflict",
                union_position="Centralised regulation for uniformity",
                state_position="State autonomy under List II",
                list_classification="Subject appears to fall under State/Concurrent List",
                legislative_power="Parliament may legislate if it satisfies List I or List III",
                repugnancy_analysis="In case of conflict, Central law prevails under Article 254",
                conclusion="The Central directive requires constitutional backing. States may challenge in Supreme Court under Article 131.",
                policy="uniform land acquisition", context=scenario,
                subject="land acquisition", assessment="Requires careful Seventh Schedule analysis",
                risk_level="HIGH", conflict_scenario=scenario,
            )

            yield SyntheticInstruction(
                instruction=template["instruction"],
                input=scenario,
                output=output,
                task_type=TaskType.FEDERAL_CONFLICT,
                domain=GovernanceDomain.FEDERAL,
                article_refs=["Article 246", "Article 254", "Seventh Schedule"],
            )

    def _generate_implementation_gap(self, n: int) -> Iterator[SyntheticInstruction]:
        for _ in range(n):
            scheme = random.choice(SCHEMES_BANK)
            policy_text = random.choice(POLICY_TEXTS_BANK)
            template = IMPLEMENTATION_GAP_TEMPLATES[0]

            output = template["output"].format(
                gap_1="Mandatory biometric authentication without offline fallback",
                implication_1="Violates Article 21 — effective exclusion from right to food/health",
                gap_2="No grievance redressal mechanism mentioned",
                implication_2="Violates Article 32 — effective denial of constitutional remedies",
                gap_3="No special provisions for persons with disabilities",
                implication_3="Violates Article 15(3) — absence of protective discrimination",
                systemic_risk="The scheme may systematically exclude the most vulnerable while appearing inclusive on paper.",
                short_term="Deploy offline authentication pilots; establish block-level grievance cells",
                long_term="Amend scheme guidelines with mandatory inclusion audits and constitutional impact assessments",
            )

            yield SyntheticInstruction(
                instruction=template["instruction"],
                input=policy_text,
                output=output,
                task_type=TaskType.IMPLEMENTATION_GAP,
                domain=GovernanceDomain.POLICY_ANALYSIS,
                article_refs=["Article 21", "Article 14", "Article 32"],
            )

    def _generate_c3_chains(self, n: int) -> Iterator[SyntheticInstruction]:
        """Generate examples using the full Constitutional Chain Compression engine."""
        scenarios = [
            "A welfare scheme excludes biometric-failure citizens from receiving food rations.",
            "The government requires all pension beneficiaries to appear in-person annually despite physical disabilities.",
            "A new policy makes digital literacy mandatory to access government services.",
            "Migrant workers crossing state boundaries lose all welfare entitlements.",
            "RTI applications in regional languages are automatically rejected by the national portal.",
            "A housing scheme denies allocation to single women heads of household.",
            "The education scheme excludes out-of-school children above age 12.",
            "Farmers without Aadhaar-linked bank accounts are excluded from crop insurance.",
            "Healthcare scheme limits coverage to citizens in urban areas only.",
            "A dam project displaces tribal communities without rehabilitation plan.",
            "Caste-based discrimination occurs in MNREGA job allocation.",
            "Police refuse to register FIRs for domestic violence without male witness.",
            "School mid-day meals are withheld as punishment for students.",
            "Disability certificates are denied without constitutional due process.",
            "Land acquisition compensation is calculated at below-market rates arbitrarily.",
            "A scheme requires citizens to waive privacy rights to access public housing.",
            "Police surveillance systems are deployed without legislative authority.",
            "State withholds Centrally-sponsored funds without explanation or appeal process.",
            "Uniform Civil Code is mandated for tribal communities without consultation.",
            "Environmental clearances are granted without public hearing as required by law.",
        ] * (n // 20 + 1)

        for scenario in scenarios[:n]:
            try:
                result = self.chain_engine.analyze(scenario)
                yield result.instruction
            except Exception as e:
                log.warning(f"C³ generation failed for scenario: {e}")
                continue
