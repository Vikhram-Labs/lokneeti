"""
apps/gradio_demo.py
====================
Lokneeti-3B Gradio Demo — Constitutional Governance Intelligence Interface

Features:
  - Policy text analysis via Constitutional Chain Compression
  - Constitutional risk detection
  - Citizen grievance abstraction
  - Welfare inclusion analysis
  - Example gallery
  - CPU / CUDA auto-detection

Usage:
  python apps/gradio_demo.py                         # local
  python apps/gradio_demo.py --share                 # Colab public link
  python apps/gradio_demo.py --model local_path/     # local model
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lokneeti.reasoning.constitutional_chain import ConstitutionalChainEngine
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Demo Examples
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "Constitutional Risk Detection",
        "A welfare scheme excludes biometric-failure citizens from receiving food rations under the Public Distribution System.",
    ],
    [
        "Policy Contradiction Analysis",
        "The government requires all pension beneficiaries to appear in-person annually for identity verification, with no exemptions for persons with disabilities.",
    ],
    [
        "Grievance Abstraction",
        "I am a tribal woman from a remote village. My MNREGA wages have not been paid for 6 months. The Block Development Officer refuses to meet me.",
    ],
    [
        "Federal Conflict Analysis",
        "The Central Government has issued a directive mandating all States to implement a uniform land acquisition policy under LARR Act without State consultation.",
    ],
    [
        "Inclusion Analysis",
        "A new digital literacy scheme excludes citizens without smartphones from accessing government services, creating a class of digitally disenfranchised citizens.",
    ],
]

TASK_PROMPTS = {
    "Constitutional Risk Detection": (
        "Analyse the following governance scenario using Constitutional Chain Compression. "
        "Identify constitutional risks, enumerate the reasoning chain, and provide a structured governance conclusion."
    ),
    "Policy Contradiction Analysis": (
        "Detect constitutional contradictions in the following policy text. "
        "Identify implicated constitutional articles and assess constitutional risk level."
    ),
    "Grievance Abstraction": (
        "Compress the following citizen grievance into a structured constitutional rights violation report "
        "with recommended legal remedies."
    ),
    "Federal Conflict Analysis": (
        "Analyse the federal dimension of this policy implementation conflict. "
        "Identify relevant Seventh Schedule provisions and Centre-State constitutional implications."
    ),
    "Inclusion Analysis": (
        "Identify marginalized groups at constitutional risk from this policy design. "
        "Cite relevant constitutional articles on equal protection and welfare rights."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Engine (rule-based C³ for CPU demo; model if loaded)
# ─────────────────────────────────────────────────────────────────────────────
_chain_engine = ConstitutionalChainEngine()
_pipeline = None   # Will hold LoknetiPipeline if model is loaded


def load_model(model_id: str, adapter_path: Optional[str] = None) -> bool:
    """Attempt to load the Lokneeti model. Returns True on success."""
    global _pipeline
    try:
        from lokneeti.inference.pipeline import LoknetiPipeline
        _pipeline = LoknetiPipeline.from_pretrained(
            model_id=model_id,
            adapter_path=adapter_path,
        )
        log.info(f"✅ Model loaded: {model_id}")
        return True
    except Exception as e:
        log.warning(f"Model not loaded ({e}) — using rule-based C³ engine")
        return False


def analyze_governance(task_type: str, policy_text: str) -> tuple[str, str, str]:
    """
    Core analysis function called by Gradio.

    Returns:
        (chain_output, risk_level, conclusion)
    """
    if not policy_text.strip():
        return "⚠️ Please enter a policy text or scenario.", "", ""

    # Use full model if available, else rule-based C³
    if _pipeline is not None:
        instruction = TASK_PROMPTS.get(task_type, TASK_PROMPTS["Constitutional Risk Detection"])
        response = _pipeline.analyze(policy_text=policy_text, instruction=instruction)
        output = response.output

        # Parse structured output fields
        chain_section = ""
        risk = "UNKNOWN"
        conclusion = output

        lines = output.split("\n")
        for i, line in enumerate(lines):
            if "CHAIN:" in line:
                chain_section = "\n".join(lines[i:i+8])
            if "RISK LEVEL:" in line:
                risk = line.replace("RISK LEVEL:", "").strip()
            if "CONCLUSION:" in line:
                conclusion = "\n".join(lines[i+1:])

        return chain_section or output, risk, conclusion

    else:
        # Rule-based C³ analysis
        result = _chain_engine.analyze(policy_text)
        chain = result.chain

        chain_text = "\n".join(
            f"  {node.concept} {node.relation} {node.target}"
            for node in chain.chain_nodes
        )
        chain_display = (
            f"[Constitutional Chain Compression — Rule-Based]\n\n"
            f"CHAIN:\n{chain_text}\n\n"
            f"ARTICLES: {', '.join(chain.articles_implicated) or 'Auto-detected'}"
        )

        risk_display = (
            f"🔴 {chain.risk_level.upper()}"
            if chain.risk_level in ("high", "critical")
            else f"🟡 {chain.risk_level.upper()}"
        )

        return chain_display, risk_display, chain.conclusion


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1100px; margin: 0 auto; }
#header { text-align: center; padding: 24px 0 8px; }
#header h1 { font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460, #533483);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header p { color: #6b7280; font-size: 1rem; margin-top: 4px; }
.risk-box { border-radius: 8px; padding: 12px; font-weight: 700; font-size: 1.1rem; }
.chain-box { font-family: monospace; font-size: 0.9rem; }
.footer { text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 16px 0; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Lokneeti-3B — Constitutional Governance Intelligence",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div id="header">
            <h1>⚖️ Lokneeti-3B</h1>
            <p><em>Reasoning for Democratic Governance</em> &nbsp;|&nbsp;
               Vikhram Labs &nbsp;|&nbsp;
               Constitutional Chain Compression (C³)</p>
            <p style="font-size:0.8rem;color:#9ca3af;">
              Indian Constitutional Reasoning · Policy Analysis · Welfare Inclusion ·
              Governance Risk Intelligence
            </p>
        </div>
        """)

        # ── Main Interface ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=5):
                task_dropdown = gr.Dropdown(
                    choices=list(TASK_PROMPTS.keys()),
                    value="Constitutional Risk Detection",
                    label="Analysis Task",
                    info="Select the type of governance reasoning to perform",
                )
                policy_input = gr.Textbox(
                    label="Policy Text / Governance Scenario",
                    placeholder=(
                        "Enter a policy description, governance scenario, "
                        "welfare scheme detail, or citizen grievance...\n\n"
                        "Example: 'A welfare scheme excludes biometric-failure citizens "
                        "from receiving food rations.'"
                    ),
                    lines=7,
                    max_lines=15,
                )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "⚖️ Analyse",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button("🗑 Clear", size="lg")

            with gr.Column(scale=5):
                chain_output = gr.Textbox(
                    label="Constitutional Chain Compression (C³)",
                    lines=8,
                    interactive=False,
                    elem_classes=["chain-box"],
                )
                risk_output = gr.Textbox(
                    label="Constitutional Risk Level",
                    lines=1,
                    interactive=False,
                    elem_classes=["risk-box"],
                )
                conclusion_output = gr.Textbox(
                    label="Governance Conclusion",
                    lines=6,
                    interactive=False,
                )

        # ── Examples ────────────────────────────────────────────────────
        gr.Markdown("### 📚 Example Scenarios")
        example_gallery = gr.Examples(
            examples=EXAMPLES,
            inputs=[task_dropdown, policy_input],
            label="Click to load an example",
        )

        # ── About ────────────────────────────────────────────────────────
        with gr.Accordion("ℹ️ About Lokneeti-3B", open=False):
            gr.Markdown("""
**Lokneeti-3B** is a Small Language Model fine-tuned for Indian constitutional and
governance reasoning, developed by **Vikhram Labs**.

**Specializations:**
- Constitutional conflict detection
- Policy contradiction reasoning
- Welfare inclusion analysis
- Governance risk abstraction
- Multilingual citizen grievance compression
- Democratic institutional reasoning

**Reasoning Paradigm:** Constitutional Chain Compression (C³) — a novel symbolic
reasoning schema that encodes the pathway from a raw governance scenario to a
structured constitutional conclusion via ordered conceptual hops.

**Base Model:** Qwen2.5-3B-Instruct | **Training:** QLoRA (4-bit, LoRA r=16)
| **License:** Apache 2.0

> ⚠️ This tool is for educational and research purposes only. It does not constitute
> legal advice. Always consult qualified legal professionals for constitutional matters.
            """)

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
            Lokneeti-3B · Vikhram Labs · Apache 2.0 ·
            <a href="https://huggingface.co/vikhram-labs/Lokneeti-3B" target="_blank">
            HuggingFace</a> ·
            <a href="https://github.com/vikhram-labs/lokneeti" target="_blank">GitHub</a>
        </div>
        """)

        # ── Events ────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_governance,
            inputs=[task_dropdown, policy_input],
            outputs=[chain_output, risk_output, conclusion_output],
        )
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[policy_input, chain_output, risk_output, conclusion_output],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lokneeti-3B Gradio Demo")
    ap.add_argument("--model",   type=str, default=None,
                    help="HuggingFace model ID or local path (optional)")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--port",    type=int, default=7860)
    ap.add_argument("--share",   action="store_true",
                    help="Create public Gradio link (required for Colab)")
    args = ap.parse_args()

    if args.model:
        load_model(args.model, args.adapter)
    else:
        log.info("No model specified — using rule-based Constitutional Chain engine")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
