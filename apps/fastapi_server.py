"""
apps/fastapi_server.py
=======================
Lokneeti-3B REST API — Constitutional Governance Intelligence Service

Endpoints:
  POST /analyze          — Full governance analysis
  POST /detect-risk      — Constitutional risk detection
  POST /abstract-grievance — Grievance abstraction
  GET  /health           — Health check
  GET  /examples         — Example scenarios

Usage:
  uvicorn apps.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lokneeti.reasoning.constitutional_chain import ConstitutionalChainEngine
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App Initialisation
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lokneeti-3B API",
    description=(
        "Constitutional Governance Reasoning API — "
        "Reasoning for Democratic Governance. "
        "Developed by Vikhram Labs."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_chain_engine: Optional[ConstitutionalChainEngine] = None
_pipeline = None
_startup_time = time.time()


@app.on_event("startup")
async def startup_event() -> None:
    global _chain_engine, _pipeline
    _chain_engine = ConstitutionalChainEngine()
    log.info("✅ Constitutional Chain Engine loaded")

    model_id = os.environ.get("LOKNEETI_MODEL")
    if model_id:
        try:
            from lokneeti.inference.pipeline import LoknetiPipeline
            adapter = os.environ.get("LOKNEETI_ADAPTER")
            _pipeline = LoknetiPipeline.from_pretrained(model_id, adapter_path=adapter)
            log.info(f"✅ Model pipeline loaded: {model_id}")
        except Exception as e:
            log.warning(f"Model not loaded: {e} — using rule-based C³")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Policy text or governance scenario")
    instruction: Optional[str] = Field(None, description="Optional task instruction")
    task: str = Field(
        "constitutional_chain",
        description="Task type: constitutional_chain | risk_detection | grievance | policy_contradiction",
    )


class ChainNodeResponse(BaseModel):
    concept: str
    relation: str
    target: str
    weight: float


class AnalyzeResponse(BaseModel):
    input_text: str
    chain_nodes: List[ChainNodeResponse]
    articles_implicated: List[str]
    risk_level: str
    conclusion: str
    model_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    engine_loaded: bool
    uptime_seconds: float
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        engine_loaded=_chain_engine is not None,
        uptime_seconds=round(time.time() - _startup_time, 1),
        version="0.1.0",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Governance Analysis"])
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Perform full Constitutional Chain Compression analysis on a policy text.

    Returns a structured constitutional reasoning chain, risk assessment,
    and governance conclusion.
    """
    if _chain_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    t0 = time.time()

    if _pipeline is not None:
        # Use full model
        task_instructions = {
            "constitutional_chain": (
                "Analyse using Constitutional Chain Compression. "
                "Enumerate the reasoning chain and provide a structured conclusion."
            ),
            "risk_detection":    "Detect constitutional risks in this policy text.",
            "grievance":         "Abstract this citizen grievance into a constitutional rights report.",
            "policy_contradiction": "Detect constitutional contradictions in this policy.",
        }
        instr = request.instruction or task_instructions.get(request.task, task_instructions["constitutional_chain"])
        response = _pipeline.analyze(policy_text=request.text, instruction=instr)
        latency = (time.time() - t0) * 1000

        return AnalyzeResponse(
            input_text=request.text,
            chain_nodes=[],
            articles_implicated=[],
            risk_level="see_conclusion",
            conclusion=response.output,
            model_used=response.model_id,
            latency_ms=round(latency, 1),
        )

    # Rule-based C³ fallback
    result = _chain_engine.analyze(request.text)
    latency = (time.time() - t0) * 1000

    return AnalyzeResponse(
        input_text=request.text,
        chain_nodes=[
            ChainNodeResponse(
                concept=n.concept,
                relation=n.relation,
                target=n.target,
                weight=n.weight,
            )
            for n in result.chain.chain_nodes
        ],
        articles_implicated=result.chain.articles_implicated,
        risk_level=result.chain.risk_level,
        conclusion=result.chain.conclusion,
        model_used="constitutional-chain-engine-v1 (rule-based)",
        latency_ms=round(latency, 1),
    )


@app.post("/detect-risk", tags=["Governance Analysis"])
async def detect_risk(request: AnalyzeRequest) -> dict:
    """Shortcut for constitutional risk detection."""
    request.task = "risk_detection"
    result = await analyze(request)
    return {
        "risk_level": result.risk_level,
        "articles":   result.articles_implicated,
        "conclusion": result.conclusion,
    }


@app.post("/abstract-grievance", tags=["Governance Analysis"])
async def abstract_grievance(request: AnalyzeRequest) -> dict:
    """Compress a citizen grievance into a constitutional rights report."""
    request.task = "grievance"
    result = await analyze(request)
    return {"report": result.conclusion, "articles": result.articles_implicated}


@app.get("/examples", tags=["System"])
async def get_examples() -> dict:
    """Return example governance scenarios for API exploration."""
    return {
        "examples": [
            {
                "task": "constitutional_chain",
                "text": "A welfare scheme excludes biometric-failure citizens from food rations.",
            },
            {
                "task": "grievance",
                "text": "My MNREGA wages have not been paid for 6 months. The BDO refuses to meet.",
            },
            {
                "task": "risk_detection",
                "text": "The policy mandates Aadhaar for all government services without exemptions.",
            },
        ]
    }
