"""
Lokneeti-3B: Constitutional Governance Reasoning SLM
=====================================================
"Reasoning for Democratic Governance"

Developed by Vikhram Labs Hugging Face Organization.

Core capabilities:
  - Constitutional conflict detection
  - Policy contradiction reasoning
  - Welfare inclusion analysis
  - Governance risk abstraction
  - Multilingual citizen grievance compression
  - Democratic institutional reasoning
  - Policy implementation ambiguity detection

Reasoning paradigm: Constitutional Chain Compression (C³)
"""

__version__ = "0.1.0"
__author__ = "Vikhram Labs"
__license__ = "Apache-2.0"
__model__ = "Lokneeti-3B"
__base_model__ = "Qwen/Qwen2.5-3B-Instruct"
__hf_repo__ = "vikhram-labs/Lokneeti-3B"

from lokneeti.utils.logging import get_logger  # noqa: F401

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__model__",
    "__base_model__",
    "__hf_repo__",
    "get_logger",
]
