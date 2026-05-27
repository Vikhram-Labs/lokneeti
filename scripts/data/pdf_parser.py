"""
scripts/data/pdf_parser.py
===========================
Generic PDF parser for governance documents.

Handles:
  - NITI Aayog reports
  - Finance Commission reports
  - Supreme Court summary PDFs
  - Ministry policy documents

Features:
  - pypdf text extraction
  - Page-level chunking with overlap
  - Metadata extraction from filename/path
  - Batch directory processing
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.data.cleaner import TextCleaner
from lokneeti.data.chunker import DocumentChunker, ChunkerConfig
from lokneeti.schemas.datasets import GovernanceDomain, GovernanceExample, Language
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

OUTPUT_DIR = Path("./data/raw")

# Filename → domain heuristics
DOMAIN_HINTS = {
    "niti": GovernanceDomain.POLICY_ANALYSIS,
    "finance_commission": GovernanceDomain.BUDGET_FINANCE,
    "budget": GovernanceDomain.BUDGET_FINANCE,
    "supreme_court": GovernanceDomain.JUDICIARY,
    "sc_": GovernanceDomain.JUDICIARY,
    "welfare": GovernanceDomain.WELFARE,
    "constitution": GovernanceDomain.CONSTITUTIONAL,
    "parliament": GovernanceDomain.PARLIAMENTARY,
    "rti": GovernanceDomain.RTI,
    "panchayat": GovernanceDomain.FEDERAL,
}


def guess_domain(filename: str) -> GovernanceDomain:
    """Heuristically determine governance domain from filename."""
    lower = filename.lower()
    for hint, domain in DOMAIN_HINTS.items():
        if hint in lower:
            return domain
    return GovernanceDomain.POLICY_ANALYSIS


def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract full text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        log.error(f"PDF extraction failed for {pdf_path.name}: {e}")
        return None


def parse_pdf(
    pdf_path: Path,
    source_name: Optional[str] = None,
    domain: Optional[GovernanceDomain] = None,
    chunk_size: int = 600,
    chunk_overlap: int = 64,
) -> List[GovernanceExample]:
    """
    Parse a single PDF into a list of GovernanceExample chunks.

    Args:
        pdf_path:    Path to the PDF file.
        source_name: Optional source label (defaults to filename stem).
        domain:      Override domain detection.
        chunk_size:  Approximate characters per chunk.
        chunk_overlap: Character overlap between chunks.
    """
    cleaner = TextCleaner()
    chunker = DocumentChunker(ChunkerConfig(
        strategy="semantic",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ))

    source = source_name or pdf_path.stem.replace("_", " ").replace("-", " ")
    dom = domain or guess_domain(pdf_path.name)

    raw_text = extract_pdf_text(pdf_path)
    if not raw_text:
        return []

    cleaned = cleaner.clean(raw_text)
    if not cleaned.passed_filter:
        log.warning(f"PDF failed cleaning filter: {pdf_path.name}")
        return []

    chunks = chunker.chunk(
        text=cleaned.text,
        source_id=pdf_path.stem,
        source=source,
    )

    examples: List[GovernanceExample] = []
    for chunk in chunks:
        doc_hash = hashlib.sha256(
            f"{pdf_path.stem}-{chunk.chunk_index}".encode()
        ).hexdigest()[:10]

        examples.append(GovernanceExample(
            id=f"pdf-{doc_hash}",
            source=source,
            url=str(pdf_path),
            title=f"{source} — Chunk {chunk.chunk_index}",
            text=chunk.text,
            language=Language.ENGLISH,
            domain=dom,
            article_refs=chunk.article_refs,
            metadata={
                "filename": pdf_path.name,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        ))

    log.info(f"✅ {pdf_path.name}: {len(examples)} chunks extracted")
    return examples


def parse_directory(
    pdf_dir: Path,
    output_file: Optional[Path] = None,
) -> List[GovernanceExample]:
    """
    Parse all PDFs in a directory.

    Args:
        pdf_dir:     Directory containing PDF files.
        output_file: Optional path to save JSONL output.
    """
    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        log.warning(f"No PDFs found in {pdf_dir}")
        return []

    log.info(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    all_examples: List[GovernanceExample] = []

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs", unit="file"):
        examples = parse_pdf(pdf_path)
        all_examples.extend(examples)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(asdict(ex), default=str) + "\n")
        log.info(f"✅ Saved {len(all_examples)} chunks to {output_file}")

    return all_examples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse governance PDFs → JSONL")
    ap.add_argument("--dir", type=str, required=True, help="Directory with PDFs")
    ap.add_argument("--output", type=str, default="./data/raw/pdf_corpus.jsonl")
    args = ap.parse_args()

    examples = parse_directory(Path(args.dir), Path(args.output))
    log.info(f"✅ Total chunks: {len(examples)}")
