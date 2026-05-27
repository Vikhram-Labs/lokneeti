"""
scripts/data/parliament_scraper.py
====================================
Parliamentary debates and Q&A scraper for Lok Sabha / Rajya Sabha.

Collects starred/unstarred questions and answers from Sansad.in,
which are a rich source of policy reasoning and constitutional dialogue.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.data.cleaner import TextCleaner
from lokneeti.schemas.datasets import GovernanceDomain, GovernanceExample, Language
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

OUTPUT_DIR  = Path("./data/raw")
OUTPUT_FILE = OUTPUT_DIR / "parliament_debates.jsonl"

BASE_URLS = [
    "https://sansad.in/ls/questions/questions-and-answers",
    "https://rajyasabha.nic.in/rsnew/questions/questionsearch.aspx",
]

HEADERS = {
    "User-Agent": "Lokneeti-Research-Bot/1.0 (academic; governance-ai-research)",
    "Accept-Language": "en-IN,en;q=0.9",
}

GOVERNANCE_KEYWORDS = [
    "constitutional", "welfare", "fundamental rights", "article 21",
    "MNREGA", "ration", "Aadhaar", "pension", "healthcare", "education",
    "tribal", "scheduled caste", "scheduled tribe", "RTI", "grievance",
    "federal", "state government", "central government", "policy",
]


async def fetch_page(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = 3,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            async with session.get(url, headers=HEADERS,
                                   timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status == 200:
                    return await r.text()
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            log.debug(f"Fetch error {url}: {e}")
    return None


def extract_qa_pairs(html: str, source_url: str) -> List[GovernanceExample]:
    """Extract Q&A pairs from a parliamentary questions page."""
    cleaner = TextCleaner()
    examples: List[GovernanceExample] = []

    try:
        soup = BeautifulSoup(html, "lxml")
        # Look for question/answer blocks (structure varies by session)
        qa_blocks = (
            soup.find_all("div", class_="question-answer")
            or soup.find_all("div", class_="qa-block")
            or soup.find_all("table", class_="qtable")
            or [soup.find("body")]
        )

        for i, block in enumerate(qa_blocks[:20]):
            if not block:
                continue
            text = block.get_text(separator="\n", strip=True)

            # Filter by governance relevance
            text_lower = text.lower()
            if not any(kw in text_lower for kw in GOVERNANCE_KEYWORDS):
                continue

            cleaned = cleaner.clean(text)
            if not cleaned.passed_filter:
                continue

            doc_hash = hashlib.sha256(
                f"parliament-{source_url}-{i}".encode()
            ).hexdigest()[:10]

            examples.append(GovernanceExample(
                id=f"parl-{doc_hash}",
                source="parliament_debates",
                url=source_url,
                title=f"Parliamentary Q&A — Session Block {i}",
                text=cleaned.text,
                language=Language.ENGLISH,
                domain=GovernanceDomain.PARLIAMENTARY,
                metadata={"block_index": i, "scrape_time": time.time()},
            ))

    except Exception as e:
        log.error(f"HTML parse error: {e}")

    return examples


async def scrape_parliament(max_pages: int = 30) -> List[GovernanceExample]:
    """Scrape parliamentary Q&A data."""
    all_examples: List[GovernanceExample] = []

    connector = aiohttp.TCPConnector(limit=2)
    async with aiohttp.ClientSession(connector=connector) as session:
        for base_url in BASE_URLS:
            for page in tqdm(range(1, min(max_pages, 11)), desc=f"Parliament pages"):
                url = f"{base_url}?page={page}"
                html = await fetch_page(session, url)
                if not html:
                    continue
                examples = extract_qa_pairs(html, url)
                all_examples.extend(examples)
                await asyncio.sleep(2.5)

    log.info(f"Parliament scrape complete: {len(all_examples)} examples")
    return all_examples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scrape parliamentary debates → JSONL")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = asyncio.run(scrape_parliament(max_pages=args.max_pages))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), default=str) + "\n")
    log.info(f"✅ Saved {len(examples)} parliamentary examples → {out}")
