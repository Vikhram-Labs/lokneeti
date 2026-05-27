"""
scripts/data/scraper.py
========================
Async governance corpus scraper for Lokneeti-3B.

Scrapes:
  - India Code acts and regulations
  - NITI Aayog policy reports
  - Public welfare scheme descriptions
  - Parliamentary question summaries

Features:
  - Async HTTP with aiohttp
  - Exponential backoff retry logic
  - Rate limiting per domain
  - Incremental / resumable execution (checkpoint file)
  - Metadata extraction and JSONL storage
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.data.cleaner import TextCleaner
from lokneeti.schemas.datasets import GovernanceDomain, GovernanceExample, Language
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScrapeTarget:
    name: str
    urls: List[str]
    domain: GovernanceDomain
    rate_limit: float = 2.0   # seconds between requests
    max_pages: int = 50


TARGETS: List[ScrapeTarget] = [
    ScrapeTarget(
        name="niti_aayog",
        urls=["https://niti.gov.in/publications"],
        domain=GovernanceDomain.POLICY_ANALYSIS,
        rate_limit=2.0,
        max_pages=30,
    ),
    ScrapeTarget(
        name="india_code",
        urls=["https://www.indiacode.nic.in/acts-in-force"],
        domain=GovernanceDomain.CONSTITUTIONAL,
        rate_limit=1.5,
        max_pages=50,
    ),
    ScrapeTarget(
        name="welfare_pmjay",
        urls=["https://pmjay.gov.in/about/pmjay"],
        domain=GovernanceDomain.WELFARE,
        rate_limit=1.0,
        max_pages=10,
    ),
    ScrapeTarget(
        name="welfare_nrega",
        urls=["https://nrega.nic.in/netnrega/home.aspx"],
        domain=GovernanceDomain.WELFARE,
        rate_limit=1.0,
        max_pages=10,
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Lokneeti-Research-Bot/1.0; "
        "+https://github.com/vikhram-labs/lokneeti)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-IN,en;q=0.9",
}

OUTPUT_DIR = Path("./data/raw")
CHECKPOINT_FILE = OUTPUT_DIR / ".scraper_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint() -> Dict[str, List[str]]:
    """Load previously scraped URLs to enable resumable execution."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: Dict[str, List[str]]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Async Fetcher
# ─────────────────────────────────────────────────────────────────────────────
async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Optional[str]:
    """Fetch a URL with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.text()
                elif resp.status == 429:
                    wait = backoff ** (attempt + 1)
                    log.warning(f"Rate limited — waiting {wait}s before retry")
                    await asyncio.sleep(wait)
                else:
                    log.warning(f"HTTP {resp.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            log.warning(f"Timeout on {url} (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff ** attempt)
        except aiohttp.ClientError as e:
            log.error(f"Client error on {url}: {e}")
            return None
    return None


def extract_text_from_html(html: str, url: str) -> Optional[str]:
    """Extract meaningful text content from HTML."""
    try:
        soup = BeautifulSoup(html, "lxml")
        # Remove boilerplate
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer main content areas
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.find("body")
        )
        if main:
            return main.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        log.error(f"HTML parse error for {url}: {e}")
        return None


def make_doc_id(url: str, text: str) -> str:
    """Create deterministic document ID from URL + content hash."""
    content_hash = hashlib.sha256((url + text[:200]).encode()).hexdigest()[:12]
    domain = urlparse(url).netloc.replace(".", "-")
    return f"{domain}-{content_hash}"


# ─────────────────────────────────────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────────────────────────────────────
class GovernanceScraper:
    """Async governance corpus scraper."""

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = TextCleaner()
        self.checkpoint = load_checkpoint()

    async def scrape_target(self, target: ScrapeTarget) -> List[GovernanceExample]:
        """Scrape all URLs for a single ScrapeTarget."""
        scraped_urls = set(self.checkpoint.get(target.name, []))
        examples: List[GovernanceExample] = []

        connector = aiohttp.TCPConnector(limit=3)
        async with aiohttp.ClientSession(connector=connector) as session:
            urls_to_scrape = [u for u in target.urls if u not in scraped_urls]

            for url in tqdm(urls_to_scrape, desc=f"Scraping {target.name}", unit="url"):
                html = await fetch_with_retry(session, url)
                if not html:
                    continue

                text = extract_text_from_html(html, url)
                if not text:
                    continue

                cleaned = self.cleaner.clean(text)
                if not cleaned.passed_filter:
                    continue

                doc_id = make_doc_id(url, cleaned.text)
                example = GovernanceExample(
                    id=doc_id,
                    source=target.name,
                    url=url,
                    title=urlparse(url).path.split("/")[-1],
                    text=cleaned.text,
                    language=Language(cleaned.language or "en"),
                    domain=target.domain,
                    metadata={"scrape_timestamp": time.time()},
                )
                examples.append(example)
                scraped_urls.add(url)

                # Respect rate limit
                await asyncio.sleep(target.rate_limit)

        # Update checkpoint
        self.checkpoint[target.name] = list(scraped_urls)
        save_checkpoint(self.checkpoint)
        return examples

    async def run_all(self) -> None:
        """Run the scraper for all configured targets."""
        all_examples: List[GovernanceExample] = []

        for target in TARGETS:
            examples = await self.scrape_target(target)
            all_examples.extend(examples)
            log.info(f"✅ {target.name}: {len(examples)} documents scraped")

            # Save per-source JSONL
            out_file = self.output_dir / f"{target.name}.jsonl"
            with open(out_file, "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.model_dump(), default=str) + "\n")

        log.info(f"🎯 Total documents scraped: {len(all_examples)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🌐 Starting Lokneeti governance corpus scraper...")
    scraper = GovernanceScraper()
    asyncio.run(scraper.run_all())
    log.info("✅ Scraping complete. Data saved to ./data/raw/")
