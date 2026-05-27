"""
scripts/data/constitution_parser.py
=====================================
Indian Constitution parser — article-level chunking with metadata.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lokneeti.data.cleaner import TextCleaner
from lokneeti.schemas.datasets import GovernanceDomain, GovernanceExample, Language
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

OUTPUT_DIR = Path("./data/raw")
OUTPUT_FILE = OUTPUT_DIR / "constitution_articles.jsonl"

PARTS = {
    "III": "Fundamental Rights",
    "IV": "Directive Principles of State Policy",
    "IVA": "Fundamental Duties",
    "V": "The Union",
    "VI": "The States",
    "XI": "Relations between Union and States",
    "XII": "Finance, Property, Contracts",
}

ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+[A-Z]?)\.\s+([A-Z][^\n]+)\n",
    re.MULTILINE,
)

PART_PATTERN = re.compile(
    r"PART\s+([IVXABC]+)\s*\n([A-Z][^\n]+)",
    re.IGNORECASE,
)


class ConstitutionParser:
    """Parses the Indian Constitution into article-level GovernanceExample objects."""

    def __init__(
        self,
        pdf_path: Optional[str | Path] = None,
        output_file: Path = OUTPUT_FILE,
    ) -> None:
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.output_file = output_file
        self.cleaner = TextCleaner()

    def parse(self) -> List[GovernanceExample]:
        text = self._load_text()
        if not text:
            log.error("Failed to load Constitution text from any source.")
            return []
        articles = list(self._extract_articles(text))
        log.info(f"✅ Parsed {len(articles)} constitutional articles")
        return articles

    def save(self, articles: List[GovernanceExample]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            for article in tqdm(articles, desc="Saving articles"):
                f.write(json.dumps(article.model_dump(), default=str) + "\n")
        log.info(f"✅ Saved {len(articles)} articles to {self.output_file}")

    def _load_text(self) -> Optional[str]:
        if self.pdf_path and self.pdf_path.exists():
            text = self._load_from_pdf(self.pdf_path)
            if text:
                return text
        text = self._load_from_package()
        if text:
            return text
        log.warning("Using seed constitutional text (limited articles for demo)")
        return self._seed_text()

    def _load_from_pdf(self, pdf_path: Path) -> Optional[str]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in tqdm(reader.pages, desc="PDF pages")]
            return "\n".join(pages)
        except Exception as e:
            log.error(f"PDF extraction failed: {e}")
            return None

    @staticmethod
    def _load_from_package() -> Optional[str]:
        try:
            from indianconstitution import get_constitution_text  # type: ignore
            return get_constitution_text()
        except Exception:
            return None

    def _extract_articles(self, text: str) -> Iterator[GovernanceExample]:
        part_boundaries = [(m.start(), m.group(1)) for m in PART_PATTERN.finditer(text)]
        splits = list(ARTICLE_PATTERN.finditer(text))

        for i, match in enumerate(splits):
            article_num = match.group(1)
            article_title = match.group(2).strip()
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            article_body = text[start:end].strip()

            full_text = f"Article {article_num}. {article_title}\n\n{article_body}"
            cleaned = self.cleaner.clean(full_text)
            if not cleaned.passed_filter:
                continue

            part = "III"
            for pos, p in part_boundaries:
                if pos <= match.start():
                    part = p

            doc_hash = hashlib.sha256(f"const-{article_num}".encode()).hexdigest()[:10]
            yield GovernanceExample(
                id=f"const-art-{doc_hash}",
                source="constitution_of_india",
                url="https://www.india.gov.in/sites/upload_files/npi/files/coi-full.pdf",
                title=f"Article {article_num} — {article_title}",
                text=cleaned.text,
                language=Language.ENGLISH,
                domain=GovernanceDomain.CONSTITUTIONAL,
                article_refs=[f"Article {article_num}"],
                metadata={
                    "article_number": article_num,
                    "article_title": article_title,
                    "part": part,
                    "part_name": PARTS.get(part, "General"),
                },
            )

    @staticmethod
    def _seed_text() -> str:
        return """
PART III FUNDAMENTAL RIGHTS

14. Equality before law
The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

21. Protection of life and personal liberty
No person shall be deprived of his life or personal liberty except according to procedure established by law.

21A. Right to education
The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.

32. Remedies for enforcement of rights conferred by this Part
The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.

PART IV DIRECTIVE PRINCIPLES OF STATE POLICY

41. Right to work, to education and to public assistance in certain cases
The State shall, within the limits of its economic capacity and development, make effective provision for securing the right to work, to education and to public assistance in cases of unemployment, old age, sickness and disablement.
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse Indian Constitution → JSONL")
    ap.add_argument("--pdf", type=str, default=None)
    ap.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = ap.parse_args()

    parser = ConstitutionParser(pdf_path=args.pdf, output_file=Path(args.output))
    articles = parser.parse()
    parser.save(articles)
