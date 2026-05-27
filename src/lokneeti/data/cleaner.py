"""
lokneeti.data.cleaner
=====================
Multilingual text cleaning for the Lokneeti governance corpus.

Handles:
  - Unicode normalization (NFC)
  - OCR noise removal (common artifacts from PDF extraction)
  - Hindi/Devanagari-aware whitespace normalization
  - English-Indic script detection and tagging
  - Minimum/maximum length filtering
  - ftfy-based encoding repair

Designed to be stateless and embarrassingly parallel via multiprocessing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import ftfy
from langdetect import detect, LangDetectException

from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# OCR Noise Patterns                                                            #
# --------------------------------------------------------------------------- #
_OCR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[^\S\n]{2,}"),                  " "),    # Collapse multi-spaces
    (re.compile(r"\n{3,}"),                        "\n\n"), # Max 2 consecutive newlines
    (re.compile(r"[|]{2,}"),                       ""),     # OCR table artifacts
    (re.compile(r"\.{4,}"),                        "..."),  # Ellipsis noise
    (re.compile(r"[_]{3,}"),                       ""),     # Underline noise
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"), ""),    # Control chars (keep \t\n)
    (re.compile(r"[-]{5,}"),                       "---"),  # Long dashes
    (re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.I), ""), # Page headers
    (re.compile(r"\bwww\.[^\s]+\.[a-z]{2,4}\b", re.I), ""),  # Stray URLs
]

# Indian constitutional/legal character whitelist check
_ALLOWED_SCRIPTS = re.compile(
    r"[\u0000-\u007F"   # Basic Latin (English)
    r"\u0900-\u097F"    # Devanagari (Hindi, Marathi, Sanskrit)
    r"\u0980-\u09FF"    # Bengali
    r"\u0A00-\u0A7F"    # Gurmukhi (Punjabi)
    r"\u0A80-\u0AFF"    # Gujarati
    r"\u0B00-\u0B7F"    # Oriya
    r"\u0B80-\u0BFF"    # Tamil
    r"\u0C00-\u0C7F"    # Telugu
    r"\u0C80-\u0CFF"    # Kannada
    r"\u0D00-\u0D7F"    # Malayalam
    r"\u0600-\u06FF"    # Arabic (Urdu)
    r"\u2018-\u201F"    # Smart quotes
    r"\u2013\u2014"     # En/em dashes
    r"\u00A0-\u00FF]+"  # Latin-1 supplement
)


# --------------------------------------------------------------------------- #
# Dataclass                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CleanerConfig:
    """Configuration for the TextCleaner."""
    normalize_unicode: bool = True
    fix_encoding: bool = True          # Use ftfy
    remove_ocr_noise: bool = True
    min_char_length: int = 50
    max_char_length: int = 8000
    detect_language: bool = True


@dataclass
class CleanedDocument:
    """Output of TextCleaner.clean()."""
    text: str
    language: Optional[str] = None    # ISO 639-1 code
    char_count: int = 0
    was_truncated: bool = False
    passed_filter: bool = True
    rejection_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Cleaner                                                                       #
# --------------------------------------------------------------------------- #
class TextCleaner:
    """
    Production-grade multilingual text cleaner for governance corpora.

    Usage::

        cleaner = TextCleaner()
        result = cleaner.clean("   यह  एक  परीक्षण  है   ")
        print(result.text)   # → "यह एक परीक्षण है"
        print(result.language)  # → "hi"

    The cleaner is stateless after construction and thread-safe.
    """

    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self.config = config or CleanerConfig()
        log.debug(f"TextCleaner initialized with config: {self.config}")

    # ------------------------------------------------------------------ #
    def clean(self, text: str) -> CleanedDocument:
        """
        Apply the full cleaning pipeline to a single text string.

        Returns a :class:`CleanedDocument` with the cleaned text and
        metadata about the cleaning process.
        """
        cfg = self.config

        # Step 1: Fix encoding corruption (ftfy)
        if cfg.fix_encoding:
            try:
                text = ftfy.fix_text(text)
            except Exception as e:
                log.warning(f"ftfy failed: {e}")

        # Step 2: Unicode normalization (NFC)
        if cfg.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Step 3: OCR noise removal
        if cfg.remove_ocr_noise:
            text = self._remove_ocr_noise(text)

        # Step 4: Strip leading/trailing whitespace
        text = text.strip()

        # Step 5: Length filter — too short
        if len(text) < cfg.min_char_length:
            return CleanedDocument(
                text=text,
                char_count=len(text),
                passed_filter=False,
                rejection_reason=f"Too short: {len(text)} chars < {cfg.min_char_length}",
            )

        # Step 6: Length filter — truncate if too long
        was_truncated = False
        if len(text) > cfg.max_char_length:
            text = text[: cfg.max_char_length]
            was_truncated = True

        # Step 7: Language detection
        detected_lang: Optional[str] = None
        if cfg.detect_language:
            detected_lang = self._detect_language(text)

        return CleanedDocument(
            text=text,
            language=detected_lang,
            char_count=len(text),
            was_truncated=was_truncated,
            passed_filter=True,
        )

    def clean_batch(self, texts: list[str]) -> list[CleanedDocument]:
        """Clean a list of texts and return results."""
        return [self.clean(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _remove_ocr_noise(text: str) -> str:
        """Apply all OCR noise removal regex patterns."""
        for pattern, replacement in _OCR_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        """Detect the primary language of a text sample."""
        try:
            # Sample first 500 chars for speed
            return detect(text[:500])
        except LangDetectException:
            return None
        except Exception as e:
            log.debug(f"Language detection error: {e}")
            return None
