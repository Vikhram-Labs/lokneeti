"""
lokneeti.data.chunker
=====================
Semantic and fixed-size document chunking for governance corpora.

Three strategies:
  1. ``fixed``    — Split at N tokens with K overlap (fast, baseline)
  2. ``semantic`` — Split at sentence/paragraph boundaries respecting semantics
  3. ``article``  — Constitutional article-aware split (preserves article integrity)

Chunks are tagged with source metadata and constitutional article references
detected via lightweight regex matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constitutional Article Reference Regex                                        #
# --------------------------------------------------------------------------- #
_ARTICLE_PATTERN = re.compile(
    r"\b(Article|Art\.?)\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)",
    re.IGNORECASE,
)

_SCHEDULE_PATTERN = re.compile(
    r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
    r"Eleventh|Twelfth)\s+Schedule\b",
    re.IGNORECASE,
)

# Sentence boundary splitter — handles English and Devanagari ।
_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?।])\s+")


# --------------------------------------------------------------------------- #
# Data Structures                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class TextChunk:
    """A single text chunk with governance metadata."""
    text: str
    chunk_index: int
    source_id: str
    source: str
    article_refs: List[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self) -> None:
        # Auto-detect article references if not pre-populated
        if not self.article_refs:
            self.article_refs = extract_article_refs(self.text)


@dataclass
class ChunkerConfig:
    """Configuration for DocumentChunker."""
    strategy: str = "semantic"        # "fixed" | "semantic" | "article"
    chunk_size: int = 512             # In approximate characters (not tokens)
    chunk_overlap: int = 64
    respect_article_boundaries: bool = True
    min_chunk_chars: int = 80


# --------------------------------------------------------------------------- #
# Utility                                                                       #
# --------------------------------------------------------------------------- #
def extract_article_refs(text: str) -> List[str]:
    """
    Extract constitutional article references from text.

    Returns a de-duplicated list of article strings.
    e.g. ["Article 21", "Article 14", "Article 32"]
    """
    refs: list[str] = []
    for match in _ARTICLE_PATTERN.finditer(text):
        refs.append(f"Article {match.group(2)}")
    for match in _SCHEDULE_PATTERN.finditer(text):
        refs.append(match.group(0).title())
    return list(dict.fromkeys(refs))  # Deduplicate preserving order


# --------------------------------------------------------------------------- #
# Chunker                                                                       #
# --------------------------------------------------------------------------- #
class DocumentChunker:
    """
    Governance-aware document chunker.

    Usage::

        chunker = DocumentChunker()
        chunks = chunker.chunk(
            text=long_policy_text,
            source_id="pol-001",
            source="niti_aayog",
        )
        for chunk in chunks:
            print(chunk.text[:100], chunk.article_refs)
    """

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(
        self,
        text: str,
        source_id: str,
        source: str,
    ) -> List[TextChunk]:
        """
        Chunk a document into TextChunk objects.

        Dispatches to the configured strategy.
        """
        strategy = self.config.strategy
        if strategy == "fixed":
            return list(self._fixed_chunk(text, source_id, source))
        elif strategy == "semantic":
            return list(self._semantic_chunk(text, source_id, source))
        elif strategy == "article":
            return list(self._article_chunk(text, source_id, source))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    # ------------------------------------------------------------------ #
    # Strategy: Fixed                                                       #
    # ------------------------------------------------------------------ #
    def _fixed_chunk(
        self, text: str, source_id: str, source: str
    ) -> Iterator[TextChunk]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = size - overlap
        idx = 0
        chunk_i = 0
        while idx < len(text):
            end = min(idx + size, len(text))
            chunk_text = text[idx:end].strip()
            if len(chunk_text) >= self.config.min_chunk_chars:
                yield TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_i,
                    source_id=source_id,
                    source=source,
                    char_start=idx,
                    char_end=end,
                )
                chunk_i += 1
            idx += step

    # ------------------------------------------------------------------ #
    # Strategy: Semantic (paragraph/sentence-aware)                        #
    # ------------------------------------------------------------------ #
    def _semantic_chunk(
        self, text: str, source_id: str, source: str
    ) -> Iterator[TextChunk]:
        # Split on blank lines first (paragraph boundaries)
        paragraphs = re.split(r"\n{2,}", text.strip())
        buffer = ""
        char_cursor = 0
        chunk_i = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph stays within size — append
            if len(buffer) + len(para) <= self.config.chunk_size:
                buffer += ("\n\n" if buffer else "") + para
            else:
                # Flush buffer
                if len(buffer) >= self.config.min_chunk_chars:
                    yield TextChunk(
                        text=buffer.strip(),
                        chunk_index=chunk_i,
                        source_id=source_id,
                        source=source,
                        char_start=char_cursor,
                        char_end=char_cursor + len(buffer),
                    )
                    char_cursor += len(buffer)
                    chunk_i += 1
                # Start new buffer with overlap
                buffer = (buffer[-self.config.chunk_overlap:] + "\n\n" + para
                          if self.config.chunk_overlap > 0 else para)

        # Flush remaining buffer
        if len(buffer.strip()) >= self.config.min_chunk_chars:
            yield TextChunk(
                text=buffer.strip(),
                chunk_index=chunk_i,
                source_id=source_id,
                source=source,
                char_start=char_cursor,
                char_end=char_cursor + len(buffer),
            )

    # ------------------------------------------------------------------ #
    # Strategy: Article-aware (Constitutional document specific)           #
    # ------------------------------------------------------------------ #
    def _article_chunk(
        self, text: str, source_id: str, source: str
    ) -> Iterator[TextChunk]:
        """
        Split at constitutional article boundaries.
        Each article becomes its own chunk (with context from previous article
        title as prefix).
        """
        # Match Article N or ARTICLE N headings
        article_split = re.compile(
            r"(?=\b(?:ARTICLE|Article)\s+\d+[A-Z]?\.?\s)",
        )
        parts = article_split.split(text)
        chunk_i = 0
        char_cursor = 0

        for part in parts:
            part = part.strip()
            if len(part) < self.config.min_chunk_chars:
                char_cursor += len(part)
                continue

            # If part is large, fall back to semantic sub-chunking
            if len(part) > self.config.chunk_size * 2:
                sub_chunker = DocumentChunker(
                    ChunkerConfig(strategy="semantic", chunk_size=self.config.chunk_size)
                )
                for sub in sub_chunker.chunk(part, source_id, source):
                    sub.chunk_index = chunk_i
                    sub.char_start += char_cursor
                    sub.char_end += char_cursor
                    yield sub
                    chunk_i += 1
            else:
                yield TextChunk(
                    text=part,
                    chunk_index=chunk_i,
                    source_id=source_id,
                    source=source,
                    char_start=char_cursor,
                    char_end=char_cursor + len(part),
                )
                chunk_i += 1

            char_cursor += len(part)
