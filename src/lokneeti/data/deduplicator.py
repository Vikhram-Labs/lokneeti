"""
lokneeti.data.deduplicator
==========================
MinHash-based near-duplicate detection for governance corpora.

Uses a lightweight shingling + Jaccard approximation approach
that operates without external C++ extensions (pure Python fallback).

For production at scale, replace with datasketch MinHashLSH.
This implementation is Colab-compatible with zero additional deps.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from tqdm.auto import tqdm

from lokneeti.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class DeduplicatorConfig:
    """Configuration for MinHashDeduplicator."""
    shingle_size: int = 5           # Character n-gram shingle size
    num_hashes: int = 128           # Number of hash functions for MinHash
    similarity_threshold: float = 0.85   # Jaccard similarity threshold


# --------------------------------------------------------------------------- #
# MinHash Implementation                                                        #
# --------------------------------------------------------------------------- #
def _shingles(text: str, k: int = 5) -> Set[str]:
    """Generate character-level k-shingles from text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip().lower())
    return {text[i: i + k] for i in range(len(text) - k + 1)}


def _minhash_signature(shingles: Set[str], num_hashes: int = 128) -> List[int]:
    """
    Compute a MinHash signature using MD5 with different seeds.

    This is a simplified pure-Python MinHash — deterministic and reproducible.
    """
    signature: List[int] = []
    for i in range(num_hashes):
        min_val = float("inf")
        for shingle in shingles:
            h = hashlib.md5(f"{i}:{shingle}".encode()).digest()
            val = int.from_bytes(h[:4], "big")
            if val < min_val:
                min_val = val
        signature.append(int(min_val))
    return signature


def _jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    assert len(sig1) == len(sig2), "Signatures must have the same length"
    matches = sum(a == b for a, b in zip(sig1, sig2))
    return matches / len(sig1)


# --------------------------------------------------------------------------- #
# Deduplicator                                                                  #
# --------------------------------------------------------------------------- #
class MinHashDeduplicator:
    """
    Near-duplicate removal for governance text corpora.

    Usage::

        dedup = MinHashDeduplicator()
        unique_texts = dedup.deduplicate(list_of_texts)
        print(f"Kept {len(unique_texts)} / {len(list_of_texts)} documents")

    For large corpora (>100K docs), use datasketch LSH instead.
    """

    def __init__(self, config: Optional[DeduplicatorConfig] = None) -> None:
        self.config = config or DeduplicatorConfig()
        self._signatures: List[List[int]] = []
        self._seen_exact: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """
        Check if a text is a near-duplicate of any previously seen document.
        Adds the text to the seen set if it is not a duplicate.
        """
        cfg = self.config

        # Fast path: exact hash check
        exact_hash = hashlib.sha256(text.strip().lower().encode()).hexdigest()
        if exact_hash in self._seen_exact:
            return True

        # Compute MinHash signature
        sh = _shingles(text, k=cfg.shingle_size)
        if not sh:
            return False
        sig = _minhash_signature(sh, num_hashes=cfg.num_hashes)

        # Compare against all stored signatures
        for stored_sig in self._signatures:
            sim = _jaccard_estimate(sig, stored_sig)
            if sim >= cfg.similarity_threshold:
                return True

        # Not a duplicate — register
        self._seen_exact.add(exact_hash)
        self._signatures.append(sig)
        return False

    def deduplicate(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Remove near-duplicates from a list of texts.

        Args:
            texts:         Input list of raw texts.
            show_progress: Show tqdm progress bar.

        Returns:
            Filtered list with near-duplicates removed.
        """
        unique: List[str] = []
        duplicates_removed = 0

        iterator = tqdm(texts, desc="Deduplicating", unit="doc") if show_progress else texts

        for text in iterator:
            if not self.is_duplicate(text):
                unique.append(text)
            else:
                duplicates_removed += 1

        log.info(
            f"Deduplication complete — kept {len(unique)}/{len(texts)} "
            f"({duplicates_removed} duplicates removed)"
        )
        return unique

    def reset(self) -> None:
        """Clear all stored signatures (reset the deduplicator state)."""
        self._signatures.clear()
        self._seen_exact.clear()
