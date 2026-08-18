"""Exact and near-duplicate removal.

Non-optional: Gutenberg carries multiple editions of the same work and news is
syndicated near-verbatim, so without this the frequency statistics reflect
whichever text happened to be duplicated.

stdlib MinHash so the logic is testable without datasketch; swap in
datasketch's MinHashLSH for the Pass 1 scale, where the index gets large.
"""

import hashlib
import re

from . import config

_WORD = re.compile(r"[a-z0-9']+")
_MASK64 = (1 << 64) - 1


def _shingles(text: str, k: int = config.SHINGLE_SIZE) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _hash64(value: str, seed: int) -> int:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest, "little")


def minhash(text: str, permutations: int = config.MINHASH_PERMUTATIONS) -> tuple[int, ...]:
    shingles = _shingles(text)
    if not shingles:
        return tuple([_MASK64] * permutations)
    return tuple(
        min(_hash64(s, seed) for s in shingles) for seed in range(permutations)
    )


def estimate_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def exact_key(text: str) -> str:
    """Whitespace- and case-insensitive exact-duplicate key."""
    normalized = " ".join(_WORD.findall(text.lower()))
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


class Deduplicator:
    """Exact dedup, plus banded LSH for near-duplicates.

    Bands trade recall for speed: with 128 permutations in 32 bands of 4, two
    sentences collide when any band matches exactly, which approximates the
    0.8 Jaccard threshold closely enough at corpus scale.
    """

    def __init__(
        self,
        threshold: float = config.JACCARD_THRESHOLD,
        permutations: int = config.MINHASH_PERMUTATIONS,
        bands: int = 32,
    ):
        if permutations % bands:
            raise ValueError("permutations must divide evenly into bands")
        self.threshold = threshold
        self.permutations = permutations
        self.bands = bands
        self.rows = permutations // bands
        self._exact: set[str] = set()
        self._buckets: dict[tuple[int, tuple[int, ...]], list[tuple[int, ...]]] = {}
        self.stats = {"seen": 0, "exact_dupes": 0, "near_dupes": 0, "kept": 0}

    def is_duplicate(self, text: str) -> bool:
        self.stats["seen"] += 1

        key = exact_key(text)
        if key in self._exact:
            self.stats["exact_dupes"] += 1
            return True

        signature = minhash(text, self.permutations)
        candidates = []
        band_keys = []
        for b in range(self.bands):
            band = (b, signature[b * self.rows : (b + 1) * self.rows])
            band_keys.append(band)
            candidates.extend(self._buckets.get(band, ()))

        for other in candidates:
            if estimate_jaccard(signature, other) >= self.threshold:
                self.stats["near_dupes"] += 1
                return True

        self._exact.add(key)
        for band in band_keys:
            self._buckets.setdefault(band, []).append(signature)
        self.stats["kept"] += 1
        return False

    @property
    def removal_rate(self) -> float:
        seen = self.stats["seen"]
        if not seen:
            return 0.0
        return (self.stats["exact_dupes"] + self.stats["near_dupes"]) / seen
