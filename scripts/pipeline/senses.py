"""Sense constraints and the context gate.

Two layers, cheapest first:

  check_constraints  Deterministic. Each entry is keyed to one sense of the
                     trigger and carries block/require word lists derived from
                     observed failures. Free, exact, and auditable.

  ContextGate        Statistical backstop. Compares the proposed sentence with
                     the corpus contexts where the candidate actually occurs.
                     Measured: 81.5% on a signal test, but only ~33% of
                     wrong-sense blocked at a recall-preserving cutoff -- so it
                     runs as a cheap PRE-FILTER ahead of the teacher, never as
                     the decision.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import clean

INVENTORY = Path(__file__).resolve().parents[2] / "data" / "inventory"

# Measured operating point: keeps 92% of proposals, blocks 33% of constructed
# wrong-sense negatives. Deliberately permissive -- its job is to reduce what
# the teacher must adjudicate, not to make the call.
GATE_PERCENTILE = 0.34


def load_inversions(path: Path | None = None) -> list[dict]:
    path = path or (INVENTORY / "inversions_v3.tsv")
    with path.open() as fh:
        rows = [r for r in csv.DictReader(
            (l for l in fh if not l.startswith("#")), delimiter="\t")
            if r.get("trigger")]
    deferred = load_deferred()
    return [r for r in rows if r["hard_word"] not in deferred]


def load_deferred() -> set[str]:
    path = INVENTORY / "deferred_polysemous.tsv"
    if not path.exists():
        return set()
    with path.open() as fh:
        return {r["word"] for r in csv.DictReader(fh, delimiter="\t")}


def _terms(field: str) -> set[str]:
    return {t.strip().lower() for t in (field or "").split("|") if t.strip()}


def check_constraints(sentence: str, rule: dict) -> tuple[bool, str]:
    """Deterministic sense check. Returns (ok, reason_if_blocked)."""
    words = set(clean.words(sentence))
    # The trigger's own tokens are always present, so a rule that listed one of
    # them in block_context would veto every sentence it matched. Caught by
    # test_terse_accepts_speech, where "brief" blocked itself.
    words -= set(clean.words(rule.get("trigger", "")))
    lowered = " " + " ".join(clean.words(sentence)) + " "

    for term in _terms(rule.get("block_context", "")):
        if _present(term, words, lowered):
            return False, f"block_context: {term}"
    require = _terms(rule.get("require_context", ""))
    if require and not any(_present(t, words, lowered) for t in require):
        return False, "require_context: none present"
    return True, ""


def _present(term: str, words: set[str], lowered: str) -> bool:
    """Match single words by token, multi-word terms as a phrase.

    The generated table is full of phrase constraints ("fake tan", "false
    alarm"). Set-membership alone silently never fires on those, so 68% of
    rows carried block lists that were partly inert.
    """
    if " " in term:
        return f" {term} " in lowered
    return term in words


class ContextGate:
    """Embedding similarity against a candidate's attested corpus contexts."""

    def __init__(self, banks: dict[str, "object"], model):
        self.banks = banks
        self.model = model
        self.words = sorted(banks)

    def percentile(self, sentence: str, candidate: str) -> float | None:
        """Where the candidate ranks among all context banks for this sentence.

        Relative, not absolute: per-word similarity scales differ, which is the
        same reason the design records v1's fixed syntax floor as a bug.
        """
        candidate = candidate.lower()
        if candidate not in self.banks:
            return None
        v = self.model.encode([sentence], normalize_embeddings=True)[0]
        scores = {w: float((self.banks[w] @ v).max()) for w in self.words}
        mine = scores[candidate]
        return sum(1 for s in scores.values() if s < mine) / len(scores)

    def passes(self, sentence: str, candidate: str,
               cutoff: float = GATE_PERCENTILE) -> bool:
        p = self.percentile(sentence, candidate)
        return True if p is None else p >= cutoff
