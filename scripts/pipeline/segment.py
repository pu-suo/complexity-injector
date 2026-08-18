"""Sentence segmentation.

Three backends, selected by availability:

  blingfire  ~10x spaCy's throughput -- what makes the tier-1 pass over 8.7B
             words affordable. x86_64 wheels only, so it loads on the Pass 1
             Debian box but NOT on Apple silicon.
  pysbd      pure python, accurate, slower. The Pass 0 backend on this laptop.
  regex      stdlib fallback so the module is testable with no dependencies.

Backends disagree, so `segmenter_name()` is recorded alongside every artifact
and Pass 0 measures pysbd against the regex fallback directly.
"""

import re

from . import clean, config

try:
    from blingfire import text_to_sentences

    text_to_sentences("probe")  # x86_64-only .dylib; import alone does not fail
    _HAVE_BLINGFIRE = True
except Exception:  # pragma: no cover - arch dependent
    _HAVE_BLINGFIRE = False

try:
    import pysbd

    _PYSBD = pysbd.Segmenter(language="en", clean=False)
except Exception:  # pragma: no cover - exercised only without the dep
    _PYSBD = None


# Abbreviations that must not end a sentence. Not exhaustive -- the fallback is
# a correctness net for tests, not the production segmenter.
_ABBREV = frozenset(
    """Mr Mrs Ms Dr Prof Sr Jr St Rev Hon Gen Col Capt Lt Sgt
    cf vs etc al Inc Ltd Co Corp No vol pp Fig approx
    Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec""".split()
)

# Candidate boundary: terminator (+ optional closer), whitespace, then something
# that looks like the start of a new sentence.
_BOUNDARY = re.compile(r"[.!?]['\")\]]?\s+(?=[\"'(\[]?[A-Z0-9])")

# The token immediately before the terminator, used for the abbreviation check.
# Python's re requires fixed-width lookbehind, so this is done by scanning.
_PRECEDING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_false_boundary(text: str, terminator_index: int) -> bool:
    head = text[:terminator_index]
    # An ellipsis is a real terminator, not an abbreviation. Without this the
    # dotted-abbreviation rule below reads "by.." out of 'Standing by..." The'
    # and welds two sentences together.
    if head.endswith(".."):
        return False
    match = _PRECEDING_WORD.search(head)
    if not match:
        return False
    word = match.group(1)
    if word in _ABBREV:
        return True
    # A single capital letter is an initial ("A. Writer"), not a sentence end.
    if len(word) == 1 and word.isupper():
        return True
    # Dotted abbreviations reach here as "e.g", "i.e", "U.S".
    if "." in word:
        return True
    return False


def _fallback_split(text: str) -> list[str]:
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        if _is_false_boundary(text, match.start()):
            continue
        end = match.start() + len(match.group().rstrip())
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str, backend: str | None = None) -> list[str]:
    """Split one paragraph of normalized text into sentences."""
    backend = backend or segmenter_name()
    if backend == "blingfire":
        return [s for s in text_to_sentences(text).split("\n") if s.strip()]
    if backend == "pysbd":
        return [s.strip() for s in _PYSBD.segment(text) if s.strip()]
    return _fallback_split(text)


def segment_document(
    text: str, backend: str | None = None, apply_filters: bool = True
) -> list[str]:
    """Paragraph-aware segmentation, returning only eligible sentences.

    Splitting per paragraph rather than over the whole document keeps a missing
    terminator at a paragraph end from welding two paragraphs together.
    """
    backend = backend or segmenter_name()
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sentence in split_sentences(para, backend):
            sentence = sentence.strip()
            if not apply_filters or clean.is_acceptable_sentence(sentence):
                out.append(sentence)
    return out


def segmenter_name() -> str:
    if _HAVE_BLINGFIRE:
        return "blingfire"
    if _PYSBD is not None:
        return "pysbd"
    return "regex-fallback"


def available_backends() -> list[str]:
    names = ["regex-fallback"]
    if _PYSBD is not None:
        names.insert(0, "pysbd")
    if _HAVE_BLINGFIRE:
        names.insert(0, "blingfire")
    return names


__all__ = [
    "split_sentences",
    "segment_document",
    "segmenter_name",
    "config",
]
