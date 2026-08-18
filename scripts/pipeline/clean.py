"""Extraction and normalization.

Pure stdlib on purpose: this is the stage with the documented correctness
hazards, so it must be testable without installing anything.
"""

import re
import unicodedata

from . import config

# --- Project Gutenberg ------------------------------------------------------

_PG_START = re.compile(
    r"^\s*\*\*\*\s*START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PG_END = re.compile(
    r"^\s*\*\*\*\s*END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop everything outside the START/END markers.

    PG permits removing its header and footer; what remains is public domain.
    Falls back to the whole text when markers are absent (a few old files).
    """
    start = _PG_START.search(text)
    body = text[start.end():] if start else text
    end = _PG_END.search(body)
    return body[: end.start()] if end else body


# Typeset text hyphenates across line breaks: "Chris-\ntian" must rejoin as
# "Christian", not "Chris- tian". Found by hand-reading Pass 0 Gutenberg
# samples. This one is not cosmetic -- it silently corrupts the frequency
# counts that the whole inventory is derived from.
_LINE_HYPHEN = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-z])")

# Gutenberg marks italics with underscores: _Corallium rubrum_.
_UNDERSCORE_EMPHASIS = re.compile(r"_([^_\n]{1,80})_")


def rejoin_hyphenation(text: str) -> str:
    """Rejoin words split by an end-of-line hyphen."""
    return _LINE_HYPHEN.sub(r"\1\2", text)


def strip_emphasis_markers(text: str) -> str:
    """Remove Gutenberg's _italics_ markers, keeping the text."""
    return _UNDERSCORE_EMPHASIS.sub(r"\1", text)


def collapse_double_spacing(text: str) -> str:
    """Collapse blank-line-per-wrapped-line formatting before unwrapping.

    Some Gutenberg redistributions are double-spaced: every hard-wrapped line
    is followed by a blank line, and paragraph breaks are two or more blank
    lines. Paragraph-based unwrapping then treats each *line* as a paragraph
    and never rejoins anything, which truncates sentences at line boundaries.

    Caught by comparing mean sentence length against raw mirror files: 9 tokens
    versus the 32 Pass 0 measured. Nothing else in the pipeline would have
    flagged it -- the sentences are individually well-formed, just cut short.
    """
    lines = text.split("\n")
    if not lines:
        return text
    blank = sum(1 for line in lines if not line.strip())
    if blank / len(lines) < 0.35:
        return text
    # Order matters: mark real paragraph breaks (2+ blanks) before collapsing
    # the single blank lines that are merely wrap artifacts.
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\x00", text)
    text = re.sub(r"\n[ \t]*\n", "\n", text)
    return text.replace("\x00", "\n\n")


def unwrap_paragraphs(text: str) -> str:
    """Rejoin hard-wrapped lines into paragraphs.

    Gutenberg texts wrap at ~70 columns. Segmenting before unwrapping turns
    every line break into a false sentence boundary -- the single most common
    way this corpus is processed wrong.

    A blank line separates paragraphs. A single newline inside a paragraph is
    a wrap artifact and becomes a space -- except after a line that ends in a
    sentence terminator AND is short, which indicates real structure (verse,
    headings, dialogue lines) rather than wrapping.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = []
        for i, line in enumerate(lines):
            joined.append(line)
            if i + 1 < len(lines):
                # Preserve the break only for short terminated lines (verse,
                # headings); otherwise this is a wrap and gets a space.
                if len(line) < 45 and re.search(r"[.!?:]['\")\]]?$", line):
                    joined.append("\n")
                else:
                    joined.append(" ")
        out.append("".join(joined))
    return "\n\n".join(out)


def drop_gutenberg_frontmatter(text: str) -> str:
    """Remove transcriber's notes, indices, and produced-by lines."""
    text = re.sub(
        r"^\s*(?:Transcriber'?s? Note|Produced by|E-text prepared by)\b.*?(?=\n\s*\n)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    return text


# --- Stack Exchange ---------------------------------------------------------

_CODE_BLOCK = re.compile(r"<(?:pre|code)\b.*?</(?:pre|code)>", re.IGNORECASE | re.DOTALL)
_BLOCKQUOTE = re.compile(r"<blockquote\b.*?</blockquote>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def clean_stackexchange_body(html: str) -> str:
    """Strip code, quoted text, and markup from a Posts.xml body.

    Code and blockquotes go entirely: substituting inside quoted text is
    error 11, and code is error 10's cousin.
    """
    text = _CODE_BLOCK.sub(" ", html)
    text = _BLOCKQUOTE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return unescape_entities(text)


def unescape_entities(text: str) -> str:
    import html as _html

    return _html.unescape(text)


# --- Wikipedia --------------------------------------------------------------

# The HuggingFace dump has already stripped wikitext markup, so section
# headings arrive as bare lines -- "References", not "== References ==". A
# pattern requiring "=+" silently matches nothing, which is what happened.
_WIKI_TAIL_SECTIONS = re.compile(
    r"\n=*[ \t]*(?:References|External links|See also|Further reading|Notes|"
    r"Footnotes|Bibliography|Sources|Citations)[ \t]*=*[ \t]*\n.*$",
    re.IGNORECASE | re.DOTALL,
)


def drop_wikipedia_tail_sections(text: str) -> str:
    """Remove reference and category tails.

    Measured impact while this was silently a no-op: 0.010% of retained
    Wikipedia sentences, because the sentence filters (terminator required,
    length bounds, proper-noun ratio) catch the boilerplate anyway. What it
    did pollute was the raw bigram counts, which is why "external links"
    topped the collocation table by association score.
    """
    return _WIKI_TAIL_SECTIONS.sub("", text)


# --- Shared normalization ---------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_HANDLE = re.compile(r"(?<!\w)[@/][ur]/\w+|(?<!\w)@\w{2,}")
_WS = re.compile(r"[ \t]+")

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "--", "‒": "-", "―": "--",
    "…": "...", " ": " ", "​": "",
}
_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)


# Stripping inline markup (<i>, <a>, <code>) leaves a space before the
# following punctuation: "<i>Catch 22</i>," becomes "Catch 22 ,". Pervasive in
# Stack Exchange bodies and it corrupts tokenization downstream.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")


def normalize(text: str) -> str:
    """NFKC, quote/dash folding, PII scrub, whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_TABLE)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    text = re.sub(r"(?<=\w)--+(?=\w)", " -- ", text)  # pitch--seemed
    text = _WS.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def nonalpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    non = sum(1 for c in text if not (c.isalpha() or c.isspace()))
    return non / len(text)


def is_acceptable_document(text: str) -> bool:
    """Document-level quality gate."""
    if len(text) < 200:
        return False
    return nonalpha_ratio(text) <= config.MAX_NONALPHA_RATIO


# --- Sentence-level filters ----------------------------------

_TOKEN = re.compile(r"\S+")

# Word tokenizer for frequency counting.
#
# The naive form [A-Za-z][A-Za-z'-]* absorbs Gutenberg's "--" em-dash, so
# "pitch--seemed" and "deferred--or" count as single wordforms. That inflated
# the vocabulary to 202k forms of which 65% were hapax, and those hapax are
# what the retention rule selects on -- so the bug was steering retention
# toward noise. An internal hyphen or apostrophe must be followed by a letter.
WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def words(text: str) -> list[str]:
    """Lowercased word tokens for frequency counting."""
    return WORD.findall(text.lower())


def token_count(sentence: str) -> int:
    return len(_TOKEN.findall(sentence))


_TERMINATED = re.compile(r"[.!?]['\")\]]?$")
_VERSE_REF = re.compile(r"^\d{1,3}:\d{1,3}\b")
# Numbered reference entries: "937. harmless; inoffensive, innoxious[obs3]..."
_ENTRY_NUM = re.compile(r"^\d{1,4}\.\s")
_BRACKET_TAG = re.compile(r"\[[a-z]{2,6}\d?\]")


def is_acceptable_sentence(sentence: str) -> bool:
    """Drop degenerate sentences.

    Beyond the majority-proper-noun / numeral / ALL-CAPS rules, two filters
    come from hand-reading the Pass 0 samples:

    - **Unterminated fragments.** List items and text truncated by anchor-tag
      stripping ("EDIT in response to thought-provoking comment by") arrive
      without a terminator. They are not sentences and give the judge no
      context to work with.
    - **Repetition residue.** Stripped link text collapses to things like
      "This and this and this and this .", which is not English.
    """
    tokens = _TOKEN.findall(sentence)
    n = len(tokens)
    if not (config.MIN_SENTENCE_TOKENS <= n <= config.MAX_SENTENCE_TOKENS):
        return False

    if not _TERMINATED.search(sentence.strip()):
        return False

    # Verse/line references leak in from scripture and numbered editions:
    # "27:41 And Esau hated Jacob...". One numeric token in twenty-five slips
    # under the numeral-ratio rule below, so match the shape directly.
    stripped = sentence.strip()
    if _VERSE_REF.match(stripped) or _ENTRY_NUM.match(stripped):
        return False

    # Thesaurus / glossary / index lines: real prose does not carry this much
    # separator punctuation per token. A Roget's entry reached the teacher in
    # the first pilot and was graded as though it were a sentence.
    seps = sentence.count(",") + sentence.count(";")
    if n and seps / n > 0.25:
        return False
    if _BRACKET_TAG.search(sentence):
        return False

    lowered = [t.lower() for t in tokens]
    if len(set(lowered)) / n < config.MIN_UNIQUE_TOKEN_RATIO:
        return False

    alpha = [t for t in tokens if any(c.isalpha() for c in t)]
    if len(alpha) < config.MIN_SENTENCE_TOKENS:
        return False

    upper = sum(1 for t in alpha if t.isupper() and len(t) > 1)
    if upper / len(alpha) > 0.5:
        return False

    numeric = sum(1 for t in tokens if any(c.isdigit() for c in t))
    if numeric / n > 0.3:
        return False

    # Majority capitalized mid-sentence suggests a name list or a heading.
    interior = alpha[1:]
    if interior:
        capped = sum(1 for t in interior if t[0].isupper())
        if capped / len(interior) > 0.5:
            return False

    return True
