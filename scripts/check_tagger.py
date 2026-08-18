#!/usr/bin/env python3
"""spaCy vs compromise.js POS agreement.

The claim under test is the design notes row 1: wrong-part-of-speech is "Blocking,
handled by POS tagger, deterministic". That only holds if the offline tagger
(spaCy, which the inventory's POS patterns are written against) and the runtime
tagger (compromise.js, which decides what matches in the browser) agree.

Overall agreement is a misleading number -- it is inflated by determiners and
punctuation that no span pattern keys on. What matters is agreement restricted
to MATCH-ELIGIBLE tokens: the content words the proposer table actually keys on
(ADJ, ADV, VERB per the design notes span patterns).

    python scripts/check_tagger.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "data" / "tables"
TAGGER_JS = Path(__file__).resolve().parent / "tag_compromise.js"

SENTENCES_PER_SOURCE = 200
MATCH_ELIGIBLE = {"ADJ", "ADV", "VERB"}
CONTENT = {"ADJ", "ADV", "VERB", "NOUN", "PROPN"}

# Auxiliaries and modals are tagged AUX by spaCy and Verb by compromise. The
# proposer table keys on lexical verbs, never on these, so counting them as
# disagreements understates agreement on tokens that can actually match.
AUXILIARY_LEMMAS = {
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "ought", "'s", "'re", "'ve", "'ll", "'d", "'m",
}


def log(msg: str) -> None:
    print(f"[tagger] {msg}", flush=True)


def compromise_tag(sentences: list[str]) -> list[list[list[str]]]:
    proc = subprocess.run(
        ["node", str(TAGGER_JS)],
        input=json.dumps({"sentences": sentences}),
        capture_output=True, text=True, timeout=600,
        cwd=str(TAGGER_JS.parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compromise failed: {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def align(spacy_toks, compromise_toks):
    """Align two tokenizations by surface form, skipping where they diverge."""
    pairs = []
    i = j = 0
    while i < len(spacy_toks) and j < len(compromise_toks):
        st, sp = spacy_toks[i]
        ct, cp = compromise_toks[j]
        if st.lower() == ct.lower():
            pairs.append((st, sp, cp))
            i += 1
            j += 1
        elif len(st) < len(ct):
            i += 1
        else:
            j += 1
    return pairs


def main() -> None:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    report = {"sentences_per_source": SENTENCES_PER_SOURCE, "sources": {}}
    confusion: Counter = Counter()

    for parquet in sorted(INTERIM.glob("*.parquet")):
        source = parquet.stem
        texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
        if not texts:
            continue
        rng = random.Random(7)
        sample = rng.sample(texts, min(SENTENCES_PER_SOURCE, len(texts)))

        spacy_docs = [
            [(t.text, t.pos_) for t in doc if not t.is_punct and not t.is_space]
            for doc in nlp.pipe(sample, batch_size=64)
        ]
        comp_docs = compromise_tag(sample)

        n_aligned = n_eligible = n_agree_eligible = 0
        n_agree_all = 0
        # Same, excluding auxiliary/modal tokens the table cannot key on.
        n_lex = n_agree_lex = 0
        # The dangerous direction: spaCy says not-eligible, compromise says
        # eligible -> the runtime matches a span the inventory never sanctioned.
        n_false_match = n_false_match_content = 0
        # The cheap direction: spaCy eligible, compromise not -> recall loss.
        n_missed_match = 0

        for sd, cd in zip(spacy_docs, comp_docs):
            for tok, sp, cp in align(sd, cd):
                n_aligned += 1
                is_aux = tok.lower() in AUXILIARY_LEMMAS
                if sp == cp:
                    n_agree_all += 1
                if sp in MATCH_ELIGIBLE:
                    n_eligible += 1
                    if not is_aux:
                        n_lex += 1
                    if sp == cp:
                        n_agree_eligible += 1
                        if not is_aux:
                            n_agree_lex += 1
                    else:
                        n_missed_match += cp not in MATCH_ELIGIBLE
                        confusion[f"{sp}->{cp}"] += 1
                elif cp in MATCH_ELIGIBLE:
                    n_false_match += 1
                    if sp in CONTENT and not is_aux:
                        n_false_match_content += 1
                    confusion[f"{sp}->{cp}"] += 1

        stats = {
            "aligned_tokens": n_aligned,
            "overall_agreement": round(n_agree_all / max(1, n_aligned), 4),
            "match_eligible_tokens": n_eligible,
            "match_eligible_agreement": round(n_agree_eligible / max(1, n_eligible), 4),
            "lexical_tokens": n_lex,
            "lexical_agreement": round(n_agree_lex / max(1, n_lex), 4),
            "false_match_tokens": n_false_match,
            "false_match_rate": round(n_false_match / max(1, n_aligned), 4),
            "false_match_content_tokens": n_false_match_content,
            "false_match_content_rate": round(n_false_match_content / max(1, n_aligned), 4),
            "missed_match_tokens": n_missed_match,
        }
        report["sources"][source] = stats
        log(f"{source}: eligible {stats['match_eligible_agreement']:.1%} "
            f"lexical-only {stats['lexical_agreement']:.1%} "
            f"false-match {stats['false_match_rate']:.2%} "
            f"(content-only {stats['false_match_content_rate']:.2%})")

    if report["sources"]:
        srcs = report["sources"].values()
        elig = sum(s["match_eligible_tokens"] for s in srcs)
        agree = sum(s["match_eligible_agreement"] * s["match_eligible_tokens"] for s in srcs)
        report["pooled_match_eligible_agreement"] = round(agree / max(1, elig), 4)
        lex = sum(s["lexical_tokens"] for s in srcs)
        lex_agree = sum(s["lexical_agreement"] * s["lexical_tokens"] for s in srcs)
        report["pooled_lexical_agreement"] = round(lex_agree / max(1, lex), 4)
        report["top_confusions"] = confusion.most_common(15)

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "pass0_tagger_agreement.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"pooled match-eligible agreement: "
        f"{report.get('pooled_match_eligible_agreement', 0):.1%}")
    log(f"report -> {out}")


if __name__ == "__main__":
    main()
