"""Pipeline constants. Every threshold in the data notes lives here, once."""

from dataclasses import dataclass, field

# --- Sentence eligibility -----------------------------
# Floor: shorter sentences carry no inferable context, which is the 2 bar.
# Ceiling: the judge has 128 positions and needs room for the candidate segment.
MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 40

# Repetition guard. Stripped link text collapses to "This and this and this .";
# real prose sits well above 0.5 unique-token ratio.
MIN_UNIQUE_TOKEN_RATIO = 0.5

# --- Document quality --------------------------------
MAX_NONALPHA_RATIO = 0.20
MIN_LANGID_CONFIDENCE = 0.90

# --- Retention rule ----------------------------------
# Retention selects sentences containing a word inside a RARITY BAND, not below
# a ceiling. Measured on the 216M-word probe, GRE-level vocabulary sits at
# 0.1-20 occurrences per million (obstreperous 0.10, gargantuan 0.65,
# clandestine 2.2, enormous 19.6) while the sub-0.05 tail is foreign-language
# leakage, proper nouns, dialect and typos. Selecting "rarest" selects noise.
#
# Rarity is computed on POOLED counts across all sources: a word's rarity is a
# property of English, not of the source it was drawn from. (Per-source counts
# are still kept -- the formality score in 4 stage 7 needs them.)
BASELINE_SAMPLE_RATE = 0.02

# The baseline sample supplies contexts for spans made only of common words
# ("very big"), which the rarity band by definition excludes. It is bounded for
# the same reason retention is: at 8.7B words a 2% rate yields ~8.7M sentences
# -- twice the coverage set and 67% of the store -- to serve a need of roughly
# 3,000 entries x 20 contexts = 60,000. The cap keeps ~8x headroom over that.
BASELINE_MAX_SENTENCES = 500_000

# The band, anchored to measured GRE-word rates rather than tuned to hit a
# retention percentage: obstreperous 0.10, sycophant 0.18, gargantuan 0.65,
# clandestine 2.2, opaque 3.0, lucid 3.2, mitigate 5.4 per million. Above ~8
# the words are common enough that a reader already knows them; below 0.05 is
# the noise tail.
RARITY_BAND = (0.05, 8.0)

# Volume is bounded per word, not globally. Pass 0 measured that NO rarity band
# lands retention in 4-8% -- the narrowest gives 32%, because a per-million
# rate is scale-invariant and nearly every sentence contains some uncommon
# word. A global rate cannot express what retention is actually for, which is
# "enough contexts for each word we might ship". A per-word cap expresses it
# exactly and bounds the store predictably.
CONTEXTS_PER_WORD = 50

# the design notes: a word needs 10 real occurrences to be shippable. This is the
# number Pass 0 should be judged on, not a retention percentage.
MIN_CONTEXTS_TO_SHIP = 10

# Diagnostic sweep only; retention is no longer chosen from this. The original
# 4-8% target is kept purely as a reference line on that curve -- Pass 0 showed
# it is unreachable by any band (narrowest gives 32%), which is what motivated
# the per-word cap above.
TARGET_RETENTION_RANGE = (0.04, 0.08)
RARITY_BAND_FLOORS = [0.02, 0.05, 0.1, 0.2]
RARITY_BAND_CEILINGS = [1.0, 2.0, 3.0, 5.0, 8.0]

# --- Near-duplicate removal --------------------------
SHINGLE_SIZE = 5
MINHASH_PERMUTATIONS = 128
JACCARD_THRESHOLD = 0.8

# --- Collocation mining -----------------------------
COLLOCATION_MIN_COUNT = 50
COLLOCATION_TOP_K = 20_000_000

# --- Probe sizing ------------------------------------
PROBE_SAMPLE_RATE = 0.05
PROBE_TAGGER_AGREEMENT_SENTENCES = 200


@dataclass(frozen=True)
class Source:
    name: str
    role: str
    formal: bool  # feeds the formality_score numerator


SOURCES = {
    s.name: s
    for s in [
        Source("gutenberg", "advanced-vocab evidence", formal=True),
        Source("wikipedia", "baseline frequency + names", formal=True),
        Source("c4news", "contemporary formal prose", formal=True),
        Source("stackexchange", "deployment register", formal=False),
    ]
}

# Stack Exchange sites to take -- deliberately excluding Stack Overflow and the
# programming sites so we are not training on code-adjacent prose.
STACKEXCHANGE_SITES = [
    "english", "writers", "history", "skeptics", "interpersonal",
    "movies", "literature", "philosophy", "politics", "travel",
    "cooking", "worldbuilding",
]
