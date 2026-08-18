"""Correctness tests for the network-free half of the pipeline.

Run: python3 -m unittest discover -s scripts/tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import clean, dedup, segment  # noqa: E402


GUTENBERG_SAMPLE = """\
The Project Gutenberg eBook of Some Novel, by A. Writer

*** START OF THE PROJECT GUTENBERG EBOOK SOME NOVEL ***

Produced by Someone and the Online Distributed Proofreading Team.

The sweltering afternoon dragged on and everyone in the small
parlour grew steadily more irritable as the clock advanced.

Mrs. Hale had put off the difficult conversation for a week. She
did not intend to postpone it again, however, and so she spoke.

*** END OF THE PROJECT GUTENBERG EBOOK SOME NOVEL ***

This eBook is for the use of anyone anywhere at no cost.
"""


class TestGutenberg(unittest.TestCase):
    def test_boilerplate_stripped_from_both_ends(self):
        body = clean.strip_gutenberg_boilerplate(GUTENBERG_SAMPLE)
        self.assertNotIn("Project Gutenberg eBook of Some Novel", body)
        self.assertNotIn("for the use of anyone anywhere", body)
        self.assertIn("sweltering afternoon", body)

    def test_frontmatter_dropped(self):
        body = clean.drop_gutenberg_frontmatter(
            clean.strip_gutenberg_boilerplate(GUTENBERG_SAMPLE)
        )
        self.assertNotIn("Online Distributed Proofreading", body)

    def test_unwrapping_rejoins_wrapped_lines(self):
        """The documented gotcha: wrapped lines must not become sentences."""
        body = clean.unwrap_paragraphs(
            clean.strip_gutenberg_boilerplate(GUTENBERG_SAMPLE)
        )
        self.assertIn("in the small parlour grew", body)
        self.assertIn("for a week. She did not intend", body)

    def test_unwrapping_preserves_paragraph_breaks(self):
        body = clean.unwrap_paragraphs("Line one here.\nWrapped on.\n\nNew para.")
        self.assertEqual(body.count("\n\n"), 1)

    def test_unwrapping_preserves_short_terminated_lines(self):
        verse = "Roses are red.\nViolets are blue.\nSugar is sweet."
        self.assertEqual(clean.unwrap_paragraphs(verse).count("\n"), 2)

    def test_line_hyphenation_rejoined(self):
        """Found by hand-reading Pass 0 samples: 'Chris- tian Science'.

        Silently corrupts frequency counts, so it matters more than it looks.
        """
        text = "The elucidation of Chris-\ntian Science lies in its sense."
        self.assertIn("Christian", clean.rejoin_hyphenation(text))

    def test_real_hyphenated_compound_preserved(self):
        """A hyphen before a capital or a line end is not a wrap artifact."""
        self.assertIn("well-\nKnown", clean.rejoin_hyphenation("well-\nKnown"))

    def test_emphasis_markers_stripped(self):
        self.assertEqual(
            clean.strip_emphasis_markers("_Corallium rubrum_ has been studied"),
            "Corallium rubrum has been studied",
        )

    def test_end_to_end_segmentation(self):
        text = clean.normalize(
            clean.unwrap_paragraphs(
                clean.drop_gutenberg_frontmatter(
                    clean.strip_gutenberg_boilerplate(GUTENBERG_SAMPLE)
                )
            )
        )
        sentences = segment.segment_document(text)
        # One sentence in the first paragraph, two in the second.
        self.assertEqual(len(sentences), 3, sentences)
        # The wrap must not split: the whole first paragraph is one sentence.
        self.assertTrue(sentences[0].startswith("The sweltering afternoon"))
        self.assertTrue(sentences[0].endswith("as the clock advanced."))
        # "Mrs." must not split.
        self.assertTrue(any(s.startswith("Mrs. Hale had put off") for s in sentences))


class TestNormalize(unittest.TestCase):
    def test_smart_punctuation_folded(self):
        out = clean.normalize("He said “it’s fine” — really…")
        self.assertEqual(out, 'He said "it\'s fine" -- really...')

    def test_pii_and_urls_scrubbed(self):
        out = clean.normalize(
            "Mail me at a.b@example.com or see https://example.com/x now"
        )
        self.assertNotIn("@example.com", out)
        self.assertNotIn("https", out)

    def test_whitespace_collapsed(self):
        self.assertEqual(clean.normalize("a   b \t c"), "a b c")


class TestSentenceFilters(unittest.TestCase):
    def test_length_bounds_enforced(self):
        self.assertFalse(clean.is_acceptable_sentence("Too short here."))
        self.assertTrue(
            clean.is_acceptable_sentence(
                "The sweltering afternoon dragged on and everyone got annoyed."
            )
        )
        self.assertFalse(clean.is_acceptable_sentence(" ".join(["word"] * 41)))

    def test_all_caps_rejected(self):
        self.assertFalse(
            clean.is_acceptable_sentence("CLICK HERE NOW TO SUBSCRIBE TODAY PLEASE")
        )

    def test_name_list_rejected(self):
        self.assertFalse(
            clean.is_acceptable_sentence(
                "John Smith, Mary Jones, Peter Brown, Alice Green, Bob White."
            )
        )

    def test_numeral_heavy_rejected(self):
        self.assertFalse(
            clean.is_acceptable_sentence("In 1994 1995 1996 1997 1998 1999 2000 rose.")
        )

    def test_unterminated_fragment_rejected(self):
        """Found by hand-reading Pass 0 samples: anchor-stripped truncations."""
        self.assertFalse(
            clean.is_acceptable_sentence(
                "EDIT in response to thought-provoking comment by"
            )
        )
        self.assertFalse(
            clean.is_acceptable_sentence(
                "People downloading to sample who then go on to pay or wouldn't anyway"
            )
        )

    def test_repetition_residue_rejected(self):
        """Found by hand-reading Pass 0 samples: collapsed link text."""
        self.assertFalse(
            clean.is_acceptable_sentence(
                "This and this and this and this and this and this and this."
            )
        )

    def test_lead_in_colon_rejected(self):
        self.assertFalse(
            clean.is_acceptable_sentence(
                "Yossarian describes how he makes a boring task interesting thus:"
            )
        )


class TestSpacingArtifacts(unittest.TestCase):
    def test_space_before_punctuation_removed(self):
        """Inline-markup stripping leaves 'Catch 22 ,' — corrupts tokenization."""
        self.assertEqual(
            clean.normalize("In chapter 1 of Catch 22 , Yossarian describes ."),
            "In chapter 1 of Catch 22, Yossarian describes.",
        )

    def test_space_after_open_bracket_removed(self):
        self.assertEqual(clean.normalize("a ( b ) c"), "a (b) c")

    def test_em_dash_split_from_words(self):
        self.assertEqual(clean.normalize("pitch--seemed"), "pitch -- seemed")


class TestWordTokenizer(unittest.TestCase):
    """The tokenizer bug that was steering retention toward noise."""

    def test_em_dash_not_absorbed(self):
        self.assertEqual(clean.words("pitch--seemed"), ["pitch", "seemed"])
        self.assertEqual(clean.words("deferred--or"), ["deferred", "or"])

    def test_real_compounds_preserved(self):
        self.assertEqual(clean.words("cross-beams"), ["cross-beams"])
        self.assertEqual(clean.words("oyster-bed"), ["oyster-bed"])

    def test_trailing_apostrophe_dropped(self):
        self.assertEqual(clean.words("kissin'"), ["kissin"])

    def test_contraction_preserved(self):
        self.assertEqual(clean.words("don't"), ["don't"])


class TestStackExchange(unittest.TestCase):
    def test_code_and_quotes_removed(self):
        html = (
            "<p>You should <em>defer</em> the meeting.</p>"
            "<pre><code>print('hello')</code></pre>"
            "<blockquote>Someone else said this.</blockquote>"
            "<p>That&#39;s my view.</p>"
        )
        out = clean.clean_stackexchange_body(html)
        self.assertIn("defer the meeting", " ".join(out.split()))
        self.assertNotIn("print", out)
        self.assertNotIn("Someone else said", out)
        self.assertIn("That's my view", " ".join(out.split()))


class TestDedup(unittest.TestCase):
    def test_exact_duplicates_caught(self):
        d = dedup.Deduplicator()
        s = "The sweltering afternoon dragged on and everyone got annoyed."
        self.assertFalse(d.is_duplicate(s))
        self.assertTrue(d.is_duplicate(s))
        self.assertTrue(d.is_duplicate("the  SWELTERING afternoon dragged on and everyone got annoyed"))
        self.assertEqual(d.stats["exact_dupes"], 2)

    def test_near_duplicates_caught(self):
        d = dedup.Deduplicator()
        base = (
            "The committee announced on Tuesday that the proposal would be "
            "deferred until the following year, citing budget concerns."
        )
        edited = (
            "The committee announced on Tuesday that the proposal would be "
            "deferred until the following year, citing budgetary concerns."
        )
        self.assertFalse(d.is_duplicate(base))
        self.assertTrue(d.is_duplicate(edited))

    def test_distinct_sentences_kept(self):
        d = dedup.Deduplicator()
        self.assertFalse(
            d.is_duplicate("The sweltering afternoon dragged on and everyone got annoyed.")
        )
        self.assertFalse(
            d.is_duplicate("She decided to postpone the difficult conversation again.")
        )
        self.assertEqual(d.stats["kept"], 2)


if __name__ == "__main__":
    unittest.main()


class TestDoubleSpacing(unittest.TestCase):
    """Caught by comparing mean sentence length against raw mirror files."""

    DOUBLE = (
        "We both started from our seats, and ran to\n\n"
        "the window, which was open to admit the\n\n"
        "cool air, though the blind was down.\n\n\n"
        "A new paragraph begins here and it also\n\n"
        "wraps across two lines."
    )

    def test_collapses_to_two_paragraphs(self):
        out = clean.collapse_double_spacing(self.DOUBLE)
        self.assertEqual(out.count("\n\n"), 1)

    def test_unwraps_into_full_sentences(self):
        body = clean.unwrap_paragraphs(clean.collapse_double_spacing(self.DOUBLE))
        text = clean.normalize(body)
        self.assertIn("ran to the window", text)
        self.assertIn("paragraph begins here and it also wraps", text)

    def test_normal_text_untouched(self):
        normal = "Line one wraps\nonto line two here.\n\nSecond paragraph now."
        self.assertEqual(clean.collapse_double_spacing(normal), normal)


class TestVerseReferences(unittest.TestCase):
    """Scripture verse numbers slipped past the numeral-ratio rule."""

    def test_verse_prefixed_sentence_rejected(self):
        self.assertFalse(clean.is_acceptable_sentence(
            "27:41 And Esau hated Jacob because of the blessing his father gave."))

    def test_ordinary_sentence_with_time_kept(self):
        self.assertTrue(clean.is_acceptable_sentence(
            "The train leaves at 9:45 and arrives before the meeting starts."))


class TestDegenerateContexts(unittest.TestCase):
    """A Roget's Thesaurus entry reached the teacher in the first label pilot."""

    def test_thesaurus_entry_rejected(self):
        self.assertFalse(clean.is_acceptable_sentence(
            "937. harmless; inoffensive, innoxious, innocuous; dove-like, "
            "lamblike; pure, harmless as doves; innocent as a lamb."))

    def test_bracket_tagged_gloss_rejected(self):
        self.assertFalse(clean.is_acceptable_sentence(
            "harmless; inoffensive, innoxious[obs3], innocuous, dove-like here."))

    def test_ordinary_sentence_with_commas_kept(self):
        self.assertTrue(clean.is_acceptable_sentence(
            "She packed the bags, locked the door, and drove to the airport."))
