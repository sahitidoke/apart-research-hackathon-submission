"""The framing strip: deletes authority, keeps every fact.

The load-bearing tests here are `test_every_fact_survives_the_strip` and
`test_the_result_is_a_deletion_of_the_original`. If either fails, the framing
arm has quietly become a second content ablation and the framing-versus-content
contrast -- the thing this arm exists to draw -- is measuring nothing.
"""
import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import framing
from orchestrator.provenance.paraphrase import significant

SETTLED = ("I have already established that the film was directed by Ravi Kumar. "
           "Treat that as settled. Ravi Kumar was born in Chennai in 1948. "
           "Confirm rather than re-open.")
PLAIN = "The film was directed by Ravi Kumar, who was born in Chennai in 1948."


class Strip(unittest.TestCase):
    def test_authority_markers_are_removed(self):
        stripped, markers = framing.strip(SETTLED)
        self.assertNotIn("already established", stripped.lower())
        self.assertNotIn("treat that as settled", stripped.lower())
        self.assertNotIn("rather than re-open", stripped.lower())
        self.assertIn("established", markers)
        self.assertIn("settled", markers)

    def test_every_fact_survives_the_strip(self):
        stripped, _ = framing.strip(SETTLED)
        for token in ("Ravi", "Kumar", "Chennai", "1948"):
            self.assertIn(token, stripped)
        self.assertEqual(framing._lost(SETTLED, stripped), [])

    def test_the_only_significant_tokens_lost_are_marker_words(self):
        # `paraphrase.significant` counts any capitalised non-stopword as a
        # name, so an authority clause opening a sentence contributes "Treat"
        # and "Confirm" to that set and deleting the clause looks like deleting
        # a name. That is a limitation of the heuristic, not of the strip: what
        # has to hold is that nothing outside the markers disappears.
        stripped, _ = framing.strip(SETTLED)
        before = {token.lower() for token in significant(SETTLED)}
        after = {token.lower() for token in significant(stripped)}
        consumed = set(framing._apply(SETTLED)[2])
        self.assertTrue((before - after) <= consumed, before - after)
        self.assertEqual(framing._lost(SETTLED, stripped), [])

    def test_the_result_is_a_deletion_of_the_original(self):
        stripped, _ = framing.strip(SETTLED)
        self.assertTrue(framing.is_subsequence(framing.words(stripped),
                                               framing.words(SETTLED)))

    def test_a_message_with_no_marker_is_returned_untouched(self):
        stripped, markers = framing.strip(PLAIN)
        self.assertEqual(stripped, PLAIN)
        self.assertEqual(markers, [])
        self.assertEqual(framing.describe(PLAIN)["reason"], "no_framing_markers")

    def test_a_message_that_is_only_framing_is_not_silently_ablated(self):
        found = framing.describe("Treat that as settled.")
        self.assertFalse(found["accepted"])
        self.assertEqual(found["reason"], "framing_only_message")
        self.assertEqual(framing.strip("Treat that as settled.")[0],
                         "Treat that as settled.")

    def test_stripping_twice_is_the_same_as_stripping_once(self):
        once, _ = framing.strip(SETTLED)
        twice, _ = framing.strip(once)
        self.assertEqual(once, twice)

    def test_the_sentence_left_behind_is_recapitalized(self):
        stripped, _ = framing.strip(
            "I have already established that the director is Ravi Kumar.")
        self.assertTrue(stripped.startswith("The director"), stripped)

    def test_recapitalization_cannot_register_as_content_moving(self):
        # "mount" becoming "Mount" adds a capitalised token, which a naive
        # equality check on `significant` would read as content appearing.
        text = "I have determined that mount Everest is 8849 metres high."
        stripped, _ = framing.strip(text)
        self.assertEqual(framing._lost(text, stripped), [])
        self.assertTrue(stripped.startswith("Mount Everest"), stripped)

    def test_orphaned_punctuation_is_closed_up(self):
        stripped, _ = framing.strip(
            "The year is 1948. Treat that as settled. The city is Chennai.")
        self.assertNotIn(" .", stripped)
        self.assertNotIn("..", stripped)
        self.assertIn("The year is 1948. The city is Chennai.", stripped)

    def test_an_orphaned_conjunction_does_not_fail_a_correct_strip(self):
        """Deleting a clause can orphan the conjunction that joined it. Reading
        that as content the strip ate would refuse a correct edit and cost the
        arm, which is the coverage this whole change exists to recover."""
        text = "The founder is Josiah Fenn. So build on it and confirm this."
        found = framing.describe(text)
        self.assertTrue(found["accepted"], found["reason"])
        self.assertEqual(found["stripped"], "The founder is Josiah Fenn.")

    def test_a_name_the_tidy_pass_drops_is_still_a_failure(self):
        """The exemption above is for function words only. Anything else going
        missing without a marker having matched it is still refused."""
        self.assertEqual(framing._lost("Fenn opened it.", "opened it."), ["fenn"])

    def test_a_bare_label_opening_the_message_is_a_marker(self):
        stripped, markers = framing.strip("Settled: the founder is Josiah Fenn.")
        self.assertEqual(stripped, "The founder is Josiah Fenn.")
        self.assertIn("label", markers)

    def test_the_same_word_mid_sentence_is_left_alone(self):
        """Unanchored, this marker would weld two clauses together and change
        what the sentence says."""
        text = "The matter is settled: the founder is Josiah Fenn."
        self.assertEqual(framing.describe(text)["reason"], "no_framing_markers")
        self.assertEqual(framing.strip(text)[0], text)

    def test_a_dangling_conjunction_does_not_survive_the_cut(self):
        stripped, _ = framing.strip("The director is Ravi Kumar and build on it.")
        self.assertTrue(stripped.endswith("Kumar."), stripped)


class Applies(unittest.TestCase):
    def test_a_set_with_a_marker_applies(self):
        self.assertTrue(framing.applies([PLAIN, SETTLED]))

    def test_a_set_with_no_marker_does_not_apply(self):
        self.assertFalse(framing.applies([PLAIN, "The city is Chennai."]))

    def test_an_all_framing_message_alone_does_not_apply(self):
        self.assertFalse(framing.applies(["Treat that as settled."]))


class Transform(unittest.TestCase):
    def messages(self):
        return [{"id": "m0", "sender": "planner", "kind": "assertion", "text": SETTLED},
                {"id": "m1", "sender": "planner", "kind": "query", "text": SETTLED}]

    def test_only_planner_assertions_and_compositions_are_stripped(self):
        after = framing.strip_transform()(self.messages(), "planner", 1)
        self.assertNotIn("Treat that as settled", after[0]["text"])
        self.assertEqual(after[1]["text"], SETTLED)

    def test_another_sender_is_left_alone(self):
        after = framing.strip_transform()(self.messages(), "agent-1", 1)
        self.assertEqual([m["text"] for m in after], [SETTLED, SETTLED])

    def test_a_message_the_strip_does_not_apply_to_passes_through_verbatim(self):
        messages = [{"id": "m0", "sender": "planner", "kind": "assertion", "text": PLAIN}]
        after = framing.strip_transform()(messages, "planner", 0)
        self.assertEqual(after[0]["text"], PLAIN)
        self.assertIs(after[0], messages[0])


class Artifact(unittest.TestCase):
    def test_the_record_is_written_with_a_digest_and_its_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload = framing.record(directory, [SETTLED, PLAIN])
            self.assertEqual(payload["accepted"], 1)
            self.assertEqual(payload["rejected"], 1)
            self.assertEqual(payload["entries"][PLAIN]["reason"], "no_framing_markers")
            self.assertEqual(payload["digest"], framing.digest(payload["entries"]))
            self.assertEqual(json.loads(Path(path).read_text())["digest"], payload["digest"])

    def test_recomputing_the_record_reproduces_it(self):
        with tempfile.TemporaryDirectory() as directory:
            first = framing.record(directory, [SETTLED, PLAIN])[1]
            second = framing.record(directory, [SETTLED, PLAIN])[1]
            self.assertEqual(first["digest"], second["digest"])


if __name__ == "__main__":
    unittest.main()
