"""The frozen paraphrase set: what it accepts, what it refuses, and why.

A paraphrase that changes content turns the framing arm of the battery into a
second, weaker content ablation, and nothing downstream would show it. These
tests are the only place that failure mode is caught.
"""
import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import paraphrase
from orchestrator.provenance.generator import ScriptedGenerator

ORIGINAL = "The founder is Josiah Fenn, who opened the Aldric Preserve in 1976."


class Check(unittest.TestCase):
    def accept(self, rewrite):
        return paraphrase.check(ORIGINAL, rewrite)

    def test_a_faithful_rewrite_is_accepted(self):
        accepted, reason = self.accept(
            "Josiah Fenn founded it; he opened the Aldric Preserve in 1976.")
        self.assertTrue(accepted, reason)

    def test_a_rewrite_that_drops_a_name_is_rejected(self):
        accepted, reason = self.accept("The founder opened the preserve in 1976.")
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("dropped:"), reason)

    def test_a_rewrite_that_drops_a_date_is_rejected(self):
        accepted, reason = self.accept(
            "Josiah Fenn is the founder, who opened the Aldric Preserve.")
        self.assertFalse(accepted)
        self.assertIn("1976", reason)

    def test_a_rewrite_that_invents_a_number_is_rejected(self):
        accepted, reason = self.accept(
            "Josiah Fenn, 42, founded and opened the Aldric Preserve in 1976.")
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("added_numbers:"), reason)

    def test_an_unchanged_rewrite_is_rejected_rather_than_run_as_a_no_op(self):
        self.assertEqual(self.accept(ORIGINAL), (False, "unchanged"))
        self.assertEqual(self.accept(""), (False, "empty"))

    def test_a_rewrite_of_wildly_different_length_is_rejected(self):
        accepted, reason = self.accept(ORIGINAL + " " + ORIGINAL + " " + ORIGINAL)
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("length_ratio_"), reason)


class Split(unittest.TestCase):
    def test_the_split_is_lossless(self):
        text = "The founder is Josiah Fenn. He opened it in 1976. It closed later."
        self.assertEqual("".join(core + gap for core, gap in paraphrase.split(text)), text)
        self.assertEqual(len(paraphrase.sentences(text)), 3)

    def test_an_abbreviation_does_not_end_a_sentence(self):
        # Splitting here would hand "Dr." and "Kell directed it." to the
        # rewriter separately, and half a rewrite of half a name is the exact
        # failure this module exists to stop.
        text = "Dr. Kell directed it. The year was 1976."
        self.assertEqual(paraphrase.sentences(text),
                         ["Dr. Kell directed it.", "The year was 1976."])

    def test_an_initial_does_not_end_a_sentence(self):
        self.assertEqual(paraphrase.sentences("J. Fenn founded it."),
                         ["J. Fenn founded it."])

    def test_a_decimal_does_not_end_a_sentence(self):
        self.assertEqual(paraphrase.sentences("It is 8.5 metres wide."),
                         ["It is 8.5 metres wide."])

    def test_a_single_sentence_is_one_piece(self):
        self.assertEqual(paraphrase.sentences(ORIGINAL), [ORIGINAL])


class Sentencewise(unittest.TestCase):
    """The fix for the binding constraint: 86 of 99 planner messages had their
    whole-message rewrite rejected, every one for dropping a name out of a long
    fact-dense assertion. Per sentence, only the offending clause fails."""

    TEXT = ("The founder is Josiah Fenn. He opened the Aldric Preserve in 1976. "
            "The site covers 40 hectares.")

    def rewriter(self, script):
        return ScriptedGenerator(
            lambda agent, step, messages:
            script(messages[-1]["content"].split("MESSAGE:\n")[-1].strip()))

    def faithful(self, sentence):
        return f"Put differently: {sentence}"

    def test_one_bad_sentence_does_not_cost_the_whole_message(self):
        def script(sentence):
            # The middle sentence loses a name and a date, exactly the failure
            # that rejected 87% of whole-message rewrites.
            if "Aldric" in sentence:
                return "He opened the preserve."
            return self.faithful(sentence)
        entries = paraphrase.build_sentencewise(self.rewriter(script), [self.TEXT])
        entry = entries[self.TEXT]
        self.assertTrue(entry["accepted"], entry["reason"])
        self.assertEqual(entry["sentences_total"], 3)
        self.assertEqual(entry["sentences_rewritten"], 2)
        # The failed sentence survives verbatim, so no content was lost.
        self.assertIn("He opened the Aldric Preserve in 1976.", entry["rewrite"])
        self.assertIn("Put differently: The founder is Josiah Fenn.", entry["rewrite"])

    def test_the_whole_message_check_still_gates_the_reassembly(self):
        entries = paraphrase.build_sentencewise(
            self.rewriter(lambda sentence: self.faithful(sentence)), [self.TEXT])
        accepted, reason = paraphrase.check(self.TEXT, entries[self.TEXT]["rewrite"])
        self.assertTrue(accepted, reason)

    def test_a_message_where_every_sentence_fails_is_rejected_not_run(self):
        entries = paraphrase.build_sentencewise(
            self.rewriter(lambda sentence: "nope"), [self.TEXT])
        entry = entries[self.TEXT]
        self.assertFalse(entry["accepted"])
        self.assertEqual(entry["reason"], "no_sentence_rewritten")
        self.assertEqual(entry["sentences_rewritten"], 0)

    def test_the_failed_sentences_keep_their_reasons(self):
        def script(sentence):
            # Same length, one name and one date gone: rejected for content
            # rather than for length, which is the failure mode that mattered.
            return ("He opened the nature reserve during that decade."
                    if "Aldric" in sentence else self.faithful(sentence))
        entries = paraphrase.build_sentencewise(self.rewriter(script), [self.TEXT])
        reasons = entries[self.TEXT]["sentence_reasons"]
        self.assertEqual(list(reasons), ["He opened the Aldric Preserve in 1976."])
        self.assertTrue(reasons["He opened the Aldric Preserve in 1976."]
                        .startswith("dropped:"), reasons)

    def test_a_repeated_sentence_is_only_asked_about_once(self):
        generator = self.rewriter(self.faithful)
        repeated = "The founder is Josiah Fenn. The founder is Josiah Fenn."
        paraphrase.build_sentencewise(generator, [repeated, "The founder is Josiah Fenn."])
        self.assertEqual(len(generator.calls), 1)

    def test_coverage_is_recorded_rather_than_assumed(self):
        def script(sentence):
            return "nope" if "Aldric" in sentence else self.faithful(sentence)
        entries = paraphrase.build_sentencewise(self.rewriter(script), [self.TEXT])
        self.assertEqual(paraphrase.coverage(entries),
                         {"sentences_total": 3, "sentences_rewritten": 2,
                          "rewritten_share": round(2 / 3, 4)})

    def test_a_late_message_is_minted_in_the_same_mode(self):
        frozen = paraphrase.Frozen({}, self.rewriter(self.faithful),
                                   mode=paraphrase.SENTENCE)
        rewrite = frozen(self.TEXT)
        self.assertIsNotNone(rewrite)
        self.assertEqual(frozen.late[self.TEXT]["mode"], paraphrase.SENTENCE)
        self.assertEqual(frozen.late[self.TEXT]["sentences_rewritten"], 3)


class Freeze(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        episode = self.root / "episodes" / "e1"
        episode.mkdir(parents=True)
        (episode / "bus.jsonl").write_text("\n".join(json.dumps(m) for m in [
            {"id": "m0", "step": 0, "sender": "planner", "to": "all",
             "kind": "assertion", "text": ORIGINAL},
            {"id": "m1", "step": 0, "sender": "planner", "to": "agent-1",
             "kind": "query", "text": "What did Kell direct?"},
            {"id": "m2", "step": 1, "sender": "agent-1", "to": "planner",
             "kind": "evidence", "text": "Aldric Preserve."}]) + "\n")

    def rewriter(self, text):
        return ScriptedGenerator(lambda agent, step, messages: text)

    def test_only_planner_assertions_and_compositions_are_collected(self):
        self.assertEqual(paraphrase.planner_texts(self.root), [ORIGINAL])

    def test_a_frozen_set_records_its_rejects_rather_than_regenerating(self):
        _, payload = paraphrase.freeze(self.rewriter("nope"), self.root)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["rejected"], 1)
        self.assertFalse(payload["entries"][ORIGINAL]["accepted"])
        self.assertTrue(payload["entries"][ORIGINAL]["reason"])

    def test_a_frozen_set_is_never_silently_regenerated(self):
        paraphrase.freeze(self.rewriter("nope"), self.root)
        with self.assertRaises(FileExistsError):
            paraphrase.freeze(self.rewriter("nope"), self.root)

    def test_loading_verifies_the_digest(self):
        good = "Josiah Fenn founded it; he opened the Aldric Preserve in 1976."
        path, _ = paraphrase.freeze(self.rewriter(good), self.root)
        accepted, payload = paraphrase.load(path)
        self.assertEqual(accepted, {ORIGINAL: good})
        tampered = json.loads(path.read_text())
        tampered["entries"][ORIGINAL]["rewrite"] = "something else entirely"
        path.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            paraphrase.load(path)

    def test_the_spot_check_sample_is_marked_and_reported_as_unreviewed(self):
        good = "Josiah Fenn founded it; he opened the Aldric Preserve in 1976."
        _, payload = paraphrase.freeze(self.rewriter(good), self.root)
        self.assertTrue(payload["entries"][ORIGINAL]["sampled"])
        self.assertEqual(paraphrase.unreviewed(payload), (1, 1))
        payload["entries"][ORIGINAL]["spot_check"] = "ok"
        self.assertEqual(paraphrase.unreviewed(payload), (0, 1))


class LateMessages(unittest.TestCase):
    """Paraphrasing the planner changes what the workers read, so they answer
    differently, so the planner says things it never said in the original
    episode. Those messages cannot be in a set frozen from the original bus,
    and the first real run died on exactly that."""

    GOOD = "Josiah Fenn founded it; he opened the Aldric Preserve in 1976."
    NEW = "The university he attended was Calverton."

    def rewriter(self, text=None):
        """Echoes a faithful rewrite of whatever it is given, so a minted
        paraphrase passes the same check a frozen one does."""
        def script(agent, step, messages):
            if text is not None:
                return text
            body = messages[-1]["content"].split("MESSAGE:\n")[-1].strip()
            return f"Put differently: {body}"
        return ScriptedGenerator(script)

    def test_a_message_the_freeze_never_saw_is_minted_and_recorded(self):
        frozen = paraphrase.Frozen({ORIGINAL: self.GOOD}, self.rewriter())
        self.assertEqual(frozen(self.NEW), f"Put differently: {self.NEW}")
        self.assertIn(self.NEW, frozen.late)
        self.assertTrue(frozen.late[self.NEW]["late"])

    def test_a_late_rewrite_still_has_to_pass_the_same_check(self):
        frozen = paraphrase.Frozen({}, self.rewriter("nope"))
        self.assertIsNone(frozen("The founder is Josiah Fenn."))
        self.assertIn("The founder is Josiah Fenn.", frozen.rejected)
        self.assertEqual(frozen.late, {})

    def test_a_reloaded_set_does_not_ask_about_its_own_rejects_again(self):
        """Reloading a frozen set used to forget which texts it had already
        refused, so a resumed pass asked the model about them a second time --
        which is the "regenerate until one passes" this module forbids."""
        payload = {"entries": {"bad text": {"rewrite": "", "accepted": False,
                                            "reason": "dropped:Fenn"}}}
        generator = self.rewriter()
        frozen = paraphrase.Frozen({}, generator,
                                   rejected=paraphrase.refusals(payload))
        self.assertIsNone(frozen("bad text"))
        self.assertEqual(generator.calls, [])

    def test_a_rejected_message_is_not_asked_about_twice(self):
        generator = self.rewriter("nope")
        frozen = paraphrase.Frozen({}, generator)
        frozen("The founder is Josiah Fenn.")
        frozen("The founder is Josiah Fenn.")
        self.assertEqual(len(generator.calls), 1)

    def test_without_a_generator_an_unseen_message_stays_unavailable(self):
        frozen = paraphrase.Frozen({ORIGINAL: self.GOOD})
        self.assertEqual(frozen(ORIGINAL), self.GOOD)
        self.assertIsNone(frozen("something else"))

    def test_late_additions_are_written_back_and_re_digested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episodes" / "e1"
            episode.mkdir(parents=True)
            (episode / "bus.jsonl").write_text(json.dumps(
                {"id": "m0", "step": 0, "sender": "planner", "to": "all",
                 "kind": "assertion", "text": ORIGINAL}) + "\n")
            path, payload = paraphrase.freeze(self.rewriter(self.GOOD), root)
            frozen = paraphrase.Frozen({ORIGINAL: self.GOOD}, self.rewriter())
            frozen(self.NEW)
            updated = frozen.record(path)
            self.assertEqual(updated["late"], 1)
            # The digest still matches its own entries, so `load` accepts it.
            accepted, reloaded = paraphrase.load(path)
            self.assertIn(self.NEW, accepted)
            self.assertTrue(reloaded["entries"][self.NEW]["late"])


if __name__ == "__main__":
    unittest.main()
