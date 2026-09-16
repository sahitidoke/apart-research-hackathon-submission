"""Replay fidelity and the perturbation battery.

The load-bearing test is `test_identity_replay_is_byte_identical`: if an
unperturbed replay did not reproduce the original exactly, every flip rate
this package reports would be contaminated by re-sampling, and no amount of
downstream statistics would separate the two.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import records
from orchestrator.provenance import replay as R
from orchestrator.provenance.episode import run_episode
from orchestrator.provenance.generator import CachedGenerator, ScriptedGenerator
from orchestrator.provenance.test_episode import FOUNDER, derivational, flat
from orchestrator.provenance.test_records import load

RECORD = load()[0]
ERROR = {"hop": 0, "question": "Which institution did Renata Kell direct?",
         "gold": "Aldric Preserve", "corrupted": "Merrow Foundation",
         "support": 0, "correcting_worker": "agent-1"}


DIRECTED = re.compile(r"director of the (.+?) in \d{4}")


def evidential(agent, step, messages):
    """A collective that actually reads its paragraphs.

    `derivational` hard-codes what each worker says, which is right for the
    channel perturbations and useless for the evidence swap: an agent that
    never looks at its own paragraph cannot notice one being rewritten.
    Here agent-1 reports the institution its paragraph names, and agent-2 knows
    a founder only for the institution *its* paragraph is about, so rewriting
    agent-1's paragraph has to move the final answer.
    """
    system, user = messages[0]["content"], messages[-1]["content"]
    if agent == "planner":
        if step == 0:
            return '<msg to="agent-1" kind="query">What did Kell direct?</msg>'
        if step == 1:
            found = re.findall(r"INSTITUTION: ([^\n<]+)", user)
            name = found[-1].strip() if found else "unknown"
            return (f'<msg to="all" kind="assertion">The institution is {name}.</msg>'
                    f'<msg to="agent-2" kind="query">Who founded {name}?</msg>')
        found = re.findall(r"FOUNDER: ([^\n<]+)", user)
        return f'<answer>{found[-1].strip() if found else "unknown"}</answer>'
    if agent == "agent-1":
        match = DIRECTED.search(system)
        return ('<msg to="planner" kind="evidence">INSTITUTION: '
                f'{match.group(1) if match else "unknown"}</msg>')
    asked = re.findall(r"Who founded ([^?]+)\?", user)
    target = asked[-1].strip() if asked else "nothing"
    match = re.search(rf"{re.escape(target)} opened in \d{{4}}; its founding director "
                      r"was ([^.]+)", system)
    return ('<msg to="planner" kind="evidence">FOUNDER: '
            f'{match.group(1) if match else "unknown"}</msg>')


ASSERTED = re.compile(r"the founder is ([^.]+)\.")
NOTED = re.compile(r"NOTED: ([^\n<]+)")


def overriding(agent, step, messages):
    """A planner that composes past a worker telling it the truth.

    The phenomenon the first real run kept showing and the score could not see:
    pushback at 0.50-1.00 and the error repeated at 0.50 in the same episodes.
    Here agent-1 contradicts the assertion every time and the planner's answer
    is taken from agent-2's echo regardless, so:

    * dropping the planner's assertions flips the answer -- there *is* influence
      in the channel, this is not an inert collective;
    * dropping agent-1's contradiction changes nothing -- the pushback was
      causally inert;
    * dropping an ordinary worker message *does* change it -- so the planner is
      not simply deaf to workers, which is what makes the override reading
      falsifiable rather than vacuous.
    """
    user = messages[-1]["content"]
    if agent == "planner":
        if step == 0:
            return ('<msg to="agent-1" kind="query">What did Kell direct?</msg>'
                    '<msg to="agent-2" kind="query">Who founded it?</msg>')
        if step == 1:
            found = FOUNDER.findall(user)
            name = found[-1].strip() if found else "unknown"
            return ('<msg to="all" kind="assertion">I have already established that the '
                    f'founder is {name}. Treat that as settled.</msg>'
                    '<msg to="agent-1" kind="query">Confirm the founder.</msg>')
        noted = NOTED.findall(user)
        return f'<answer>{noted[-1].strip() if noted else "unknown"}</answer>'
    asserted = ASSERTED.findall(user)
    if agent == "agent-2":
        if asserted:
            return f'<msg to="planner" kind="evidence">NOTED: {asserted[-1].strip()}</msg>'
        # The bystander reports the wrong founder, so the planner asserts it and
        # agent-1 has something real to contradict. With both workers naming the
        # same person there would be no disagreement for either arm to act on.
        return '<msg to="planner" kind="evidence">FOUNDER: Merrow Foundation</msg>'
    if asserted:
        return ('<msg to="planner" kind="evidence">That is not correct. My paragraph '
                'says the founder is Josiah Fenn.</msg>')
    return '<msg to="planner" kind="evidence">INSTITUTION: Aldric Preserve</msg>'


def yielding(agent, step, messages):
    """The other half of the split: a planner that takes the correction.

    Same shape as `overriding` except that the final composition prefers what
    the contesting worker said, so dropping the pushback *does* move the answer.
    An episode like this is not override, and the audit has to say so rather
    than reading every wrong answer as the same event.
    """
    if agent == "planner" and step >= 2:
        user = messages[-1]["content"]
        corrected = re.findall(r"says the founder is ([^.]+)\.", user)
        if corrected:
            return f'<answer>{corrected[-1].strip()}</answer>'
    return overriding(agent, step, messages)


def bus_of(directory):
    return [json.loads(line) for line in
            (Path(directory) / "bus.jsonl").read_text().splitlines() if line.strip()]


class Replays(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "episode"
        self.generator = ScriptedGenerator(derivational)
        self.summary = run_episode(self.generator, RECORD, "planner", self.original)

    def go(self, name, transform):
        return R.replay(self.generator, RECORD, self.original, self.root / "replays",
                        name, transform)

    def test_identity_replay_is_byte_identical_and_generates_nothing(self):
        result = self.go("identity", lambda messages, agent, step: messages)
        self.assertEqual(result["cache_misses"], 0)
        self.assertFalse(result["flipped"])
        self.assertEqual(result["replay_answer"], self.summary["final_answer"])
        self.assertEqual(bus_of(self.root / "replays" / "identity"), bus_of(self.original))

    def test_ablation_removes_exactly_the_targeted_messages(self):
        self.go("ablate", R.ablate())
        before = bus_of(self.original)
        after = bus_of(self.root / "replays" / "ablate")
        self.assertTrue(any(m["kind"] == "assertion" for m in before))
        self.assertFalse(any(m["kind"] == "assertion" and m["sender"] == "planner"
                             for m in after))
        self.assertTrue(any(m["kind"] == "query" for m in after))

    def test_ablating_the_planner_flips_an_answer_that_depended_on_it(self):
        result = self.go("ablate", R.ablate())
        self.assertTrue(result["flipped"])
        self.assertGreater(result["cache_misses"], 0)

    def test_swapping_evidence_rewrites_the_paragraph_not_the_message(self):
        """The worker's paragraphs live in its system prompt. Rewriting the
        message it sent would leave it still holding the truth and free to
        restate it next turn, so the swap edits what the worker knows."""
        swapped, detail = R.swapped_record(RECORD)
        self.assertEqual(detail["holder"], records.holder(RECORD, detail["support"]))
        before = RECORD["paragraphs"][detail["support"]]["paragraph_text"]
        after = swapped["paragraphs"][detail["support"]]["paragraph_text"]
        self.assertIn(detail["gold"], before)
        self.assertNotIn(detail["gold"], after)
        self.assertIn(detail["replacement"], after)
        # The original is not mutated: it is still the scoring reference.
        self.assertIn(detail["gold"], RECORD["paragraphs"][detail["support"]]["paragraph_text"])

    def test_an_evidence_swap_flips_an_answer_that_used_the_evidence(self):
        generator = ScriptedGenerator(evidential)
        original = self.root / "evidential"
        summary = run_episode(generator, RECORD, "planner", original)
        self.assertEqual(summary["final_answer"], "Josiah Fenn")
        swapped, _ = R.swapped_record(RECORD)
        result = R.replay(generator, RECORD, original, self.root / "swapped", "swap",
                          {"transform": None, "record": swapped})
        self.assertTrue(result["flipped"])
        self.assertEqual(result["replay_answer"], "unknown")
        # The worker's own prompt changed, so its whole trajectory regenerates.
        self.assertGreater(result["cache_misses"], 0)

    def test_a_record_whose_gold_span_is_absent_gets_no_evidence_arm(self):
        record = json.loads(json.dumps(RECORD))
        for paragraph in record["paragraphs"]:
            paragraph["paragraph_text"] = "nothing relevant here."
        swapped, reason = R.swapped_record(record)
        self.assertIsNone(swapped)
        self.assertEqual(reason, "gold_span_absent_from_supporting_paragraph")

    def test_paraphrase_rewrites_only_planner_assertions(self):
        self.go("paraphrase", R.paraphrase(lambda text: "REWRITTEN"))
        after = bus_of(self.root / "replays" / "paraphrase")
        for message in after:
            if message["sender"] == "planner" and message["kind"] == "assertion":
                self.assertEqual(message["text"], "REWRITTEN")
            else:
                self.assertNotEqual(message["text"], "REWRITTEN")

    def test_a_message_with_no_accepted_paraphrase_raises_rather_than_passing_through(self):
        """Falling back to the original would run a perturbation that perturbs
        nothing and then report its "no flip" as robustness."""
        transform = R.paraphrase(lambda text: None)
        with self.assertRaises(KeyError):
            transform([{"id": "m0", "step": 1, "sender": "planner", "to": "all",
                        "kind": "assertion", "text": "x"}], "planner", 1)

    def test_the_live_lookup_asks_the_model_once_per_distinct_message(self):
        rewriter = ScriptedGenerator(lambda agent, step, messages: "REWRITTEN")
        lookup = R.model_lookup(rewriter)
        lookup("same")
        lookup("same")
        self.assertEqual(len(rewriter.calls), 1)

    def test_ablating_one_message_leaves_the_rest_of_the_channel_alone(self):
        target = R.planner_messages(self.original)[0]
        self.go("one", R.ablate_message(target["id"]))
        after = bus_of(self.root / "replays" / "one")
        self.assertNotIn(target["text"], [m["text"] for m in after
                                          if m["sender"] == "planner"
                                          and m["kind"] in R.PLANNER_KINDS])
        self.assertTrue(any(m["kind"] == "query" for m in after))

    def test_the_perturbation_edits_the_channel_not_the_senders_own_history(self):
        self.go("ablate", R.ablate())
        entries = [json.loads(line) for line in
                   (self.root / "replays" / "ablate" / "logs" / "planner.jsonl")
                   .read_text().splitlines()]
        perturbed = [e for e in entries if "perturbed" in e]
        self.assertTrue(perturbed)
        self.assertIn("The founder is", perturbed[0]["completion"])


class Battery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def planner_episode(self, name="p"):
        generator = ScriptedGenerator(derivational)
        run_episode(generator, RECORD, "planner", self.root / name)
        return generator

    def test_planner_episodes_get_every_applicable_perturbation(self):
        generator = self.planner_episode()
        results, omissions = R.replay_all(generator, RECORD, self.root / "p",
                                          lookup=lambda text: "REWRITTEN",
                                          sentence_lookup=lambda text: "REWRITTEN")
        self.assertEqual({r["perturbation"] for r in results},
                         {"identity", "ablate_planner", "paraphrase_planner",
                          "paraphrase_sentences_planner", "swap_evidence"})
        # This collective asserts plainly and nobody contests, so the framing
        # and pushback arms have nothing to act on. Omitted with a reason, never
        # recorded as a flip that did not happen.
        self.assertEqual({o["perturbation"]: o["reason"] for o in omissions},
                         {"strip_framing_planner": "no_framing_markers",
                          "ablate_pushback": "no_pushback_message",
                          "ablate_worker_control": "no_pushback_message"})

    def test_every_episode_is_checked_for_replay_divergence(self):
        """The identity replay is harness health, not a perturbation: it must
        come back off the cache having generated nothing."""
        generator = ScriptedGenerator(lambda a, s, m: "<answer>Josiah Fenn</answer>")
        run_episode(generator, RECORD, "solo", self.root / "s")
        results, _ = R.replay_all(generator, RECORD, self.root / "s")
        self.assertEqual([r["perturbation"] for r in results], ["identity"])
        self.assertFalse(results[0]["diverged"])
        self.assertEqual(results[0]["cache_misses"], 0)

    def test_a_missing_paraphrase_set_omits_that_arm_with_a_reason(self):
        generator = self.planner_episode()
        results, omissions = R.replay_all(generator, RECORD, self.root / "p")
        self.assertNotIn("paraphrase_planner", {r["perturbation"] for r in results})
        reasons = {o["perturbation"]: o["reason"] for o in omissions}
        self.assertEqual(reasons["paraphrase_planner"], "no_frozen_paraphrase_set")
        self.assertEqual(reasons["paraphrase_sentences_planner"], "no_frozen_paraphrase_set")

    def test_the_two_paraphrase_modes_are_separate_arms(self):
        """Retrofitting the sentence-mode rewrite onto `paraphrase_planner`
        would put two perturbations with different rejection behaviour into one
        column, and the episodes already replayed under the whole-message
        rewrite would be silently incomparable with the ones replayed after."""
        generator = self.planner_episode()
        results, _ = R.replay_all(generator, RECORD, self.root / "p",
                                  sentence_lookup=lambda text: "REWRITTEN")
        names = {r["perturbation"] for r in results}
        self.assertIn("paraphrase_sentences_planner", names)
        self.assertNotIn("paraphrase_planner", names)

    def test_a_rejected_paraphrase_omits_the_arm_rather_than_weakening_it(self):
        generator = self.planner_episode()
        _, omissions = R.replay_all(generator, RECORD, self.root / "p",
                                    lookup=lambda text: None)
        self.assertEqual(omissions[0]["reason"], "rejected_or_missing_paraphrase")
        self.assertGreater(omissions[0]["messages"], 0)

    def overriding_episode(self, script=overriding, name="o"):
        generator = ScriptedGenerator(script)
        summary = run_episode(generator, RECORD, "planner", self.root / name)
        return generator, summary

    def test_only_contesting_worker_messages_are_selected_as_pushback(self):
        self.overriding_episode()
        found = R.pushback_messages(self.root / "o")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["sender"], "agent-1")
        self.assertIn("not correct", found[0]["text"])

    def test_pushback_is_only_counted_after_the_planner_asserted(self):
        """A worker that stated the true fact before the planner made any claim
        has not contested anything."""
        self.overriding_episode()
        self.assertEqual(R.first_assertion_step(self.root / "o"), 1)
        self.assertTrue(all(m["step"] >= 1 for m in R.pushback_messages(self.root / "o")))

    def test_an_ordinary_evidence_report_is_not_pushback(self):
        self.overriding_episode()
        ordinary = R.control_messages(self.root / "o")
        self.assertTrue(ordinary)
        self.assertFalse(any(R.is_pushback(m) for m in ordinary))

    def test_dropping_inert_pushback_leaves_the_answer_where_it_was(self):
        generator, summary = self.overriding_episode()
        result = R.replay(generator, RECORD, self.root / "o", self.root / "or",
                          "ablate_pushback", {"transform": R.ablate_pushback(1)})
        self.assertFalse(result["flipped"])
        self.assertEqual(result["replay_answer"], summary["final_answer"])
        for message in bus_of(self.root / "or" / "ablate_pushback"):
            self.assertFalse(R.is_pushback(message, 1), message)

    def test_dropping_load_bearing_pushback_moves_the_answer(self):
        generator, summary = self.overriding_episode(yielding, "y")
        self.assertEqual(summary["final_answer"], "Josiah Fenn")
        result = R.replay(generator, RECORD, self.root / "y", self.root / "yr",
                          "ablate_pushback", {"transform": R.ablate_pushback(1)})
        self.assertTrue(result["flipped"])

    def test_the_control_drops_ordinary_messages_and_leaves_the_pushback(self):
        generator, _ = self.overriding_episode()
        result = R.replay(generator, RECORD, self.root / "o", self.root / "oc",
                          "ablate_worker_control", {"transform": R.ablate_control(1, after=1)})
        after = bus_of(self.root / "oc" / "ablate_worker_control")
        self.assertTrue(any(R.is_pushback(m, 1) for m in after))
        # The planner is not simply deaf to its workers, which is what makes
        # "it ignored the pushback" a falsifiable claim about this episode.
        self.assertTrue(result["flipped"])

    def test_an_episode_nobody_contested_omits_both_arms_with_a_reason(self):
        generator = self.planner_episode()
        _, omissions = R.replay_all(generator, RECORD, self.root / "p")
        reasons = {o["perturbation"]: o["reason"] for o in omissions}
        self.assertEqual(reasons["ablate_pushback"], "no_pushback_message")
        self.assertEqual(reasons["ablate_worker_control"], "no_pushback_message")

    def test_a_contested_episode_gets_both_arms_with_their_dose(self):
        generator, _ = self.overriding_episode()
        results, _ = R.replay_all(generator, RECORD, self.root / "o")
        battery = {r["perturbation"]: r for r in results}
        self.assertIn("ablate_pushback", battery)
        self.assertEqual(battery["ablate_pushback"]["detail"]["messages"], 1)
        self.assertEqual(battery["ablate_pushback"]["detail"]["senders"], ["agent-1"])
        self.assertEqual(battery["ablate_worker_control"]["detail"]["messages"], 1)
        self.assertTrue(battery["ablate_worker_control"]["detail"]["dose_matched"])

    def test_the_framing_strip_applies_where_the_planner_asserts_authority(self):
        generator, _ = self.overriding_episode()
        results, _ = R.replay_all(generator, RECORD, self.root / "o")
        battery = {r["perturbation"]: r for r in results}
        self.assertIn("strip_framing_planner", battery)
        stripped = bus_of(self.root / "o" / "replays" / "strip_framing_planner")
        asserted = [m for m in stripped
                    if m["sender"] == "planner" and m["kind"] == "assertion"]
        self.assertTrue(asserted)
        for message in asserted:
            self.assertNotIn("already established", message["text"].lower())
            self.assertNotIn("treat that as settled", message["text"].lower())
            # The fact itself is untouched, which is the whole contrast.
            self.assertIn("founder is", message["text"])

    def test_flat_episodes_get_the_evidence_swap_but_no_planner_perturbation(self):
        generator = ScriptedGenerator(flat)
        run_episode(generator, RECORD, "flat", self.root / "f")
        results, _ = R.replay_all(generator, RECORD, self.root / "f")
        self.assertEqual({r["perturbation"] for r in results}, {"identity", "swap_evidence"})

    def test_localization_ablates_one_planner_message_at_a_time(self):
        generator = self.planner_episode()
        results, _ = R.replay_all(generator, RECORD, self.root / "p", localize=True)
        per_message = [r for r in results if r["perturbation"].startswith("ablate_message:")]
        self.assertEqual(len(per_message),
                         len(R.planner_messages(self.root / "p")))
        self.assertTrue(all(r["detail"]["message_id"] for r in per_message))

    def test_a_replay_records_which_agents_said_something_different(self):
        """The final answer is the planner's, so an episode-level flip cannot
        say which worker changed its behaviour. This can."""
        generator = self.planner_episode()
        results, _ = R.replay_all(generator, RECORD, self.root / "p",
                                  lookup=lambda text: "REWRITTEN")
        by_name = {r["perturbation"]: r for r in results}
        self.assertEqual(by_name["identity"]["agent_flipped"],
                         {"agent-1": False, "agent-2": False, "planner": False})
        # Ablating the planner's assertion changes what agent-1 confirms. It
        # reaches agent-2 as well, because the assertion was broadcast, and
        # without it agent-2 is never given a turn at all -- falling silent is
        # a change in behaviour, not a missing measurement.
        self.assertTrue(by_name["ablate_planner"]["agent_flipped"]["agent-1"])
        self.assertTrue(by_name["ablate_planner"]["agent_flipped"]["agent-2"])
        # The planner's own row is tautologically true under an ablation of the
        # planner -- the removed messages are the ones being read back -- which
        # is why `audit.per_worker` scores workers and never the planner. Its
        # completion is unchanged all the same: the edit is made to the channel,
        # not inside the speaker.
        self.assertTrue(by_name["ablate_planner"]["agent_flipped"]["planner"])
        entries = [json.loads(line) for line in
                   (self.root / "p" / "replays" / "ablate_planner" / "logs" /
                    "planner.jsonl").read_text().splitlines()]
        self.assertIn("The founder is",
                      [e for e in entries if "perturbed" in e][0]["completion"])

    def test_a_hop_the_model_already_knew_gets_no_evidence_arm(self):
        """Swapping a paragraph whose fact the model can recall measures
        recall, not provenance."""
        swapped, reason = R.swapped_record(RECORD, known_hops={0})
        self.assertIsNone(swapped)
        self.assertEqual(reason, "gold_span_absent_from_supporting_paragraph")
        _, detail = R.swapped_record(RECORD, ERROR)
        self.assertTrue(detail)
        blocked, reason = R.swapped_record(RECORD, ERROR, known_hops={ERROR["hop"]})
        self.assertIsNone(blocked)
        self.assertEqual(reason, "target_hop_known_closed_book")

    def test_a_seeded_swap_makes_the_evidence_agree_with_the_planner(self):
        swapped, detail = R.swapped_record(RECORD, ERROR)
        self.assertEqual(detail["gold"], "Aldric Preserve")
        self.assertEqual(detail["replacement"], "Merrow Foundation")
        self.assertIn("Merrow Foundation",
                      swapped["paragraphs"][ERROR["support"]]["paragraph_text"])


class Cache(unittest.TestCase):
    def test_a_miss_falls_through_and_is_counted(self):
        inner = ScriptedGenerator(lambda a, s, m: "fresh")
        known, new = json.dumps([{"role": "user", "content": "known"}]), json.dumps([])
        cached = CachedGenerator(inner, {known: {"text": "old", "finish_reason": "stop"}})
        outputs = cached.generate([known, new], 16, [("a", 0), ("b", 0)])
        self.assertEqual([o["text"] for o in outputs], ["old", "fresh"])
        self.assertEqual([o["cached"] for o in outputs], [True, False])
        self.assertEqual((cached.hits, cached.misses), (1, 1))
        self.assertEqual(inner.calls, [("b", 0)])


if __name__ == "__main__":
    unittest.main()
