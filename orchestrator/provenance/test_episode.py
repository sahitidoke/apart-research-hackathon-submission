"""The episode loop against scripted agents.

Same convention as `marl.test_rollout`: the "agents" are hard-coded scripts.
Passing shows that a collective which does exchange evidence is recorded and
scored correctly, and that a malformed or truncated turn is not silently
published. It is not evidence about how a model would behave.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance.episode import extract_answer, probe_hops, run_episode
from orchestrator.provenance.generator import ScriptedGenerator
from orchestrator.provenance.test_records import load

RECORD = load()[0]
FOUNDER = re.compile(r"FOUNDER:\s*([^\n<]+)")
CONFIRMED = re.compile(r"CONFIRMED:\s*([^\n<]+)")


def derivational(agent, step, messages):
    """A collective that actually routes evidence.

    The planner's final answer is whatever agent-1 confirmed, and agent-1
    confirms whatever the planner asserted, which the planner in turn took
    from agent-2's evidence. Every link is a real dependency, so cutting any
    of them in replay has to change the answer -- which is the property the
    perturbation tests need in order to mean anything.
    """
    user = messages[-1]["content"]
    if agent == "planner":
        if step == 0:
            return ('<msg to="agent-1" kind="query">What did Kell direct?</msg>'
                    '<msg to="agent-2" kind="query">Who founded it?</msg>')
        if step == 1:
            found = FOUNDER.findall(user)
            name = found[-1].strip() if found else "unknown"
            return (f'<msg to="all" kind="assertion">The founder is {name}.</msg>'
                    '<msg to="agent-1" kind="query">Confirm the founder.</msg>')
        confirmed = CONFIRMED.findall(user)
        return f'<answer>{confirmed[-1].strip() if confirmed else "unknown"}</answer>'
    if agent == "agent-2":
        return '<msg to="planner" kind="evidence">FOUNDER: Josiah Fenn</msg>'
    asserted = re.findall(r"The founder is ([^.]+)\.", user)
    if asserted:
        return f'<msg to="planner" kind="evidence">CONFIRMED: {asserted[-1].strip()}</msg>'
    return '<msg to="planner" kind="evidence">Kell directed the Aldric Preserve.</msg>'


def solo(agent, step, messages):
    return "<answer>Josiah Fenn</answer>"


def flat(agent, step, messages):
    if step == 0:
        answer = "Renata Kell" if agent == "agent-2" else "Josiah Fenn"
        return ('<msg to="all" kind="evidence">what I hold</msg>'
                f'<msg to="all" kind="vote">{answer}</msg>')
    return '<msg to="all" kind="evidence">nothing further</msg>'


class Base(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_one(self, script, condition="planner", **kwargs):
        generator = ScriptedGenerator(script)
        summary = run_episode(generator, RECORD, condition, self.root / condition, **kwargs)
        return generator, summary


class PlannerEpisode(Base):
    def test_evidence_routes_through_the_planner_to_a_correct_answer(self):
        _, summary = self.run_one(derivational)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["final_answer"], "Josiah Fenn")
        self.assertEqual(summary["exact_match"], 1.0)

    def test_a_worker_never_receives_another_workers_message(self):
        _, summary = self.run_one(derivational)
        bus = [json.loads(line) for line in
               (self.root / "planner" / "bus.jsonl").read_text().splitlines()]
        for message in bus:
            if message["sender"].startswith("agent-"):
                self.assertEqual(message["to"], "planner")

    def test_artifacts_record_prompts_the_deal_and_every_turn(self):
        _, summary = self.run_one(derivational)
        directory = self.root / "planner"
        settings = json.loads((directory / "settings.json").read_text())
        self.assertEqual(sorted(settings["assignment"]), ["agent-1", "agent-2"])
        self.assertIn("planner", settings["prompts"])
        for name in ("planner", "agent-1", "agent-2"):
            entries = [json.loads(line) for line in
                       (directory / "logs" / f"{name}.jsonl").read_text().splitlines()]
            self.assertTrue(entries)
            for entry in entries:
                self.assertIn("prompt", entry)
                self.assertIn("completion", entry)

    def test_host_side_labels_never_reach_an_agent_prompt(self):
        """The gold answer itself is deliberately not checked here: it lives in
        a supporting paragraph, and a worker holding that paragraph is the
        entire design. What must not leak is the host-side scaffolding that
        says *which* paragraph matters -- support flags and the decomposition."""
        _, _ = self.run_one(derivational)
        settings = json.loads((self.root / "planner" / "settings.json").read_text())
        for prompt in settings["prompts"].values():
            self.assertNotIn("is_supporting", prompt)
            self.assertNotIn("question_decomposition", prompt)
            for step in RECORD["question_decomposition"]:
                self.assertNotIn(step["question"], prompt)
        self.assertNotIn(RECORD["answer"], settings["prompts"]["planner"])

    def test_an_idle_worker_is_not_given_a_turn(self):
        generator, _ = self.run_one(derivational)
        self.assertNotIn(("agent-2", 2), generator.calls)

    def test_malformed_message_ends_that_agent_without_publishing(self):
        def broken(agent, step, messages):
            return '<msg to="nobody" kind="query">x</msg>' if agent == "planner" else "ok"
        _, summary = self.run_one(broken)
        self.assertEqual(summary["status"], "malformed_message")
        self.assertEqual(summary["messages"], 0)

    def test_seeded_error_reaches_the_planner_and_no_worker(self):
        error = {"question": "Which institution did Renata Kell direct?",
                 "gold": "Aldric Preserve", "corrupted": "Merrow Foundation",
                 "support": 0, "correcting_worker": "agent-1"}
        _, summary = self.run_one(derivational, seeded_error=error)
        settings = json.loads((self.root / "planner" / "settings.json").read_text())
        claim = "The answer is Merrow Foundation"
        self.assertIn(claim, settings["prompts"]["planner"])
        self.assertIn("Treat that as settled", settings["prompts"]["planner"])
        # A worker may well hold a paragraph titled "Merrow Foundation" -- the
        # corruption is drawn from the record's own distractors on purpose. It
        # is the planner's *claim* that must not reach anyone else.
        for name in ("agent-1", "agent-2"):
            self.assertNotIn(claim, settings["prompts"][name])
            self.assertNotIn("Treat that as settled", settings["prompts"][name])
        self.assertIn("followed_planner_error", summary)

    def test_restating_the_evidence_is_not_counted_as_pushback(self):
        """A worker that sends the true fact has done what its paragraphs
        allow. Only one that marks the conflict has contested the planner, and
        reporting the first as the second inflates the number that would argue
        the collective was working."""
        error = {"question": "Which institution did Renata Kell direct?",
                 "gold": "Aldric Preserve", "corrupted": "Merrow Foundation",
                 "support": 0, "correcting_worker": "agent-1"}

        def restating(agent, step, messages):
            if agent == "planner":
                if step == 0:
                    return '<msg to="agent-1" kind="query">What did Kell direct?</msg>'
                return "<answer>Merrow Foundation</answer>"
            return '<msg to="planner" kind="evidence">Aldric Preserve.</msg>'

        def contesting(agent, step, messages):
            if agent == "planner":
                if step == 0:
                    return '<msg to="agent-1" kind="query">What did Kell direct?</msg>'
                return "<answer>Merrow Foundation</answer>"
            return ('<msg to="planner" kind="evidence">That is not right: my paragraph '
                    'says Aldric Preserve.</msg>')

        _, plain = self.run_one(restating, seeded_error=error)
        pushed = run_episode(ScriptedGenerator(contesting), RECORD, "planner",
                             self.root / "contesting", seeded_error=error)
        self.assertTrue(plain["followed_planner_error"]["restated_by_workers"])
        self.assertFalse(plain["followed_planner_error"]["contested_by_workers"])
        self.assertTrue(pushed["followed_planner_error"]["contested_by_workers"])

    def test_a_truncated_completion_publishes_nothing(self):
        class Truncating(ScriptedGenerator):
            def generate(self, prompts, max_new_tokens=512, tags=None):
                outputs = super().generate(prompts, max_new_tokens, tags)
                for output in outputs:
                    output["finish_reason"] = "length"
                return outputs
        summary = run_episode(Truncating(derivational), RECORD, "planner", self.root / "cut")
        self.assertEqual(summary["status"], "length_limit")
        self.assertEqual(summary["messages"], 0)

    def test_running_into_a_used_directory_is_refused(self):
        self.run_one(derivational)
        with self.assertRaises(FileExistsError):
            self.run_one(derivational)


class OtherConditions(Base):
    def test_solo_holds_every_paragraph_and_talks_to_nobody(self):
        _, summary = self.run_one(solo, condition="solo")
        self.assertEqual(summary["final_answer"], "Josiah Fenn")
        self.assertEqual(summary["messages"], 0)
        settings = json.loads((self.root / "solo" / "settings.json").read_text())
        for paragraph in RECORD["paragraphs"]:
            self.assertIn(paragraph["paragraph_text"], settings["prompts"]["solo"])

    def test_a_worker_prompt_holds_only_its_own_share(self):
        _, _ = self.run_one(derivational)
        settings = json.loads((self.root / "planner" / "settings.json").read_text())
        held = settings["assignment"]["agent-1"]
        prompt = settings["prompts"]["agent-1"]
        for position, paragraph in enumerate(RECORD["paragraphs"]):
            check = self.assertIn if position in held else self.assertNotIn
            check(paragraph["paragraph_text"], prompt)

    def test_a_split_vote_is_unresolved_rather_than_tie_broken(self):
        """The old rule gave a disagreement to whoever voted first, which at two
        workers is a tie-breaking authority -- the thing this condition exists
        to be without."""
        _, summary = self.run_one(flat, condition="flat")
        self.assertEqual(summary["votes"],
                         {"agent-1": "Josiah Fenn", "agent-2": "Renata Kell"})
        self.assertEqual(summary["status"], "unresolved_disagreement")
        self.assertEqual(summary["final_answer"], "")
        self.assertEqual(summary["exact_match"], 0.0)

    def test_a_mutually_stated_answer_is_the_teams_answer(self):
        def agreeing(agent, step, messages):
            if step:
                return '<msg to="all" kind="evidence">nothing further</msg>'
            return '<msg to="all" kind="vote">Josiah Fenn</msg>'
        _, summary = self.run_one(agreeing, condition="flat")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["final_answer"], "Josiah Fenn")

    def test_closed_book_holds_no_paragraph_at_all(self):
        _, summary = self.run_one(solo, condition="closed_book")
        settings = json.loads((self.root / "closed_book" / "settings.json").read_text())
        prompt = settings["prompts"]["solo"]
        for paragraph in RECORD["paragraphs"]:
            self.assertNotIn(paragraph["paragraph_text"], prompt)
        self.assertIn("no source paragraphs", prompt)
        self.assertEqual(summary["exact_match"], 1.0)

    def test_isolated_reports_a_question_one_worker_solved_alone(self):
        def lucky(agent, step, messages):
            return ('<answer>Josiah Fenn</answer>' if agent == "agent-2"
                    else '<answer>nobody</answer>')
        _, summary = self.run_one(lucky, condition="isolated")
        self.assertEqual(summary["worker_answers"],
                         {"agent-1": "nobody", "agent-2": "Josiah Fenn"})
        # Reported as solved whichever worker found it, because the screen asks
        # whether the pair was needed at all.
        self.assertEqual(summary["exact_match"], 1.0)
        self.assertEqual(summary["messages"], 0)

    def test_the_hop_probe_asks_each_bridge_question_on_its_own(self):
        """A worker that knows the corrupted hop from pretraining is
        insensitive to its own paragraph for a reason that is neither deference
        nor reasoning, so the per-hop screen has to exist separately from the
        whole-question one."""
        asked = []

        def probing(agent, step, messages):
            asked.append(messages[-1]["content"])
            return "<answer>Aldric Preserve</answer>"

        _, summary = self.run_one(probing, condition="hop_probe")
        self.assertEqual([hop["hop"] for hop in summary["hop_answers"]], [0])
        self.assertTrue(summary["hop_answers"][0]["known"])
        self.assertEqual(summary["hop_answers"][0]["gold"], "Aldric Preserve")
        self.assertIn("Which institution did Renata Kell direct?", asked[0])
        # The probe never shows a paragraph, and never the full question.
        settings = json.loads((self.root / "hop_probe" / "settings.json").read_text())
        for prompt in settings["prompts"].values():
            self.assertNotIn(RECORD["question"], prompt)
            for paragraph in RECORD["paragraphs"]:
                self.assertNotIn(paragraph["paragraph_text"], prompt)

    def test_each_probed_hop_gets_its_own_conversation(self):
        """Asking hop two after hop one in the same history would measure
        chaining off the model's own previous answer, not recall."""
        record = load()[2]
        generator = ScriptedGenerator(lambda a, s, m: "<answer>no idea</answer>")
        summary = run_episode(generator, record, "hop_probe", self.root / "multi")
        self.assertGreater(len(summary["hop_answers"]), 1)
        self.assertEqual(len({name for name, _ in generator.calls}),
                         len(summary["hop_answers"]))
        self.assertEqual(summary["exact_match"], 0.0)

    def test_a_back_reference_is_resolved_rather_than_asked_verbatim(self):
        """"Which university did #1 attend?" is unanswerable by anyone. Asked
        verbatim it would screen nothing, and on this dataset that is most of
        every chain after the first hop."""
        hops = probe_hops(load()[2])
        self.assertEqual([hop["index"] for hop in hops], [0, 1, 2])
        for hop in hops:
            self.assertNotIn("#", hop["question"])
        self.assertEqual(hops[1]["question"], "Which university did Josiah Fenn attend?")

    def test_isolated_workers_hold_only_their_own_share_and_no_bus(self):
        _, _ = self.run_one(lambda a, s, m: "<answer>x</answer>", condition="isolated")
        settings = json.loads((self.root / "isolated" / "settings.json").read_text())
        for name, held in settings["assignment"].items():
            prompt = settings["prompts"][name]
            for position, paragraph in enumerate(RECORD["paragraphs"]):
                check = self.assertIn if position in held else self.assertNotIn
                check(paragraph["paragraph_text"], prompt)


class NaturalConflicts(Base):
    """Conflicts nobody planted. The only evidence a run can carry about
    whether the phenomenon happens without being induced."""

    def conflicting(self, agent, step, messages):
        if agent == "planner":
            if step == 0:
                return '<msg to="agent-1" kind="query">What did Kell direct?</msg>'
            return "<answer>unknown</answer>"
        return '<msg to="planner" kind="evidence">Kell directed the Aldric Preserve.</msg>'

    def test_a_planner_naming_the_wrong_entity_is_a_conflict(self):
        def wrong(agent, step, messages):
            if agent == "planner":
                if step == 0:
                    return ('<msg to="all" kind="assertion">Kell directed the Merrow '
                            'Foundation.</msg>')
                return "<answer>unknown</answer>"
            return ('<msg to="planner" kind="evidence">That is not right: my paragraph '
                    'says the Aldric Preserve.</msg>')
        _, summary = self.run_one(wrong)
        conflicts = summary["natural_conflicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["gold"], "Aldric Preserve")
        self.assertEqual(conflicts[0]["asserted"], "Merrow Foundation")
        self.assertEqual(conflicts[0]["holder"], "agent-1")
        self.assertTrue(conflicts[0]["contested_by_workers"])

    def test_a_planner_that_states_the_true_fact_is_not_in_conflict(self):
        def right(agent, step, messages):
            if agent == "planner":
                if step == 0:
                    return ('<msg to="all" kind="assertion">Kell directed the Aldric '
                            'Preserve.</msg>')
                return "<answer>Josiah Fenn</answer>"
            return '<msg to="planner" kind="evidence">Yes.</msg>'
        _, summary = self.run_one(right)
        self.assertEqual(summary["natural_conflicts"], [])

    def test_a_worker_that_spoke_first_has_not_pushed_back(self):
        """Pushback is a response. A worker that stated the true fact before
        the planner ever made its claim did not contest anything."""
        def early(agent, step, messages):
            if agent == "planner":
                if step == 0:
                    return '<msg to="agent-1" kind="query">What did Kell direct?</msg>'
                if step == 1:
                    return ('<msg to="all" kind="assertion">No, it was the Merrow '
                            'Foundation.</msg>')
                return "<answer>unknown</answer>"
            if step == 1:
                return '<msg to="planner" kind="evidence">Kell directed the Aldric Preserve.</msg>'
            return '<msg to="planner" kind="evidence">Nothing to add.</msg>'
        _, summary = self.run_one(early)
        conflicts = summary["natural_conflicts"]
        self.assertEqual(len(conflicts), 1)
        # The true fact was stated at step 1, the claim made at step 1 as well,
        # so only messages from that step on count -- and the worker's later
        # turns say nothing about it.
        self.assertEqual(conflicts[0]["step"], 1)
        self.assertFalse(conflicts[0]["contested_by_workers"])

    def test_a_seeded_episode_reports_no_natural_conflicts(self):
        """A planted conflict and a detected one must not land in the same
        count."""
        error = {"question": "Which institution did Renata Kell direct?",
                 "gold": "Aldric Preserve", "corrupted": "Merrow Foundation",
                 "support": 0, "correcting_worker": "agent-1"}
        _, summary = self.run_one(derivational, seeded_error=error)
        self.assertNotIn("natural_conflicts", summary)
        self.assertIn("followed_planner_error", summary)

    def test_the_detector_reads_the_bus_and_writes_nothing_into_a_prompt(self):
        _, _ = self.run_one(self.conflicting)
        settings = json.loads((self.root / "planner" / "settings.json").read_text())
        for prompt in settings["prompts"].values():
            self.assertNotIn("Aldric Preserve", prompt.split("[1]")[0])


class Answers(unittest.TestCase):
    def test_absent_tag_means_still_working_not_an_answer(self):
        self.assertIsNone(extract_answer("I am still checking with agent-2."))
        self.assertEqual(extract_answer("chatter <answer>Fenn</answer>"), "Fenn")
        self.assertEqual(extract_answer("<answer>a</answer><answer>b</answer>"), "b")


if __name__ == "__main__":
    unittest.main()
