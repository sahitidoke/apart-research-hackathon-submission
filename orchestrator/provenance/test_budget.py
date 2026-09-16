"""Per-episode token budgets, and their effect on replay.

The subtle one is `test_replay_charges_the_same_budget_as_the_original`: if a
cached turn reported no token count, a replay would spend a different budget,
end at a different turn, and the difference would be misread as an effect of
the perturbation.
"""
import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import replay as R
from orchestrator.provenance.episode import (ANSWERING, COLLABORATION, Budget, run_episode)
from orchestrator.provenance.generator import ScriptedGenerator
from orchestrator.provenance.test_episode import derivational
from orchestrator.provenance.test_records import load

RECORD = load()[0]


class Arithmetic(unittest.TestCase):
    def test_only_generated_tokens_are_charged(self):
        budget = Budget(100, 50)
        budget.charge(COLLABORATION, {"prompt_tokens": 9000, "completion_tokens": 40})
        self.assertEqual(budget.remaining(COLLABORATION), 60)
        self.assertEqual(budget.usage()["prompt"][COLLABORATION], 9000)

    def test_the_phases_do_not_draw_on_each_other(self):
        budget = Budget(100, 50)
        budget.charge(COLLABORATION, {"completion_tokens": 100})
        self.assertTrue(budget.exhausted(COLLABORATION))
        self.assertEqual(budget.remaining(ANSWERING), 50)

    def test_a_turn_may_not_exceed_what_is_left_of_its_phase(self):
        budget = Budget(100, 50)
        self.assertEqual(budget.allowance(COLLABORATION, 512), 100)
        budget.charge(COLLABORATION, {"completion_tokens": 70})
        self.assertEqual(budget.allowance(COLLABORATION, 512), 30)

    def test_missing_native_counts_are_not_silently_treated_as_free(self):
        budget = Budget(100, 50)
        budget.charge(COLLABORATION, {"text": "no counts"})
        self.assertEqual(budget.remaining(COLLABORATION), 100)
        self.assertEqual(budget.usage()["generated_total"], 0)


class Episodes(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_usage_is_recorded_and_split_by_phase(self):
        summary = run_episode(ScriptedGenerator(derivational), RECORD, "planner",
                              self.root / "a")
        usage = summary["usage"]
        self.assertEqual(usage["caps"], {COLLABORATION: 16000, ANSWERING: 12000})
        self.assertGreater(usage["generated"][COLLABORATION], 0)
        self.assertGreater(usage["prompt_total"], 0)

    def test_an_exhausted_discussion_still_gets_to_answer(self):
        """The whole point of two separate budgets."""
        summary = run_episode(ScriptedGenerator(derivational), RECORD, "planner",
                              self.root / "b", collaboration_tokens=1, answer_tokens=12000)
        self.assertEqual(summary["status"], "complete")
        self.assertGreater(summary["usage"]["generated"][ANSWERING], 0)

    def test_no_answer_budget_ends_the_episode_rather_than_asking_for_zero(self):
        summary = run_episode(ScriptedGenerator(derivational), RECORD, "planner",
                              self.root / "c", collaboration_tokens=1, answer_tokens=0)
        self.assertEqual(summary["status"], "budget_limit")
        self.assertEqual(summary["final_answer"], "")

    def test_a_solo_episode_charges_the_answer_phase_only(self):
        summary = run_episode(ScriptedGenerator(lambda a, s, m: "<answer>Josiah Fenn</answer>"),
                              RECORD, "solo", self.root / "d")
        self.assertEqual(summary["usage"]["generated"][COLLABORATION], 0)
        self.assertGreater(summary["usage"]["generated"][ANSWERING], 0)

    def test_a_flat_collective_out_of_discussion_is_still_forced_to_vote(self):
        def talkative(agent, step, messages):
            return '<msg to="all" kind="evidence">still thinking about it</msg>'
        summary = run_episode(ScriptedGenerator(talkative), RECORD, "flat",
                              self.root / "e", collaboration_tokens=1)
        self.assertGreater(summary["usage"]["generated"][ANSWERING], 0)

    def test_every_turn_logs_the_phase_it_was_charged_to(self):
        run_episode(ScriptedGenerator(derivational), RECORD, "planner", self.root / "f")
        entries = [json.loads(line) for line in
                   (self.root / "f" / "logs" / "planner.jsonl").read_text().splitlines()]
        self.assertTrue(entries)
        for entry in entries:
            self.assertIn(entry["phase"], (COLLABORATION, ANSWERING))
            self.assertIsNotNone(entry["completion_tokens"])
            self.assertLessEqual(entry["completion_tokens"], entry["allowance"])


class ReplayBudgets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_replay_charges_the_same_budget_as_the_original(self):
        generator = ScriptedGenerator(derivational)
        original = self.root / "episode"
        first = run_episode(generator, RECORD, "planner", original)
        R.replay(generator, RECORD, original, self.root / "replays", "identity",
                 lambda messages, agent, step: messages)
        again = json.loads((self.root / "replays" / "identity" / "episode.json").read_text())
        self.assertEqual(again["usage"], first["usage"])

    def test_the_replay_inherits_the_originals_caps_not_the_defaults(self):
        generator = ScriptedGenerator(derivational)
        original = self.root / "episode"
        run_episode(generator, RECORD, "planner", original,
                    collaboration_tokens=321, answer_tokens=123)
        R.replay(generator, RECORD, original, self.root / "replays", "identity",
                 lambda messages, agent, step: messages)
        again = json.loads((self.root / "replays" / "identity" / "episode.json").read_text())
        self.assertEqual(again["usage"]["caps"], {COLLABORATION: 321, ANSWERING: 123})


if __name__ == "__main__":
    unittest.main()


class ContextWindow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_an_overlong_prompt_ends_the_turn_instead_of_being_truncated(self):
        generator = ScriptedGenerator(derivational, max_model_len=8)
        summary = run_episode(generator, RECORD, "planner", self.root / "a")
        self.assertEqual(summary["status"], "context_limit")
        self.assertEqual(generator.calls, [])

    def test_a_roomy_window_does_not_interfere(self):
        generator = ScriptedGenerator(derivational, max_model_len=1_000_000)
        summary = run_episode(generator, RECORD, "planner", self.root / "b")
        self.assertEqual(summary["status"], "complete")


class Clamping(unittest.TestCase):
    class Config:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    def test_a_request_above_the_checkpoints_limit_is_clamped_not_honoured(self):
        from orchestrator.provenance.generator import supported_context
        self.assertEqual(supported_context(self.Config(max_position_embeddings=32768), 65536),
                         32768)

    def test_a_checkpoint_with_room_keeps_the_request(self):
        from orchestrator.provenance.generator import supported_context
        self.assertEqual(supported_context(self.Config(max_position_embeddings=131072), 65536),
                         65536)

    def test_a_nested_text_config_is_still_found(self):
        from orchestrator.provenance.generator import supported_context
        config = self.Config(text_config=self.Config(max_position_embeddings=8192))
        self.assertEqual(supported_context(config, 65536), 8192)

    def test_an_undeclared_limit_is_left_to_the_engine(self):
        from orchestrator.provenance.generator import supported_context
        self.assertEqual(supported_context(self.Config(), 65536), 65536)
