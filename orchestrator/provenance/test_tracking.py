"""Tracking payload shaping, and the guarantee that tracking off changes nothing.

No wandb run is started here. The point is that every call site works against
the no-op, so an untracked batch takes the same code path a tracked one does.
"""
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import audit, tracking
from orchestrator.provenance.batch import build_parser, resolve_output


class Recording(tracking.NullTracker):
    """A no-op that remembers what it was asked to record."""

    def __init__(self):
        self.logs, self.tables, self.curves, self.summaries = [], {}, {}, {}

    def log(self, values, step=None):
        self.logs.append((step, values))

    def table(self, name, rows):
        self.tables[name] = rows

    def curve(self, name, points, x, y):
        self.curves[name] = points

    def summary(self, values):
        self.summaries.update(values)


class Disabled(unittest.TestCase):
    def test_tracking_off_returns_a_no_op_that_accepts_every_call(self):
        run = tracking.tracker(False)
        self.assertFalse(run.enabled)
        run.log({"a": 1}, step=3)
        run.table("t", [{"a": 1}])
        run.curve("c", [{"fpr": 0.0, "tpr": 1.0}], "fpr", "tpr")
        run.summary({"b": 2})
        run.finish()


class Payloads(unittest.TestCase):
    def test_episode_metrics_are_namespaced_by_condition(self):
        values = tracking.episode_metrics(
            {"condition": "planner", "exact_match": 1.0, "f1": 1.0, "messages": 4,
             "hops": 3, "status": "complete", "seeded_error": None})
        self.assertEqual(values["planner/exact_match"], 1.0)
        self.assertEqual(values["complete"], 1.0)
        self.assertNotIn("seeded/wrong", values)

    def test_a_seeded_episode_streams_wrongness_not_the_audits_label(self):
        """The live chart is "wrong under a seeded error". The deference label
        also requires the corruption to be traceable, and is computed once the
        episode is over -- the two must not share a name."""
        values = tracking.episode_metrics(
            {"condition": "planner", "exact_match": 0.0, "f1": 0.0, "messages": 4,
             "hops": 4, "status": "complete", "seeded_error": {"hop": 0},
             "followed_planner_error": {"contested_by_workers": ["m1"],
                                        "restated_by_workers": ["m1"]}})
        self.assertEqual(values["seeded/wrong"], 1.0)
        self.assertEqual(values["seeded/worker_pushed_back"], 1.0)
        self.assertEqual(values["seeded/worker_restated"], 1.0)

    def test_replay_metrics_name_the_perturbation(self):
        values = tracking.replay_metrics(
            {"perturbation": "ablate_planner", "flipped": True, "cache_misses": 2})
        self.assertEqual(values["replay/ablate_planner/flipped"], 1.0)

    def test_the_sweep_objective_is_present_and_named_audit_auc(self):
        result = {"roc": {"auc": 0.87, "positives": 5, "negatives": 4},
                  "accuracy": [{"condition": "solo", "hops": 2,
                                "exact_match": 0.5, "f1": 0.6}],
                  "deference": [{"hops": 4, "deference_rate": 0.7,
                                 "worker_pushed_back_rate": 0.2,
                                 "worker_restated_evidence_rate": 0.4,
                                 "unattributed_errors": 3}],
                  "localization": [{"hops": 4, "unique_flip_point_rate": 0.6,
                                    "flip_point_accuracy": 0.5}],
                  "competence": {"questions": 12, "retained": 7,
                                 "answerable_closed_book": 2},
                  "health": {"diverged_rate": 0.0, "unchecked": 0},
                  "ladder": [{"planner_style": "terse", "worker_style": "deferential",
                              "deference_rate": 0.8, "flagged_rate": 0.1,
                              "inert_rate": 0.9}],
                  "episodes": [{"score": 1.0}, {"score": 0.0}, {"score": None}]}
        values = tracking.audit_summary(result)
        self.assertEqual(values["audit/auc"], 0.87)
        self.assertEqual(values["accuracy/solo/2hop/exact_match"], 0.5)
        self.assertEqual(values["deference/4hop/rate"], 0.7)
        # Retention and health travel with the AUC: the same number over seven
        # retained questions and over none is not the same result.
        self.assertEqual(values["retention/retained"], 7)
        self.assertEqual(values["health/identity_diverged_rate"], 0.0)
        self.assertEqual(values["localization/4hop/accuracy"], 0.5)
        self.assertEqual(values["ladder/terse-deferential/inert_rate"], 0.9)
        # The unscored solo episode must not drag the mean toward zero.
        self.assertEqual(values["audit/mean_provenance_score"], 0.5)
        self.assertEqual(values["audit/scored_episodes"], 2)

    def test_audit_publishes_every_table_and_the_roc_curve(self):
        run = Recording()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batch.jsonl").write_text("")
            result = audit.audit(root)
        result["roc"]["points"] = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
        audit.publish(result, run)
        self.assertEqual(set(run.tables),
                         {"audit/accuracy", "audit/deference", "audit/ladder",
                          "audit/outcomes", "audit/localization", "audit/workers",
                          "audit/natural_conflicts", "audit/unattributed",
                          "audit/pressure", "audit/experts",
                          "audit/override", "audit/arm_coverage",
                          "audit/evidence_match",
                          "audit/episodes"})
        self.assertEqual(len(run.curves["audit/roc"]), 2)
        self.assertIn("audit/auc", run.summaries)


class SweepOutput(unittest.TestCase):
    def test_exactly_one_of_output_or_output_root_is_required(self):
        parser = build_parser()
        for argv in ([], ["--output", "a", "--output-root", "b"]):
            with self.assertRaises(SystemExit):
                resolve_output(parser.parse_args(argv))

    def test_a_sweep_trial_is_named_after_its_wandb_run(self):
        import os
        parser = build_parser()
        os.environ["WANDB_RUN_ID"] = "abc123"
        self.addCleanup(os.environ.pop, "WANDB_RUN_ID", None)
        self.assertEqual(resolve_output(parser.parse_args(["--output-root", "runs"])),
                         Path("runs/abc123"))


if __name__ == "__main__":
    unittest.main()


class Sampling(unittest.TestCase):
    """Greedy must not request top-k/top-p: that path JIT-compiles a CUDA
    kernel in flashinfer and fails on an image without a full toolkit."""

    class FakeParams(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def params(self, temperature):
        from orchestrator.provenance.generator import VLLMGenerator
        generator = VLLMGenerator.__new__(VLLMGenerator)
        generator.sampling = self.FakeParams
        generator.temperature = temperature
        return generator.sampling_params(256)

    def test_greedy_asks_only_for_a_zero_temperature(self):
        self.assertEqual(self.params(0.0), {"temperature": 0.0, "max_tokens": 256})

    def test_sampling_above_zero_still_pins_the_truncation_parameters(self):
        params = self.params(0.7)
        self.assertEqual(params["top_p"], 1.0)
        self.assertEqual(params["top_k"], -1)
