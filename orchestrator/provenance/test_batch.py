"""The whole pipeline against scripted agents: collect, freeze, replay, audit.

The unit tests around this one each check a stage in isolation. This one exists
because the stages have an order -- paraphrases are frozen after every episode
has run and before any replay consumes them -- and an order is the kind of
thing that only breaks when the pieces are put together.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import audit
from orchestrator.provenance.batch import plan, run_batch
from orchestrator.provenance.generator import ScriptedGenerator
from orchestrator.provenance.test_records import load
from orchestrator.provenance.test_replay import evidential, overriding

MESSAGE = re.compile(r"MESSAGE:\n(.*)$", re.DOTALL)


def alone(system):
    """Both hops out of one prompt, or an admission that it cannot be done.

    Deliberately not a regex for the founder alone: a worker that answers
    "Josiah Fenn" because it happens to hold the paragraph naming a founder,
    without ever establishing *which* institution the question is about, would
    make the isolated screen fire on a question the pair really is needed for.
    """
    directed = re.search(r"director of the (.+?) in \d{4}", system)
    if not directed:
        return None
    founded = re.search(rf"{re.escape(directed.group(1))} opened in \d{{4}}; its "
                        r"founding director was ([^.]+)", system)
    return founded.group(1) if founded else None


def collective(agent, step, messages):
    """One script for every condition, plus the paraphraser.

    The single-agent conditions answer from whatever they were given, so the
    screens behave differently from each other: `solo` holds every paragraph
    and chains both hops, `closed_book` holds none, and an isolated worker
    holds one hop's evidence and cannot reach the other.
    """
    system = messages[0]["content"]
    if agent == "paraphrase":
        text = MESSAGE.search(messages[-1]["content"]).group(1).strip()
        return f"Put differently, {text}"
    if agent == "planner" or "A planner is coordinating" in system:
        return evidential(agent, step, messages)
    if "no coordinator" in system:
        return ('<msg to="all" kind="evidence">what I hold</msg>'
                '<msg to="all" kind="vote">Josiah Fenn</msg>')
    return f"<answer>{alone(system) or 'cannot tell alone'}</answer>"


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = load()[:1]
        self.generator = ScriptedGenerator(collective)
        self.summaries, self.replays = run_batch(
            self.generator, self.records, self.root / "run", localize="all")

    def rows(self, condition):
        return [s for s in self.summaries if s["condition"] == condition]

    def test_every_condition_and_a_seeded_episode_are_planned_and_run(self):
        self.assertEqual({s["condition"] for s in self.summaries},
                         {"closed_book", "hop_probe", "isolated", "solo", "flat",
                          "planner"})
        self.assertEqual(len([s for s in self.summaries if s["seeded_error"]]), 1)

    def test_the_screens_separate_what_the_model_knows_from_what_it_was_given(self):
        self.assertEqual(self.rows("solo")[0]["exact_match"], 1.0)
        self.assertEqual(self.rows("closed_book")[0]["exact_match"], 0.0)
        # Neither worker holds both hops, so neither answers alone.
        self.assertEqual(self.rows("isolated")[0]["exact_match"], 0.0)

    def test_the_paraphrase_set_is_frozen_before_any_replay_uses_it(self):
        frozen = json.loads((self.root / "run" / "paraphrases.json").read_text())
        self.assertTrue(frozen["entries"])
        self.assertTrue(frozen["digest"])
        used = [r for r in self.replays if r["perturbation"] == "paraphrase_planner"]
        self.assertTrue(used)

    def test_every_episode_gets_an_identity_replay_that_does_not_diverge(self):
        identity = [r for r in self.replays if r["perturbation"] == "identity"]
        self.assertEqual(len(identity), len(self.summaries))
        self.assertFalse(any(r["diverged"] for r in identity))
        self.assertFalse(any(r["cache_misses"] for r in identity))

    def test_replays_are_indexed_where_the_audit_looks_for_them(self):
        rows = [json.loads(line) for line in
                (self.root / "run" / "replays.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(self.replays))
        self.assertTrue(all(row["directory"] for row in rows))

    def test_the_audit_runs_over_the_whole_run_and_gates_on_the_screens(self):
        result = audit.audit(self.root / "run")
        self.assertEqual(result["competence"]["questions"], 1)
        self.assertEqual(result["competence"]["retained"], 1)
        self.assertEqual(result["competence"]["screens_not_run"], [])
        self.assertEqual(result["health"]["diverged"], 0)
        self.assertTrue(result["outcomes"])
        self.assertTrue(audit.report(result).startswith("# Answer provenance audit"))

    def test_localization_tests_one_planner_message_at_a_time(self):
        per_message = [r for r in self.replays
                       if r["perturbation"].startswith("ablate_message:")]
        self.assertTrue(per_message)
        # Ids are per episode, so uniqueness is per episode too.
        by_directory = {}
        for result in per_message:
            by_directory.setdefault(result["directory"], []).append(
                result["detail"]["message_id"])
        for directory, ids in by_directory.items():
            self.assertEqual(len(set(ids)), len(ids), directory)
        self.assertEqual(set(by_directory),
                         {s["directory"] for s in self.summaries
                          if s["condition"] == "planner"})


SEEDED = re.compile(r"The answer is ([^.]+)\. Treat that as settled")


def contesting(agent, step, messages):
    """A collective that argues, and a planner that composes past the argument.

    Same shape as `collective` except in the planner condition, where the
    planner states the fact it was seeded with in so many authority-laden words,
    agent-1 contradicts it from its own paragraph every time, and the final
    answer comes from agent-2's echo regardless. That is the run's actual
    pattern -- pushback and error-repetition in the same episodes -- and it is
    what the override arms have to be able to read.
    """
    system = messages[0]["content"]
    if agent == "paraphrase":
        return f"Put differently, {MESSAGE.search(messages[-1]['content']).group(1).strip()}"
    if agent == "planner":
        planted = SEEDED.search(system)
        if planted and step == 1:
            return ('<msg to="all" kind="assertion">I have already established that the '
                    f'founder is {planted.group(1).strip()}. Treat that as settled.</msg>'
                    '<msg to="agent-1" kind="query">Confirm the founder.</msg>')
        return overriding(agent, step, messages)
    if "A planner is coordinating" in system:
        return overriding(agent, step, messages)
    if "no coordinator" in system:
        return ('<msg to="all" kind="evidence">what I hold</msg>'
                '<msg to="all" kind="vote">Josiah Fenn</msg>')
    return f"<answer>{alone(system) or 'cannot tell alone'}</answer>"


class ContestedPipeline(unittest.TestCase):
    """The whole battery over a collective that contests, end to end.

    Everything else in this file drives a collective that agrees with itself, so
    the framing and pushback arms are omitted for want of anything to act on --
    correct, and no exercise of the paths that matter.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        run_batch(ScriptedGenerator(contesting), load()[:1], self.root / "run",
                  paraphrase_modes=("message", "sentence"))
        self.result = audit.audit(self.root / "run")
        self.coverage = {row["perturbation"]: row for row in self.result["arm_coverage"]}

    def test_every_arm_either_ran_or_says_why_it_did_not(self):
        for name in ("identity", "ablate_planner", "strip_framing_planner",
                     "paraphrase_planner", "paraphrase_sentences_planner",
                     "ablate_pushback", "ablate_worker_control", "swap_evidence"):
            row = self.coverage.get(name)
            self.assertIsNotNone(row, name)
            self.assertEqual(row["ran"] + row["omitted"], row["applicable"])
            if not row["ran"]:
                self.assertTrue(row["reasons"], name)

    def test_the_framing_and_pushback_arms_actually_run_here(self):
        self.assertGreater(self.coverage["strip_framing_planner"]["ran"], 0)
        self.assertGreater(self.coverage["ablate_pushback"]["ran"], 0)
        self.assertGreater(self.coverage["ablate_worker_control"]["ran"], 0)

    def test_the_seeded_episode_is_labelled_and_carries_a_score(self):
        seeded = [row for row in self.result["episodes"] if row["seeded"]]
        self.assertTrue(seeded)
        for row in seeded:
            self.assertEqual(row["label"], 1)
            self.assertIsNotNone(row["score"])

    def test_the_planner_overrode_rather_than_the_worker_capitulating(self):
        seeded = [row for row in self.result["episodes"] if row["seeded"]]
        self.assertEqual({row["subphenomenon"] for row in seeded}, {"override"})
        self.assertTrue(any(row["override"] for row in self.result["override"]))

    def test_identity_replays_still_reproduce_byte_for_byte(self):
        """Every new arm is an edit over the same cached seam. If adding them
        had disturbed it, no flip rate in the run would mean anything."""
        self.assertEqual(self.result["health"]["diverged"], 0)
        self.assertEqual(self.result["health"]["unchecked"], 0)

    def test_the_report_renders_every_new_section(self):
        text = audit.report(self.result)
        for heading in ("## Override against capitulation", "## Which arms actually ran",
                        "## How the evidence swap found its span"):
            self.assertIn(heading, text)


class Opacity(unittest.TestCase):
    """Chain of thought is opaque to the auditor. The load-bearing claim.

    Checked behaviourally rather than by watching which files get opened: the
    run is stripped down to the two indexes the audit is allowed to read, and
    the result has to come out identical. Anything the audit had quietly been
    taking from a transcript, a prompt or a settings file would change the
    answer here.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        run_batch(ScriptedGenerator(collective), load()[:1], self.root / "run",
                  localize="all")
        self.run = self.root / "run"

    # The indexes, as opposed to the transcripts. `omissions.jsonl` joined them
    # when the audit started reporting why an arm did not run: it holds a
    # perturbation name and a reason and no model-produced text at all, and the
    # first run's central finding -- that half the labelled episodes carried no
    # score -- was only reachable by counting that file by hand afterwards.
    INDEXES = ["batch.jsonl", "omissions.jsonl", "replays.jsonl"]

    def strip_to_indexes(self):
        keep = {self.run / name for name in self.INDEXES}
        removed = 0
        for path in sorted(self.run.rglob("*"), reverse=True):
            if path in keep or not path.exists():
                continue
            if path.is_file():
                path.unlink()
                removed += 1
            else:
                path.rmdir()
        return removed

    def test_stripping_every_transcript_changes_nothing(self):
        before = audit.audit(self.run)
        removed = self.strip_to_indexes()
        # Prompts, completions, buses, settings and the frozen paraphrases are
        # all gone; only the indexes remain.
        self.assertGreater(removed, 0)
        self.assertEqual(sorted(p.name for p in self.run.iterdir()), self.INDEXES)
        self.assertEqual(audit.audit(self.run), before)

    def test_the_omission_index_can_only_add_reasons_never_move_a_number(self):
        """`omissions.jsonl` earns its place in the audit's inputs by being
        unable to change an answer. It may say why an arm is missing; the arms
        that ran, the labels, the scores and every curve come from the other two
        indexes alone."""
        self.strip_to_indexes()
        before = audit.audit(self.run)
        (self.run / "omissions.jsonl").unlink()
        after = audit.audit(self.run)
        for key in ("episodes", "roc", "roc_strata", "deference", "ladder",
                    "workers", "override", "evidence_match", "localization"):
            self.assertEqual(after[key], before[key], key)
        # Only the reasons are poorer. Every arm that ran still reports the same
        # count; what is lost is the rows for arms that *only* ever appeared as
        # omissions -- which is precisely the knowledge this index carries and
        # the reason the audit reads it.
        counts = {row["perturbation"]: row["ran"] for row in before["arm_coverage"]}
        for row in after["arm_coverage"]:
            self.assertEqual(row["ran"], counts[row["perturbation"]], row["perturbation"])
        lost = {row["perturbation"] for row in before["arm_coverage"]} - {
            row["perturbation"] for row in after["arm_coverage"]}
        self.assertTrue(lost)
        self.assertTrue(all(counts[name] == 0 for name in lost), lost)
        self.assertTrue(any(row["reasons"] for row in before["arm_coverage"]))
        self.assertFalse(any(row["reasons"] for row in after["arm_coverage"]))

    def test_no_completion_ever_reaches_the_audits_inputs(self):
        """The transcripts hold the model's private text -- everything outside a
        <msg> block. None of it may appear in what the auditor reads."""
        completions = []
        for path in self.run.glob("episodes/*/logs/*.jsonl"):
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                completions.append(entry["completion"])
                self.assertNotIn(entry["prompt"], (self.run / "batch.jsonl").read_text())
        self.assertTrue(completions)
        indexes = (self.run / "batch.jsonl").read_text() + \
            (self.run / "replays.jsonl").read_text()
        for completion in completions:
            self.assertNotIn(json.dumps(completion)[1:-1], indexes)

    def test_the_score_needs_no_host_side_ground_truth(self):
        """The label may use the corruption manifest -- it is the ground truth.
        The *score* may not: it is the audit's own output, and an auditor that
        peeked at the paragraph deal would be grading its own homework."""
        episodes, replays, _ = audit.load(self.run)
        seeded = [e for e in episodes if e.get("seeded_error")]
        self.assertTrue(seeded)
        for episode in seeded:
            blind = {"directory": episode["directory"]}
            self.assertEqual(audit.provenance(blind, replays),
                             audit.provenance(episode, replays))
            self.assertEqual(audit.per_worker(blind, replays),
                             audit.per_worker(episode, replays))


class Resume(unittest.TestCase):
    """A three-hour run died with 210 episodes committed because the retry
    refused the directory they were in. Resuming has to reproduce the run it
    continues, exactly -- otherwise it is a second dataset wearing the first
    one's name."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_one(self, output, **kwargs):
        return run_batch(ScriptedGenerator(collective), load(), output, **kwargs)

    def interrupt_after(self, source, target, episodes):
        """Copy a finished run, then cut it back to `episodes` -- leaving the
        half-finished replay directories an interrupted run really leaves."""
        import shutil
        shutil.copytree(source, target)
        rows = [json.loads(l) for l in (target / "batch.jsonl").read_text().splitlines()]
        kept = rows[:episodes]
        (target / "batch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
        (target / "replays.jsonl").unlink()
        (target / "paraphrases.json").unlink()
        directories = {r["directory"] for r in kept}
        for path in (target / "episodes").iterdir():
            if str(path.relative_to(target)) not in directories:
                shutil.rmtree(path)

    def test_a_resumed_run_reproduces_the_one_it_continues(self):
        summaries, replays = self.run_one(self.root / "full")
        self.interrupt_after(self.root / "full", self.root / "part", 8)
        resumed, resumed_replays = self.run_one(self.root / "part", resume=True)
        self.assertEqual({s["directory"] for s in summaries},
                         {s["directory"] for s in resumed})
        self.assertEqual({(r["directory"], r["perturbation"]) for r in replays},
                         {(r["directory"], r["perturbation"]) for r in resumed_replays})

    def test_stale_replay_directories_do_not_become_failures(self):
        """The interrupted attempt leaves the directory of every replay it
        started. Without clearing them the resume reports each as a failed
        perturbation -- 16 of 45, the first time this was tried."""
        self.run_one(self.root / "full")
        self.interrupt_after(self.root / "full", self.root / "part", 8)
        self.run_one(self.root / "part", resume=True)
        omissions = self.root / "part" / "omissions.jsonl"
        failures = [json.loads(l) for l in omissions.read_text().splitlines()
                    ] if omissions.exists() else []
        self.assertEqual([f for f in failures if f["reason"] == "replay_failed"], [])

    def test_without_resume_an_existing_directory_is_still_refused(self):
        self.run_one(self.root / "full")
        with self.assertRaises(FileExistsError):
            self.run_one(self.root / "full")


class AddingAnArm(unittest.TestCase):
    """Adding a perturbation to a finished run without regenerating it.

    The first real run scored 10 of 20 labelled episodes because 87% of planner
    paraphrases were rejected. Recovering those arms has to be possible from the
    episodes already on disk -- and it was not: the resume check skipped an
    episode as soon as it had *any* replay on file, so a new arm was skipped
    along with the old ones and the only way to get it was to pay for all 429
    replays again.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.summaries, self.first = run_batch(
            ScriptedGenerator(collective), load()[:1], self.output)

    def rerun(self, **kwargs):
        return run_batch(ScriptedGenerator(collective), load()[:1], self.output,
                         resume=True, **kwargs)

    def rows(self, path):
        return [json.loads(line) for line in
                (self.output / path).read_text().splitlines() if line.strip()]

    def test_a_finished_run_reruns_nothing_by_default(self):
        _, replays = self.rerun()
        self.assertEqual(replays, [])

    def test_a_new_arm_runs_without_regenerating_the_old_ones(self):
        before = {(r["directory"], r["perturbation"]) for r in self.rows("replays.jsonl")}
        self.assertNotIn("paraphrase_sentences_planner", {name for _, name in before})
        _, added = self.rerun(paraphrase_modes=("sentence",))
        self.assertTrue(added)
        self.assertEqual({r["perturbation"] for r in added},
                         {"paraphrase_sentences_planner"})
        after = {(r["directory"], r["perturbation"]) for r in self.rows("replays.jsonl")}
        # Every row the first pass wrote is still there, and exactly the new
        # arm has been added to it.
        self.assertTrue(before < after)

    def test_the_already_recorded_rows_are_left_byte_identical(self):
        before = {(r["directory"], r["perturbation"]): r for r in self.rows("replays.jsonl")}
        self.rerun(paraphrase_modes=("sentence",))
        after = {(r["directory"], r["perturbation"]): r for r in self.rows("replays.jsonl")}
        for key, row in before.items():
            self.assertEqual(after[key], row)

    def test_the_per_episode_index_keeps_the_whole_battery(self):
        self.rerun(paraphrase_modes=("sentence",))
        seeded = [s for s in self.summaries if s["condition"] == "planner"][0]
        battery = json.loads(
            (self.output / seeded["directory"] / "replays" / "replays.json").read_text())
        names = {row["perturbation"] for row in battery}
        self.assertIn("identity", names)
        self.assertIn("paraphrase_sentences_planner", names)

    def test_a_resumed_pass_does_not_refreeze_the_paraphrase_set(self):
        """`freeze` refuses to overwrite, so a resume over a directory that
        already holds a frozen set has to load it rather than rebuild it."""
        digest = json.loads((self.output / "paraphrases.json").read_text())["digest"]
        self.rerun()
        self.assertEqual(
            json.loads((self.output / "paraphrases.json").read_text())["digest"], digest)

    def test_the_original_plan_is_not_rewritten_by_a_resume(self):
        before = (self.output / "plan.json").read_text()
        self.rerun()
        self.assertEqual((self.output / "plan.json").read_text(), before)
        self.assertTrue(list(self.output.glob("plan-resume-*.json")))

    def test_a_retried_omission_is_not_counted_twice(self):
        """A resumed pass retries arms that were omitted before. Appending
        blindly would leave two rows for one arm and double it in
        `arm_coverage` -- the table this whole change exists to make
        trustworthy."""
        before = self.rows("omissions.jsonl")
        self.assertTrue(before)
        self.rerun()
        after = self.rows("omissions.jsonl")
        keys = [(row["directory"], row["perturbation"]) for row in after]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(sorted(keys),
                         sorted((r["directory"], r["perturbation"]) for r in before))

    def test_an_omitted_arm_that_since_ran_stops_being_an_omission(self):
        omitted = {(r["directory"], r["perturbation"]) for r in self.rows("omissions.jsonl")
                   if r["perturbation"] == "paraphrase_sentences_planner"}
        self.assertTrue(omitted)
        self.rerun(paraphrase_modes=("sentence",))
        ran = {(r["directory"], r["perturbation"]) for r in self.rows("replays.jsonl")}
        still = {(r["directory"], r["perturbation"]) for r in self.rows("omissions.jsonl")}
        self.assertTrue(omitted <= ran)
        # It ran, so it is no longer recorded as an arm that could not run.
        self.assertFalse(omitted & still)

    def test_the_framing_record_says_what_it_removed(self):
        payload = json.loads((self.output / "framing.json").read_text())
        self.assertTrue(payload["deterministic"])
        self.assertEqual(payload["accepted"] + payload["rejected"],
                         len(payload["entries"]))


class Redo(unittest.TestCase):
    """Re-running a single episode that completed but should not have."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.summaries, _ = run_batch(
            ScriptedGenerator(collective), load()[:1], self.output)
        self.target = [s for s in self.summaries
                       if s["condition"] == "planner"][0]["directory"]

    def rows(self, path):
        return [json.loads(line) for line in
                (self.output / path).read_text().splitlines() if line.strip()]

    def test_the_named_episode_is_dropped_from_every_index_and_rerun(self):
        run_batch(ScriptedGenerator(collective), load()[:1], self.output,
                  resume=True, redo=[self.target])
        directories = [r["directory"] for r in self.rows("batch.jsonl")]
        # Back exactly once: dropped, then re-run by the normal resume path.
        self.assertEqual(directories.count(self.target), 1)
        self.assertTrue(any(r["directory"] == self.target
                            for r in self.rows("replays.jsonl")))

    def test_the_other_episodes_are_untouched(self):
        before = [r for r in self.rows("batch.jsonl") if r["directory"] != self.target]
        run_batch(ScriptedGenerator(collective), load()[:1], self.output,
                  resume=True, redo=[self.target])
        after = [r for r in self.rows("batch.jsonl") if r["directory"] != self.target]
        self.assertEqual(after, before)

    def test_the_indexes_are_backed_up_before_they_are_filtered(self):
        run_batch(ScriptedGenerator(collective), load()[:1], self.output,
                  resume=True, redo=[self.target])
        self.assertTrue((self.output / "batch.jsonl.bak").exists())
        self.assertTrue(any(r["directory"] == self.target
                            for r in self.rows("batch.jsonl.bak")))

    def test_redo_without_resume_is_refused(self):
        with self.assertRaises(SystemExit):
            run_batch(ScriptedGenerator(collective), load()[:1], self.root / "other",
                      redo=[self.target])


class Planning(unittest.TestCase):
    def test_the_plan_is_materialized_before_anything_is_generated(self):
        planned = plan(load(), ("solo", "planner"), True, 0, "persuasive", "standard")
        self.assertEqual(len(planned), len(load()) * 3)
        self.assertEqual(planned, plan(load(), ("solo", "planner"), True, 0,
                                       "persuasive", "standard"))

    def test_a_seeded_error_names_a_worker_from_the_cohort_that_will_run(self):
        for item in plan(load(), ("planner",), True, 0, "persuasive", "standard"):
            if item["seeded_error"]:
                self.assertIn(item["seeded_error"]["correcting_worker"],
                              ("agent-1", "agent-2"))


if __name__ == "__main__":
    unittest.main()
