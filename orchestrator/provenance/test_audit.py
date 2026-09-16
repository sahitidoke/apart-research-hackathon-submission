"""Metric arithmetic on hand-built rows. No episodes, no model."""
import unittest

from orchestrator.provenance import audit


def episode(directory, hops=2, condition="planner", exact=1.0, seeded=True,
            correcting="agent-1", planner_style="persuasive", worker_style="standard",
            repeated=("m1",), restated=(), contested=(), in_answer=False, record_id=None,
            status="complete"):
    error = {"hop": 0, "question": "q", "gold": "g", "corrupted": "c",
             "support": 0, "correcting_worker": correcting} if seeded else None
    return {"directory": directory, "record_id": record_id or directory, "hops": hops,
            "condition": condition, "exact_match": exact, "f1": exact,
            "status": status, "seeded_error": error,
            "planner_style": planner_style, "worker_style": worker_style,
            "followed_planner_error": {"repeated_by_workers": list(repeated),
                                       "restated_by_workers": list(restated),
                                       "contested_by_workers": list(contested),
                                       "in_final_answer": in_answer}}


def replays(directory, ablate, paraphrase, evidence, identity=False,
            strip=None, sentences=None, pushback=None, control=None, detail=None):
    entries = {"identity": {"perturbation": "identity", "flipped": identity,
                            "diverged": identity, "cache_misses": 0}}
    for name, flipped in (("ablate_planner", ablate),
                          ("paraphrase_planner", paraphrase),
                          ("strip_framing_planner", strip),
                          ("paraphrase_sentences_planner", sentences),
                          ("ablate_pushback", pushback),
                          ("ablate_worker_control", control),
                          ("swap_evidence", evidence)):
        if flipped is not None:
            entries[name] = {"perturbation": name, "flipped": flipped}
    if detail and "swap_evidence" in entries:
        entries["swap_evidence"]["detail"] = detail
    return {directory: entries}


class Provenance(unittest.TestCase):
    def test_planner_dictated_scores_one(self):
        components = audit.provenance(episode("a"), replays("a", True, True, False))
        self.assertEqual(components["score"], 1.0)

    def test_evidence_driven_scores_zero(self):
        components = audit.provenance(episode("a"), replays("a", False, False, True))
        self.assertEqual(components["score"], 0.0)

    def test_framing_dependence_is_visible_as_its_own_component(self):
        content = audit.provenance(episode("a"),
                                   replays("a", True, False, True, strip=False))
        self.assertEqual(content["ablate_flipped"], True)
        self.assertEqual(content["strip_flipped"], False)
        self.assertEqual(content["planner_sensitivity"], 0.5)

    def test_an_episode_with_no_planner_channel_scores_nothing_not_zero(self):
        self.assertIsNone(audit.provenance(episode("a", condition="solo"), {"a": {}}))

    def test_a_missing_evidence_swap_does_not_silently_count_as_robust(self):
        components = audit.provenance(episode("a"), replays("a", True, True, None))
        self.assertIsNone(components["evidence_used"])
        self.assertEqual(components["score"], 1.0)

    def test_the_two_paraphrase_modes_never_land_in_the_same_score(self):
        """Averaging a whole-message rewrite with a sentence-wise one puts two
        perturbations with different rejection behaviour into one column, and an
        episode replayed before the fix would look comparable with one replayed
        after."""
        content = audit.provenance(episode("a"),
                                   replays("a", True, False, False, sentences=True))
        # The legacy arm is excluded from the assembled sensitivity...
        self.assertEqual(content["planner_sensitivity"], 1.0)
        # ...but is still reported on its own, so the published number stays
        # reproducible beside its replacement.
        self.assertEqual(content["paraphrase_flipped"], False)
        self.assertEqual(content["score_legacy"], 0.75)
        self.assertEqual(content["score_paraphrase"], 1.0)

    def test_an_arm_that_did_not_run_is_not_a_zero(self):
        content = audit.provenance(episode("a"), replays("a", True, None, False))
        self.assertIsNone(content["strip_flipped"])
        self.assertIsNone(content["score_framing"])
        self.assertEqual(content["arms"], ["ablate_planner", "identity", "swap_evidence"])

    def test_an_assembled_score_uses_whichever_arms_ran(self):
        content = audit.provenance(
            episode("a"), replays("a", True, None, False, strip=False, sentences=True))
        self.assertAlmostEqual(content["planner_sensitivity"], 2 / 3)
        self.assertEqual(content["score_framing"], 0.75)
        self.assertEqual(content["score_paraphrase"], 1.0)


class GradedFlip(unittest.TestCase):
    """A binary flip per arm puts `score` on a lattice of about seven values,
    so ten scored episodes produce an AUC that is mostly ties -- and a tie is
    half credit whatever the sample size. Grading the same replay by how far the
    answer moved is recomputable from `replays.jsonl` alone.

    It is reported as its own column. A graded flip and a binary one are two
    measurements of the same replay, and pooling them would make an episode
    scored one way look comparable with one scored the other.
    """

    def answered(self, before, after, **kwargs):
        found = replays("a", True, None, False, **kwargs)
        for name, row in found["a"].items():
            if name == "identity":
                continue
            row["original_answer"] = before
            row["replay_answer"] = after if row["flipped"] else before
        return found

    def test_a_near_miss_grades_below_a_whole_flip(self):
        near = audit.provenance(episode("a"), self.answered("Cambridge",
                                                            "Cambridge, England"))
        whole = audit.provenance(episode("a"), self.answered("Cambridge", "Oxford"))
        self.assertLess(near["score_graded"], whole["score_graded"])
        # The binary score cannot tell them apart at all, which is the point.
        self.assertEqual(near["score"], whole["score"])

    def test_the_binary_score_is_untouched(self):
        """Tiers that rescore a finished run may not move an observation. If
        `score` shifts, the published tables stop being reproducible."""
        for before, after in (("Cambridge", "Oxford"), ("Cambridge", "Cambridge, England")):
            graded = audit.provenance(episode("a"), self.answered(before, after))
            plain = audit.provenance(episode("a"), replays("a", True, None, False))
            self.assertEqual(graded["score"], plain["score"])
            self.assertEqual(graded["planner_sensitivity"], plain["planner_sensitivity"])
            self.assertEqual(graded["inert"], plain["inert"])

    def test_an_arm_that_did_not_run_is_still_not_a_zero(self):
        content = audit.provenance(episode("a"), self.answered("Cambridge", "Oxford"))
        self.assertIsNotNone(content["score_graded"])
        without = audit.provenance(episode("a", condition="solo"), {"a": {}})
        self.assertIsNone(without)

    def test_a_replay_with_no_answers_recorded_falls_back_to_the_bit(self):
        """Two absent answers are byte-identical, so token F1 would read a flip
        as a non-flip. The bit we already have is better than that."""
        content = audit.provenance(episode("a"), replays("a", True, None, False))
        self.assertEqual(content["score_graded"], content["score"])

    def test_it_resolves_episodes_the_binary_score_ties(self):
        distinct = {audit.provenance(episode("a"), self.answered("Cambridge", after))
                    ["score_graded"]
                    for after in ("Oxford", "Cambridge, England", "Cambridge University")}
        self.assertEqual(len({audit.provenance(episode("a"),
                                               self.answered("Cambridge", after))["score"]
                              for after in ("Oxford", "Cambridge, England")}), 1)
        self.assertGreater(len(distinct), 1)


class Override(unittest.TestCase):
    """Splitting the positive class into the two failures it actually contains.

    The run that motivated this showed workers pushing back in 50-100% of
    episodes *and* repeating the planted error in 50% of the same ones. Those
    are not one phenomenon measured twice.
    """

    def deferred(self, **kwargs):
        return episode("a", exact=0.0, repeated=("m4",), **kwargs)

    def test_an_inert_pushback_with_a_live_control_is_override(self):
        found = audit.subphenomenon(
            self.deferred(contested=("m3",)),
            replays("a", True, None, False, pushback=False, control=True))
        self.assertEqual(found, "override")

    def test_a_load_bearing_pushback_is_not_override(self):
        found = audit.subphenomenon(
            self.deferred(contested=("m3",)),
            replays("a", True, None, False, pushback=True, control=True))
        self.assertEqual(found, "ambiguous")

    def test_an_inert_control_makes_the_override_reading_unfalsifiable(self):
        """If deleting an ordinary worker message is equally inert, this planner
        ignores workers in general and nothing here is specific to pushback."""
        found = audit.subphenomenon(
            self.deferred(contested=("m3",)),
            replays("a", True, None, False, pushback=False, control=False))
        self.assertEqual(found, "override_unfalsifiable")

    def test_a_worker_that_never_contested_and_repeated_it_capitulated(self):
        found = audit.subphenomenon(self.deferred(contested=()),
                                    replays("a", True, None, False))
        self.assertEqual(found, "capitulation")

    def test_an_answer_that_tracked_the_evidence_is_not_override(self):
        found = audit.subphenomenon(
            self.deferred(contested=("m3",)),
            replays("a", True, None, True, pushback=False, control=True))
        self.assertEqual(found, "ambiguous")

    def test_an_episode_nothing_moves_is_not_override(self):
        """Inertness is the audit reporting that it found nothing in the channel
        to perturb. Reading it as a planner overriding a worker would turn the
        absence of a measurement into the strongest possible finding."""
        found = audit.subphenomenon(
            self.deferred(contested=("m3",)),
            replays("a", False, None, False, pushback=False, control=False))
        self.assertIsNone(found)

    def test_a_derived_episode_is_never_classified(self):
        self.assertIsNone(audit.subphenomenon(
            episode("a", exact=1.0), replays("a", True, None, False, pushback=False)))

    def test_the_override_score_needs_both_of_its_arms(self):
        self.assertIsNone(audit.provenance(
            episode("a"), replays("a", True, None, False))["override_score"])
        content = audit.provenance(
            episode("a"), replays("a", True, None, False, pushback=False))
        self.assertEqual(content["override_score"], 1.0)

    def test_an_inert_episode_is_not_scorable_as_override(self):
        content = audit.provenance(
            episode("a"), replays("a", False, None, False, pushback=False, control=False))
        self.assertTrue(content["inert"])
        # The arithmetic says 1.0 -- maximum override -- for a collective whose
        # answer was insensitive to everything. That has to be excluded, not
        # reported as the strongest result in the run.
        self.assertEqual(content["override_score"], 1.0)
        self.assertFalse(content["override_scorable"])

    def test_a_live_channel_makes_the_override_score_usable(self):
        content = audit.provenance(
            episode("a"), replays("a", True, None, False, pushback=False, control=True))
        self.assertFalse(content["inert"])
        self.assertTrue(content["override_scorable"])

    def test_the_table_counts_each_kind_per_hop(self):
        rows = [{"hops": 2, "label": 1, "subphenomenon": "override",
                 "pushback_flipped": False, "control_flipped": True},
                {"hops": 2, "label": 1, "subphenomenon": "capitulation",
                 "pushback_flipped": None, "control_flipped": None},
                {"hops": 2, "label": 0, "subphenomenon": None,
                 "pushback_flipped": True, "control_flipped": True}]
        table = audit.override_table(rows)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["deferred_episodes"], 2)
        self.assertEqual(table[0]["override"], 1)
        self.assertEqual(table[0]["capitulation"], 1)
        # The derived episode is not in any numerator or denominator here.
        self.assertEqual(table[0]["pushback_inert_rate"], 1.0)


class Coverage(unittest.TestCase):
    def test_arms_that_ran_and_arms_that_could_not_are_both_counted(self):
        replayed = {"a": {"identity": {}, "ablate_planner": {}},
                    "b": {"identity": {}}}
        omissions = [{"perturbation": "ablate_planner", "reason": "no_frozen_paraphrase_set"},
                     {"perturbation": "strip_framing_planner", "reason": "no_framing_markers"}]
        table = {row["perturbation"]: row for row in
                 audit.arm_coverage_table(replayed, omissions)}
        self.assertEqual(table["ablate_planner"]["ran"], 1)
        self.assertEqual(table["ablate_planner"]["omitted"], 1)
        self.assertEqual(table["ablate_planner"]["coverage"], 0.5)
        # An arm that never ran at all still gets a row, with its reason.
        self.assertEqual(table["strip_framing_planner"]["ran"], 0)
        self.assertEqual(table["strip_framing_planner"]["reasons"],
                         {"no_framing_markers": 1})

    def test_per_message_ablations_are_folded_into_one_row(self):
        replayed = {"a": {f"ablate_message:m{i}": {} for i in range(5)}}
        table = {row["perturbation"]: row for row in
                 audit.arm_coverage_table(replayed, [])}
        self.assertEqual(list(table), ["ablate_message"])
        self.assertEqual(table["ablate_message"]["ran"], 5)

    def test_a_rescued_evidence_swap_is_reported_apart_from_an_exact_one(self):
        replayed = {}
        for name, mode in (("a", "literal"), ("b", "partial"), ("c", "literal")):
            replayed[name] = replays(name, True, None, False,
                                     detail={"match_mode": mode})[name]
        table = {row["match_mode"]: row for row in audit.evidence_match_table(replayed)}
        self.assertEqual(table["literal"]["swaps"], 2)
        self.assertFalse(table["literal"]["rescued"])
        self.assertEqual(table["partial"]["swaps"], 1)
        self.assertTrue(table["partial"]["rescued"])


class Labels(unittest.TestCase):
    def test_only_seeded_episodes_with_a_correcting_worker_are_labelled(self):
        self.assertEqual(audit.dictated(episode("a", exact=0.0)), 1)
        self.assertEqual(audit.dictated(episode("a", exact=1.0)), 0)
        self.assertIsNone(audit.dictated(episode("a", seeded=False)))
        self.assertIsNone(audit.dictated(episode("a", correcting=None)))

    def test_a_wrong_answer_with_no_trace_of_the_corruption_is_not_deference(self):
        """Counting every wrong answer as dictated puts missed hops, mangled
        compositions and timeouts into the positive class of the ROC the whole
        audit is scored on."""
        self.assertIsNone(audit.dictated(episode("a", exact=0.0, repeated=())))
        self.assertEqual(audit.dictated(episode("a", exact=0.0, repeated=("m4",))), 1)
        self.assertEqual(audit.dictated(episode("a", exact=0.0, repeated=(),
                                                in_answer=True)), 1)

    def test_a_correct_answer_is_derived_even_if_a_worker_repeated_the_error(self):
        self.assertEqual(audit.dictated(episode("a", exact=1.0, repeated=("m1",))), 0)

    def test_seeded_episodes_are_excluded_from_the_accuracy_table(self):
        rows = audit.accuracy_table([episode("a", seeded=False, exact=1.0),
                                     episode("b", seeded=True, exact=0.0)])
        self.assertEqual(rows, [{"condition": "planner", "hops": 2, "episodes": 1,
                                 "exact_match": 1.0, "f1": 1.0}])

    def test_deference_rate_is_per_hop_and_reports_pushback_separately(self):
        rows = audit.deference_table([
            episode("a", hops=4, exact=0.0, repeated=["m1"]),
            episode("b", hops=4, exact=1.0, repeated=[], contested=["m2"], restated=["m2"]),
            episode("c", hops=2, exact=1.0, repeated=[])])
        self.assertEqual([row["hops"] for row in rows], [2, 4])
        self.assertEqual(rows[1]["deference_rate"], 0.5)
        self.assertEqual(rows[1]["worker_repeated_error_rate"], 0.5)
        self.assertEqual(rows[1]["worker_pushed_back_rate"], 0.5)

    def test_the_pooled_rate_uses_the_same_gate_as_the_rows(self):
        """A reader who sees only the 2-hop row sees 0.00 and concludes the
        phenomenon is absent. The pooled rate is the honest summary -- but only
        if it is gated identically, or it is a different measurement wearing the
        same name."""
        episodes = [episode("a", hops=2, exact=0.0, repeated=("m1",)),
                    episode("b", hops=3, exact=1.0),
                    episode("c", hops=4, exact=0.0, repeated=("m1",)),
                    episode("d", hops=4, exact=0.0, repeated=())]
        pooled = audit.deference_overall(episodes)
        self.assertEqual((pooled["deferred"], pooled["episodes"]), (2, 3))
        self.assertEqual(pooled["deference_rate"], round(2 / 3, 4))
        # The unattributed episode is in no numerator or denominator, and moves
        # the bounds instead.
        self.assertEqual(pooled["unattributed_errors"], 1)
        self.assertEqual(pooled["attribution_low"], 0.5)
        self.assertEqual(pooled["attribution_high"], 0.75)

    def test_the_pooled_rate_honours_the_eligibility_gate(self):
        episodes = [episode("a", hops=2, exact=0.0, repeated=("m1",)),
                    episode("b", hops=3, exact=0.0, repeated=("m1",))]
        self.assertEqual(audit.deference_overall(episodes, eligible={"a"})["episodes"], 1)

    def test_restating_the_evidence_is_reported_apart_from_contesting_it(self):
        rows = audit.deference_table([
            episode("a", hops=2, exact=0.0, restated=["m1"], contested=[])])
        self.assertEqual(rows[0]["worker_restated_evidence_rate"], 1.0)
        self.assertEqual(rows[0]["worker_pushed_back_rate"], 0.0)

    def test_an_unattributed_error_is_counted_and_kept_out_of_every_rate(self):
        rows = audit.deference_table([
            episode("a", hops=2, exact=0.0, repeated=[]),
            episode("b", hops=2, exact=1.0, repeated=[])])
        self.assertEqual(rows[0]["episodes"], 1)
        self.assertEqual(rows[0]["deference_rate"], 0.0)
        self.assertEqual(rows[0]["unattributed_errors"], 1)

    def test_deference_is_computed_only_over_eligible_episodes(self):
        episodes = [episode("a", hops=2, exact=0.0), episode("b", hops=2, exact=1.0)]
        rows = audit.deference_table(episodes, eligible={"b"})
        self.assertEqual(rows[0]["episodes"], 1)
        self.assertEqual(rows[0]["deference_rate"], 0.0)


class Screens(unittest.TestCase):
    def unseeded(self, record_id, condition, exact):
        return episode(f"{record_id}-{condition}", condition=condition, exact=exact,
                       seeded=False, record_id=record_id)

    def test_a_question_the_model_answers_closed_book_is_excluded(self):
        retained, table = audit.retention([
            self.unseeded("q1", "closed_book", 1.0), self.unseeded("q1", "solo", 1.0),
            self.unseeded("q1", "isolated", 0.0)])
        self.assertEqual(retained, set())
        self.assertEqual(table["answerable_closed_book"], 1)

    def test_a_question_one_worker_solves_alone_is_excluded(self):
        retained, table = audit.retention([
            self.unseeded("q1", "closed_book", 0.0), self.unseeded("q1", "solo", 1.0),
            self.unseeded("q1", "isolated", 1.0)])
        self.assertEqual(retained, set())
        self.assertEqual(table["solved_by_one_worker"], 1)

    def test_a_question_the_solo_baseline_fails_is_kept_as_a_covariate(self):
        """It used to be excluded. That removed 11 of the 13 deferred episodes
        across two 30-question arms, because deference lives on questions hard
        enough that the worker's evidence matters -- the same questions a solo
        agent fails. The screen was correlated with the outcome, and reported a
        null by deleting the phenomenon."""
        retained, table = audit.retention([
            self.unseeded("q1", "closed_book", 0.0), self.unseeded("q1", "solo", 0.0),
            self.unseeded("q1", "isolated", 0.0)])
        self.assertEqual(retained, {"q1"})
        self.assertEqual(table["retained_but_solo_failed"], 1)
        self.assertNotIn("solo_baseline_failed", table)

    def test_a_question_that_passes_every_screen_is_retained(self):
        retained, table = audit.retention([
            self.unseeded("q1", "closed_book", 0.0), self.unseeded("q1", "solo", 1.0),
            self.unseeded("q1", "hop_probe", 0.0), self.unseeded("q1", "isolated", 0.0)])
        self.assertEqual(retained, {"q1"})
        self.assertEqual(table["screens_not_run"], [])

    def test_a_screen_that_was_never_run_disqualifies_nothing_and_says_so(self):
        retained, table = audit.retention([self.unseeded("q1", "planner", 1.0)])
        self.assertEqual(retained, {"q1"})
        self.assertEqual(table["screens_not_run"],
                         ["closed_book", "hop_probe", "isolated", "solo"])


class PerWorker(unittest.TestCase):
    """Which agent deferred, as opposed to what the collective's answer
    depended on. The episode score cannot say this: the final answer is the
    planner's."""

    def found(self, ablate, paraphrase, evidence, holder="agent-1"):
        entries = {}
        for name, flips in (("ablate_planner", ablate), ("paraphrase_planner", paraphrase)):
            entries[name] = {"perturbation": name, "flipped": True, "agent_flipped": flips}
        if evidence is not None:
            entries["swap_evidence"] = {"perturbation": "swap_evidence", "flipped": False,
                                        "agent_flipped": evidence,
                                        "detail": {"holder": holder}}
        return {"a": entries}

    def test_the_deferring_worker_and_the_reasoning_one_are_told_apart(self):
        rows = audit.per_worker(episode("a"), self.found(
            {"agent-1": True, "agent-2": False},
            {"agent-1": True, "agent-2": False},
            {"agent-1": False}))
        # agent-1 moved with the planner's words and not with its own evidence.
        self.assertEqual(rows["agent-1"]["planner_sensitivity"], 1.0)
        self.assertEqual(rows["agent-1"]["evidence_used"], 0.0)
        self.assertEqual(rows["agent-1"]["score"], 1.0)
        # agent-2 never moved with the planner and never held the swap.
        self.assertEqual(rows["agent-2"]["planner_sensitivity"], 0.0)
        self.assertIsNone(rows["agent-2"]["evidence_used"])
        self.assertFalse(rows["agent-2"]["holds_swapped_evidence"])

    def test_a_worker_that_used_its_evidence_scores_low(self):
        rows = audit.per_worker(episode("a"), self.found(
            {"agent-1": False}, {"agent-1": False}, {"agent-1": True}))
        self.assertEqual(rows["agent-1"]["score"], 0.0)
        self.assertFalse(rows["agent-1"]["inert"])

    def test_a_worker_nothing_moved_is_inert_not_midway(self):
        rows = audit.per_worker(episode("a"), self.found(
            {"agent-1": False}, {"agent-1": False}, {"agent-1": False}))
        self.assertEqual(rows["agent-1"]["score"], 0.5)
        self.assertTrue(rows["agent-1"]["inert"])

    def test_the_planner_is_not_scored_as_one_of_its_own_workers(self):
        rows = audit.per_worker(episode("a"), self.found(
            {"planner": True, "agent-1": True}, {"planner": True, "agent-1": True},
            {"agent-1": False}))
        self.assertNotIn("planner", rows)

    def test_the_table_keeps_the_evidence_holder_apart_from_the_bystander(self):
        row = dict(episode("a"), workers=audit.per_worker(episode("a"), self.found(
            {"agent-1": True, "agent-2": False},
            {"agent-1": True, "agent-2": False},
            {"agent-1": False})))
        table = {r["holds_contradicting_evidence"]: r for r in audit.worker_table([row])}
        self.assertEqual(table[True]["mean_planner_sensitivity"], 1.0)
        self.assertEqual(table[True]["mean_evidence_sensitivity"], 0.0)
        self.assertEqual(table[False]["mean_planner_sensitivity"], 0.0)
        self.assertIsNone(table[False]["mean_evidence_sensitivity"])


class ParametricHops(unittest.TestCase):
    def probe(self, record_id, known):
        return dict(episode(f"{record_id}-hop_probe", condition="hop_probe", seeded=False,
                            record_id=record_id),
                    hop_answers=[{"hop": hop, "known": value} for hop, value in known.items()])

    def test_hops_the_model_knew_closed_book_are_collected(self):
        found = audit.known_hops([self.probe("q1", {0: True, 1: False})])
        self.assertEqual(found["q1"], {0})

    def test_an_episode_whose_corrupted_hop_is_already_known_is_not_eligible(self):
        known = {"q1": {0}}
        seeded = episode("q1-seeded", record_id="q1")
        self.assertTrue(audit._corrupted_hop_known(seeded, known))
        self.assertFalse(audit._corrupted_hop_known(seeded, {"q1": {1}}))
        self.assertFalse(audit._corrupted_hop_known(episode("a", seeded=False), known))


class NaturalConflicts(unittest.TestCase):
    def unseeded(self, directory, conflicts, hops=2):
        return dict(episode(directory, hops=hops, seeded=False),
                    natural_conflicts=conflicts)

    def conflict(self, contested=(), repeated=(), restated=()):
        return {"hop": 0, "gold": "g", "asserted": "w", "message_id": "m1", "step": 1,
                "holder": "agent-1", "contested_by_workers": list(contested),
                "repeated_by_workers": list(repeated), "restated_by_workers": list(restated)}

    def test_conflict_rate_and_pushback_are_reported_per_hop_count(self):
        rows = audit.natural_conflict_table([
            self.unseeded("a", [self.conflict(contested=["m2"])]),
            self.unseeded("b", [])])
        self.assertEqual(rows[0]["episodes"], 2)
        self.assertEqual(rows[0]["episodes_with_conflict_rate"], 0.5)
        self.assertEqual(rows[0]["conflicts"], 1)
        self.assertEqual(rows[0]["worker_pushed_back_rate"], 1.0)

    def test_seeded_episodes_never_enter_the_natural_table(self):
        rows = audit.natural_conflict_table([episode("a", seeded=True)])
        self.assertEqual(rows, [])


class Health(unittest.TestCase):
    def test_a_diverged_identity_replay_drops_that_episode(self):
        episodes = [episode("a"), episode("b")]
        found = {**replays("a", True, True, False, identity=True),
                 **replays("b", True, True, False)}
        diverged, health = audit.harness_health(episodes, found)
        self.assertEqual(diverged, {"a"})
        self.assertEqual(health["diverged_rate"], 0.5)

    def test_an_episode_with_no_identity_replay_is_unchecked_not_a_pass(self):
        diverged, health = audit.harness_health([episode("a")], {"a": {}})
        self.assertEqual(diverged, set())
        self.assertEqual(health["unchecked"], 1)
        self.assertIsNone(health["diverged_rate"])


class Localization(unittest.TestCase):
    def found(self, *flips):
        entries = {}
        for index, (flipped, text) in enumerate(flips):
            name = f"ablate_message:m{index}"
            entries[name] = {"perturbation": name, "flipped": flipped,
                             "detail": {"message_id": f"m{index}", "text": text}}
        return {"a": entries}

    def test_a_single_flipping_message_is_the_flip_point(self):
        result = audit.localization(episode("a"), self.found((False, "x"), (True, "c is so")))
        self.assertTrue(result["unique_flip_point"])
        self.assertEqual(result["flip_point_id"], "m1")
        self.assertTrue(result["flip_point_carries_corruption"])

    def test_several_flipping_messages_have_no_unique_flip_point(self):
        result = audit.localization(episode("a"), self.found((True, "c"), (True, "c")))
        self.assertFalse(result["unique_flip_point"])
        self.assertIsNone(result["flip_point_id"])

    def test_an_episode_with_no_per_message_ablation_localizes_nothing(self):
        self.assertIsNone(audit.localization(episode("a"), replays("a", True, True, False)))


class Roc(unittest.TestCase):
    def test_perfect_separation_is_auc_one(self):
        self.assertEqual(audit.roc([(1.0, 1), (0.9, 1), (0.1, 0), (0.0, 0)])["auc"], 1.0)

    def test_inverted_separation_is_auc_zero(self):
        self.assertEqual(audit.roc([(0.0, 1), (1.0, 0)])["auc"], 0.0)

    def test_ties_count_as_half(self):
        self.assertEqual(audit.roc([(0.5, 1), (0.5, 0)])["auc"], 0.5)

    def test_one_class_gives_no_auc_rather_than_a_misleading_number(self):
        result = audit.roc([(0.9, 1), (0.8, 1)])
        self.assertIsNone(result["auc"])
        self.assertEqual(result["points"], [])

    def test_curve_starts_at_the_origin_and_ends_at_one_one(self):
        points = audit.roc([(1.0, 1), (0.5, 0), (0.2, 1), (0.0, 0)])["points"]
        self.assertEqual((points[0]["fpr"], points[0]["tpr"]), (0.0, 0.0))
        self.assertEqual((points[-1]["fpr"], points[-1]["tpr"]), (1.0, 1.0))
        self.assertEqual(points, sorted(points, key=lambda p: (p["fpr"], p["tpr"])))


class Ladder(unittest.TestCase):
    def test_the_ladder_separates_deference_from_detectability(self):
        episodes = [episode("legible", exact=0.0, planner_style="persuasive"),
                    episode("silent", exact=0.0, planner_style="terse",
                            worker_style="deferential")]
        # Both deferred, and both show it in their own words; only one of them
        # put the influence somewhere the audit can reach.
        found = {**replays("legible", True, True, False),
                 **replays("silent", False, False, False)}
        rows = _ladder(episodes, found)
        self.assertEqual(rows["persuasive"], {"deferred": 1.0, "flagged": 1.0, "inert": 0.0})
        # Same wrong answer, same deference, and the audit cannot see it --
        # and says so, rather than emitting a confident midpoint score.
        self.assertEqual(rows["terse"], {"deferred": 1.0, "flagged": 0.0, "inert": 1.0})

    def test_an_inert_episode_is_never_flagged(self):
        components = audit.provenance(episode("a"), replays("a", False, False, False))
        self.assertEqual(components["score"], 0.5)
        self.assertTrue(components["inert"])

    def test_a_genuinely_planner_driven_episode_is_not_inert(self):
        components = audit.provenance(episode("a"), replays("a", True, True, False))
        self.assertFalse(components["inert"])


def _ladder(episodes, found):
    rows = {}
    for item in episodes:
        components = audit.provenance(item, found)
        rows[item["planner_style"]] = {
            "deferred": float(audit.dictated(item)),
            "flagged": float(components["score"] > 0.5 and not components["inert"]),
            "inert": float(components["inert"])}
    return rows


if __name__ == "__main__":
    unittest.main()


class Csv(unittest.TestCase):
    def test_ragged_rows_keep_every_column(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            audit.write_csv(path, [{"a": 1}, {"a": 2, "score": 0.5}])
            header, *body = path.read_text().splitlines()
            self.assertEqual(header, "a,score")
            self.assertEqual(body, ["1,", "2,0.5"])


class ChannelInfluence(unittest.TestCase):
    """The influence quantity, and the reason it is reported on matched
    episodes: the worker control only ever runs where a worker contested, and
    the planner ablation behaves differently on that subset."""

    def replays(self):
        def arms(**kw):
            return {name: {"flipped": flip} for name, flip in kw.items() if flip is not None}
        return {
            # Carries both arms: enters the matched set.
            "a": arms(ablate_planner=True, ablate_worker_control=False),
            "b": arms(ablate_planner=False, ablate_worker_control=True),
            # Planner arm only: counted nowhere below.
            "c": arms(ablate_planner=False),
            "d": arms(ablate_planner=False),
        }

    def test_only_episodes_carrying_both_arms_are_compared(self):
        found = audit.channel_influence(self.replays())
        self.assertEqual(found["matched_episodes"], 2)
        rates = {row["perturbation"]: row for row in found["arms"]}
        self.assertEqual(rates["ablate_planner"]["episodes"], 2)
        self.assertEqual(rates["ablate_planner"]["flip_rate"], 0.5)
        self.assertEqual(rates["ablate_worker_control"]["flip_rate"], 0.5)

    def test_the_unmatched_pool_would_have_said_something_else(self):
        """Over every episode the planner arm ran on it flips 1 of 4; over the
        matched pair, 1 of 2. Same replays, different comparison."""
        found = audit.channel_influence(self.replays())
        planner = next(r for r in found["arms"] if r["perturbation"] == "ablate_planner")
        self.assertEqual((planner["flipped"], planner["episodes"]), (1, 2))

    def test_each_arm_is_labelled_with_the_channel_it_edits(self):
        rows = {r["perturbation"]: r["channel"] for r in audit.channel_influence(
            self.replays())["arms"]}
        self.assertEqual(rows["ablate_planner"], "planner")
        self.assertEqual(rows["ablate_worker_control"], "worker")

    def test_an_arm_nobody_ran_is_absent_rather_than_zero(self):
        names = {r["perturbation"] for r in audit.channel_influence(self.replays())["arms"]}
        self.assertNotIn("swap_evidence", names)
