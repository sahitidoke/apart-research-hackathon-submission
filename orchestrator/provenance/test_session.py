"""Sessions: the schedule, the blocks, and the deal that inverts the screens.

The test that matters most here is `test_the_isolated_screen_inverts...`:
with the expert deal, the old screen would exclude every question in the run
and the report would come back tidy and empty.
"""
import random
import unittest

from orchestrator.provenance import audit, records, seed, session
from orchestrator.provenance.batch import session_plan
from orchestrator.provenance.test_records import load


class Schedule(unittest.TestCase):
    def test_the_declining_arm_steps_evenly_from_start_to_end(self):
        self.assertEqual(session.steps(100, 20, 5), [100, 80, 60, 40, 20])

    def test_the_flat_arm_sits_at_the_midpoint_for_the_same_session_total(self):
        declining = session.steps(100, 20, 5)
        flat = session.steps(100, 20, 5, shape="flat")
        self.assertEqual(set(flat), {60})
        self.assertEqual(sum(flat), sum(declining))

    def test_a_schedule_is_calibrated_from_what_a_run_really_spent(self):
        """Levels as percentiles of observed spend, not round numbers somebody
        picked: that is what makes "the budget is tight" a fact about this
        model on this task."""
        spent = list(range(1000, 11000, 1000))
        levels = session.levels_from_usage(spent, 4)
        self.assertEqual(levels[0], session.percentile(spent, 0.90))
        self.assertEqual(levels[-1], session.percentile(spent, 0.25))
        self.assertEqual(sorted(levels, reverse=True), levels)

    def test_the_answer_allowance_never_declines_to_nothing(self):
        """An episode that cannot emit an answer is no measurement, and a
        schedule that manufactures no-answer episodes at its tight end looks
        exactly like one that manufactures deference."""
        rounds = session.budgets(16000, 12000, 8, 6, start=1.0, end=0.0)
        self.assertGreaterEqual(rounds[-1]["answer_tokens"], 12000 * 0.5)
        self.assertGreaterEqual(rounds[-1]["max_steps"], 2)

    def test_budgets_decline_across_the_session(self):
        rounds = session.budgets(16000, 12000, 8, 5)
        caps = [r["collaboration_tokens"] for r in rounds]
        self.assertEqual(sorted(caps, reverse=True), caps)
        self.assertLess(caps[-1], caps[0])

    def test_a_schedule_of_fractions_is_not_floored_into_an_on_off_switch(self):
        """`steps` is called with token counts in one place and with fractions
        of a budget in another. Rounding inside it turned the second into
        1, 1, ... 0 -- full budget for most of the session and one token for
        the rest, which is not a declining schedule."""
        caps = [r["collaboration_tokens"] for r in session.budgets(16000, 12000, 8, 10)]
        self.assertEqual(len(set(caps)), 10)
        self.assertNotIn(1, caps)
        self.assertEqual(caps[0], 16000)
        self.assertEqual(caps[-1], round(16000 * session.DEFAULT_END))

    def test_both_arms_spend_the_same_session_total(self):
        """The flat arm is the control. If it spent a different total, a
        difference between the arms would be a difference in total budget
        rather than in whether the budget moves."""
        declining = session.budgets(16000, 12000, 8, 10)
        flat = session.budgets(16000, 12000, 8, 10, shape="flat")
        self.assertEqual(sum(r["collaboration_tokens"] for r in declining),
                         sum(r["collaboration_tokens"] for r in flat))

    def test_the_turn_cap_is_the_time_pressure_and_declines_too(self):
        """Wall-clock is not a controllable quantity for inference, so turns
        are the analog and they shrink with the tokens."""
        turns = [r["max_steps"] for r in session.budgets(16000, 12000, 8, 10)]
        self.assertEqual(sorted(turns, reverse=True), turns)
        self.assertLess(turns[-1], turns[0])
        self.assertGreaterEqual(min(turns), 2)


class Sequence(unittest.TestCase):
    def test_blocks_hand_the_evidence_to_one_worker_at_a_time(self):
        plan = session.sequence(load() * 4, rounds=4, block=2, base_seed=0, cohort=0)
        self.assertEqual([slot["expert"] for slot in plan],
                         ["agent-1", "agent-1", "agent-2", "agent-2"])
        self.assertEqual([slot["block"] for slot in plan], [0, 0, 1, 1])

    def test_block_order_is_counterbalanced_across_cohorts(self):
        """Otherwise agent-1 is always the expert early, and every "the expert
        deferred less in round one" reading is also "agent-1 went first"."""
        first = session.sequence(load() * 4, 4, 2, base_seed=0, cohort=0)
        second = session.sequence(load() * 4, 4, 2, base_seed=0, cohort=1)
        self.assertEqual(first[0]["expert"], "agent-1")
        self.assertEqual(second[0]["expert"], "agent-2")

    def test_question_order_is_randomised_per_cohort(self):
        pool = [{**r, "id": f"{r['id']}_{i}"} for i in range(4) for r in load()]
        first = [s["record_id"] for s in session.sequence(pool, 6, 3, 0, cohort=0)]
        second = [s["record_id"] for s in session.sequence(pool, 6, 3, 0, cohort=1)]
        self.assertNotEqual(first, second)
        self.assertEqual(first, [s["record_id"] for s in
                                 session.sequence(pool, 6, 3, 0, cohort=0)])

    def test_evidence_overlap_between_neighbours_is_reported(self):
        found = session.overlap(load())
        self.assertEqual(found["pairs"], 2)
        self.assertIn("mean_shared_titles", found)


class ExpertDeal(unittest.TestCase):
    def test_the_expert_holds_every_support_and_the_bystander_none(self):
        for record in load():
            assignment = records.expert_deal(record, "agent-2")
            supports = set(records.supporting_positions(record))
            self.assertTrue(supports <= set(assignment["agent-2"]))
            self.assertFalse(supports & set(assignment["agent-1"]))

    def test_every_paragraph_is_still_dealt_exactly_once(self):
        for record in load():
            assignment = records.expert_deal(record, "agent-1")
            dealt = sorted(p for held in assignment.values() for p in held)
            self.assertEqual(dealt, list(range(len(record["paragraphs"]))))

    def test_the_expert_also_holds_distractors(self):
        """A prompt of nothing but supporting paragraphs is a much easier task
        than anyone else here is given, and would flatter the expert's accuracy
        for a reason unrelated to deference."""
        record = load()[2]
        assignment = records.expert_deal(record, "agent-1")
        supports = set(records.supporting_positions(record))
        self.assertTrue(set(assignment["agent-1"]) - supports)

    def test_naming_a_worker_outside_the_cohort_is_refused(self):
        with self.assertRaises(ValueError):
            records.expert_deal(load()[0], "agent-9")


class ScreenInversion(unittest.TestCase):
    """With the evidence split, a question one worker solves alone never needed
    the pair. With one worker holding the whole chain, a question that worker
    *cannot* solve alone is the useless one -- an expert that yields was then
    guessing, not deferring. Getting this backwards excludes every question and
    reports an empty run rather than an error.
    """

    def episodes(self, expert, isolated_solved):
        rows = []
        for condition, exact in (("closed_book", 0.0), ("solo", 1.0),
                                 ("isolated", 1.0 if isolated_solved else 0.0)):
            rows.append({"directory": f"q1-{condition}", "record_id": "q1", "hops": 2,
                         "condition": condition, "exact_match": exact, "f1": exact,
                         "status": "complete", "seeded_error": None, "expert": expert,
                         "planner_style": "persuasive", "worker_style": "standard"})
        return rows

    def test_under_the_expert_deal_an_expert_who_cannot_answer_alone_is_excluded(self):
        retained, table = audit.retention(self.episodes("agent-1", isolated_solved=False))
        self.assertEqual(retained, set())
        self.assertEqual(table["expert_could_not_answer_alone"], 1)
        self.assertEqual(table["deal"], "expert")

    def test_under_the_expert_deal_an_expert_who_can_answer_alone_is_kept(self):
        retained, table = audit.retention(self.episodes("agent-1", isolated_solved=True))
        self.assertEqual(retained, {"q1"})
        self.assertEqual(table["deal"], "expert")

    def test_under_the_split_deal_the_rule_is_the_other_way_round(self):
        self.assertEqual(audit.retention(self.episodes(None, isolated_solved=True))[0],
                         set())
        self.assertEqual(audit.retention(self.episodes(None, isolated_solved=False))[0],
                         {"q1"})


class Planning(unittest.TestCase):
    def plan(self, **kwargs):
        pool = [{**r, "id": f"{r['id']}_{i}"} for i in range(4) for r in load()]
        return session_plan(pool, ("planner",), True, 0, "persuasive", "standard",
                            rounds=4, block=2, **kwargs)

    def test_each_episode_carries_its_round_budget_and_expert(self):
        planned = self.plan()
        rounds = {item["session"]["round"] for item in planned}
        self.assertEqual(rounds, {0, 1, 2, 3})
        caps = {item["session"]["round"]: item["collaboration_tokens"] for item in planned}
        self.assertGreater(caps[0], caps[3])
        self.assertTrue(all(item["expert"] in ("agent-1", "agent-2") for item in planned))

    def test_the_flat_arm_holds_one_cap_for_the_whole_session(self):
        caps = {item["collaboration_tokens"] for item in self.plan(shape="flat")}
        self.assertEqual(len(caps), 1)

    def test_cohorts_are_independent_sessions(self):
        planned = self.plan(cohorts=2)
        self.assertEqual({item["session"]["cohort"] for item in planned}, {0, 1})
        directories = {(item["session"]["cohort"], item["session"]["round"],
                        item["record_id"], item["condition"], bool(item["seeded_error"]))
                       for item in planned}
        self.assertEqual(len(directories), len(planned))


class CorruptionUnderTheExpertDeal(unittest.TestCase):
    """`correcting_worker` is the ground truth every deference rate is
    conditioned on, and `seed.corrupt` used to resolve it through
    `records.holder`, which re-derives the *round-robin split* deal and takes no
    expert. Under the expert deal one worker holds every support, so the
    correcting worker is always the expert -- but the split deal named the
    bystander, who holds no supporting paragraph at all, in about half of
    episodes.

    Nothing crashed. The run completed, the tables were well formed, and the
    labels were attached to the wrong agent. That is the same failure shape as a
    flattened dataset mirror: empty and plausible.
    """

    def corrupted(self, record, expert):
        assignment = records.expert_deal(record, expert)
        return seed.corrupt(record, random.Random(0), [], 2, assignment=assignment)

    def test_the_correcting_worker_is_the_expert(self):
        for record in load():
            for expert in ("agent-1", "agent-2"):
                error = self.corrupted(record, expert)
                self.assertIsNotNone(error, record["id"])
                self.assertEqual(error["correcting_worker"], expert,
                                 f"{record['id']} with {expert} as expert")

    def test_the_named_worker_actually_holds_the_contradicting_paragraph(self):
        """The point of the label. A worker dealt only distractors cannot
        contradict anything, so naming it makes the episode unlabellable while
        still counting it."""
        for record in load():
            for expert in ("agent-1", "agent-2"):
                assignment = records.expert_deal(record, expert)
                error = self.corrupted(record, expert)
                self.assertIn(error["support"], assignment[error["correcting_worker"]])

    def test_the_bystander_is_never_named(self):
        for record in load():
            for expert, bystander in (("agent-1", "agent-2"), ("agent-2", "agent-1")):
                self.assertNotEqual(self.corrupted(record, expert)["correcting_worker"],
                                    bystander)

    def test_holder_and_share_honour_an_explicit_assignment(self):
        record = load()[0]
        assignment = records.expert_deal(record, "agent-2")
        support = records.supporting_positions(record)[0]
        self.assertEqual(records.holder(record, support, assignment=assignment), "agent-2")
        held = records.share(record, support, assignment=assignment)
        self.assertEqual(held[0], support)
        self.assertTrue(set(held) <= set(assignment["agent-2"]))

    def test_the_swap_target_is_searched_in_the_experts_share(self):
        """`locatable` decides whether a hop may be corrupted at all. Run
        against the wrong share it rejects hops the expert could have been
        swapped on, and accepts ones it cannot."""
        record = load()[0]
        assignment = records.expert_deal(record, "agent-2")
        error = self.corrupted(record, "agent-2")
        self.assertTrue(records.locatable(
            record, error["gold"],
            records.share(record, error["support"], assignment=assignment)))

    def test_the_split_deal_is_unchanged(self):
        """The expert deal is a second experiment, not a replacement. With no
        assignment the round-robin answer has to be exactly what it always was."""
        for record in load():
            error = seed.corrupt(record, random.Random(0), [], 2)
            self.assertEqual(error["correcting_worker"],
                             records.holder(record, error["support"], 2))

    def test_a_session_plan_names_the_expert_of_the_slot_it_planted_in(self):
        pool = [{**r, "id": f"{r['id']}_{i}"} for i in range(4) for r in load()]
        planned = session_plan(pool, ("planner",), True, 0, "persuasive", "standard",
                               rounds=4, block=2)
        seeded = [item for item in planned if item["seeded_error"]]
        self.assertTrue(seeded)
        for item in seeded:
            self.assertEqual(item["seeded_error"]["correcting_worker"], item["expert"])


class InductionArms(unittest.TestCase):
    """Deference can be induced -- the plan says so, once the pilot measures it
    near zero. What it may not do is move the default, because that is the arm
    whose rate is reported as propensity."""

    def test_the_default_arm_says_nothing_about_deferring(self):
        from orchestrator.provenance import roles
        natural = roles.worker_prompt(load()[0], "agent-1", [0], ["agent-2"], "standard")
        for phrase in ("defer", "take precedence", "scored together", "converge"):
            self.assertNotIn(phrase, natural)
        # It says the opposite, in fact.
        self.assertIn("say what your paragraphs", natural)

    def test_the_induced_arms_are_named_so_a_rate_cannot_pass_as_natural(self):
        from orchestrator.provenance import roles
        self.assertNotIn("standard", roles.INDUCED["worker"])
        self.assertNotIn("persuasive", roles.INDUCED["planner"])
        self.assertEqual(set(roles.INDUCED["worker"]), {"cooperative", "deferential"})
        self.assertEqual(set(roles.INDUCED["planner"]), {"directive"})

    def test_cooperative_prices_convergence_without_instructing_deference(self):
        """The middle rung is the interesting one: it never says the planner is
        right, it makes agreeing cheap and dissent costly, and whether
        deference follows is the measurement rather than the instruction."""
        from orchestrator.provenance import roles
        prompt = roles.worker_prompt(load()[0], "agent-1", [0], ["agent-2"], "cooperative")
        self.assertIn("scored together", prompt)
        self.assertNotIn("defer to", prompt)
        self.assertNotIn("take precedence", prompt)

    def test_every_style_still_builds_a_prompt(self):
        from orchestrator.provenance import roles
        for style in roles.WORKERS:
            self.assertTrue(roles.worker_prompt(load()[0], "agent-1", [0], ["agent-2"], style))
        for style in roles.PLANNERS:
            self.assertTrue(roles.planner_prompt(["agent-1", "agent-2"], style))


if __name__ == "__main__":
    unittest.main()
