"""Record selection and the paragraph deal. No model, no dataset."""
import json
import random
import unittest
from pathlib import Path

from orchestrator.provenance import records, seed

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smoke.jsonl"


def load():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


class Hops(unittest.TestCase):
    def test_hop_count_from_id(self):
        self.assertEqual(records.hop_count({"id": "2hop__a"}), 2)
        self.assertEqual(records.hop_count({"id": "3hop1__a"}), 3)
        self.assertEqual(records.hop_count({"id": "4hop2__a"}), 4)

    def test_unparseable_id_raises(self):
        for bad in ("", "hop__a", "1hop__a", "musique-42"):
            with self.assertRaises(ValueError):
                records.hop_count({"id": bad})

    def test_final_hop_is_never_offered_for_corruption(self):
        for record in load():
            steps = record["question_decomposition"]
            offered = {hop["index"] for hop in records.bridge_hops(record)}
            self.assertNotIn(len(steps) - 1, offered)
            self.assertTrue(offered)

    def test_placeholder_answers_are_dropped(self):
        record = {"id": "3hop1__x", "answer": "z", "paragraphs": [{}, {}, {}],
                  "question_decomposition": [
                      {"answer": "#2", "paragraph_support_idx": 0},
                      {"answer": "real", "paragraph_support_idx": 1},
                      {"answer": "z", "paragraph_support_idx": 2}]}
        self.assertEqual([hop["answer"] for hop in records.bridge_hops(record)], ["real"])


class Deal(unittest.TestCase):
    def test_the_cohort_is_two_workers_at_every_hop_count(self):
        """Cohort size is held fixed so that hop count, the axis every rate is
        stratified by, is not also moving the number of agents."""
        for record in load():
            self.assertEqual(records.worker_count(record), 2)
            self.assertEqual(len(records.deal(record)), 2)

    def test_each_worker_holds_evidence_and_neither_holds_all_of_it(self):
        for record in load():
            assignment = records.deal(record)
            supports = set(records.supporting_positions(record))
            for positions in assignment.values():
                held = supports.intersection(positions)
                self.assertTrue(held)
                self.assertNotEqual(held, supports)

    def test_deal_partitions_every_paragraph_exactly_once(self):
        for record in load():
            dealt = [p for positions in records.deal(record).values() for p in positions]
            self.assertEqual(sorted(dealt), list(range(len(record["paragraphs"]))))

    def test_holder_finds_the_worker_given_a_paragraph(self):
        record = load()[1]
        for position in range(len(record["paragraphs"])):
            holder = records.holder(record, position)
            self.assertIn(position, records.deal(record)[holder])


class RelationStyleHops(unittest.TestCase):
    """Real MuSiQue writes its decomposition as `subject >> relation`, not as
    English, and every step after the first refers back with "#1". The bundled
    fixture uses the English form, so this pins the shape the dataset actually
    has -- written by hand rather than vendored, so the suite still needs no
    download and no licence.
    """

    RECORD = {
        "id": "4hop1__a_b_c_d", "question": "What is the capital?", "answer": "Green Bay",
        "paragraphs": [
            {"title": "Western Islands", "paragraph_text": "Its headquarters are in Appleton.",
             "is_supporting": True},
            {"title": "Appleton", "paragraph_text": "Appleton is in Outagamie County.",
             "is_supporting": True},
            {"title": "Outagamie County", "paragraph_text": "It shares a border with Brown County.",
             "is_supporting": True},
            {"title": "Brown County", "paragraph_text": "Its capital is Green Bay.",
             "is_supporting": True},
            # Real records carry ~16 distractors alongside their supports, and
            # a corruption is drawn from their titles -- with none, there is
            # nothing plausible to swap a bridge answer for.
            {"title": "Winnebago County", "paragraph_text": "A neighbouring county.",
             "is_supporting": False},
            {"title": "Fox Valley", "paragraph_text": "A region of Wisconsin.",
             "is_supporting": False}],
        "question_decomposition": [
            {"question": "Western Islands >> headquarters location", "answer": "Appleton",
             "paragraph_support_idx": 0},
            {"question": "#1 >> located in the administrative territorial entity",
             "answer": "Outagamie County", "paragraph_support_idx": 1},
            {"question": "#2 >> shares border with", "answer": "Brown County",
             "paragraph_support_idx": 2},
            {"question": "#3 >> capital", "answer": "Green Bay", "paragraph_support_idx": 3}]}

    def test_every_bridge_hop_becomes_a_standalone_query(self):
        """Unresolved, only the first hop of a chain could ever be probed --
        and on this dataset that is most of every chain left unscreened while
        the audit reports it screened."""
        from orchestrator.provenance.episode import probe_hops
        asked = [hop["question"] for hop in probe_hops(self.RECORD)]
        self.assertEqual(asked, [
            "Western Islands >> headquarters location",
            "Appleton >> located in the administrative territorial entity",
            "Outagamie County >> shares border with"])

    def test_the_reference_names_the_answer_it_points_at(self):
        self.assertEqual(
            records.resolve_question(self.RECORD, "#3 >> capital"),
            "Brown County >> capital")

    def test_a_reference_past_the_end_resolves_to_nothing(self):
        self.assertIsNone(records.resolve_question(self.RECORD, "#9 >> capital"))

    def test_the_record_is_usable_and_corruptible(self):
        self.assertEqual(records.usable(self.RECORD), (True, None))
        label = seed.corrupt(self.RECORD, random.Random(0))
        self.assertIsNotNone(label)
        self.assertLess(label["hop"], 3)


class Selection(unittest.TestCase):
    def test_selection_is_stable_and_extends_rather_than_resamples(self):
        pool = load() * 4
        pool = [{**record, "id": f"{record['id']}_{index}"} for index, record in enumerate(pool)]
        one, _ = records.select(pool, [2], 2, seed=7)
        two, _ = records.select(pool, [2], 3, seed=7)
        self.assertEqual([r["id"] for r in one], [r["id"] for r in two][:2])

    def test_unusable_records_are_counted_not_dropped_silently(self):
        pool = load() + [{"id": "bogus", "question": "q", "answer": "a", "paragraphs": []}]
        _, excluded = records.select(pool, [2], 1, seed=0)
        self.assertEqual(excluded, {"unparseable_id": 1})

    def test_asking_for_more_than_exists_raises(self):
        with self.assertRaises(ValueError):
            records.select(load(), [2], 5, seed=0)

    def test_episode_seed_is_stable_across_processes(self):
        record = load()[0]
        self.assertEqual(records.episode_seed(0, record, "planner"),
                         records.episode_seed(0, record, "planner"))
        self.assertNotEqual(records.episode_seed(0, record, "planner"),
                            records.episode_seed(0, record, "flat"))


class Corruption(unittest.TestCase):
    def test_corruption_labels_the_worker_holding_the_contradicting_paragraph(self):
        for record in load():
            label = seed.corrupt(record, random.Random(3))
            self.assertIsNotNone(label)
            self.assertEqual(label["correcting_worker"],
                             records.holder(record, label["support"]))
            self.assertIn(label["support"], records.deal(record)[label["correcting_worker"]])

    def test_corruption_never_equals_the_gold_hop_or_the_final_answer(self):
        for record in load():
            for attempt in range(12):
                label = seed.corrupt(record, random.Random(attempt))
                self.assertNotEqual(label["corrupted"].lower(), label["gold"].lower())
                self.assertNotEqual(label["corrupted"].lower(), record["answer"].lower())

    def test_corruption_never_targets_the_final_hop(self):
        for record in load():
            label = seed.corrupt(record, random.Random(11))
            self.assertLess(label["hop"], len(record["question_decomposition"]) - 1)

    def test_the_correcting_worker_matches_the_cohort_the_episode_will_run(self):
        """A label naming a worker from a differently-sized cohort would name
        an agent that was dealt something else, and the deference rate is
        conditioned on that name."""
        record = load()[2]
        label = seed.corrupt(record, random.Random(5), agents=3)
        self.assertEqual(label["correcting_worker"],
                         records.holder(record, label["support"], agents=3))
        self.assertIn(label["correcting_worker"], records.workers(record, 3))

    def test_no_usable_hop_returns_none_rather_than_guessing(self):
        record = {"id": "2hop__x", "question": "q", "answer": "a",
                  "paragraphs": [{"title": "t", "paragraph_text": "x", "is_supporting": True}],
                  "question_decomposition": []}
        self.assertIsNone(seed.corrupt(record, random.Random(0)))


if __name__ == "__main__":
    unittest.main()
