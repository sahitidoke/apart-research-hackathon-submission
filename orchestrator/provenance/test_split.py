"""The deal is the reason the bus carries anything. Check it actually splits.

If one agent ends up holding every supporting paragraph, the collective has no
reason to talk and an answer proves nothing about message provenance -- so these
are load-bearing for the experiment, not just for the function.
"""
import json
import unittest

from orchestrator.provenance.split import agent_names, split_positions

RECORD = {
    "id": "2hop__test",
    "question": "Who founded the institution Renata Kell later directed, and when did it open?",
    "answer": "Josiah Fenn",
    "paragraphs": [
        {"title": "Renata Kell", "paragraph_text": "Kell became the second director of the Aldric Preserve in 1991.", "is_supporting": True},
        {"title": "Aldric Preserve", "paragraph_text": "The Aldric Preserve opened in 1976 under Josiah Fenn.", "is_supporting": True},
        {"title": "Larch Institute", "paragraph_text": "An unrelated institute in Bergen.", "is_supporting": False},
        {"title": "Fenn Point", "paragraph_text": "A headland with no connection to the Preserve.", "is_supporting": False},
    ],
}


class Names(unittest.TestCase):
    def test_names_are_one_indexed_and_ordered(self):
        self.assertEqual(agent_names(3), ["agent-1", "agent-2", "agent-3"])

    def test_zero_agents_is_an_empty_cohort_not_an_error(self):
        self.assertEqual(agent_names(0), [])


class Positions(unittest.TestCase):
    def test_supporting_paragraphs_are_dealt_round_robin(self):
        assignment = split_positions(RECORD, 2)
        self.assertEqual(assignment["agent-1"][0], 0)
        self.assertEqual(assignment["agent-2"][0], 1)

    def test_no_agent_holds_every_supporting_paragraph(self):
        assignment = split_positions(RECORD, 2)
        supporting = {i for i, p in enumerate(RECORD["paragraphs"]) if p["is_supporting"]}
        for positions in assignment.values():
            self.assertTrue(supporting - set(positions), "an agent could answer alone")

    def test_every_paragraph_is_assigned_exactly_once(self):
        assignment = split_positions(RECORD, 2)
        assigned = [p for positions in assignment.values() for p in positions]
        self.assertEqual(sorted(assigned), list(range(len(RECORD["paragraphs"]))))

    def test_a_record_with_too_few_supports_is_refused(self):
        """One support and two agents means one agent is a bystander -- that is
        not a cooperative task and must not enter the run."""
        single = {**RECORD, "paragraphs": [
            {**RECORD["paragraphs"][0]},
            {**RECORD["paragraphs"][1], "is_supporting": False}]}
        with self.assertRaises(ValueError):
            split_positions(single, 2)

    def test_one_agent_is_refused(self):
        with self.assertRaises(ValueError):
            split_positions(RECORD, 1)

    def test_the_deal_returns_positions_only_and_carries_no_labels(self):
        """`is_supporting` decides the deal and then disappears: the return value
        is integers, so a support label cannot ride along into a prompt."""
        body = json.dumps(split_positions(RECORD, 2))
        self.assertNotIn("is_supporting", body)
        self.assertNotIn("Josiah", body)


if __name__ == "__main__":
    unittest.main()
