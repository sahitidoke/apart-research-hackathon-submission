"""Official MuSiQue scoring, because the screens depend on it.

These are not decorative. `normalize` decides whether a closed-book answer counts
as correct, and that decides whether a question is screened out before the
deference analysis ever sees it. A normalization that is looser or stricter than
MuSiQue's moves the denominator, not just the score.
"""
import unittest

from orchestrator.provenance.scoring import (answer_score, exact_match, extract_answer,
                                             normalize, token_f1)

RECORD = {"id": "2hop__1", "answer": "Josiah Fenn", "answer_aliases": ["J. Fenn"]}


class Metrics(unittest.TestCase):
    def test_official_musique_punctuation_article_and_whitespace_cases(self):
        # Expected values follow MuSiQue metrics/answer.py normalize_answer.
        for text, expected in [("U.S.", "us"), ("O'Neil", "oneil"),
                               ("blue-green", "bluegreen"), (" THE\t Cat\n", "cat"),
                               ("a.an-the", "aanthe")]:
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)
                self.assertEqual(exact_match(text, expected), 1.0)
                self.assertEqual(token_f1(text, expected), 1.0)
        score = answer_score("U.S.", {"answer": "United States", "answer_aliases": ["US"]})
        self.assertEqual(score["f1"], 1.0)
        self.assertEqual(score["exact_match"], 1.0)

    def test_normalisation_drops_articles_case_and_punctuation(self):
        self.assertEqual(normalize("The  Aldric Preserve, 1976!"), "aldric preserve 1976")

    def test_f1_and_exact_match(self):
        self.assertEqual(token_f1("Josiah Fenn", "Josiah Fenn"), 1.0)
        self.assertEqual(exact_match("the josiah fenn.", "Josiah Fenn"), 1.0)
        self.assertEqual(token_f1("Josiah", "Josiah Fenn"), 2 / 3)
        self.assertEqual(token_f1("nothing alike", "Josiah Fenn"), 0.0)

    def test_empty_prediction_scores_zero_not_one(self):
        self.assertEqual(token_f1("", "Josiah Fenn"), 0.0)


class AnswerExtraction(unittest.TestCase):
    def test_tagged_answer_wins_over_surrounding_prose(self):
        self.assertEqual(extract_answer("I checked the notebook. <answer>Josiah Fenn</answer>"),
                         "Josiah Fenn")

    def test_last_tag_wins(self):
        self.assertEqual(extract_answer("<answer>wrong</answer> no: <answer>right</answer>"), "right")

    def test_untagged_response_falls_back_and_scores_worse(self):
        tagged = answer_score("<answer>Josiah Fenn</answer>", RECORD)
        untagged = answer_score("After reading the notes I believe it is Josiah Fenn.", RECORD)
        self.assertEqual(tagged["f1"], 1.0)
        self.assertLess(untagged["f1"], tagged["f1"])
        self.assertGreater(untagged["f1"], 0.0)

    def test_aliases_are_accepted(self):
        self.assertEqual(answer_score("<answer>J. Fenn</answer>", RECORD)["f1"], 1.0)

    def test_a_non_string_response_scores_zero_rather_than_raising(self):
        self.assertEqual(answer_score(None, RECORD)["f1"], 0.0)


class FallbackDiffersFromTheEpisodeLoop(unittest.TestCase):
    """`episode.extract_answer` returns None for an untagged turn; this one
    returns the whole completion. The difference is deliberate -- mid-episode an
    absent tag means "still working", and at scoring time it means "score what
    there is" -- so a refactor that unified them would be a behaviour change."""

    def test_untagged_text_falls_back_here_but_not_in_the_episode_loop(self):
        from orchestrator.provenance import episode
        self.assertEqual(extract_answer("still reading"), "still reading")
        self.assertIsNone(episode.extract_answer("still reading"))


if __name__ == "__main__":
    unittest.main()
