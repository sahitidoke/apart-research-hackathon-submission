"""Getting the dataset in, and refusing to quietly run without one.

The schema check earns its place here: several MuSiQue mirrors on the Hub drop
`question_decomposition` or flatten `paragraphs`, and against one of those this
package does not crash -- `records.usable` rejects every record, `select`
reports a tidy exclusion tally, and the run comes back empty and plausible.
"""
import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance import dataset
from orchestrator.provenance.batch import build_parser, resolve_dataset

GOOD = {"id": "2hop__1_2", "question": "q", "answer": "a", "answer_aliases": [],
        "paragraphs": [{"idx": 0, "title": "t", "paragraph_text": "x",
                        "is_supporting": True}],
        "question_decomposition": [{"id": 1, "question": "s >> r", "answer": "a",
                                    "paragraph_support_idx": 0}]}


class Spec(unittest.TestCase):
    def test_a_hub_spec_defaults_its_config_and_split(self):
        self.assertEqual(dataset.parse_source("hf:bdsaglam/musique"),
                         ("hub", ("bdsaglam/musique", "answerable", "validation")))
        self.assertEqual(dataset.parse_source("hf:a/b:full:train"),
                         ("hub", ("a/b", "full", "train")))

    def test_anything_without_the_prefix_is_a_path(self):
        self.assertEqual(dataset.parse_source("/root/musique.jsonl"),
                         ("file", "/root/musique.jsonl"))

    def test_a_hub_spec_naming_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            dataset.parse_source("hf:")


class Schema(unittest.TestCase):
    def test_the_original_release_schema_passes(self):
        self.assertEqual(dataset.problems(GOOD), [])

    def test_a_mirror_missing_the_decomposition_is_named_not_silently_empty(self):
        flattened = {k: v for k, v in GOOD.items() if k != "question_decomposition"}
        self.assertIn("question_decomposition", dataset.problems(flattened))
        with self.assertRaises(ValueError) as caught:
            dataset.check([flattened], "hf:some/mirror")
        self.assertIn("question_decomposition", str(caught.exception))
        self.assertIn("bdsaglam/musique", str(caught.exception))

    def test_a_mirror_without_support_labels_is_caught(self):
        stripped = json.loads(json.dumps(GOOD))
        del stripped["paragraphs"][0]["is_supporting"]
        self.assertIn("paragraphs[].is_supporting", dataset.problems(stripped))

    def test_an_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            dataset.check([], "hf:some/mirror")


class Files(unittest.TestCase):
    def test_a_jsonl_export_loads_and_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "musique.jsonl"
            path.write_text(json.dumps(GOOD) + "\n\n")
            self.assertEqual(dataset.load(str(path)), [GOOD])

    def test_a_broken_line_names_its_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "musique.jsonl"
            path.write_text(json.dumps(GOOD) + "\n{not json\n")
            with self.assertRaises(ValueError) as caught:
                dataset.load(str(path))
            self.assertIn("line 2", str(caught.exception))


class NoSilentFixture(unittest.TestCase):
    """A default of three invented questions is how a smoke test gets reported
    as a run: every command still works and the numbers look like numbers."""

    def parse(self, *argv):
        return build_parser().parse_args(["--output", "x", *argv])

    def test_a_run_without_a_dataset_is_refused_and_told_what_to_pass(self):
        with self.assertRaises(SystemExit) as caught:
            resolve_dataset(self.parse())
        self.assertIn(dataset.DEFAULT_SOURCE, str(caught.exception))

    def test_the_fixture_has_to_be_asked_for_by_name(self):
        self.assertTrue(resolve_dataset(self.parse("--smoke")).endswith("smoke.jsonl"))

    def test_a_dataset_is_passed_through_untouched(self):
        self.assertEqual(resolve_dataset(self.parse("--dataset", "hf:a/b")), "hf:a/b")

    def test_asking_for_both_is_refused_rather_than_silently_ordered(self):
        with self.assertRaises(SystemExit):
            resolve_dataset(self.parse("--dataset", "hf:a/b", "--smoke"))


if __name__ == "__main__":
    unittest.main()
