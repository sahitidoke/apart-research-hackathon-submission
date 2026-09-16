"""The rendered page. No browser, so these check the contract, not the pixels.

The one that matters is `test_a_single_class_stratum_says_so`: an AUC that
cannot be computed must look like a stated absence, because an empty axis and a
curve fitted to one class both read as results.
"""
import unittest

from orchestrator.provenance import audit, plots


def rows(pairs, arm=("persuasive", "standard"), hops=2):
    return [{"directory": f"q{i}", "record_id": f"{hops}hop__q{i}", "hops": hops,
             "condition": "planner", "eligible": True, "label": label, "score": score,
             "planner_style": arm[0], "worker_style": arm[1],
             "workers": {"agent-1": {"holds_swapped_evidence": True, "score": score,
                                     "planner_sensitivity": score, "evidence_used": 0.0,
                                     "inert": False}}}
            for i, (score, label) in enumerate(pairs)]


def result(episodes):
    return {"episodes": episodes,
            "roc": audit.roc([(r["score"], r["label"]) for r in episodes]),
            "roc_strata": audit.roc_strata(episodes),
            "competence": {"questions": 9, "retained": len(episodes), "screens_not_run": []},
            "health": {"diverged_rate": 0.0}}


class Page(unittest.TestCase):
    def setUp(self):
        self.episodes = rows([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)])
        self.html = plots.render(result(self.episodes))

    def test_it_is_one_self_contained_file(self):
        self.assertTrue(self.html.startswith("<!doctype html>"))
        for external in ("<script src", "<link ", "http://", "https://"):
            self.assertNotIn(external, self.html)

    def test_every_stratum_gets_its_own_panel(self):
        for name in ("Overall", "2-hop", "persuasive / standard",
                     "Evidence holder, per-agent score"):
            self.assertIn(name, self.html)

    def test_the_curve_is_drawn_and_the_auc_stated(self):
        self.assertIn('class="curve"', self.html)
        self.assertIn("AUC 1", self.html)

    def test_every_chart_has_a_table_view(self):
        """Tooltips enhance, never gate: a value a reader cannot reach without
        hovering is a value some readers cannot reach at all."""
        self.assertEqual(self.html.count("<summary>Table</summary>"),
                         self.html.count("<svg "))

    def test_the_two_class_chart_carries_a_legend(self):
        self.assertIn('class="legend"', self.html)
        self.assertIn("Deferred", self.html)
        self.assertIn("Derived", self.html)

    def test_dark_mode_is_declared_for_both_the_os_and_the_toggle(self):
        self.assertIn("@media (prefers-color-scheme:dark)", self.html)
        self.assertIn(":root[data-theme=dark]", self.html)


class Absences(unittest.TestCase):
    def test_a_single_class_stratum_says_so(self):
        html = plots.render(result(rows([(0.9, 1), (0.8, 1)])))
        self.assertIn("not computable", html)
        self.assertIn("2 deferred, 0 derived", html)
        self.assertNotIn('class="curve"', html)

    def test_an_empty_audit_renders_rather_than_crashing(self):
        html = plots.render({"episodes": [], "roc": audit.roc([]), "roc_strata": [],
                             "competence": {"questions": 0, "screens_not_run": []},
                             "health": {"diverged_rate": None}})
        self.assertIn("no AUC to compare", html)
        self.assertIn("No labelled, eligible episode", html)

    def test_an_ungated_run_is_flagged_on_the_page_itself(self):
        found = result(rows([(0.9, 1), (0.1, 0)]))
        found["competence"]["screens_not_run"] = ["closed_book", "hop_probe"]
        html = plots.render(found)
        self.assertIn("not fully gated", html)
        self.assertIn("closed_book, hop_probe", html)

    def test_a_record_id_cannot_inject_markup(self):
        episodes = rows([(0.9, 1), (0.1, 0)])
        episodes[0]["record_id"] = '<script>alert("x")</script>'
        html = plots.render(result(episodes))
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
