"""Deterministic plumbing and boundary checks; no network or model calls."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser, MAX_TEXT, search_snippet
from orchestrator.simulated_web.runner import ModelResponse, Ollama, ScriptedClients, main, run_cohort

FIXTURE = Path(__file__).parent / "fixtures"


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.browser = Browser(json.loads((FIXTURE / "pages.json").read_text()), Path(self.temp.name) / "wiki.db")

    def tearDown(self):
        self.browser.close()
        self.temp.cleanup()

    def save(self, agent="a", text="Useful unique fact"):
        return self.browser.call(agent, "open", {"url": "https://wiki.test/save?" + urlencode(
            {"slug": "notes", "title": "Research", "text": text})})

    def test_shared_notes_and_audit_identity(self):
        self.assertIn("saved", self.save())
        result = self.browser.call("b", "search", {"query": "unique"})
        self.assertEqual(result["results"][0]["url"], "https://wiki.test/page/notes")
        page = self.browser.call("b", "open", {"url": result["results"][0]["url"]})
        self.assertEqual(page["text"], "Useful unique fact")
        self.assertEqual(self.browser.db.execute("SELECT agent FROM revisions").fetchone()[0], "a")
        self.assertEqual(self.browser.db.execute("SELECT count(*) FROM audit").fetchone()[0], 3)

    def test_bm25_rare_terms_length_and_frequency(self):
        pages = [{"url": f"https://docs.test/{i}", "title": "", "text": text}
                 for i, text in enumerate([
                     "common ordinary", "quasar", "common ordinary", "common ordinary",
                     "signal " + "padding " * 100, "signal", "signal signal signal",
                 ] + ["common ordinary"] * 20)]
        browser = Browser(pages, Path(self.temp.name) / "ranking.db")
        try:
            def urls(query):
                return [hit["url"] for hit in browser.search(query)["results"]]
            # Two ubiquitous matches must not beat a discriminative rare term.
            self.assertEqual(urls("common ordinary quasar")[0], "https://docs.test/1")
            ranked = urls("signal")
            self.assertLess(ranked.index("https://docs.test/5"), ranked.index("https://docs.test/4"))
            self.assertLess(ranked.index("https://docs.test/6"), ranked.index("https://docs.test/5"))
            self.assertEqual(urls("SIGNAL signal!!!"), ranked)
            self.assertEqual(urls("???"), [])
            self.assertEqual(urls("absentword"), [])
        finally:
            browser.close()

    def test_search_refreshes_replaced_wiki_notes(self):
        self.save(text="quasar")
        self.assertEqual(self.browser.search("quasar")["results"][0]["url"], "https://wiki.test/page/notes")
        self.save(text="pulsar")
        self.assertEqual(self.browser.search("quasar")["results"], [])
        self.assertEqual(self.browser.search("pulsar")["results"][0]["url"], "https://wiki.test/page/notes")

    def test_snippet_finds_late_passage_and_preserves_source(self):
        text = "Unrelated introduction. " * 30 + "The observatory opened in 1842 near Larch village. " + "Other material. " * 30
        snippet = search_snippet(text, {"observatory": 2, "1842": 3})
        self.assertIn("The observatory opened in 1842", snippet)
        self.assertIn(snippet, text)
        self.assertLessEqual(len(snippet), 300)

    def test_snippet_prefers_distinct_informative_terms(self):
        text = "common " * 90 + "Quasar and pulsar discoveries were announced together. " + "filler " * 70
        snippet = search_snippet(text, {"common": 0.1, "quasar": 2, "pulsar": 2})
        self.assertIn("Quasar and pulsar", snippet)
        self.assertEqual(snippet, search_snippet(text, {"pulsar": 2, "common": 0.1, "quasar": 2}))
        self.assertEqual(search_snippet(text, {"absent": 2}), text[:300])
        self.assertEqual(search_snippet("Short source.", {"source": 1}), "Short source.")

    def test_search_snippet_tracks_live_wiki_replacement(self):
        self.save(text="Background. " * 50 + "quasar evidence here")
        hit = self.browser.search("quasar")["results"][0]
        self.assertIn("quasar evidence here", hit["snippet"])
        self.save(text="Different background. " * 50 + "quasar revised evidence")
        updated = self.browser.search("quasar")["results"][0]
        self.assertEqual(hit["url"], updated["url"])
        self.assertIn("quasar revised evidence", updated["snippet"])

    def test_bad_urls_and_arguments(self):
        urls = ["file:///etc/passwd", "https://example.com", "https://wiki.test:443/", "https://user@wiki.test/",
                "https://wiki.test/#secret", "https://docs.test/../../etc/passwd", "https://wiki.test/page/%2e%2e",
                "https://wiki.test/save?slug=a&title=b&text=c&agent=other", "https://wiki.test/\n"]
        for url in urls:
            with self.subTest(url=url):
                self.assertIn("error", self.browser.call("a", "open", {"url": url}))
        for operation, args in [("shell", {}), ("open", {"url": [], "agent": "b"}), ("search", {"query": []}),
                                ("click", {"page_id": "p1", "link_id": True}), ("open", None)]:
            self.assertIn("error", self.browser.call("a", operation, args))

    def test_private_handles_and_trials(self):
        page = self.browser.call("a", "open", {"url": "https://wiki.test/"})
        self.assertIn("error", self.browser.call("b", "click", {"page_id": page["page_id"], "link_id": 1}))
        self.assertEqual(self.browser.call("a", "click", {"page_id": page["page_id"], "link_id": 1})["title"], "Edit a page")
        self.save()
        other = Browser([], Path(self.temp.name) / "other.db")
        try:
            self.assertIn("error", other.call("a", "open", {"url": "https://wiki.test/page/notes"}))
        finally:
            other.close()

    def test_bounds_and_reference_separation(self):
        self.assertIn("error", self.save(text="a" * (MAX_TEXT + 1)))
        self.assertIn("error", self.browser.call("a", "search", {"query": "a" * 501}))
        self.assertIn("error", self.browser.call("a", "open", {"url": "https://docs.test/task.json"}))
        for _ in range(100):
            self.assertIn("saved", self.save())
        self.assertIn("error", self.save())


class RunnerTests(unittest.TestCase):
    def assert_terminal(self, run_dir, result, agent_index=0):
        self.assertEqual(json.loads((run_dir / "results.json").read_text()), result)
        events = [json.loads(line) for line in (run_dir / "logs" / (result[agent_index]["agent"] + ".jsonl")).read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "final")
        for key, value in result[agent_index].items():
            self.assertEqual(events[-1][key], value)

    def test_empty_final_response_and_preserved_nonempty_answer(self):
        for response in [{}, {"content": ""}, {"content": " \n\t"}, {"content": " Answer \n"}]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                run_dir = Path(root) / "run"
                result = run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json",
                                    lambda agent, messages, timeout: response, agents=1)
                answer = response.get("content", "")
                self.assertEqual(result[0]["status"], "complete" if answer.strip() else "empty_response")
                self.assertEqual(result[0]["answer"], answer)
                self.assert_terminal(run_dir, result)

    def test_scripted_save_failure_does_not_signal_publication(self):
        for response in [{"error": "write rejected"}, {}, {"saved": "https://wiki.test/page/other"}]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                run_dir = Path(root) / "run"
                client = ScriptedClients()
                with patch.object(Browser, "call", return_value=response):
                    result = run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json", client, agents=1)
                self.assertEqual(result[0]["status"], "error")
                self.assertIn("Scripted publication failed", result[0]["error"])
                self.assertFalse(client.published.is_set())
                self.assert_terminal(run_dir, result)

    def test_scripted_readback_failure_is_terminal_error(self):
        original_call = Browser.call
        for response in [{"error": "page missing"}, {}, {"text": "wrong note"}]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                run_dir = Path(root) / "run"
                def intercept(browser, agent, operation, args):
                    if agent == "agent-2":
                        return response
                    return original_call(browser, agent, operation, args)
                with patch.object(Browser, "call", new=intercept):
                    result = run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json", ScriptedClients())
                self.assertEqual(result[0]["status"], "complete")
                self.assertEqual(result[1]["status"], "error")
                self.assertIn("Scripted retrieval failed", result[1]["error"])
                self.assert_terminal(run_dir, result, agent_index=1)

    def test_cli_exit_status_for_terminal_outcomes(self):
        for status in ["complete", "empty_response", "error"]:
            with self.subTest(status=status), patch("sys.argv", ["runner", "--dry-run", "--run-dir", "unused"]), \
                    patch("orchestrator.simulated_web.runner.run_cohort", return_value=[{"status": "complete"}, {"status": status}]), \
                    patch("builtins.print"):
                self.assertEqual(main(), 0 if status == "complete" else 1)

    def test_scripted_lifecycle_and_existing_run_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run"
            result = run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json", ScriptedClients())
            self.assertEqual([r["status"] for r in result], ["complete", "complete"])
            self.assertEqual(result[1]["answer"], "Mira Vale; 1978")
            events = [json.loads(line) for line in (run_dir / "logs" / "agent-2.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "initial")
            self.assertEqual(events[-1]["event"], "final")
            self.assertNotIn("reference_answer", json.dumps(events))
            self.assertTrue((run_dir / "settings.json").exists())
            self.assertTrue((run_dir / "web" / "wiki.sqlite3").exists())
            with self.assertRaises(FileExistsError):
                run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json", ScriptedClients())

    def test_malformed_responses_and_step_limit(self):
        for response, status in [(None, "error"), ({"content": 4}, "error"),
                                 ({"tool_calls": [{}] * 9}, "error"),
                                 ({"tool_calls": [{"function": {"name": "open", "arguments": "not-json"}}]}, "step_limit")]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                result = run_cohort(Path(root) / "run", FIXTURE / "pages.json", FIXTURE / "task.json",
                                    lambda agent, messages, timeout: response, agents=1, steps=1)
                self.assertEqual(result[0]["status"], status)

    def test_conversations_are_distinct(self):
        identities = []
        lock = threading.Lock()
        def client(agent, messages, timeout):
            with lock:
                identities.append(messages)
            return {"content": agent}
        with tempfile.TemporaryDirectory() as root:
            run_cohort(Path(root) / "run", FIXTURE / "pages.json", FIXTURE / "task.json", client)
        self.assertEqual(len({id(messages) for messages in identities}), 2)


class ModelMetadataTests(unittest.TestCase):
    def test_transport_generation_budget_and_metadata(self):
        message = {"content": "", "thinking": "Unfinished"}
        with patch("orchestrator.simulated_web.runner.http.client.HTTPConnection") as factory:
            connection = factory.return_value
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.read.return_value = json.dumps({
                "message": message, "done_reason": "length", "eval_count": 8192,
                "prompt_eval_count": 3000, "done": True
            }).encode()
            reply = Ollama("test", max_output_tokens=8192)("agent-1", [], 10)
            payload = json.loads(connection.request.call_args.args[2])
            self.assertEqual(payload["options"]["num_predict"], 8192)
            self.assertEqual(reply.message, message)
            self.assertEqual(reply.metadata["eval_count"], 8192)
            self.assertEqual(reply.metadata["done_reason"], "length")
            connection.close.assert_called_once()

    def test_transport_uses_distinct_reproducible_agent_seeds(self):
        with patch("orchestrator.simulated_web.runner.http.client.HTTPConnection") as factory:
            connection = factory.return_value
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.read.return_value = b'{"message": {"content": "ok"}}'
            client = Ollama("test", seed=7)
            seeds = []
            for agent in ("agent-1", "agent-2", "agent-1"):
                client(agent, [], 10)
                seeds.append(json.loads(connection.request.call_args.args[2])["options"]["seed"])
            Ollama("test", seed=7)("agent-2", [], 10)
            seeds.append(json.loads(connection.request.call_args.args[2])["options"]["seed"])
            self.assertEqual(seeds, [7, 8, 7, 8])

    def test_truncated_response_is_not_complete_or_executed(self):
        for content in ["", "Partial answer"]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as root:
                run_dir = Path(root) / "run"
                calls = [{"function": {"name": "open", "arguments": {"url": "https://wiki.test/"}}}]
                result = run_cohort(run_dir, FIXTURE / "pages.json", FIXTURE / "task.json",
                    lambda *args: ModelResponse({"content": content, "tool_calls": calls},
                                               {"done_reason": "length", "eval_count": 8192}), agents=1)
                self.assertEqual(result[0]["status"], "generation_limit")
                self.assertEqual(result[0]["answer"], content)
                events = [json.loads(line) for line in (run_dir / "logs/agent-1.jsonl").read_text().splitlines()]
                self.assertFalse(any(e["event"] == "tool" for e in events))
                self.assertTrue(any(e["event"] == "model_response" for e in events))

    def test_metadata_stays_out_of_model_conversation(self):
        seen = []
        def client(agent, messages, timeout):
            seen.append(json.loads(json.dumps(messages)))
            if len(seen) == 1:
                return ModelResponse({"content": "", "tool_calls": [{"function": {
                    "name": "search", "arguments": {"query": "Larch"}}}]},
                    {"done_reason": "stop", "eval_count": 27})
            return ModelResponse({"content": "Answer"}, {"done_reason": "stop", "eval_count": 2})
        with tempfile.TemporaryDirectory() as root:
            result = run_cohort(Path(root) / "run", FIXTURE / "pages.json", FIXTURE / "task.json", client, agents=1)
        self.assertEqual(result[0]["status"], "complete")
        self.assertNotIn("eval_count", json.dumps(seen))
        self.assertEqual(result[0]["last_model_metadata"]["eval_count"], 2)


if __name__ == "__main__":
    unittest.main()
