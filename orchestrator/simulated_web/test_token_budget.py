"""Focused pressure-budget plumbing coverage; no model or service required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.simulated_web.runner import ModelResponse, Ollama, SYSTEM, TokenBudget, run_agent
from orchestrator.simulated_web.session import FINAL_ANSWER_INSTRUCTION, prepare, resolve_token_budget, run_session
from orchestrator.simulated_web.test_session import records


TOOL_MESSAGE = {"content": "", "tool_calls": [
    {"function": {"name": "open", "arguments": {"url": "https://wiki.test/"}}}]}


class TokenBudgetTests(unittest.TestCase):
    def run_fixture(self, client, steps=36, budget=None, **kwargs):
        kwargs.setdefault("response_token_limit", 8192)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            browser = Mock()
            browser.call.return_value = {"text": "source text" * 1000}
            result = run_agent("agent-1", browser, "Question?", client, root, steps, 600,
                               token_budget=budget or TokenBudget(), **kwargs)
            events = [json.loads(line) for line in (root / "logs/agent-1.jsonl").read_text().splitlines()]
            return result, events, browser

    def test_cumulative_allowance_reserve_and_truncated_tool_not_executed(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append((copy.deepcopy(messages), options))
            if len(requests) == 1:
                return ModelResponse(copy.deepcopy(TOOL_MESSAGE), {"eval_count": 1000})
            if len(requests) == 2:
                message = {**copy.deepcopy(TOOL_MESSAGE), "thinking": "partial reasoning"}
                return ModelResponse(message, {"eval_count": 2744, "done_reason": "length"})
            return ModelResponse({"content": "Answer"}, {"eval_count": 256})
        result, events, browser = self.run_fixture(client)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([options["num_predict"] for _, options in requests], [3744, 2744, 256])
        self.assertEqual([options["final_only"] for _, options in requests], [False, False, True])
        self.assertEqual(browser.call.call_count, 1)
        self.assertNotIn("tool_calls", requests[-1][0][-2])
        self.assertEqual(requests[-1][0][-2]["thinking"], "partial reasoning")
        self.assertEqual(result["token_budget"]["generated_tokens"], 4000)
        self.assertTrue(result["token_budget"]["cap_verified"])
        raw = [event["message"] for event in events if event["event"] == "assistant"]
        self.assertIn("tool_calls", raw[1])
        self.assertEqual(raw[1]["thinking"], "partial reasoning")

    def test_final_turn_is_within_step_cap_and_uses_unused_balance(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append(options)
            return ModelResponse({"content": "Answer"} if options["final_only"]
                                 else copy.deepcopy(TOOL_MESSAGE), {"eval_count": 1})
        result, _, browser = self.run_fixture(client)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(requests), 36)
        self.assertEqual(browser.call.call_count, 35)
        self.assertTrue(requests[-1]["final_only"])
        self.assertEqual(requests[-1]["num_predict"], 3965)

    def test_missing_invalid_and_excess_counts_fail_closed(self):
        for count in (None, True, -1, 1.5, "10", 4001):
            with self.subTest(count=count):
                client = Mock(return_value=ModelResponse(copy.deepcopy(TOOL_MESSAGE), {"eval_count": count}))
                result, _, browser = self.run_fixture(client)
                self.assertEqual(result["status"], "budget_error")
                self.assertEqual(client.call_count, 1)
                browser.call.assert_not_called()
                self.assertFalse(result["token_budget"]["cap_verified"])
                if count == 4001:
                    self.assertEqual(result["token_budget"]["generated_tokens"], 4001)
                    self.assertTrue(result["token_budget"]["backend_allowance_violation"])
                else:
                    self.assertIsNone(result["token_budget"]["generated_tokens"])

    def test_final_truncation_or_tools_fail_without_execution(self):
        for final_message, metadata, status in (
                ({"content": "Partial"}, {"eval_count": 256, "done_reason": "length"}, "budget_limit"),
                (TOOL_MESSAGE, {"eval_count": 1}, "budget_error")):
            with self.subTest(status=status):
                client = Mock(side_effect=[
                    ModelResponse({"content": "", "thinking": "partial"},
                                  {"eval_count": 3744, "done_reason": "length"}),
                    ModelResponse(copy.deepcopy(final_message), metadata)])
                result, _, browser = self.run_fixture(client)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["answer"], "")
                self.assertEqual(client.call_count, 2)
                browser.call.assert_not_called()

    def test_transport_final_only_and_legacy_request(self):
        with patch("orchestrator.simulated_web.runner.http.client.HTTPConnection") as factory:
            connection = factory.return_value
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.read.return_value = (
                b'{"message":{"content":"Answer"},"eval_count":2}')
            client = Ollama("test", seed=0, max_output_tokens=8192)
            response = client("agent-1", [], 600, num_predict=256, final_only=True)
            payload = json.loads(connection.request.call_args.args[2])
            self.assertEqual(payload["options"]["num_predict"], 256)
            self.assertEqual(payload["options"]["seed"], 0)
            self.assertFalse(payload["think"])
            self.assertNotIn("tools", payload)
            self.assertEqual(response.metadata["requested_num_predict"], 256)
            client("agent-1", [], 600)
            payload = json.loads(connection.request.call_args.args[2])
            self.assertEqual(payload["options"]["num_predict"], 8192)
            self.assertNotIn("think", payload)
            self.assertIn("tools", payload)

    def test_session_budgets_reset_and_failures_export_blank(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append(options)
            return ModelResponse({"content": "Answer"}, {"eval_count": 1 if len(requests) == 1 else None})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            result = run_session(root, records(2), client, agents=1, prompt_condition="pressure")
            self.assertEqual([request["num_predict"] for request in requests], [3744, 3744])
            self.assertEqual([row["status"] for row in result], ["complete", "budget_error"])
            predictions = [json.loads(line) for line in
                           (root / "predictions/agent-1.predictions.jsonl").read_text().splitlines()]
            answers = {row["id"]: row["predicted_answer"] for row in predictions}
            self.assertEqual(answers[result[0]["id"]], "Answer")
            self.assertEqual(answers[result[1]["id"]], "")
            config = json.loads((root / "settings.json").read_text())
            self.assertEqual(config["token_budget"]["total_generated_token_limit"], 4000)

    def test_pressure_prevalidation_and_full_corpus_selector_shards(self):
        source = records(20)
        baseline = prepare(source, 1, 0, "neutral")
        for options in ({"shard": 1, "shards": 4}, {"question_id": "q7"}):
            prepared = prepare(source, 1, 0, "pressure", **options)
            self.assertEqual(prepared[:2], baseline[:2])
            self.assertEqual(prepared[2]["model_seeds"], baseline[2]["model_seeds"])
            self.assertIn("3000", prepared[3])
            self.assertIn("4000", prepared[3])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            for options in ({"target_tokens": 4000}, {"total_token_budget": 0},
                            {"final_reserve": 4000}, {"target_tokens": True}, {"steps": 1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_session(root, source, None, agents=1, prompt_condition="pressure", **options)
                self.assertFalse(root.exists())
        with self.assertRaises(ValueError):
            resolve_token_budget("neutral", total_token_budget=4000)

    def test_maximal_pressure_composes_same_prompt_and_schedule(self):
        source = records(20)
        maximal = prepare(source, 2, 0, "maximal")
        pressure = prepare(source, 2, 0, "pressure")
        combined = prepare(source, 2, 0, "maximal-pressure")
        self.assertEqual(combined[:3], maximal[:3])
        self.assertEqual(combined[3],
                         maximal[3][:-len(FINAL_ANSWER_INSTRUCTION) - 1] + pressure[3][len(SYSTEM):])
        self.assertEqual(resolve_token_budget("maximal-pressure"), resolve_token_budget("pressure"))
        self.assertIsNone(resolve_token_budget("maximal"))

    def test_maximal_pressure_budgets_reset_for_both_agents_and_questions(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append((agent, next(m["content"] for m in reversed(messages) if m["role"] == "user"), options["num_predict"]))
            return ModelResponse({"content": "Answer"}, {"eval_count": 100})
        with tempfile.TemporaryDirectory() as temporary:
            results = run_session(Path(temporary) / "run", records(2), client,
                                  agents=2, prompt_condition="maximal-pressure")
        self.assertEqual(len(requests), 4)
        self.assertEqual(len({(agent, question) for agent, question, _ in requests}), 4)
        self.assertEqual({agent for agent, _, _ in requests}, {"agent-1", "agent-2"})
        self.assertEqual({allowance for _, _, allowance in requests}, {3744})
        self.assertTrue(all(row["status"] == "complete" for row in results))
        self.assertTrue(all(row["token_budget"]["generated_tokens"] == 100 for row in results))

    def test_maximal_pressure_rejects_single_agent_sharding_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            for options in ({"agents": 1}, {"agents": 2, "shards": 2},
                            {"agents": 2, "question_id": "q0"}, {"agents": 2, "steps": 1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_session(root, records(2), None, prompt_condition="maximal-pressure", **options)
                self.assertFalse(root.exists())

    def test_uncapped_client_signature_and_behavior_unchanged(self):
        def client(agent, messages, timeout):
            return ModelResponse({"content": "Partial"}, {"done_reason": "length"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            result = run_session(root, records(1), client, agents=1, prompt_condition="neutral")
            self.assertEqual(result[0]["status"], "generation_limit")
            self.assertNotIn("token_budget", result[0])


if __name__ == "__main__":
    unittest.main()
