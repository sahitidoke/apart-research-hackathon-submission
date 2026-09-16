"""Mock-only contracts for combined growth and proactive phase boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.context import ContextPolicy, ManagedContext
from orchestrator.simulated_web.growth import bounded_result, result_bytes
from orchestrator.simulated_web.runner import ModelResponse, PreparationBudget, TokenBudget, run_agent
from orchestrator.simulated_web.session import REFLECTION_PROTOCOL, run_session
from orchestrator.simulated_web.test_session import records


def tool(url='https://docs.test/source'):
    return {'function': {'name': 'open', 'arguments': {'url': url}}}


class CombinedBudgetTests(unittest.TestCase):
    def test_structured_excerpts_preserve_unicode_json_handles_and_link_ids(self):
        raw = {'page_id': 'p7', 'url': 'https://docs.test/source', 'title': 'Long title',
               'text': '漢字🙂' * 2000,
               'links': [{'id': 9, 'url': 'https://docs.test/edit', 'label': 'Edit'}]}
        original = copy.deepcopy(raw)
        result = bounded_result(raw, 600)
        self.assertLessEqual(result_bytes(result), 600)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(result['page_id'], 'p7')
        self.assertEqual(result['links'][0]['id'], 9)
        self.assertEqual(result['links'][0]['url'], raw['links'][0]['url'])
        self.assertIn('omitted', result)
        self.assertEqual(raw, original)
        self.assertEqual(result_bytes(result), len(json.dumps(result).encode('utf-8')))

    def test_huge_metadata_has_explicit_paired_status_fallback(self):
        for status, raw in [('saved', {'saved': 'https://docs.test/' + 'a' * 10000}),
                            ('error', {'error': 'bad' * 10000}),
                            ('completed', {'page_id': 'p1', 'url': 'x' * 10000})]:
            result = bounded_result(raw, 256)
            self.assertEqual(result['status'], status)
            self.assertLessEqual(result_bytes(result), 256)
            self.assertIn('omitted', result)

    def run_bounded(self, raw, combined=1000, generated=500, reserve=100, prep=False):
        calls = []
        seen = []
        class Browser:
            def call(self, agent, name, arguments):
                calls.append(arguments)
                return raw
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            message = ({'content': '', 'tool_calls': [tool()]} if len(seen) == 1 else {'content': 'done'})
            return ModelResponse(message, {'eval_count': min(10, options['num_predict']),
                                           'prompt_eval_count': 100})
        history = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            budget = PreparationBudget(generated) if prep else TokenBudget(generated - reserve, generated, reserve)
            result = run_agent('agent-1', Browser(), 'Question', client, root, 4, 60,
                               token_budget=budget, history=history,
                               phase='preparation' if prep else 'answer', combined_token_limit=combined)
            events = [json.loads(line) for line in (root / 'logs/agent-1.jsonl').read_text().splitlines()]
        return result, history, events, calls

    def test_new_delivered_results_charged_once_raw_audit_preserved(self):
        raw = {'page_id': 'p1', 'url': 'https://docs.test/source', 'text': 'x' * 20000, 'links': []}
        result, history, events, calls = self.run_bounded(raw)
        self.assertEqual(result['status'], 'complete')
        ledger = result['combined_budget']
        delivered = [m for m in history if m['role'] == 'tool']
        self.assertEqual(ledger['estimated_observation_tokens'], sum(len(m['content'].encode('utf-8')) for m in delivered))
        self.assertEqual(ledger['raw_observation_bytes'], result_bytes(raw))
        self.assertLessEqual(ledger['combined_used'], 1000)
        self.assertTrue(ledger['cap_verified'])
        self.assertLessEqual(result['token_budget']['observed_generated_tokens'], 500)
        self.assertEqual(next(e['response'] for e in events if e['event'] == 'tool'), raw)
        delivery = next(e for e in events if e['event'] == 'observation_budget')
        self.assertEqual(delivery['response'], json.loads(delivered[0]['content']))
        self.assertNotEqual(delivery['response'], raw)
        self.assertEqual(len(calls), 1)

    def test_no_tool_execution_when_acknowledgment_reserve_will_not_fit(self):
        result, history, events, calls = self.run_bounded({'saved': 'https://docs.test/source'}, combined=200)
        self.assertEqual(calls, [])
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(any(m.get('tool_calls') for m in history))
        self.assertTrue(result['token_budget']['final_phase_started'])
        self.assertLessEqual(result['combined_budget']['combined_used'], 200)

    def test_preparation_finishes_gracefully_when_result_reserve_unavailable(self):
        result, history, events, calls = self.run_bounded({'text': 'large'}, combined=200, prep=True)
        self.assertEqual(result['status'], 'prepared_budget_limit')
        self.assertEqual(calls, [])
        self.assertFalse(any(m.get('tool_calls') for m in history))

    def test_write_acknowledgment_is_delivered_and_paired(self):
        result, history, events, calls = self.run_bounded({'saved': 'https://docs.test/source'})
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(next(m['content'] for m in history if m['role'] == 'tool')),
                         {'saved': 'https://docs.test/source'})
        self.assertEqual(len([m for m in history if m.get('tool_calls')]), 1)

    def test_preblock_answer_summarizes_completed_history_before_generation(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append((copy.deepcopy(messages), options))
            return ModelResponse({'content': 'summary' if options.get('summary_only') else 'answer'},
                                 {'prompt_eval_count': 1000, 'eval_count': 5})
        managed = ManagedContext(client, ContextPolicy(context_length=32768, observation_window=10,
                                                       combined_phase_budgets=True))
        history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'old' * 29000}]
        managed.previous_reflection_anchor = history[1]
        managed.begin_phase('answer', history)
        current = {'role': 'user', 'content': 'CURRENT QUESTION'}
        history.append(current)
        with tempfile.TemporaryDirectory() as directory:
            managed.log_path = Path(directory) / 'context.jsonl'
            managed('agent-1', history, 60, num_predict=100)
            self.assertTrue(seen[0][1]['summary_only'])
            self.assertNotIn('CURRENT QUESTION', json.dumps(seen[0][0]))
            self.assertIs(history[-1], current)
            self.assertLess(managed.estimate(history) + 4000 + 1024, 32768)
            event = next(json.loads(line) for line in managed.log_path.read_text().splitlines()
                         if json.loads(line)['event'] == 'pre_block_context')
            self.assertTrue(event['compacted'])
        self.assertEqual(len(seen), 2)

    def test_preblock_reflection_keeps_current_answer_handoff(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            return ModelResponse({'content': 'summary'}, {'prompt_eval_count': 1000, 'eval_count': 5})
        managed = ManagedContext(client, ContextPolicy(context_length=32768, observation_window=10,
                                                       combined_phase_budgets=True))
        answer = {'role': 'user', 'content': 'CURRENT QUESTION'}
        response = {'role': 'assistant', 'content': 'CURRENT ANSWER', 'thinking': 'answer reasoning'}
        history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'old' * 25000}, answer, response]
        managed.answer_anchor = answer
        managed.begin_phase('reflection', history)
        reflection = {'role': 'user', 'content': 'REFLECT'}
        history.append(reflection)
        with tempfile.TemporaryDirectory() as directory:
            managed.log_path = Path(directory) / 'context.jsonl'
            managed('agent-1', history, 60, num_predict=100)
        self.assertNotIn('CURRENT ANSWER', json.dumps(seen[0]))
        self.assertIn('CURRENT ANSWER', json.dumps(seen[1]))
        self.assertEqual(managed.finish_reflection(history), 2)
        self.assertIn(reflection, history)

    def test_optional_defaults_and_session_phase_limits(self):
        self.assertIsNone(ManagedContext(None, ContextPolicy()).combined_token_limit)
        with self.assertRaises(ValueError):
            ContextPolicy(combined_phase_budgets=1)
        def client(agent, messages, timeout, **options):
            self.assertFalse(options.get('summary_only'))
            return ModelResponse({'content': 'done'}, {'prompt_eval_count': 1000, 'eval_count': 5})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            result = run_session(root, records(10), client, agents=1,
                prompt_condition='neutral-reflection', protocol=REFLECTION_PROTOCOL, preparation_topic='Topic',
                editable_source_title='Document 0',
                editable_source_text_sha256=hashlib.sha256(b'Evidence 0').hexdigest(),
                context_policy=ContextPolicy(context_length=32768, observation_window=10, combined_phase_budgets=True))
            self.assertTrue(all(row['status'] == 'complete' for row in result))
            self.assertEqual([p['combined_budget']['limit'] for p in result[0]['phases']], [4000, 8000])
            settings = json.loads((root / 'settings.json').read_text())
            self.assertTrue(settings['context_policy']['combined_phase_budgets'])
            self.assertNotIn('that previous reflection', settings['system_prompt'])
            self.assertIn('several times the actual model token count', settings['system_prompt'])
            self.assertEqual(settings['context_policy']['combined_growth_limits']['initial_research'], 16000)
