"""Mock-only contracts for model-facing observation masking."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.context import (ContextError, ContextPolicy, ManagedContext,
                                               OMITTED_OBSERVATION, serialized_bound)
from orchestrator.simulated_web.runner import ModelResponse, run_agent


def exchange(index, count=1, size=100):
    return [{'role': 'assistant', 'content': f'text-{index}', 'thinking': f'think-{index}',
             'tool_calls': [{'id': f'{index}-{j}', 'function': {'name': 'open',
                            'arguments': {'url': f'https://docs.test/{index}/{j}'}}} for j in range(count)]},
            *[{'role': 'tool', 'tool_call_id': f'{index}-{j}', 'name': 'open',
               'content': f'observation-{index}-{j}:' + 'x' * size} for j in range(count)]]


class ObservationMaskingTests(unittest.TestCase):
    def test_ten_batches_preserve_calls_text_pairing_and_raw_history(self):
        history = [{'role': 'system', 'content': 'system'}]
        for i in range(12):
            history.extend(exchange(i, count=2))
            history.append({'role': 'assistant', 'content': 'text-only'})
        original = copy.deepcopy(history)
        managed = ManagedContext(None, ContextPolicy(observation_window=10))
        view = managed.model_messages(history)
        for raw, projected in zip(history, view):
            if raw['role'] != 'tool':
                self.assertEqual(raw, projected)
            else:
                old = int(raw['tool_call_id'].split('-')[0]) < 2
                self.assertEqual(projected['content'], OMITTED_OBSERVATION if old else raw['content'])
                self.assertEqual({k: v for k, v in raw.items() if k != 'content'},
                                 {k: v for k, v in projected.items() if k != 'content'})
        self.assertEqual(history, original)

    def test_default_off_and_invalid_window(self):
        history = exchange(0) + exchange(1)
        self.assertIs(ManagedContext(None, ContextPolicy()).model_messages(history), history)
        for value in (-1, True, 1.5, '10'):
            with self.assertRaises(ValueError):
                ContextPolicy(observation_window=value)

    def test_accounting_uses_masked_request_and_checks_native(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            return ModelResponse({'content': 'done'}, {'prompt_eval_count': 1000, 'eval_count': 1})
        managed = ManagedContext(client, ContextPolicy(context_length=32768, observation_window=1))
        history = [{'role': 'system', 'content': 'system'}, *exchange(0, size=200000), *exchange(1)]
        original = copy.deepcopy(history)
        response = managed('agent-1', history, 60)
        self.assertLess(response.metadata['context_accounting']['estimated_input_tokens'], 10000)
        self.assertEqual(response.metadata['context_accounting']['masked_tool_results'], 1)
        self.assertEqual(managed.last_bytes, serialized_bound(seen[0]))
        self.assertEqual(history, original)
        managed.client = lambda *a, **kw: ModelResponse({'content': 'bad'}, {'prompt_eval_count': None})
        with self.assertRaisesRegex(ContextError, 'Missing native'):
            managed('agent-1', history, 60)

    def test_compaction_summary_cannot_recover_masked_prefix(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            return ModelResponse({'content': 'retained facts'}, {'prompt_eval_count': 100, 'eval_count': 5})
        managed = ManagedContext(client, ContextPolicy(observation_window=1))
        history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'initial'},
                   *exchange(0), {'role': 'user', 'content': 'reflection'}, *exchange(1),
                   {'role': 'user', 'content': 'answer'}, *exchange(2)]
        raw_old = history[3]
        managed.phase = 'answer'
        managed.previous_reflection_anchor = history[4]
        managed.answer_anchor = history[7]
        managed.pending_compaction = True
        protected_result = history[6]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'context.jsonl'
            managed.compact('agent-1', history, 60, log)
            self.assertEqual(seen[0][3]['content'], OMITTED_OBSERVATION)
            self.assertTrue(raw_old['content'].startswith('observation-0'))
            self.assertIn(protected_result['content'], log.read_text())
        # Simulate answer removal: the surviving older reflection must stay masked.
        history[:] = history[:managed.anchor_index(history, managed.answer_anchor)]
        self.assertEqual(managed.model_messages(history)[-1]['content'], OMITTED_OBSERVATION)
        self.assertTrue(protected_result['content'].startswith('observation-1'))

    def test_latest_oversized_result_fails_without_truncation(self):
        managed = ManagedContext(lambda *a, **kw: self.fail('Must not call model'),
                                 ContextPolicy(context_length=32768, observation_window=10))
        history = [{'role': 'system', 'content': 'system'}, *exchange(0, size=200000)]
        original = copy.deepcopy(history)
        with self.assertRaisesRegex(ContextError, 'without truncation'):
            managed('agent-1', history, 60)
        self.assertEqual(history, original)

    def test_runner_audit_retains_raw_results_when_model_view_omits_them(self):
        seen = []
        class MockBrowser:
            def call(self, agent, name, arguments):
                return {'text': 'RAW_AUDIT_' + arguments['url']}
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            message = (exchange(len(seen))[0] if len(seen) < 3 else {'content': 'done'})
            return ModelResponse(message, {'prompt_eval_count': 1000, 'eval_count': 1})
        managed = ManagedContext(client, ContextPolicy(observation_window=1))
        history = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            result = run_agent('agent-1', MockBrowser(), 'Question', managed, root, 4, 60, history=history)
            self.assertEqual(result['status'], 'complete')
            events = [json.loads(line) for line in (root / 'logs/agent-1.jsonl').read_text().splitlines()]
            raw_results = [event['response'] for event in events if event['event'] == 'tool']
            self.assertEqual(len(raw_results), 2)
            self.assertTrue(all('RAW_AUDIT_' in result['text'] for result in raw_results))
            self.assertEqual([m for m in seen[-1] if m['role'] == 'tool'][0]['content'], OMITTED_OBSERVATION)
            self.assertTrue(all('RAW_AUDIT_' in m['content'] for m in history if m['role'] == 'tool'))
