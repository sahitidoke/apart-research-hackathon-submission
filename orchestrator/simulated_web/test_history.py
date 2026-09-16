"""Prepared mocked history regressions; do not execute without authorization."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse, TokenBudget, run_agent
from orchestrator.simulated_web.session import prepare, run_session
from orchestrator.simulated_web.test_session import records


def tool(name, **arguments):
    return {'function': {'name': name, 'arguments': arguments}}


class HistoryTests(unittest.TestCase):
    def test_own_memory_tools_handles_and_current_metrics(self):
        observed = {}
        def client(agent, messages, timeout):
            users = [i for i, message in enumerate(messages) if message['role'] == 'user']
            current = messages[users[-1] + 1:]
            prior = messages[:users[-1]]
            serialized = json.dumps(messages)
            other = 'agent-2' if agent == 'agent-1' else 'agent-1'
            self.assertNotIn('PRIVATE-' + other, serialized)
            self.assertNotIn('PRIVATE_GOLD', serialized)
            self.assertNotIn('PRIVATE_DECOMPOSITION', serialized)
            self.assertEqual(sum(m['role'] == 'system' for m in messages), 1)
            returned = [m for m in current if m['role'] == 'tool']
            if not returned:
                observed[(agent, len(users))] = copy.deepcopy(messages)
                if len(users) == 1:
                    url = messages[users[-1]]['content'].split('Document collection: ')[1].splitlines()[0]
                    calls = [tool('open', url=url)]
                else:
                    self.assertIn('PRIVATE-' + agent, json.dumps(prior))
                    old_page = json.loads(next(m['content'] for m in prior if m['role'] == 'tool'))
                    self.assertEqual(old_page['page_id'], 'p1')
                    self.assertIn('New question boundary', messages[users[-1]]['content'])
                    # Open another page first; p1 must still refer to the earlier collection.
                    calls = [tool('open', url='https://wiki.test/'),
                             tool('click', page_id=old_page['page_id'], link_id=1)]
                return ModelResponse({'content': '', 'thinking': 'PRIVATE-' + agent,
                                      'tool_calls': calls}, {'eval_count': 7, 'prompt_eval_count': 30})
            if len(users) == 2:
                prior_page = json.loads(next(m['content'] for m in prior if m['role'] == 'tool'))
                self.assertEqual(json.loads(returned[-1]['content'])['url'], prior_page['links'][0]['url'])
                self.assertNotEqual(json.loads(returned[0]['content'])['page_id'], 'p1')
            return ModelResponse({'content': 'PRIVATE-' + agent + '-answer'},
                                 {'eval_count': 3, 'prompt_eval_count': 40})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            with patch.object(Browser, 'reset_views', side_effect=AssertionError('Must preserve handles')):
                results = run_session(root, records(2), client, steps=3)
            self.assertEqual(len(observed), 4)
            self.assertTrue(all(row['status'] == 'complete' for row in results))
            for row in results:
                events = [json.loads(line) for line in (root / row['log_path']).read_text().splitlines()]
                self.assertEqual(sum(e['event'] == 'assistant' for e in events), 2)
                self.assertEqual(sum(e['event'] == 'model_response' for e in events), 2)
                self.assertEqual(row['metrics']['generated_tokens'], 10)
                self.assertEqual(row['metrics']['browser_calls'], row['slot'])
                self.assertEqual(events[0]['history_mode'], 'persistent')
                self.assertEqual(events[0]['carried_messages'] > 0, row['slot'] == 2)
            for filename in ('settings.json', 'manifest.json'):
                self.assertEqual(json.loads((root / filename).read_text())['history_mode'], 'persistent')

    def test_reset_and_bad_mode(self):
        def client(agent, messages, timeout):
            self.assertEqual([m['role'] for m in messages], ['system', 'user'])
            return {'content': 'Answer'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            with self.assertRaises(ValueError):
                run_session(root, records(2), client, history_mode='typo')
            self.assertFalse(root.exists())
            original = Browser.reset_views
            calls = []
            def reset(browser):
                calls.append(True)
                return original(browser)
            with patch.object(Browser, 'reset_views', reset):
                outcomes = run_session(root, records(2), client, history_mode='reset')
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(row['status'] == 'complete' for row in outcomes))
            self.assertIn('reset between questions', prepare(records(2), 2, 0, 'maximal', history_mode='reset')[3])

    def test_question_after_final_only_renews_budget_and_tools(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append(copy.deepcopy((messages, options)))
            if options['final_only']:
                return ModelResponse({'content': 'Done'}, {'eval_count': 10})
            return ModelResponse({'content': '', 'thinking': 'retained reasoning',
                                  'tool_calls': [tool('open', url='https://wiki.test/')]},
                                 {'done_reason': 'length', 'eval_count': 3744})
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = run_session(Path(temporary) / 'run', records(2), client,
                                   agents=1, prompt_condition='pressure', steps=2)
        self.assertTrue(all(len([m for m in messages if m['role'] == 'system']) == 1
                            for messages, _ in requests))
        self.assertEqual([options['num_predict'] for _, options in requests], [3744, 256, 3744, 256])
        self.assertEqual([options['final_only'] for _, options in requests], [False, True, False, True])
        resumed = requests[2][0]
        self.assertIn('retained reasoning', json.dumps(resumed))
        self.assertIn('final-only instruction', resumed[-1]['content'])
        self.assertIn('renewed for this question', resumed[-1]['content'])
        self.assertFalse(any(m.get('tool_calls') for m in resumed))
        self.assertTrue(all(row['token_budget']['generated_tokens'] == 3754 for row in outcomes))
        self.assertTrue(all(row['status'] == 'complete' for row in outcomes))

    def test_failed_tool_batch_keeps_completed_pairs_only(self):
        class PartialBrowser:
            def __init__(self):
                self.calls = []
            def call(self, agent, name, args):
                self.calls.append(args['url'])
                if len(self.calls) == 2:
                    raise RuntimeError('browser stopped')
                return {'text': 'completed useful result'}
        browser = PartialBrowser()
        history = []
        response = {'content': '', 'thinking': 'useful partial reasoning', 'tool_calls': [
            tool('open', url='https://wiki.test/'), tool('open', url='https://wiki.test/edit'),
            tool('open', url='https://wiki.test/never-execute')]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for slot in ('first', 'second'):
                (root / slot / 'logs').mkdir(parents=True)
            result = run_agent('agent-1', browser, 'First?', lambda *a: response,
                               root / 'first', 3, 60, history=history)
            self.assertEqual(result['status'], 'error')
            assistant = next(m for m in history if m['role'] == 'assistant')
            self.assertEqual(assistant['thinking'], 'useful partial reasoning')
            self.assertEqual(len(assistant['tool_calls']), 1)
            self.assertEqual(sum(m['role'] == 'tool' for m in history), 1)
            events = [json.loads(line) for line in (root / 'first/logs/agent-1.jsonl').read_text().splitlines()]
            self.assertEqual(next(e for e in events if e['event'] == 'history_cleanup')['discarded_unexecuted_tool_calls'], 2)
            self.assertEqual(len(next(e for e in events if e['event'] == 'assistant')['message']['tool_calls']), 3)
            result = run_agent('agent-1', browser, 'Second?', lambda *a: {'content': 'Done'},
                               root / 'second', 3, 60, history=history)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(browser.calls), 2)

    def test_truncated_malformed_and_budget_rejected_carryover(self):
        pending = {'content': 'partial', 'thinking': 'useful',
                   'tool_calls': [tool('open', url='https://wiki.test/') ]}
        cases = [
            (ModelResponse(pending, {'done_reason': 'length'}), None, 'generation_limit', True),
            ({'content': 123, 'thinking': 'invalid'}, None, 'error', False),
            ({'content': 'invalid', 'thinking': {}}, None, 'error', False),
            (ModelResponse(pending, {'eval_count': None}), TokenBudget(3000, 4000, 256), 'budget_error', False),
            (ModelResponse(pending, {'eval_count': 4001}), TokenBudget(3000, 4000, 256), 'budget_error', False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for index, (response, budget, status, retained) in enumerate(cases):
                with self.subTest(index=index):
                    root = Path(temporary) / str(index)
                    (root / 'logs').mkdir(parents=True)
                    history = []
                    result = run_agent('agent-1', None, 'Question?', lambda *a, **k: response,
                                       root, 2, 60, history=history, token_budget=budget)
                    self.assertEqual(result['status'], status)
                    self.assertFalse(any(m.get('tool_calls') for m in history))
                    self.assertEqual(any(m['role'] == 'assistant' for m in history), retained)
                    if retained:
                        self.assertEqual(history[-1]['thinking'], 'useful')


if __name__ == '__main__':
    unittest.main()
