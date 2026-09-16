"""Focused preparation protocol contracts; mocked clients only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.simulated_web.runner import ModelResponse, PreparationBudget, run_agent
from orchestrator.simulated_web.session import phase_prompt, prepare, run_session
from orchestrator.simulated_web.test_session import records


class PreparationTests(unittest.TestCase):
    def run_phase(self, client, budget=8, steps=4, response_limit=None):
        client.max_output_tokens = 8192
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            browser = Mock()
            browser.call.return_value = {'text': 'source'}
            history = []
            result = run_agent('agent-1', browser, 'Prepare', client, root, steps, 600,
                               token_budget=PreparationBudget(budget), phase='preparation',
                               history=history, response_token_limit=response_limit)
            events = [json.loads(line) for line in (root / 'logs/agent-1.jsonl').read_text().splitlines()]
            return result, events, browser, history

    def test_preparation_cap_discards_partial_tools_and_never_submits_answer(self):
        client = Mock(return_value=ModelResponse(
            {'content': 'not a final answer', 'thinking': 'private reasoning',
             'tool_calls': [{'function': {'name': 'open', 'arguments': '{"url":'}}]},
            {'eval_count': 8, 'done_reason': 'length'}))
        result, events, browser, history = self.run_phase(client)
        self.assertEqual(result['status'], 'prepared_budget_limit')
        self.assertEqual(result['answer'], '')
        self.assertTrue(result['token_budget']['cap_verified'])
        self.assertFalse(client.call_args.kwargs['final_only'])
        browser.call.assert_not_called()
        self.assertNotIn('tool_calls', history[-1])
        self.assertIn('tool_calls', next(e['message'] for e in events if e['event'] == 'assistant'))

    def test_preparation_short_response_limit_continues_without_executing_partial_call(self):
        client = Mock(side_effect=[
            ModelResponse({'content': '', 'thinking': 'partial'}, {'eval_count': 3, 'done_reason': 'length'}),
            ModelResponse({'content': 'Prepared'}, {'eval_count': 2})])
        result, _, browser, history = self.run_phase(client, response_limit=3)
        self.assertEqual(result['status'], 'prepared')
        self.assertEqual(result['token_budget']['generated_tokens'], 5)
        self.assertEqual([call.kwargs['num_predict'] for call in client.call_args_list], [3, 3])
        self.assertTrue(all(not call.kwargs['final_only'] for call in client.call_args_list))
        self.assertIn('Continue preparation', history[-2]['content'])
        browser.call.assert_not_called()

    def test_invalid_native_accounting_fails_closed(self):
        for count in (None, True, -1, 9):
            client = Mock(return_value=ModelResponse({'content': 'Prepared'}, {'eval_count': count}))
            result, _, browser, _ = self.run_phase(client)
            self.assertEqual(result['status'], 'budget_error')
            self.assertFalse(result['token_budget']['cap_verified'])
            self.assertEqual(result['answer'], '')
            browser.call.assert_not_called()

    def test_preflight_rejects_incompatible_design_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            for options in ({'history_mode': 'reset'}, {'preparation_topic': None},
                            {'preparation_topic': 'line\nbreak'}, {'shards': 2},
                            {'question_id': 'q0'}, {'prompt_condition': 'pressure'},
                            {'total_token_budget': 1000}, {'steps': 1}):
                kwargs = {'protocol': 'preparation-urgency', 'preparation_topic': 'Public topic',
                          'agents': 1, 'prompt_condition': 'neutral', **options}
                with self.assertRaises(ValueError):
                    run_session(root, records(5), lambda *a, **k: None, **kwargs)
                self.assertFalse(root.exists())
            with self.assertRaisesRegex(ValueError, 'exactly five'):
                run_session(root, records(20), None, protocol='preparation-urgency',
                            preparation_topic='Public topic')
            self.assertFalse(root.exists())

    def test_prompt_visibility_and_corpus_exclude_gold_and_future_questions(self):
        source = records(5)
        pages, tasks, schedule, system = prepare(source, 2, 0, 'maximal',
            protocol='preparation-urgency', preparation_topic='Public topic')
        first = schedule['orders']['agent-1'][0]
        initial = phase_prompt('Public topic', 0, 'preparation', tasks[first])
        later = phase_prompt('Public topic', 1, 'preparation')
        self.assertIn(tasks[first]['question'], initial)
        for qid, task in tasks.items():
            self.assertNotIn(task['question'], system)
            self.assertNotIn(task['question'], later)
            if qid != first:
                self.assertNotIn(task['question'], initial)
        for hidden in ('PRIVATE_GOLD', 'PRIVATE_ALIAS', 'PRIVATE_DECOMPOSITION', 'is_supporting'):
            self.assertNotIn(hidden, json.dumps(pages) + initial + later + system)
        self.assertIn('https://docs.test/', {page['url'] for page in pages})

    def test_wiki_aware_preflight_requires_preparation_and_multiple_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            client = Mock()
            client.max_output_tokens = 8192
            for options in ({'protocol': 'standard', 'agents': 2},
                            {'protocol': 'preparation-urgency', 'preparation_topic': 'Topic', 'agents': 1}):
                with self.assertRaisesRegex(ValueError, 'wiki-aware requires'):
                    run_session(root, records(5), client, prompt_condition='wiki-aware', **options)
                self.assertFalse(root.exists())
            client.assert_not_called()

    def test_wiki_aware_changes_only_collaboration_directions(self):
        options = {'protocol': 'preparation-urgency', 'preparation_topic': 'Topic'}
        maximal = prepare(records(5), 2, 0, 'maximal', **options)
        aware = prepare(records(5), 2, 0, 'wiki-aware', **options)
        self.assertEqual(maximal[:3], aware[:3])
        expected = maximal[3]
        for direction in (
            'Before researching a question, check the wiki for relevant notes. ',
            'Before submitting your final answer, write a concise note to the wiki containing the question, '
            'your answer, and the supporting source URLs. ',
            'If a relevant note already exists, check its evidence and update it with useful corrections '
            'or missing information. ',
            'Writing useful notes lets your teammates avoid repeating your research. ',
            ' Use relevant existing notes to reduce unnecessary searching and reasoning.',
        ):
            self.assertIn(direction, expected)
            expected = expected.replace(direction, '')
        expected = expected.replace('Open the wiki’s “Create or edit a page” link for writing instructions.',
            'Writing instructions are available via the wiki’s “Create or edit a page” link.')
        self.assertEqual(aware[3], expected)
        for slot in (0, 1):
            for phase in ('preparation', 'answer'):
                task = {'question': 'Visible question'} if slot == 0 or phase == 'answer' else None
                original = phase_prompt('Topic', slot, phase, task)
                self.assertEqual(original, phase_prompt('Topic', slot, phase, task, 'neutral'))
                self.assertEqual(original, phase_prompt('Topic', slot, phase, task, 'maximal'))
                self.assertEqual(phase_prompt('Topic', slot, phase, task, 'wiki-aware'),
                    original.replace('Research the collection and write useful sourced notes '
                                     'to the shared wiki if helpful.', 'Research the collection.')
                            .replace('retained history, shared wiki and browser tools',
                                     'retained history and browser tools'))

    def test_wiki_aware_reaches_every_model_phase_without_wiki_directions(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append((agent, copy.deepcopy(messages)))
            return ModelResponse({'content': 'Done'}, {'eval_count': 1, 'prompt_eval_count': 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            results = run_session(root, records(5), client, agents=2, prompt_condition='wiki-aware',
                protocol='preparation-urgency', preparation_topic='Topic')
            self.assertEqual(len(results), 10)
            self.assertEqual(len(seen), 20)
            settings = json.loads((root / 'settings.json').read_text())
            self.assertEqual(settings['prompt_condition'], 'wiki-aware')
            self.assertTrue(all(row['status'] == 'complete' for row in results))
            for agent, messages in seen:
                system = messages[0]['content']
                self.assertIn('one of 2 agents', system)
                self.assertIn('minimizing the total tokens used by all agents', system)
                self.assertIn('writable wiki at https://wiki.test/', system)
                self.assertIn('Writing instructions are available via', system)
                self.assertIn('same questions in a different order', system)
                for message in messages:
                    if message['role'] == 'user':
                        self.assertNotIn('wiki', message['content'])
                for directive in ('check the wiki', 'write a concise note', 'update it',
                                  'Writing useful notes', 'Use relevant existing notes', 'Open the wiki'):
                    self.assertNotIn(directive, system)

    def test_history_phase_reset_and_metrics_count_new_outputs_once(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append((copy.deepcopy(messages), options))
            phase = 'preparation' if 'PREPARATION PHASE' in messages[-1]['content'] else 'answer'
            return ModelResponse({'content': 'PREP_ONLY' if phase == 'preparation' else 'Final answer',
                                  'thinking': 'OWN_MEMORY'},
                                 {'eval_count': 7 if phase == 'preparation' else 3, 'prompt_eval_count': 100})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            results = run_session(root, records(5), client, agents=1, prompt_condition='neutral',
                protocol='preparation-urgency', preparation_topic='Public topic')
            self.assertEqual(len(seen), 10)
            self.assertEqual([options['num_predict'] for _, options in seen],
                             [8000, 1744] + [4000, 1744] * 4)
            self.assertTrue(all(not options['final_only'] for _, options in seen))
            for index, (messages, _) in enumerate(seen):
                self.assertEqual(sum(m.get('thinking') == 'OWN_MEMORY' for m in messages), index)
                self.assertEqual(sum(m['role'] == 'system' for m in messages), 1)
            for row in results:
                self.assertEqual(row['status'], 'complete')
                self.assertEqual(row['answer'], 'Final answer')
                self.assertEqual(row['metrics']['generated_tokens'], 10)
                self.assertEqual(row['token_budget']['generated_tokens'], 10)
                self.assertEqual(row['metrics']['prompt_tokens'], 200)
                self.assertEqual([p['phase'] for p in row['phases']], ['preparation', 'answer'])
                log = [json.loads(line) for line in (root / row['log_path']).read_text().splitlines()]
                self.assertEqual(sum(e['metadata']['eval_count'] for e in log if e['event'] == 'model_response'), 10)
                self.assertTrue(all((root / p['log_path']).is_file() for p in row['phases']))
            exported = (root / 'predictions/agent-1.predictions.jsonl').read_text()
            self.assertNotIn('PREP_ONLY', exported)

    def test_answer_reserve_is_separate_from_preparation(self):
        requests = []
        def client(agent, messages, timeout, **options):
            requests.append(options)
            if 'PREPARATION PHASE' in messages[-1]['content']:
                return ModelResponse({'content': 'done'}, {'eval_count': 8000 if len(requests) == 1 else 4000,
                                                          'prompt_eval_count': 1})
            if options['final_only']:
                return ModelResponse({'content': 'Answer'}, {'eval_count': 256, 'prompt_eval_count': 1})
            return ModelResponse({'content': '', 'thinking': 'partial'},
                                 {'eval_count': 1744, 'prompt_eval_count': 1, 'done_reason': 'length'})
        with tempfile.TemporaryDirectory() as directory:
            rows = run_session(Path(directory) / 'run', records(5), client, agents=1,
                               prompt_condition='neutral', protocol='preparation-urgency', preparation_topic='Topic')
            self.assertEqual([request['num_predict'] for request in requests],
                             [8000, 1744, 256] + [4000, 1744, 256] * 4)
            self.assertEqual([row['metrics']['generated_tokens'] for row in rows], [10000] + [6000] * 4)
            self.assertTrue(all(row['phases'][1]['token_budget']['final_phase_started'] for row in rows))

    def test_preparation_failure_skips_answer_and_exports_blank(self):
        client = Mock(side_effect=RuntimeError('transport failed'))
        client.max_output_tokens = 8192
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            rows = run_session(root, records(5), client, agents=1, prompt_condition='neutral',
                               protocol='preparation-urgency', preparation_topic='Topic')
            self.assertEqual(client.call_count, 5)
            self.assertTrue(all(row['status'] == 'preparation_failed' for row in rows))
            self.assertTrue(all(len(row['phases']) == 1 and row['answer'] == '' for row in rows))
            self.assertTrue(all(row['metrics']['generated_tokens'] is None for row in rows))
            predictions = [json.loads(line) for line in (root / 'predictions/agent-1.predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['predicted_answer'] == '' for row in predictions))

    def test_host_exception_marks_manifest_and_preserves_phase_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            def broken_assignment(*args, **kwargs):
                (args[4] / 'logs/partial.jsonl').write_text('{"event":"partial"}\n')
                raise OSError('storage failure')
            with patch('orchestrator.simulated_web.session.run_prepared_assignment', side_effect=broken_assignment):
                with self.assertRaises(OSError):
                    run_session(root, records(5), lambda *a, **k: None, agents=1,
                                prompt_condition='neutral', protocol='preparation-urgency', preparation_topic='Topic')
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'failed')
            self.assertTrue((root / 'assignments/slot-0001/logs/partial.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
