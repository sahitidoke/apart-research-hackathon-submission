"""Paired initial-research protocol contracts, with no model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.session import CYCLES_CONDITIONS, CYCLES_PROTOCOL, prepare, run_session
from orchestrator.simulated_web.test_session import records


class PreparationCyclesTests(unittest.TestCase):
    def test_paired_phases_visibility_continuity_and_costs(self):
        schedules = []
        for condition in CYCLES_CONDITIONS:
            seen = {'agent-1': [], 'agent-2': []}
            def client(agent, messages, timeout, **options):
                seen[agent].append((copy.deepcopy(messages), options))
                return ModelResponse({'content': 'Done', 'thinking': agent + '_MEMORY'},
                                     {'eval_count': 7, 'prompt_eval_count': 10})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                results = run_session(root, records(5), client, agents=2, seed=0,
                    prompt_condition=condition, protocol=CYCLES_PROTOCOL, preparation_topic='Topic')
                settings = json.loads((root / 'settings.json').read_text())
                schedules.append(settings['schedule'])
                initial = json.loads((root / 'initial-preparation/results.json').read_text())
                self.assertEqual(len(initial), 2)
                self.assertTrue(all(row['phase'] == 'initial_preparation' and 'id' not in row for row in initial))
                self.assertEqual(len(results), 10)
                self.assertTrue(all(len(row['phases']) == 2 for row in results))
                self.assertTrue(all(row['token_budget']['total_generated_token_limit'] == 6000 for row in results))
                usage = json.loads((root / 'session-usage.json').read_text())
                for agent, calls in seen.items():
                    self.assertEqual(len(calls), 11)
                    self.assertEqual([opts['num_predict'] for _, opts in calls], [8000] + [4000, 1744] * 5)
                    self.assertEqual(usage[agent]['metrics']['generated_tokens'], 77)
                    self.assertEqual(usage[agent]['initial_generated_tokens'], 7)
                    self.assertEqual(usage[agent]['answer_observed_generated_tokens'], 35)
                    for index, (messages, _) in enumerate(calls):
                        self.assertEqual(sum(m.get('thinking') == agent + '_MEMORY' for m in messages), index)
                        self.assertFalse(any(m.get('thinking') == ('agent-2' if agent == 'agent-1' else 'agent-1') + '_MEMORY' for m in messages))
                        for authored in (messages[0]['content'], messages[-1]['content']):
                            for cue in ('wiki', 'peer', 'team', 'write notes', 'read notes'):
                                self.assertNotIn(cue, authored.lower())
                    for record in records(5):
                        self.assertNotIn(record['question'], str(calls[0][0]))
                    order = settings['schedule']['orders'][agent]
                    questions = {r['id']: r['question'] for r in records(5)}
                    for slot, qid in enumerate(order):
                        prep = calls[1 + 2 * slot][0][-1]['content']
                        answer = calls[2 + 2 * slot][0][-1]['content']
                        self.assertIn(questions[qid], answer)
                        if condition == 'neutral-all-cost':
                            self.assertIn(questions[qid], prep)
                        else:
                            self.assertNotIn(questions[qid], prep)
                    exported = (root / f'predictions/{agent}.predictions.jsonl').read_text().splitlines()
                    self.assertEqual(len(exported), 5)
                    raw_counts = sum(json.loads(line).get('metadata', {}).get('eval_count', 0)
                                     for row in initial + results if row['agent'] == agent
                                     for line in (root / row['log_path']).read_text().splitlines()
                                     if json.loads(line)['event'] == 'model_response')
                    self.assertEqual(raw_counts, 77)
                objective = settings['system_prompt']
                self.assertIn('number_correct - 0.1 *', objective)
                self.assertIn('uncharged' if condition == 'neutral-answer-cost' else 'exactly once', objective)
        self.assertEqual(*schedules)

    def test_initial_failure_preserves_cost_and_skips_assignments(self):
        def client(*args, **kwargs):
            return ModelResponse({'content': 'Partial'}, {'eval_count': 8001, 'prompt_eval_count': 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'Initial preparation failed'):
                run_session(root, records(5), client, agents=2, prompt_condition='neutral-all-cost',
                            protocol=CYCLES_PROTOCOL, preparation_topic='Topic')
            initial = json.loads((root / 'initial-preparation/results.json').read_text())
            self.assertEqual(len(initial), 2)
            self.assertTrue(all(row['status'] == 'budget_error' for row in initial))
            self.assertTrue(all(row['metrics']['observed_generated_tokens'] == 8001 for row in initial))
            self.assertEqual(json.loads((root / 'results.json').read_text()), [])
            self.assertFalse((root / 'assignments').exists())
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'failed')

    def test_invalid_config_precedes_artifacts_and_model_calls(self):
        def client(*args, **kwargs):
            self.fail('Invalid configuration reached model')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            for options in ({'protocol': 'standard'}, {'protocol': 'preparation-urgency'},
                            {'prompt_condition': 'neutral'}, {'history_mode': 'reset'},
                            {'shards': 2}, {'question_id': 'q0'}, {'steps': 1},
                            {'total_token_budget': 2000}, {'preparation_topic': None}):
                kwargs = {'protocol': CYCLES_PROTOCOL, 'prompt_condition': 'neutral-answer-cost',
                          'preparation_topic': 'Topic', 'agents': 2, **options}
                with self.assertRaises(ValueError):
                    run_session(root, records(5), client, **kwargs)
                self.assertFalse(root.exists())
            for condition in CYCLES_CONDITIONS:
                with self.assertRaisesRegex(ValueError, 'exactly five'):
                    prepare(records(4), 2, 0, condition, protocol=CYCLES_PROTOCOL, preparation_topic='Topic')

    def test_full_caps_and_answer_reserve_total_38000(self):
        for condition in CYCLES_CONDITIONS:
            requests = []
            def client(agent, messages, timeout, **options):
                requests.append(options)
                if 'PREPARATION PHASE' in messages[-1]['content']:
                    return ModelResponse({'content': 'Prepared'},
                        {'eval_count': options['num_predict'], 'prompt_eval_count': 1})
                if options['final_only']:
                    return ModelResponse({'content': 'Answer'}, {'eval_count': 256, 'prompt_eval_count': 1})
                return ModelResponse({'content': '', 'thinking': 'partial'},
                    {'eval_count': 1744, 'prompt_eval_count': 1, 'done_reason': 'length'})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                rows = run_session(root, records(5), client, agents=1, prompt_condition=condition,
                    protocol=CYCLES_PROTOCOL, preparation_topic='Topic')
                self.assertEqual([request['num_predict'] for request in requests], [8000] + [4000, 1744, 256] * 5)
                self.assertTrue(all(row['metrics']['generated_tokens'] == 6000 for row in rows))
                usage = json.loads((root / 'session-usage.json').read_text())['agent-1']
                self.assertEqual(usage['metrics']['generated_tokens'], 38000)
                self.assertEqual(usage['answer_observed_generated_tokens'], 10000)
