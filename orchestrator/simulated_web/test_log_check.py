"""Three answers per agent over an unchanged full corpus, with mock transport."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.browser_retention import ANSWER_OMITTED
from orchestrator.simulated_web.log_exposure import LOG_URL
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair, validate_pair


class InvalidFinalClient(Client):
    def __call__(self, agent, history, timeout, **kwargs):
        self.calls.append((agent, json.loads(json.dumps(history))))
        if kwargs['final_only']:
            return ModelResponse({'content': 'not accepted', 'thinking': 'forbidden thought', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://wiki.test/save?slug=bad&title=bad&text=bad'}}}]},
                                 {'eval_count': 2, 'prompt_eval_count': 100, 'done_reason': 'stop', 'final_only_contract_violation': True})
        return ModelResponse({'content': '', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]},
                             {'eval_count': 3, 'prompt_eval_count': 100, 'done_reason': 'stop'})


class LogCheckTests(unittest.TestCase):
    def test_six_answers_full_corpus_logs_and_final_checkpoint(self):
        records, selectors = inputs()
        ids = [r['id'] for r in records[:3]]
        policy = pair_policy(compaction_enabled=False, browser_retention='question_boundary')
        pages, _, _, _ = validate_pair(records, 'topic', policy, selectors)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = Client()
            result = run_token_pair(root / 'run', records, 'topic', client, policy, selectors,
                                    pair_protocol='answers_only', question_ids=ids, log_exposure='forced')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_rounds'], 3)
            saved_pages = json.loads((root / 'run/pages.json').read_text())
            self.assertEqual(saved_pages, pages)
            settings = json.loads((root / 'run/settings.json').read_text())
            self.assertEqual(settings['selected_question_count'], 3)
            self.assertEqual(settings['corpus_question_count'], 10)
            self.assertEqual(settings['schedule']['orders'], {'agent-1': ids, 'agent-2': ids})
            self.assertIn('answer 3 related questions', settings['system_prompt'])
            self.assertNotIn('reflection', settings['system_prompt'].lower())
            self.assertNotIn('preparation', settings['system_prompt'].lower())
            rows = json.loads((root / 'run/results.json').read_text())
            self.assertEqual([(r['agent'],r['phase'],r['question_id']) for r in rows], [(a,'answer',q) for q in ids for a in ('agent-1','agent-2')])
            self.assertEqual(len(list((root / 'run').glob('forced-log-*'))), 6)
            first = json.loads((root / 'run/forced-log-00-agent-1.json').read_text())
            self.assertEqual(first['response']['text'], 'No requests.')
            second = json.loads((root / 'run/forced-log-01-agent-2.json').read_text())
            third = json.loads((root / 'run/forced-log-02-agent-1.json').read_text())
            self.assertIn('https://docs.test/', second['response']['text'])
            self.assertIn('https://docs.test/', third['response']['text'])
            loaded = load_pair_checkpoint(root / 'run/checkpoints/rounds-003')
            try:
                self.assertTrue(loaded.complete)
                self.assertEqual(len(loaded.data['results.json']), 6)
                self.assertEqual(loaded.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 12)
                for history in loaded.data['histories.json'].values():
                    self.assertTrue(all(m['content'] == ANSWER_OMITTED for m in history if m['role'] == 'tool'))
            finally:
                loaded.close()
            no_op = run_token_pair(root / 'unused', None, None, Client(), resume_from=root / 'run/checkpoints/rounds-003')
            self.assertEqual(no_op['status'], 'already_complete')
            self.assertFalse((root / 'unused').exists())

    def test_invalid_final_records_counts_without_tool_execution_and_continues(self):
        records, selectors = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = pair_policy(compaction_enabled=False, answer_browser_calls=1)
            result = run_token_pair(root / 'run', records, 'topic', InvalidFinalClient(), policy, selectors,
                                    pair_protocol='answers_only', question_ids=[r['id'] for r in records[:3]])
            self.assertEqual(result['status'], 'complete')
            rows = json.loads((root / 'run/results.json').read_text())
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['status'] == 'invalid_final_response' and r['answer'] == '' for r in rows))
            self.assertTrue(all(r['browser_calls'] == 1 and r['generated_tokens_observed'] == 5 for r in rows))
            loaded = load_pair_checkpoint(root / 'run/checkpoints/rounds-003')
            try:
                self.assertEqual(loaded.browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 0)
                self.assertEqual(loaded.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 6)
                self.assertIn('forbidden thought', json.dumps(loaded.data['histories.json']))
                log_path = root / 'run' / rows[0]['log_path']
                self.assertIn('not accepted', log_path.read_text())
                self.assertIn('slug=bad', log_path.read_text())
            finally:
                loaded.close()

    def test_invalid_selection_does_not_create_output(self):
        records, selectors = inputs()
        with tempfile.TemporaryDirectory() as directory:
            for ids in ([], ['missing'] * 3, [r['id'] for r in records[:2]]):
                path = Path(directory) / 'run'
                with self.assertRaises(ValueError):
                    run_token_pair(path, records, 'topic', Client(), pair_policy(), selectors,
                                   pair_protocol='answers_only', question_ids=ids)
                self.assertFalse(path.exists())
