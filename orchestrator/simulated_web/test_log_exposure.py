"""Forced observation diagnostic tests; no model or cloud execution."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.browser_retention import OMITTED
from orchestrator.simulated_web.log_exposure import EXPOSURE_NOTICE, LOG_URL, expose_log
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair


class LogExposureTests(unittest.TestCase):
    def test_forced_order_budget_retention_and_resume(self):
        records, selectors = inputs()
        policy = pair_policy(compaction_enabled=False, browser_retention='question_boundary')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = Client()
            run_token_pair(root / 'forced', records, 'topic', client, policy, selectors, log_exposure='forced')
            normal_client = Client()
            run_token_pair(root / 'normal', records, 'topic', normal_client, policy, selectors)
            self.assertNotIn(EXPOSURE_NOTICE, json.dumps(normal_client.calls))
            forced = json.loads((root / 'forced/results.json').read_text())
            normal = json.loads((root / 'normal/results.json').read_text())
            self.assertEqual([(r['browser_calls'], r['generated_tokens_observed']) for r in forced],
                             [(r['browser_calls'], r['generated_tokens_observed']) for r in normal])
            self.assertEqual(len(list((root / 'forced').glob('forced-log-*.json'))), 22)
            self.assertEqual(list((root / 'normal').glob('forced-log-*.json')), [])
            for row in forced:
                self.assertEqual('forced_log_exposure' in row, row['phase'] in ('preparation', 'answer'))
            # Each first request sees the supplied browser exchange before the real phase prompt.
            first_requests = [h for _, h in client.calls if h[-1]['role'] == 'user']
            for row, history in zip(forced, first_requests):
                if row['phase'] in ('preparation', 'answer'):
                    self.assertEqual(history[-4]['content'], EXPOSURE_NOTICE)
                    self.assertEqual(history[-2]['tool_name'], 'open')
                    self.assertIn('Host-injected', history[-3]['content'])
            evidence = json.loads((root / 'forced/forced-log-01-agent-2.json').read_text())
            self.assertIn('https://docs.test/', evidence['response']['text'])
            self.assertEqual(evidence['forced_browser_calls'], 1)
            self.assertEqual(evidence['model_browser_calls'], 0)
            self.assertEqual(evidence['audit_after']['audit'] - evidence['audit_before']['audit'], 1)
            completed = load_pair_checkpoint(root / 'forced/checkpoints/rounds-010')
            try:
                count = completed.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0]
                self.assertEqual(count, sum(row['browser_calls'] for row in forced) + 22)
                audit = completed.browser.db.execute('SELECT agent,operation,args FROM audit WHERE id=?', (evidence['audit_after']['audit'],)).fetchone()
                self.assertEqual(audit[:2], ('agent-2', 'open'))
                self.assertEqual(json.loads(audit[2]), {'url': LOG_URL})
            finally:
                completed.close()
            histories = json.loads((root / 'forced/histories.json').read_text())
            for history in histories.values():
                injected = [m for m in history if m.get('tool_call_id', '').startswith('host-log-')]
                self.assertTrue(injected)
                self.assertTrue(all(m['content'] == OMITTED for m in injected))
            self.assertNotEqual(evidence['response'], OMITTED)
            checkpoint = root / 'forced/checkpoints/rounds-005'
            run_token_pair(root / 'resumed', None, None, Client(), resume_from=checkpoint)
            self.assertEqual(len(list((root / 'resumed').glob('forced-log-*.json'))), 10)
            loaded = load_pair_checkpoint(root / 'resumed/checkpoints/rounds-010')
            self.assertEqual(loaded.data['settings.json']['log_exposure'], 'forced')
            loaded.close()
            with self.assertRaises(ValueError):
                run_token_pair(root / 'bad', None, None, Client(), resume_from=checkpoint, log_exposure='spontaneous')

    def test_error_is_preserved_and_not_success(self):
        browser = Browser([{'url': 'https://docs.test/', 'title': 'Home', 'text': 'home'}], ':memory:')
        try:
            with tempfile.TemporaryDirectory() as directory:
                history = []
                with self.assertRaisesRegex(RuntimeError, 'Forced log exposure failed'):
                    expose_log(browser, history, Path(directory), 0, 'agent-1', 'preparation')
                event = json.loads((Path(directory) / 'forced-log-00-agent-1.json').read_text())
                self.assertEqual(event['status'], 'error')
                self.assertIn('error', json.loads(history[-1]['content']))
        finally:
            browser.close()

    def test_invalid_mode_before_output(self):
        records, selectors = inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid'
            with self.assertRaises(ValueError):
                run_token_pair(path, records, 'topic', Client(), pair_policy(), selectors, log_exposure='invalid')
            self.assertFalse(path.exists())
