"""Pair scheduler, failure and durable restore tests; only mocked model responses."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser

from orchestrator.simulated_web import modal_token_pair as launcher
from orchestrator.simulated_web.modal_token_pair import validate_resume_selector
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.token_pair import DIGEST, load_pair_checkpoint, pair_policy, run_token_pair, sequence, validate_pair


class Client:
    deadline_cancellation_guaranteed = True
    native_context_preflight = True

    def __init__(self):
        self.calls = []

    def inspect_model(self):
        return {'digest': DIGEST}

    def ensure_ready(self, **kwargs):
        return {'status': 'ready'}

    def count_context(self, *args, **kwargs):
        return {'prompt_tokens': 100}

    def __call__(self, agent, history, timeout, **kwargs):
        self.calls.append((agent, json.loads(json.dumps(history))))
        # Make a real Browser tool request once per phase, then return.
        if history[-1]['role'] == 'user':
            message = {'content': '', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]}
        else:
            message = {'content': f'{agent} private answer'}
        return ModelResponse(message, {'eval_count': 3, 'prompt_eval_count': 100, 'done_reason': 'stop'})


def inputs():
    data = records(10)
    selectors = []
    for row in data[:5]:
        p = row['paragraphs'][0]
        selectors.append({'title': p['title'], 'text_sha256': hashlib.sha256(p['paragraph_text'].encode()).hexdigest()})
    return data, selectors


class PairTests(unittest.TestCase):
    def run_pair(self, path, **kwargs):
        data, selectors = inputs()
        return run_token_pair(path, data, 'topic', kwargs.pop('client', Client()),
                              pair_policy(compaction_enabled=False), selectors, **kwargs)

    def test_schedule_private_history_and_checkpoint_restore(self):
        client = Client()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            callbacks = []
            result = self.run_pair(root / 'first', client=client, checkpoint_callback=lambda p: callbacks.append(p))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([p.name for p in callbacks], ['rounds-005', 'rounds-010'])
            settings = json.loads((root / 'first/settings.json').read_text())
            orders = settings['schedule']['orders']
            self.assertEqual(orders['agent-2'], orders['agent-1'][5:] + orders['agent-1'][:5])
            prompts = [h[-1]['content'] for _, h in client.calls if h[-1]['role'] == 'user']
            self.assertEqual(len(prompts), 42)
            reflections = [p for p in prompts if p.startswith('Reflection and continued research:')]
            self.assertEqual(len(reflections), 20)
            self.assertTrue(all('Formulate the check without assuming your previous answer is correct.' in p for p in reflections))
            self.assertTrue(all('corrections here' in p.lower() for p in reflections))
            self.assertTrue(all('generated tokens' in p for p in prompts))
            expected = sequence(settings['schedule'])
            rows = json.loads((root / 'first/results.json').read_text())
            self.assertEqual([(r['agent'], r['phase'], r['question_id']) for r in rows], [s[:3] for s in expected])
            for agent, history in client.calls:
                other = 'agent-2' if agent == 'agent-1' else 'agent-1'
                self.assertFalse(any(other + ' private answer' in m.get('content', '') for m in history))
            with_checkpoint = load_pair_checkpoint(callbacks[0])
            try:
                self.assertEqual(set(with_checkpoint.browser.views), {'agent-1', 'agent-2'})
                owners = {r[0] for r in with_checkpoint.browser.db.execute('SELECT owner FROM request_events')}
                self.assertEqual(owners, {'agent-1', 'agent-2'})
            finally:
                with_checkpoint.close()
            before = {p.name: p.read_bytes() for p in callbacks[0].iterdir()}
            resumed_client = Client()
            resumed = self.run_pair(root / 'child', client=resumed_client, resume_from=callbacks[0])
            self.assertEqual(resumed['completed_rounds'], 10)
            self.assertEqual(len(resumed_client.calls), 40)  # 20 phases, tool + answer each.
            self.assertIn(orders['agent-1'][5], (root / 'child/results.json').read_text())
            self.assertEqual(before, {p.name: p.read_bytes() for p in callbacks[0].iterdir()})
            no_op = self.run_pair(root / 'unused', resume_from=callbacks[1])
            self.assertEqual(no_op['status'], 'already_complete')
            self.assertFalse((root / 'unused').exists())

    def test_checkpoint_preserves_shared_edits_and_both_history_windows(self):
        initial = {}
        def factory(*args, **kwargs):
            browser = Browser(*args, **kwargs)
            identity = next(iter(browser.source_urls))
            browser.call('agent-2', 'open', {'url': 'https://docs.test/source/save?' + urlencode(
                {'source': identity, 'title': 'Edited shared title', 'text': 'retained shared source'})})
            for index in range(100):
                browser.call('agent-1', 'search', {'query': f'request-{index}'})
            for agent in ('agent-1', 'agent-2'):
                browser.call(agent, 'open', {'url': 'https://docs.test/request-history'})
            initial.update(identity=identity, windows=json.loads(json.dumps(browser.history_windows)))
            return browser
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'parent'
            with patch('orchestrator.simulated_web.token_pair.Browser', side_effect=factory):
                self.run_pair(destination)
            checkpoint = load_pair_checkpoint(destination / 'checkpoints/rounds-005')
            try:
                self.assertEqual(checkpoint.browser.history_windows, initial['windows'])
                self.assertEqual(checkpoint.browser.db.execute('SELECT body FROM source_pages WHERE identity=?',
                    (initial['identity'],)).fetchone()[0], 'retained shared source')
                self.assertEqual(checkpoint.browser.db.execute('SELECT agent FROM revisions').fetchall(), [('agent-2',)])
                for owner in ('agent-1', 'agent-2'):
                    view = checkpoint.browser.views[owner]
                    cursor = next(links[0]['url'] for links in view.values() if links and 'request-history?' in links[0]['url'])
                    self.assertNotIn('error', checkpoint.browser.call(owner, 'open', {'url': cursor}))
            finally:
                checkpoint.close()

    def test_invalid_inputs_create_no_output(self):
        data, selectors = inputs()
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'bad'
            with self.assertRaises(ValueError):
                run_token_pair(destination, data, 'topic', Client(), pair_policy(), selectors[:4])
            self.assertFalse(destination.exists())

    def test_failure_preserves_both_histories_and_stops(self):
        client = Client()
        client.count_context = lambda *args, **kwargs: {'prompt_tokens': 60000}
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'failed'
            with patch('orchestrator.simulated_web.token_pair.compact_between_questions', side_effect=ValueError('compaction failed')):
                data, selectors = inputs()
                with self.assertRaisesRegex(ValueError, 'compaction failed'):
                    run_token_pair(destination, data, 'topic', client, pair_policy(), selectors)
            histories = json.loads((destination / 'histories.json').read_text())
            self.assertEqual(set(histories), {'agent-1', 'agent-2'})
            self.assertEqual(json.loads((destination / 'manifest.json').read_text())['status'], 'failed')
            self.assertEqual(len(json.loads((destination / 'results.json').read_text())), 4)

    def test_commit_failure_retains_checkpoint_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'failed'
            def fail(path):
                raise OSError('commit failed')
            with self.assertRaisesRegex(OSError, 'commit failed'):
                self.run_pair(destination, checkpoint_callback=fail)
            checkpoint = load_pair_checkpoint(destination / 'checkpoints/rounds-005')
            checkpoint.close()
            self.assertEqual(len(json.loads((destination / 'results.json').read_text())), 22)

    def test_job_safety_stop_before_model_generation(self):
        client = Client()
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_pair(Path(temp) / 'stopped', client=client, job_deadline=0)
            self.assertEqual(result['status'], 'job_safety_stop')
            self.assertEqual(client.calls, [])

    def test_checkpoint_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'first'
            self.run_pair(destination)
            checkpoint = destination / 'checkpoints/rounds-005'
            (checkpoint / 'histories.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_pair_checkpoint(checkpoint)


class LauncherTests(unittest.TestCase):
    def test_default_is_read_only_and_budget_explicit(self):
        data, selectors = inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data.jsonl').write_text('\n'.join(json.dumps(r) for r in data))
            (root / 'topic.txt').write_text('topic')
            (root / 'selectors.json').write_text(json.dumps(selectors))
            (root / 'modal.toml').write_text('[research-profile]\n')
            args = ['--run-id', 'test', '--dataset', str(root / 'data.jsonl'), '--topic-file', str(root / 'topic.txt'), '--editable-sources', str(root / 'selectors.json')]
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(root / 'modal.toml')}), patch.object(launcher.app, 'run') as cloud, patch('builtins.print') as output:
                launcher.main(args)
                cloud.assert_not_called()
                report = json.loads(output.call_args.args[0])
                self.assertEqual(report['resources']['maximum_phase_generated_tokens'], 221184)
                self.assertEqual(report['resources']['hard_job_seconds'], 21600)
                self.assertFalse(report['remote_freshness_and_model_cache_checked'])
                with self.assertRaises(ValueError):
                    launcher.main(args + ['--expected-model-digest', '0' * 64])
                cloud.assert_not_called()

    def test_resume_selector_rejects_traversal_and_same_run(self):
        for selector in ('../parent/checkpoints/rounds-005', 'child/checkpoints/rounds-005', 'parent/checkpoints/rounds-004'):
            with self.assertRaises(ValueError):
                validate_resume_selector(selector, 'child')


if __name__ == '__main__':
    unittest.main()
