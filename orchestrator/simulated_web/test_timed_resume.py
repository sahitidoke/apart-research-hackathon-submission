"""Finite checkpoint continuation tests with fake models; no inference or cloud."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.test_timed import Client
from orchestrator.simulated_web.timed import TimedPolicy, run_timed_session, resume_timed_session
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint


MODEL = {'name': 'mock', 'digest': 'a' * 64}


class ResumeClient(Client):
    def __init__(self):
        self.calls = []
        self.events = []

    def inspect_model(self):
        return MODEL

    def ensure_ready(self, timeout):
        self.events.append('ready')
        return {'status': 'ready'}

    def count_context(self, history, timeout):
        self.events.append('count')
        return {'prompt_tokens': 100}

    def __call__(self, agent, history, timeout, **options):
        self.calls.append(json.loads(json.dumps(history)))
        if options['num_predict'] == 1:
            self.events.append('warm')
            return ModelResponse({'content': 'DISCARDED WARMUP'}, {})
        self.events.append('phase')
        return super().__call__(agent, history, timeout, **options)


class ResumeTests(unittest.TestCase):
    def make_parent(self, root, count=20, compaction=False):
        source = records(count)
        paragraph = source[0]['paragraphs'][0]
        selector = {'title': paragraph['title'],
                    'text_sha256': hashlib.sha256(paragraph['paragraph_text'].encode()).hexdigest()}
        def browser_factory(*args, **kwargs):
            browser = Browser(*args, **kwargs)
            url = next(iter(browser.editable_urls))
            browser.call('agent-1', 'open', {'url': url})  # Retain p1's Edit link.
            identity = browser.source_identity
            browser.call('agent-1', 'open', {'url': 'https://docs.test/source/save?' + urlencode(
                {'source': identity, 'title': 'Saved title', 'text': 'Saved edited evidence'})})
            return browser
        def compact(client, history, policy, run_dir, index):
            if compaction:
                history[:] = [history[0], {'role': 'user', 'content': f'private memory {index}'}]
        with patch('orchestrator.simulated_web.timed.Browser', side_effect=browser_factory), patch('orchestrator.simulated_web.timed.compact_between_questions', side_effect=compact), redirect_stdout(io.StringIO()):
            run_timed_session(root / 'parent', source, 'topic', ResumeClient(),
                              TimedPolicy(question_count=count, compaction_enabled=compaction), [selector],
                              {'model': MODEL})
        return root / 'parent/checkpoints/questions-010'

    def test_resume_starts_q11_restores_browser_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint_path = self.make_parent(root)
            parent_before = {str(p.relative_to(root)): p.read_bytes() for p in (root / 'parent').rglob('*') if p.is_file()}
            loaded = load_checkpoint(checkpoint_path)
            try:
                browser = loaded.browser
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 1)
                editor = browser.call('agent-1', 'click', {'page_id': 'p1', 'link_id': 1})
                self.assertEqual(editor['title'], 'Edit source')
                self.assertIn('Saved edited evidence', browser.open('agent-1', next(iter(browser.editable_urls)))['text'])
            finally:
                loaded.close()
            client = ResumeClient()
            callbacks = []
            destination = root / 'resumed'
            with redirect_stdout(io.StringIO()):
                result = resume_timed_session(destination, checkpoint_path, client, checkpoint_callback=callbacks.append)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(client.calls), 21)  # One discarded warmup, ten answer/reflection pairs.
            order = json.loads((destination / 'settings.json').read_text())['schedule']['orders']['agent-1']
            data = {r['id']: r for r in records(20)}
            self.assertIn(data[order[10]]['question'], client.calls[1][-1]['content'])
            self.assertTrue(client.calls[1][-1]['content'].startswith('Answer phase:'))
            self.assertEqual(client.events[:4], ['ready', 'count', 'warm', 'ready'])
            self.assertNotIn('DISCARDED WARMUP', (destination / 'history.json').read_text())
            self.assertEqual(sorted(p.name for p in destination.glob('phase-*.jsonl')), [f'phase-{i:02d}.jsonl' for i in range(21, 41)])
            self.assertEqual(len(json.loads((destination / 'results.json').read_text())), 41)
            self.assertEqual([p.name for p in callbacks], ['questions-015', 'questions-020'])
            self.assertEqual(parent_before, {str(p.relative_to(root)): p.read_bytes() for p in (root / 'parent').rglob('*') if p.is_file()})
            final = load_checkpoint(destination / 'checkpoints/questions-020')
            final.close()

    def test_five_question_checkpoint_resumes_sixth_and_old_ten_interval_still_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tenth = self.make_parent(root, count=10)
            fifth = tenth.with_name('questions-005')
            client = ResumeClient()
            with redirect_stdout(io.StringIO()):
                result = resume_timed_session(root / 'resumed-five', fifth, client)
            self.assertEqual(result['completed_questions'], 10)
            self.assertEqual(len(client.calls), 11)  # Warmup then five answer/reflection pairs.
            settings = json.loads((root / 'resumed-five/settings.json').read_text())
            order = settings['schedule']['orders']['agent-1']
            questions = {row['id']: row['question'] for row in records(10)}
            self.assertIn(questions[order[5]], client.calls[1][-1]['content'])
            self.assertEqual(settings['checkpoint_interval_questions'], 5)
            self.assertEqual(sorted(p.name for p in (root / 'resumed-five').glob('phase-*.jsonl')),
                             [f'phase-{i:02d}.jsonl' for i in range(11, 21)])
            self.assertTrue((root / 'resumed-five/checkpoints/questions-010/checkpoint.json').is_file())
            # V1 historical snapshots recorded interval 10; their boundaries remain valid.
            settings_path = tenth / 'settings.json'
            old_settings = json.loads(settings_path.read_text())
            old_settings['checkpoint_interval_questions'] = 10
            settings_path.write_text(json.dumps(old_settings))
            manifest_path = tenth / 'checkpoint.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['settings.json'] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            loaded = load_checkpoint(tenth)
            self.assertTrue(loaded.complete)
            loaded.close()

    def test_compacted_history_resumes_at_original_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.make_parent(root, compaction=True)
            client = ResumeClient()
            with patch('orchestrator.simulated_web.timed.compact_between_questions') as compact, redirect_stdout(io.StringIO()):
                resume_timed_session(root / 'resumed', checkpoint, client)
            self.assertEqual(client.calls[0][-1]['content'], 'private memory 20')
            self.assertEqual([call.args[-1] for call in compact.call_args_list], list(range(22, 40, 2)))

    def test_completed_checkpoint_is_loadable_and_resume_is_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.make_parent(root, count=10)
            loaded = load_checkpoint(checkpoint)
            self.assertTrue(loaded.complete)
            loaded.close()
            result = resume_timed_session(root / 'unused', checkpoint)
            self.assertEqual(result['status'], 'already_complete')
            self.assertFalse((root / 'unused').exists())

    def test_tampering_invalid_progress_and_model_fail_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.make_parent(root)
            path = checkpoint / 'history.json'
            original = path.read_bytes()
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                resume_timed_session(root / 'bad', checkpoint, ResumeClient())
            self.assertFalse((root / 'bad').exists())
            path.write_bytes(original)
            manifest_path = checkpoint / 'checkpoint.json'
            manifest = json.loads(manifest_path.read_text())
            # Historical v1 snapshots with this flag false remain loadable.
            manifest['resume_implemented'] = False
            manifest['next_phase_index'] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'progress boundary'):
                resume_timed_session(root / 'bad', checkpoint, ResumeClient())
            self.assertFalse((root / 'bad').exists())
            manifest['next_phase_index'] = 21
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(ResumeClient, 'inspect_model', return_value={'digest': 'b' * 64}):
                with self.assertRaisesRegex(ValueError, 'model digest differs'):
                    resume_timed_session(root / 'bad', checkpoint, ResumeClient())
            self.assertFalse((root / 'bad').exists())
            settings_path = checkpoint / 'settings.json'
            settings = json.loads(settings_path.read_text())
            settings['policy']['context_length'] = 0
            settings_path.write_text(json.dumps(settings))
            manifest['files_sha256']['settings.json'] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint)

    def test_warmup_failure_preserves_diagnostics_and_starts_no_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.make_parent(root)
            with patch.object(ResumeClient, 'count_context', return_value={'prompt_tokens': 65536}), patch('orchestrator.simulated_web.timed.run_phase') as phase:
                with self.assertRaisesRegex(ValueError, 'cannot fit'):
                    resume_timed_session(root / 'failed', checkpoint, ResumeClient())
                phase.assert_not_called()
            self.assertEqual(json.loads((root / 'failed/resume-warmup.json').read_text())['status'], 'failed')
            self.assertEqual(json.loads((root / 'failed/manifest.json').read_text())['status'], 'failed')
