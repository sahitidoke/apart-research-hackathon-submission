"""Finite mocked memory-policy contracts; no model, server or cloud execution."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser, TOOLS
from orchestrator.simulated_web.private_memory import PrivateScratchpad, UPDATE_TOOL
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web import test_modal_timed as modal_tests

from orchestrator.simulated_web.test_timed import Clock, Client
from orchestrator.simulated_web.test_timed_resume import MODEL, ResumeClient
from orchestrator.simulated_web.timed import TimedPolicy, run_phase, run_timed_session, resume_timed_session
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint
from orchestrator.simulated_web.timed_transport import OwnedOllama


modal_timed = modal_tests.modal_timed


def counted(text, tokens=None):
    return {'method': 'owned-native-text-tokenize', 'tokens': len(text.split()) if tokens is None else tokens,
            'add_special': False, 'parse_special': False, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()}


class MemoryClient(ResumeClient):
    native_context_preflight = True

    def count_text(self, text, timeout):
        return counted(text)

    def __call__(self, agent, history, timeout, **options):
        self.calls.append(json.loads(json.dumps(history)))
        if options['num_predict'] == 1:
            return ModelResponse({'content': 'DISCARDED WARMUP'}, {})
        if history[-1]['role'] == 'user' and history[-1]['content'].startswith('Preparation:'):
            return ModelResponse({'content': '', 'tool_calls': [{'function': {
                'name': 'private_scratchpad_update', 'arguments': {'text': 'SECRET_PREP_FACT'}}}]}, {})
        return ModelResponse({'content': 'UNSAVED_TRACE', 'thinking': 'UNSAVED_THINKING'}, {})


class MemoryTests(unittest.TestCase):
    def test_atomic_native_cap_deadline_invalid_and_agent_scope(self):
        memory = PrivateScratchpad('agent-1', 2)
        client = MagicMock()
        clock = Clock()
        client.count_text.return_value = counted('many characters', 1)
        self.assertEqual(memory.update('agent-1', {'text': 'many characters'}, client, 10, clock)['status'], 'updated')
        self.assertEqual(client.count_text.call_args.kwargs, {'timeout': 10})
        old = memory.snapshot()
        client.count_text.return_value = counted('x', 3)
        self.assertIn('cap exceeded', memory.update('agent-1', {'text': 'x'}, client, 10, clock)['error'])
        self.assertEqual(memory.snapshot(), old)
        for args in ({'text': 5}, {'text': '', 'agent': 'agent-2'}, []):
            self.assertIn('error', memory.update('agent-1', args, client, 10, clock))
        self.assertEqual(memory.update('agent-2', {'text': 'stolen'}, client, 10, clock), {'error': 'Private memory access denied'})
        client.count_text.return_value = {'tokens': 1}
        self.assertIn('error', memory.update('agent-1', {'text': 'x'}, client, 10, clock))
        def late(text, timeout):
            clock.now = 11
            return counted(text)
        client.count_text.side_effect = late
        self.assertIn('deadline', memory.update('agent-1', {'text': 'x'}, client, 10, clock)['error'])
        self.assertEqual(memory.snapshot(), old)
        clock.now = 0
        client.count_text.side_effect = None
        client.count_text.return_value = counted('', 0)
        self.assertEqual(memory.update('agent-1', {'text': ''}, client, 10, clock)['tokens'], 0)
        self.assertEqual(memory.text, '')

    def test_reset_checkpoint_resume_and_shared_state_are_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = records(10)
            paragraph = source[0]['paragraphs'][0]
            selectors = [{'title': paragraph['title'], 'text_sha256': hashlib.sha256(paragraph['paragraph_text'].encode()).hexdigest()}]
            def factory(*args, **kwargs):
                browser = Browser(*args, **kwargs)
                browser.call('agent-1', 'open', {'url': 'https://docs.test/source/save?' + urlencode({
                    'source': browser.source_identity, 'title': 'edited', 'text': 'SHARED_EDIT'})})
                return browser
            client = MemoryClient()
            policy = TimedPolicy(memory_mode='private_scratchpad')
            with patch('orchestrator.simulated_web.timed.Browser', side_effect=factory), patch('orchestrator.simulated_web.timed.compact_between_questions') as compact, redirect_stdout(io.StringIO()):
                run_timed_session(root / 'parent', source, 'TOPIC', client, policy, selectors, {'model': MODEL})
            compact.assert_not_called()
            first_answer = next(h for h in client.calls if h[-1]['content'].startswith('Answer phase:'))
            self.assertIn('Preparation:', json.dumps(first_answer))
            answers = [h for h in client.calls if h[-1]['content'].startswith('Answer phase:')]
            self.assertEqual(len(answers[1]), 3)
            self.assertIn('SECRET_PREP_FACT', json.dumps(answers[1]))
            self.assertIn('TOPIC', answers[1][1]['content'])
            self.assertEqual(answers[1][0], first_answer[0])
            self.assertNotIn('other agents', first_answer[0]['content'])
            self.assertNotIn('peers', first_answer[0]['content'])
            self.assertNotIn('UNSAVED_TRACE', json.dumps(answers[1]))
            self.assertNotIn('UNSAVED_THINKING', json.dumps(answers[1]))
            self.assertNotIn(first_answer[-1]['content'], json.dumps(answers[1]))
            history = json.loads((root / 'parent/history.json').read_text())
            self.assertEqual(len(history), 2)  # Final reflection resets too.
            self.assertEqual(len(list((root / 'parent').glob('memory-reset-*.json'))), 10)
            checkpoint = root / 'parent/checkpoints/questions-005'
            loaded = load_checkpoint(checkpoint)
            try:
                self.assertEqual(loaded.data['private-scratchpad.json']['text'], 'SECRET_PREP_FACT')
                self.assertNotIn('SECRET_PREP_FACT', '\n'.join(loaded.browser.db.iterdump()))
                self.assertIn('SHARED_EDIT', loaded.browser.open('agent-2', next(iter(loaded.browser.editable_urls)))['text'])
                self.assertNotIn('SECRET_PREP_FACT', json.dumps(loaded.browser.call('agent-2', 'search', {'query': 'SECRET_PREP_FACT'})))
                self.assertIn('error', loaded.browser.call('agent-2', 'private_scratchpad_update', {'text': 'x'}))
            finally:
                loaded.close()
            child = MemoryClient()
            with redirect_stdout(io.StringIO()):
                resume_timed_session(root / 'child', checkpoint, child)
            self.assertIn('SECRET_PREP_FACT', json.dumps(child.calls[0]))
            self.assertIn('TOPIC', json.loads((root / 'child/history.json').read_text())[1]['content'])
            self.assertEqual(json.loads((root / 'child/settings.json').read_text())['policy']['memory_mode'], 'private_scratchpad')
            final = load_checkpoint(root / 'child/checkpoints/questions-010')
            final.close()
            rows = json.loads((root / 'parent/results.json').read_text())
            self.assertEqual(rows[0]['scratchpad_calls'], 1)
            self.assertEqual(rows[0]['browser_calls'], 0)
            self.assertIn('scratchpad_update', (root / 'parent/phase-00.jsonl').read_text())

    def test_validation_before_output_and_legacy_checkpoint_defaults(self):
        for kwargs in ({'memory_mode': 'bad'}, {'scratchpad_tokens': True}, {'scratchpad_tokens': 0},
                       {'memory_mode': 'private_scratchpad', 'context_length': 4096}):
            with self.assertRaises(ValueError):
                TimedPolicy(**kwargs)
        TimedPolicy(context_length=4096, compaction_enabled=False)  # Existing default remains valid.
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'bad'
            with self.assertRaisesRegex(ValueError, 'native text'):
                run_timed_session(destination, records(10), 'topic', Client(), TimedPolicy(memory_mode='private_scratchpad'))
            self.assertFalse(destination.exists())
            with redirect_stdout(io.StringIO()):
                run_timed_session(Path(folder) / 'old', records(10), 'topic', Client(), TimedPolicy(compaction_enabled=False))
            checkpoint = Path(folder) / 'old/checkpoints/questions-005'
            settings_path = checkpoint / 'settings.json'
            settings = json.loads(settings_path.read_text())
            del settings['policy']['memory_mode']
            del settings['policy']['scratchpad_tokens']
            settings_path.write_text(json.dumps(settings))
            manifest_path = checkpoint / 'checkpoint.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['settings.json'] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            loaded = load_checkpoint(checkpoint)
            loaded.close()

    def test_updates_charge_time_and_preserve_failure_diagnostics(self):
        for failure in ('late', 'native_failure', 'malformed'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                clock = Clock()
                memory = PrivateScratchpad('agent-1', 2, 'old', 1, counted('old'))
                client = MagicMock()
                client.native_context_preflight = True
                args = '{broken' if failure == 'malformed' else {'text': 'new'}
                client.return_value = ModelResponse({'content': '', 'tool_calls': [{'function': {
                    'name': 'private_scratchpad_update', 'arguments': args}}]}, {})
                def count(text, timeout):
                    self.assertEqual(timeout, 20)
                    if failure == 'native_failure':
                        raise RuntimeError('native identity failed')
                    clock.now = 21
                    return counted(text)
                client.count_text.side_effect = count
                browser = MagicMock()
                path = Path(folder) / 'phase.jsonl'
                row = run_phase(browser, client, [], 'reflect', 'reflection', 20,
                                TimedPolicy(memory_mode='private_scratchpad'), path, clock, memory)
                self.assertEqual(memory.text, 'old')
                browser.call.assert_not_called()
                self.assertEqual(row['browser_calls'], 0)
                self.assertIn('scratchpad_update_attempt', path.read_text())
                if failure == 'native_failure':
                    self.assertEqual(row['status'], 'error')
                    self.assertIn('scratchpad_update_failed', path.read_text())
                elif failure == 'late':
                    self.assertEqual(row['status'], 'deadline_reached')
                    self.assertEqual(row['elapsed_seconds'], 21)
                else:
                    client.count_text.assert_not_called()
                    self.assertIn('Expected exactly one string', path.read_text())

    @unittest.skipIf(modal_timed is None, 'Optional Modal SDK not installed')
    def test_cli_memory_flags_validate_without_cloud_and_resume_rejects_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            args = modal_tests.ModalValidationTests().inputs(folder)
            config = Path(folder) / 'modal.toml'
            config.write_text('[research-profile]\n')
            output = io.StringIO()
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(config)}), patch.object(modal_timed.app, 'run') as run, redirect_stdout(output):
                modal_timed.main(args + ['--memory-mode', 'private_scratchpad', '--scratchpad-tokens', '1024'])
                run.assert_not_called()
            policy = json.loads(output.getvalue())['policy']
            self.assertEqual(policy['memory_mode'], 'private_scratchpad')
            self.assertEqual(policy['scratchpad_tokens'], 1024)
            for flag in (['--memory-mode', 'private_scratchpad'], ['--scratchpad-tokens', '1024']):
                with self.assertRaisesRegex(ValueError, 'does not accept'):
                    modal_timed.main(['--run-id', 'child', '--resume-from', 'parent/checkpoints/questions-005', '--validate-only', *flag])

    def test_actual_request_tool_schema_is_opt_in_and_matches_preflight(self):
        for mode in ('in_context', 'private_scratchpad'):
            with self.subTest(mode=mode):
                with patch('orchestrator.simulated_web.timed_transport.sys.platform', 'linux'), patch('orchestrator.simulated_web.timed_transport.shutil.which', return_value='/mock/ollama'):
                    client = OwnedOllama(TimedPolicy(memory_mode=mode), '/tmp/model', Path('/tmp/mock.log'))
                client.metadata, client.ready = MODEL, True
                response = io.BytesIO(b'{"message":{"content":"answer"},"done":true,"prompt_eval_count":100}\n')
                response.status = 200
                connection = MagicMock()
                connection.getresponse.return_value = response
                with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}) as preflight, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
                    client('agent-1', [], 20, num_predict=10)
                sent = json.loads(connection.request.call_args.args[2])['tools']
                self.assertEqual(sent, preflight.call_args.args[0]['tools'])
                self.assertEqual(UPDATE_TOOL in sent, mode == 'private_scratchpad')
                self.assertEqual([t for t in sent if t != UPDATE_TOOL], TOOLS)

    def test_raw_native_tokenizer_contract_and_no_generation(self):
        with patch('orchestrator.simulated_web.timed_transport.sys.platform', 'linux'), patch('orchestrator.simulated_web.timed_transport.shutil.which', return_value='/mock/ollama'):
            client = OwnedOllama(TimedPolicy(memory_mode='private_scratchpad'), '/tmp/model', Path('/tmp/mock.log'))
        client.metadata, client.ready = MODEL, True
        connection = MagicMock()
        connection.getresponse.return_value.status = 200
        connection.getresponse.return_value.read.return_value = b'{"tokens":[1,2,3]}'
        identity = (42, '100', 1234)
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=(identity, 7)), patch.object(client, '_runner_identity', return_value=identity), patch.object(client, '_count_prompt') as count_prompt, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
            result = client.count_text('<|im_end|> hello', timeout=1)
        self.assertEqual(result['tokens'], 3)
        count_prompt.assert_not_called()
        connection.request.assert_called_once()
        args = connection.request.call_args.args
        self.assertEqual(args[:2], ('POST', '/tokenize'))
        self.assertEqual(json.loads(args[2]), {'content': '<|im_end|> hello', 'add_special': False, 'parse_special': False})
        self.assertNotIn(UPDATE_TOOL, TOOLS)
