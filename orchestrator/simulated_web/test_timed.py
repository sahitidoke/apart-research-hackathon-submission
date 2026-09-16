"""Deterministic mocked deadline, history, source and identity contracts; no models."""
import hashlib
import json
import io
from pathlib import Path
import tempfile
import socket
import sqlite3
import threading
import time
import unittest
from urllib.parse import urlencode
from unittest.mock import MagicMock, patch

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.timed import TimedPolicy, run_phase, run_timed_session, save_checkpoint, validate_inputs
from orchestrator.simulated_web.timed_transport import ContextExhausted, DeadlineExpired, FINAL_PREFILL, FINAL_RENDER_SUFFIX, MODEL, OwnedOllama, check_loopback_port_available, validate_model_metadata


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class Client:
    deadline_cancellation_guaranteed = True

    def __call__(self, agent, history, timeout, **kwargs):
        return ModelResponse({'content': 'retained answer'}, {'eval_count': 3, 'prompt_eval_count': 100})


class TimedTests(unittest.TestCase):
    def test_checkpoints_capture_post_compaction_state_and_final_reflection(self):
        snapshots = []
        snapshot_bytes = []
        browsers = []
        source = records(20)
        def make_browser(*args, **kwargs):
            browser = Browser(*args, **kwargs)
            browser.open('agent-1', 'https://docs.test/')
            browsers.append(browser)
            return browser
        def compact(client, history, policy, run_dir, index):
            browsers[0].call('agent-1', 'open', {'url': 'https://docs.test/'})
            if index == 22:
                self.assertEqual(len(snapshots), 2)  # Q5/Q10 published before Q11 advances.
            history[:] = [history[0], {'role': 'user', 'content': f'memory after phase {index}'}]
        def checkpoint_published(path):
            manifest = json.loads((path / 'checkpoint.json').read_text())
            snapshots.append(manifest)
            snapshot_bytes.append({item.name: item.read_bytes() for item in path.iterdir()})
            for name, digest in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), digest)
            state = json.loads((path / 'browser.json').read_text())
            self.assertEqual(state['views'], browsers[0].views)
            with sqlite3.connect(path / 'wiki.sqlite3') as db:
                self.assertEqual(list(db.iterdump()), list(browsers[0].db.iterdump()))
            self.assertEqual(len(json.loads((path / 'results.json').read_text())),
                             manifest['completed_questions'] * 2 + 1)
        client = Client()
        client.count_context = lambda *args: None
        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.Browser', side_effect=make_browser), patch('orchestrator.simulated_web.timed.compact_between_questions', side_effect=compact) as compactor:
            path = Path(folder) / 'run'
            run_timed_session(path, source, 'topic', client, TimedPolicy(question_count=20),
                              checkpoint_callback=checkpoint_published)
            first = path / 'checkpoints/questions-005'
            last = path / 'checkpoints/questions-020'
            first_history = json.loads((first / 'history.json').read_text())
            last_history = json.loads((last / 'history.json').read_text())
            self.assertEqual(first_history[-1]['content'], 'memory after phase 10')
            self.assertIn('no further questions remain', last_history[-2]['content'])
            self.assertEqual(compactor.call_count, 19)
            self.assertEqual([item['completed_questions'] for item in snapshots], [5, 10, 15, 20])
            self.assertEqual([item['next_phase_index'] for item in snapshots], [11, 21, 31, 41])
            self.assertEqual(snapshots[0]['next_phase'], 'answer')
            self.assertIsNotNone(snapshots[0]['next_question_id'])
            self.assertTrue(snapshots[-1]['session_complete'])
            self.assertIsNone(snapshots[-1]['next_question_id'])
            self.assertTrue(snapshots[0]['resume_implemented'])
            self.assertEqual({item.name: item.read_bytes() for item in first.iterdir()}, snapshot_bytes[0])
            first_views = json.loads((first / 'browser.json').read_text())['views']
            last_views = json.loads((last / 'browser.json').read_text())['views']
            self.assertGreater(len(last_views['agent-1']), len(first_views['agent-1']))
            with sqlite3.connect(first / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM audit').fetchone()[0], 5)
            with sqlite3.connect(last / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM audit').fetchone()[0], 19)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                save_checkpoint(path, None, [], 10, 21, 'unused')

    def test_checkpoint_snapshot_failure_preserves_diagnostics_without_publication(self):
        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.shutil.copyfile', side_effect=OSError('snapshot copy failed')):
            path = Path(folder) / 'run'
            with self.assertRaisesRegex(OSError, 'snapshot copy failed'):
                run_timed_session(path, records(10), 'topic', Client(), TimedPolicy(compaction_enabled=False))
            self.assertFalse((path / 'checkpoints/questions-010').exists())
            failure = path / 'checkpoints/.questions-005.incomplete/failure.json'
            self.assertEqual(json.loads(failure.read_text())['status'], 'incomplete')
            self.assertTrue((path / 'history.json').is_file())
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['status'], 'failed')

    def test_checkpoint_commit_failure_stops_but_keeps_complete_local_snapshot(self):
        def fail_commit(path):
            self.assertTrue((path / 'checkpoint.json').is_file())
            raise OSError('volume commit failed')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run'
            with self.assertRaisesRegex(OSError, 'volume commit failed'):
                run_timed_session(path, records(10), 'topic', Client(), TimedPolicy(compaction_enabled=False),
                                  checkpoint_callback=fail_commit)
            self.assertTrue((path / 'checkpoints/questions-005/checkpoint.json').is_file())
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['status'], 'failed')

    def test_session_search_marks_editable_sources_without_snippets(self):
        observed = []
        source = records(10)
        paragraph = source[0]['paragraphs'][0]
        selector = {'title': paragraph['title'], 'text_sha256': hashlib.sha256(paragraph['paragraph_text'].encode()).hexdigest()}
        def make_browser(*args, **kwargs):
            browser = Browser(*args, **kwargs)
            observed.append(browser.search(paragraph['title']))
            self.assertFalse(browser.search_snippets)
            self.assertTrue(browser.editable_title_marker)
            return browser
        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.Browser', side_effect=make_browser):
            run_dir = Path(folder) / 'run'
            run_timed_session(run_dir, source, 'topic', Client(), TimedPolicy(compaction_enabled=False), [selector])
            settings = json.loads((run_dir / 'settings.json').read_text())
        self.assertFalse(settings['search_snippets'])
        self.assertTrue(settings['editable_title_marker'])
        for response in observed:
            self.assertTrue(response['results'])
            self.assertTrue(any(hit['title'].endswith(' [Editable]') for hit in response['results']))
            for hit in response['results']:
                self.assertEqual(set(hit), {'title', 'url'})

    def phase(self, client, clock, phase='answer', browser=None, policy=None):
        policy = policy or TimedPolicy()
        history = []
        with tempfile.TemporaryDirectory() as folder:
            row = run_phase(browser, client, history, 'Current question?', phase, 20, policy,
                            Path(folder) / 'log.jsonl', clock)
            log = (Path(folder) / 'log.jsonl').read_text()
        return row, history, log

    def test_confirmed_kill_misses_answer_then_recovers_before_next_phase(self):
        clock = Clock()
        class RecoveringClient(Client):
            def __init__(self):
                self.ready = True
                self.calls = 0
                self.recoveries = []
                self.options = []
            def ensure_ready(self, timeout):
                if not self.ready:
                    self.recoveries.append(clock.now)
                    clock.now += 30  # Outside the next phase clock.
                    self.ready = True
                return {'status': 'ready'}
            def __call__(self, agent, history, timeout, **options):
                if not self.ready:
                    raise AssertionError('Generation attempted before between-phase recovery')
                self.calls += 1
                self.options.append(options)
                if self.calls == 2:  # First answer research, after preparation.
                    clock.now += timeout + 2.1
                    self.ready = False
                    error = DeadlineExpired({'thinking': 'useful interrupted reasoning'}, 2.1)
                    error.cancellation = {'event': 'request_cancelled_killed', 'server_retained': False,
                                          'elapsed_seconds': 2.1}
                    raise error
                return super().__call__(agent, history, timeout, **options)
        client = RecoveringClient()
        original_phase = run_phase
        def phase_with_clock(*args, **kwargs):
            return original_phase(*args, **kwargs, clock=clock)
        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.run_phase', side_effect=phase_with_clock):
            run_dir = Path(folder) / 'run'
            rows = run_timed_session(run_dir, records(10), 'topic', client, policy=TimedPolicy(compaction_enabled=False))
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            history = json.loads((run_dir / 'history.json').read_text())
        missed = rows[1]
        self.assertEqual(missed['status'], 'deadline_reached')
        self.assertIn('cancellation_fallback', missed['limits_reached'])
        self.assertEqual(missed['final_skipped_reason'], 'confirmed_cancellation_fallback')
        self.assertFalse(missed['final_attempted'])
        self.assertEqual(missed['answer'], '')
        self.assertAlmostEqual(missed['elapsed_seconds'], 17.1)
        self.assertEqual(client.recoveries, [17.1])
        self.assertEqual(rows[2]['status'], 'complete')
        self.assertEqual(rows[2]['elapsed_seconds'], 0)
        self.assertEqual(len(rows), 21)
        self.assertEqual(client.calls, 21)
        self.assertFalse(any(option['final_only'] for option in client.options))
        self.assertTrue(any(message.get('thinking') == 'useful interrupted reasoning' for message in history))
        self.assertEqual(manifest['status'], 'complete')

    def test_unconfirmed_stop_and_unknown_errors_still_abort_session(self):
        for error in (RuntimeError('Deadline cleanup failed; no further inference is safe'), ValueError('unknown backend failure')):
            with self.subTest(error=str(error)):
                client = MagicMock(side_effect=[ModelResponse({'content': 'prep'}, {}), error])
                client.deadline_cancellation_guaranteed = True
                client.native_context_preflight = False
                client.ensure_ready = MagicMock(return_value={'status': 'ready'})
                with tempfile.TemporaryDirectory() as folder:
                    run_dir = Path(folder) / 'run'
                    with self.assertRaises(RuntimeError):
                        run_timed_session(run_dir, records(10), 'topic', client, policy=TimedPolicy(compaction_enabled=False))
                    rows = json.loads((run_dir / 'results.json').read_text())
                    self.assertEqual(rows[-1]['status'], 'error')
                    self.assertEqual(json.loads((run_dir / 'manifest.json').read_text())['status'], 'failed')
                self.assertEqual(client.call_count, 2)
                self.assertEqual(client.ensure_ready.call_count, 2)

    def test_answer_research_timeout_uses_remaining_window_final_reserve(self):
        clock, requests = Clock(), []

        def client(agent, messages, remaining, **options):
            requests.append((clock.now, remaining, options))
            if not options['final_only']:
                clock.now += remaining
                raise DeadlineExpired({'thinking': 'useful partial', 'tool_calls': [{'broken': True}]}, 0)
            clock.now += 2
            return ModelResponse({'content': 'answer'}, {'eval_count': 1})

        row, history, log = self.phase(client, clock)
        self.assertEqual([(r[0], r[1]) for r in requests], [(0, 15), (15, 5)])
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['elapsed_seconds'], 17)
        self.assertTrue(any(m.get('thinking') == 'useful partial' for m in history))
        self.assertFalse(any(m.get('tool_calls') for m in history))
        self.assertIn('time_reserve', log)
        self.assertEqual(history[-2]['content'],
                         'Finalize now. Give only the shortest complete answer to the current question using the information already available. Do not make further tool calls.')
        self.assertFalse(row['token_accounting_complete'])

    def test_cleanup_overrun_cannot_start_final_outside_deadline(self):
        clock = Clock()
        calls = []

        def client(*args, **kwargs):
            calls.append(kwargs)
            clock.now = 22
            raise DeadlineExpired({}, 7)

        row, _, _ = self.phase(client, clock)
        self.assertEqual(len(calls), 1)
        self.assertEqual(row['status'], 'deadline_reached')
        self.assertEqual(row['deadline_overrun_seconds'], 2)

    def test_timed_preparation_and_reflection_are_usable_transitions(self):
        for phase in ('preparation', 'reflection'):
            clock = Clock()

            def client(agent, messages, remaining, **kwargs):
                clock.now += remaining
                raise DeadlineExpired({'content': 'partial'}, 0)

            row, history, _ = self.phase(client, clock, phase)
            self.assertEqual(row['status'], 'deadline_reached')
            self.assertEqual(history[-1]['content'], 'partial')

    def test_history_keeps_complete_answer_blocks_and_future_questions_hidden(self):
        source = records(10)
        observed = []

        class RecordingClient(Client):
            def __call__(self, agent, history, remaining, **kwargs):
                observed.append(json.loads(json.dumps(history)))
                return super().__call__(agent, history, remaining, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run'
            rows = run_timed_session(path, source, 'Topic only', RecordingClient(), policy=TimedPolicy(compaction_enabled=False))
            self.assertEqual(len(rows), 21)
            history = json.loads((path / 'history.json').read_text())
            self.assertEqual(sum(m.get('content') == 'retained answer' for m in history), 21)
            self.assertNotIn('Question ', json.dumps(observed[0]))
            self.assertNotIn('PRIVATE_', json.dumps(observed))
            self.assertNotIn('wiki', observed[0][0]['content'])
            self.assertNotIn('editable', observed[0][0]['content'].lower())
            system = observed[0][0]['content']
            self.assertIn('answer 10 related questions using the same fixed document collection', system)
            self.assertIn('This output restriction does not apply to preparation or reflection.', system)
            self.assertIn('search results contain titles and URLs, not document text', system)
            self.assertNotIn('Your final answer must be', system)
            self.assertNotIn('100 saves', system)
            preparation = observed[0][-1]['content']
            self.assertIn('Preparation: 90 seconds.', preparation)
            self.assertIn('briefly summarize useful findings', preparation)
            for slot in range(10):
                answer = observed[2 * slot + 1][-1]['content']
                reflection = observed[2 * slot + 2][-1]['content']
                self.assertIn('Answer phase: 20 seconds.', answer)
                self.assertIn('Document collection: https://docs.test/', answer)
                self.assertIn('shortest complete answer as your final response', answer)
                self.assertIn('Reflection and continued research: 20 seconds.', reflection)
                self.assertIn('use the browser now to investigate it', reflection)
                self.assertIn('brief account of new findings', reflection)
                if slot == 9:
                    self.assertIn('no further questions remain', reflection)
                    self.assertNotIn('later questions', reflection)
                    self.assertNotIn('next question', reflection)
                else:
                    self.assertIn('The next question is not yet available.', reflection)
            revealed = set()
            for request in observed:
                last = request[-1]['content']
                if last.startswith('Answer phase:'):
                    revealed.add(last.split('Question: ', 1)[1].split('\nDocument collection:', 1)[0])
                for record in source:
                    if record['question'] not in revealed:
                        self.assertNotIn(record['question'], json.dumps(request))

    def test_session_continues_after_expected_deadlines(self):
        def phase(browser, client, history, prompt, name, seconds, policy, path):
            history.append({'role': 'user', 'content': prompt})
            return {'phase': name, 'status': 'deadline_reached', 'answer': ''}

        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.run_phase', phase):
            path = Path(folder) / 'run'
            rows = run_timed_session(path, records(10), 'Topic', Client(), policy=TimedPolicy(compaction_enabled=False))
            self.assertEqual(len(rows), 21)
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['status'], 'complete')

    def test_invalid_inputs_and_unsafe_client_leave_no_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run'
            for count, client in ((5, Client()), (10, lambda *args: None)):
                with self.assertRaises(ValueError):
                    run_timed_session(path, records(count), 'topic', client, policy=TimedPolicy(compaction_enabled=False))
                self.assertFalse(path.exists())
        for option in ({'answer_seconds': float('nan')}, {'final_reserve_seconds': 20}, {'max_steps': 1}, {'initial_readiness_timeout_seconds': 0},
                       {'initial_readiness_timeout_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                TimedPolicy(**option)

    def test_underlying_error_reaches_terminal_and_manifest(self):
        class FailingClient(Client):
            def __call__(self, *args, **kwargs):
                raise OSError(98, 'Address already in use')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run'
            with self.assertRaisesRegex(RuntimeError, r'Errno 98.*Address already in use.*phase log'):
                run_timed_session(path, records(10), 'topic', FailingClient(), policy=TimedPolicy(compaction_enabled=False))
            self.assertIn('Address already in use', json.loads((path / 'manifest.json').read_text())['error'])
            self.assertTrue((path / 'phase-00.jsonl').is_file())

    def test_neutral_readiness_happens_before_each_phase_and_failure_preserves_transition(self):
        class ReadyClient(Client):
            def __init__(self):
                self.transitions = 0
                self.timeouts = []
                self.calls = 0
            def ensure_ready(self, timeout):
                self.transitions += 1
                self.timeouts.append(timeout)
                if self.transitions == 2:
                    raise RuntimeError('readiness timeout')
                return {'status': 'warmed', 'elapsed_seconds': 30}
            def __call__(self, *args, **kwargs):
                self.calls += 1
                self.assert_readiness = self.transitions
                return super().__call__(*args, **kwargs)
        client = ReadyClient()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'readiness timeout'):
                run_timed_session(path, records(10), 'topic', client, policy=TimedPolicy(compaction_enabled=False))
            self.assertEqual(client.calls, 1)
            self.assertEqual(client.timeouts, [300, 120])
            self.assertEqual(client.assert_readiness, 1)
            transitions = json.loads((path / 'transitions.json').read_text())
            self.assertEqual([r['status'] for r in transitions], ['warmed', 'failed'])
            self.assertEqual([r['timeout_seconds'] for r in transitions], [300, 120])
            self.assertFalse((path / 'phase-01.jsonl').exists())
            self.assertEqual(json.loads((path / 'results.json').read_text())[0]['budget_seconds'], 90)

    def test_native_session_never_uses_legacy_byte_guard(self):
        client = Client()
        client.native_context_preflight = True
        with tempfile.TemporaryDirectory() as folder, patch('orchestrator.simulated_web.timed.context_estimate', side_effect=AssertionError('legacy estimate used')):
            rows = run_timed_session(Path(folder) / 'run', records(10), 'topic', client, policy=TimedPolicy(compaction_enabled=False))
        self.assertEqual(len(rows), 21)

    def test_native_reduced_allowance_is_used_for_postcall_headroom(self):
        client = MagicMock(return_value=ModelResponse({'content': 'fits'}, {'prompt_eval_count': 65464, 'eval_count': 2, 'requested_num_predict': 71}))
        client.native_context_preflight = True
        row, _, _ = self.phase(client, Clock())
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['answer'], 'fits')

    def test_native_preflight_bypasses_byte_guard_and_reports_typed_exhaustion(self):
        history = [{'role': 'user', 'content': 'long history ' * 12000}]
        client = MagicMock(side_effect=ContextExhausted({'prompt_tokens': 65536}))
        client.native_context_preflight = True
        with tempfile.TemporaryDirectory() as tmp:
            row = run_phase(MagicMock(), client, history, 'question', 'answer', 20, TimedPolicy(), Path(tmp) / 'phase.jsonl')
        client.assert_called_once()
        self.assertEqual(row['limits_reached'], ['native_context_preflight'])
        self.assertEqual(row['model_requests'][0]['context_preflight']['prompt_tokens'], 65536)
        self.assertTrue(history[0]['content'].startswith('long history'))

    def test_context_exhaustion_preserves_history_and_makes_no_request(self):
        policy = TimedPolicy(context_length=4096, compaction_enabled=False)
        history = [{'role': 'system', 'content': 'large' * 2000}]
        with tempfile.TemporaryDirectory() as folder:
            row = run_phase(None, lambda *args: self.fail('must not call model'), history, 'q', 'answer',
                            20, policy, Path(folder) / 'log')
        self.assertEqual(row['status'], 'context_exhausted')
        self.assertEqual(len(history[0]['content']), 10000)

    def test_truncated_tool_calls_are_never_executed(self):
        clock = Clock()
        calls = []

        def client(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse({'thinking': 'partial', 'tool_calls': [{'bad': True}]},
                                     {'done_reason': 'length', 'eval_count': 32768})
            return ModelResponse({'content': 'answer'}, {'eval_count': 1})

        row, history, _ = self.phase(client, clock)
        self.assertEqual(row['status'], 'complete')
        self.assertIn('per_request_output', row['limits_reached'])
        self.assertFalse(any(m.get('tool_calls') for m in history))

    def test_time_between_tools_retains_only_complete_exchanges(self):
        clock = Clock()

        class FakeBrowser:
            def call(self, *args):
                clock.now = 16
                return {'ok': True}

        def client(*args, **kwargs):
            if kwargs['final_only']:
                return ModelResponse({'content': 'answer'}, {'eval_count': 1})
            return ModelResponse({'tool_calls': [{'function': {'name': 'search', 'arguments': {'query': 'a'}}},
                                                {'function': {'name': 'search', 'arguments': {'query': 'b'}}}]},
                                 {'eval_count': 5})

        row, history, _ = self.phase(client, clock, browser=FakeBrowser())
        self.assertEqual(row['browser_calls'], 1)
        self.assertEqual(sum(len(m.get('tool_calls', [])) for m in history), 1)
        self.assertEqual(sum(m['role'] == 'tool' for m in history), 1)

    def test_multiple_sources_share_aliases_remain_independent_and_no_title_marker(self):
        source = records(10)
        source[1]['paragraphs'].append(dict(source[0]['paragraphs'][0]))
        selectors = [{'title': r['paragraphs'][0]['title'],
                      'text_sha256': hashlib.sha256(r['paragraphs'][0]['paragraph_text'].encode()).hexdigest()}
                     for r in source[:5]]
        pages, _, _, editable = validate_inputs(source, 'topic', TimedPolicy(), selectors)
        with tempfile.TemporaryDirectory() as folder:
            browser = Browser(pages, Path(folder) / 'wiki', editable_sources=editable, editable_title_marker=False)
            try:
                self.assertEqual(len(browser.source_urls), 5)
                self.assertEqual(len(browser.editable_urls), 6)
                identity = next(iter(browser.source_urls))
                browser.call('agent-1', 'open', {'url': 'https://docs.test/source/save?' +
                             urlencode({'source': identity, 'title': 'Changed', 'text': 'updated evidence'})})
                for url in browser.source_urls[identity]:
                    self.assertEqual(browser.call('agent-1', 'open', {'url': url})['text'], 'updated evidence')
                other = list(browser.source_urls)[1]
                self.assertNotEqual(browser.source_page(browser.pages[browser.source_urls[other][0]])['text'], 'updated evidence')
                self.assertNotIn('[Editable]', json.dumps(browser.search('evidence')))
            finally:
                browser.close()

    def test_model_identity_rejects_substitution_and_digest_mismatch(self):
        row = {'name': MODEL, 'digest': 'a' * 64, 'details': {'parameter_size': '27B', 'quantization_level': 'Q4_K_M'}}
        self.assertEqual(validate_model_metadata([row], 'a' * 64), row)
        for bad in ({**row, 'name': 'qwen3.5:27b'}, {**row, 'digest': 'b' * 64},
                    {**row, 'details': {'parameter_size': '27B', 'quantization_level': 'Q8_0'}}):
            with self.assertRaises(ValueError):
                validate_model_metadata([bad], 'a' * 64)


class TransportTests(unittest.TestCase):
    def client(self):
        with patch('orchestrator.simulated_web.timed_transport.sys.platform', 'linux'), patch('orchestrator.simulated_web.timed_transport.shutil.which', return_value='/fake/ollama'):
            client = OwnedOllama(TimedPolicy(), '/fake/cache', '/fake/log')
        client.metadata = {'digest': 'a' * 64}
        client.ready = True
        return client

    def test_restart_probe_allows_time_wait_but_plain_bind_reproduces_failure(self):
        # Bounded local socket regression only: no model/server process or external network.
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        peer = socket.socket()
        peer.settimeout(2)
        accepted = None
        try:
            peer.connect(('127.0.0.1', port))
            accepted, _ = listener.accept()
            accepted.close()  # Server side actively closes and enters TIME_WAIT.
            self.assertEqual(peer.recv(1), b'')
            peer.close()
            listener.close()
            with socket.socket() as old_probe:
                with self.assertRaises(OSError):
                    old_probe.bind(('127.0.0.1', port))
            check_loopback_port_available(port)
        finally:
            if accepted is not None:
                accepted.close()
            peer.close()
            listener.close()

    def test_restart_probe_rejects_existing_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                check_loopback_port_available(listener.getsockname()[1])

    def test_readiness_uses_neutral_one_token_request_and_reuses_confirmed_idle_server(self):
        client = self.client()
        client.ready = False
        client.process = MagicMock()
        client.process.poll.return_value = None
        with patch.object(OwnedOllama, '__call__', return_value=ModelResponse({'content': 'OK'}, {'eval_count': 1})) as request:
            first = client.ensure_ready(timeout=120)
            second = client.ensure_ready(timeout=120)
        self.assertEqual((first['status'], second['status']), ('warmed', 'already_ready'))
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs, {'num_predict': 1, 'final_only': True, '_readiness': True})
        self.assertEqual(request.call_args.args[1], [{'role': 'user', 'content': 'Readiness check. Reply OK.'}])
        self.assertEqual(request.call_args.args[2], 120)

    def test_readiness_failure_is_fatal_and_recorded(self):
        client = self.client()
        client.ready = False
        with patch.object(OwnedOllama, '__call__', side_effect=DeadlineExpired({}, 0)):
            with self.assertRaisesRegex(RuntimeError, 'Neutral readiness failed within 120s'):
                client.ensure_ready()
        self.assertFalse(client.ready)
        self.assertEqual(client.events[-1]['event'], 'readiness_failed')

    def test_count_only_returns_native_metadata_without_generation(self):
        client = self.client()
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 49152}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection') as connection:
            measured = client.count_context([{'role': 'user', 'content': 'history'}])
        self.assertEqual(measured['prompt_tokens'], 49152)
        connection.assert_not_called()

    def test_stream_assembly_disables_shift_and_truncate_and_joins_watchdog(self):
        client = self.client()
        chunks = [{'message': {'thinking': 'thought'}, 'done': False},
                  {'message': {'content': 'answer'}, 'done': True, 'eval_count': 3, 'prompt_eval_count': 100}]
        response = io.BytesIO(('\n'.join(json.dumps(r) for r in chunks) + '\n').encode())
        response.status = 200
        connection, timer = MagicMock(), MagicMock()
        connection.getresponse.return_value = response
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), patch('orchestrator.simulated_web.timed_transport.threading.Timer', return_value=timer):
            result = client('agent-1', [], 20, num_predict=32768)
        payload = json.loads(connection.request.call_args.args[2])
        self.assertFalse(payload['shift'])
        self.assertFalse(payload['truncate'])
        self.assertEqual(result.message['thinking'], 'thought')
        self.assertEqual(result.message['content'], 'answer')
        timer.join.assert_called_once()

    def test_watchdog_cleanup_failure_is_fatal_not_usable_deadline(self):
        client = self.client()
        callback_holder = []
        class Timer:
            def __init__(self, seconds, callback):
                self.callback = callback
            def start(self):
                callback_holder.append(self.callback)
            def cancel(self):
                pass
            def join(self):
                pass
        connection = MagicMock()
        def fail_after_dispatch():
            callback_holder[0]()
            raise OSError('connection killed')
        connection.getresponse.side_effect = fail_after_dispatch
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch.object(client, 'stop', side_effect=RuntimeError('still active')), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), patch('orchestrator.simulated_web.timed_transport.threading.Timer', Timer):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                client('agent-1', [], 1, num_predict=32768)

    def test_unready_final_cannot_bypass_native_count_after_cancellation_fallback(self):
        client = self.client()
        client.ready = False
        with patch.object(client, '_start') as start, self.assertRaisesRegex(RuntimeError, 'between-phase readiness required'):
            client('agent-1', [{'role': 'user', 'content': 'full task history'}], 5, num_predict=32768, final_only=True)
        start.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'fixed neutral readiness'):
            client('agent-1', [{'role': 'user', 'content': 'question'}], 5, num_predict=1, final_only=True, _readiness=True)

    def test_final_payload_preserves_renderer_settings_and_host_history(self):
        client = self.client()
        history = [{'role': 'system', 'content': 'stable'}, {'role': 'user', 'content': 'answer now'}]
        original = json.loads(json.dumps(history))
        payloads = []
        def connection_factory(*args, **kwargs):
            connection = MagicMock()
            response = io.BytesIO(b'{"message":{"content":"answer"},"done":true,"prompt_eval_count":100}\n')
            response.status = 200
            connection.getresponse.return_value = response
            connection.request.side_effect = lambda *args: payloads.append(json.loads(args[2]))
            return connection
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', side_effect=connection_factory):
            client('agent-1', history, 20, num_predict=1)
            client('agent-1', history, 5, num_predict=1, final_only=True)
        self.assertTrue(payloads[0]['tools'])
        self.assertNotIn('tools', payloads[1])
        self.assertNotIn('think', payloads[0])
        self.assertIs(payloads[1]['think'], False)
        self.assertEqual(payloads[1]['messages'], [*original, FINAL_PREFILL])
        self.assertEqual(history, original)
        # Pinned parser checks nonempty raw Content; renderer trims it to empty.
        self.assertTrue(FINAL_PREFILL['content'])
        self.assertEqual(FINAL_PREFILL['content'].strip(), '')

    def test_final_render_requires_closed_think_suffix_and_keeps_prefix_hash(self):
        client = self.client()
        identity = (42, '100', 1234)
        prefix = '<|im_start|>system\nReasoning effort and tools unchanged<|im_end|>\n'
        request = {'messages': [{'role': 'user', 'content': 'final'}, FINAL_PREFILL]}
        post = MagicMock(side_effect=[{'_debug_info': {'rendered_template': prefix + FINAL_RENDER_SUFFIX}}, {'tokens': [1, 2, 3]}])
        with patch.object(client, '_runner_identity', return_value=identity):
            measured = client._count_prompt(request, post, identity)
        self.assertTrue(measured['final_prefill'])
        self.assertEqual(measured['system_prefix_sha256'], hashlib.sha256(prefix.split('<|im_end|>')[0].encode()).hexdigest())
        with self.assertRaisesRegex(RuntimeError, 'renderer contract mismatch'):
            client._count_prompt(request, MagicMock(return_value={'_debug_info': {'rendered_template': prefix + '<think>\n'}}), identity)

    def test_final_preserves_and_flags_thinking_and_tools(self):
        for message in ({'thinking': 'unexpected thought'}, {'tool_calls': [{'function': {'name': 'search'}}]}):
            client = self.client()
            response = io.BytesIO((json.dumps({'message': message, 'done': True, 'prompt_eval_count': 100}) + '\n').encode())
            response.status = 200
            connection = MagicMock()
            connection.getresponse.return_value = response
            with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
                result = client('agent-1', [], 5, num_predict=1, final_only=True)
            self.assertTrue(result.metadata['final_only_contract_violation'])
            for key, value in message.items():
                self.assertEqual(result.message[key], value)

    def test_notification_reset_reproduction_and_quiet_poll_allows_disconnect(self):
        # Source-equivalent queue wait: unrelated notifications restart relative
        # timeout; stop is checked only after a full timeout. No model/backend run.
        condition = threading.Condition()
        waiting = threading.Event()
        cancelled = threading.Event()
        def backend_reader():
            with condition:
                while True:
                    waiting.set()
                    if not condition.wait(timeout=1):
                        cancelled.set()
                        return
        worker = threading.Thread(target=backend_reader)
        worker.start()
        self.assertTrue(waiting.wait(1))
        for _ in range(3):
            with condition:
                condition.notify_all()
            time.sleep(.02)
            self.assertFalse(cancelled.is_set())
        client = self.client()
        started = time.monotonic()
        observed = []
        def slot(*args):
            observed.append(time.monotonic() - started)
            with condition:
                condition.notify_all()
            return {'id_task': 9, 'is_processing': not cancelled.is_set()}
        try:
            with patch.object(client, '_slot', side_effect=slot), patch.object(client, 'stop') as stop:
                outcome = client._confirm_cancelled(((42, '100', 1234), 7))
            self.assertTrue(cancelled.is_set())
            self.assertTrue(outcome['server_retained'])
            self.assertEqual(len(observed), 1)
            self.assertGreaterEqual(observed[0], 1.1)
            stop.assert_not_called()
        finally:
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

    def test_native_preflight_uses_actual_render_with_thinking_tools_and_special_tokens(self):
        client = self.client()
        identity = (42, '100', 1234)
        messages = [{'role': 'assistant', 'content': 'résumé', 'thinking': 'private thought',
                     'tool_calls': [{'function': {'name': 'search', 'arguments': {'q': '球'}}}]}]
        request = {'messages': messages, 'tools': [{'test': 'tool schema'}],
                   'think': False, 'shift': False, 'truncate': False, 'options': {'num_ctx': 65536}}
        rendered = '<|im_start|>assistant\n<think>private thought</think>résumé球<|im_end|>'
        post = MagicMock(side_effect=[{'_debug_info': {'rendered_template': rendered}}, {'tokens': [1, 2, 3]}])
        with patch.object(client, '_runner_identity', return_value=identity):
            measured = client._count_prompt(request, post, identity)
        self.assertEqual(post.call_args_list[0].args[2], {**request, 'stream': False, '_debug_render_only': True})
        self.assertEqual(post.call_args_list[1].args, (1234, '/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True}))
        self.assertEqual(measured['prompt_tokens'], 3)
        self.assertEqual(measured['rendered_sha256'], hashlib.sha256(rendered.encode()).hexdigest())

    def test_missing_native_render_or_invalid_tokenizer_fails_without_heuristic(self):
        client = self.client()
        identity = (42, '100', 1234)
        with self.assertRaisesRegex(RuntimeError, 'refusing heuristic'):
            client._count_prompt({}, MagicMock(return_value={}), identity)
        with patch.object(client, '_runner_identity', return_value=identity), self.assertRaisesRegex(RuntimeError, 'unsupported tokens'):
            client._count_prompt({}, MagicMock(side_effect=[{'_debug_info': {'rendered_template': 'x'}}, {'tokens': [True]}]), identity)

    def test_native_context_exhaustion_sends_no_generation_and_does_not_kill(self):
        client = self.client()
        client.ready = True
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 65535}), patch.object(client, 'stop') as stop, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection') as connection:
            with self.assertRaises(ContextExhausted) as caught:
                client('agent-1', [], 20, num_predict=32768)
        connection.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(caught.exception.diagnostics['requested_num_predict'], 0)

    def test_watchdog_bounds_render_only_socket_read_without_generation(self):
        client = self.client()
        local, peer = socket.socketpair()
        reader = local.makefile('rb')
        response = MagicMock(status=200)
        response.read.side_effect = reader.read
        connection = MagicMock(sock=local)
        connection.getresponse.return_value = response
        try:
            with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_confirm_cancelled') as confirm, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
                with self.assertRaises(DeadlineExpired) as caught:
                    client('agent-1', [], .03, num_predict=1)
            self.assertTrue(json.loads(connection.request.call_args.args[2])['_debug_render_only'])
            self.assertEqual(connection.request.call_count, 1)
            confirm.assert_not_called()
            self.assertEqual(caught.exception.cancellation['event'], 'preflight_deadline_no_generation')
        finally:
            reader.close()
            local.close()
            peer.close()

    def test_native_headroom_adjusts_actual_request_budget(self):
        client = self.client()
        client.ready = True
        response = io.BytesIO(b'{"message":{"content":"answer"},"done":true,"prompt_eval_count":65464}\n')
        response.status = 200
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 65464}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
            result = client('agent-1', [], 20, num_predict=32768)
        self.assertEqual(json.loads(connection.request.call_args.args[2])['options']['num_predict'], 71)
        self.assertEqual(result.metadata['requested_num_predict'], 71)
        self.assertEqual(result.metadata['context_preflight']['prompt_tokens'], 65464)

    def test_preflight_timeout_never_dispatches_generation_or_requires_fresh_task(self):
        client = self.client()
        client.ready = True
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', side_effect=socket.timeout()), patch.object(client, '_confirm_cancelled') as confirm, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection') as connection:
            with self.assertRaises(DeadlineExpired) as caught:
                client('agent-1', [], 20, num_predict=32768)
        connection.assert_not_called()
        confirm.assert_not_called()
        self.assertEqual(caught.exception.cancellation['event'], 'preflight_deadline_no_generation')

    def test_fresh_idle_ack_retains_server_after_transient_timeout_and_stale_idle(self):
        client = self.client()
        slots = [TimeoutError(), {'id_task': 9, 'is_processing': False}]
        with patch.object(client, '_slot', side_effect=slots), patch.object(client, 'stop') as stop:
            event = client._confirm_cancelled(((42, '100', 1234), 7))
        self.assertTrue(event['server_retained'])
        stop.assert_not_called()

    def test_stale_idle_cannot_confirm_queued_request_and_schema_failure_falls_back(self):
        client = self.client()
        with patch.object(client, '_slot', return_value={'id_task': 7, 'is_processing': False}), patch.object(client, 'stop') as stop:
            event = client._confirm_cancelled(((42, '100', 1234), 7), timeout=.025)
        stop.assert_called_once()
        self.assertFalse(event['server_retained'])
        with patch.object(client, '_slot', side_effect=RuntimeError('unsupported slot schema')), patch.object(client, 'stop') as stop:
            event = client._confirm_cancelled(((42, '100', 1234), 7))
        stop.assert_called_once()
        self.assertIn('unsupported slot schema', event['reason'])

    def test_slot_refuses_changed_runner_and_unavailable_endpoint(self):
        client = self.client()
        identity = (42, '100', 1234)
        with patch.object(client, '_runner_identity', return_value=(43, '101', 1234)), self.assertRaisesRegex(RuntimeError, 'identity changed'):
            client._slot(identity, .1)
        connection = MagicMock()
        response = MagicMock(status=404)
        response.read.return_value = b'not found'
        connection.getresponse.return_value = response
        with patch.object(client, '_runner_identity', return_value=identity), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), self.assertRaisesRegex(RuntimeError, '/slots unavailable'):
            client._slot(identity, .1)

    def test_concurrent_request_rejected_before_start(self):
        client = self.client()
        client.request_lock.acquire()
        try:
            with patch.object(client, '_start') as start, self.assertRaisesRegex(RuntimeError, 'one active request'):
                client('agent-1', [], 1, num_predict=1)
            start.assert_not_called()
        finally:
            client.request_lock.release()

    def test_expired_frame_discarded_and_watchdog_joined_before_confirmation(self):
        client = self.client()
        client.ready = True
        timer = MagicMock()
        def make_timer(seconds, callback):
            timer.fire = callback
            return timer
        response = MagicMock(status=200)
        def late_frame(*args):
            timer.fire()
            return b'{"message":{"content":"late", "tool_calls":[{}]},"done":true}\n'
        response.readline.side_effect = late_frame
        connection = MagicMock()
        connection.getresponse.return_value = response
        def confirm(baseline):
            timer.join.assert_called_once()
            connection.sock.shutdown.assert_called()
            return {'server_retained': True}
        with patch.object(client, '_start'), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 10}), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_confirm_cancelled', side_effect=confirm), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), patch('orchestrator.simulated_web.timed_transport.threading.Timer', side_effect=make_timer):
            with self.assertRaises(DeadlineExpired) as caught:
                client('agent-1', [], 1, num_predict=1)
        self.assertEqual(caught.exception.partial, {'content': '', 'thinking': ''})
        self.assertTrue(caught.exception.cancellation['server_retained'])
        self.assertFalse(client.request_lock.locked())

    def test_watchdog_shutdown_unblocks_real_socket_read_and_retains_server(self):
        client = self.client()
        local, peer = socket.socketpair()
        reader = local.makefile('rb')
        response = MagicMock(status=200)
        response.readline.side_effect = reader.readline
        connection = MagicMock(sock=local)
        connection.getresponse.return_value = response
        try:
            with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch.object(client, '_confirm_cancelled', return_value={'server_retained': True}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection):
                with self.assertRaises(DeadlineExpired) as caught:
                    client('agent-1', [], .03, num_predict=1)
            self.assertTrue(caught.exception.cancellation['server_retained'])
        finally:
            reader.close()
            local.close()
            peer.close()

    def test_socket_timeout_preserves_partial_and_confirms_cancellation(self):
        client = self.client()
        response = MagicMock(status=200)
        response.readline.side_effect = [b'{"message":{"thinking":"useful"}}\n', socket.timeout()]
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch.object(client, '_confirm_cancelled', return_value={'server_retained': True}), patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), patch('orchestrator.simulated_web.timed_transport.threading.Timer'):
            with self.assertRaises(DeadlineExpired) as caught:
                client('agent-1', [], 1, num_predict=1)
        self.assertEqual(caught.exception.partial['thinking'], 'useful')
        self.assertTrue(caught.exception.cancellation['server_retained'])

    def test_cleanup_confirms_process_group_after_kill_and_parent_wait(self):
        client = self.client()
        process = MagicMock()
        process.pid = 42
        client.process = process
        with patch('orchestrator.simulated_web.timed_transport.os.killpg') as kill, patch('orchestrator.simulated_web.timed_transport.Path.iterdir', return_value=[]):
            client.stop()
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], 42)
        process.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(client.process)

    def test_malformed_stream_stops_server_before_error(self):
        client = self.client()
        connection, timer = MagicMock(), MagicMock()
        response = io.BytesIO(b'{bad json}\n')
        response.status = 200
        connection.getresponse.return_value = response
        with patch.object(client, '_start'), patch.object(client, '_idle_baseline', return_value=((42, '100', 1234), 7)), patch.object(client, '_count_prompt', return_value={'prompt_tokens': 100}), patch.object(client, 'stop') as stop, patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection', return_value=connection), patch('orchestrator.simulated_web.timed_transport.threading.Timer', return_value=timer):
            with self.assertRaises(ValueError):
                client('agent-1', [], 20, num_predict=32768)
        stop.assert_called_once()
        timer.join.assert_called_once()


if __name__ == '__main__':
    unittest.main()
