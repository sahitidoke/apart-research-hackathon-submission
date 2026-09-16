"""Mock-only public request-history and durable snapshot contracts."""
from contextlib import redirect_stdout
from html import unescape
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web.browser import Browser, MAX_TEXT
from orchestrator.simulated_web import test_modal_timed
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.test_timed_resume import ResumeClient, MODEL
from orchestrator.simulated_web.timed import TimedPolicy, run_timed_session, resume_timed_session
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint


ROOT = 'https://docs.test/'
HISTORY = ROOT + 'request-history'
PAGES = [{'url': ROOT, 'title': 'Collection', 'text': 'SECRET DOCUMENT BODY',
          'links': [{'label': 'Missing', 'url': ROOT + 'missing'}]}]


def browse(browser, agent='alpha', url=HISTORY, interleave=False):
    page = browser.call(agent, 'open', {'url': url})
    chunks = []
    while True:
        assert 'error' not in page, page
        assert len(page['text']) <= MAX_TEXT
        chunks.append(unescape(page['text']))
        if not page['links']:
            break
        if interleave:
            browser.call('beta', 'search', {'query': 'INTERLEAVED'})
        page = browser.call(agent, 'click', {'page_id': page['page_id'], 'link_id': 1})
    text = ''.join(chunks)
    return [] if text == 'No requests.' else [json.loads(row) for row in text.splitlines()]


class RequestHistoryTests(unittest.TestCase):
    def browser(self, mode='shared'):
        browser = Browser(PAGES, ':memory:', request_history_mode=mode)
        self.addCleanup(browser.close)
        return browser

    def test_empty_default_and_validation(self):
        b = self.browser()
        self.assertEqual(browse(b), [])
        self.assertEqual(browse(b)[0]['requested'], HISTORY)
        disabled = self.browser('disabled')
        self.assertEqual(len(disabled.call('alpha', 'open', {'url': ROOT})['links']), 1)
        self.assertIn('error', disabled.call('alpha', 'open', {'url': HISTORY}))
        self.assertEqual(disabled.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.sqlite3'
            with self.assertRaises(ValueError):
                Browser(PAGES, path, request_history_mode='bad')
            self.assertFalse(path.exists())
        with self.assertRaises(ValueError):
            TimedPolicy(request_history_mode='bad')

    def test_shared_isolated_and_no_private_fields(self):
        for mode in ('shared', 'isolated'):
            b = self.browser(mode)
            b.call('alpha', 'search', {'query': 'alpha request'})
            b.call('beta', 'search', {'query': 'beta request'})
            b.call('alpha', 'private_scratchpad_update', {'text': 'SECRET NOTES'})
            b.call('alpha', 'search', {'query': 'SECRET OVERSIZED' * 100})
            b.call('alpha', 'open', {'url': ROOT, 'extra': 'SECRET INVALID'})
            b.call('alpha', 'open', {'url': 'SECRET OVERSIZED' * 3000})
            page = b.call('beta', 'open', {'url': ROOT})
            b.call('alpha', 'click', {'page_id': page['page_id'], 'link_id': 2})
            rows = browse(b)
            values = [r['requested'] for r in rows]
            self.assertIn('alpha request', values)
            self.assertEqual('beta request' in values, mode == 'shared')
            for row in rows:
                self.assertEqual(set(row), {'timestamp', 'operation', 'requested', 'status'})
            payload = json.dumps(rows)
            self.assertNotIn('SECRET', payload)
            self.assertNotIn('owner', payload)
            self.assertNotIn('page_id', payload)

    def test_success_failure_click_and_inert_values(self):
        b = self.browser()
        page = b.call('alpha', 'open', {'url': ROOT})
        b.call('alpha', 'click', {'page_id': page['page_id'], 'link_id': 2})
        bad = 'https://outside.test/<script>?x=1&y=2'
        b.call('alpha', 'open', {'url': bad})
        query = '<img src=x onerror=alert(1)>\n[go](javascript:alert(1))'
        b.call('alpha', 'search', {'query': query})
        raw = b.call('alpha', 'open', {'url': HISTORY})
        self.assertNotIn('<img', raw['text'])
        self.assertEqual(raw['links'], [])
        rows = [json.loads(row) for row in unescape(raw['text']).splitlines()]
        self.assertEqual([(r['operation'], r['requested'], r['status']) for r in rows],
                         [('search', query, 'success'), ('open', bad, 'error'),
                          ('click', ROOT + 'missing', 'error'), ('open', ROOT, 'success')])
        before = b.db.execute('SELECT * FROM request_events').fetchall()
        for url in (HISTORY + '/save?text=replace', HISTORY + '/edit', HISTORY + '/delete', HISTORY + '?text=replace'):
            self.assertIn('error', b.call('alpha', 'open', {'url': url}))
        self.assertEqual(b.db.execute('SELECT * FROM request_events WHERE id<=?', (len(before),)).fetchall(), before)

    def test_stable_pagination_beyond_100_and_long_requests(self):
        b = self.browser()
        values = [f'query-{i}' for i in range(125)]
        for value in values:
            b.call('alpha', 'search', {'query': value})
        long_url = ROOT + 'missing?' + ('<&x' * 9000)
        b.call('alpha', 'open', {'url': long_url})
        rows = browse(b, interleave=True)
        self.assertEqual([r['requested'] for r in rows], [long_url] + values[::-1])
        self.assertTrue(any(r['requested'] == 'INTERLEAVED' for r in browse(b)))

    def test_landing_preserves_twenty_collection_links(self):
        links = [{'label': str(i), 'url': ROOT + str(i)} for i in range(20)]
        b = Browser([{**PAGES[0], 'links': links}], ':memory:', request_history_mode='shared')
        self.addCleanup(b.close)
        page = b.call('alpha', 'open', {'url': ROOT})
        self.assertEqual(page['links'][1:], [{'id': i + 2, **link} for i, link in enumerate(links)])
        self.assertEqual(page['links'][0]['label'], 'Request history')

    def test_owner_bound_pagination(self):
        for mode in ('shared', 'isolated'):
            b = self.browser(mode)
            for i in range(100):
                b.call('alpha', 'search', {'query': str(i)})
            page = b.call('alpha', 'open', {'url': HISTORY})
            cursor = page['links'][0]['url']
            self.assertIn('error', b.call('beta', 'open', {'url': cursor}))
            self.assertNotIn('error', b.call('alpha', 'open', {'url': cursor}))

    def make_checkpoint(self, root, mode='shared'):
        def factory(*args, **kwargs):
            b = Browser(*args, **kwargs)
            for i in range(100):
                b.call('agent-1', 'search', {'query': f'checkpoint-{i}'})
            b.call('agent-1', 'open', {'url': HISTORY})
            return b
        with patch('orchestrator.simulated_web.timed.Browser', side_effect=factory), redirect_stdout(io.StringIO()):
            run_timed_session(root / 'parent', records(10), 'topic', ResumeClient(),
                              TimedPolicy(question_count=10, compaction_enabled=False, request_history_mode=mode),
                              provenance={'model': MODEL})
        return root / 'parent/checkpoints/questions-005'

    def test_checkpoint_and_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.make_checkpoint(root)
            with_load = load_checkpoint(checkpoint)
            try:
                b = with_load.browser
                self.assertEqual(b.request_history_mode, 'shared')
                cursor = b.views['agent-1']['p1'][0]['url']
                rows = browse(b, 'agent-1', cursor)
                self.assertTrue(rows[-1]['requested'] == 'checkpoint-0')
            finally:
                with_load.close()
            with redirect_stdout(io.StringIO()):
                resume_timed_session(root / 'child', checkpoint, ResumeClient())
            child = load_checkpoint(root / 'child/checkpoints/questions-010')
            try:
                self.assertEqual(child.browser.request_history_mode, 'shared')
                self.assertEqual(len(child.browser.history_windows), 1)
                self.assertNotIn('error', child.browser.call('agent-1', 'click', {'page_id': 'p1', 'link_id': 1}))
            finally:
                child.close()

    def test_legacy_checkpoint_without_history_table_or_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = self.make_checkpoint(Path(folder), 'disabled')
            for name in ('settings.json', 'browser.json'):
                path = checkpoint / name
                data = json.loads(path.read_text())
                if name == 'settings.json':
                    data['policy'].pop('request_history_mode')
                else:
                    data.pop('request_history_mode')
                    data.pop('history_windows')
                path.write_text(json.dumps(data))
            db = sqlite3.connect(checkpoint / 'wiki.sqlite3')
            db.execute('DROP TABLE request_events')
            db.commit()
            db.close()
            manifest_path = checkpoint / 'checkpoint.json'
            manifest = json.loads(manifest_path.read_text())
            for name in manifest['files_sha256']:
                manifest['files_sha256'][name] = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            loaded = load_checkpoint(checkpoint)
            try:
                self.assertEqual(loaded.browser.request_history_mode, 'disabled')
                self.assertEqual(loaded.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 0)
            finally:
                loaded.close()


@unittest.skipIf(test_modal_timed.modal_timed is None, 'Optional Modal SDK not installed')
class RequestHistoryCliTests(unittest.TestCase):
    def test_validate_only_modes_and_resume_override(self):
        launcher = test_modal_timed.modal_timed
        with tempfile.TemporaryDirectory() as folder:
            args = test_modal_timed.ModalValidationTests().inputs(folder)
            config = Path(folder) / 'modal.toml'
            config.write_text('[research-profile]\n')
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(config)}), patch.object(launcher.app, 'run') as cloud:
                for mode in ('disabled', 'shared', 'isolated'):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        launcher.main(args + ['--request-history-mode', mode])
                    self.assertEqual(json.loads(output.getvalue())['policy']['request_history_mode'], mode)
                cloud.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'does not accept'):
                launcher.main(['--resume-from', 'parent/checkpoints/questions-005', '--run-id', 'child',
                               '--validate-only', '--request-history-mode', 'shared'])
