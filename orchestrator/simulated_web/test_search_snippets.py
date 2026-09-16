"""Titles/URLs-only search contracts, using local fixtures and mocked models."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.session import main, run_session
from orchestrator.simulated_web.test_session import records


class SearchSnippetsTests(unittest.TestCase):
    def test_uniform_omission_preserves_ranking_labels_and_live_content(self):
        corpus = [{'url': f'https://docs.test/{i}', 'title': 'Research source',
                   'text': f'Research evidence {i}'} for i in range(3)]
        selector = ('Research source', hashlib.sha256(b'Research evidence 0').hexdigest())
        for editable in (None, selector):
            with self.subTest(editable=editable):
                on = Browser(corpus, ':memory:', editable_source=editable)
                off = Browser(corpus, ':memory:', editable_source=editable, search_snippets=False)
                try:
                    for browser in (on, off):
                        browser.call('a', 'open', {'url': 'https://wiki.test/save?' + urlencode(
                            {'slug': 'research', 'title': 'Research notes', 'text': 'Research evidence'})})
                    for query in ('research', 'evidence', 'absent'):
                        expected = on.call('a', 'search', {'query': query})
                        actual = off.call('a', 'search', {'query': query})
                        self.assertEqual(actual, {'results': [
                            {key: value for key, value in hit.items() if key != 'snippet'}
                            for hit in expected['results']]})
                        self.assertTrue(all('snippet' in hit for hit in expected['results']))
                    hits = off.search('research')['results']
                    self.assertTrue(any(hit['url'].startswith('https://wiki.test/page/') for hit in hits))
                    if editable:
                        self.assertTrue(any(hit['title'].endswith(' [Editable]') for hit in hits))
                    else:
                        self.assertTrue(any(hit['url'] == 'https://wiki.test/' for hit in hits))
                    self.assertEqual(on.open('a', corpus[0]['url']), off.open('a', corpus[0]['url']))
                    audit = off.db.execute("SELECT result FROM audit WHERE operation='search'").fetchall()
                    self.assertTrue(all('snippet' not in row[0] for row in audit))
                finally:
                    on.close()
                    off.close()

    def test_session_records_and_delivers_selected_mode(self):
        for enabled in (True, False):
            seen = []
            def client(agent, messages, timeout, **options):
                if messages[-1]['role'] != 'tool':
                    return ModelResponse({'content': '', 'tool_calls': [{'function': {
                        'name': 'search', 'arguments': {'query': 'Evidence'}}}]},
                        {'eval_count': 1, 'prompt_eval_count': 10})
                seen.append(json.loads(messages[-1]['content']))
                return ModelResponse({'content': 'Answer'}, {'eval_count': 1, 'prompt_eval_count': 10})
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'run'
                kwargs = {} if enabled else {'search_snippets': False}
                results = run_session(root, records(1), client, agents=1,
                                      prompt_condition='neutral', **kwargs)
                self.assertEqual(results[0]['status'], 'complete')
                self.assertIs(json.loads((root / 'settings.json').read_text())['search_snippets'], enabled)
                self.assertTrue(seen[0]['results'])
                self.assertTrue(all(('snippet' in hit) == enabled for hit in seen[0]['results']))

    def test_invalid_values_create_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            database = Path(temporary) / 'wiki.sqlite3'
            for invalid in (None, 0, 1, 'off'):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, 'search_snippets'):
                        run_session(root, records(1), None, search_snippets=invalid)
                    with self.assertRaisesRegex(ValueError, 'search_snippets'):
                        Browser([], database, search_snippets=invalid)
                    self.assertFalse(root.exists())
                    self.assertFalse(database.exists())
            with patch('sys.argv', ['session', '--dataset', str(Path(temporary) / 'absent'),
                                   '--run-dir', str(root), '--model', 'mock', '--search-snippets', 'bad']), \
                    patch('orchestrator.simulated_web.session.model_metadata') as metadata:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                metadata.assert_not_called()
                self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
