"""Prepared host-seed plumbing tests; use mocked clients only."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web.session import HOST_SEED_LABEL, load_wiki_seed, main, prepare, run_session
from orchestrator.simulated_web.test_session import records


class WikiSeedTests(unittest.TestCase):
    def note(self):
        pages, _, _, _ = prepare(records(), 2, 0, 'maximal')
        url = pages[0]['url']
        return {'slug': 'starter', 'title': 'Document 0 evidence',
                'text': 'Evidence 0. Source: ' + url,
                'provenance': {'source_urls': [url], 'construction': 'Fixture public document only.'}}

    def test_invalid_seed_leaves_no_run_artifacts(self):
        valid = self.note()
        malformed = [[], {}, dict(valid, slug='../escape'), dict(valid, text=''),
                     dict(valid, title='x' * 201), dict(valid, text='x' * 8000),
                     dict(valid, provenance={}), dict(valid, text='Missing source URL')]
        for source_urls in ([], ['https://docs.test/missing'], [17],
                            valid['provenance']['source_urls'] * 2):
            note = copy.deepcopy(valid)
            note['provenance']['source_urls'] = source_urls
            malformed.append(note)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            seed = Path(temporary) / 'seed.json'
            for note in malformed:
                seed.write_text(json.dumps(note))
                with self.subTest(note=note), self.assertRaises(ValueError):
                    run_session(root, records(), lambda *args: {'content': 'answer'}, wiki_seed_file=seed)
                self.assertFalse(root.exists())
            seed.write_text('{')
            with self.assertRaises(ValueError):
                run_session(root, records(), lambda *args: None, wiki_seed_file=seed)
            self.assertFalse(root.exists())
            with self.assertRaises(FileNotFoundError):
                run_session(root, records(), lambda *args: None, wiki_seed_file=seed.with_name('missing'))
            self.assertFalse(root.exists())

    def test_seed_is_visible_before_assignments_and_attributed_to_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            seed = Path(temporary) / 'seed.json'
            seed.write_text(json.dumps(self.note()))

            def client(*args):
                with sqlite3.connect(root / 'web/wiki.sqlite3') as db:
                    self.assertEqual(db.execute('SELECT agent FROM revisions').fetchall(), [('host-seeded',)])
                    body = db.execute('SELECT body FROM pages').fetchone()[0]
                    self.assertTrue(body.startswith(HOST_SEED_LABEL))
                    self.assertNotIn('PRIVATE_GOLD', body)
                return {'content': 'answer'}

            results = run_session(root, records(), client, wiki_seed_file=seed)
            self.assertTrue(all(row['status'] == 'complete' for row in results))
            settings = json.loads((root / 'settings.json').read_text())
            self.assertEqual((root / 'wiki-seed.json').read_bytes(), seed.read_bytes())
            self.assertEqual(settings['wiki_seed']['input_sha256'], hashlib.sha256(seed.read_bytes()).hexdigest())
            self.assertEqual(settings['wiki_seed']['author'], 'host-seeded')
            self.assertTrue(all(row['wiki_boundary']['before']['revisions'] == 1 for row in results))
            with sqlite3.connect(root / 'web/wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT agent,operation FROM audit').fetchall(), [('host-seeded', 'open')])

    def test_default_remains_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            run_session(root, records(), lambda *args: {'content': 'answer'})
            self.assertIsNone(json.loads((root / 'settings.json').read_text())['wiki_seed'])
            self.assertFalse((root / 'wiki-seed.json').exists())
            with sqlite3.connect(root / 'web/wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0], 0)

    def test_cli_rejects_seed_before_contacting_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / 'dataset.jsonl'
            dataset.write_text(''.join(json.dumps(record) + '\n' for record in records()))
            root = Path(temporary) / 'run'
            with patch('sys.argv', ['session', '--dataset', str(dataset), '--run-dir', str(root),
                                    '--wiki-seed-file', str(Path(temporary) / 'missing')]), \
                    patch('orchestrator.simulated_web.session.http.client.HTTPConnection') as connection:
                with self.assertRaises(FileNotFoundError):
                    main()
                connection.assert_not_called()
                self.assertFalse(root.exists())

    def test_encoded_url_limit_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary) / 'seed.json'
            note = self.note()
            note['text'] = '\U0001f600' * 3000 + note['text']
            seed.write_text(json.dumps(note))
            pages, _, _, _ = prepare(records(), 2, 0, 'maximal')
            with self.assertRaisesRegex(ValueError, 'Encoded'):
                load_wiki_seed(seed, pages)


if __name__ == '__main__':
    unittest.main()
