"""Small offline histories; no model or service calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.search_replay import identity, main, replay, support_index
from orchestrator.simulated_web.session import prepare


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'session'
        (self.root / 'web').mkdir(parents=True)
        p = {'idx': 91, 'title': 'Shared source', 'paragraph_text': 'alpha evidence', 'is_supporting': True}
        self.records = [
            {'id': 'q1', 'question': 'First?', 'paragraphs': [p,
             {'idx': 7, 'title': 'Second source', 'paragraph_text': 'beta evidence', 'is_supporting': True}]},
            {'id': 'q2', 'question': 'Second?', 'paragraphs': [dict(p, is_supporting=False),
             {'idx': 4, 'title': 'Other source', 'paragraph_text': 'gamma evidence', 'is_supporting': True}]}]
        self.pages, _, _, _ = prepare(self.records, 2, 0, 'neutral')
        (self.root / 'dataset.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.records))
        (self.root / 'pages.json').write_text(json.dumps(self.pages))
        b = Browser(self.pages, self.root / 'web' / 'wiki.sqlite3')
        before = b.checkpoint()
        b.call('agent-1', 'search', {'query': 'futureunique'})
        save = lambda text: 'https://wiki.test/save?' + urlencode({'slug': 'note', 'title': 'Note', 'text': text})
        b.call('agent-2', 'open', {'url': save('futureunique')})
        b.call('agent-1', 'search', {'query': 'futureunique'})
        b.call('agent-1', 'open', {'url': save('replacementunique')})
        b.call('agent-2', 'search', {'query': 'futureunique'})
        b.call('agent-2', 'search', {'query': 'replacementunique'})
        b.call('agent-1', 'search', {'query': 'alpha'})
        after = b.checkpoint()
        b.close()
        self.results = [dict(agent=agent, id=qid, slot=slot, wiki_boundary={'before': lo, 'after': hi})
                        for agent, qid, slot, lo, hi in [
                            ('agent-1', 'q1', 1, before, after), ('agent-2', 'q2', 1, before, after),
                            ('agent-1', 'q2', 2, after, after), ('agent-2', 'q1', 2, after, after)]]
        self.write_results()

    def write_results(self):
        (self.root / 'results.json').write_text(json.dumps(self.results))

    def mutate(self, sql, parameters=()):
        with sqlite3.connect(self.root / 'web' / 'wiki.sqlite3') as c:
            c.execute(sql, parameters)

    def test_exact_history_replacement_and_no_final_page_dependency(self):
        self.mutate("UPDATE pages SET body='poisonfuture',title='Poison'")
        report = replay(self.root)
        queries = {q['audit_id']: q for q in report['queries']}
        note = 'https://wiki.test/page/note'
        self.assertEqual(queries[1]['bm25_urls'], [])
        self.assertEqual(queries[3]['bm25_urls'], [note])
        self.assertEqual(queries[5]['bm25_urls'], [])
        self.assertEqual(queries[6]['bm25_urls'], [note])
        self.assertEqual([queries[i]['wiki_revisions_visible'] for i in (1, 3, 5, 6)], [0, 1, 2, 2])
        self.assertEqual(report['summary']['revisions_replayed'], 2)

    def test_copy_identity_and_paragraph_position_not_idx(self):
        mapping, supports, _ = support_index(self.records, self.pages)
        prefix = 'https://docs.test/q/' + hashlib.sha256(b'q2').hexdigest()
        copy = prefix + '/p/0/0'
        self.assertIn(mapping[copy], supports['q1'])
        self.assertNotIn(mapping[copy], supports['q2'])
        self.mutate('UPDATE audit SET result=? WHERE id=7',
                    (json.dumps({'results': [{'url': copy}]}),))
        report = replay(self.root)
        row = report['queries'][-1]['top_k']['3']
        self.assertEqual(row['original_supports'], [identity(self.records[0]['paragraphs'][0])])
        self.assertEqual(len(row['bm25_supports']), 1)  # Two URLs, one supporting paragraph.
        self.assertEqual(row['support_denominator'], 2)

    def test_denominators_include_zero_search_assignments(self):
        report = replay(self.root)
        s = report['summary']
        self.assertEqual(s['queries']['denominator'], 5)
        self.assertEqual(s['assignments']['denominator'], 4)
        self.assertEqual(s['zero_search_assignments'], 2)
        self.assertEqual(s['assignments']['3']['support_denominator'], 6)
        self.assertEqual(s['assignments']['3']['original']['all_support']['count'], 0)
        self.assertEqual(s['assignments']['3']['original']['any_support']['rate'], 0.25)

    def test_errors_retained_and_excluded_from_valid_query_denominator(self):
        self.mutate('UPDATE audit SET args=?,result=? WHERE id=1',
                    (json.dumps({'query': 9}), json.dumps({'error': 'Invalid query'})))
        report = replay(self.root)
        self.assertEqual(report['summary']['search_count'], 5)
        self.assertEqual(report['summary']['failed_search_count'], 1)
        self.assertEqual(report['summary']['queries']['denominator'], 4)
        self.assertEqual(report['assignments'][0]['search_count'], 3)

    def test_cli_immutable_deterministic_and_fresh_output(self):
        original = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--run-root', str(self.root)]), 0)
        for p, content in original.items():
            self.assertEqual(p.read_bytes(), content)
        output = self.root / 'search-replay'
        first = (output / 'report.json').read_bytes()
        other = Path(self.temp.name) / 'other'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--run-root', str(self.root), '--output-dir', str(other)])
        self.assertEqual(first, (other / 'report.json').read_bytes())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(['--run-root', str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(first, (output / 'report.json').read_bytes())

    def test_rejects_mismatched_save_and_orphan_revision(self):
        self.mutate("UPDATE revisions SET agent='wrong' WHERE id=1")
        with self.assertRaisesRegex(ValueError, 'agent or slug mismatch'):
            replay(self.root)
        self.mutate("UPDATE revisions SET agent='agent-2' WHERE id=1")
        self.mutate("UPDATE revisions SET body='different' WHERE id=1")
        with self.assertRaisesRegex(ValueError, 'differs from revision'):
            replay(self.root)
        self.mutate("UPDATE revisions SET body='futureunique' WHERE id=1")
        self.mutate("INSERT INTO revisions VALUES(3,'agent-1','extra','Extra','orphan')")
        with self.assertRaisesRegex(ValueError, 'without a successful'):
            replay(self.root)

    def test_rejects_ambiguous_assignment_and_wrong_boundary(self):
        self.results.append(dict(self.results[0], slot=3))
        self.write_results()
        with self.assertRaisesRegex(ValueError, 'exactly one assignment'):
            replay(self.root)
        self.results.pop()
        self.results[0]['wiki_boundary']['after']['revisions'] = 1
        self.write_results()
        with self.assertRaisesRegex(ValueError, 'boundary disagrees'):
            replay(self.root)

    def test_rejects_corpus_mismatch_and_future_original_urls(self):
        self.pages[0]['text'] = 'corrupt'
        (self.root / 'pages.json').write_text(json.dumps(self.pages))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            replay(self.root)
        self.pages[0]['text'] = 'alpha evidence'
        (self.root / 'pages.json').write_text(json.dumps(self.pages))
        self.mutate('UPDATE audit SET result=? WHERE id=1',
                    (json.dumps({'results': [{'url': 'https://wiki.test/page/note'}]}),))
        with self.assertRaisesRegex(ValueError, 'historically unavailable'):
            replay(self.root)


if __name__ == '__main__':
    unittest.main()
