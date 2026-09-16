"""Focused synthetic histories for the offline selection diagnostic."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser, MAX_TEXT
from orchestrator.simulated_web.selection_audit import audit_selection, main
from orchestrator.simulated_web.session import prepare


class SelectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'session'
        (self.root / 'web').mkdir(parents=True)
        paragraph = lambda title, text, supporting: {'idx': 91, 'title': title, 'paragraph_text': text,
                                                     'is_supporting': supporting}
        alpha = paragraph('Alpha', 'alpha evidence', True)
        beta = paragraph('Beta', 'beta ' + 'x' * MAX_TEXT, True)
        delta = paragraph('Delta', 'delta evidence', True)
        gamma = paragraph('Gamma', 'gamma evidence', True)
        self.records = [
            {'id': 'q1', 'question': 'Question one?', 'paragraphs': [alpha, beta, delta, dict(gamma, is_supporting=False)]},
            {'id': 'q2', 'question': 'Question two?', 'paragraphs': [dict(alpha, is_supporting=False),
             dict(beta, is_supporting=False), gamma]}]
        self.pages, _, _, _ = prepare(self.records, 2, 0, 'neutral')
        (self.root / 'dataset.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.records))
        (self.root / 'pages.json').write_text(json.dumps(self.pages))
        b = Browser(self.pages, self.root / 'web' / 'wiki.sqlite3')
        before = b.checkpoint()
        b.call('agent-1', 'open', {'url': self.url('q1', 'p/0/0')})  # 1: alpha before exposure
        b.call('agent-1', 'search', {'query': 'recorded query'})  # 2
        selected = [self.url('q2', 'p/0/0'), self.url('q1', 'p/1/0'),
                    self.url('q1', 'p/1/1'), self.url('q1', 'p/2/0')]
        b.db.execute('UPDATE audit SET result=? WHERE id=2', (json.dumps({'results': [{'url': u} for u in selected]}),))
        index = b.call('agent-1', 'open', {'url': self.url('q1', '')})  # 3
        b.call('agent-1', 'click', {'page_id': index['page_id'], 'link_id': 2})  # 4: beta
        b.call('agent-1', 'open', {'url': self.url('q2', 'p/1/0')})  # 5: identical beta copy
        b.call('agent-1', 'open', {'url': self.url('q1', 'p/1/1')})  # 6: second beta chunk
        b.call('agent-1', 'open', {'url': self.url('q1', 'p/3/0')})  # 7: irrelevant gamma
        b.call('agent-1', 'open', {'url': 'https://wiki.test/'})  # 8
        b.call('agent-1', 'open', {'url': 'https://wiki.test/save?' + urlencode(
            {'slug': 'note', 'title': 'Note', 'text': 'content'})})  # 9: save, not read
        b.call('agent-1', 'click', {'page_id': 'invalid', 'link_id': 1})  # 10: error
        b.call('agent-2', 'search', {'query': 'gamma'})  # 11: gamma is shown
        b.call('agent-2', 'open', {'url': 'https://docs.test/missing'})  # 12: error, not read
        after = b.checkpoint()
        b.close()
        self.results = [dict(agent=agent, id=qid, slot=slot, wiki_boundary={'before': lo, 'after': hi})
                        for agent, qid, slot, lo, hi in [
                            ('agent-1', 'q1', 1, before, after), ('agent-2', 'q2', 1, before, after),
                            ('agent-1', 'q1', 2, after, after)]]
        self.write_results()

    def url(self, qid, suffix):
        return 'https://docs.test/q/' + hashlib.sha256(qid.encode()).hexdigest() + '/' + suffix

    def write_results(self):
        (self.root / 'results.json').write_text(json.dumps(self.results))

    def mutate(self, sql, parameters=()):
        with sqlite3.connect(self.root / 'web' / 'wiki.sqlite3') as db:
            db.execute(sql, parameters)

    def test_copy_chunk_duplicate_and_click_opens(self):
        report = audit_selection(self.root)
        first = report['assignments'][0]
        self.assertEqual(first['document_open_calls'], 5)
        self.assertEqual(len(first['opened_documents']), 3)
        self.assertEqual(first['metrics']['opened_document_precision']['rate'], 2 / 3)
        self.assertEqual(first['metrics']['opened_support_recall']['rate'], 2 / 3)
        beta = next(s for s in first['support_details'] if s['title'] == 'Beta')
        self.assertEqual(beta['first_exposure_audit_id'], 2)
        self.assertEqual(beta['open_audit_ids'], [4, 5, 6])
        self.assertEqual(beta['state'], 'opened_after_first_exposure')
        self.assertEqual(first['index_open_calls'], 1)
        self.assertEqual(first['wiki_open_calls'], 1)
        self.assertEqual(first['failed_calls'], 1)

    def test_early_open_not_mislabeled_as_skip(self):
        report = audit_selection(self.root)
        first = report['assignments'][0]
        metrics = first['metrics']
        self.assertEqual(metrics['shown_support_count'], 3)
        self.assertEqual(metrics['already_opened_before_first_exposure'], 1)
        self.assertEqual(metrics['shown_support_then_subsequently_opened']['rate'], 1 / 3)
        self.assertEqual(metrics['newly_shown_support_selection']['rate'], 1 / 2)
        self.assertEqual(metrics['shown_but_never_opened'], 1)
        self.assertEqual([s['title'] for s in report['shown_but_never_opened']], ['Delta', 'Gamma'])
        self.assertEqual(report['shown_but_never_opened'][0]['first_exposure_audit_id'], 2)

    def test_early_open_then_reopen_reported_separately(self):
        result = {'url': self.url('q2', 'p/0/0'), 'page_id': 'p99', 'text': 'alpha evidence', 'links': []}
        self.mutate('UPDATE audit SET result=? WHERE id=7', (json.dumps(result),))
        metrics = audit_selection(self.root)['assignments'][0]['metrics']
        self.assertEqual(metrics['already_opened_and_reopened_after_exposure'], 1)
        self.assertEqual(metrics['shown_support_then_subsequently_opened']['rate'], 2 / 3)
        self.assertEqual(metrics['newly_shown_support_selection']['rate'], 1 / 2)

    def test_aggregate_denominators_and_empty_assignment(self):
        report = audit_selection(self.root)
        summary = report['summary']
        self.assertEqual(summary['assignment_count'], 3)
        self.assertEqual(summary['zero_search_assignments'], 1)
        self.assertEqual(summary['zero_document_open_assignments'], 2)
        self.assertEqual(summary['opened_support_recall'], {'numerator': 2, 'denominator': 7, 'rate': 2 / 7})
        self.assertEqual(summary['opened_document_precision']['denominator'], 3)
        self.assertEqual(summary['newly_shown_support_selection']['rate'], 1 / 3)
        self.assertEqual(summary['failed_calls'], 2)
        empty = report['assignments'][2]['metrics']
        self.assertIsNone(empty['opened_document_precision']['rate'])
        self.assertIsNone(empty['newly_shown_support_selection']['rate'])
        self.assertEqual(empty['opened_support_recall']['rate'], 0)

    def test_invalid_search_not_exposure(self):
        self.mutate('UPDATE audit SET result=? WHERE id=11', (json.dumps({'error': 'bad query'}),))
        row = audit_selection(self.root)['assignments'][1]
        self.assertEqual(row['search_calls'], 1)
        self.assertEqual(row['successful_search_calls'], 0)
        self.assertEqual(row['shown_supports'], {})
        self.assertEqual(row['failed_calls'], 2)

    def test_input_immutability_determinism_and_existing_output_refusal(self):
        snapshots = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--run-root', str(self.root)]), 0)
        for path, content in snapshots.items():
            self.assertEqual(path.read_bytes(), content)
        output = self.root / 'selection-audit'
        first = (output / 'report.json').read_bytes()
        other = Path(self.temp.name) / 'another'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--run-root', str(self.root), '--output-dir', str(other)])
        self.assertEqual(first, (other / 'report.json').read_bytes())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(['--run-root', str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(first, (output / 'report.json').read_bytes())

    def test_inconsistent_assignment_rejected_before_output_creation(self):
        self.results.append(dict(self.results[0], slot=4))
        self.write_results()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--run-root', str(self.root)])
        self.assertFalse((self.root / 'selection-audit').exists())
        self.results.pop()
        self.results[0]['wiki_boundary']['after']['audit'] = 0
        self.write_results()
        with self.assertRaisesRegex(ValueError, 'exactly one assignment'):
            audit_selection(self.root)

    def test_unknown_successful_document_rejected(self):
        result = {'url': 'https://docs.test/unknown', 'page_id': 'p1', 'text': 'unmatched'}
        self.mutate('UPDATE audit SET result=? WHERE id=1', (json.dumps(result),))
        with self.assertRaisesRegex(ValueError, 'unknown opened URL'):
            audit_selection(self.root)


if __name__ == '__main__':
    unittest.main()
