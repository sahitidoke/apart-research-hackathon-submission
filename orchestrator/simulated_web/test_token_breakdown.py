"""Synthetic checks for the opt-in offline token breakdown (no model calls)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestrator.simulated_web import token_breakdown as breakdown


class FakeTokenizer:
    """Each nonempty field is one synthetic token, independent of its length."""
    seen = []

    @classmethod
    def from_file(cls, path):
        json.loads(Path(path).read_text())
        return cls()

    def no_truncation(self):
        self.truncation_disabled = True

    def no_padding(self):
        self.padding_disabled = True

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        assert self.truncation_disabled and self.padding_disabled
        self.seen.append(text)
        return SimpleNamespace(ids=[123] if text else [])


class TokenBreakdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'session'
        self.root.mkdir()
        self.tokenizer = Path(self.temp.name) / 'tokenizer.json'
        self.tokenizer.write_text('{}')
        (self.root / 'settings.json').write_text(json.dumps({'settings': {'model': 'fixture'}}))
        self.patcher = patch.object(breakdown, 'Tokenizer', FakeTokenizer)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        FakeTokenizer.seen = []

    def fixture(self, messages, status='error'):
        events = [{'event': 'model_response', 'metadata': {'eval_count': 99}}]
        events.extend({'event': 'assistant', 'message': m} for m in messages)
        events.append({'event': 'final', 'answer': 'duplicate terminal answer', 'status': status})
        (self.root / 'agent.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        (self.root / 'results.json').write_text(json.dumps([
            {'agent': 'agent-1', 'id': 'q1', 'slot': 1, 'status': status, 'log_path': 'agent.jsonl'}]))

    def call(self, name, arguments):
        return {'function': {'name': name, 'arguments': arguments}}

    def test_categories_all_arguments_and_excluded_final(self):
        raw = '{ "url": "https://wiki.test/save?text=a%20b" }'
        self.fixture([{'thinking': 'visible reasoning', 'content': '', 'tool_calls': [
            self.call('search', {'query': 'q'}),
            self.call('open', {'url': 'https://docs.test/doc'}),
            self.call('click', {'page_id': 'a', 'link_id': 1}),
            self.call('open', raw), self.call('hallucinated', {'b': 2, 'a': 'é'}),
            self.call('open', {'url': 'https://wiki.test.evil/save'})]},
            {'content': 'a very long final answer'}])
        report = breakdown.build_report(self.root, self.tokenizer)
        counts = report['aggregate']
        self.assertEqual(counts['denominator_tokens'], 7)
        self.assertEqual(counts['search_read_tokens'], 3)
        self.assertEqual(counts['wiki_write_tokens'], 1)
        self.assertEqual(counts['other_unknown_tokens'], 2)
        self.assertEqual(counts['content_tokens'], 1)
        self.assertEqual(counts['final_content_tokens'], 1)
        self.assertEqual(counts['native_eval_count_tokens'], 99)
        self.assertEqual(counts['shares']['wiki_write_tokens'], 1 / 7)
        self.assertEqual(counts['reasoning_availability'], 'partial')
        self.assertEqual(report['assignments'][0]['status'], 'error')
        self.assertEqual(report['agents']['agent-1'], counts)
        self.assertIn(raw, FakeTokenizer.seen)
        self.assertIn('{"a":"é","b":2}', FakeTokenizer.seen)
        self.assertNotIn('duplicate terminal answer', FakeTokenizer.seen)

    def test_missing_and_visible_zero_reasoning_differ(self):
        self.fixture([{'content': ''}])
        missing = breakdown.build_report(self.root, self.tokenizer)['aggregate']
        self.fixture([{'thinking': '', 'content': ''}])
        empty = breakdown.build_report(self.root, self.tokenizer)['aggregate']
        self.assertEqual(missing['reasoning_availability'], 'unavailable')
        self.assertEqual(empty['reasoning_availability'], 'available')
        self.assertEqual(empty['denominator_tokens'], 0)
        self.assertTrue(all(value is None for value in empty['shares'].values()))

    def test_strict_save_route_and_unknown(self):
        for url in ('http://wiki.test/save', 'https://wiki.test:443/save',
                    'https://wiki.test/save#fragment', 'https://user@wiki.test/save'):
            self.assertEqual(breakdown.category('open', {'url': url}), 'other_unknown')
        self.assertEqual(breakdown.category('open', {'url': 'https://wiki.test/page/note'}), 'search_read')
        self.assertEqual(breakdown.category('fake', '{bad json'), 'other_unknown')

    def test_missing_arguments_and_native_metadata_completeness(self):
        self.fixture([{'content': '', 'tool_calls': [{'function': {'name': 'fake'}}]}])
        with (self.root / 'agent.jsonl').open('a') as log:
            log.write(json.dumps({'event': 'model_response', 'metadata': {}}) + '\n')
        counts = breakdown.build_report(self.root, self.tokenizer)['aggregate']
        self.assertEqual(counts['missing_argument_calls'], 1)
        self.assertEqual(counts['native_eval_count_missing_events'], 1)

    def cli_fails_without_artifacts(self):
        output = self.root / 'new-parent' / 'output'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            breakdown.main(['--run-root', str(self.root), '--tokenizer', str(self.tokenizer),
                            '--output-dir', str(output)])
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse(output.parent.exists())

    def test_bad_log_missing_log_and_escaping_path_leave_no_output(self):
        self.fixture([{'thinking': 4}])
        self.cli_fails_without_artifacts()
        (self.root / 'agent.jsonl').unlink()
        self.cli_fails_without_artifacts()
        outside = Path(self.temp.name) / 'outside.jsonl'
        outside.write_text('{}\n')
        rows = [{'agent': 'agent-1', 'log_path': '../outside.jsonl'}]
        (self.root / 'results.json').write_text(json.dumps(rows))
        self.cli_fails_without_artifacts()
        (self.root / 'agent.jsonl').symlink_to(outside)
        rows[0]['log_path'] = 'agent.jsonl'
        (self.root / 'results.json').write_text(json.dumps(rows))
        self.cli_fails_without_artifacts()

    def test_invalid_tokenizer_and_optional_dependency_leave_no_output(self):
        self.fixture([])
        self.tokenizer.write_text('invalid json')
        self.cli_fails_without_artifacts()
        self.tokenizer.write_text('{}')
        with patch.object(breakdown, 'Tokenizer', None):
            self.cli_fails_without_artifacts()

    def test_writes_fresh_reports_and_preserves_inputs_and_existing_output(self):
        self.fixture([{'reasoning': 'visible', 'content': 'answer'}], status='complete')
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        argv = ['--run-root', str(self.root), '--tokenizer', str(self.tokenizer.parent)]
        self.assertEqual(breakdown.main(argv), 0)
        output = self.root / 'token-breakdown'
        first = (output / 'report.json').read_bytes()
        report = json.loads(first)
        self.assertEqual(report['tokenizer']['path'], str(self.tokenizer))
        self.assertEqual(len(report['tokenizer']['sha256']), 64)
        self.assertIn('Content (excluded)', (output / 'report.md').read_text())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            breakdown.main(argv)
        self.assertEqual(first, (output / 'report.json').read_bytes())
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))


if __name__ == '__main__':
    unittest.main()
