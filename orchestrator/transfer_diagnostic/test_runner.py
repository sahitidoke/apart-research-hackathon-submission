import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestrator.transfer_diagnostic.runner import (
    BASE_DOCS, INDEX, LOG, PRIVATE, browser, exact_contains, initial_docs,
    initial_messages, main, make_pair, private_text, run_pair, trajectory, validate_action)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.pair = make_pair(1)
        self.target = self.pair['records'][self.pair['target_index']]['accession']
        self.args = SimpleNamespace(steps=3, model='test:1', base_url='http://127.0.0.1:11434',
                                    timeout=1, max_tokens=512, positive_control=False)

    def test_private_source_acl(self):
        for url in (PRIVATE, PRIVATE + '/', PRIVATE + '?x=1', PRIVATE.replace('private', '%70rivate')):
            for kind in ('open', 'edit'):
                chosen = {'action': kind, 'url': url}
                if kind == 'edit':
                    chosen['text'] = 'replacement'
                self.assertIn('error', browser(self.pair, 'B', initial_docs(), chosen))
        result = browser(self.pair, 'B', initial_docs(), {'action': 'search', 'query': ''})
        self.assertNotIn(self.target, json.dumps(result))
        self.assertNotIn(PRIVATE, json.dumps(result))
        self.assertIn(self.target, json.dumps(browser(self.pair, 'A', initial_docs(), {'action': 'open', 'url': PRIVATE})))

    def test_initial_prompts_do_not_share_tasks_or_finals(self):
        for condition in ('neutral', 'encouraged'):
            a = json.dumps(initial_messages(self.pair, 'A', condition))
            b = json.dumps(initial_messages(self.pair, 'B', condition))
            self.assertNotIn(self.pair['b_task'], a)
            self.assertNotIn(self.pair['records'][self.pair['target_index']]['entity'], a)
            self.assertNotIn(self.target, b)
            self.assertEqual('other workers' in b, condition == 'encouraged')
            self.assertNotIn('other workers', json.dumps(initial_messages(self.pair, 'A', 'neutral')))

    def test_shared_snapshot_and_condition_reset(self):
        starts = []
        def fake(pair, actor, condition, docs, args, log, branch, positive=False):
            starts.append((actor, condition, branch, dict(docs)))
            if actor == 'A':
                docs[LOG] = 'sender edit'
            else:
                docs[INDEX] = 'recipient edit'
            return {'actor': actor, 'branch': branch}
        with patch('orchestrator.transfer_diagnostic.runner.trajectory', side_effect=fake):
            run_pair(self.pair, self.args, io.StringIO())
        self.assertEqual([x[1] for x in starts if x[0] == 'A'], ['neutral', 'encouraged'])
        for actor, condition, branch, docs in starts:
            if branch in ('sender', 'withheld_A_edits'):
                self.assertEqual(docs, BASE_DOCS)
            if branch == 'intact':
                self.assertEqual(docs[LOG], 'sender edit')
                self.assertEqual(docs[INDEX], BASE_DOCS[INDEX])
        self.pair['pair'] = 2
        starts.clear()
        with patch('orchestrator.transfer_diagnostic.runner.trajectory', side_effect=fake):
            run_pair(self.pair, self.args, io.StringIO())
        self.assertEqual([x[1] for x in starts if x[0] == 'A'], ['encouraged', 'neutral'])

    def response(self, chosen):
        return {'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(chosen)}}

    def test_emission_and_exposure_require_real_events(self):
        docs = initial_docs()
        answers = [self.response({'action': 'edit', 'url': LOG, 'text': self.target}),
                   self.response({'action': 'final', 'answer': 'finished'})]
        with patch('orchestrator.transfer_diagnostic.runner.request_json', side_effect=answers):
            a = trajectory(self.pair, 'A', 'neutral', docs, self.args, io.StringIO(), 'sender')
        self.assertTrue(a['target_emitted'])
        self.assertFalse(a['target_exposed'])
        with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value=self.response({'action': 'final', 'answer': 'unknown'})):
            b = trajectory(self.pair, 'B', 'neutral', docs, self.args, io.StringIO(), 'intact')
        self.assertFalse(b['target_exposed'])
        answers = [self.response({'action': 'open', 'url': LOG}), self.response({'action': 'final', 'answer': self.target})]
        with patch('orchestrator.transfer_diagnostic.runner.request_json', side_effect=answers):
            b = trajectory(self.pair, 'B', 'neutral', docs, self.args, io.StringIO(), 'intact')
        self.assertTrue(b['target_exposed'])
        self.assertTrue(b['exact_answer'])

    def test_sender_own_task_score_is_separate(self):
        own = self.pair['records'][0]['accession']
        with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value=self.response({'action': 'final', 'answer': own})):
            a = trajectory(self.pair, 'A', 'neutral', initial_docs(), self.args, io.StringIO(), 'sender')
        self.assertTrue(a['task_exact_answer'])
        self.assertFalse(a['exact_answer'])

    def test_sender_final_never_mutates_shared_docs(self):
        docs = initial_docs()
        with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value=self.response({'action': 'final', 'answer': private_text(self.pair)})):
            a = trajectory(self.pair, 'A', 'neutral', docs, self.args, io.StringIO(), 'sender')
        self.assertFalse(a['target_emitted'])
        self.assertEqual(docs, BASE_DOCS)

    def test_step_limit_retains_writes(self):
        self.args.steps = 1
        docs = initial_docs()
        with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value=self.response({'action': 'edit', 'url': INDEX, 'text': self.target})):
            a = trajectory(self.pair, 'A', 'neutral', docs, self.args, io.StringIO(), 'sender')
        self.assertEqual(a['status'], 'step_limit')
        self.assertTrue(a['target_emitted'])
        self.assertEqual(docs[INDEX], self.target)

    def test_incomplete_and_invalid_responses_preserve_logs(self):
        for response, status in (({'done': True, 'done_reason': 'length', 'message': {'content': json.dumps({'action': 'final', 'answer': self.target})}}, 'incomplete_response'),
                                 ({'done': True, 'message': {'content': '{"action":"bad"}'}}, 'error')):
            log = io.StringIO()
            with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value=response):
                row = trajectory(self.pair, 'B', 'neutral', initial_docs(), self.args, log, 'intact')
            self.assertEqual(row['status'], status)
            self.assertFalse(row['exact_answer'])
            events = [json.loads(line) for line in log.getvalue().splitlines()]
            self.assertEqual(events[1]['response'], response)
            self.assertEqual(events[-1]['event'], 'end')
        log = io.StringIO()
        with patch('orchestrator.transfer_diagnostic.runner.request_json', side_effect=OSError('transport failure')):
            row = trajectory(self.pair, 'B', 'neutral', initial_docs(), self.args, log, 'intact')
        self.assertEqual(row['status'], 'error')
        self.assertIn('transport failure', log.getvalue())

    def test_invalid_inputs_leave_no_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'absent'
            for extra in (['--pairs', '0'], ['--steps', '9'], ['--timeout', 'nan'], ['--base-url', 'http://example.com']):
                with self.assertRaises(SystemExit):
                    main(['--model', 'test:1', '--run-dir', str(target)] + extra)
                self.assertFalse(target.exists())
            with patch('orchestrator.transfer_diagnostic.runner.request_json', return_value={'models': []}):
                with self.assertRaises(SystemExit):
                    main(['--model', 'test:1', '--run-dir', str(target)])
            self.assertFalse(target.exists())

    def test_action_validation_and_numeric_boundaries(self):
        for value in ({'action': 'shell', 'text': 'x'}, {'action': 'open', 'url': PRIVATE, 'extra': 'x'},
                      {'action': 'final', 'answer': ''}, {'action': 'edit', 'url': LOG, 'text': 3}):
            with self.assertRaises(ValueError):
                validate_action(value)
        self.assertTrue(exact_contains('Number: ' + self.target + '.', self.target))
        self.assertFalse(exact_contains('1' + self.target, self.target))


if __name__ == '__main__':
    unittest.main()
