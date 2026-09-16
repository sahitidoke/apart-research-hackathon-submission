import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.simulated_web.compaction import compact_between_questions, validate_summary_links
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.timed import TimedPolicy
from orchestrator.simulated_web.timed_transport import DeadlineExpired


class CompactionTests(unittest.TestCase):
    def test_retained_ceiling_below_configured_trigger(self):
        with self.assertRaisesRegex(ValueError, 'below trigger'):
            TimedPolicy(context_length=4096)
        TimedPolicy(context_length=4096, compaction_retained_tokens=2048)

    def client(self, count):
        client = Mock()
        client.ensure_ready.return_value = {'status': 'ready'}
        client.count_context.side_effect = [{'prompt_tokens': count}, {'prompt_tokens': 3500}]
        client.side_effect = [ModelResponse({'content': 'Fact https://docs.test/source; prior answer A; uncertain B'}, {'done_reason': 'stop'}),
                              ModelResponse({'content': ''}, {'done_reason': 'length'})]
        return client

    def test_native_boundary_and_transactional_replacement(self):
        for count in (49151, 49152):
            with tempfile.TemporaryDirectory() as folder:
                history = [{'role': 'system', 'content': 'trusted'}, {'role': 'assistant', 'content': 'old https://docs.test/source'}]
                original = json.loads(json.dumps(history))
                client = self.client(count)
                row = compact_between_questions(client, history, TimedPolicy(), Path(folder), 2)
                self.assertEqual(row['status'], 'compacted' if count == 49152 else 'not_needed')
                if count == 49152:
                    self.assertEqual(history[0], original[0])
                    self.assertEqual(history[1]['role'], 'user')
                    self.assertEqual(json.loads((Path(folder)/row['raw_path']).read_text()), original)
                    self.assertTrue(client.call_args_list[0].kwargs['final_only'])
                    self.assertEqual(client.call_args_list[1].kwargs['num_predict'], 1)
                else:
                    client.assert_not_called()
                    self.assertEqual(history, original)

    def test_oversize_or_failed_warm_preserves_history(self):
        for failure in ('size', 'warm', 'length'):
            with tempfile.TemporaryDirectory() as folder:
                history = [{'role': 'system', 'content': 'trusted'}, {'role': 'user', 'content': 'old https://docs.test/source'}]
                original = list(history)
                client = self.client(100000)
                if failure == 'size':
                    client.count_context.side_effect = [{'prompt_tokens': 100000}, {'prompt_tokens': 16385}]
                elif failure == 'warm':
                    client.side_effect = [ModelResponse({'content': 'summary'}, {'done_reason': 'stop'}), TimeoutError('warm')]
                else:
                    client.side_effect = [ModelResponse({'content': 'partial'}, {'done_reason': 'length'})] * 2
                with self.assertRaises((ValueError, TimeoutError)):
                    compact_between_questions(client, history, TimedPolicy(), Path(folder), 2)
                self.assertEqual(history, original)
                self.assertEqual(json.loads((Path(folder)/'compaction-02.json').read_text())['status'], 'failed')


class SummaryLinkTests(unittest.TestCase):
    url = 'https://docs.test/q/c85e189141' + 'a' * 54 + '/p/12/0'

    def test_exact_and_unique_abbreviation_preserve_observed_url(self):
        history = [{'role': 'tool', 'content': json.dumps({'links': [{'url': self.url}]})}]
        for reference in (self.url, 'https://docs.test/q/c85e189141.../p/12/0', '/q/c85e189141…/p/12/0'):
            fixed, metadata = validate_summary_links('Source: ' + reference, history)
            self.assertEqual(fixed, 'Source: ' + self.url)
            self.assertFalse(metadata['errors'])

    def test_unobserved_and_ambiguous_links_fail(self):
        other = self.url.replace('a' * 54, 'b' * 54)
        history = [{'content': self.url + ' ' + other}]
        for reference, reason in [('/p/12/0', 'ambiguous'), ('/q/c85e189141.../p/12/0', 'ambiguous'),
                                  ('https://docs.test/q/invented/p/12/0', 'unobserved')]:
            _, metadata = validate_summary_links(reference, history)
            self.assertEqual(metadata['errors'][0]['reason'], reason)

    def test_bad_link_preserves_history_without_count_or_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            history = [{'role': 'system', 'content': 'trusted'}, {'role': 'tool', 'content': self.url}]
            original = json.loads(json.dumps(history))
            client = CompactionTests().client(50000)
            client.side_effect = [ModelResponse({'content': 'Source /p/13/0'}, {'done_reason': 'stop'})] * 2
            with self.assertRaisesRegex(ValueError, 'document links'):
                compact_between_questions(client, history, TimedPolicy(), Path(folder), 2)
            self.assertEqual(history, original)
            self.assertEqual(client.call_count, 2)
            self.assertEqual(client.count_context.call_count, 1)
            report = json.loads((Path(folder) / 'compaction-02.json').read_text())
            self.assertEqual(report['summary'], 'Source /p/13/0')
            self.assertEqual(report['link_validation']['errors'][0]['reason'], 'unobserved')

    def test_repairs_are_counted_and_warmed_with_raw_summary_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            history = [{'role': 'system', 'content': 'trusted'}, {'role': 'tool', 'content': self.url}]
            raw = 'Source /q/c85e189141.../p/12/0'
            client = CompactionTests().client(50000)
            client.side_effect = [ModelResponse({'content': raw}, {'done_reason': 'stop'}),
                                  ModelResponse({'content': ''}, {'done_reason': 'length'})]
            report = compact_between_questions(client, history, TimedPolicy(), Path(folder), 2)
            self.assertEqual(report['summary'], raw)
            self.assertEqual(report['validated_summary'], 'Source ' + self.url)
            self.assertIn(self.url, client.count_context.call_args.args[0][1]['content'])
            self.assertIn(self.url, client.call_args.args[1][1]['content'])

    def test_missing_scheme_cannot_create_malformed_partial_repair(self):
        _, report = validate_summary_links(self.url.removeprefix('https://'), [{'content': self.url}])
        self.assertEqual(report['errors'][0]['reason'], 'missing_scheme')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.history = [{'role': 'system', 'content': 'trusted'},
                        {'role': 'tool', 'content': 'Established fact https://docs.test/source'}]
        self.original = json.loads(json.dumps(self.history))
        self.client = CompactionTests().client(50000)
        self.good = ModelResponse({'content': 'Established fact https://docs.test/source'}, {'done_reason': 'stop'})
        self.warm = ModelResponse({'content': ''}, {'done_reason': 'length'})

    def run_compaction(self, **kwargs):
        return compact_between_questions(self.client, self.history, TimedPolicy(**kwargs), self.path, 1)

    def report(self):
        return json.loads((self.path / 'compaction-01.json').read_text())

    def test_recovery_candidates_and_original_input(self):
        for message, metadata in [({'content': 'partial'}, {'done_reason': 'length'}),
                                  ({'content': 'unknown https://docs.test/missing'}, {'done_reason': 'stop'}),
                                  ({'content': ''}, {'done_reason': 'stop'}),
                                  ({'content': 'fact', 'tool_calls': [{'function': {}}]}, {'done_reason': 'stop'}),
                                  ({'content': 'fact', 'thinking': 'reason'}, {'done_reason': 'stop'}),
                                  ({'content': 'partial'}, {}),
                                  ({'content': 'partial'}, {'done_reason': 'unknown'})]:
            with self.subTest(message=message, metadata=metadata):
                self.history[:] = self.original
                self.client = CompactionTests().client(50000)
                self.client.side_effect = [ModelResponse(message, metadata), self.good, self.warm]
                report = self.run_compaction()
                self.assertEqual(report['status'], 'compacted')
                self.assertEqual(len(report['attempts']), 2)
                self.assertEqual(report['attempts'][0]['raw_message'], message)
                self.assertEqual(report['attempts'][0]['summary_metadata'], metadata)
                self.assertIn('rejection', report['attempts'][0])
                self.assertEqual(report['attempts'][1]['target_tokens'], 1024)
                for call in self.client.call_args_list[:2]:
                    self.assertEqual(call.args[1][:-1], self.original)
                    self.assertTrue(call.kwargs['final_only'])
                    self.assertEqual(call.kwargs['num_predict'], 4096)
                self.assertEqual(self.client.ensure_ready.call_count, 2)
                self.assertIn(report['attempts'][0]['rejection'], report['attempts'][1]['request'])
                self.assertEqual(report['attempts'][0]['validation_reserve_seconds'], 35)
                if metadata != {'done_reason': 'stop'}:
                    self.assertNotIn('link_validation', report['attempts'][0])

    def test_target_scales_with_output_cap(self):
        self.client.side_effect = [self.good, self.warm]
        report = self.run_compaction(compaction_output_tokens=256)
        self.assertEqual(report['attempts'][0]['target_tokens'], 102)
        self.assertIn('at most 102 tokens', self.client.call_args_list[0].args[1][-1]['content'])
        self.assertEqual(self.client.call_args_list[0].kwargs['num_predict'], 256)

    def test_failure_retains_all_attempt_artifacts(self):
        self.client.side_effect = [ModelResponse({'content': 'partial'}, {'done_reason': 'length'})] * 2
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_compaction()
        report = self.report()
        self.assertEqual(self.history, self.original)
        self.assertEqual(len(report['attempts']), 2)
        self.assertEqual([row['status'] for row in report['attempts']], ['rejected'] * 2)
        self.assertEqual(self.client.count_context.call_count, 1)
        self.assertFalse((self.path / 'history-after-compaction-01.json').exists())

    def test_certified_timeout_can_recover_without_partial_history(self):
        error = DeadlineExpired({'content': 'partial', 'tool_calls': [{'partial': True}]}, 1)
        error.cancellation = {'event': 'request_cancelled_killed', 'server_retained': False}
        self.client.side_effect = [error, self.good, self.warm]
        report = self.run_compaction()
        self.assertEqual(report['status'], 'compacted')
        self.assertEqual(report['attempts'][0]['raw_message'], error.partial)
        self.assertEqual(self.client.call_args_list[1].args[1][:-1], self.original)
        self.assertEqual(self.client.ensure_ready.call_count, 2)

    def test_uncertified_transport_error_is_terminal(self):
        for error in (DeadlineExpired({'content': 'partial'}, 2), RuntimeError('cleanup failed')):
            with self.subTest(error=error):
                self.client = CompactionTests().client(50000)
                self.client.side_effect = error
                with self.assertRaises(type(error)):
                    self.run_compaction()
                self.assertEqual(self.client.call_count, 1)
                self.assertEqual(self.history, self.original)

    def test_count_and_warm_failures_never_replace_history(self):
        for failure in ('count', 'warm', 'warm_ack'):
            with self.subTest(failure=failure):
                self.client = CompactionTests().client(50000)
                self.client.side_effect = [self.good, self.warm]
                if failure == 'count':
                    self.client.count_context.side_effect = [{'prompt_tokens': 50000}, TimeoutError('count')]
                elif failure == 'warm':
                    self.client.side_effect = [self.good, TimeoutError('warm')]
                else:
                    self.client.side_effect = [self.good, ModelResponse({'content': ''}, {})]
                with self.assertRaises((TimeoutError, ValueError)):
                    self.run_compaction()
                self.assertEqual(self.history, self.original)
                self.assertEqual(len(self.report()['attempts']), 1)

    def test_late_calls_and_insufficient_budget_fail_closed(self):
        for stage in ('initial_count', 'summary', 'retry', 'recovery', 'count', 'warm'):
            with self.subTest(stage=stage):
                clock = [0.0]
                self.client = CompactionTests().client(50000)
                counts = [0]
                requests = [0]
                def count(*args, **kwargs):
                    counts[0] += 1
                    if stage == 'initial_count' or (stage == 'count' and counts[0] == 2):
                        clock[0] = 181
                    return {'prompt_tokens': 50000 if counts[0] == 1 else 3500}
                def generate(*args, **kwargs):
                    requests[0] += 1
                    if requests[0] == 1:
                        if stage in ('summary', 'retry'):
                            clock[0] = 181 if stage == 'summary' else 146
                        if stage in ('retry', 'recovery'):
                            return ModelResponse({'content': 'partial'}, {'done_reason': 'length'})
                        return self.good
                    if stage == 'warm':
                        clock[0] = 181
                    return self.warm
                def ready(*args, **kwargs):
                    if stage == 'recovery' and self.client.ensure_ready.call_count == 2:
                        clock[0] = 150
                    return {'status': 'ready'}
                self.client.count_context.side_effect = count
                self.client.side_effect = generate
                self.client.ensure_ready.side_effect = ready
                with patch('orchestrator.simulated_web.compaction.time.monotonic', side_effect=lambda: clock[0]):
                    with self.assertRaises(TimeoutError):
                        self.run_compaction()
                self.assertEqual(self.history, self.original)
                if stage in ('summary', 'retry', 'recovery'):
                    self.assertEqual(requests[0], 1)
                if stage == 'count':
                    self.assertEqual(requests[0], 1)
                self.assertEqual(self.report()['status'], 'failed')
