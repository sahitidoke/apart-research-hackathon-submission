from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.async_notebooks import AsyncBudget, EventLog, build_async_settings, make_browser, run_async_notebooks
from orchestrator.simulated_web.append_notebooks import append_entry, append_open
from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, VLLM_VERSION, KV_CACHE_DTYPE
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_paired_notebook_views import inputs as paired_inputs


def inputs():
    result = paired_inputs()
    result.pop('round_leaders')
    return result


def tool(name, **arguments):
    return {'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}


class ScriptedClient:
    native_context_preflight = deadline_cancellation_guaranteed = True

    def __init__(self, scripts=None):
        self.scripts = {agent: list(script) for agent, script in (scripts or {}).items()}
        self.calls = []
        self.allowances = []

    def inspect_model(self):
        return {'name': MODEL, 'revision': REVISION, 'digest': 'hf:' + REVISION,
                'backend': 'vllm', 'backend_version': VLLM_VERSION, 'kv_cache_dtype': KV_CACHE_DTYPE,
                'all_artifact_checksums_verified': True}

    def ensure_ready(self, timeout):
        return {'status': 'mock_ready'}

    def count_context(self, history, timeout):
        return {'prompt_tokens': 100}

    def count_text(self, text, timeout=15):
        return {'tokens': 1}

    def __call__(self, agent, history, timeout, *, num_predict):
        self.calls.append((agent, json.loads(json.dumps(history))))
        self.allowances.append((agent, num_predict))
        script = self.scripts.get(agent, [])
        message = script.pop(0) if script else {'content': 'final answer'}
        if isinstance(message, BaseException):
            raise message
        if isinstance(message, ModelResponse):
            return message
        return ModelResponse(message, {'eval_count': min(3, num_predict), 'prompt_eval_count': 100, 'done_reason': 'stop'})


class AsyncTests(unittest.TestCase):
    def test_optional_no_notes_and_no_forced_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            client = ScriptedClient()
            result = run_async_notebooks(path, client, **inputs())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['results']), 4)
            with sqlite3.connect(path / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT count(*) FROM audit').fetchone()[0], 0)
            rows = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            available = [r['sequence'] for r in rows if r['event'] == 'question_available'][:2]
            first_slot = next(r['sequence'] for r in rows if r['event'] == 'inference_slot_granted')
            self.assertTrue(all(seq < first_slot for seq in available))
            self.assertFalse(any('notebook_' in row['event'] for row in rows))

    def test_publication_read_before_author_final_and_independent_advancement(self):
        settings, _, _ = build_async_settings(**inputs())
        entry = 'https://wiki.test/page/' + settings['notebooks']['agent-1'] + '-entry-000001'
        client = ScriptedClient({'agent-1': [tool('append_notebook', text='Early evidence'),
                                            tool('open', url='https://docs.test/'), {'content': 'A final'}],
                                 'agent-2': [tool('read_notebook', url=entry, revision=''), {'content': 'B final'}]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            run_async_notebooks(path, client, **inputs())
            rows = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            publish = next(r for r in rows if r['event'] == 'notebook_published_available')
            read = next(r for r in rows if r['event'] == 'notebook_read_returned')
            afinal = next(r for r in rows if r['event'] == 'question_terminated' and r['agent'] == 'agent-1')
            bnext = next(r for r in rows if r['event'] == 'question_available' and r['agent'] == 'agent-2' and r['question_index'] == 1)
            self.assertLess(publish['sequence'], read['sequence'])
            self.assertLess(read['sequence'], afinal['sequence'])
            self.assertLess(bnext['sequence'], afinal['sequence'])
            self.assertEqual(publish['revision'], read['revision'])
            self.assertEqual(read['agent'], 'agent-2')
            self.assertIn('Early evidence', json.dumps(client.calls[3][1]))
            with sqlite3.connect(path / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0], 1)

    def test_atomic_concurrent_appends_and_failed_append_rollback(self):
        settings, pages, _ = build_async_settings(**inputs())
        with tempfile.TemporaryDirectory() as temp:
            browser = make_browser(settings, pages, Path(temp) / 'db', EventLog(Path(temp) / 'events'))
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda i: browser.call('agent-1', 'append_notebook', {'text': f'entry {i}'}), range(12)))
                self.assertEqual(len({r['saved'] for r in results}), 12)
                self.assertEqual(len({r['revision'] for r in results}), 12)
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 12)
                def fail(*args, **kwargs):
                    browser.db.execute('INSERT INTO pages VALUES(?,?,?)', ('bad-entry', 'bad', 'bad'))
                    raise ValueError('injected failure')
                with patch('orchestrator.simulated_web.async_notebooks.append_entry', side_effect=fail):
                    self.assertIn('error', browser.call('agent-1', 'append_notebook', {'text': 'not committed'}))
                self.assertIsNone(browser.db.execute('SELECT 1 FROM pages WHERE slug=?', ('bad-entry',)).fetchone())
            finally:
                browser.close()

    def test_budget_bounds_append_loops_and_native_token_accounting(self):
        repeated = [tool('append_notebook', text='bounded') for _ in range(10)]
        client = ScriptedClient({'agent-1': repeated, 'agent-2': repeated})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            result = run_async_notebooks(path, client, **inputs(), budget=AsyncBudget(
                generated_tokens=6, tool_calls=1, model_requests=2))
            self.assertTrue(all(r['status'] == 'budget_exhausted' for r in result['results']))
            self.assertTrue(all(r['generated_tokens'] == 6 and r['tool_calls'] == 1 for r in result['results']))
            with sqlite3.connect(path / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0], 4)

    def test_backend_failure_preserves_peer_and_partial_state(self):
        client = ScriptedClient({'agent-1': [TimeoutError('injected backend deadline')]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            with self.assertRaises(TimeoutError):
                run_async_notebooks(path, client, **inputs())
            states = json.loads((path / 'state.json').read_text())
            self.assertTrue(all(s['status'] == 'interrupted' for s in states.values()))
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['status'], 'failed')
            self.assertEqual(len(client.calls), 1)
            self.assertTrue((path / 'histories.json').is_file())

    def test_invalid_budget_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            with self.assertRaises(ValueError):
                run_async_notebooks(path, ScriptedClient(), **inputs(), budget=AsyncBudget(tool_calls=0))
            self.assertFalse(path.exists())

    def test_truncated_append_is_charged_but_not_executed(self):
        truncated = ModelResponse(tool('append_notebook', text='must not save'),
                                  {'eval_count': 3, 'prompt_eval_count': 100, 'done_reason': 'length'})
        client = ScriptedClient({'agent-1': [truncated]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            result = run_async_notebooks(path, client, **inputs())
            first = next(row for row in result['results'] if row['agent'] == 'agent-1')
            self.assertEqual(first['generated_tokens'], 3)
            self.assertEqual(first['status'], 'generation_limit_reached')
            self.assertEqual(first['termination_detail']['question_tokens_remaining'], 4093)
            self.assertFalse(first['termination_detail']['continuation_attempted'])
            with sqlite3.connect(path / 'wiki.sqlite3') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0], 0)

    def test_full_remaining_allowance_supports_tool_then_final_without_artificial_restart(self):
        research = ModelResponse({**tool('append_notebook', text='Complete research'), 'thinking': 'Long completed reasoning'},
                                 {'eval_count': 3000, 'prompt_eval_count': 100, 'done_reason': 'tool_calls'})
        final = ModelResponse({'content': 'Final answer'},
                              {'eval_count': 100, 'prompt_eval_count': 100, 'done_reason': 'stop'})
        client = ScriptedClient({'agent-1': [research, final]})
        with tempfile.TemporaryDirectory() as temp:
            result = run_async_notebooks(Path(temp) / 'run', client, **inputs())
            first = next(row for row in result['results'] if row['agent'] == 'agent-1')
            self.assertEqual(first['status'], 'answered')
            self.assertEqual(first['generated_tokens'], 3100)
            self.assertEqual(first['tool_calls'], 1)
            self.assertEqual([limit for agent, limit in client.allowances if agent == 'agent-1'][:2], [4096, 1096])
            self.assertEqual([agent for agent, _ in client.calls][:3], ['agent-1', 'agent-2', 'agent-1'])

    def test_repeated_length_terminals_end_distinct_questions_never_resume_same_question(self):
        partial = ModelResponse({'content': '', 'thinking': 'Partial reasoning'},
                                {'eval_count': 3, 'prompt_eval_count': 100, 'done_reason': 'length', 'requested_num_predict': 3})
        client = ScriptedClient({'agent-1': [partial, partial, tool('append_notebook', text='must not execute')]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            result = run_async_notebooks(path, client, **inputs())
            rows = [row for row in result['results'] if row['agent'] == 'agent-1']
            self.assertEqual([row['status'] for row in rows], ['generation_limit_reached'] * 2)
            self.assertEqual([row['question_index'] for row in rows], [0, 1])
            self.assertTrue(all(row['termination_detail']['backend_requested_num_predict'] == 3 for row in rows))
            histories = [history for agent, history in client.calls if agent == 'agent-1']
            self.assertEqual(len(histories), 2)
            self.assertEqual(histories[-1][-1]['role'], 'user')
            self.assertIn('Question:', histories[-1][-1]['content'])
            self.assertNotEqual(histories[0][-1]['content'], histories[-1][-1]['content'])

    def test_unicode_direct_append_has_character_limit_and_preserves_legacy_url_limit(self):
        settings, pages, _ = build_async_settings(**inputs())
        with tempfile.TemporaryDirectory() as temp:
            browser = make_browser(settings, pages, Path(temp) / 'db', EventLog(Path(temp) / 'events'))
            try:
                for char in ('界', '😀'):
                    text = char * 6000
                    saved = browser.call('agent-1', 'append_notebook', {'text': text})
                    self.assertEqual(saved['author'], settings['visible_labels']['agent-1'])
                    read = browser.call('agent-2', 'read_notebook', {'url': saved['saved'], 'revision': saved['revision']})
                    self.assertEqual(read['text'], text)
                    with self.assertRaisesRegex(ValueError, 'Invalid URL'):
                        append_open(browser, 'agent-1', 'https://wiki.test/append?' + urlencode({
                            'slug': settings['notebooks']['agent-1'], 'text': text}), settings['notebooks'], True)
                for text in ('界' * 6001, '😀' * 6001, '   '):
                    self.assertIn('error', browser.call('agent-1', 'append_notebook', {'text': text}))
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 2)
                with browser.lock, browser.db:
                    with self.assertRaisesRegex(ValueError, 'access denied'):
                        append_entry(browser, 'agent-1', settings['notebooks']['agent-2'], 'foreign', settings['notebooks'], True)
                    handled, saved = append_open(browser, 'agent-1', 'https://wiki.test/append?' + urlencode({
                        'slug': settings['notebooks']['agent-1'], 'text': 'legacy'}), settings['notebooks'], True)
                    self.assertTrue(handled)
                    self.assertEqual(set(saved), {'saved', 'notebook', 'operation'})
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 3)
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
