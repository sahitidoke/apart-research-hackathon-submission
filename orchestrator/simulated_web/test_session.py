"""Mocked session plumbing checks; no inference or external services."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import urlencode
from unittest.mock import patch

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse, Ollama, SYSTEM
from orchestrator.simulated_web.session import FINAL_ANSWER_INSTRUCTION, export_session, make_schedule, prepare, run_session


def records(count=3):
    return [{'id': f'q{i}', 'question': f'Question {i}?', 'answerable': True,
             'answer': 'PRIVATE_GOLD', 'answer_aliases': ['PRIVATE_ALIAS'],
             'question_decomposition': [{'answer': 'PRIVATE_DECOMPOSITION'}],
             'paragraphs': [{'title': f'Document {i}', 'paragraph_text': f'Evidence {i}',
                             'is_supporting': True, 'idx': 100 + i}]}
            for i in range(count)]


class ScheduleTests(unittest.TestCase):
    def test_reproducible_collision_free_and_separate_seeds(self):
        ids = [f'q{i}' for i in range(20)]
        first = make_schedule(ids, 5, 13)
        self.assertEqual(first, make_schedule(list(reversed(ids)), 5, 13))
        self.assertNotEqual(first['orders'], make_schedule(ids, 5, 14)['orders'])
        for order in first['orders'].values():
            self.assertEqual(sorted(order), sorted(ids))
        for slot in zip(*first['orders'].values()):
            self.assertEqual(len(set(slot)), 5)
        self.assertEqual(first['model_seeds'], {f'agent-{i}': 12 + i for i in range(1, 6)})
        self.assertEqual(len(set(first['order_seeds'].values())), 5)
        self.assertTrue(set(first['order_seeds'].values()).isdisjoint(first['model_seeds'].values()))

    def test_four_shards_preserve_corpus_prompt_and_generation_seed(self):
        source = records(20)
        pages, tasks, baseline, prompt = prepare(source, 1, 0, 'neutral')
        combined = []
        for shard in range(4):
            shard_pages, shard_tasks, schedule, shard_prompt = prepare(
                source, 1, 0, 'neutral', shard, 4)
            order = schedule['orders']['agent-1']
            self.assertEqual(len(order), 5)
            self.assertEqual(order, baseline['orders']['agent-1'][shard::4])
            self.assertEqual((shard_pages, shard_tasks, shard_prompt), (pages, tasks, prompt))
            self.assertEqual(schedule['model_seeds'], baseline['model_seeds'])
            self.assertEqual(schedule['shard']['full_order'], baseline['orders']['agent-1'])
            combined.extend(order)
        self.assertEqual(sorted(combined), sorted(row['id'] for row in source))
        self.assertEqual(Ollama('model', 11434, 0, 262144, 8192).seed_for_agent('agent-1'), 0)

    def test_invalid_shards_leave_no_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            for shard, shards, agents, condition in (
                    (-1, 4, 1, 'neutral'), (4, 4, 1, 'neutral'),
                    (0, 0, 1, 'neutral'), (0, 21, 1, 'neutral'),
                    (0, 4, 2, 'neutral'), (0, 4, 2, 'maximal')):
                with self.assertRaises(ValueError):
                    run_session(root, records(20), None, agents=agents,
                                prompt_condition=condition, shard=shard, shards=shards)
                self.assertFalse(root.exists())
            source = records(20)
            source[-1]['paragraphs'] = []
            with self.assertRaises(ValueError):
                run_session(root, source, None, agents=1, prompt_condition='neutral',
                            shard=0, shards=4)
            self.assertFalse(root.exists())

    def test_question_selection_preserves_full_corpus_and_schedule_provenance(self):
        source = records(20)
        pages, tasks, baseline, prompt = prepare(source, 1, 0, 'neutral')
        selected_pages, selected_tasks, schedule, selected_prompt = prepare(
            source, 1, 0, 'neutral', question_id='q7')
        self.assertEqual((selected_pages, selected_tasks, selected_prompt), (pages, tasks, prompt))
        self.assertEqual(schedule['orders'], {'agent-1': ['q7']})
        self.assertEqual(schedule['model_seeds'], baseline['model_seeds'])
        self.assertEqual(schedule['selection']['full_order'], baseline['orders']['agent-1'])
        self.assertEqual(schedule['selection']['original_slot'],
                         baseline['orders']['agent-1'].index('q7') + 1)
        self.assertEqual(schedule['selection']['question_id'], 'q7')

    def test_invalid_selection_leaves_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            for question_id, agents, condition, shards in (
                    ('missing', 1, 'neutral', 1), ('', 1, 'neutral', 1),
                    (7, 1, 'neutral', 1), ('q7', 2, 'neutral', 1),
                    ('q7', 2, 'maximal', 1), ('q7', 1, 'neutral', 4)):
                with self.assertRaises(ValueError):
                    run_session(root, records(20), None, agents=agents,
                                prompt_condition=condition, shards=shards, question_id=question_id)
                self.assertFalse(root.exists())
            duplicate = records(20)
            duplicate[-1]['id'] = 'q7'
            invalid = records(20)
            invalid[-1]['paragraphs'] = []
            for source in (duplicate, invalid):
                with self.assertRaises(ValueError):
                    run_session(root, source, None, agents=1, prompt_condition='neutral',
                                question_id='q7')
                self.assertFalse(root.exists())

    def test_condition_does_not_change_schedule(self):
        maximal = prepare(records(), 2, 4, 'maximal')
        neutral = prepare(records(), 2, 4, 'neutral')
        self.assertEqual(maximal[:3], neutral[:3])
        self.assertEqual(neutral[3], SYSTEM + ' ' + FINAL_ANSWER_INSTRUCTION)
        self.assertTrue(maximal[3].endswith(FINAL_ANSWER_INSTRUCTION))
        self.assertIn('Create or edit a page', maximal[3])
        self.assertIn('write a concise note to the wiki', maximal[3])

    def test_invalid_inputs_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run'
            for inputs, agents in (([], 2), (records(1), 2), (records(), 33), (records(), 1)):
                with self.assertRaises(ValueError):
                    run_session(path, inputs, None, agents=agents)
                self.assertFalse(path.exists())
            invalid = records()
            invalid[1]['id'] = invalid[0]['id']
            with self.assertRaises(ValueError):
                run_session(path, invalid, None)

    def test_stable_links_and_no_private_labels(self):
        source = records()
        original = copy.deepcopy(source)
        pages, tasks, _, _ = prepare(source, 2, 0, 'maximal')
        self.assertEqual(source, original)
        serialized = json.dumps(pages)
        for private in ('PRIVATE_GOLD', 'PRIVATE_ALIAS', 'PRIVATE_DECOMPOSITION', 'is_supporting'):
            self.assertNotIn(private, serialized)
        urls = {page['url'] for page in pages}
        self.assertEqual(len(urls), len(pages))
        for page in pages:
            for link in page['links']:
                if link['url'].startswith('https://docs.test/'):
                    self.assertIn(link['url'], urls)
        self.assertTrue(all(task['collection_url'] in urls for task in tasks.values()))
        reverse_pages = prepare(list(reversed(source)), 2, 0, 'maximal')[0]
        self.assertEqual({p['url']: p for p in pages}, {p['url']: p for p in reverse_pages})

    def test_page_limit_before_creating_run(self):
        source = records(2)
        def many_pages(record):
            return [{'url': f'https://docs.test/p/{i}', 'title': 'Title', 'text': '', 'links': []}
                    for i in range(5001)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run'
            with patch('orchestrator.simulated_web.session.convert', side_effect=many_pages):
                with self.assertRaisesRegex(ValueError, '10000'):
                    run_session(path, source, None)
            self.assertFalse(path.exists())


class SessionTests(unittest.TestCase):
    def test_persistent_peer_notes_fresh_chats_and_export_coverage(self):
        source = records(2)
        starts = []
        reads = []
        schedule = make_schedule(['q0', 'q1'], 2, 0)['orders']
        def client(agent, messages, timeout):
            question = messages[1]['content'].splitlines()[0]
            qid = 'q' + question.split()[1].rstrip('?')
            tools = [m for m in messages if m['role'] == 'tool']
            if not tools:
                self.assertEqual(len(messages), 2)
                starts.append((agent, qid))
                self.assertIn('Document collection: https://docs.test/q/', messages[1]['content'])
                url = 'https://wiki.test/save?' + urlencode({'slug': qid, 'title': qid,
                    'text': f'Note from {agent} for {qid} https://docs.test/q/{hashlib.sha256(qid.encode()).hexdigest()}/p/0/0'}) if qid == schedule[agent][0] else 'https://wiki.test/page/' + qid
                return {'content': '', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': url}}}]}
            if qid != schedule[agent][0] and len(tools) == 1:
                note = json.loads(tools[-1]['content'])['text']
                self.assertNotIn(agent, note)
                reads.append(note)
                return {'content': '', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': note.split()[-1]}}}]}
            if qid != schedule[agent][0]:
                self.assertEqual(json.loads(tools[-1]['content'])['text'], 'Evidence ' + qid[1:])
            return ModelResponse({'content': 'Answer ' + qid}, {'eval_count': 5, 'prompt_eval_count': 10})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            outcomes = run_session(root, source, client, steps=3, history_mode='reset')
            self.assertEqual(len(starts), 4)
            self.assertEqual(len(reads), 2)
            self.assertTrue(all(r['status'] == 'complete' for r in outcomes))
            self.assertFalse(any(r['metrics']['token_reporting_complete'] for r in outcomes))
            self.assertTrue(all(r['metrics']['generated_tokens'] is None for r in outcomes))
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'complete')
            for agent in ('agent-1', 'agent-2'):
                predictions = [json.loads(line) for line in (root / 'predictions' / (agent + '.predictions.jsonl')).read_text().splitlines()]
                self.assertEqual([p['id'] for p in predictions], ['q0', 'q1'])
            for row in source:
                task = root / 'tasks' / hashlib.sha256(row['id'].encode()).hexdigest() / 'task.json'
                self.assertEqual(json.loads(task.read_text())['question'], row['question'])
            with sqlite3.connect(root / 'web' / 'wiki.sqlite3') as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM revisions').fetchone()[0], 2)
            self.assertEqual(outcomes[0]['wiki_boundary']['after'], outcomes[2]['wiki_boundary']['before'])

    def test_shard_export_and_full_dataset_provenance(self):
        source = records(20)
        raw = ''.join(json.dumps(record) + '\n' for record in source).encode()
        def client(agent, messages, timeout):
            self.assertEqual(agent, 'agent-1')
            return ModelResponse({'content': 'answer'},
                                 {'eval_count': 1, 'prompt_eval_count': 2})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            outcomes = run_session(root, source, client, agents=1, prompt_condition='neutral',
                                   dataset_bytes=raw, shard=2, shards=4)
            expected = make_schedule([row['id'] for row in source], 1, 0)['orders']['agent-1'][2::4]
            self.assertEqual([row['id'] for row in outcomes], expected)
            self.assertEqual((root / 'dataset.jsonl').read_bytes(), raw)
            config = json.loads((root / 'settings.json').read_text())
            self.assertEqual(config['dataset_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(config['schedule']['shard']['assigned_ids'], expected)
            self.assertEqual(json.loads((root / 'pages.json').read_text()),
                             prepare(source, 1, 0, 'neutral')[0])
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual((manifest['slots'], manifest['assignments'], manifest['completed_slots']),
                             (5, 5, 5))
            ids_in_dataset_order = [row['id'] for row in source if row['id'] in expected]
            for filename in ('gold.jsonl', 'agent-1.predictions.jsonl'):
                exported = [json.loads(line) for line in
                            (root / 'predictions' / filename).read_text().splitlines()]
                self.assertEqual([row['id'] for row in exported], ids_in_dataset_order)

    def test_selected_question_export_keeps_full_dataset(self):
        source = records(20)
        raw = ''.join(json.dumps(record) + '\n' for record in source).encode()
        calls = []
        def client(agent, messages, timeout):
            calls.append((agent, messages[1]['content']))
            return ModelResponse({'content': 'answer'},
                                 {'eval_count': 1, 'prompt_eval_count': 2})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            outcomes = run_session(root, source, client, agents=1, steps=36,
                                   prompt_condition='neutral', dataset_bytes=raw, question_id='q7')
            self.assertEqual([row['id'] for row in outcomes], ['q7'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 'agent-1')
            self.assertEqual((root / 'dataset.jsonl').read_bytes(), raw)
            config = json.loads((root / 'settings.json').read_text())
            self.assertEqual(config['dataset_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(config['steps'], 36)
            self.assertEqual(config['schedule']['selection']['question_id'], 'q7')
            self.assertEqual(json.loads((root / 'pages.json').read_text()),
                             prepare(source, 1, 0, 'neutral')[0])
            self.assertEqual(len(list((root / 'tasks').glob('*/task.json'))), 20)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual((manifest['slots'], manifest['assignments'], manifest['completed_slots']),
                             (1, 1, 1))
            for filename in ('gold.jsonl', 'agent-1.predictions.jsonl'):
                exported = [json.loads(line) for line in
                            (root / 'predictions' / filename).read_text().splitlines()]
                self.assertEqual([row['id'] for row in exported], ['q7'])

    def test_limits_and_transport_errors_have_blank_predictions(self):
        calls = {}
        def client(agent, messages, timeout):
            key = (agent, messages[1]['content'])
            calls[key] = calls.get(key, 0) + 1
            if agent == 'agent-1':
                return ModelResponse({'content': 'truncated'}, {'done_reason': 'length', 'eval_count': 2, 'prompt_eval_count': 3})
            if calls[key] == 1:
                return ModelResponse({'content': '', 'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://wiki.test/'}}}]},
                                     {'eval_count': 2, 'prompt_eval_count': 3})
            raise TimeoutError('mock failure')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            outcomes = run_session(root, records(2), client, steps=3, history_mode='reset')
            self.assertEqual({r['status'] for r in outcomes}, {'error', 'generation_limit'})
            for row in outcomes:
                self.assertEqual(row['metrics']['observed_generated_tokens'], 2)
                self.assertEqual(row['metrics']['token_reporting_complete'], row['agent'] == 'agent-1')
            for path in (root / 'predictions').glob('agent-*.jsonl'):
                self.assertTrue(all(json.loads(line)['predicted_answer'] == '' for line in path.read_text().splitlines()))

    def test_export_rejects_missing_or_duplicate_pairs(self):
        rows = [{'agent': 'agent-1', 'id': 'q0'}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'predictions'
            for malformed in ([], rows + rows):
                with self.assertRaises(ValueError):
                    export_session(records(1), malformed, 1, output)
                self.assertFalse(output.exists())

    def test_page_handles_reset_but_wiki_remains(self):
        with tempfile.TemporaryDirectory() as temporary:
            browser = Browser([], Path(temporary) / 'wiki.sqlite3')
            try:
                page = browser.call('agent-1', 'open', {'url': 'https://wiki.test/'})
                browser.reset_views()
                self.assertIn('error', browser.call('agent-1', 'click', {'page_id': page['page_id'], 'link_id': 1}))
                self.assertEqual(browser.checkpoint()['audit'], 2)
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
