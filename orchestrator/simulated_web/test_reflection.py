"""Unexecuted mock contracts for reflection, context failure safety and ordinary-source edits."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.context import ContextError, ContextPolicy, ManagedContext
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.session import (REFLECTION_PROTOCOL, REFLECTION_NO_EFFICIENCY_CONDITION,
    phase_prompt, prepare, protocol_settings, run_reflected_assignment, run_session)
from orchestrator.simulated_web.test_session import records


class ReflectionTests(unittest.TestCase):
    def test_answer_visible_to_reflection_then_removed_including_final_round(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            phase = messages[-1]['content']
            text = ('ANSWER_MARKER' if 'ANSWER PHASE.' in phase else
                    'REFLECTION_MARKER' if 'REFLECTION PHASE.' in phase else 'INITIAL_MARKER')
            return ModelResponse({'content': text, 'thinking': text + '_THINKING'},
                                 {'eval_count': 7, 'prompt_eval_count': 10})
        source = records(10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            results = run_session(root, source, client, agents=1,
                prompt_condition='neutral-reflection', protocol=REFLECTION_PROTOCOL, preparation_topic='Topic',
                editable_source_title='Document 0',
                editable_source_text_sha256=hashlib.sha256(b'Evidence 0').hexdigest(),
                context_policy=ContextPolicy())
            self.assertEqual(len(seen), 21)
            self.assertEqual(len(results), 10)
            self.assertNotIn('Question 0?', str(seen[0]))
            for slot in range(10):
                answer, reflection = seen[1 + 2 * slot:3 + 2 * slot]
                self.assertNotIn('ANSWER_MARKER', str(answer))
                self.assertIn('ANSWER_MARKER_THINKING', str(reflection))
                self.assertNotIn('Question', reflection[-1]['content'])
                self.assertEqual([p['phase'] for p in results[slot]['phases']], ['answer', 'reflection'])
            usage = json.loads((root / 'session-usage.json').read_text())['agent-1']
            self.assertEqual(usage['answer_observed_generated_tokens'], 70)
            self.assertEqual(usage['metrics']['generated_tokens'], 147)
            self.assertTrue(all(row['status'] == 'complete' for row in results))

    def test_calibration_condition_changes_only_score_prompt_and_metadata(self):
        source = records(20)
        baseline = prepare(source, 1, 0, 'neutral-reflection',
                           protocol=REFLECTION_PROTOCOL, preparation_topic='Topic')
        calibration = prepare(source, 1, 0, REFLECTION_NO_EFFICIENCY_CONDITION,
                              protocol=REFLECTION_PROTOCOL, preparation_topic='Topic')
        self.assertEqual(baseline[:3], calibration[:3])
        score_start = baseline[3].index('Your individual score per question is ')
        score_end = baseline[3].index('Initial research and reflections remain')
        self.assertEqual(calibration[3], baseline[3][:score_start] +
                         'No correctness feedback is provided. ' + baseline[3][score_end:])
        for phase, slot, task in [('preparation', -1, None), ('answer', 0, source[0]),
                                   ('reflection', 0, None)]:
            self.assertEqual(phase_prompt('Topic', slot, phase, task, 'neutral-reflection'),
                             phase_prompt('Topic', slot, phase, task, REFLECTION_NO_EFFICIENCY_CONDITION))
        settings = [protocol_settings(REFLECTION_PROTOCOL, 'Topic', source, 'persistent', condition,
                    0, 1, None, (None, None, None))
                    for condition in ('neutral-reflection', REFLECTION_NO_EFFICIENCY_CONDITION)]
        self.assertEqual(settings[1]['score_generated_token_scope'], 'none')
        for key in settings[0]:
            if not key.startswith('score_'):
                self.assertEqual(settings[0][key], settings[1][key])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'invalid'
            for protocol, extra in [('standard', {}), (REFLECTION_PROTOCOL, {'total_token_budget': 9999})]:
                with self.assertRaises(ValueError):
                    run_session(root, source, lambda *a, **kw: self.fail('Unexpected model call'),
                                prompt_condition=REFLECTION_NO_EFFICIENCY_CONDITION,
                                protocol=protocol, preparation_topic='Topic', **extra)
                self.assertFalse(root.exists())

    def test_calibration_runs_twenty_rounds_with_matching_retention_and_accounting(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(copy.deepcopy(messages))
            phase = messages[-1]['content']
            text = ('ANSWER_MARKER' if 'ANSWER PHASE.' in phase else
                    'REFLECTION_MARKER' if 'REFLECTION PHASE.' in phase else 'INITIAL_MARKER')
            self.assertNotIn('score', messages[0]['content'])
            self.assertNotIn('efficien', messages[0]['content'])
            return ModelResponse({'content': text}, {'eval_count': 7, 'prompt_eval_count': 10})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            result = run_session(root, records(20), client, agents=1,
                prompt_condition=REFLECTION_NO_EFFICIENCY_CONDITION, protocol=REFLECTION_PROTOCOL,
                preparation_topic='Topic', editable_source_title='Document 0',
                editable_source_text_sha256=hashlib.sha256(b'Evidence 0').hexdigest(),
                context_policy=ContextPolicy())
            self.assertEqual(len(result), 20)
            self.assertEqual(len(seen), 41)
            for slot in range(20):
                answer, reflection = seen[1 + 2 * slot:3 + 2 * slot]
                self.assertNotIn('ANSWER_MARKER', str(answer))
                self.assertIn('ANSWER_MARKER', str(reflection))
                self.assertIn('REFLECTION PHASE.', reflection[-1]['content'])
            usage = json.loads((root / 'session-usage.json').read_text())['agent-1']
            self.assertEqual(usage['score_generated_token_scope'], 'none')
            self.assertEqual(usage['answer_observed_generated_tokens'], 140)
            self.assertEqual(usage['metrics']['generated_tokens'], 287)
            self.assertTrue(all(row['status'] == 'complete' for row in result))

    def test_invalid_source_creates_no_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with self.assertRaisesRegex(ValueError, 'Editable source not found'):
                run_session(root, records(10), lambda *a, **kw: self.fail('Unexpected model call'), agents=1,
                    prompt_condition='neutral-reflection', protocol=REFLECTION_PROTOCOL,
                    preparation_topic='Topic', editable_source_title='Missing',
                    editable_source_text_sha256='0' * 64, context_policy=ContextPolicy())
            self.assertFalse(root.exists())

    def test_shared_source_identity_updates_aliases_and_search(self):
        corpus = [{'url': f'https://docs.test/q/{i}/p/0', 'title': 'Source', 'text': 'Original source'}
                  for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            browser = Browser(corpus, Path(directory) / 'wiki.sqlite3',
                              editable_source=('Source', hashlib.sha256(b'Original source').hexdigest()))
            try:
                opened = browser.call('agent-1', 'open', {'url': corpus[0]['url']})
                self.assertEqual(opened['links'][0]['label'], 'Edit')
                revision = browser.source_editor_revisions[browser.source_identity]
                self.assertNotIn(revision, json.dumps(opened))
                search = browser.call('agent-1', 'search', {'query': 'Source'})
                self.assertNotIn(revision, json.dumps(search))
                editor = browser.call('agent-1', 'click', {'page_id': opened['page_id'], 'link_id': 1})
                self.assertIn(f'Editor revision: {revision}.', editor['text'])
                self.assertIn('To save this source, open', editor['text'])
                saved = browser.call('agent-1', 'open', {'url': 'https://docs.test/source/save?' +
                    urlencode({'source': browser.source_identity, 'title': 'Source', 'text': 'Shared new evidence'})})
                self.assertIn('saved', saved)
                other = browser.call('agent-2', 'open', {'url': corpus[1]['url']})
                self.assertEqual(other['text'], 'Shared new evidence')
                hits = browser.call('agent-2', 'search', {'query': 'Shared new evidence'})['results']
                self.assertEqual({hit['url'] for hit in hits}, {p['url'] for p in corpus})
                self.assertEqual(browser.checkpoint()['revisions'], 1)
            finally:
                browser.close()

    def test_summary_failure_preserves_original_history_and_raw_response(self):
        original = [{'role': 'system', 'content': 'Sequence task'},
                    {'role': 'user', 'content': 'Research'},
                    {'role': 'assistant', 'content': 'Evidence ' * 2000}]
        for message, metadata in [({'content': ''}, {'eval_count': 2, 'prompt_eval_count': 5000}),
                                  ({'content': 'Partial'}, {'done_reason': 'length', 'eval_count': 2, 'prompt_eval_count': 5000}),
                                  ({'content': 'Summary'}, {'eval_count': 100, 'prompt_eval_count': 65500})]:
            managed = ManagedContext(lambda *a, **kw: ModelResponse(message, metadata), ContextPolicy())
            managed.pending_compaction = True
            history = copy.deepcopy(original)
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / 'context.jsonl'
                with self.assertRaises(ContextError):
                    managed.compact('agent-1', history, 60, log)
                self.assertEqual(history, original)
                self.assertIn('compaction_response', log.read_text())

    def test_protected_overflow_does_not_request_or_mutate_history(self):
        managed = ManagedContext(lambda *a, **kw: self.fail('Overflow reached model'), ContextPolicy())
        managed.allow_compaction = False
        history = [{'role': 'system', 'content': 'x' * 250000}]
        original = copy.deepcopy(history)
        with self.assertRaisesRegex(ContextError, 'exceeds context'):
            managed('agent-1', history, 60, num_predict=1000)
        self.assertEqual(history, original)

    def test_answer_context_failure_skips_reflection_and_keeps_answer_block(self):
        def fail(*args, **kwargs):
            raise ContextError('Backend prompt accounting failed')
        managed = ManagedContext(fail, ContextPolicy())
        history = [{'role': 'system', 'content': 'Sequence task'},
                   {'role': 'user', 'content': 'Initial research'},
                   {'role': 'assistant', 'content': 'Initial evidence'}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'logs').mkdir()
            browser = Browser([], folder / 'wiki.sqlite3')
            try:
                result = run_reflected_assignment('agent-1', browser, {'question': 'New question?'},
                    managed, folder, 4, 60, 'Sequence task', history, 'Topic', 0)
                self.assertTrue(result['reflection_failed'])
                self.assertEqual(len(result['phases']), 1)
                self.assertIn('New question?', history[-1]['content'])
                self.assertFalse((folder / 'reflection').exists())
            finally:
                browser.close()

    def test_initial_compaction_keeps_phase_instruction_and_tool_exchange(self):
        def summary(*args, **kwargs):
            self.assertTrue(kwargs['summary_only'])
            return ModelResponse({'content': 'Retained evidence', 'thinking': 'Private summary reasoning'},
                                 {'eval_count': 100, 'prompt_eval_count': 5000})
        managed = ManagedContext(summary, ContextPolicy())
        managed.pending_compaction = True
        managed.phase_instruction = 'INITIAL RESEARCH. Research the topic.'
        history = [{'role': 'system', 'content': 'Sequence task'},
                   {'role': 'user', 'content': managed.phase_instruction},
                   {'role': 'assistant', 'content': 'Evidence ' * 2000,
                    'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]},
                   {'role': 'tool', 'tool_name': 'open', 'content': 'Source'}]
        with tempfile.TemporaryDirectory() as directory:
            managed.compact('agent-1', history, 60, Path(directory) / 'context.jsonl')
        self.assertEqual(history[-1]['content'], managed.phase_instruction)
        self.assertIn('Retained evidence', str(history))
        self.assertNotIn('Private summary reasoning', str(history))
        self.assertEqual(managed.compactions[0]['eval_count'], 100)

    def test_render_mode_changes_reset_native_append_comparison(self):
        native_counts = iter((1000, 400, 300, 100))
        def client(*args, **kwargs):
            return ModelResponse({'content': 'Done'},
                                 {'eval_count': 7, 'prompt_eval_count': next(native_counts)})
        managed = ManagedContext(client, ContextPolicy())
        managed.allow_compaction = False
        history = [{'role': 'system', 'content': 'Task'}, {'role': 'user', 'content': 'Answer'}]
        for mode, final_only in [('tools', False), ('final_only', True), ('tools', False)]:
            response = managed('agent-1', history, 60, num_predict=256, final_only=final_only)
            self.assertEqual(response.metadata['context_accounting']['render_mode'], mode)
            history.append({'role': 'user', 'content': 'Next boundary'})
        # With an unchanged renderer, a falling native count remains suspicious.
        with self.assertRaisesRegex(ContextError, 'decreased on appended history'):
            managed('agent-1', history, 60, num_predict=256, final_only=False)

    def test_phase_compaction_rebases_answer_removal_and_preserves_previous_reflection(self):
        seen_summaries = []
        phase_calls = {}
        current_slot = 0
        managed = None
        def client(agent, messages, timeout, **options):
            if options.get('summary_only'):
                seen_summaries.append(copy.deepcopy(messages))
                self.assertNotIn('ANSWER_MARKER', str(messages))
                return ModelResponse({'content': 'Summary of older findings'},
                                     {'eval_count': 20, 'prompt_eval_count': 500})
            phase = managed.phase
            key = current_slot, phase
            step = phase_calls.get(key, 0)
            phase_calls[key] = step + 1
            tools = 2 if phase == 'answer' else 3
            # Repeated threshold checks include a prefix already reduced to its summary.
            if (phase == 'reflection' and step in (0, 1)) or (current_slot == 2 and phase == 'answer' and step == 0):
                managed.pending_compaction = True
            marker = f'ANSWER_MARKER_{current_slot}' if phase == 'answer' else f'REFLECTION_MARKER_{current_slot}'
            message = {'content': marker + f'_step{step}'}
            if step < tools:
                message['tool_calls'] = [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]
            return ModelResponse(message, {'eval_count': 7, 'prompt_eval_count': 500})
        managed = ManagedContext(client, ContextPolicy())
        history = [{'role': 'system', 'content': 'Sequence task'},
                   {'role': 'user', 'content': 'Initial research'},
                   {'role': 'assistant', 'content': 'Old initial finding'},
                   {'role': 'user', 'content': 'More initial research'},
                   {'role': 'assistant', 'content': 'Other old finding'}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            browser = Browser([{'url': 'https://docs.test/', 'title': 'Source', 'text': 'Source result'}], root / 'wiki.sqlite3')
            try:
                previous_reflection = []
                for current_slot in range(3):
                    folder = root / f'slot{current_slot}'
                    (folder / 'logs').mkdir(parents=True)
                    result = run_reflected_assignment('agent-1', browser, {'question': f'Question {current_slot}?'},
                        managed, folder, 8, 60, 'Sequence task', history, 'Topic', current_slot)
                    self.assertEqual(result['status'], 'complete')
                    self.assertFalse(result['reflection_failed'])
                    self.assertNotIn('ANSWER_MARKER', str(history))
                    # The last reflection is preserved verbatim throughout the next cycle.
                    for message in previous_reflection:
                        self.assertTrue(any(retained is message for retained in history))
                    start = managed.anchor_index(history, managed.previous_reflection_anchor)
                    previous_reflection = history[start:]
                    self.assertEqual(sum(m['role'] == 'tool' for m in previous_reflection), 3)
                    events = [json.loads(line) for line in (folder / 'logs/agent-1.jsonl').read_text().splitlines()]
                    boundary = events[-1]
                    self.assertTrue(boundary['removed'])
                    self.assertEqual(boundary['answer_messages'], 6)
                self.assertEqual(len(seen_summaries), 2)
                self.assertIn('REFLECTION_MARKER_0', str(seen_summaries[-1]))
                self.assertNotIn('REFLECTION_MARKER_1', str(seen_summaries[-1]))
                self.assertNotIn('Old initial finding', str(history))
                self.assertEqual(len(managed.compactions), 2)
            finally:
                browser.close()

    def test_protected_phase_overflow_fails_without_summarizing_or_losing_history(self):
        managed = ManagedContext(lambda *a, **kw: self.fail('Protected overflow reached backend'), ContextPolicy())
        history = [{'role': 'system', 'content': 'Task'}]
        managed.begin_phase('answer', history)
        history.append({'role': 'user', 'content': 'Question?'})
        history.extend([{'role': 'assistant', 'content': 'Large active answer',
                         'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]},
                        {'role': 'tool', 'tool_name': 'open', 'content': 'x' * 250000}])
        original = copy.deepcopy(history)
        with tempfile.TemporaryDirectory() as directory:
            managed.log_path = Path(directory) / 'context.jsonl'
            with self.assertRaisesRegex(ContextError, 'Protected previous reflection'):
                managed('agent-1', history, 60, num_predict=1000)
            self.assertIn('compaction_deferred', managed.log_path.read_text())
        self.assertEqual(history, original)


class SourceRevisionLabelTests(unittest.TestCase):
    def test_five_unique_alias_stable_editor_only_labels_survive_save_and_reset(self):
        corpus = [{'url': f'https://docs.test/q/{alias}/p/{i}', 'title': f'Source{i}', 'text': f'Body{i}'}
                  for i in range(5) for alias in range(2)]
        selectors = [(f'Source{i}', hashlib.sha256(f'Body{i}'.encode()).hexdigest()) for i in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            browser = Browser(corpus, Path(folder) / 'wiki.sqlite3', editable_sources=selectors)
            try:
                labels = browser.source_editor_revisions.copy()
                self.assertEqual(len(set(labels.values())), 5)
                for identity, label in labels.items():
                    self.assertNotIn(label.replace('R-', '').replace('-', '').lower(), identity)
                    for alias in browser.source_urls[identity]:
                        opened = browser.call('agent-1', 'open', {'url': alias})
                        self.assertNotIn(label, json.dumps(opened))
                        editor = browser.call('agent-1', 'click', {'page_id': opened['page_id'], 'link_id': 1})
                        self.assertIn(f'Editor revision: {label}.', editor['text'])
                    save_url = 'https://docs.test/source/save?' + urlencode({'source': identity, 'title': 'Changed', 'text': 'Changed body'})
                    self.assertNotIn(label, save_url)
                    saved = browser.call('agent-1', 'open', {'url': save_url})
                    self.assertNotIn(label, json.dumps(saved))
                    # Private conversation reset leaves Browser state and canonical identity intact.
                    reopened = browser.call('agent-2', 'open', {'url': browser.source_urls[identity][1]})
                    self.assertNotIn(label, json.dumps(reopened))
                    editor = browser.call('agent-2', 'click', {'page_id': reopened['page_id'], 'link_id': 1})
                    self.assertIn(f'Editor revision: {label}.', editor['text'])
                search = browser.call('agent-2', 'search', {'query': 'Changed'})
                for label in labels.values():
                    self.assertNotIn(label, json.dumps(search))
                self.assertEqual(labels, browser.source_editor_revisions)
            finally:
                browser.close()

    def test_label_collision_fails_before_creating_database(self):
        corpus = [{'url': f'https://docs.test/p/{i}', 'title': f'Source{i}', 'text': f'Body{i}'} for i in range(2)]
        selectors = [(p['title'], hashlib.sha256(p['text'].encode()).hexdigest()) for p in corpus]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'wiki.sqlite3'
            with patch('orchestrator.simulated_web.browser.source_editor_revision', return_value='R-COLLISION'):
                with self.assertRaisesRegex(ValueError, 'collision'):
                    Browser(corpus, path, editable_sources=selectors)
            self.assertFalse(path.exists())
