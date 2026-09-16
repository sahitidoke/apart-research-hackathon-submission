"""Mock-only terminal recognition probe contracts; no model or service calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.context import ContextPolicy, ManagedContext, OMITTED_OBSERVATION
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.session import run_editability_probe, run_session
from orchestrator.simulated_web.test_session import records


class EditabilityProbeTests(unittest.TestCase):
    def test_terminal_order_separate_usage_and_failure_preserves_task(self):
        for fail in (False, True):
            seen = []
            def client(agent, messages, timeout, **options):
                seen.append(copy.deepcopy(messages))
                phase = messages[-1]['content']
                if 'memory-only follow-up' in phase:
                    self.assertEqual(options, {'num_predict': 1024, 'final_only': True})
                    self.assertNotIn('ANSWER_MARKER', str(messages))
                    self.assertIn('REFLECTION_MARKER', str(messages))
                    if fail:
                        raise TimeoutError('probe failed')
                    return ModelResponse({'content': 'Unsure.'}, {'eval_count': 3, 'prompt_eval_count': 20})
                text = 'ANSWER_MARKER' if 'ANSWER PHASE.' in phase else 'REFLECTION_MARKER'
                return ModelResponse({'content': text}, {'eval_count': 7, 'prompt_eval_count': 10})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                result = run_session(root, records(10), client, agents=1,
                    prompt_condition='neutral-reflection', protocol='answer-reflection', preparation_topic='Topic',
                    editable_source_title='Document 0',
                    editable_source_text_sha256=hashlib.sha256(b'Evidence 0').hexdigest(),
                    context_policy=ContextPolicy(), editability_probe=True)
                self.assertEqual(len(seen), 22)
                self.assertEqual(len(result), 10)
                self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'complete')
                usage = json.loads((root / 'session-usage.json').read_text())['agent-1']
                self.assertEqual(usage['metrics']['generated_tokens'], 147)
                probe = json.loads((root / 'editability-probe.json').read_text())[0]
                self.assertEqual(probe['status'], 'failed' if fail else 'complete')
                self.assertTrue((root / 'predictions').is_dir())

    def test_masked_snapshot_is_not_unmasked_or_changed(self):
        seen = []
        def client(agent, messages, timeout, **options):
            seen.append(messages)
            return ModelResponse({'content': 'No.'}, {'eval_count': 2, 'prompt_eval_count': 10})
        managed = ManagedContext(client, ContextPolicy(observation_window=1))
        history = [{'role': 'system', 'content': 'Task'}]
        for text in ('SECRET_OLD_OBSERVATION', 'recent'):
            history.extend([{'role': 'assistant', 'tool_calls': [{'function': {'name': 'open'}}]},
                            {'role': 'tool', 'content': text}])
        managed.model_messages(history)
        del history[-2:]  # Removing a later phase must not restore the earlier observation.
        before = copy.deepcopy(history)
        result = run_editability_probe('agent-1', managed, history, 60)
        self.assertEqual(result['status'], 'complete')
        self.assertNotIn('SECRET_OLD_OBSERVATION', str(seen))
        self.assertIn(OMITTED_OBSERVATION, str(seen))
        self.assertEqual(history, before)
        self.assertEqual(managed.compactions, [])

    def test_context_skip_and_tool_rejection(self):
        managed = ManagedContext(lambda *a, **k: self.fail('Backend called on overflow'), ContextPolicy())
        history = [{'role': 'system', 'content': 'x' * 300000}]
        self.assertEqual(run_editability_probe('agent-1', managed, history, 60)['status'], 'skipped_context_limit')
        managed.client = lambda *a, **k: ModelResponse(
            {'content': '', 'tool_calls': [{'function': {'name': 'open'}}]},
            {'eval_count': 3, 'prompt_eval_count': 10})
        result = run_editability_probe('agent-1', managed, [], 60)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('no actions executed', result['error'])

    def test_invalid_option_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with self.assertRaisesRegex(ValueError, 'requires answer-reflection'):
                run_session(root, records(10), lambda *a: None, editability_probe=True)
            self.assertFalse(root.exists())
