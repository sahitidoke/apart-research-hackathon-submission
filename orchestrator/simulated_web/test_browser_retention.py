"""Selective retention contracts using mock clients and temporary browser state."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web import modal_timed, modal_token_pair
from orchestrator.simulated_web.browser_retention import OMITTED, retain_at_question_boundary
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.timed import run_timed_session, resume_timed_session, session_prompt
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import pair_policy, run_token_pair, load_pair_checkpoint
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint


def read(path):
    return json.loads(path.read_text())


class RetentionTests(unittest.TestCase):
    def test_mask_content_only_preserves_pairing_and_nonbrowser(self):
        history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'},
                   {'role': 'assistant', 'thinking': 'private reasoning', 'content': 'answer', 'tool_calls': [
                       {'id': 'one', 'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}]},
                   {'role': 'tool', 'tool_call_id': 'one', 'tool_name': 'open', 'content': 'raw page'},
                   {'role': 'tool', 'tool_name': 'private_scratchpad_update', 'content': 'private note'}]
        original = deepcopy(history)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            retain_at_question_boundary(history, TimedPolicy(), root, 2, 'agent-1', 'q1')
            self.assertEqual(history, original)
            self.assertEqual(list(root.iterdir()), [])
            policy = TimedPolicy(browser_retention='question_boundary')
            retain_at_question_boundary(history, policy, root, 2, 'agent-1', 'q1')
            expected = deepcopy(original)
            expected[3]['content'] = OMITTED
            self.assertEqual(history, expected)
            self.assertEqual(read(root / 'history-before-retention-02-agent-1.json'), original)
            metrics = read(root / 'browser-retention-02-agent-1.json')
            self.assertEqual(metrics['removed_message_contents'], 1)
            self.assertEqual(metrics['removed_content_characters'], 8)
            retain_at_question_boundary(history, policy, root, 4, 'agent-1', 'q2')
            self.assertEqual(read(root / 'browser-retention-04-agent-1.json')['removed_message_contents'], 0)

    def test_pair_boundary_privacy_final_checkpoint_and_resume(self):
        data, selectors = inputs()
        client = Client()
        policy = pair_policy(browser_retention='question_boundary')
        compacted = []
        def compact(client, history, policy, run_dir, index):
            self.assertTrue(all(m['content'] == OMITTED for m in history if m['role'] == 'tool'))
            compacted.append(index)
        with tempfile.TemporaryDirectory() as temp, patch('orchestrator.simulated_web.token_pair.compact_between_questions', side_effect=compact):
            root = Path(temp)
            run_token_pair(root / 'parent', data, 'topic', client, policy, selectors)
            self.assertEqual(len(compacted), 18)
            for agent in ('agent-1', 'agent-2'):
                calls = [h for owner, h in client.calls if owner == agent and h[-1]['role'] == 'user']
                # Both agents retain prep when entering Q1 and both prep+answer through reflection.
                self.assertNotEqual([m for m in calls[1] if m['role'] == 'tool'][0]['content'], OMITTED)
                self.assertTrue(all(m['content'] != OMITTED for m in calls[2] if m['role'] == 'tool'))
                self.assertTrue(all(m['content'] == OMITTED for m in calls[3] if m['role'] == 'tool'))
                other = 'agent-2' if agent == 'agent-1' else 'agent-1'
                self.assertTrue(all(other + ' private answer' not in json.dumps(h) for h in calls))
            checkpoint_path = root / 'parent/checkpoints/rounds-005'
            before = {p.name: p.read_bytes() for p in checkpoint_path.iterdir()}
            loaded = load_pair_checkpoint(checkpoint_path)
            self.assertEqual(loaded.data['settings.json']['policy']['browser_retention'], 'question_boundary')
            loaded.close()
            child = Client()
            run_token_pair(root / 'child', None, None, child, resume_from=checkpoint_path)
            self.assertEqual(read(root / 'child/settings.json')['policy']['browser_retention'], 'question_boundary')
            self.assertTrue(all(m['content'] == OMITTED for m in child.calls[0][1] if m['role'] == 'tool'))
            self.assertEqual(before, {p.name: p.read_bytes() for p in checkpoint_path.iterdir()})
            final = read(root / 'parent/checkpoints/rounds-010/histories.json')
            self.assertTrue(all(m['content'] == OMITTED for h in final.values() for m in h if m['role'] == 'tool'))
            self.assertEqual(len(list((root / 'parent').glob('browser-retention-*.json'))), 20)
            # The raw host event stream still contains actual browser observations.
            self.assertIn('https://docs.test/', (root / 'parent/phase-00.jsonl').read_text())

    def test_timed_checkpoint_resume_and_historical_default(self):
        data, _ = inputs()
        policy = TimedPolicy(browser_retention='question_boundary', compaction_enabled=False)
        client = Client()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_timed_session(root / 'parent', data, 'topic', client, policy,
                              provenance={'model': client.inspect_model()})
            checkpoint_path = root / 'parent/checkpoints/questions-005'
            loaded = load_checkpoint(checkpoint_path)
            loaded.close()
            resume_timed_session(root / 'child', checkpoint_path, Client())
            self.assertEqual(read(root / 'child/settings.json')['policy']['browser_retention'], 'question_boundary')
            final = read(root / 'parent/checkpoints/questions-010/history.json')
            self.assertTrue(all(m['content'] == OMITTED for m in final if m['role'] == 'tool'))
            # Historical schema/policy contains no retention field and retains full by default.
            settings = read(checkpoint_path / 'settings.json')
            settings['policy'].pop('browser_retention')
            (checkpoint_path / 'settings.json').write_text(json.dumps(settings))
            manifest = read(checkpoint_path / 'checkpoint.json')
            manifest['files_sha256']['settings.json'] = hashlib.sha256((checkpoint_path / 'settings.json').read_bytes()).hexdigest()
            (checkpoint_path / 'checkpoint.json').write_text(json.dumps(manifest))
            loaded = load_checkpoint(checkpoint_path)
            self.assertEqual(TimedPolicy(**loaded.data['settings.json']['policy']).browser_retention, 'full')
            loaded.close()

    def test_resume_cli_rejects_policy_override_before_cloud_access(self):
        for launcher, checkpoint in [(modal_timed, 'parent/checkpoints/questions-005'),
                                     (modal_token_pair, 'parent/checkpoints/rounds-005')]:
            with self.subTest(launcher=launcher.__name__), self.assertRaisesRegex(ValueError, 'overrides'):
                launcher.main(['--run-id', 'child', '--resume-from', checkpoint,
                               '--browser-retention', 'question_boundary', '--validate-only'])

    def test_validation_and_truthful_notice(self):
        with self.assertRaisesRegex(ValueError, 'browser retention'):
            TimedPolicy(browser_retention='typo')
        policy = TimedPolicy(browser_retention='question_boundary', compaction_enabled=False)
        notice = session_prompt(policy)
        self.assertNotIn('Your full conversation is retained', notice)
        self.assertIn('After each reflection', notice)
        baseline = asdict(TimedPolicy())
        selective = asdict(TimedPolicy(browser_retention='question_boundary'))
        baseline.pop('browser_retention'); selective.pop('browser_retention')
        self.assertEqual(baseline, selective)
