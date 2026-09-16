"""Read-only launcher validation; SDK handles are mocked and never hydrated."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.timed import TimedPolicy
from orchestrator.simulated_web import test_timed_resume

try:
    from orchestrator.simulated_web import modal_timed
except ModuleNotFoundError as error:
    if error.name != 'modal':
        raise
    modal_timed = None


@unittest.skipIf(modal_timed is None, 'Optional Modal SDK not installed')
class ModalValidationTests(unittest.TestCase):
    def inputs(self, folder):
        root = Path(folder)
        source = records(10)
        (root / 'data.jsonl').write_text('\n'.join(json.dumps(r) for r in source))
        (root / 'topic.txt').write_text('Topic')
        (root / 'selectors.json').write_text(json.dumps([
            {'title': r['paragraphs'][0]['title'], 'text_sha256': hashlib.sha256(r['paragraphs'][0]['paragraph_text'].encode()).hexdigest()}
            for r in source[:5]]))
        return ['--dataset', str(root / 'data.jsonl'), '--topic-file', str(root / 'topic.txt'),
                '--editable-sources', str(root / 'selectors.json'), '--run-id', 'mock',
                '--expected-model-digest', 'a' * 64, '--validate-only']

    def test_resume_selector_validation_is_explicitly_local_only(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'modal.toml'
            config.write_text('[research-profile]\n')
            args = ['--resume-from', 'parent/checkpoints/questions-010', '--run-id', 'continued', '--validate-only']
            output = io.StringIO()
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(config)}), patch.object(modal_timed.app, 'run') as run, redirect_stdout(output):
                modal_timed.main(args)
                run.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertEqual(result['status'], 'validated_resume_selector_only_no_cloud_actions')
            self.assertFalse(result['checkpoint_contents_validated'])
            for extra in (['--question-count', '20'], ['--dataset', 'unread'], ['--download-model']):
                with self.assertRaisesRegex(ValueError, 'does not accept'):
                    modal_timed.main(args + extra)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                modal_timed.main(args + ['--reflection-s=120'])
            for value in ('../parent/checkpoints/questions-010', '/parent/checkpoints/questions-010', 'parent/other/questions-010'):
                with self.assertRaises(ValueError):
                    modal_timed.validate_resume_selector(value, 'continued')
            with self.assertRaisesRegex(ValueError, 'fresh run ID'):
                modal_timed.validate_resume_selector('parent/checkpoints/questions-010', 'parent')

    def test_remote_resume_validates_snapshot_before_setup_and_wires_saved_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = test_timed_resume.ResumeTests().make_parent(root)
            def local_path(value):
                return root if value == '/runs' else Path(value)
            client = MagicMock()
            client.inspect_model.return_value = test_timed_resume.MODEL
            client.events = []
            with patch.object(modal_timed, 'Path', side_effect=local_path), patch.object(modal_timed, 'OwnedOllama', return_value=client) as owner, patch.object(modal_timed, 'resume_timed_session') as resume, patch.object(modal_timed.runs, 'commit'), redirect_stdout(io.StringIO()):
                with self.assertWarnsRegex(UserWarning, 'executing locally'):
                    result = modal_timed.execute.local(None, None, None, {}, 'continued', None, False, 'parent/checkpoints/questions-010')
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(owner.call_args.kwargs['expected_digest'], test_timed_resume.MODEL['digest'])
                self.assertEqual(resume.call_args.args[:2], (root / 'continued', checkpoint))
                self.assertIn('checkpoint_callback', resume.call_args.kwargs)
            (checkpoint / 'history.json').write_text('corrupted')
            with patch.object(modal_timed, 'Path', side_effect=local_path), patch.object(modal_timed, 'OwnedOllama') as owner, redirect_stdout(io.StringIO()):
                with self.assertWarnsRegex(UserWarning, 'executing locally'), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    modal_timed.execute.local(None, None, None, {}, 'bad', None, False, 'parent/checkpoints/questions-010')
                owner.assert_not_called()
                self.assertFalse((root / 'bad-setup').exists())
                self.assertFalse((root / 'bad').exists())

    def test_remote_complete_resume_does_not_start_model_or_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test_timed_resume.ResumeTests().make_parent(root, count=10)
            def local_path(value):
                return root if value == '/runs' else Path(value)
            with patch.object(modal_timed, 'Path', side_effect=local_path), patch.object(modal_timed, 'OwnedOllama') as owner, redirect_stdout(io.StringIO()):
                with self.assertWarnsRegex(UserWarning, 'executing locally'):
                    result = modal_timed.execute.local(None, None, None, {}, 'continued', None, False, 'parent/checkpoints/questions-010')
                self.assertEqual(result['status'], 'already_complete')
                owner.assert_not_called()
                self.assertFalse((root / 'continued-setup').exists())
                self.assertFalse((root / 'continued').exists())

    def test_checkpoint_commits_volume_immediately_and_retains_final_commit(self):
        client = MagicMock()
        client.inspect_model.return_value = {'name': 'mock-model', 'digest': 'a' * 64}
        client.events = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def local_path(value):
                return root if value == '/runs' else Path(value)
            def session(destination, *args, checkpoint_callback, **kwargs):
                checkpoint = destination / 'checkpoints/questions-010'
                checkpoint.mkdir(parents=True)
                checkpoint_callback(checkpoint)
                commit.assert_called_once()  # Persist before the session continues/returns.
            with patch.object(modal_timed, 'Path', side_effect=local_path), patch.object(modal_timed, 'OwnedOllama', return_value=client), patch.object(modal_timed, 'run_timed_session', side_effect=session), patch.object(modal_timed.runs, 'commit') as commit, redirect_stdout(io.StringIO()):
                with self.assertWarnsRegex(UserWarning, 'executing locally'):
                    result = modal_timed.execute.local(records(10), 'topic', [], {}, 'mock-run', None, False)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(commit.call_count, 2)
                client.close.assert_called_once()

    def test_validate_only_does_not_create_cloud_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.inputs(folder)
            config = Path(folder) / 'modal.toml'
            config.write_text('[research-profile]\n')
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(config)}), patch.object(modal_timed.app, 'run') as run, patch.object(modal_timed, 'validate_remote_budget', wraps=modal_timed.validate_remote_budget) as budget, redirect_stdout(io.StringIO()):
                modal_timed.main(args)
                policy = budget.call_args.args[0]
                self.assertEqual(policy.context_length, 65536)
                self.assertEqual(policy.context_length * policy.compaction_trigger_fraction, 49152)
                self.assertEqual(policy.compaction_retained_tokens, 16384)
                run.assert_not_called()

    def test_missing_profile_and_invalid_budget_fail_before_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.inputs(folder)
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(Path(folder) / 'absent')}), patch.object(modal_timed.app, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'not configured'):
                    modal_timed.main(args)
                run.assert_not_called()
        with self.assertRaises(ValueError):
            modal_timed.validate_remote_budget(TimedPolicy(question_count=100, answer_seconds=60), True)

    def test_download_announces_progress_and_preserves_log(self):
        process = MagicMock()
        process.args = ['ollama', 'pull', 'model']
        process.wait.side_effect = [subprocess.TimeoutExpired(process.args, 15), 0]
        process.poll.return_value = 0
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, patch.object(modal_timed.subprocess, 'Popen', return_value=process), redirect_stdout(output):
            path = Path(folder) / 'pull.log'
            modal_timed.pull_model(MagicMock(port=11434), path)
            self.assertTrue(path.is_file())
        self.assertIn('pull running', output.getvalue())
        self.assertIn('pull complete', output.getvalue())
        process.kill.assert_not_called()

    def test_download_failure_is_reported_without_losing_log(self):
        process = MagicMock()
        process.args = ['ollama', 'pull', 'model']
        process.wait.return_value = 1
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as folder, patch.object(modal_timed.subprocess, 'Popen', return_value=process), redirect_stdout(io.StringIO()):
            path = Path(folder) / 'pull.log'
            with self.assertRaises(subprocess.CalledProcessError):
                modal_timed.pull_model(MagicMock(port=11434), path)
            self.assertTrue(path.is_file())

    def test_initial_readiness_is_counted_once_in_job_envelope(self):
        #20 questions:300 initial +40*120 transitions +90+20*40 phases +30 setup=6020.
        policy = TimedPolicy(question_count=20, compaction_enabled=False)
        with patch.object(modal_timed, 'JOB_TIMEOUT_SECONDS', 6320):
            modal_timed.validate_remote_budget(policy, False)
        with patch.object(modal_timed, 'JOB_TIMEOUT_SECONDS', 6319):
            with self.assertRaises(ValueError):
                modal_timed.validate_remote_budget(policy, False)

    def test_compaction_added_once_per_between_question_boundary(self):
        policy = TimedPolicy(question_count=20)
        with patch.object(modal_timed, 'JOB_TIMEOUT_SECONDS', 9740):
            modal_timed.validate_remote_budget(policy, False)
        with patch.object(modal_timed, 'JOB_TIMEOUT_SECONDS', 9739):
            with self.assertRaises(ValueError):
                modal_timed.validate_remote_budget(policy, False)
