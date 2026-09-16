"""No scheduler, model, download or CUDA operations in these regressions."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.simulated_web import farmshare_parallel_notebooks as farm


def test_device_mapping_retains_slurm_namespace():
    assert farm.allocated_devices('2,3') == ['2', '3']
    for value in ('', '0', '0,0', '0,1,2', 'GPU-uuid,GPU-other', '0, 1'):
        with pytest.raises(ValueError):
            farm.allocated_devices(value)


def test_fresh_paths_rejects_existing_setup_and_dangling_symlink(tmp_path):
    root = tmp_path / 'no-created-root'
    with pytest.raises(ValueError):
        farm.fresh_paths(root, '../bad')
    assert not root.exists()
    name = 'async-fp8-parallel-test'
    (tmp_path / (name + '-setup')).symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError):
        farm.fresh_paths(tmp_path, name)


def test_missing_runtime_creates_no_artifacts(tmp_path, monkeypatch):
    args = SimpleNamespace(run_root=tmp_path / 'new', run_id='async-fp8-parallel-test')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='Slurm'):
        farm.execute(args, {}, None, {})
    assert not args.run_root.exists()


def test_runtime_checks_exact_versions_and_allocated_devices(monkeypatch):
    monkeypatch.setattr(farm.sys, 'platform', 'linux')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    monkeypatch.setattr(farm.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(farm.importlib.metadata, 'version', lambda name: {'vllm': '0.24.0', 'transformers': '5.8.0'}[name])
    seen = []
    def query(command, **kwargs):
        seen.append(command[2])
        return 'NVIDIA L40S\n'
    monkeypatch.setattr(farm.subprocess, 'check_output', query)
    assert farm.runtime_preflight() == ['2', '3']
    assert seen == ['2', '3']
    monkeypatch.setattr(farm.importlib.metadata, 'version', lambda name: 'wrong')
    with pytest.raises(ValueError, match='exactly'):
        farm.runtime_preflight()


def test_execution_failure_preserves_diagnostics_and_cancels_both(tmp_path, monkeypatch):
    args = SimpleNamespace(run_root=tmp_path / 'runs', run_id='async-fp8-parallel-test', cache_dir=tmp_path, wall_seconds=300)
    monkeypatch.setattr(farm, 'runtime_preflight', lambda: ['2', '3'])
    monkeypatch.setattr(farm, 'validate_snapshot', lambda path: {'verified': True})
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(farm, 'TimedPolicy', lambda **kwargs: object())
    monkeypatch.setattr(farm, 'free_ports', lambda: [8000, 8001])
    clients = []
    class Client:
        def __init__(self, *args, **kwargs):
            self.cuda_visible_devices = kwargs['cuda_visible_devices']
            self.port = kwargs['port']
            self.events = []
            self.cancelled = self.closed = False
            clients.append(self)
        def cancel(self):
            self.cancelled = True
        def close(self):
            self.closed = True
    monkeypatch.setattr(farm, 'OwnedVllm', Client)
    def fail(*args, **kwargs):
        raise RuntimeError('injected runner failure')
    monkeypatch.setattr(farm, 'run_parallel_notebooks', fail)
    with pytest.raises(RuntimeError, match='injected runner failure'):
        farm.execute(args, {}, None, {'policy': {}})
    assert len(clients) == 2 and all(c.cancelled and c.closed for c in clients)
    setup = args.run_root / (args.run_id + '-setup')
    assert 'injected runner failure' in (setup / 'failure.json').read_text()
    assert (setup / 'transport-events.json').is_file()


def test_progress_exposes_note_phase_attempts_tokens_and_persistence():
    state = {'agent-1': {'question_index': 0, 'phase': 'note', 'status': 'active'}}
    results = [{'agent': 'agent-1', 'question_index': 0, 'mandatory_note': {
        'status': 'pending', 'attempts': 1, 'generated_tokens': 123, 'persistence_verified': False}}]
    summary = farm.progress_summary(state, results)
    assert summary['agent-1']['phase'] == 'note'
    assert summary['agent-1']['latest_mandatory_note'] == {
        'question_index': 0, 'status': 'pending', 'attempts': 1, 'generated_tokens': 123, 'persistence_verified': False}
    results[0]['mandatory_note'].update(status='published', attempts=2, generated_tokens=180, persistence_verified=True)
    assert farm.progress_summary(state, results) != summary
