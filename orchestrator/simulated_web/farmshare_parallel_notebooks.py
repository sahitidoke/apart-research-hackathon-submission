"""FarmShare pinned FP8 pilot. Default validates inputs; download and execution are explicit."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

from orchestrator.simulated_web.async_notebooks import AsyncBudget
from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, VLLM_VERSION, download_command, validate_snapshot
from orchestrator.simulated_web.hf_transport import OwnedVllm
from orchestrator.simulated_web.parallel_notebooks import build_parallel_settings, run_parallel_notebooks
from orchestrator.simulated_web.timed_policy import TimedPolicy

FIELDS = ('dataset', 'topic-file', 'editable-sources', 'question-ids', 'access-manifest', 'visible-labels')


def allocated_devices(value):
    devices = value.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or any(not d.isdecimal() for d in devices):
        raise ValueError('Exactly two distinct numeric Slurm CUDA_VISIBLE_DEVICES required; UUID allocation is unsupported by this transport')
    return devices


def fresh_paths(root, run_id):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id) is None:
        raise ValueError('Invalid run ID')
    if any(word not in run_id.lower() for word in ('async', 'fp8', 'parallel')):
        raise ValueError('Run ID must contain async, fp8 and parallel')
    paths = root / run_id, root / (run_id + '-setup')
    if any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError('Fresh run and setup directories required')
    return paths


def runtime_preflight():
    if sys.platform != 'linux' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Execution requires Linux within an explicit Slurm allocation')
    devices = allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    for executable in ('vllm', 'nvidia-smi'):
        if shutil.which(executable) is None:
            raise ValueError('Missing executable: ' + executable)
    for package, required in (('vllm', VLLM_VERSION), ('transformers', '5.8.0')):
        if importlib.metadata.version(package) != required:
            raise ValueError(f'{package} must be exactly {required}')
    # Query only the allocated numeric devices; no model/CUDA initialization.
    for device in devices:
        name = subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        if 'L40S' not in name:
            raise ValueError('This pilot requires allocated L40S GPUs, found: ' + name)
    return devices


def free_ports():
    handles = [socket.socket(), socket.socket()]
    try:
        for handle in handles:
            handle.bind(('127.0.0.1', 0))
        return [handle.getsockname()[1] for handle in handles]
    finally:
        for handle in handles:
            handle.close()



def progress_summary(state, results):
    summary = {a: {k: v.get(k) for k in ('question_index', 'status', 'phase', 'generated_tokens', 'model_requests')}
               for a, v in state.items()}
    for result in results:
        note = result.get('mandatory_note')
        if note is not None and result['agent'] in summary:
            summary[result['agent']]['latest_mandatory_note'] = {
                'question_index': result['question_index'],
                **{key: note.get(key) for key in ('status', 'attempts', 'generated_tokens', 'persistence_verified')}}
    return summary


def execute(args, inputs, budget, settings):
    started = time.monotonic()
    destination, setup = fresh_paths(args.run_root, args.run_id)
    devices = runtime_preflight()
    snapshot = args.cache_dir / 'models--Qwen--Qwen3.8-27B-FP8' / 'snapshots' / REVISION
    print('Checking all pinned model artifacts before creating run output...', flush=True)
    metadata = validate_snapshot(snapshot)
    clients = {agent: OwnedVllm(TimedPolicy(**settings['policy']), snapshot, setup / (agent + '-vllm.log'),
               port=port, cuda_visible_devices=device, seed=0)
               for agent, port, device in zip(('agent-1', 'agent-2'), free_ports(), devices)}
    for client in clients.values():
        client.metadata = metadata
    provenance = {'backend': 'FarmShare Slurm', 'job_id': os.environ['SLURM_JOB_ID'],
                  'hostname': socket.gethostname(), 'models': metadata,
                  'workers': {a: {'cuda_visible_devices': c.cuda_visible_devices, 'port': c.port} for a, c in clients.items()},
                  'wall_seconds': args.wall_seconds, 'gpu_fit_verified': False}
    if time.monotonic() >= started + args.wall_seconds - 120:
        raise TimeoutError('Model validation exhausted job allowance')
    setup.mkdir(parents=True)
    previous = None

    def progress(path):
        nonlocal previous
        state = json.loads((path / 'state.json').read_text())
        summary = progress_summary(state, json.loads((path / 'results.json').read_text()))
        line = json.dumps(summary, sort_keys=True)
        if line != previous:
            print('progress ' + line, flush=True)
            previous = line

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; preserving partial results and stopping workers')

    old_handler = signal.signal(signal.SIGTERM, stop)
    try:
        (setup / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'Starting two independent FP8 workers; logs: {setup}', flush=True)
        return run_parallel_notebooks(destination, clients, **inputs, budget=budget, provenance=provenance,
                                      checkpoint_callback=progress, job_deadline=started + args.wall_seconds - 120)
    except BaseException as error:
        (setup / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        errors = []
        for client in clients.values():
            try:
                try:
                    client.cancel()
                finally:
                    client.close()
            except BaseException as error:
                errors.append(f'{type(error).__name__}: {error}')
        (setup / 'transport-events.json').write_text(json.dumps({a: c.events for a, c in clients.items()}, indent=2) + '\n')
        if errors:
            (setup / 'cleanup-failure.json').write_text(json.dumps(errors) + '\n')
            raise RuntimeError('Worker cleanup unconfirmed: ' + '; '.join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    for field in FIELDS:
        parser.add_argument('--' + field, type=Path, required=True)
    for name, default in asdict(AsyncBudget()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--wall-seconds', type=int, default=21600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--download-model', action='store_true', help='Only fetch and hash locked artifacts; never run inference')
    mode.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if not 300 <= args.wall_seconds <= 21600:
        parser.error('wall-seconds must be within 300..21600')
    fresh_paths(args.run_root, args.run_id)
    budget = AsyncBudget(**{name: getattr(args, name) for name in asdict(AsyncBudget())})
    inputs = {'records': [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
              'topic': args.topic_file.read_text().strip(), 'source_access_mode': 'discovery_only',
              'mandatory_post_answer': True, 'shared_request_log': True}
    for key, filename in (('selectors', args.editable_sources), ('question_ids', args.question_ids),
                          ('access_manifest', args.access_manifest), ('visible_labels', args.visible_labels)):
        inputs[key] = json.loads(filename.read_text())
    settings, _, _ = build_parallel_settings(**inputs, budget=budget)
    if settings['question_count'] != 3 or budget.generated_tokens != 4096:
        raise ValueError('FarmShare pilot requires three shared questions and 4096 QA tokens each')
    if args.download_model:
        if shutil.which('hf') is None:
            raise ValueError('Install a reviewed HF CLI environment before downloading')
        subprocess.run(download_command(args.cache_dir), check=True, timeout=3600)
        print(json.dumps(validate_snapshot(args.cache_dir / 'models--Qwen--Qwen3.8-27B-FP8' / 'snapshots' / REVISION), indent=2))
    elif args.execute:
        print(json.dumps(execute(args, inputs, budget, settings), indent=2))
    else:
        print(json.dumps({'status': 'validated_no_remote_actions', 'run_id': args.run_id, 'settings': settings,
                          'model': MODEL, 'revision': REVISION, 'runtime_and_cache_checked': False,
                          'slurm': {'partition': 'gpu', 'constraint': 'GPU_SKU:L40S', 'gres': 'gpu:2',
                                    'nodes': 1, 'cpus': 32, 'memory_mib': 128000, 'wall_seconds': args.wall_seconds}}, indent=2))


if __name__ == '__main__':
    main()
