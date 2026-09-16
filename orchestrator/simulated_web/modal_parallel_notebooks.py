"""Two GPUs and independent owned inference workers, central notebook authority; validate by default."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import modal

from orchestrator.simulated_web.async_notebooks import AsyncBudget
from orchestrator.simulated_web.parallel_notebooks import build_parallel_settings, run_parallel_notebooks
from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, download_command
from orchestrator.simulated_web.hf_transport import OwnedVllm
from orchestrator.simulated_web.modal_hf_paired_views import image, models, runs, PROFILE, MODEL_VOLUME, RUN_VOLUME, JOB_SECONDS, resources as single_resources
from orchestrator.simulated_web.modal_token_pair import validate_run_id
from orchestrator.simulated_web.timed_policy import TimedPolicy

app = modal.App('germanwiki-hf-fp8-parallel-notebooks')


def resources():
    return {**single_resources(), 'gpu': 'H100', 'gpu_count': 2, 'gpu_request': 'H100:2',
            'inference_workers': 2, 'notebook_authorities': 1, 'maximum_gpu_hours': 12,
            'parallel_runtime_verified': False, 'independent_process_group_cancellation': True}


def validate_inputs(run_id, inputs, budget_values):
    validate_run_id(run_id)
    if any(token not in run_id.lower() for token in ('async', 'fp8', 'parallel')):
        raise ValueError('Fresh run ID must contain async, fp8 and parallel')
    budget = AsyncBudget(**budget_values)
    settings, _, _ = build_parallel_settings(**inputs, budget=budget)
    return settings, budget


@app.function(image=image, gpu='H100:2', cpu=8, memory=131072,
              volumes={'/hf': models, '/runs': runs}, timeout=JOB_SECONDS,
              max_containers=1, retries=0)
def execute(run_id, inputs, budget_values, download_model=False):
    started = time.monotonic()
    settings, budget = validate_inputs(run_id, inputs, budget_values)
    destination, setup = Path('/runs') / run_id, Path('/runs') / (run_id + '-setup')
    if destination.exists() or setup.exists():
        raise ValueError('Fresh run and setup IDs required')
    snapshot = Path('/hf/models--Qwen--Qwen3.8-27B-FP8/snapshots') / REVISION
    if download_model:
        subprocess.run(download_command('/hf'), check=True, timeout=3600)
        models.commit()
    clients = {agent: OwnedVllm(TimedPolicy(**settings['policy']), snapshot, setup / (agent + '-vllm.log'),
                               port=8000 + index, cuda_visible_devices=str(index), seed=0)
               for index, agent in enumerate(('agent-1', 'agent-2'))}
    metadata = {agent: client.inspect_model() for agent, client in clients.items()}
    if time.monotonic() >= started + JOB_SECONDS - 120:
        raise TimeoutError('Setup exhausted job allowance')
    provenance = {'backend': 'Modal', 'models': metadata, 'resources': resources(),
                  'run_volume': RUN_VOLUME, 'model_volume': MODEL_VOLUME,
                  'download_requested': download_model, 'requested_profile': PROFILE,
                  'scheduling': settings['scheduling'], 'simultaneous_inference': True,
                  'workers': {a: {'cuda_visible_devices': c.cuda_visible_devices, 'loopback_port': c.port,
                                  'seed': c.seed} for a, c in clients.items()}}
    setup.mkdir()
    try:
        (setup / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return run_parallel_notebooks(destination, clients, **inputs, budget=budget, provenance=provenance,
            checkpoint_callback=lambda path: runs.commit(), job_deadline=started + JOB_SECONDS - 120)
    except BaseException as error:
        (setup / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        try:
            cleanup_errors = []
            for client in clients.values():
                try:
                    try:
                        client.cancel()
                    finally:
                        client.close()
                except BaseException as error:
                    cleanup_errors.append(f'{type(error).__name__}: {error}')
            if cleanup_errors:
                (setup / 'cleanup-failure.json').write_text(json.dumps(cleanup_errors) + '\n')
                raise RuntimeError('Could not confirm all worker cleanup: ' + '; '.join(cleanup_errors))
        finally:
            (setup / 'transport-events.json').write_text(json.dumps({a: c.events for a, c in clients.items()}, indent=2) + '\n')
            runs.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-id', required=True)
    for field in ('dataset', 'topic-file', 'editable-sources', 'question-ids', 'access-manifest', 'visible-labels'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--source-access-mode', choices=('hard', 'discovery_only'), default='discovery_only')
    for name, default in asdict(AsyncBudget()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--mandatory-post-answer', action='store_true')
    parser.add_argument('--shared-request-log', action='store_true')
    args = parser.parse_args(argv)
    budget_values = {name: getattr(args, name) for name in asdict(AsyncBudget())}
    AsyncBudget(**budget_values).validate()
    inputs = {'records': [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
              'topic': args.topic_file.read_text().strip(), 'source_access_mode': args.source_access_mode,
              'mandatory_post_answer': args.mandatory_post_answer, 'shared_request_log': args.shared_request_log}
    for name, file in [('selectors', args.editable_sources), ('question_ids', args.question_ids),
                       ('access_manifest', args.access_manifest), ('visible_labels', args.visible_labels)]:
        inputs[name] = json.loads(file.read_text())
    settings, _ = validate_inputs(args.run_id, inputs, budget_values)
    if not args.launch:
        print(json.dumps({'status': 'validated_no_cloud_actions', 'run_id': args.run_id, 'settings': settings,
                          'resources': resources(), 'remote_cache_checked': False}, indent=2))
        return
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Set MODAL_PROFILE=' + PROFILE)
    config = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):
        raise ValueError('Configure required Modal profile')
    with modal.enable_output(), app.run():
        print(json.dumps(execute.remote(args.run_id, inputs, budget_values, args.download_model), indent=2))


if __name__ == '__main__':
    main()
