"""Opt-in future paired-view pilot on official Hugging Face FP8 weights.

Default is local validation only. --launch spends compute; --download-model
additionally permits downloading this pinned checkpoint on the remote worker.
No legacy checkpoint migration or resume is supported by this entry point.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import modal

from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, VLLM_VERSION, KV_CACHE_DTYPE, download_command
from orchestrator.simulated_web.hf_transport import OwnedVllm
from orchestrator.simulated_web.modal_token_pair import validate_run_id
from orchestrator.simulated_web.paired_notebook_views import build_paired_settings, run_paired_views
from orchestrator.simulated_web.timed_policy import TimedPolicy

PROFILE = 'research-profile'
MODEL_VOLUME = 'germanwiki-qwen38-hf-fp8-models'
RUN_VOLUME = 'germanwiki-timed-runs'
JOB_SECONDS = 21600
# Docker release tags include v; installed package versions do not.
IMAGE = 'vllm/vllm-openai:v' + VLLM_VERSION
app = modal.App('germanwiki-hf-fp8-paired-views')
models = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=True)
# The official final image installs vLLM into system Python 3.12, not /opt/venv.
# Modal's image builder/runtime requires the unversioned python command.
PYTHON_SETUP = ['RUN test -x /usr/bin/python3 && ln -s /usr/bin/python3 /usr/local/bin/python']
VERIFY_ENVIRONMENT = (
    "python -c \"import importlib.metadata as m, os, shutil, sys; "
    "assert sys.version_info[:2] == (3, 12); "
    "assert os.path.samefile(sys.executable, '/usr/bin/python3'); "
    "assert m.version('vllm') == '" + VLLM_VERSION + "'; "
    "assert m.version('transformers') == '5.8.0'; "
    "assert shutil.which('vllm'), 'vLLM CLI missing from PATH'; assert shutil.which('hf'), 'HF CLI missing from PATH'\""
)


def build_image():
    return (modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_SETUP).entrypoint([])
            .pip_install('transformers==5.8.0')
            .run_commands(VERIFY_ENVIRONMENT)
            .env({'PYTHONPATH': '/root/project'})
            .add_local_dir(Path(__file__).parent, '/root/project/orchestrator/simulated_web',
                           ignore=['__pycache__', '*.pyc']))


image = build_image()


def resources():
    return {'gpu': 'L40S', 'gpu_count': 1, 'context_length': 65536,
            'model': MODEL, 'revision': REVISION, 'backend_image': IMAGE,
            'kv_cache_dtype': KV_CACHE_DTYPE, 'hard_job_seconds': JOB_SECONDS,
            'runtime_memory_fit_verified': False, 'runtime_transport_verified': False,
            'precision_comparison_isolation': False}


def validate_fresh(run_id, inputs):
    validate_run_id(run_id)
    if 'fp8' not in run_id.lower():
        raise ValueError('Fresh HF pilot run ID must explicitly contain fp8')
    if inputs.get('inference_profile') != 'hf-fp8-v1':
        raise ValueError('HF FP8 inference profile required')
    return build_paired_settings(**inputs)[0]


@app.function(image=image, gpu='L40S', cpu=4, memory=65536,
              volumes={'/hf': models, '/runs': runs}, timeout=JOB_SECONDS,
              max_containers=1, retries=0)
def execute(run_id, inputs, download_model=False):
    started = time.monotonic()
    settings = validate_fresh(run_id, inputs)
    destination, setup = Path('/runs') / run_id, Path('/runs') / (run_id + '-setup')
    if destination.exists() or setup.exists():
        raise ValueError('Fresh run/setup IDs required; existing artifacts preserved')
    snapshot = Path('/hf/models--Qwen--Qwen3.8-27B-FP8/snapshots') / REVISION
    if download_model:
        # Explicit flag, immutable revision; no dependency or model substitution.
        subprocess.run(download_command('/hf'),
                       check=True, timeout=3600)
        models.commit()
    client = OwnedVllm(TimedPolicy(**settings['policy']), snapshot, setup / 'vllm.log')
    metadata = client.inspect_model()  # Cached artifacts checked before output mkdir/GPU startup.
    provenance = {'backend': 'Modal', 'model': metadata, 'resources': resources(),
                  'requested_profile': PROFILE, 'run_volume': RUN_VOLUME, 'model_volume': MODEL_VOLUME,
                  'download_requested': download_model,
                  'migration': 'New backend, HF template, FP8 weights and BF16 KV; not an isolated precision ablation'}
    setup.mkdir()
    try:
        (setup / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return run_paired_views(destination, client, **inputs, provenance=provenance,
            checkpoint_callback=lambda path: runs.commit(), job_deadline=started + JOB_SECONDS - 120)
    except BaseException as error:
        (setup / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        try:
            client.close()
        finally:
            (setup / 'transport-events.json').write_text(json.dumps(client.events, indent=2) + '\n')
            runs.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-id', required=True)
    for name in ('dataset', 'topic-file', 'editable-sources', 'question-ids', 'access-manifest', 'visible-labels', 'round-leaders'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args(argv)
    inputs = {'records': [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
              'topic': args.topic_file.read_text().strip(), 'inference_profile': 'hf-fp8-v1'}
    for field, path in [('selectors', args.editable_sources), ('question_ids', args.question_ids),
                        ('access_manifest', args.access_manifest), ('visible_labels', args.visible_labels),
                        ('round_leaders', args.round_leaders)]:
        inputs[field] = json.loads(path.read_text())
    settings = validate_fresh(args.run_id, inputs)
    if not args.launch:
        print(json.dumps({'status': 'validated_no_cloud_actions', 'run_id': args.run_id,
                          'settings': settings, 'resources': resources(), 'remote_cache_checked': False}, indent=2))
        return
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Set MODAL_PROFILE=' + PROFILE)
    config = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):
        raise ValueError('Configure required Modal profile')
    with modal.enable_output(), app.run():
        print(json.dumps(execute.remote(args.run_id, inputs, args.download_model), indent=2))


if __name__ == '__main__':
    main()
