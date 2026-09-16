"""Fresh matched 9d synchronized exchanges on official Hugging Face FP8 weights.

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

from orchestrator.simulated_web.answer_support import PROTOCOL

from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, KV_CACHE_DTYPE, download_command
from orchestrator.simulated_web.hf_transport import OwnedVllm
from orchestrator.simulated_web.modal_token_pair import validate_run_id
from orchestrator.simulated_web.timed_policy import TimedPolicy

from orchestrator.simulated_web.modal_hf_paired_views import image, models, runs, PROFILE, MODEL_VOLUME, RUN_VOLUME, JOB_SECONDS, IMAGE
from orchestrator.simulated_web.modal_synchronized_exchange import resources as exchange_resources
from orchestrator.simulated_web.synchronized_exchange import build_settings, run_exchange, SHORT_NOTE_POLICY, DISCOVERY_ACCESS_POLICY

GPU = 'L40S'
app = modal.App('germanwiki-hf-fp8-synchronized-exchange')


def resources(count,notebook_context_policy=None,notebook_quota_policy=None):
    return {**exchange_resources(count, SHORT_NOTE_POLICY),
            **({'notebook_quota_policy':notebook_quota_policy,'browser_allowance_scope':'non-notebook calls only'} if notebook_quota_policy else {}),
            **({'maximum_verifier_generated_tokens':2*count*1024,'maximum_combined_generated_tokens':22528*count,'memory_payload_native_token_cap':0} if notebook_context_policy else {}), 'gpu': GPU,
            'host_memory_mib': 65536, 'context_length': 65536,
            'model': MODEL, 'revision': REVISION, 'backend_image': IMAGE,
            'model_volume': MODEL_VOLUME, 'kv_cache_dtype': KV_CACHE_DTYPE,
            'runtime_memory_fit_verified': False, 'runtime_transport_verified': False,
            'precision_comparison_isolation': False,
            'sampling': {'seed': 0, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20},
            'chat_template': {'enable_thinking': 'research/answer only',
                              'reasoning_effort': 'medium', 'preserve_thinking': True}}


def validate_fresh(run_id, inputs):
    validate_run_id(run_id)
    if 'fp8' not in run_id.lower():
        raise ValueError('Fresh HF pilot run ID must explicitly contain fp8')
    if inputs.get('inference_profile') != 'hf-fp8-v1':
        raise ValueError('HF FP8 inference profile required')
    settings = build_settings(**inputs)[0]
    if settings['question_count'] != 6 or settings.get('note_retry_policy') != SHORT_NOTE_POLICY:
        raise ValueError('Matched FP8 rerun requires six rounds and fixed repaired note policy')
    return settings


@app.function(image=image, gpu=GPU, cpu=4, memory=65536,
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
    provenance = {'backend': 'Modal', 'model': metadata, 'resources': resources(settings['question_count'],settings.get('notebook_context_policy'),settings.get('notebook_quota_policy')),
                  'requested_profile': PROFILE, 'run_volume': RUN_VOLUME, 'model_volume': MODEL_VOLUME,
                  'download_requested': download_model,
                  'migration': 'New backend, HF template, FP8 weights and BF16 KV; not an isolated precision ablation'}
    setup.mkdir()
    try:
        (setup / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return run_exchange(destination, client, **inputs, provenance=provenance,
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
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--question-pairing-policy',choices=['same-question-v1'])
    parser.add_argument('--notebook-quota-policy',choices=['notebook-exempt-v1'])
    parser.add_argument('--notebook-context-policy',choices=[PROTOCOL])
    parser.add_argument('--source-access-policy', choices=[DISCOVERY_ACCESS_POLICY])
    parser.add_argument('--no-peer-information', action='store_true')
    parser.add_argument('--short-note-retry', action='store_true', required=True)
    args = parser.parse_args(argv)
    if (args.detach or args.download_model) and not args.launch:
        raise ValueError('--detach and --download-model require --launch')
    inputs = {'records': [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
              'topic': args.topic_file.read_text().strip(), 'inference_profile': 'hf-fp8-v1',
              'note_retry_policy': SHORT_NOTE_POLICY, 'no_peer_information': args.no_peer_information,
              'source_access_policy': args.source_access_policy,'notebook_context_policy':args.notebook_context_policy}
    if args.question_pairing_policy:inputs['question_pairing_policy']=args.question_pairing_policy
    if args.notebook_quota_policy:inputs['notebook_quota_policy']=args.notebook_quota_policy
    for field, path in [('selectors', args.editable_sources), ('question_ids', args.question_ids),
                        ('access_manifest', args.access_manifest), ('visible_labels', args.visible_labels),
                        ('round_leaders', args.round_leaders)]:
        inputs[field] = json.loads(path.read_text())
    settings = validate_fresh(args.run_id, inputs)
    if not args.launch:
        print(json.dumps({'status': 'validated_no_cloud_actions', 'run_id': args.run_id,
                          'settings': settings, 'resources': resources(settings['question_count'],settings.get('notebook_context_policy'),settings.get('notebook_quota_policy')), 'remote_cache_checked': False}, indent=2))
        return
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Set MODAL_PROFILE=' + PROFILE)
    config = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):
        raise ValueError('Configure required Modal profile')
    with modal.enable_output(), app.run(detach=args.detach):
        print(json.dumps(execute.remote(args.run_id, inputs, args.download_model), indent=2))


if __name__ == '__main__':
    main()
