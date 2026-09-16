"""Fresh concurrent 10d reference on one80GBH100 with explicit answer finalization on official FP8.

Default is local validation only. --launch spends compute; --download-model
additionally permits downloading this pinned checkpoint on the remote worker.
No legacy checkpoint migration or resume is supported by this entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import modal

from orchestrator.simulated_web.complementary_partition import SCHEMA, VERSION
from orchestrator.simulated_web.answer_support import PROTOCOL

from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, KV_CACHE_DTYPE, download_command
from orchestrator.simulated_web.concurrent_hf_transport import ConcurrentOwnedVllm
from orchestrator.simulated_web.modal_token_pair import validate_run_id
from orchestrator.simulated_web.timed_policy import TimedPolicy

from orchestrator.simulated_web.modal_hf_paired_views import image, models, runs, PROFILE, MODEL_VOLUME, RUN_VOLUME, JOB_SECONDS, IMAGE
from orchestrator.simulated_web.synchronized_exchange import SHORT_NOTE_POLICY, DISCOVERY_ACCESS_POLICY
from orchestrator.simulated_web.concurrent_research_reflection import build_settings, run_concurrent_research_reflection

GPU = 'H100'
app = modal.App('germanwiki-hf-fp8-concurrent-10d')


def resources(settings):
    return {'gpu':GPU,'gpu_count':1,'gpu_memory_gb':80,'model_processes':1,'weight_copies':1,'max_num_seqs':2,'concurrent_agent_requests':2,'host_memory_mib':65536,'context_length':65536,'hard_job_seconds':JOB_SECONDS,
            'coordinator_readiness':settings['coordinator_readiness'], 'agent_log_policy':settings['agent_log_policy'],
            'phase_records':settings['maximum_phases'],'normal_model_phases':4+4*settings['question_count'],
            'maximum_generated_tokens':settings['maximum_generated_tokens'],
            'maximum_verifier_generated_tokens':settings['maximum_verifier_generated_tokens'],
            'maximum_combined_generated_tokens':settings['maximum_combined_generated_tokens'],
            'reflection_repairs':'at most512+1024tokens per reflection publication, already inside solver ceiling',
            'initial_research_tokens':8192,'answer_tokens':2048,'reflection_tokens':4096,'answer_seconds':180,'answer_final_reserve_tokens':512,'answer_finalization_policy':'explicit-schema-final-only-v1',
            'source_call_caps':{'initial_research':16,'answer':4,'reflection':8},
            'notebook_quota_policy':settings['notebook_quota_policy'],'memory_policy':settings['memory_policy'],
            'model':MODEL,'revision':REVISION,'backend_image':IMAGE,'kv_cache_dtype':KV_CACHE_DTYPE,
            'sampling':{'seed':0,'temperature':1.0,'top_p':0.95,'top_k':20},
            'runtime_memory_fit_verified':False,'runtime_transport_verified':False,'completion_guaranteed':False}


def validate_fresh(run_id, inputs, question_count=2):
    if type(question_count) is not int or question_count != 2:
        raise ValueError("Question count must be2(pilot)")
    validate_run_id(run_id)
    if 'fp8' not in run_id.lower() or not run_id.endswith('-concurrent-10d'):
        raise ValueError('Fresh run ID must contain fp8 and end in -concurrent-10d')
    if inputs.get('inference_profile') != 'hf-fp8-v1':
        raise ValueError('HF FP8 inference profile required')
    manifest = inputs.get("access_manifest", {})
    if manifest.get("schema") != SCHEMA or manifest.get("partition_audit", {}).get("schema") != VERSION:
        raise ValueError("10d requires the audited complementary 11b partition")
    if inputs.get("source_access_policy") != DISCOVERY_ACCESS_POLICY:
        raise ValueError("10d requires discovery-only source access")
    if inputs.get("answer_final_reserve_tokens") != 512:
        raise ValueError("10d fixes-v2 requires answer-only final reserve512")
    if inputs.get("agent_log_policy") != "no-agent-history-v1":
        raise ValueError("10d requires no agent-visible history")
    settings = build_settings(**inputs)[0]
    if manifest["partition_audit"]["selected_questions"] != settings["question_ids"]["agent-1"]:
        raise ValueError("Selected questions must match the complementary partition audit")
    if settings['question_count'] != question_count or settings.get('note_retry_policy') != SHORT_NOTE_POLICY:
        raise ValueError('Question count must match the explicitly selected count and fixed repaired note policy')
    return settings


@app.function(image=image, gpu=GPU, cpu=4, memory=65536,
              volumes={'/hf': models, '/runs': runs}, timeout=JOB_SECONDS,
              max_containers=1, retries=0)
def execute(run_id, inputs, download_model=False, question_count=2):
    started = time.monotonic()
    settings = validate_fresh(run_id, inputs, question_count)
    destination, setup = Path('/runs') / run_id, Path('/runs') / (run_id + '-setup')
    if destination.exists() or setup.exists():
        raise ValueError('Fresh run/setup IDs required; existing artifacts preserved')
    snapshot = Path('/hf/models--Qwen--Qwen3.8-27B-FP8/snapshots') / REVISION
    if download_model:
        # Explicit flag, immutable revision; no dependency or model substitution.
        subprocess.run(download_command('/hf'),
                       check=True, timeout=3600)
        models.commit()
    client = ConcurrentOwnedVllm(TimedPolicy(**settings['policy']), snapshot, setup / 'vllm.log')
    metadata = client.inspect_model()  # Cached artifacts checked before output mkdir/GPU startup.
    provenance = {'backend': 'Modal', 'model': metadata, 'resources': resources(settings),
                  'requested_profile': PROFILE, 'run_volume': RUN_VOLUME, 'model_volume': MODEL_VOLUME,
                  'download_requested': download_model,
                  'experiment_variant':'10d-no-agent-history-v1',
                  'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'partition_version':inputs['access_manifest']['partition_audit']['schema'],
                  'partition_dataset_sha256':inputs['access_manifest']['dataset_sha256'],
                  'migration': 'Removes all agent-visible request history while preserving host audit/notebooks; reuses concurrent10b startup-v2 schedule with worker-owned SQLite cleanup and answer-only512 reserve inside2048 with exact11b complementary partition and two MLB questions; explicit finalization remains within budget; not a single-variable comparison with10b'}
    if question_count==2:provenance['run_scope']={'purpose':'two-question development pilot','question_count':2}
    setup.mkdir()
    try:
        (setup / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return run_concurrent_research_reflection(destination, client, **inputs, provenance=provenance,
            checkpoint_callback=lambda path: runs.commit(), job_deadline=started + JOB_SECONDS - 120)
    except BaseException as error:
        (setup / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        try:
            client.close()
        finally:
            (setup / 'transport-events.json').write_text(json.dumps({'owner':client.events,'agents':{a:e.events for a,e in client.endpoints.items()}}, indent=2) + '\n')
            runs.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-count',type=int,choices=[2],default=2)
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
              'agent_log_policy':'no-agent-history-v1', 'answer_final_reserve_tokens':512, 'note_retry_policy': SHORT_NOTE_POLICY, 'no_peer_information': args.no_peer_information,
              'source_access_policy': args.source_access_policy,'notebook_context_policy':args.notebook_context_policy}
    if args.question_pairing_policy:inputs['question_pairing_policy']=args.question_pairing_policy
    if args.notebook_quota_policy:inputs['notebook_quota_policy']=args.notebook_quota_policy
    for field, path in [('selectors', args.editable_sources), ('question_ids', args.question_ids),
                        ('access_manifest', args.access_manifest), ('visible_labels', args.visible_labels),
                        ('round_leaders', args.round_leaders)]:
        inputs[field] = json.loads(path.read_text())
    settings = validate_fresh(args.run_id, inputs, args.question_count)
    if not args.launch:
        print(json.dumps({'status': 'validated_no_cloud_actions', 'run_id': args.run_id,
                          'settings': settings, 'resources': resources(settings), 'remote_cache_checked': False,
                          'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiment_variant':'10d-no-agent-history-v1',
                          **({'run_scope':{'purpose':'two-question development pilot','question_count':2}} if args.question_count==2 else {})}, indent=2))
        return
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Set MODAL_PROFILE=' + PROFILE)
    config = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):
        raise ValueError('Configure required Modal profile')
    with modal.enable_output(), app.run(detach=args.detach):
        print(json.dumps(execute.remote(args.run_id, inputs, args.download_model, **({"question_count":2} if args.question_count==2 else {})), indent=2))


if __name__ == '__main__':
    main()
