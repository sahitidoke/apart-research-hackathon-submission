"""Explicit 8d feedback child; read-only local validation is the default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import tomllib

import modal

from orchestrator.simulated_web.modal_token_pair import GPU, JOB_TIMEOUT_SECONDS, FINALIZATION_MARGIN_SECONDS, validate_run_id
from orchestrator.simulated_web.modal_timed import image, models, runs, PROFILE, RUN_VOLUME, MODEL_VOLUME, IMAGE
from orchestrator.simulated_web.pair_extension import extension_settings, load_extension_checkpoint, run_extension
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama, MODEL
from orchestrator.simulated_web.token_pair import DIGEST, load_pair_checkpoint

app=modal.App('germanwiki-pair-feedback-extension')


def validate_selector(selector,run_id,resume):
    pattern=r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}/checkpoints/rounds-00[0-7]' if resume else r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}/checkpoints/rounds-003'
    if not isinstance(selector,str) or re.fullmatch(pattern,selector) is None or selector.split('/')[0]==run_id:
        raise ValueError('Select a different parent run and valid completed checkpoint boundary')


def inspect_checkpoint(path,resume):
    if resume:
        parent,data=load_extension_checkpoint(path)
        try:return data['settings.json'],data['state.json']['completed_rounds']
        finally:parent.close()
    parent=load_pair_checkpoint(path)
    try:return extension_settings(parent),0
    finally:parent.close()


def bounds(settings,completed):
    remaining=7-completed
    policy=TimedPolicy(**settings['policy'])
    return {'gpu':GPU,'gpu_count':1,'new_rounds_remaining':remaining,'minimum_new_phases':remaining*4,
            'maximum_new_phases':remaining*8,'maximum_new_generated_tokens':remaining*8*2048,
            'maximum_phase_safety_seconds':remaining*4*(policy.preparation_seconds+policy.answer_seconds),
            'hard_job_seconds':JOB_TIMEOUT_SECONDS,'finalization_margin_seconds':FINALIZATION_MARGIN_SECONDS,
            'completion_guaranteed':False,'context_policy':'parent native guard, no new compaction or reset'}


@app.function(image=image,gpu=GPU,cpu=4,memory=32768,volumes={'/models':models,'/runs':runs},timeout=JOB_TIMEOUT_SECONDS,max_containers=1,retries=0)
def execute(run_id,selector,resume,expected_manifest_sha256):
    started=time.monotonic()
    validate_run_id(run_id);validate_selector(selector,run_id,resume)
    checkpoint=Path('/runs')/selector
    manifest=checkpoint/('extension-checkpoint.json' if resume else 'checkpoint.json')
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected_manifest_sha256:
        raise ValueError('Remote checkpoint differs from locally validated checkpoint')
    settings,completed=inspect_checkpoint(checkpoint,resume)
    destination=Path('/runs')/run_id;attempt=Path('/runs')/(run_id+'-setup')
    if destination.exists() or attempt.exists():raise ValueError('Fresh run and setup destinations required')
    if completed==7:return {'status':'already_complete','output_created':False}
    policy=TimedPolicy(**settings['policy'])
    attempt.mkdir()
    client=OwnedOllama(policy,'/models',attempt/'ollama.log',expected_digest=DIGEST,seed=policy.seed)
    try:
        metadata=client.inspect_model()
        provenance={'backend':'Modal','requested_profile':PROFILE,'model':metadata,'expected_digest':DIGEST,
                    'requested_model':MODEL,'gpu_request':GPU,'image':IMAGE,'modal_version':modal.__version__,
                    'model_volume':MODEL_VOLUME,'run_volume':RUN_VOLUME,'resource_bound':bounds(settings,completed),
                    'source_checkpoint':selector,'source_manifest_sha256':expected_manifest_sha256}
        (attempt/'setup.json').write_text(json.dumps(provenance,indent=2)+'\n')
        def commit_checkpoint(path):
            runs.commit()
            print(f'Durable extension checkpoint: {path}',flush=True)
        return run_extension(destination,client,parent_from=None if resume else checkpoint,resume_from=checkpoint if resume else None,
                             checkpoint_callback=commit_checkpoint,job_deadline=started+JOB_TIMEOUT_SECONDS-FINALIZATION_MARGIN_SECONDS,provenance=provenance)
    except BaseException as error:
        (attempt/'failure.json').write_text(json.dumps({'error':f'{type(error).__name__}: {error}'})+'\n')
        raise
    finally:
        try:client.close()
        finally:
            (attempt/'transport-events.json').write_text(json.dumps(client.events,indent=2)+'\n')
            runs.commit()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--run-id',required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--parent-from');source.add_argument('--resume-from')
    parser.add_argument('--local-checkpoint',required=True,type=Path)
    launch=parser.add_mutually_exclusive_group()
    launch.add_argument('--launch',action='store_true');launch.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv)
    resume=args.resume_from is not None;selector=args.resume_from or args.parent_from
    validate_run_id(args.run_id);validate_selector(selector,args.run_id,resume)
    settings,completed=inspect_checkpoint(args.local_checkpoint,resume)
    filename='extension-checkpoint.json' if resume else 'checkpoint.json'
    digest=hashlib.sha256((args.local_checkpoint/filename).read_bytes()).hexdigest()
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError(f'Set MODAL_PROFILE={PROFILE}')
    config=Path(os.environ.get('MODAL_CONFIG_PATH',str(Path.home()/'.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):raise ValueError('Configure required Modal profile')
    if not args.launch:
        print(json.dumps({'status':'validated_no_cloud_actions','run_id':args.run_id,'source_checkpoint':selector,
              'source_manifest_sha256':digest,'remote_freshness_checked':False,'settings':settings,'resources':bounds(settings,completed)},indent=2))
        return
    with modal.enable_output(),app.run():print(json.dumps(execute.remote(args.run_id,selector,resume,digest),indent=2))


if __name__=='__main__':main()
