"""Run 9d synchronized notebook exchanges; local validation by default, explicit cloud launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib

import modal

from orchestrator.simulated_web.modal_private_notes import validate_selector
from orchestrator.simulated_web.modal_token_pair import GPU,JOB_TIMEOUT_SECONDS,FINALIZATION_MARGIN_SECONDS,validate_run_id
from orchestrator.simulated_web.modal_timed import image,models,runs,PROFILE,RUN_VOLUME
from orchestrator.simulated_web.synchronized_exchange import build_settings,load_checkpoint,run_exchange,migrated_settings,SHORT_NOTE_POLICY,SOURCE_DENIAL_POLICY
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama
from orchestrator.simulated_web.token_pair import DIGEST

app=modal.App('germanwiki-synchronized-notebook-exchanges')


def resources(count,note_retry_policy=None,migration=None):
    return {'gpu':GPU,'gpu_count':1,'phases':16*count,'maximum_generated_tokens':(20480 if note_retry_policy else 16384)*count,
            'note_attempt_token_limits':[512,1024] if note_retry_policy else [512,512],
            'note_policy_migration':migration,
            'research_stages':3,'research_tokens_each':1024,'research_calls_each':4,
            'note_attempts':2,'note_tokens_per_attempt':512,'note_stages_per_agent_question':4,
            'memory_payload_native_token_cap':2048,'host_reads_count_as_model_phases':False,
            'hard_job_seconds':JOB_TIMEOUT_SECONDS,'completion_guaranteed':False,'checkpoint_every_pair_round':True}


@app.function(image=image,gpu=GPU,cpu=4,memory=32768,volumes={'/models':models,'/runs':runs},timeout=JOB_TIMEOUT_SECONDS,max_containers=1,retries=0)
def execute(run_id,inputs=None,resume_from=None,expected_manifest_sha256=None,migrate_note_retry=False):
    started=time.monotonic();validate_run_id(run_id)
    if migrate_note_retry and resume_from is None:raise ValueError('Note migration requires resume')
    destination=Path('/runs')/run_id;setup=Path('/runs')/(run_id+'-setup');checkpoint=None
    if destination.exists() or setup.exists():raise ValueError('Fresh run and setup IDs required')
    if resume_from is not None:
        if inputs is not None:raise ValueError('Resume overrides prohibited')
        validate_selector(resume_from,run_id,10);checkpoint=Path('/runs')/resume_from
        if hashlib.sha256((checkpoint/'synchronized-checkpoint.json').read_bytes()).hexdigest()!=expected_manifest_sha256:raise ValueError('Remote checkpoint differs from local validation')
        data,browser,_=load_checkpoint(checkpoint)
        try:
            settings=migrated_settings(data,checkpoint) if migrate_note_retry else data['settings.json'];validate_selector(resume_from,run_id,settings['question_count'])
            if data['state.json']['completed_rounds']==settings['question_count']:return {'status':'already_complete','output_created':False}
        finally:browser.close()
    else:
        if expected_manifest_sha256 is not None:raise ValueError('Unexpected checkpoint hash')
        settings,*_=build_settings(**inputs)
    setup.mkdir()
    client=OwnedOllama(TimedPolicy(**settings['policy']),'/models',setup/'ollama.log',expected_digest=DIGEST,seed=0)
    provenance={'backend':'Modal','requested_profile':PROFILE,'run_volume':RUN_VOLUME,'resource_bound':resources(settings['question_count'],settings.get('note_retry_policy'),settings.get('note_policy_migration')),'expected_digest':DIGEST}
    try:
        (setup/'setup.json').write_text(json.dumps(provenance,indent=2)+'\n')
        return run_exchange(destination,client,**(inputs or {}),resume_from=checkpoint,migrate_note_retry=migrate_note_retry,
            checkpoint_callback=lambda path:runs.commit(),job_deadline=started+JOB_TIMEOUT_SECONDS-FINALIZATION_MARGIN_SECONDS,provenance=provenance)
    except BaseException as error:
        (setup/'failure.json').write_text(json.dumps({'error':f'{type(error).__name__}: {error}'})+'\n');raise
    finally:
        try:client.close()
        finally:
            (setup/'transport-events.json').write_text(json.dumps(client.events,indent=2)+'\n');runs.commit()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--clear-source-denial',action='store_true',help='Fresh runs only: state that denied source documents are unavailable to this session and retries cannot change access')
    parser.add_argument('--no-peer-information',action='store_true')
    parser.add_argument('--short-note-retry',action='store_true');parser.add_argument('--migrate-note-retry',action='store_true')
    parser.add_argument('--run-id',required=True);parser.add_argument('--resume-from')
    for name in ('dataset','topic-file','editable-sources','question-ids','access-manifest','visible-labels','round-leaders','local-checkpoint'):parser.add_argument('--'+name,type=Path)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--launch',action='store_true');modes.add_argument('--validate-only',action='store_true')
    parser.add_argument('--detach',action='store_true',help='Keep an explicitly launched Modal app alive after local client disconnect')
    args=parser.parse_args(argv);validate_run_id(args.run_id);digest=None;inputs=None
    if args.detach and not args.launch:raise ValueError('--detach requires --launch')
    if args.migrate_note_retry and not args.resume_from:raise ValueError('--migrate-note-retry requires resume')
    fresh=(args.dataset,args.topic_file,args.editable_sources,args.question_ids,args.access_manifest,args.visible_labels,args.round_leaders)
    if args.resume_from:
        if args.clear_source_denial or args.no_peer_information or args.short_note_retry or any(v is not None for v in fresh):raise ValueError('Resume overrides prohibited')
        if args.local_checkpoint is None:raise ValueError('Local checkpoint required')
        validate_selector(args.resume_from,args.run_id,10);data,browser,_=load_checkpoint(args.local_checkpoint)
        try:settings=migrated_settings(data,args.local_checkpoint) if args.migrate_note_retry else data['settings.json'];validate_selector(args.resume_from,args.run_id,settings['question_count'])
        finally:browser.close()
        digest=hashlib.sha256((args.local_checkpoint/'synchronized-checkpoint.json').read_bytes()).hexdigest()
    else:
        if args.local_checkpoint is not None or any(v is None for v in fresh):raise ValueError('All fresh inputs required; no local checkpoint')
        inputs={'records':[json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
                'topic':args.topic_file.read_text().strip(),'selectors':json.loads(args.editable_sources.read_text()),
                'question_ids':json.loads(args.question_ids.read_text()),'access_manifest':json.loads(args.access_manifest.read_text()),
                'visible_labels':json.loads(args.visible_labels.read_text()),'round_leaders':json.loads(args.round_leaders.read_text())}
        if args.short_note_retry:inputs['note_retry_policy']=SHORT_NOTE_POLICY
        if args.no_peer_information:inputs['no_peer_information']=True
        if args.clear_source_denial:inputs['source_denial_policy']=SOURCE_DENIAL_POLICY
        settings,*_=build_settings(**inputs)
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError(f'Set MODAL_PROFILE={PROFILE}')
    config=Path(os.environ.get('MODAL_CONFIG_PATH',str(Path.home()/'.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):raise ValueError('Configure required Modal profile')
    if not args.launch:
        print(json.dumps({'status':'validated_no_cloud_actions','run_id':args.run_id,'settings':settings,'resources':resources(settings['question_count'],settings.get('note_retry_policy'),settings.get('note_policy_migration')),'remote_freshness_checked':False},indent=2));return
    with modal.enable_output(),app.run(detach=args.detach):print(json.dumps(execute.remote(args.run_id,inputs,args.resume_from,digest,args.migrate_note_retry),indent=2))


if __name__=='__main__':main()
