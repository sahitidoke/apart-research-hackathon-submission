"""Run 9 paired notebook views; local validation by default, explicit cloud launch."""
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
from orchestrator.simulated_web.paired_notebook_views import build_paired_settings,load_checkpoint,run_paired_views
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama
from orchestrator.simulated_web.token_pair import DIGEST

app=modal.App('germanwiki-paired-notebook-views')


def resources(count):
    return {'gpu':GPU,'gpu_count':1,'phases':7*count,'maximum_generated_tokens':16384*count,
            'reader_research_blocks':2,'reader_research_tokens_each':1024,'reader_research_calls_each':2,
            'automatic_view_reads_per_round_maximum':5,'host_reads_count_as_model_phases':False,
            'hard_job_seconds':JOB_TIMEOUT_SECONDS,'completion_guaranteed':False,'checkpoint_every_pair_round':True}


@app.function(image=image,gpu=GPU,cpu=4,memory=32768,volumes={'/models':models,'/runs':runs},timeout=JOB_TIMEOUT_SECONDS,max_containers=1,retries=0)
def execute(run_id,inputs=None,resume_from=None,expected_manifest_sha256=None):
    started=time.monotonic();validate_run_id(run_id)
    destination=Path('/runs')/run_id;setup=Path('/runs')/(run_id+'-setup');checkpoint=None
    if destination.exists() or setup.exists():raise ValueError('Fresh run and setup IDs required')
    if resume_from is not None:
        if inputs is not None:raise ValueError('Resume overrides prohibited')
        validate_selector(resume_from,run_id,10);checkpoint=Path('/runs')/resume_from
        if hashlib.sha256((checkpoint/'paired-view-checkpoint.json').read_bytes()).hexdigest()!=expected_manifest_sha256:raise ValueError('Remote checkpoint differs from local validation')
        data,browser=load_checkpoint(checkpoint)
        try:
            settings=data['settings.json'];validate_selector(resume_from,run_id,settings['question_count'])
            if data['state.json']['completed_rounds']==settings['question_count']:return {'status':'already_complete','output_created':False}
        finally:browser.close()
    else:
        if expected_manifest_sha256 is not None:raise ValueError('Unexpected checkpoint hash')
        settings,*_=build_paired_settings(**inputs)
    setup.mkdir()
    client=OwnedOllama(TimedPolicy(**settings['policy']),'/models',setup/'ollama.log',expected_digest=DIGEST,seed=0)
    provenance={'backend':'Modal','requested_profile':PROFILE,'run_volume':RUN_VOLUME,'resource_bound':resources(settings['question_count']),'expected_digest':DIGEST}
    try:
        (setup/'setup.json').write_text(json.dumps(provenance,indent=2)+'\n')
        return run_paired_views(destination,client,**(inputs or {}),resume_from=checkpoint,
            checkpoint_callback=lambda path:runs.commit(),job_deadline=started+JOB_TIMEOUT_SECONDS-FINALIZATION_MARGIN_SECONDS,provenance=provenance)
    except BaseException as error:
        (setup/'failure.json').write_text(json.dumps({'error':f'{type(error).__name__}: {error}'})+'\n');raise
    finally:
        try:client.close()
        finally:
            (setup/'transport-events.json').write_text(json.dumps(client.events,indent=2)+'\n');runs.commit()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--related-append-only',action='store_true');parser.add_argument('--run-id',required=True);parser.add_argument('--resume-from')
    for name in ('dataset','topic-file','editable-sources','question-ids','access-manifest','visible-labels','round-leaders','local-checkpoint'):parser.add_argument('--'+name,type=Path)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--launch',action='store_true');modes.add_argument('--validate-only',action='store_true')
    parser.add_argument('--detach',action='store_true',help='Keep an explicitly launched Modal app alive after local client disconnect')
    args=parser.parse_args(argv);validate_run_id(args.run_id);digest=None;inputs=None
    if args.detach and not args.launch:raise ValueError('--detach requires --launch')
    fresh=(args.dataset,args.topic_file,args.editable_sources,args.question_ids,args.access_manifest,args.visible_labels,args.round_leaders)
    if args.resume_from:
        if args.related_append_only or any(v is not None for v in fresh):raise ValueError('Resume overrides prohibited')
        if args.local_checkpoint is None:raise ValueError('Local checkpoint required')
        validate_selector(args.resume_from,args.run_id,10);data,browser=load_checkpoint(args.local_checkpoint)
        try:settings=data['settings.json'];validate_selector(args.resume_from,args.run_id,settings['question_count'])
        finally:browser.close()
        digest=hashlib.sha256((args.local_checkpoint/'paired-view-checkpoint.json').read_bytes()).hexdigest()
    else:
        if args.local_checkpoint is not None or any(v is None for v in fresh):raise ValueError('All fresh inputs required; no local checkpoint')
        inputs={'records':[json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()],
                'topic':args.topic_file.read_text().strip(),'selectors':json.loads(args.editable_sources.read_text()),
                'question_ids':json.loads(args.question_ids.read_text()),'access_manifest':json.loads(args.access_manifest.read_text()),
                'visible_labels':json.loads(args.visible_labels.read_text()),'round_leaders':json.loads(args.round_leaders.read_text())}
        if args.related_append_only:inputs['related_append_only']=True
        settings,*_=build_paired_settings(**inputs)
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError(f'Set MODAL_PROFILE={PROFILE}')
    config=Path(os.environ.get('MODAL_CONFIG_PATH',str(Path.home()/'.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):raise ValueError('Configure required Modal profile')
    if not args.launch:
        print(json.dumps({'status':'validated_no_cloud_actions','run_id':args.run_id,'settings':settings,'resources':resources(settings['question_count']),'remote_freshness_checked':False},indent=2));return
    with modal.enable_output(),app.run(detach=args.detach):print(json.dumps(execute.remote(args.run_id,inputs,args.resume_from,digest),indent=2))


if __name__=='__main__':main()
