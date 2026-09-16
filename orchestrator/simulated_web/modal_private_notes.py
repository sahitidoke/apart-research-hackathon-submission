"""Private-intent/public-log fresh condition; validate locally by default."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import tomllib

import modal

from orchestrator.simulated_web.modal_token_pair import GPU, JOB_TIMEOUT_SECONDS, FINALIZATION_MARGIN_SECONDS, validate_run_id
from orchestrator.simulated_web.modal_timed import image, models, runs, PROFILE, RUN_VOLUME
from orchestrator.simulated_web.private_notes import build_settings, load_checkpoint, private_notes_policy, run_private_notes
from orchestrator.simulated_web.timed_transport import OwnedOllama
from orchestrator.simulated_web.token_pair import DIGEST

app=modal.App('germanwiki-private-notes-public-log')


def validate_selector(selector,run_id,question_count=3):
    if not isinstance(selector,str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}/checkpoints/rounds-0(?:0[0-9]|10)',selector) is None or selector.split('/')[0]==run_id or int(selector[-3:])>question_count:
        raise ValueError(f'Select a different private-notes run/checkpoints/rounds-000..{question_count:03d}')


def resources(mandatory_notes=False,question_count=3):
    return {'gpu':GPU,'gpu_count':1,'phases':6*question_count,'maximum_generated_tokens':2*question_count*(4096+(4096 if mandatory_notes else 512)),'note_tokens':4096 if mandatory_notes else 512,'note_browser_calls':0 if mandatory_notes else 2,
            'hard_job_seconds':JOB_TIMEOUT_SECONDS,'completion_guaranteed':False,'checkpoint_every_pair_round':True}


@app.function(image=image,gpu=GPU,cpu=4,memory=32768,volumes={'/models':models,'/runs':runs},timeout=JOB_TIMEOUT_SECONDS,max_containers=1,retries=0)
def execute(run_id,records=None,topic=None,selectors=None,question_ids=None,access_manifest=None,resume_from=None,expected_manifest_sha256=None,source_access_mode=None,sequence_mode=None,history_search=None,log_exposure=None,retain_context=None,mandatory_notes=None,question_count=None,bounded_context=None,append_notes=None,neutral_notebook=None,visible_labels=None,round_leaders=None,notebook_tools=None):
    started=time.monotonic();validate_run_id(run_id)
    destination=Path('/runs')/run_id;attempt=Path('/runs')/(run_id+'-setup');checkpoint=None
    if destination.exists() or attempt.exists():raise ValueError('Fresh run and setup destinations required')
    if resume_from is not None:
        if any(v is not None for v in (records,topic,selectors,question_ids,access_manifest,source_access_mode,sequence_mode,history_search,log_exposure,retain_context,mandatory_notes,question_count,bounded_context,append_notes,neutral_notebook,visible_labels,round_leaders,notebook_tools)):raise ValueError('Resume overrides prohibited')
        validate_selector(resume_from,run_id,10);checkpoint=Path('/runs')/resume_from
        if hashlib.sha256((checkpoint/'private-notes-checkpoint.json').read_bytes()).hexdigest()!=expected_manifest_sha256:raise ValueError('Remote checkpoint differs from local validation')
        data,browser=load_checkpoint(checkpoint)
        try:
            settings=data['settings.json']
            validate_selector(resume_from,run_id,settings.get('question_count',3))
            if data['state.json']['completed_rounds']==settings.get('question_count',3):return {'status':'already_complete','output_created':False}
        finally:browser.close()
    else:
        if expected_manifest_sha256 is not None:raise ValueError('Unexpected resume hash')
        settings,_,_,_=build_settings(records,topic,selectors,question_ids,access_manifest,source_access_mode or "discovery_only",sequence_mode or "interleaved",False if history_search is None else history_search,log_exposure or "forced",False if retain_context is None else retain_context,False if mandatory_notes is None else mandatory_notes,3 if question_count is None else question_count,False if bounded_context is None else bounded_context,False if append_notes is None else append_notes,False if neutral_notebook is None else neutral_notebook,visible_labels,round_leaders,False if notebook_tools is None else notebook_tools)
    attempt.mkdir()
    client=OwnedOllama(private_notes_policy(settings.get('retain_context',False)),'/models',attempt/'ollama.log',expected_digest=DIGEST,seed=0)
    try:
        provenance={'backend':'Modal','requested_profile':PROFILE,'resource_bound':resources(settings.get('mandatory_notes',False),settings.get('question_count',3)),'run_volume':RUN_VOLUME,'expected_digest':DIGEST}
        (attempt/'setup.json').write_text(json.dumps(provenance,indent=2)+'\n')
        def commit_checkpoint(path):
            runs.commit();print(f'Durable private-notes checkpoint: {path}',flush=True)
        return run_private_notes(destination,client,records,topic,selectors,question_ids,access_manifest,resume_from=checkpoint,
                                 checkpoint_callback=commit_checkpoint,job_deadline=started+JOB_TIMEOUT_SECONDS-FINALIZATION_MARGIN_SECONDS,provenance=provenance,source_access_mode=source_access_mode,sequence_mode=sequence_mode,history_search=history_search,log_exposure=log_exposure,retain_context=retain_context,mandatory_notes=mandatory_notes,question_count=question_count,bounded_context=bounded_context,append_notes=append_notes,neutral_notebook=neutral_notebook,visible_labels=visible_labels,round_leaders=round_leaders,notebook_tools=notebook_tools)
    except BaseException as error:
        (attempt/'failure.json').write_text(json.dumps({'error':f'{type(error).__name__}: {error}'})+'\n');raise
    finally:
        try:client.close()
        finally:
            (attempt/'transport-events.json').write_text(json.dumps(client.events,indent=2)+'\n');runs.commit()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--run-id',required=True)
    for name in ('dataset','topic-file','editable-sources','question-ids','access-manifest','local-checkpoint','visible-labels','round-leaders'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--resume-from')
    parser.add_argument('--source-access-mode',choices=('hard','discovery_only'))
    parser.add_argument('--sequence-mode',choices=('interleaved','agent_serial'))
    parser.add_argument('--history-search',action='store_true',default=None)
    parser.add_argument('--notebook-tools',action='store_true',default=None,help='Enable local notebook read/edit tools with versioned cross-editing')
    parser.add_argument('--neutral-notebook',action='store_true',default=None,help='Use neutral research notebook labels/URLs and disable the stated search reward')
    parser.add_argument('--append-notes',action='store_true',default=None,help='Append one freeform entry after each answer without replacing earlier entries')
    parser.add_argument('--bounded-context',action='store_true',default=None,help='Retain own messages but bound old browser/log payloads using native input counts')
    parser.add_argument('--question-count',type=int,choices=range(1,11),help='Questions per agent (default: 3)')
    parser.add_argument('--mandatory-notes',action='store_true',default=None,help='Require structured agent note and verified host save before advancing')
    parser.add_argument('--retain-context',action='store_true',default=None,help='Retain full conversation across notebook updates (default: reset after each update)')
    parser.add_argument('--log-exposure',choices=('forced','spontaneous'))
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--launch',action='store_true');modes.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv);validate_run_id(args.run_id)
    digest=None;records=topic=selectors=ids=access=labels=leaders=None
    if args.resume_from:
        if any(v is not None for v in (args.dataset,args.topic_file,args.editable_sources,args.question_ids,args.access_manifest,args.source_access_mode,args.sequence_mode,args.history_search,args.log_exposure,args.retain_context,args.mandatory_notes,args.question_count,args.bounded_context,args.append_notes,args.neutral_notebook,args.visible_labels,args.round_leaders,args.notebook_tools)):raise ValueError('Resume overrides prohibited')
        if args.local_checkpoint is None:raise ValueError('Local checkpoint required to validate resume')
        validate_selector(args.resume_from,args.run_id,10);data,browser=load_checkpoint(args.local_checkpoint)
        try:
            settings=data['settings.json']
            validate_selector(args.resume_from,args.run_id,settings.get('question_count',3))
        finally:browser.close()
        digest=hashlib.sha256((args.local_checkpoint/'private-notes-checkpoint.json').read_bytes()).hexdigest()
    else:
        if args.local_checkpoint is not None:raise ValueError('Local checkpoint requires resume')
        if any(v is None for v in (args.dataset,args.topic_file,args.editable_sources,args.question_ids,args.access_manifest)):raise ValueError('All fresh input files required')
        records=[json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
        topic=args.topic_file.read_text().strip();selectors=json.loads(args.editable_sources.read_text());ids=json.loads(args.question_ids.read_text());access=json.loads(args.access_manifest.read_text())
        labels=json.loads(args.visible_labels.read_text()) if args.visible_labels else None
        leaders=json.loads(args.round_leaders.read_text()) if args.round_leaders else None
        settings,_,_,_=build_settings(records,topic,selectors,ids,access,args.source_access_mode or "discovery_only",args.sequence_mode or "interleaved",bool(args.history_search),args.log_exposure or "forced",bool(args.retain_context),bool(args.mandatory_notes),3 if args.question_count is None else args.question_count,bool(args.bounded_context),bool(args.append_notes),bool(args.neutral_notebook),labels,leaders,bool(args.notebook_tools))
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError(f'Set MODAL_PROFILE={PROFILE}')
    config=Path(os.environ.get('MODAL_CONFIG_PATH',str(Path.home()/'.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):raise ValueError('Configure required Modal profile')
    if not args.launch:
        print(json.dumps({'status':'validated_no_cloud_actions','run_id':args.run_id,'settings':settings,'resources':resources(settings.get('mandatory_notes',False),settings.get('question_count',3)),'remote_freshness_checked':False},indent=2));return
    with modal.enable_output(),app.run():print(json.dumps(execute.remote(args.run_id,records,topic,selectors,ids,access,args.resume_from,digest,args.source_access_mode,args.sequence_mode,args.history_search,args.log_exposure,args.retain_context,args.mandatory_notes,args.question_count,args.bounded_context,args.append_notes,args.neutral_notebook,labels,leaders,args.notebook_tools),indent=2))


if __name__=='__main__':main()
