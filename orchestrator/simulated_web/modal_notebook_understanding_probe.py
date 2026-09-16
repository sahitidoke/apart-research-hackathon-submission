"""Validate locally, then explicitly launch two isolated saved-state understanding probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tomllib

import modal

from orchestrator.simulated_web.modal_private_notes import validate_selector
from orchestrator.simulated_web.modal_timed import image,models,runs,PROFILE
from orchestrator.simulated_web.modal_token_pair import GPU,validate_run_id
from orchestrator.simulated_web.notebook_understanding_probe import validate_probe_checkpoint,run_understanding_probe
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama
from orchestrator.simulated_web.token_pair import DIGEST

app=modal.App('germanwiki-notebook-understanding-probe')
JOB_SECONDS=3600


@app.function(image=image,gpu=GPU,cpu=4,memory=32768,volumes={'/models':models,'/runs':runs},timeout=JOB_SECONDS,max_containers=1,retries=0)
def execute(run_id,parent_selector,expected_manifest_sha256):
    validate_run_id(run_id);validate_selector(parent_selector,run_id,10)
    checkpoint=Path('/runs')/parent_selector;destination=Path('/runs')/run_id;setup=Path('/runs')/(run_id+'-setup')
    if destination.exists() or setup.exists():raise ValueError('Fresh probe run and setup IDs required')
    if hashlib.sha256((checkpoint/'private-notes-checkpoint.json').read_bytes()).hexdigest()!=expected_manifest_sha256:raise ValueError('Remote parent differs from validated local checkpoint')
    provenance=validate_probe_checkpoint(checkpoint)
    setup.mkdir()
    client=OwnedOllama(TimedPolicy(**provenance['settings']['policy']),'/models',setup/'ollama.log',expected_digest=DIGEST,seed=0)
    try:
        return run_understanding_probe(destination,client,checkpoint,after_agent=lambda agent,path:runs.commit())
    finally:
        try:client.close()
        finally:
            (setup/'transport-events.json').write_text(json.dumps(client.events,indent=2)+'\n');runs.commit()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--run-id',required=True);parser.add_argument('--parent',required=True)
    parser.add_argument('--local-checkpoint',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--launch',action='store_true');modes.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv);validate_run_id(args.run_id);validate_selector(args.parent,args.run_id,10)
    provenance=validate_probe_checkpoint(args.local_checkpoint)
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError(f'Set MODAL_PROFILE={PROFILE}')
    config=Path(os.environ.get('MODAL_CONFIG_PATH',str(Path.home()/'.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):raise ValueError('Configure required Modal profile')
    if not args.launch:
        print(json.dumps({'status':'validated_no_cloud_actions','run_id':args.run_id,'parent':args.parent,
            'provenance':provenance,'resources':{'gpu':GPU,'gpu_count':1,'probes':2,'maximum_generated_tokens':4096,
            'maximum_browser_calls':8,'phase_seconds':180,'hard_job_seconds':JOB_SECONDS},'remote_freshness_checked':False},indent=2));return
    with modal.enable_output(),app.run():print(json.dumps(execute.remote(args.run_id,args.parent,provenance['parent_manifest_sha256']),indent=2))


if __name__=='__main__':main()
