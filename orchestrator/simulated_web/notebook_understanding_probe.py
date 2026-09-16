"""Isolated understanding questions from a completed private-notes checkpoint."""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from orchestrator.simulated_web.browser import TOOLS as BROWSER_TOOLS
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.private_notes import load_checkpoint
from orchestrator.simulated_web.timed import run_phase_with_readiness
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS, NORMAL, validate_client

QUESTION = 'Which notebook entries can you edit with the available tools? How would you edit one?'
TOKENS = 2048
CALLS = 4
SECONDS = 180
CAPABILITY_FILES = ('browser.py','private_notes.py','append_notebooks.py','notebook_tools.py','timed_transport.py','timed.py','timed_policy.py')


def validate_probe_checkpoint(checkpoint):
    data,browser=load_checkpoint(checkpoint)
    try:
        settings=data['settings.json']
        if data['state.json']['completed_rounds']!=settings.get('question_count',3):raise ValueError('Understanding probe requires a completed checkpoint')
        if not settings.get('notebook_tools') or not settings.get('visible_labels'):raise ValueError('Require the saved notebook-tool identity condition')
        hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in CAPABILITY_FILES}
        if any(settings['source_hashes'].get(name)!=value for name,value in hashes.items()):raise ValueError('Current browser/tool/transport implementation differs from saved checkpoint')
        if settings['policy']['seed']!=0:raise ValueError('This probe preserves the saved seed0 condition')
        return {'parent_checkpoint':str(Path(checkpoint).resolve()),
                'parent_manifest_sha256':hashlib.sha256((Path(checkpoint)/'private-notes-checkpoint.json').read_bytes()).hexdigest(),
                'parent_member_sha256':json.loads((Path(checkpoint)/'private-notes-checkpoint.json').read_text())['files_sha256'],
                'capability_source_sha256':hashes,'settings':settings,
                'history_sha256':{agent:hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest() for agent,history in data['histories.json'].items()}}
    finally:browser.close()


def run_understanding_probe(run_dir,client,checkpoint,after_agent=None):
    run_dir=Path(run_dir);checkpoint=Path(checkpoint)
    if run_dir.exists():raise ValueError('Fresh probe output required')
    if run_dir.resolve().is_relative_to(checkpoint.resolve().parent.parent):raise ValueError('Probe must not write inside its parent run')
    if after_agent is not None and not callable(after_agent):raise ValueError('Invalid probe callback')
    provenance=validate_probe_checkpoint(checkpoint)
    metadata=validate_client(client)
    original_digest=provenance['settings'].get('provenance',{}).get('model',{}).get('digest')
    if original_digest and metadata.get('digest')!=original_digest:raise ValueError('Probe model differs from saved checkpoint')
    client.notebook_tools_enabled=True
    policy=replace(TimedPolicy(**provenance['settings']['policy']),preparation_generated_tokens=TOKENS,
                   preparation_browser_calls=CALLS,preparation_seconds=SECONDS)
    run_dir.mkdir(parents=True)
    shutil.copytree(checkpoint,run_dir/'seed-checkpoint')
    write_json(run_dir/'probe-settings.json',{'schema':'notebook-understanding-probe-v1','question':QUESTION,
        'generated_tokens_per_agent':TOKENS,'browser_calls_per_agent':CALLS,'seconds_per_agent':SECONDS,
        'agents':list(AGENTS),'visible_labels':provenance['settings']['visible_labels'],'model':metadata,'seed':0,
        'provenance':provenance,'tools':[ *BROWSER_TOOLS,*NOTEBOOK_TOOLS],
        'isolation':'independent browser and history copy of the same parent for each agent; tools fully executable on copies',
        'context':'original system/history preserved; no extra forced history, no masking/summarization/reset, no notebook phase',
        'mutations':'spontaneous tool writes permitted only in each isolated copy; preserved in per-agent database and audit'})
    outcomes={}
    for agent in AGENTS:
        data,browser=load_checkpoint(checkpoint)
        destination=run_dir/agent;destination.mkdir()
        history=data['histories.json'][agent];results=[];transitions=[]
        before=browser.checkpoint()
        write_json(destination/'initial-history.json',history)
        write_json(destination/'initial-browser.json',data['browser.json'])
        (destination/'initial-wiki.sqlite3').write_bytes(data['wiki.sqlite3'])
        try:
            row=run_phase_with_readiness(browser,client,history,QUESTION,'preparation',SECONDS,policy,
                destination,0,None,transitions,results,policy.initial_readiness_timeout_seconds,
                'Understanding probe '+provenance['settings']['visible_labels'][agent],agent=agent)
            outcomes[agent]={'status':row['status'],'answer':row['answer'],'generated_tokens_observed':row.get('generated_tokens_observed'),
                             'browser_calls':row.get('browser_calls'),'phase_log':f'{agent}/phase-00.jsonl'}
        except Exception as error:
            outcomes[agent]={'status':'failed','error':f'{type(error).__name__}: {error}','phase_log':f'{agent}/phase-00.jsonl'}
        finally:
            outcomes[agent]['browser_before']=before;outcomes[agent]['browser_after']=browser.checkpoint()
            write_json(destination/'final-history.json',history)
            write_json(destination/'browser.json',{'views':browser.views,'history_windows':browser.history_windows})
            write_json(destination/'results.json',results);write_json(destination/'transitions.json',transitions)
            with closing(sqlite3.connect(destination/'wiki.sqlite3')) as db:browser.db.backup(db)
            browser.close()
            write_json(destination/'manifest.json',outcomes[agent])
            write_json(run_dir/'manifest.json',{'status':'in_progress','agents':outcomes})
        if after_agent:after_agent(agent,destination)
    status={'status':'complete' if all(x['status'] in NORMAL for x in outcomes.values()) else 'partial_or_failed','agents':outcomes}
    write_json(run_dir/'manifest.json',status)
    return status
