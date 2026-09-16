"""Hash-checked completed-round forks; never mutate or resume inside a parent run."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from orchestrator.simulated_web.peer_note_exposure import validate_ledger
from orchestrator.simulated_web.research_reflection import MEMBERS, schedule
from orchestrator.simulated_web.token_pair import AGENTS, NORMAL, reset_base_history

SCHEMA='research-reflection-12a-round-resume-v1'
LEGACY='research-reflection-10b-concurrent-h100-v1'


def load_checkpoint(path,builder):
    path=Path(path)
    if path.is_symlink() or not path.is_dir():raise ValueError('Invalid checkpoint path')
    manifest=json.loads((path/'concurrent-research-reflection-checkpoint.json').read_text())
    if manifest.get('schema') not in (LEGACY,SCHEMA) or set(manifest.get('files_sha256',{}))!=MEMBERS:raise ValueError('Invalid checkpoint manifest')
    data={}
    for name in MEMBERS:
        p=path/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['files_sha256'][name]:raise ValueError('Checkpoint hash mismatch')
        if name!='wiki.sqlite3':data[name]=json.loads(p.read_text())
    saved=data['settings.json'];expected,pages,tasks=builder(**data['inputs.json'])
    ignored={'source_hashes','provenance','resume'}
    if {k:v for k,v in saved.items() if k not in ignored}!={k:v for k,v in expected.items() if k not in ignored} or pages!=data['pages.json']:raise ValueError('Checkpoint settings mismatch')
    if manifest['schema']!=saved.get('checkpoint_schema',LEGACY):raise ValueError('Checkpoint schema mismatch')
    state=data['state.json'];q=state.get('completed_rounds')
    if state.get('prepared') is not True or type(q) is not int or not 0<=q<=saved['question_count']:raise ValueError('Prepared completed-round checkpoint required')
    count=4+6*q
    if state.get('next_phase_index')!=count or len(data['results.json'])!=count or len(data['transitions.json'])!=count:raise ValueError('Invalid checkpoint phase count')
    for index,(row,transition,(_,_,agent,role,qid)) in enumerate(zip(data['results.json'],data['transitions.json'],schedule(saved))):
        for item in (row,transition):
            if (item.get('agent'),item.get('phase_role'),item.get('question_id'),item.get('global_phase_index'))!=(agent,role,qid,index):raise ValueError('Checkpoint phase identity mismatch')
        if row['status'] not in NORMAL and not (role=='reflection' and row['status']=='empty_response'):raise ValueError('Incomplete checkpoint phase')
        if row.get('safety_timeout_hit') or (role.endswith('note') and not row.get('note_preservation',{}).get('persistence_verified')):raise ValueError('Unverified checkpoint publication')
    if q and data['histories.json']!={a:reset_base_history(saved['system_prompts'][a],saved['topic']) for a in AGENTS}:raise ValueError('Expected reset history boundary')
    if set(data['histories.json'])!=set(AGENTS):raise ValueError('Invalid histories')
    with closing(sqlite3.connect(f'file:{path / "wiki.sqlite3"}?mode=ro',uri=True)) as db:
        if saved.get('peer_note_exposure_policy'):validate_ledger(db,state.get('peer_note_deliveries'),saved,q)
        if db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Invalid checkpoint database')
        entries=db.execute('SELECT e.slug,e.author,e.question_round,e.stage,p.title,p.body,r.id,r.agent,r.title,r.body FROM entry_provenance e LEFT JOIN pages p USING(slug) LEFT JOIN revisions r USING(slug)').fetchall()
        if db.execute('SELECT count(*) FROM revisions').fetchone()[0]!=len(entries):raise ValueError('Invalid revisions')
        for slug,author,round_index,stage,title,body,serial,revision_author,rtitle,rbody in entries:
            if author not in AGENTS or not 0<=round_index<=q or stage not in ((1,) if round_index==0 else (2,3)) or revision_author!=author or body!=rbody or title!=rtitle or not isinstance(body,str) or not body.strip() or serial%2!=(1 if author=='agent-1' else 0):raise ValueError('Invalid notebook provenance')
        if {r[0] for r in db.execute('SELECT slug FROM pages')}!=set(saved['notebooks'].values())|{e[0] for e in entries}:raise ValueError('Unknown notebook pages')
    return data


def prepare_resume(checkpoint,target_inputs,builder,destination=None):
    if Path(checkpoint).is_symlink():raise ValueError("Symlink checkpoint prohibited")
    checkpoint=Path(checkpoint).resolve();parent=checkpoint.parent.parent
    if destination is not None and Path(destination).resolve().is_relative_to(parent):raise ValueError('Fresh output must be outside parent run')
    data=load_checkpoint(checkpoint,builder);old=data['settings.json'];new,pages,_=builder(**target_inputs)
    if new.get('checkpoint_schema')!=SCHEMA:raise ValueError('Resume requires12a policy')
    extension=old.get('checkpoint_schema')!=SCHEMA
    old_inputs=data['inputs.json'];allowed={'question_ids','round_leaders','continuation_policy','model_seed'} if extension else set()
    timing_migration=None
    if (not extension and old.get('ablation_policy') is not None and old.get('ablation_policy')==new.get('ablation_policy')
            and old['policy']['answer_seconds']==180 and new['policy']['answer_seconds']==600):
        allowed.add('answer_safety_seconds')
        timing_migration={'policy':'ablation-answer-safety-v3','inherited_answer_seconds':180,'new_answer_seconds':600,
                          'boundary_completed_rounds':data['state.json']['completed_rounds'],'tokens_unchanged':True}
    for key in old_inputs.keys()|target_inputs.keys():
        if key not in allowed and old_inputs.get(key)!=target_inputs.get(key):raise ValueError('Resume input override prohibited: '+key)
    if old['policy']['seed']!=new['policy']['seed']:raise ValueError('Resume seed mismatch')
    if extension:
        if old['question_count']!=2 or data['state.json']['completed_rounds']!=2 or new['question_count']!=6:raise ValueError('Only completed2-to6extension allowed')
        for agent in AGENTS:
            if new['question_ids'][agent][:2]!=old['question_ids'][agent]:raise ValueError('Question prefix changed')
        if new['round_leaders'][:2]!=old['round_leaders']:raise ValueError('Round leader prefix changed')
    if pages!=data['pages.json']:raise ValueError('Frozen corpus changed')
    # A fresh fork preserves all original evidence, including incomplete later phases.
    for p in parent.rglob('*'):
        if p.is_symlink():raise ValueError('Symlink in parent evidence')
    for row in data['results.json']:
        p=Path(row['log_path'])
        if p.is_absolute() or '..' in p.parts or not (parent/p).is_file():raise ValueError('Missing checkpoint transcript')
    verification_start=16 if extension else old.get('resume',{}).get('verification_start_index',0)
    if verification_start not in (0,16) or verification_start>len(data['results.json']):raise ValueError('Invalid verifier boundary')
    summary_path=parent/'answer-support/summary.json'
    prior_judgments=summary_path.exists()
    if data['state.json']['completed_rounds']==6 and prior_judgments:
        summary=json.loads(summary_path.read_text())
        expected=[(i,r['agent'],r['question_id']) for i,r in enumerate(data['results.json']) if i>=verification_start and r.get('phase_role')=='answer']
        actual=[(r.get('phase_index'),r.get('agent'),r.get('question_id')) for r in summary.get('answers',[])]
        if summary.get('status')=='complete' and actual==expected:
            for row in summary['answers']:
                if json.loads((parent/'answer-support'/f"{row['phase_index']:04d}-result.json").read_text())!=row:raise ValueError('Completed judgment artifact mismatch')
            raise ValueError('Completed solver and verification already present; no resume work remains')
    return {'answer_timing_migration':timing_migration,'verification_start_index':verification_start,'prior_judge_artifacts_preserved':prior_judgments,'data':data,'checkpoint':checkpoint,'parent':parent,'extension':extension,'settings':new,
            'checkpoint_sha256':hashlib.sha256((checkpoint/'concurrent-research-reflection-checkpoint.json').read_bytes()).hexdigest()}
