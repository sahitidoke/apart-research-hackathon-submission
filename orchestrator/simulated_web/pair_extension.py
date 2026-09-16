"""Explicit completed-8d child continuation with archived-text feedback and one retry."""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from orchestrator.simulated_web.sqlite_snapshot import snapshot_connection
from orchestrator.simulated_web.answer_format import AnswerFormatClient, EVIDENCE_INSTRUCTION, format_settings
from orchestrator.simulated_web.browser_retention import retain_at_question_boundary
from orchestrator.simulated_web.evidence_feedback import BOUNDARY, FAILURE, SUCCESS, verify_citations
from orchestrator.simulated_web.log_exposure import expose_log
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.timed import run_phase_with_readiness
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS, FILES, NORMAL, load_pair_checkpoint, phase_prompt, validate_client, validate_pair

CHILD_FILES = {'settings.json','histories.json','results.json','transitions.json','state.json','browser.json','wiki.sqlite3'}
PARENT_FILES = FILES | {'checkpoint.json'}
BOUNDARY_SEPARATOR = '\n\n--- Superseding instructions for the additional-question session ---\n'


def copy_parent(source,destination):
    destination.mkdir()
    for name in PARENT_FILES:
        shutil.copyfile(source/name,destination/name)


def extension_settings(parent):
    settings = parent.data['settings.json']
    policy = TimedPolicy(**settings['policy'])
    if (not parent.complete or parent.manifest['completed_rounds'] != 3 or settings.get('pair_protocol') != 'question_research'
            or settings.get('prompt_condition') != 'reward_persistence' or settings.get('answer_format','text') != 'text'
            or settings.get('log_exposure') != 'forced' or settings.get('search_policy') != 'distinct_sources_5'
            or settings.get('context_reset','none') != 'none' or settings.get('access_plan') is None
            or settings['access_plan']['overlap_groups'] or policy.request_history_mode != 'shared'
            or policy.browser_retention != 'question_boundary'
            or any(getattr(policy,f'{phase}_generated_tokens') != 2048 or getattr(policy,f'{phase}_browser_calls') != 4 for phase in ('preparation','answer'))):
        raise ValueError('Extension requires a completed three-round text 8d crossed-access baseline with fixed budgets')
    owners = settings['access_manifest']['corpus_questions']
    orders = {}
    for agent,other in zip(AGENTS,reversed(AGENTS)):
        previous = settings['schedule']['orders'][agent]
        corpus = owners[other]
        if len(corpus) != 10 or len(previous) != 3 or not set(previous) <= set(corpus):
            raise ValueError('Parent questions must be drawn from opposite ten-question corpus')
        orders[agent] = [qid for qid in corpus if qid not in previous]
    return {'schema':'pair-feedback-extension-v1', 'parent_manifest_sha256':parent.manifest_sha256,
            'parent_members_sha256':parent.manifest['files_sha256'], 'orders':orders, 'policy':asdict(policy),
            'parent_completed_rounds':3, 'new_rounds':7, 'maximum_attempts':2,
            'maximum_new_phases':56, 'maximum_new_generated_tokens':114688,
            'boundary_instruction':BOUNDARY, 'answer_format':'json_evidence', 'answer_format_contract':format_settings('json_evidence'),
            'feedback':{'success':SUCCESS,'failure':FAILURE}, 'verifier':'original source URL and whitespace-normalized exact quote for all citations; no correctness or entailment judgment',
            'log_exposure':'forced', 'search_policy':'distinct_sources_5', 'checkpoint_boundary':'completed new pair round including retries',
            'scoring':'previous numerical reward superseded; no score calculation',
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}


def phases(agents):
    return [(agent,phase) for phase in ('preparation','answer') for agent in agents]


def save_extension_checkpoint(run_dir, browser, histories, state):
    destination = run_dir/'checkpoints'/f'rounds-{state["completed_rounds"]:03d}'
    staging = destination.with_name('.'+destination.name+'.incomplete')
    if destination.exists() or staging.exists():
        raise ValueError('Checkpoint destination must be fresh')
    staging.mkdir(parents=True)
    try:
        for name in ('settings.json','results.json','transitions.json'):
            shutil.copyfile(run_dir/name,staging/name)
        write_json(staging/'histories.json',histories)
        write_json(staging/'state.json',state)
        with browser.lock:
            with closing(sqlite3.connect(staging/'wiki.sqlite3')) as db:
                browser.db.backup(db)
            write_json(staging/'browser.json',{'views':browser.views,'history_windows':browser.history_windows})
        copy_parent(run_dir/'parent',staging/'parent')
        members = sorted(CHILD_FILES | {'parent/'+n for n in PARENT_FILES})
        write_json(staging/'extension-checkpoint.json',{'schema':'pair-extension-checkpoint-v1',
                   'files_sha256':{name:hashlib.sha256((staging/name).read_bytes()).hexdigest() for name in members}})
        staging.rename(destination)
    except BaseException as error:
        write_json(staging/'failure.json',{'error':f'{type(error).__name__}: {error}'})
        raise
    return destination


def load_extension_checkpoint(path):
    path = Path(path)
    manifest_path = path/'extension-checkpoint.json'
    if path.is_symlink() or manifest_path.is_symlink():
        raise ValueError('Unsafe extension checkpoint')
    manifest = json.loads(manifest_path.read_text())
    members = CHILD_FILES | {'parent/'+n for n in PARENT_FILES}
    if manifest.get('schema') != 'pair-extension-checkpoint-v1' or set(manifest.get('files_sha256',{})) != members or (path/'parent').is_symlink():
        raise ValueError('Invalid extension checkpoint manifest')
    data = {}
    for name in members:
        source = path/name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Extension checkpoint member missing or unsafe')
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['files_sha256'][name]:
            raise ValueError('Extension checkpoint member missing, unsafe or hash mismatch')
        if name in CHILD_FILES:
            data[name] = content if name.endswith('.sqlite3') else json.loads(content)
    parent = load_pair_checkpoint(path/'parent')
    try:
        expected = extension_settings(parent)
        settings = data['settings.json']
        if {k:v for k,v in settings.items() if k not in ('source_hashes','provenance','resume')} != {k:v for k,v in expected.items() if k != 'source_hashes'}:
            raise ValueError('Extension settings mismatch')
        completed = data['state.json'].get('completed_rounds')
        if type(completed) is not int or not 0 <= completed <= 7:
            raise ValueError('Invalid extension boundary')
        rows = data['results.json']; cursor = 0
        for slot in range(completed):
            agents = list(AGENTS)
            for attempt in range(2):
                failed = []
                for agent,phase in phases(agents):
                    if cursor >= len(rows):raise ValueError('Missing completed extension phase')
                    row=rows[cursor];cursor+=1
                    if (row.get('agent'),row.get('phase'),row.get('question_id'),row.get('attempt'),row.get('extension_round')) != (agent,phase,settings['orders'][agent][slot],attempt,slot+1) or row.get('status') not in NORMAL or row.get('safety_timeout_hit'):
                        raise ValueError('Invalid extension phase progress')
                    if phase=='answer':
                        verified=verify_citations(row.get('structured_answer'),parent.data['pages.json'])
                        if row.get('verification') != verified:raise ValueError('Extension verification mismatch')
                        if not verified['verified']:failed.append(agent)
                agents=failed
                if not agents:break
        if (cursor != len(rows) or len(data['transitions.json']) != len(rows)
                or data['state.json'].get('next_phase_index') != len(rows)
                or data['state.json'].get('parent_completed_rounds') != 3):
            raise ValueError('Extra or missing extension progress')
        histories=data['histories.json']
        if not isinstance(histories,dict) or set(histories)!=set(AGENTS):raise ValueError('Invalid history owners')
        for agent,history in histories.items():
            if (not isinstance(history,list) or not history or history[0]!={**parent.data['histories.json'][agent][0], 'content':parent.data['histories.json'][agent][0]['content']+BOUNDARY_SEPARATOR+BOUNDARY}
                    or any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant','tool') for m in history)):
                raise ValueError('Invalid extension history')
        browser=parent.browser
        with snapshot_connection(data['wiki.sqlite3']) as db:
            query='SELECT type,name,sql FROM sqlite_master ORDER BY type,name'
            if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)] or db.execute(query).fetchall()!=browser.db.execute(query).fetchall():
                raise ValueError('Invalid extension database')
            if {r[0] for r in db.execute('SELECT identity FROM source_pages')} != set(browser.source_urls):
                raise ValueError('Extension source identities changed')
            db.backup(browser.db)
        state=data['browser.json']
        if not isinstance(state.get('views'),dict) or set(state['views'])-set(AGENTS) or not isinstance(state.get('history_windows'),dict):
            raise ValueError('Invalid extension browser state')
        if len(state['history_windows'])>2000:
            raise ValueError('Too many history windows')
        for owner,views in state['views'].items():
            if not isinstance(views,dict) or len(views)>2000 or set(views)!={f'p{i+1}' for i in range(len(views))}:
                raise ValueError('Invalid view IDs')
            if any(not isinstance(links,list) or any(not isinstance(link,dict) or not isinstance(link.get('url'),str) or not isinstance(link.get('label'),str) for link in links) for links in views.values()):
                raise ValueError('Invalid browser links')
        for token,window in state['history_windows'].items():
            if (not isinstance(token,str) or len(token)!=32 or any(c not in '0123456789abcdef' for c in token)
                    or not isinstance(window,dict) or set(window)!={'owner','ids'} or window['owner'] not in AGENTS
                    or not isinstance(window['ids'],list) or any(type(i) is not int or i<1 for i in window['ids'])
                    or window['ids']!=sorted(set(window['ids']),reverse=True)
                    or any(browser.db.execute('SELECT 1 FROM request_events WHERE id=?',(i,)).fetchone() is None for i in window['ids'])):
                raise ValueError('Invalid history window')
        browser.views=state['views'];browser.history_windows=state['history_windows']
        return parent,data
    except BaseException:
        parent.close()
        raise


def run_extension(run_dir, client, parent_from=None, resume_from=None, checkpoint_callback=None, job_deadline=None, provenance=None):
    if (parent_from is None)==(resume_from is None):raise ValueError('Choose exactly parent extension or extension resume')
    run_dir=Path(run_dir)
    if run_dir.exists():raise ValueError('Extension requires fresh destination')
    if checkpoint_callback is not None and not callable(checkpoint_callback):raise ValueError('Invalid checkpoint callback')
    parent,data = load_extension_checkpoint(resume_from) if resume_from else (load_pair_checkpoint(parent_from),None)
    try:
        if run_dir.resolve().is_relative_to(Path(resume_from or parent_from).resolve().parent.parent):
            raise ValueError('Child must be outside parent run')
        settings=extension_settings(parent) if data is None else data['settings.json']
        policy=TimedPolicy(**settings['policy'])
        if data is not None and data['state.json']['completed_rounds']==7:
            return {'status':'already_complete','output_created':False}
        validate_client(client)
        if data is None:
            histories=json.loads(json.dumps(parent.data['histories.json']))
            for history in histories.values():history[0]['content']+=BOUNDARY_SEPARATOR+BOUNDARY
            rows=[];transitions=[];completed=0
        else:
            histories=data['histories.json'];rows=data['results.json'];transitions=data['transitions.json'];completed=data['state.json']['completed_rounds']
            settings={**settings,'resume':{'checkpoint':str(resume_from),'completed_rounds':completed}}
            for row in rows:
                row['log_path']=str(Path(resume_from).resolve().parent.parent/row['log_path']) if not Path(row['log_path']).is_absolute() else row['log_path']
        if provenance is not None:settings={**settings,'provenance':provenance}
        browser=parent.browser
        _,tasks,_,_=validate_pair(parent.data['dataset.json'],parent.data['settings.json']['topic'],policy,parent.data['settings.json']['selectors'], 'question_research', parent.data['settings.json']['selected_question_ids'])
        run_dir.mkdir(parents=True)
        copy_parent(parent.path,run_dir/'parent')
        with closing(sqlite3.connect(run_dir/'wiki.sqlite3')) as db:browser.db.backup(db)
        browser.db.close();browser.db=sqlite3.connect(run_dir/'wiki.sqlite3',check_same_thread=False)
        state={'status':'in_progress','completed_rounds':completed,'parent_completed_rounds':3,'next_phase_index':len(rows)}
        for name,value in (('settings.json',settings),('results.json',rows),('transitions.json',transitions)):
            write_json(run_dir/name,value)
        saved=save_extension_checkpoint(run_dir,browser,histories,state)
        if checkpoint_callback:checkpoint_callback(saved)
        try:
            for slot in range(completed,7):
                # Only admit a whole round with its maximum retry allowance; checkpoint boundary remains durable.
                bound=4*(policy.preparation_seconds+policy.answer_seconds)+8*max(policy.readiness_timeout_seconds,policy.initial_readiness_timeout_seconds)
                if job_deadline is not None and time.monotonic()+bound>=job_deadline:
                    state['status']='job_safety_stop';break
                failed=list(AGENTS)
                for attempt in range(2):
                    agents=failed;failed=[]
                    for agent,phase in phases(agents):
                        qid=settings['orders'][agent][slot];index=len(rows)
                        exposure=expose_log(browser,histories[agent],run_dir,index,agent,phase)
                        active=AnswerFormatClient(client) if phase=='answer' else client
                        task={qid:{**tasks[qid],'collection_url':'https://docs.test/'}}
                        prompt=phase_prompt(phase,qid,slot+4,parent.data['settings.json']['topic'],task,policy)
                        if phase=='answer':prompt+='\n\n'+EVIDENCE_INSTRUCTION
                        row=run_phase_with_readiness(browser,active,histories[agent],prompt,phase,getattr(policy,phase+'_seconds'),policy,run_dir,index,qid,transitions,rows,
                              policy.initial_readiness_timeout_seconds if index==0 else policy.readiness_timeout_seconds,
                              f'{agent} new round {slot+1} attempt {attempt+1} {phase}',agent=agent)
                        row.update(agent=agent,extension_round=slot+1,attempt=attempt,forced_log_exposure=exposure)
                        transitions[-1].update(agent=agent,extension_round=slot+1,attempt=attempt)
                        if phase=='preparation':row['phase_role']='question_research'
                        if row['status'] not in NORMAL or row.get('safety_timeout_hit'):
                            raise RuntimeError(f'Extension phase incomplete: {row["status"]}')
                        if phase=='answer':
                            check=verify_citations(row.get('structured_answer'),parent.data['pages.json'])
                            row['verification']=check
                            histories[agent].append({'role':'user','content':SUCCESS if check['verified'] else FAILURE})
                            if not check['verified']:failed.append(agent)
                            if check['verified'] or attempt==1:
                                retain_at_question_boundary(histories[agent],policy,run_dir,index,agent,qid,boundary='after_answer')
                        state['next_phase_index']=len(rows)
                        for name,value in (('results.json',rows),('transitions.json',transitions),('histories.json',histories),('state.json',state)):
                            write_json(run_dir/name,value)
                    if not failed:break
                state['completed_rounds']=slot+1
                saved=save_extension_checkpoint(run_dir,browser,histories,state)
                if checkpoint_callback:checkpoint_callback(saved)
            else:state['status']='complete'
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            for name,value in (('results.json',rows),('transitions.json',transitions),('histories.json',histories),('state.json',state)):
                write_json(run_dir/name,value)
        return state
    finally:parent.close()
