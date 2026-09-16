"""Opt-in concurrent 11a: one shared model, private workers, atomic stage barriers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import traceback

from orchestrator.simulated_web.notebook_titles import POLICY as TITLE_POLICY, INSTRUCTION as TITLE_INSTRUCTION
from orchestrator.simulated_web.answer_format import AnswerFormatClient, EVIDENCE_INSTRUCTION
from orchestrator.simulated_web.answer_support import judge_answers
from orchestrator.simulated_web.collaboration_pilot import (
    build_settings as serial_settings, RetainedContextClient, FinalEntryClient,
    persist_final_entry, NOTE, FINAL_ENTRY, MEMBERS,
)
from orchestrator.simulated_web.exchange_evidence import EvidenceIndex
from orchestrator.simulated_web.hf_fp8 import validate_client
from orchestrator.simulated_web.log_exposure import expose_log
from orchestrator.simulated_web.mandatory_notes import run_mandatory_note
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.related_notebooks import HostAppendAdapter
from orchestrator.simulated_web.synchronized_exchange import SHORT_RETRY, SHORT_NOTE_HINT
from orchestrator.simulated_web.synchronized_notebooks import make_browser, fork_stage, publish_stage
from orchestrator.simulated_web.timed import run_phase_with_readiness
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS, NORMAL, reset_base_history

PROTOCOL='collaboration-pilot-11a-concurrent-h100-v1'


def build_settings(*, answer_final_reserve_tokens=256, coordinator_readiness_policy=None, agent_log_policy=None, notebook_title_policy=None, **inputs):
    if notebook_title_policy not in (None,TITLE_POLICY):
        raise ValueError("Invalid notebook title policy")
    if agent_log_policy not in (None,"no-agent-history-v1"):
        raise ValueError("Invalid agent log policy")
    if coordinator_readiness_policy not in (None, "cold-start-600-v2"):
        raise ValueError("Invalid coordinator readiness policy")
    if type(answer_final_reserve_tokens) is not int or answer_final_reserve_tokens not in (256, 512):
        raise ValueError("Answer final reserve must be256or512tokens")
    settings,pages,tasks=serial_settings(**inputs)
    settings.update(schema=PROTOCOL,protocol=PROTOCOL,sequence_mode=PROTOCOL,
        answer_finalization_policy='explicit-schema-final-only-v1',
        concurrent_execution={'workers':2,'model_processes':1,'weight_copies':1,
            'same_stage':'concurrent private endpoints and frozen replicas',
            'publication':'coordinator only, after both workers succeed and settle',
            'failure':'cancel shared owner and both HTTP requests; settle workers; preserve private artifacts; no publication',
            'result_order':'deterministic agent then phase order; worker timestamps retained',
            'readiness':'coordinator only before each stage; no worker recovery',
            'resume':'fresh launches only; checkpoint snapshots retained for provenance, no resume entrypoint'})
    if coordinator_readiness_policy is not None:
        settings['coordinator_readiness'] = {'policy':coordinator_readiness_policy,'initial_cold_seconds':600,
            'later_stage_seconds':settings['policy']['initial_readiness_timeout_seconds'],
            'scope':'first coordinator startup only; worker readiness and phase budgets unchanged'}
    if answer_final_reserve_tokens != 256:
        settings['answer_final_reserve_tokens'] = answer_final_reserve_tokens
    if agent_log_policy is not None:
        settings['policy']['request_history_mode']='disabled'
        obsolete='Superseded automatic request-history metadata is archived; the newest payload remains. '
        settings['system_prompts']={agent:prompt.replace(obsolete,'') for agent,prompt in settings['system_prompts'].items()}
        settings.update(agent_log_policy=agent_log_policy,log_exposure='disabled',
            request_history='host-only metadata audit; no agent-visible history routes, links or search',
            history_search=False,history_search_policy={'enabled':False})
    if notebook_title_policy is not None:
        settings['notebook_title_policy']=notebook_title_policy
        settings['system_prompts']={agent:prompt+'\n\n'+TITLE_INSTRUCTION for agent,prompt in settings['system_prompts'].items()}
        settings['notebook_tool_schemas']=json.loads(json.dumps(settings['notebook_tool_schemas']))
        for tool in settings['notebook_tool_schemas']:
            if tool['function']['name']=='append_notebook':
                tool['function']['description']+=' '+TITLE_INSTRUCTION
    settings['execution']='concurrent pair per frozen stage; '+settings['execution']
    settings['schedule']['execution']=settings['execution']
    for name in ('concurrent_collaboration_pilot.py','concurrent_hf_transport.py','modal_hf_concurrent_collaboration_pilot.py','notebook_titles.py'):
        path=Path(__file__).with_name(name)
        settings['source_hashes'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return settings,pages,tasks



def readiness_with_progress(owner, timeout, label, *, interval_seconds=30):
    """Report waiting only; model readiness stays on the calling coordinator thread."""
    if interval_seconds <= 0:
        raise ValueError('Progress interval must be positive')
    started=time.monotonic();stop=threading.Event()
    log_path=getattr(owner,'log_path',None)
    print(f'{label}: coordinator readiness starting; timeout={timeout:g}s; model log={log_path if log_path is not None else "unavailable"}',flush=True)
    def heartbeat():
        while not stop.wait(interval_seconds):
            elapsed=max(0,time.monotonic()-started)
            print(f'{label}: coordinator readiness waiting; elapsed={elapsed:.1f}s; remaining={max(0,timeout-elapsed):.1f}s',flush=True)
    progress=threading.Thread(target=heartbeat,name='collaboration-readiness-progress',daemon=True)
    progress.start()
    try:
        result=owner.ensure_ready(timeout=timeout)
    except BaseException as error:
        stop.set();progress.join()
        print(f'{label}: coordinator readiness failed; elapsed={max(0,time.monotonic()-started):.1f}s; error_type={type(error).__name__}',flush=True)
        raise
    else:
        stop.set();progress.join()
        print(f'{label}: coordinator readiness ready; elapsed={max(0,time.monotonic()-started):.1f}s',flush=True)
        return result
    finally:
        stop.set()
        progress.join()


def checkpoint(path,browser,histories,settings,inputs,pages,results,transitions,prepared,rounds):
    name=f'rounds-{rounds:03d}' if prepared else 'initial'
    dest=path/'checkpoints'/name;temp=dest.with_name('.'+name+'.incomplete')
    if dest.exists() or temp.exists():raise ValueError('Fresh concurrent checkpoint required')
    temp.mkdir(parents=True)
    data={'settings.json':settings,'inputs.json':inputs,'pages.json':pages,'histories.json':histories,
          'results.json':results,'transitions.json':transitions,
          'state.json':{'prepared':prepared,'completed_rounds':rounds,'next_phase_index':len(results)},
          'browser.json':{'views':browser.views,'history_windows':browser.history_windows}}
    for filename,value in data.items():write_json(temp/filename,value)
    with closing(sqlite3.connect(temp/'wiki.sqlite3')) as db:browser.db.backup(db)
    write_json(temp/'concurrent-collaboration-pilot-checkpoint.json',{'schema':PROTOCOL,'files_sha256':{
        name:hashlib.sha256((temp/name).read_bytes()).hexdigest() for name in sorted(MEMBERS)}})
    temp.rename(dest);return dest


def worker(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,owner,gate):
    """One thread owns this replica connection and all mutable per-agent bookkeeping."""
    rows=[];transitions=[];events=[];audits=[];error=None
    qid=settings['question_ids'][agent][q-1] if q else None
    started=datetime.now(timezone.utc).isoformat()
    try:
        local.db=sqlite3.connect(folder/(agent+'.sqlite3'))
        gate.wait(timeout=30)
        for offset,role in enumerate(roles):
            if owner.cancelled.is_set():raise RuntimeError('Paired stage cancelled before next phase')
            phase_index=first_phase+offset
            audits.append(local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0])
            if role.endswith('note'):
                if role!='initial_note':
                    persist_final_entry(local,client,history,path,phase_index,agent,qid,settings['notebooks'][agent],rows[-1],rows,transitions,policy.readiness_timeout_seconds,role=role.removesuffix('_note'))
                else:
                    run_mandatory_note(HostAppendAdapter(local),client,history,path,phase_index,agent,qid,settings['notebooks'][agent],None,
                        rows,transitions,policy.readiness_timeout_seconds,append_notes=True,neutral_notebook=True,attempt_tokens=512,
                        retry_tokens=1024,retry_instruction=SHORT_RETRY,note_instruction=NOTE+SHORT_NOTE_HINT+(' '+TITLE_INSTRUCTION if settings.get('notebook_title_policy')==TITLE_POLICY else ''),research_note=True)
            else:
                semantic_phase={'initial_research':'preparation','research1':'preparation','research2':'preparation','answer':'answer','reflection':'reflection'}[role]
                if settings.get('agent_log_policy') != 'no-agent-history-v1':
                    client.before_forced_exposure(agent,history)
                    exposed=expose_log(local,history,path,phase_index,agent,'answer' if role=='answer' else 'preparation',visible_label=settings['visible_labels'][agent])
                    # Host metadata injection has its own event; never label it voluntary.
                    audits[-1]=local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
                    events.append({'event':'host_metadata_log_exposure','phase_index':phase_index,'artifact':exposed,'worker_timestamp_utc':datetime.now(timezone.utc).isoformat()})
                if role=='initial_research':prompt=f'Initial research: 600 seconds.\nTopics: {settings["topic"]}\nResearch these topics using available documents. No question has been assigned yet. Finish with useful findings and uncertainties.'
                elif role in ('research1','research2'):prompt=f'Research stage {index-1}: 600 seconds.\nQuestion: {tasks[qid]["question"]}\nInvestigate useful findings, uncertainties, and sources using browser tools. '+FINAL_ENTRY
                elif role=='answer':prompt=f'Answer phase: 180 seconds.\nQuestion: {tasks[qid]["question"]}\nUse browser tools as needed. '+EVIDENCE_INSTRUCTION
                else:prompt='Reflection: 600 seconds.\nYour submitted answer is locked. Investigate uncertainties or evidence from the completed question. Browser tools are available. No next question is assigned yet. Conclude with one natural freeform notebook entry recording useful findings, sources and uncertainties. Your exact final text will be appended to the notebook. Keep it nonblank and at most6000characters; do not repeat the whole notebook.'
                if role in ('research1','research2','reflection') and settings.get('notebook_title_policy')==TITLE_POLICY:prompt+=' '+TITLE_INSTRUCTION
                phase_policy=replace(policy, final_reserve_tokens=settings.get('answer_final_reserve_tokens',256)) if role=='answer' else policy
                engine_phase=semantic_phase
                phase_client=AnswerFormatClient(client,require_explicit_finalization=True) if role=='answer' else client
                if role in ('research1','research2','reflection'):
                    tokens=1536 if role.startswith('research') else 1024
                    phase_policy=replace(policy,answer_generated_tokens=tokens,answer_browser_calls=4,answer_seconds=600)
                    engine_phase='answer';phase_client=FinalEntryClient(client)
                run_phase_with_readiness(local,phase_client,history,prompt,engine_phase,getattr(phase_policy,engine_phase+'_seconds'),phase_policy,path,phase_index,qid,transitions,rows,policy.initial_readiness_timeout_seconds,f'{agent} round {q} {role}',agent=agent,notebook_quota_exempt=True)
                rows[-1].update(phase=semantic_phase,timing_engine_phase=engine_phase)
                transitions[-1].update(phase=semantic_phase,timing_engine_phase=engine_phase)
            row=rows[-1]
            if role=='answer' and (row['status']!='complete' or row.get('answer_format_enforcement')!='schema_and_host' or not row.get('final_attempted')):
                raise RuntimeError('Answer failed explicit schema-constrained submission')
            if row['status'] not in NORMAL and not (role in ('research1','research2','reflection') and row['status']=='empty_response'):
                raise RuntimeError('Incomplete '+role)
            if row.get('safety_timeout_hit') or (role.endswith('note') and not row['note_preservation']['persistence_verified']):raise RuntimeError('Unverified '+role)
    except BaseException as failure:
        error={'error':f'{type(failure).__name__}: {failure}','traceback':traceback.format_exc()}
        try:owner.cancel()
        except BaseException as cancellation:error['cancellation_error']=f'{type(cancellation).__name__}: {cancellation}'
        write_json(path/'failure.json',error)
    finally:
        try:
            for offset,row in enumerate(rows):
                role=roles[offset];row.update(agent=agent,phase_role=role,global_phase_index=first_phase+offset,
                    phase='answer' if role=='answer' else 'reflection' if role.startswith('reflection') else 'preparation',
                    browser_database=str(folder/(agent+'.sqlite3')))
            for offset,row in enumerate(transitions):
                role=roles[offset];row.update(agent=agent,phase_role=role,global_phase_index=first_phase+offset,
                    phase='answer' if role=='answer' else 'reflection' if role.startswith('reflection') else 'preparation')
            write_json(path/'results.json',rows);write_json(path/'transitions.json',transitions);write_json(path/'history.json',history)
            finished=datetime.now(timezone.utc).isoformat()
            write_json(path/'worker-status.json',{'agent':agent,'status':'failed' if error else 'complete','started_utc':started,'finished_utc':finished,'error':error})
        finally:
            if local.db is not None:
                try:
                    local.db.close()
                finally:
                    local.db = None
    return {'rows':rows,'transitions':transitions,'events':events,'audits':audits,'error':error,'started_utc':started,'finished_utc':finished}


def run_concurrent_collaboration_pilot(run_dir,owner,*,checkpoint_callback=None,job_deadline=None,provenance=None,**inputs):
    path=Path(run_dir);browser=None;snapshots={}
    if path.exists():raise ValueError('Fresh run destination required')
    if checkpoint_callback is not None and not callable(checkpoint_callback):raise ValueError('Invalid checkpoint callback')
    settings,pages,tasks=build_settings(**inputs)
    if getattr(owner,'concurrent_stage_supported',False) is not True or not callable(getattr(owner,'endpoint',None)) or not callable(getattr(owner,'cancel',None)) or not isinstance(getattr(owner,'cancelled',None),threading.Event):
        raise ValueError('Concurrent runner requires one cancellable shared owner with private endpoints')
    settings['provenance']={**(provenance or {}),'model':validate_client(owner)}
    owner.notebook_tools_enabled=True;owner.notebook_tool_schemas=settings['notebook_tool_schemas']
    endpoints={a:owner.endpoint(a) for a in AGENTS}
    if len({id(v) for v in endpoints.values()})!=2:raise ValueError('Distinct per-agent request endpoints required')
    policy=TimedPolicy(**settings['policy']);results=[];transitions=[]
    histories={a:reset_base_history(settings['system_prompts'][a],settings['topic']) for a in AGENTS}
    status={'status':'in_progress','prepared':False,'completed_rounds':0,'next_phase_index':0}
    path.mkdir(parents=True)
    try:
        browser=make_browser(settings,pages,path/'wiki.sqlite3');evidence=EvidenceIndex(path)
        contexts={}
        for agent in AGENTS:
            diagnostics=path/('private-'+agent);diagnostics.mkdir()
            contexts[agent]=RetainedContextClient(endpoints[agent],policy,diagnostics)
        def save():
            for filename,value in [('settings.json',settings),('inputs.json',inputs),('pages.json',pages),('results.json',results),('transitions.json',transitions),('histories.json',histories),('manifest.json',status)]:write_json(path/filename,value)
        def boundary():
            cp=checkpoint(path,browser,histories,settings,inputs,pages,results,transitions,status['prepared'],status['completed_rounds'])
            if checkpoint_callback:checkpoint_callback(cp)
        def stage(q,index,roles):
            nonlocal snapshots
            folder=path/f'round-{q:02d}-stage-{index}';folder.mkdir()
            snapshots=fork_stage(browser,settings,pages,folder,q,index)
            evidence.emit('stage_snapshots_frozen',question_round=q,stage=index,databases={a:str(folder/(a+'.sqlite3')) for a in AGENTS})
            # Transfer each SQLite connection's ownership to exactly one worker thread.
            for local in snapshots.values():
                local.db.close()
                local.db = None
            ready_timeout=settings.get('coordinator_readiness',{}).get('initial_cold_seconds',policy.initial_readiness_timeout_seconds) if q==0 else policy.initial_readiness_timeout_seconds
            write_json(folder/'shared-readiness.json',{'status':'in_progress','timeout_seconds':ready_timeout})
            try:ready=readiness_with_progress(owner,ready_timeout,f'Round {q} stage {index}')
            except BaseException as error:
                write_json(folder/'shared-readiness.json',{'status':'failed','timeout_seconds':ready_timeout,'error':f'{type(error).__name__}: {error}'})
                raise
            write_json(folder/'shared-readiness.json',{**ready,'timeout_seconds':ready_timeout})
            gate=threading.Barrier(2);outcomes={};base=len(results)
            with ThreadPoolExecutor(max_workers=2,thread_name_prefix='11a-agent') as pool:
                try:
                    futures={}
                    for offset,agent in enumerate(AGENTS):
                        workpath=folder/(agent+'-work');workpath.mkdir()
                        futures[pool.submit(worker,snapshots[agent],contexts[agent],histories[agent],settings,tasks,policy,folder,workpath,agent,q,index,roles,base+offset*len(roles),owner,gate)]=agent
                    for future in as_completed(futures):
                        agent=futures[future]
                        try:outcomes[agent]=future.result()
                        except BaseException as error:
                            diagnostic={'error':f'{type(error).__name__}: {error}'}
                            try:owner.cancel()
                            except BaseException as cancellation:diagnostic['cancellation_error']=f'{type(cancellation).__name__}: {cancellation}'
                            gate.abort()
                            outcomes[agent]={'rows':[],'transitions':[],'events':[],'audits':[],'error':diagnostic}
                except BaseException:
                    # Cancel while still inside the pool scope; __exit__ then settles workers.
                    try:
                        owner.cancel()
                    except BaseException as cancellation:
                        status['cancellation_error']=f'{type(cancellation).__name__}: {cancellation}'
                    finally:
                        gate.abort()
                    raise
            # Workers have settled; only this coordinator touches central state/evidence.
            for agent in AGENTS:
                local=snapshots[agent];local.db=sqlite3.connect(folder/(agent+'.sqlite3'))
                outcome=outcomes[agent];workpath=folder/(agent+'-work')
                for event in outcome['events']:
                    exposed=event['artifact']
                    event['artifact']={**exposed,'artifact':str((workpath/exposed['artifact']).relative_to(path))}
                    evidence.emit(event.pop('event'),question_round=q,stage=index,agent=agent,**event)
                for offset,row in enumerate(outcome['rows']):
                    row['log_path']=str((workpath/row['log_path']).relative_to(path))
                    row.update(worker_started_utc=outcome.get('started_utc'),worker_finished_utc=outcome.get('finished_utc'))
                    results.append(row)
                    audit_end=outcome['audits'][offset+1] if offset+1<len(outcome['audits']) else local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
                    evidence.phase(row,local,outcome['audits'][offset],q,index,row['global_phase_index'],folder/(agent+'.sqlite3'),audit_end=audit_end,results_index=len(results)-1)
                transitions.extend(outcome['transitions'])
            result_positions={row['global_phase_index']:position for position,row in enumerate(results)}
            for row in results:
                if 'final_entry_result_index' in row and 'final_entry_phase_index' not in row:
                    row['final_entry_phase_index']=row['final_entry_result_index']
                    row['final_entry_result_index']=result_positions[row['final_entry_phase_index']]
            status['next_phase_index']=len(results);save()
            if owner.cancelled.is_set() or any(o['error'] for o in outcomes.values()):
                write_json(folder/'barrier.json',{'published':False,'round':q,'stage':index,'workers_settled':True,'errors':{a:o['error'] for a,o in outcomes.items()}})
                raise RuntimeError('Concurrent stage failed; both workers settled; publication prohibited')
            if roles==('answer',):
                for agent in AGENTS:evidence.emit('answer_locked',agent=agent,question_round=q,results_index=outcomes[agent]['rows'][0]['global_phase_index'])
            mappings=publish_stage(browser,snapshots)
            write_json(folder/'barrier.json',{'published':True,'round':q,'stage':index,'workers_settled':True,'replica_to_central_request_ids':mappings})
            evidence.emit('stage_published',question_round=q,stage=index,artifact=str(folder/'barrier.json'))
            for local in snapshots.values():
                if local.db is not None:local.close()
            snapshots={};save()
        try:
            save();boundary()
            startup_extra=settings.get('coordinator_readiness',{}).get('initial_cold_seconds',policy.initial_readiness_timeout_seconds)-policy.initial_readiness_timeout_seconds
            if job_deadline is not None and time.monotonic()+startup_extra+2*(600+240)+4*policy.initial_readiness_timeout_seconds>=job_deadline:status['status']='job_safety_stop'
            else:
                stage(0,1,('initial_research','initial_note'));status['prepared']=True;save();boundary()
            if status['prepared']:
                for q in range(1,settings['question_count']+1):
                    if job_deadline is not None and time.monotonic()+2*(2*600+180+600+3*240)+14*policy.initial_readiness_timeout_seconds>=job_deadline:
                        status['status']='job_safety_stop';break
                    stage(q,2,('research1','research1_note'));stage(q,3,('research2','research2_note'))
                    stage(q,4,('answer',));stage(q,5,('reflection','reflection_note'))
                    write_json(path/f'round-{q:02d}-histories-before-reset.json',histories)
                    evidence.emit('question_context_reset',question_round=q,artifact=f'round-{q:02d}-histories-before-reset.json',reflection_notes_verified=True)
                    histories={a:reset_base_history(settings['system_prompts'][a],settings['topic']) for a in AGENTS}
                    status['completed_rounds']=q;save();boundary()
                else:status['status']='complete'
        except BaseException as error:
            status.update(status='failed',error=f'{type(error).__name__}: {error}')
            try:owner.cancel()
            except BaseException as cancellation:status['cancellation_error']=f'{type(cancellation).__name__}: {cancellation}'
            evidence.emit('run_failed',error=status['error'],partial_phase_artifacts_retained=True);raise
        finally:
            status['next_phase_index']=len(results);status['solver_status']=status['status']
            if status['status'] in ('complete','job_safety_stop'):status['status']='verification_pending'
            save();evidence.emit('solver_finished',status=status['solver_status'],completed_rounds=status['completed_rounds'],results_count=len(results))
        status['status']='verification_in_progress';save()
        try:status['answer_support']=judge_answers(path,owner,results,tasks,pages,settings,job_deadline)
        except BaseException as error:
            status.update(status='verification_interrupted',error=f'{type(error).__name__}: {error}');save();raise
        status.update(status=status['solver_status'],answer_support_status='complete');save();return status
    finally:
        for local in snapshots.values():
            if local.db is not None:local.close()
        if browser is not None:browser.close()
