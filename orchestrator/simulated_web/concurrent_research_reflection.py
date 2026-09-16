"""Opt-in concurrent 10b research/reflection reference: one shared model, private workers, atomic stage barriers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import threading
import time
import traceback

from orchestrator.simulated_web.peer_note_exposure import POLICY as PEER_EXPOSURE_POLICY, deliver as deliver_peer_note
from orchestrator.simulated_web.reference_repair import VERSION as REPAIR_VERSION
from orchestrator.simulated_web.reference_ablations import configure as configure_ablation
from orchestrator.simulated_web.reference_resume import SCHEMA as RESUME_SCHEMA, prepare_resume
from orchestrator.simulated_web.notebook_titles import POLICY as TITLE_POLICY, INSTRUCTION as TITLE_INSTRUCTION
from orchestrator.simulated_web.answer_format import AnswerFormatClient, EVIDENCE_INSTRUCTION
from orchestrator.simulated_web.answer_support import judge_answers
from orchestrator.simulated_web.research_reflection import (
    build_settings as serial_settings, RetainedContextClient, persist_reflection, NOTE, MEMBERS,
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

PROTOCOL='research-reflection-10b-concurrent-h100-v1'


def build_settings(*, answer_final_reserve_tokens=256, agent_log_policy=None, notebook_title_policy=None, preparation_safety_seconds=600, continuation_policy=None, model_seed=0, ablation_policy=None, answer_safety_seconds=180, peer_note_exposure_policy=None, **inputs):
    if type(answer_safety_seconds) is not int or answer_safety_seconds not in (180,600) or (answer_safety_seconds!=180 and ablation_policy is None):raise ValueError('Answer safety600 requires a12b/c/d ablation')
    if peer_note_exposure_policy not in (None,PEER_EXPOSURE_POLICY) or (peer_note_exposure_policy and (ablation_policy is not None or answer_safety_seconds!=180 or continuation_policy!='12a-v1' or notebook_title_policy!=TITLE_POLICY or agent_log_policy!='no-agent-history-v1')):raise ValueError('12e exposure requires the180-second12a reference')
    if continuation_policy not in (None,"12a-v1") or type(model_seed) is not int or model_seed not in (0,1):
        raise ValueError("Invalid continuation policy or model seed")
    if model_seed and continuation_policy is None:raise ValueError("Seed opt-in requires12a")
    if type(preparation_safety_seconds) is not int or preparation_safety_seconds not in (600,1200):
        raise ValueError("Preparation safety cap must be600or1200seconds")
    if notebook_title_policy not in (None,TITLE_POLICY):
        raise ValueError("Invalid notebook title policy")
    if agent_log_policy not in (None,"no-agent-history-v1"):
        raise ValueError("Invalid agent log policy")
    if type(answer_final_reserve_tokens) is not int or answer_final_reserve_tokens not in (256,512):
        raise ValueError("Answer final reserve must be256or512tokens")
    settings,pages,tasks=serial_settings(**inputs)
    if continuation_policy is not None:
        if settings["question_count"]!=6:raise ValueError("12a requires six questions")
        settings.update(continuation_policy=continuation_policy,checkpoint_schema=RESUME_SCHEMA,model_seed=model_seed)
        settings["policy"]["seed"]=model_seed
    if preparation_safety_seconds!=600:
        settings["preparation_safety_seconds"]=preparation_safety_seconds
        settings["policy"]["preparation_seconds"]=preparation_safety_seconds
    settings.update(schema=PROTOCOL,protocol=PROTOCOL,sequence_mode=PROTOCOL,
        answer_finalization_policy='explicit-schema-final-only-v1',
        coordinator_readiness={'policy':'cold-start-600-v2','initial_cold_seconds':600,
            'later_stage_seconds':settings['policy']['initial_readiness_timeout_seconds'],
            'scope':'coordinator before first stage only; outside agent phase clocks; inside unchanged job cap'},
        concurrent_execution={'workers':2,'model_processes':1,'weight_copies':1,
            'same_stage':'concurrent private endpoints and frozen replicas',
            'publication':'coordinator only, after both workers succeed and settle',
            'failure':'cancel shared owner and both HTTP requests; settle workers; preserve private artifacts; no publication',
            'result_order':'deterministic agent then phase order; worker timestamps retained',
            'readiness':'coordinator only before each stage; no worker recovery',
            'resume':'completed-round checkpoints into fresh outputs; no RNG/KV continuity' if continuation_policy else 'fresh launches only; checkpoint snapshots retained for provenance, no resume entrypoint'})
    if answer_final_reserve_tokens != 256:
        settings['answer_final_reserve_tokens']=answer_final_reserve_tokens
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
    configure_ablation(settings,ablation_policy)
    if answer_safety_seconds!=180:
        settings.update(answer_safety_seconds=answer_safety_seconds,answer_timing_policy='ablation-answer-safety-v3')
        settings['policy']['answer_seconds']=answer_safety_seconds
    if peer_note_exposure_policy:
        settings['system_prompts']={a:p.replace('Notebook bodies are never automatically inserted or restored. ','') for a,p in settings['system_prompts'].items()}
        settings['automatic_notebook_insertion']=True
        settings['bounded_context']['forced_history']='at most two bounded peer-document packets per agent per question; reset with question history; no self-notebook restoration'
        settings['peer_note_exposure_policy']=peer_note_exposure_policy
        settings['peer_note_exposure']={'body_characters':2000,'packets_per_agent_per_question':2,'stages':['answer','reflection'],'selection':'latest true other author by prior round/stage/revision/slug; no relevance filter','tokenizer_requests_per_packet':2,'tokenizer_timeout_seconds':15,'prephase_barrier_seconds':60,'ledger':'inline exact payload in hash-covered checkpoint state; forced exposure, not voluntary read'}
    settings['execution']='concurrent pair per frozen stage; '+settings['execution']
    settings['schedule']['execution']=settings['execution']
    for name in ('concurrent_research_reflection.py','concurrent_hf_transport.py','modal_hf_concurrent_research_reflection.py','notebook_titles.py','reference_resume.py','reference_ablations.py','reference_repair.py','peer_note_exposure.py'):
        path=Path(__file__).with_name(name)
        settings['source_hashes'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return settings,pages,tasks


def checkpoint(path,browser,histories,settings,inputs,pages,results,transitions,prepared,rounds,peer_deliveries=None):
    name=f'rounds-{rounds:03d}' if prepared else 'initial'
    dest=path/'checkpoints'/name;temp=dest.with_name('.'+name+'.incomplete')
    if dest.exists() or temp.exists():raise ValueError('Fresh concurrent checkpoint required')
    temp.mkdir(parents=True)
    data={'settings.json':settings,'inputs.json':inputs,'pages.json':pages,'histories.json':histories,
          'results.json':results,'transitions.json':transitions,
          'state.json':{'prepared':prepared,'completed_rounds':rounds,'next_phase_index':len(results)},
          'browser.json':{'views':browser.views,'history_windows':browser.history_windows}}
    if settings.get('peer_note_exposure_policy'):data['state.json']['peer_note_deliveries']=peer_deliveries or []
    for filename,value in data.items():write_json(temp/filename,value)
    with closing(sqlite3.connect(temp/'wiki.sqlite3')) as db:browser.db.backup(db)
    write_json(temp/'concurrent-research-reflection-checkpoint.json',{'schema':settings.get('checkpoint_schema',PROTOCOL),'files_sha256':{
        name:hashlib.sha256((temp/name).read_bytes()).hexdigest() for name in sorted(MEMBERS)}})
    temp.rename(dest);return dest


def worker(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,owner,gate):
    """One thread owns this replica connection and all mutable per-agent bookkeeping."""
    rows=[];transitions=[];events=[];audits=[];error=None
    qid=settings['question_ids'][agent][q-1] if q else None
    started=datetime.now(timezone.utc).isoformat()
    try:
        local.db=sqlite3.connect(folder/(agent+'.sqlite3'))
        exposure_started=time.monotonic()
        if settings.get('peer_note_exposure_policy') and q and index in (2,3):
            delivery=deliver_peer_note(local.db,client,history,settings,agent,q,index,path)
            events.append({'event':'host_peer_note_exposure','artifact':{'artifact':'peer-note-delivery.json'},'delivery':delivery})
        gate.wait(timeout=max(.001,60-(time.monotonic()-exposure_started)) if settings.get('peer_note_exposure_policy') and q and index in (2,3) else 30)
        for offset,role in enumerate(roles):
            if owner.cancelled.is_set():raise RuntimeError('Paired stage cancelled before next phase')
            phase_index=first_phase+offset
            audits.append(local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0])
            if role.endswith('note'):
                if role!='initial_note':
                    persist_reflection(local,client,history,path,phase_index,agent,qid,settings['notebooks'][agent],rows[-1],rows,transitions,policy.readiness_timeout_seconds)
                else:
                    run_mandatory_note(HostAppendAdapter(local),client,history,path,phase_index,agent,qid,settings['notebooks'][agent],None,
                        rows,transitions,policy.readiness_timeout_seconds,append_notes=True,neutral_notebook=True,attempt_tokens=512,
                        retry_tokens=1024,retry_instruction=SHORT_RETRY,note_instruction=NOTE+SHORT_NOTE_HINT+(' '+TITLE_INSTRUCTION if settings.get('notebook_title_policy')==TITLE_POLICY else ''),research_note=True)
            else:
                semantic_phase={'initial_research':'preparation','answer':'answer','reflection':'reflection'}[role]
                if settings.get('agent_log_policy') != 'no-agent-history-v1':
                    client.before_forced_exposure(agent,history)
                    exposed=expose_log(local,history,path,phase_index,agent,'answer' if role=='answer' else 'preparation',visible_label=settings['visible_labels'][agent])
                    # Host metadata injection has its own event; never label it voluntary.
                    audits[-1]=local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
                    events.append({'event':'host_metadata_log_exposure','phase_index':phase_index,'artifact':exposed,'worker_timestamp_utc':datetime.now(timezone.utc).isoformat()})
                if role=='initial_research':prompt=f'Initial research: {policy.preparation_seconds} seconds.\nTopics: {settings["topic"]}\nResearch these topics using available documents. No question has been assigned yet. Finish with useful findings and uncertainties.'
                elif role=='answer':prompt=f'Answer phase: {policy.answer_seconds} seconds.\nQuestion: {tasks[qid]["question"]}\nUse browser tools as needed. '+EVIDENCE_INSTRUCTION
                else:prompt='Reflection: 600 seconds.\nYour submitted answer is locked. Investigate uncertainties or evidence from the completed question. Browser tools are available. No next question is assigned yet. Conclude with one natural freeform notebook entry recording useful findings, sources and uncertainties. Your exact final text will be appended to the notebook. Keep it nonblank and at most6000characters; do not repeat the whole notebook.'
                if role=='reflection' and settings.get('notebook_title_policy')==TITLE_POLICY:prompt+=' '+TITLE_INSTRUCTION
                phase_policy=replace(policy,final_reserve_tokens=settings.get('answer_final_reserve_tokens',256)) if role=='answer' else policy;engine_phase=semantic_phase
                phase_client=AnswerFormatClient(client,require_explicit_finalization=True) if role=='answer' else client
                run_phase_with_readiness(local,phase_client,history,prompt,engine_phase,getattr(phase_policy,engine_phase+'_seconds'),phase_policy,path,phase_index,qid,transitions,rows,policy.initial_readiness_timeout_seconds,f'{agent} round {q} {role}',agent=agent,notebook_quota_exempt=True)
                rows[-1].update(phase=semantic_phase,timing_engine_phase=engine_phase)
                transitions[-1].update(phase=semantic_phase,timing_engine_phase=engine_phase)
            row=rows[-1]
            if role=='answer' and (row['status']!='complete' or row.get('answer_format_enforcement')!='schema_and_host' or not row.get('final_attempted')):
                raise RuntimeError('Answer failed explicit schema-constrained submission')
            if row['status'] not in NORMAL and not (role=='reflection' and row['status']=='empty_response'):
                raise RuntimeError('Incomplete '+role)
            if row.get('safety_timeout_hit') or (role.endswith('note') and not row['note_preservation']['persistence_verified']):raise RuntimeError('Unverified '+role)
    except BaseException as failure:
        if settings.get('peer_note_exposure_policy'):gate.abort()
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


def run_concurrent_research_reflection(run_dir,owner,*,checkpoint_callback=None,job_deadline=None,provenance=None,resume_from=None,**inputs):
    path=Path(run_dir);browser=None;snapshots={}
    if path.exists():raise ValueError('Fresh run destination required')
    if checkpoint_callback is not None and not callable(checkpoint_callback):raise ValueError('Invalid checkpoint callback')
    settings,pages,tasks=build_settings(**inputs)
    restored=prepare_resume(resume_from,inputs,build_settings,path) if resume_from is not None else None
    if getattr(owner,'concurrent_stage_supported',False) is not True or not callable(getattr(owner,'endpoint',None)) or not callable(getattr(owner,'cancel',None)) or not isinstance(getattr(owner,'cancelled',None),threading.Event):
        raise ValueError('Concurrent runner requires one cancellable shared owner with private endpoints')
    settings['provenance']={**(provenance or {}),'model':validate_client(owner),'runtime_repair_version':REPAIR_VERSION}
    owner.notebook_tools_enabled=True;owner.notebook_tool_schemas=settings['notebook_tool_schemas']
    endpoints={a:owner.endpoint(a) for a in AGENTS}
    if len({id(v) for v in endpoints.values()})!=2:raise ValueError('Distinct per-agent request endpoints required')
    policy=TimedPolicy(**settings['policy']);results=[];transitions=[]
    histories={a:reset_base_history(settings['system_prompts'][a],settings['topic']) for a in AGENTS}
    status={'status':'in_progress','prepared':False,'completed_rounds':0,'next_phase_index':0}
    if settings.get('peer_note_exposure_policy'):status['peer_note_deliveries']=[]
    if restored:
        data=restored['data'];results=data['results.json'];transitions=data['transitions.json']
        status.update(data['state.json'])
        histories={a:reset_base_history(settings['system_prompts'][a],settings['topic']) for a in AGENTS} if restored['extension'] else data['histories.json']
        timing_lineage=restored.get('answer_timing_migration') or data['settings.json'].get('resume',{}).get('answer_timing_migration')
        settings['resume']={**({'answer_timing_migration':timing_lineage} if timing_lineage else {}),'verification_start_index':restored['verification_start_index'],'prior_judge_artifacts_preserved':restored['prior_judge_artifacts_preserved'],'scoped_verification_restart':restored['prior_judge_artifacts_preserved'] and not restored['extension'],'policy':RESUME_SCHEMA,'source_checkpoint_sha256':restored['checkpoint_sha256'],'inherited_rounds':status['completed_rounds'],'task_count_migration':restored['extension'],'rng_kv_continuity':False,'parent_source_hashes':data['settings.json']['source_hashes'],'current_source_hashes':settings['source_hashes']}
        for row in results:
            row['log_path']='inherited/'+row['log_path']
            if 'browser_database' in row:
                original=Path(row['browser_database'])
                row.setdefault('original_browser_database',str(original))
                row['browser_database']='inherited/'+str(Path(*original.parts[-2:]) if original.is_absolute() else original)
    path.mkdir(parents=True)
    try:
        if restored:shutil.copytree(restored['parent'],path/'inherited')
        browser=make_browser(settings,pages,path/'wiki.sqlite3');evidence=EvidenceIndex(path)
        if restored:
            with closing(sqlite3.connect(f"file:{restored['checkpoint'] / 'wiki.sqlite3'}?mode=ro",uri=True)) as previous:previous.backup(browser.db)
            browser.views=restored['data']['browser.json']['views'];browser.history_windows=restored['data']['browser.json']['history_windows']
            evidence.emit('completed_round_checkpoint_fork',**settings['resume'],parent_evidence='inherited')
        contexts={}
        for agent in AGENTS:
            diagnostics=path/('private-'+agent);diagnostics.mkdir()
            contexts[agent]=RetainedContextClient(endpoints[agent],policy,diagnostics)
        inherited_count=len(restored['data']['results.json']) if restored else 0
        inherited_rounds=restored['data']['state.json']['completed_rounds'] if restored else 0
        verification_start=restored['verification_start_index'] if restored else 0
        judge_rounds=settings['question_count']-(2 if verification_start==16 else 0)
        def save():
            if settings.get('continuation_policy'):
                remaining=settings['question_count']-inherited_rounds
                status['work_accounting']={'inherited_phase_records':inherited_count,'new_phase_records':len(results)-inherited_count,
                    'inherited_solver_tokens_observed':sum(r.get('generated_tokens_observed') or 0 for r in results[:inherited_count]),
                    'new_solver_tokens_observed':sum(r.get('generated_tokens_observed') or 0 for r in results[inherited_count:]),
                    'new_accounting_complete':all(r.get('token_accounting_complete',False) for r in results[inherited_count:]),
                    'new_solver_token_ceiling':remaining*15360+(19456 if not restored else 0),
                    'new_judge_token_ceiling':judge_rounds*2048,'verification_start_index':verification_start,'original_parent_answers_rejudged':False,'scoped_verification_restart':bool(restored and restored['prior_judge_artifacts_preserved'] and not restored['extension'])}
            for filename,value in [('settings.json',settings),('inputs.json',inputs),('pages.json',pages),('results.json',results),('transitions.json',transitions),('histories.json',histories),('manifest.json',status)]:write_json(path/filename,value)
        def boundary():
            cp=checkpoint(path,browser,histories,settings,inputs,pages,results,transitions,status['prepared'],status['completed_rounds'],peer_deliveries=status.get('peer_note_deliveries'))
            if checkpoint_callback:checkpoint_callback(cp)
        first_stage=True
        def stage(q,index,roles):
            nonlocal snapshots,first_stage
            folder=path/f'round-{q:02d}-stage-{index}';folder.mkdir()
            snapshots=fork_stage(browser,settings,pages,folder,q,index)
            evidence.emit('stage_snapshots_frozen',question_round=q,stage=index,databases={a:str(folder/(a+'.sqlite3')) for a in AGENTS})
            # Transfer each SQLite connection's ownership to exactly one worker thread.
            for local in snapshots.values():
                local.db.close()
                local.db = None
            ready_timeout=settings['coordinator_readiness']['initial_cold_seconds'] if first_stage else policy.initial_readiness_timeout_seconds
            write_json(folder/'shared-readiness.json',{'status':'in_progress','timeout_seconds':ready_timeout})
            try:ready=owner.ensure_ready(timeout=ready_timeout)
            except BaseException as error:
                write_json(folder/'shared-readiness.json',{'status':'failed','timeout_seconds':ready_timeout,'error':f'{type(error).__name__}: {error}'})
                raise
            write_json(folder/'shared-readiness.json',{**ready,'timeout_seconds':ready_timeout})
            first_stage=False
            gate=threading.Barrier(2);outcomes={};base=len(results)
            with ThreadPoolExecutor(max_workers=2,thread_name_prefix='10b-agent') as pool:
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
                    if event['event']=='host_peer_note_exposure':
                        delivery=event['delivery']
                        if any(d['boundary_id']==delivery['boundary_id'] for d in status['peer_note_deliveries']):raise ValueError('Duplicate peer delivery boundary')
                        status['peer_note_deliveries'].append(delivery)
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
                if 'reflection_result_index' in row and 'reflection_phase_index' not in row:
                    row['reflection_phase_index']=row['reflection_result_index']
                    row['reflection_result_index']=result_positions[row['reflection_phase_index']]
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
            if not status['prepared']:
                if job_deadline is not None and time.monotonic()+settings['coordinator_readiness']['initial_cold_seconds']+2*(policy.preparation_seconds+240)+4*policy.initial_readiness_timeout_seconds>=job_deadline:status['status']='job_safety_stop'
                else:
                    stage(0,1,('initial_research','initial_note'));status['prepared']=True;save();boundary()
            if status['prepared']:
                for q in range(status['completed_rounds']+1,settings['question_count']+1):
                    if job_deadline is not None and time.monotonic()+(settings['coordinator_readiness']['initial_cold_seconds'] if first_stage else 0)+(120 if settings.get('peer_note_exposure_policy') else 0)+2*(policy.answer_seconds+policy.reflection_seconds+240)+6*policy.initial_readiness_timeout_seconds>=job_deadline:
                        status['status']='job_safety_stop';break
                    stage(q,2,('answer',));stage(q,3,('reflection','reflection_note'))
                    write_json(path/f'round-{q:02d}-histories-before-reset.json',histories)
                    evidence.emit('question_context_reset',question_round=q,artifact=f'round-{q:02d}-histories-before-reset.json',reflection_notes_verified=True)
                    histories={a:reset_base_history(settings['system_prompts'][a],settings['topic']) for a in AGENTS}
                    status['completed_rounds']=q;save();boundary()
                else:status['status']='complete'
        except BaseException as error:
            try:owner.cancel()
            except BaseException as cancellation:status['cancellation_error']=f'{type(cancellation).__name__}: {cancellation}'
            status.update(status='failed',error=f'{type(error).__name__}: {error}')
            evidence.emit('run_failed',error=status['error'],partial_phase_artifacts_retained=True);raise
        finally:
            status['next_phase_index']=len(results);status['solver_status']=status['status']
            if status['status'] in ('complete','job_safety_stop'):status['status']='verification_pending'
            save();evidence.emit('solver_finished',status=status['solver_status'],completed_rounds=status['completed_rounds'],results_count=len(results))
        status['status']='verification_in_progress';save()
        try:
            if first_stage and restored and inherited_rounds==settings['question_count']:
                cold=settings['coordinator_readiness']['initial_cold_seconds']
                if job_deadline is not None and time.monotonic()+cold+2*judge_rounds*120>=job_deadline:raise TimeoutError('Insufficient job time for cold verification startup and bounded judgments')
                write_json(path/'verification-readiness.json',{'status':'in_progress','timeout_seconds':cold})
                ready=owner.ensure_ready(timeout=cold)
                write_json(path/'verification-readiness.json',{**ready,'timeout_seconds':cold})
                first_stage=False
            judge_settings=dict(settings)
            judge_settings['maximum_verifier_generated_tokens']=judge_rounds*2048
            judge_settings['answer_support_verifier']={**settings['answer_support_verifier'],'maximum_calls':2*judge_rounds}
            status['answer_support']=judge_answers(path,owner,[{}]*verification_start+results[verification_start:],tasks,pages,judge_settings,job_deadline)
        except BaseException as error:
            status.update(status='verification_interrupted',error=f'{type(error).__name__}: {error}');save();raise
        status.update(status=status['solver_status'],answer_support_status='complete');save();return status
    finally:
        for local in snapshots.values():
            if local.db is not None:local.close()
        if browser is not None:browser.close()
