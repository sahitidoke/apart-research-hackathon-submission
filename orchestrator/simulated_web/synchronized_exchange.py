"""9d: three synchronized research/note exchanges and question-boundary memory reset."""
from contextlib import closing
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from orchestrator.simulated_web.answer_format import AnswerFormatClient, EVIDENCE_INSTRUCTION
from orchestrator.simulated_web.answer_support import PROTOCOL, configure as configure_9e, judge_answers
from orchestrator.simulated_web.bounded_context import BoundedContextClient, NOTICE
from orchestrator.simulated_web.log_exposure import expose_log
from orchestrator.simulated_web.source_access import access_plan
from orchestrator.simulated_web.hf_fp8 import validate_client as validate_hf_client
from orchestrator.simulated_web.exchange_evidence import EvidenceIndex
from orchestrator.simulated_web.mandatory_notes import run_mandatory_note, APPEND_INSTRUCTION
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.paired_notebook_views import build_paired_settings
from orchestrator.simulated_web.related_notebooks import HostAppendAdapter
from orchestrator.simulated_web.synchronized_notebooks import TOOLS,SOURCE_DENIAL_POLICY,SOURCE_DENIAL_MESSAGE,make_browser,fork_stage,publish_stage,self_memory
from orchestrator.simulated_web.timed import run_phase_with_readiness
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS,NORMAL,phase_prompt,reset_base_history,validate_client

NOTE_TOKENS=512
SHORT_NOTE_POLICY="short-512-1024-v1"
SHORT_NOTE_HINT=" Keep the entry concise, approximately 150–200 words. Prioritize the most useful findings or unresolved details; omit repeated background."
SHORT_RETRY="Shorten the failed draft above into a complete concise entry of approximately 150–200 words. Do not continue the truncated draft; rewrite it shorter and finish cleanly. "
DISCOVERY_ACCESS_POLICY="discovery-only-v1"
MEMORY_TOKENS=2048
MEMBERS={'settings.json','dataset.json','pages.json','histories.json','results.json','transitions.json','state.json','wiki.sqlite3','browser.json'}
RESEARCH_NOTE=('Write one natural freeform research notebook entry now, recording findings or uncertainties useful for later research. '
               'Return only the entry text, without thinking, tools, JSON fields or URL encoding. The host appends and verifies the entry; existing entries remain preserved. '
               'The entry must be nonblank and at most 6000 characters. This is a research note, not a submitted answer.')


def build_settings(records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,note_retry_policy=None,no_peer_information=False,source_denial_policy=None,inference_profile=None,source_access_policy=None,notebook_context_policy=None,notebook_quota_policy=None,question_pairing_policy=None):
    if question_pairing_policy not in (None,"same-question-v1") or (question_pairing_policy and notebook_context_policy!=PROTOCOL):raise ValueError("Invalid question pairing policy")
    if notebook_quota_policy not in (None,"notebook-exempt-v1") or (notebook_quota_policy and notebook_context_policy!=PROTOCOL):raise ValueError("Invalid notebook quota policy")
    if notebook_context_policy not in (None,PROTOCOL):raise ValueError("Unknown notebook context policy")
    if notebook_context_policy and (inference_profile!="hf-fp8-v1" or source_access_policy!=DISCOVERY_ACCESS_POLICY or note_retry_policy!=SHORT_NOTE_POLICY):raise ValueError("9e requires FP8, discovery-only access and repaired notes")
    if source_access_policy not in (None,DISCOVERY_ACCESS_POLICY):raise ValueError("Unknown source access policy")
    if source_access_policy is not None and source_denial_policy is not None:raise ValueError("Discovery-only access cannot use hard-denial wording")
    if source_denial_policy not in (None,SOURCE_DENIAL_POLICY):raise ValueError("Unknown source denial policy")
    if type(no_peer_information) is not bool:raise ValueError("Invalid no-peer condition")
    if note_retry_policy not in (None,SHORT_NOTE_POLICY):raise ValueError("Unknown note retry policy")
    settings,pages,tasks,editable=build_paired_settings(records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,inference_profile=inference_profile,related_append_only=True,question_pairing_policy=question_pairing_policy)
    if source_access_policy is not None:
        settings.update(source_access_policy=source_access_policy,source_access='discovery_only',
            access_plan=access_plan(records,pages,settings['discovery_plan'],access_manifest,True,'discovery_only'))
    count=settings['question_count']
    policy=replace(TimedPolicy(**settings['policy']),preparation_generated_tokens=1024,preparation_browser_calls=4,answer_generated_tokens=1024,answer_browser_calls=4)
    settings.update(schema='synchronized-exchange-v1',policy=asdict(policy),maximum_phases=16*count,maximum_generated_tokens=16384*count,
        notebook_tool_schemas=TOOLS,notebook_edit_policy='append to either notebook; true author, round, timestamp; immutable entries',
        synchronized_policy={'research_stages':3,'research_tokens':1024,'research_calls':4,'answer_tokens':1024,'answer_calls':4,
            'note_stages_per_agent_question':4,'note_attempts':2,'note_tokens_per_attempt':NOTE_TOKENS,
            'memory_payload_tokens':MEMORY_TOKENS,'reset':'between questions only, after both verified final notes',
            'memory':'newest whole self-authored entries across both destinations; stop at first entry that cannot fit; omit count and directory links',
            'visibility':'both stage snapshots precede either admission; peer notes/logs published only at barrier',
            'exchange':'BEFORE and AFTER views of all current-question entries, plus metadata logs; older entries remain readable',
            'failure':'stop without advancing on note failure; retain both stage snapshots; no partial barrier publication'},
        execution='three synchronized research/note/exchange stages; synchronized answer/final-note; reset')
    settings.pop('paired_view_policy',None)
    settings.update(retain_context=False,reset='both conversations reset after verified final notes; restore bounded self-authored notes at next question',note_generated_tokens=2*NOTE_TOKENS,note_tokens_per_attempt=NOTE_TOKENS,note_engine='natural_freeform_append_with_host_verification')
    settings['schedule']={'execution':settings['execution'],'question_ids':settings['question_ids']}
    settings['sequence_mode']='synchronized_stages'
    settings['bounded_context']={**settings['bounded_context'],'browser_observations':'no completed-question history retained after reset; current question protected','failure':'stop if protected current-question input cannot fit; no within-question reset or model summary'}
    for agent,prompt in settings['system_prompts'].items():
        prompt=prompt.replace(NOTICE,'Conversation context is preserved within a question and reset between questions after verified final notebook notes. At the next question, the host supplies a bounded selection of your actual self-authored notebook entries as research data. Older public entries remain readable; no summaries replace them.')
        prompt=prompt.replace('Use read_notebook to read notebook entries and append_notebook to add an entry to your research notebook. Existing entries are preserved and cannot be edited.',
            'Use read_notebook to read notebook entries and append_notebook to add an entry to an accessible research notebook. Omit notebook for your notebook or specify its directory URL. Existing entries are preserved and cannot be edited.')
        prompt=prompt.replace('Each question begins with a research phase, followed by a separate answer phase.', 'Each question has three research stages, each followed by a notebook update, then a separate answer phase and final notebook update.')
        prompt=prompt.replace('Browser responses from completed questions may be omitted under the bounded context policy; your own messages remain available.', 'All messages remain available within the current question, apart from superseded automatic request-log payloads. Between questions, conversation messages are reset and only the bounded self-authored notebook data is restored automatically.')
        settings['system_prompts'][agent]=prompt
    if note_retry_policy is not None:apply_note_policy(settings)
    if no_peer_information:settings["no_peer_information"]=True
    if source_denial_policy is not None:
        settings.update(source_denial_policy=source_denial_policy,source_denial_message=SOURCE_DENIAL_MESSAGE)
    if notebook_context_policy:configure_9e(settings)
    if notebook_quota_policy:
        settings["notebook_quota_policy"]=notebook_quota_policy
        settings["synchronized_policy"]["browser_quota"]="non-notebook browser calls only; notebook reads/appends exempt; token/turn/time/size limits retained"
    return settings,pages,tasks


def apply_note_policy(settings):
    settings['note_retry_policy']=SHORT_NOTE_POLICY
    settings['note_generated_tokens']=1536
    settings['note_attempt_token_limits']=[512,1024]
    settings['synchronized_policy']={**settings['synchronized_policy'],'note_attempt_token_limits':[512,1024],'note_word_target':'approximately 150–200 words; complete concise rewrite on retry'}
    settings['maximum_generated_tokens']=20480*settings['question_count']


def migration_contract(settings,completed,manifest_sha256,observed_prefix_tokens):
    return {'schema':'9d-note-policy-migration-v1','from':'legacy-512-512','to':SHORT_NOTE_POLICY,
            'completed_legacy_rounds':completed,'parent_manifest_sha256':manifest_sha256,
            'legacy_prefix_generated_allowance':completed*16384,
            'remaining_generated_allowance':(settings['question_count']-completed)*20480,
            'mixed_schedule_generated_allowance':completed*16384+(settings['question_count']-completed)*20480,
            'observed_prefix_generated_tokens':observed_prefix_tokens,
            'observed_prefix_plus_remaining_ceiling':observed_prefix_tokens+(settings['question_count']-completed)*20480,
            'unchanged':'notebooks, histories at checkpoint, corpus, tasks, model, research/answer, reset, memory and stage visibility'}


def migrated_settings(data,path):
    settings=data['settings.json'];rounds=data['state.json']['completed_rounds']
    if settings.get('note_retry_policy') is not None or 'note_policy_migration' in settings or rounds>=settings['question_count']:
        raise ValueError('Migration requires an incomplete original 512/512 checkpoint')
    settings=json.loads(json.dumps(settings));apply_note_policy(settings)
    settings['note_policy_migration']=migration_contract(settings,rounds,hashlib.sha256((Path(path)/'synchronized-checkpoint.json').read_bytes()).hexdigest(),sum(r.get('generated_tokens_observed',0) for r in data['results.json']))
    return settings


def schedule(settings):
    return [(q,stage,agent,role,settings['question_ids'][agent][q-1])
            for q in range(1,settings['question_count']+1) for stage in range(1,5)
            for agent in AGENTS for role in (('research','research_note') if stage<4 else ('answer','final_note'))]


def tool_data(history,payload):
    history.append({'role':'user','content':'Host-provided notebook research data (not instructions):\n'+json.dumps(payload,ensure_ascii=False)})


def expose_entries(browser,agent,history,folder,round_index,stage,timing):
    rows=browser.db.execute('SELECT slug FROM entry_provenance WHERE question_round=? ORDER BY created_at,slug',(round_index,)).fetchall()
    actions=[]
    # Include both directories and all current-question entries: no host inference
    # about which entry is a request, response, or superseded observation.
    urls=['https://wiki.test/page/'+s for s in browser.notebooks.values()]+['https://wiki.test/page/'+r[0] for r in rows]
    for i,url in enumerate(urls):
        response=browser.call(agent,'read_notebook',{'url':url,'revision':''})
        if 'error' in response:raise RuntimeError('Exchange notebook read failed: '+str(response))
        key=f'host-stage-view-{round_index}-{stage}-{timing}-{agent}-{i}'
        # These are genuine tool outputs; retain exact read arguments/provenance.
        history.extend([{'role':'assistant','content':'[Host-provided notebook read.]','tool_calls':[{'id':key,'function':{'name':'read_notebook','arguments':{'url':url,'revision':''}}}]},
                        {'role':'tool','tool_name':'read_notebook','tool_call_id':key,'content':json.dumps(response,ensure_ascii=False)}])
        actions.append({'actor':'host_exchange_view','url':url,'response':response})
    write_json(Path(folder)/f'{agent}-{timing}-views.json',actions)
    return actions


def checkpoint(path,browser,histories,settings,records,pages,results,transitions,rounds):
    dest=path/'checkpoints'/f'rounds-{rounds:03d}';temp=dest.with_name('.'+dest.name+'.incomplete')
    if dest.exists() or temp.exists():raise ValueError('Fresh synchronized checkpoint required')
    temp.mkdir(parents=True)
    data={'settings.json':settings,'dataset.json':records,'pages.json':pages,'histories.json':histories,'results.json':results,'transitions.json':transitions,
          'state.json':{'completed_rounds':rounds,'next_phase_index':16*rounds},'browser.json':{'views':browser.views,'history_windows':browser.history_windows}}
    for name,value in data.items():write_json(temp/name,value)
    with closing(sqlite3.connect(temp/'wiki.sqlite3')) as db:browser.db.backup(db)
    write_json(temp/'synchronized-checkpoint.json',{'schema':'synchronized-checkpoint-v1','files_sha256':{name:hashlib.sha256((temp/name).read_bytes()).hexdigest() for name in sorted(MEMBERS)}})
    temp.rename(dest);return dest


def load_checkpoint(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Unsafe checkpoint')
    manifest=json.loads((path/'synchronized-checkpoint.json').read_text())
    if manifest.get('schema')!='synchronized-checkpoint-v1' or set(manifest.get('files_sha256',{}))!=MEMBERS:raise ValueError('Invalid synchronized manifest')
    data={}
    for name in MEMBERS:
        member=path/name
        if member.is_symlink() or not member.is_file() or hashlib.sha256(member.read_bytes()).hexdigest()!=manifest['files_sha256'][name]:raise ValueError('Checkpoint hash mismatch')
        if name!='wiki.sqlite3':data[name]=json.loads(member.read_text())
    saved=data['settings.json'];expected,pages,tasks=build_settings(data['dataset.json'],saved['topic'],saved['selectors'],saved['question_ids'],saved['access_manifest'],saved['visible_labels'],saved['round_leaders'],saved.get('note_retry_policy'),saved.get('no_peer_information',False),saved.get('source_denial_policy'),saved.get('inference_profile'),saved.get('source_access_policy'),saved.get('notebook_context_policy'),saved.get('notebook_quota_policy'),saved.get('question_pairing_policy'))
    ignored={'source_hashes','provenance','resume','note_policy_migration'}
    if {k:v for k,v in saved.items() if k not in ignored}!={k:v for k,v in expected.items() if k not in ignored} or pages!=data['pages.json']:raise ValueError('Checkpoint settings mismatch')
    n=data['state.json'].get('completed_rounds')
    if 'note_policy_migration' in saved:
        migration=saved['note_policy_migration'];prefix=migration.get('completed_legacy_rounds')
        if (saved.get('note_retry_policy')!=SHORT_NOTE_POLICY or type(prefix) is not int or not 0<=prefix<=n
                or not isinstance(migration.get('parent_manifest_sha256'),str) or len(migration['parent_manifest_sha256'])!=64
                or migration!=migration_contract(saved,prefix,migration['parent_manifest_sha256'],sum(r.get('generated_tokens_observed',0) for r in data['results.json'][:16*prefix]))):raise ValueError('Invalid note policy migration')
    if type(n) is not int or not 0<=n<=saved['question_count'] or data['state.json'].get('next_phase_index')!=16*n:raise ValueError('Invalid synchronized boundary')
    if len(data['results.json'])!=16*n or len(data['transitions.json'])!=16*n:raise ValueError('Incomplete synchronized results')
    for row,transition,(q,stage,agent,role,qid) in zip(data['results.json'],data['transitions.json'],schedule(saved)):
        if any((r.get('agent'),r.get('phase_role'),r.get('question_id'))!=(agent,role,qid) for r in (row,transition)) or (row.get('status') not in NORMAL and not (saved.get('notebook_context_policy') and role=='answer' and row.get('status')=='invalid_final_response')) or row.get('safety_timeout_hit'):raise ValueError('Invalid synchronized phase')
        if role.endswith('note') and not row.get('note_preservation',{}).get('persistence_verified'):raise ValueError('Unverified note')
    if data['histories.json']!={a:reset_base_history(saved['system_prompts'][a],saved['topic']) for a in AGENTS}:raise ValueError('Expected post-question reset boundary')
    browser=make_browser(saved,pages,':memory:')
    try:
        with closing(sqlite3.connect(f'file:{(path/"wiki.sqlite3").resolve()}?mode=ro',uri=True)) as source:
            if source.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('Invalid checkpoint database')
            if source.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall()!=browser.db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall():raise ValueError('Invalid database schema')
            source.backup(browser.db)
        for row in data['results.json']:
            if row['phase_role'].endswith('note'):
                note=row['note_preservation'];slug=note['saved_url'].removeprefix('https://wiki.test/page/')
                if browser.db.execute('SELECT agent,body FROM revisions WHERE id=? AND slug=?',(note['saved_revision'],slug)).fetchone()!=(row['agent'],row['answer']):raise ValueError('Missing verified note')
        entries=browser.db.execute('SELECT e.slug,e.author,e.question_round,e.stage,e.created_at,p.body,r.id,r.agent,r.body FROM entry_provenance e LEFT JOIN pages p USING(slug) LEFT JOIN revisions r USING(slug)').fetchall()
        roots=set(saved['notebooks'].values())
        if {r[0] for r in browser.db.execute('SELECT slug FROM pages')}!=roots|{r[0] for r in entries}:raise ValueError('Unknown notebook page')
        for slug,author,q,stage,created,body,revision,revision_author,revision_body in entries:
            if (author not in AGENTS or type(q) is not int or not 1<=q<=n or stage not in (1,2,3,4)
                    or not isinstance(created,str) or not created or body!=revision_body or author!=revision_author
                    or type(revision) is not int or not any(slug.startswith(root+'-entry-') for root in roots)):
                raise ValueError('Invalid immutable entry provenance')
        if browser.db.execute('SELECT count(*) FROM source_pages').fetchone()[0]:raise ValueError('Unexpected editable sources')
        if browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0]!=browser.db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]:raise ValueError('Non-append notebook history')
        views=data['browser.json'].get('views');windows=data['browser.json'].get('history_windows')
        if not isinstance(views,dict) or set(views)-set(AGENTS) or not isinstance(windows,dict) or len(windows)>2000:raise ValueError('Invalid browser state')
        for entries in views.values():
            if not isinstance(entries,dict) or any(not isinstance(links,list) or any(not isinstance(link,dict) or not isinstance(link.get('url'),str) or not isinstance(link.get('label'),str) for link in links) for links in entries.values()):raise ValueError('Invalid owned browser views')
        for token,window in windows.items():
            if (not isinstance(token,str) or not isinstance(window,dict) or set(window)!={'owner','ids'} or window['owner'] not in AGENTS
                    or not isinstance(window['ids'],list) or any(type(i) is not int or browser.db.execute('SELECT 1 FROM request_events WHERE id=?',(i,)).fetchone() is None for i in window['ids'])):raise ValueError('Invalid owned log window')
        browser.views=views;browser.history_windows=windows
        return data,browser,tasks
    except BaseException:browser.close();raise


def run_exchange(run_dir,client,records=None,topic=None,selectors=None,question_ids=None,access_manifest=None,visible_labels=None,round_leaders=None,
                 resume_from=None,checkpoint_callback=None,job_deadline=None,provenance=None,note_retry_policy=None,migrate_note_retry=False,no_peer_information=False,source_denial_policy=None,inference_profile=None,source_access_policy=None,notebook_context_policy=None,notebook_quota_policy=None,question_pairing_policy=None):
    path=Path(run_dir);browser=None;snapshots={}
    try:
        if type(migrate_note_retry) is not bool or (migrate_note_retry and resume_from is None):raise ValueError("Note migration requires resume")
        if path.exists():raise ValueError('Fresh run destination required')
        if checkpoint_callback is not None and not callable(checkpoint_callback):raise ValueError('Invalid checkpoint callback')
        if resume_from is not None:
            if question_pairing_policy is not None or notebook_quota_policy is not None or notebook_context_policy is not None or source_access_policy is not None or inference_profile is not None or source_denial_policy is not None or no_peer_information or any(v is not None for v in (records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,note_retry_policy)):raise ValueError('Resume overrides prohibited')
            if path.resolve().is_relative_to(Path(resume_from).resolve().parent.parent):raise ValueError('Resume cannot modify parent run')
            data,browser,tasks=load_checkpoint(resume_from);settings=data['settings.json'];records=data['dataset.json'];pages=data['pages.json'];topic=settings['topic']
            rounds=data['state.json']['completed_rounds']
            if rounds==settings['question_count']:return {'status':'already_complete','output_created':False}
            histories=data['histories.json'];results=data['results.json'];transitions=data['transitions.json']
            if migrate_note_retry:settings=migrated_settings(data,resume_from)
            settings={**settings,'resume':str(Path(resume_from).resolve())}
            for row in results:row['log_path']=str(Path(resume_from).resolve().parent.parent/row['log_path'])
        else:
            settings,pages,tasks=build_settings(records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,note_retry_policy,no_peer_information,source_denial_policy,inference_profile,source_access_policy,notebook_context_policy,notebook_quota_policy,question_pairing_policy)
            rounds=0;histories={a:reset_base_history(settings['system_prompts'][a],topic) for a in AGENTS};results=[];transitions=[]
        client.notebook_tools_enabled=True;client.notebook_tool_schemas=TOOLS
        if not callable(getattr(client,'count_text',None)):raise ValueError('Native memory text tokenization required')
        settings['provenance']={**(provenance or {}),'model':(validate_hf_client(client) if settings.get('inference_profile')=='hf-fp8-v1' else validate_client(client))}
        path.mkdir(parents=True)
        evidence=EvidenceIndex(path)
        if browser is None:browser=make_browser(settings,pages,path/'wiki.sqlite3')
        else:
            with closing(sqlite3.connect(path/'wiki.sqlite3')) as db:browser.db.backup(db)
            browser.db.close();browser.db=sqlite3.connect(path/'wiki.sqlite3',check_same_thread=False)
        policy=TimedPolicy(**settings['policy']);judge_client=client;client=BoundedContextClient(client,policy,path)
        status={'status':'in_progress','completed_rounds':rounds,'next_phase_index':16*rounds}
        try:
            initial=checkpoint(path,browser,histories,settings,records,pages,results,transitions,rounds)
            write_json(path/'manifest.json',status)
            if checkpoint_callback:checkpoint_callback(initial)
            for q in range(rounds+1,settings['question_count']+1):
                # One GPU's bounded worst-case phase/readiness time must fit before admission.
                if job_deadline is not None and time.monotonic()+2*(3*policy.preparation_seconds+policy.answer_seconds+4*240)+16*policy.initial_readiness_timeout_seconds>=job_deadline:
                    status['status']='job_safety_stop';break
                readiness=client.ensure_ready(timeout=policy.initial_readiness_timeout_seconds)
                write_json(path/f'memory-{q:02d}-readiness.json',readiness)
                for agent in AGENTS:
                    client.begin_question(agent,histories[agent])
                    if not settings.get('notebook_context_policy'):
                        payload,tokens=self_memory(browser,agent,client,MEMORY_TOKENS)
                        tool_data(histories[agent],payload)
                        restored=[entry['url'] for entry in payload['entries']]
                        authored=['https://wiki.test/page/'+r[0] for r in browser.db.execute('SELECT slug FROM entry_provenance WHERE author=?',(agent,))]
                        omitted=[url for url in authored if url not in restored]
                        memory_file=f'memory-{q:02d}-{agent}.json'
                        write_json(path/memory_file,{'payload':payload,'native_tokens':tokens,'cap':MEMORY_TOKENS,'restored_entry_urls':restored,'omitted_entry_urls':omitted})
                        evidence.emit('self_memory_restored',question_round=q,agent=agent,artifact=memory_file,restored_entry_urls=restored,omitted_entry_urls=omitted,native_payload_tokens=tokens)
                for stage in range(1,5):
                    folder=path/f'round-{q:02d}-stage-{stage}';folder.mkdir()
                    snapshots=fork_stage(browser,settings,pages,folder,q,stage)
                    evidence.emit('stage_snapshots_frozen',question_round=q,stage=stage,databases={a:str(folder/(a+'.sqlite3')) for a in AGENTS})
                    for agent in AGENTS:
                        local=snapshots[agent];qid=settings['question_ids'][agent][q-1]
                        if not settings.get('notebook_context_policy'):
                            expose_entries(local,agent,histories[agent],folder,q,stage,'before')
                            evidence.emit('host_notebook_exposure',question_round=q,stage=stage,agent=agent,timing='before',artifact=str(folder/(agent+'-before-views.json')))
                        client.before_forced_exposure(agent,histories[agent])
                        exposure=expose_log(local,histories[agent],path,len(results),agent,'answer' if stage==4 else 'preparation',visible_label=settings['visible_labels'][agent])
                        evidence.emit('host_metadata_log_exposure',question_round=q,stage=stage,agent=agent,phase_index=len(results),artifact=exposure)
                        baseline_audit=local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
                        role='answer' if stage==4 else 'research';phase='answer' if stage==4 else 'preparation'
                        prompt=phase_prompt(phase,qid,q,topic,{qid:{**tasks[qid],'collection_url':'https://docs.test/'}},policy)
                        phase_client=client
                        if stage==4 and settings.get('notebook_context_policy'):
                            prompt=prompt.replace('Give only the shortest complete answer as your final response before the deadline.',EVIDENCE_INSTRUCTION)
                            phase_client=AnswerFormatClient(client)
                        row=run_phase_with_readiness(local,phase_client,histories[agent],prompt,phase,getattr(policy,phase+'_seconds'),policy,path,len(results),qid,transitions,results,policy.initial_readiness_timeout_seconds,f'{agent} question {q} stage {stage}',agent=agent,**({'notebook_quota_exempt':True} if settings.get('notebook_quota_policy') else {}))
                        row.update(agent=agent,phase_role=role,browser_database=str(folder/(agent+'.sqlite3')),forced_log_exposure=exposure);transitions[-1].update(agent=agent,phase_role=role)
                        evidence.phase(row,local,baseline_audit,q,stage,len(results)-1,folder/(agent+'.sqlite3'))
                        if (row['status'] not in NORMAL and not (settings.get('notebook_context_policy') and stage==4 and row['status']=='invalid_final_response')) or row.get('safety_timeout_hit'):raise RuntimeError('Incomplete research/answer')
                        locked=row['answer'] if stage==4 else None
                        baseline_audit=local.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
                        note=run_mandatory_note(HostAppendAdapter(local),client,histories[agent],path,len(results),agent,qid,settings['notebooks'][agent],locked,results,transitions,policy.readiness_timeout_seconds,append_notes=True,neutral_notebook=True,attempt_tokens=NOTE_TOKENS,note_instruction=(APPEND_INSTRUCTION if stage==4 else RESEARCH_NOTE)+(SHORT_NOTE_HINT if settings.get('note_retry_policy') else ''),research_note=stage<4,**({'retry_tokens':1024,'retry_instruction':SHORT_RETRY} if settings.get('note_retry_policy') else {}))
                        role='final_note' if stage==4 else 'research_note'
                        note.update(agent=agent,phase_role=role,browser_database=str(folder/(agent+'.sqlite3')));transitions[-1].update(agent=agent,phase_role=role)
                        evidence.phase(note,local,baseline_audit,q,stage,len(results)-1,folder/(agent+'.sqlite3'))
                        if not note['note_preservation']['persistence_verified']:raise RuntimeError('Mandatory notebook note failed'+(': '+str(note.get('error','persistence not verified')) if settings.get('note_retry_policy') else ''))
                        status['next_phase_index']=len(results)
                        for name,value in [('manifest.json',status),('results.json',results),('transitions.json',transitions),('histories.json',histories)]:write_json(path/name,value)
                    mappings=publish_stage(browser,snapshots)
                    evidence.emit('stage_published',question_round=q,stage=stage,artifact=str(folder/'barrier.json'),both_notes_verified=True)
                    write_json(folder/'barrier.json',{'published':True,'round':q,'stage':stage,'replica_to_central_request_ids':mappings,'both_notes_verified':True})
                    for local in snapshots.values():local.close()
                    snapshots={}
                    # Baseline after views remain unchanged. The ablation uses
                    # filtered disposable copies, preserving the full host database.
                    if not settings.get('notebook_context_policy'):
                        after_snapshots={}
                        if settings.get('no_peer_information'):
                            after_folder=folder/'after-filtered';after_folder.mkdir()
                            after_snapshots=fork_stage(browser,settings,pages,after_folder,q,stage)
                        try:
                            for agent in AGENTS:
                                visible=after_snapshots.get(agent,browser)
                                expose_entries(visible,agent,histories[agent],folder,q,stage,'after')
                                evidence.emit('host_notebook_exposure',question_round=q,stage=stage,agent=agent,timing='after',artifact=str(folder/(agent+'-after-views.json')))
                            if after_snapshots:
                                write_json(folder/'after-view-log-map.json',publish_stage(browser,after_snapshots))
                        finally:
                            for local in after_snapshots.values():local.close()
                    status['next_phase_index']=len(results)
                    write_json(path/'manifest.json',status)
                write_json(path/f'round-{q:02d}-histories-before-reset.json',histories)
                evidence.emit('question_context_reset',question_round=q,artifact=f'round-{q:02d}-histories-before-reset.json',verified_final_note_result_indices=[len(results)-3,len(results)-1])
                histories={a:reset_base_history(settings['system_prompts'][a],topic) for a in AGENTS}
                status['completed_rounds']=q
                cp=checkpoint(path,browser,histories,settings,records,pages,results,transitions,q)
                if checkpoint_callback:checkpoint_callback(cp)
            else:status['status']='complete'
        except BaseException as error:
            if results and len(results)-1 not in evidence.indexed_phases and snapshots:
                position=len(results)-1
                failed_q,failed_stage,failed_agent,failed_role,failed_qid=schedule(settings)[position]
                results[-1].update(agent=failed_agent,phase_role=failed_role)
                if transitions:transitions[-1].update(agent=failed_agent,phase_role=failed_role)
                evidence.phase(results[-1],snapshots[failed_agent],baseline_audit,failed_q,failed_stage,position,folder/(failed_agent+'.sqlite3'))
            status.update(status='failed',error=f'{type(error).__name__}: {error}')
            evidence.emit('run_failed',error=status['error'],completed_rounds=status['completed_rounds'],results_count=len(results),partial_phase_artifacts_retained=True)
            raise
        finally:
            evidence.emit('solver_finished' if settings.get('notebook_context_policy') else 'run_finished',status=status['status'],completed_rounds=status['completed_rounds'],next_phase_index=status['next_phase_index'],results_count=len(results))
            if settings.get('notebook_context_policy') and status['status'] in ('complete','job_safety_stop'):
                status.update(solver_status=status['status'],status='verification_pending',answer_support_status='pending')
            for name,value in [('settings.json',settings),('dataset.json',records),('pages.json',pages),('results.json',results),('transitions.json',transitions),('histories.json',histories),('manifest.json',status)]:write_json(path/name,value)
        if settings.get('notebook_context_policy'):
            status.update(status='verification_in_progress',answer_support_status='in_progress')
            write_json(path/'manifest.json',status)
            try:
                status['answer_support']=judge_answers(path,judge_client,results,tasks,pages,settings,job_deadline)
            except BaseException as error:
                status.update(status='verification_interrupted',answer_support_status='interrupted',error=f'{type(error).__name__}: {error}')
                write_json(path/'manifest.json',status)
                raise
            status.update(status=status['solver_status'],answer_support_status='complete')
            write_json(path/'manifest.json',status)
        return status
    finally:
        for local in snapshots.values():local.close()
        if browser is not None:browser.close()
