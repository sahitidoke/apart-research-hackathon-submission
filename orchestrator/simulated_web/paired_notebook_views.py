"""Paired genuine notebook views around a writer turn, with a split reader research budget."""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import re
from pathlib import Path
import shutil
import sqlite3
import time

from orchestrator.simulated_web.related_notebooks import configure_related, related_browser, HostAppendAdapter
from orchestrator.simulated_web.append_notebooks import entry_owner
from orchestrator.simulated_web.bounded_context import BoundedContextClient
from orchestrator.simulated_web.log_exposure import expose_log
from orchestrator.simulated_web.hf_fp8 import validate_client as validate_hf_client
from orchestrator.simulated_web.mandatory_notes import run_mandatory_note,SECONDS as NOTE_SECONDS
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.private_notes import build_settings,PersonalNotebookBrowser,MEMBERS
from orchestrator.simulated_web.source_access import access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery
from orchestrator.simulated_web.sqlite_snapshot import snapshot_connection
from orchestrator.simulated_web.timed import run_phase_with_readiness
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS,NORMAL,phase_prompt,reset_base_history,validate_client,search_browser_options

VIEW_POLICY={'before':'notebook directory and latest existing entry','after':'same directory/prior entry and verified new append',
             'research_split':{'blocks':2,'generated_tokens_each':1024,'browser_calls_each':2},
             'host_reads_outside_research_allowance':True,'reader_context':'continuous; both views protected through answer/note',
             'permission_notice':'You may edit any accessible notebook entry.'}
TOOLS=json.loads(json.dumps(NOTEBOOK_TOOLS))
for tool in TOOLS:
    if tool['function']['name']=='edit_notebook':tool['function']['description']='You may edit any accessible notebook entry. '+tool['function']['description']


class PairedBrowser(PersonalNotebookBrowser):
    def call(self,agent,operation,args):
        if operation=='edit_notebook' and isinstance(args,dict) and args.get('url') in ['https://wiki.test/page/'+s for s in self.notebooks.values()]:
            result=super().call(agent,operation,args)
            slug=args['url'].removeprefix('https://wiki.test/page/')
            links=[{'url':'https://wiki.test/page/'+entry,'revision':f'r-{revision}'} for entry,revision in self.db.execute(
                'SELECT p.slug,max(r.id) FROM pages p JOIN revisions r ON r.slug=p.slug WHERE p.slug GLOB ? GROUP BY p.slug ORDER BY p.slug DESC LIMIT 5',(slug+'-entry-*',))]
            result={**result,'error':'This URL is a notebook directory. You may edit any accessible notebook entry; use an entry URL and its current revision.',
                    'entries':links,'entries_order':'latest five; read the directory to navigate all entries'}
            # Persist the exact response actually returned, preserving the rejected request's real audit/event.
            self.db.execute('UPDATE audit SET result=? WHERE id=(SELECT max(id) FROM audit)',(json.dumps(result),));self.db.commit()
            return result
        return super().call(agent,operation,args)


def build_paired_settings(records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,inference_profile=None,related_append_only=False,question_pairing_policy=None):
    if question_pairing_policy not in (None,"same-question-v1") or (question_pairing_policy and not related_append_only):raise ValueError("Invalid question pairing policy")
    if inference_profile not in (None,'hf-fp8-v1'):raise ValueError('Unknown paired inference profile')
    count=len(round_leaders) if isinstance(round_leaders,list) else 0
    base,pages,tasks,editable=build_settings(records,topic,selectors,question_ids,access_manifest,sequence_mode='agent_serial',
        source_access_mode='hard' if related_append_only else 'discovery_only',history_search=True,log_exposure='forced',retain_context=True,mandatory_notes=True,
        question_count=count,bounded_context=True,append_notes=True,neutral_notebook=True,visible_labels=visible_labels,
        round_leaders=round_leaders,notebook_tools=True)
    if not isinstance(visible_labels,dict):raise ValueError('Visible identities required')
    if not related_append_only and base['question_ids']['agent-1']!=base['question_ids']['agent-2']:raise ValueError('Paired views require identical questions')
    base.update(schema='paired-notebook-views-v1',paired_view_policy=VIEW_POLICY,maximum_phases=7*count,
                execution='reader research1; writer research/answer/note; reader research2/answer/note',
                notebook_tool_schemas=TOOLS)
    for agent in AGENTS:base['system_prompts'][agent]+='\nYou may edit any accessible notebook entry.'
    if inference_profile is not None:base['inference_profile']=inference_profile
    if related_append_only:configure_related(base,question_pairing_policy)
    return base,pages,tasks,editable


def browser_for(settings,pages,editable,database):
    if settings.get("related_append_only"):return related_browser(settings,pages,database)
    return PairedBrowser(pages,database,append_notes=True,notebooks=settings['notebooks'],neutral_notebook=True,
        visible_labels=settings['visible_labels'],notebook_tools=True,editable_sources=editable,search_snippets=False,
        editable_title_marker=True,request_history_mode='shared',history_search=True,**browser_discovery(settings['discovery_plan']),
        **search_browser_options(settings['discovery_plan'],'distinct_sources_5'),**access_browser_options(settings['access_plan']))


def sequence(settings):
    items=[]
    for slot,writer in enumerate(settings['round_leaders']):
        reader=next(a for a in AGENTS if a!=writer);qid=settings['question_ids'][writer][slot]
        items.extend((agent,role,settings['question_ids'][agent][slot],slot+1,writer,reader) for agent,role in
                     [(reader,'research1'),(writer,'research'),(writer,'answer'),(writer,'note'),
                      (reader,'research2'),(reader,'answer'),(reader,'note')])
    return items


def save_checkpoint(run_dir,browser,histories,state):
    destination=run_dir/'checkpoints'/f'rounds-{state["completed_rounds"]:03d}';staging=destination.with_name('.'+destination.name+'.incomplete')
    if destination.exists() or staging.exists():raise ValueError('Fresh paired checkpoint boundary required')
    sources=MEMBERS-{'wiki.sqlite3','browser.json','histories.json','state.json'}
    if any(not (run_dir/n).is_file() for n in sources):raise ValueError('Missing checkpoint prerequisite')
    staging.mkdir(parents=True)
    for name in sources:shutil.copyfile(run_dir/name,staging/name)
    write_json(staging/'histories.json',histories);write_json(staging/'state.json',state)
    with closing(sqlite3.connect(staging/'wiki.sqlite3')) as db:browser.db.backup(db)
    write_json(staging/'browser.json',{'views':browser.views,'history_windows':browser.history_windows})
    write_json(staging/'paired-view-checkpoint.json',{'schema':'paired-view-checkpoint-v1',
        'files_sha256':{name:hashlib.sha256((staging/name).read_bytes()).hexdigest() for name in sorted(MEMBERS)}})
    staging.rename(destination);return destination


def load_checkpoint(path):
    path=Path(path);manifest_path=path/'paired-view-checkpoint.json'
    if path.is_symlink() or manifest_path.is_symlink():raise ValueError('Unsafe paired checkpoint')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='paired-view-checkpoint-v1' or set(manifest.get('files_sha256',{}))!=MEMBERS:raise ValueError('Invalid paired checkpoint manifest')
    data={}
    for name in MEMBERS:
        source=path/name
        if source.is_symlink() or not source.is_file():raise ValueError('Unsafe checkpoint member')
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['files_sha256'][name]:raise ValueError('Checkpoint hash mismatch')
        data[name]=raw if name.endswith('.sqlite3') else json.loads(raw)
    settings=data['settings.json'];expected,pages,_,editable=build_paired_settings(data['dataset.json'],settings['topic'],settings['selectors'],settings['question_ids'],settings['access_manifest'],settings['visible_labels'],settings['round_leaders'],settings.get('inference_profile'),settings.get('related_append_only',False),settings.get('question_pairing_policy'))
    ignored={'source_hashes','provenance','resume'}
    if {k:v for k,v in settings.items() if k not in ignored}!={k:v for k,v in expected.items() if k not in ignored} or pages!=data['pages.json']:raise ValueError('Paired checkpoint configuration mismatch')
    state=data['state.json'];n=state.get('completed_rounds')
    if type(n) is not int or not 0<=n<=len(settings['round_leaders']) or state.get('next_phase_index')!=7*n:raise ValueError('Invalid paired progress')
    if len(data['results.json'])!=7*n or len(data['transitions.json'])!=7*n:raise ValueError('Invalid paired phase count')
    for row,transition,(agent,role,qid,slot,_,_) in zip(data['results.json'],data['transitions.json'],sequence(settings)):
        if any((item.get('agent'),item.get('phase_role'),item.get('question_id'),item.get('phase'))!=(agent,role,qid,'answer' if role=='answer' else 'preparation') for item in (row,transition)) or row.get('status') not in NORMAL or row.get('safety_timeout_hit'):raise ValueError('Invalid paired phase')
        if role=='note' and not row.get('note_preservation',{}).get('persistence_verified'):raise ValueError('Unverified paired note')
    histories=data['histories.json']
    if set(histories)!=set(AGENTS) or any(not isinstance(h,list) or h[:2]!=reset_base_history(settings['system_prompts'][a],settings['topic']) for a,h in histories.items()):raise ValueError('Invalid paired histories')
    browser=browser_for(settings,pages,editable,':memory:')
    try:
        with snapshot_connection(data['wiki.sqlite3']) as db:
            if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)] or db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall()!=browser.db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall():raise ValueError('Invalid paired database')
            if ({r[0] for r in db.execute('SELECT identity FROM source_pages')}!=set(browser.source_urls)
                    or not set(settings['notebooks'].values()).issubset({r[0] for r in db.execute('SELECT slug FROM pages')})
                    or any(slug not in settings['notebooks'].values() and entry_owner(slug,settings['notebooks']) is None for slug, in db.execute('SELECT slug FROM pages'))
                    or any(agent not in AGENTS for agent, in db.execute('SELECT agent FROM revisions'))):raise ValueError('Invalid paired notebook ownership')
            for row in data['results.json']:
                if row['phase_role']=='note':
                    saved=row['note_preservation'];slug=saved['saved_url'].removeprefix('https://wiki.test/page/')
                    if (entry_owner(slug,settings['notebooks'])!=row['agent']
                            or db.execute('SELECT min(id) FROM revisions WHERE slug=?',(slug,)).fetchone()[0]!=saved['saved_revision']
                            or db.execute('SELECT agent,body FROM revisions WHERE slug=? AND id=?',(slug,saved['saved_revision'])).fetchone()!=(row['agent'],row['answer'])
                            or db.execute('SELECT body FROM pages WHERE slug=?',(slug,)).fetchone()!=db.execute('SELECT body FROM revisions WHERE slug=? ORDER BY id DESC LIMIT 1',(slug,)).fetchone()):raise ValueError('Missing original/current paired note')
            db.backup(browser.db)
        views=data['browser.json'].get('views');windows=data['browser.json'].get('history_windows')
        if not isinstance(views,dict) or set(views)-set(AGENTS) or not isinstance(windows,dict) or len(windows)>2000:raise ValueError('Invalid browser owners/windows')
        for entries in views.values():
            if not isinstance(entries,dict) or len(entries)>2000 or set(entries)!={f'p{i+1}' for i in range(len(entries))}:raise ValueError('Invalid browser views')
            if any(not isinstance(links,list) or any(not isinstance(link,dict) or not isinstance(link.get('url'),str) or not isinstance(link.get('label'),str) for link in links) for links in entries.values()):raise ValueError('Invalid browser links')
        for token,window in windows.items():
            if (not isinstance(token,str) or re.fullmatch('[0-9a-f]{32}',token) is None or not isinstance(window,dict)
                    or set(window)!={'owner','ids'} or window['owner'] not in AGENTS or not isinstance(window['ids'],list)
                    or any(type(i) is not int or i<1 for i in window['ids']) or window['ids']!=sorted(set(window['ids']),reverse=True)
                    or any(browser.db.execute('SELECT 1 FROM request_events WHERE id=?',(i,)).fetchone() is None for i in window['ids'])):raise ValueError('Invalid history window')
        browser.views=views;browser.history_windows=windows
        return data,browser
    except BaseException:browser.close();raise


def show_view(browser,history,run_dir,index,round_index,reader,writer,settings,timing,before=None,new_note=None):
    root='https://wiki.test/page/'+settings['notebooks'][writer]
    entry_urls=['https://wiki.test/page/'+r[0] for r in browser.db.execute('SELECT slug FROM pages WHERE slug GLOB ? ORDER BY slug',(settings['notebooks'][writer]+'-entry-*',))]
    prior=entry_urls[-1] if timing=='before' and entry_urls else before.get('prior_entry') if before else None
    urls=[root]+([prior] if prior else [])
    if new_note is not None:
        saved=new_note['note_preservation'];new_url=saved['saved_url']
        if not saved.get('persistence_verified') or new_url in before['entry_urls_before']:raise ValueError('Expected verified newly appended entry')
        urls.append(new_url)
    artifact={'timing':timing,'round':round_index,'reader':reader,'writer':writer,'notebook':root,'prior_entry':prior,
              'entry_urls_before':entry_urls if timing=='before' else before['entry_urls_before'],'host_actions':[]}
    history.append({'role':'user','content':'The following notebook reads were supplied by the host.'})
    for offset,url in enumerate(urls):
        start=browser.db.execute('SELECT coalesce(max(id),0) FROM request_events').fetchone()[0]
        response=browser.call(reader,'read_notebook',{'url':url,'revision':''})
        artifact['last_read']={'url':url,'response':response}
        write_json(run_dir/f'paired-view-{round_index:02d}-{timing}.json',artifact)
        if 'error' in response:raise RuntimeError('Automatic notebook read failed: '+str(response['error']))
        if new_note is not None and url==new_note['note_preservation']['saved_url']:
            if response.get('text')!=new_note['answer'] or response.get('revision')!=f"r-{new_note['note_preservation']['saved_revision']}" or response.get('author')!=settings['visible_labels'][writer]:raise RuntimeError('New note view verification failed')
        call_id=f'host-notebook-view-{index:02d}-{offset}-{settings["visible_labels"][reader]}'
        history.extend([{'role':'assistant','content':'[Host-provided notebook read.]','tool_calls':[{'id':call_id,'function':{'name':'read_notebook','arguments':{'url':url,'revision':''}}}]},
                        {'role':'tool','tool_name':'read_notebook','tool_call_id':call_id,'content':json.dumps(response)}])
        artifact['host_actions'].append({'actor':'host_paired_notebook_view','url':url,'response':response,'history_message_index':len(history)-1,
            'request_event_ids':[r[0] for r in browser.db.execute('SELECT id FROM request_events WHERE id>? ORDER BY id',(start,))]})
        write_json(run_dir/f'paired-view-{round_index:02d}-{timing}.json',artifact)
    return artifact


def run_paired_views(run_dir,client,records=None,topic=None,selectors=None,question_ids=None,access_manifest=None,
                     visible_labels=None,round_leaders=None,resume_from=None,checkpoint_callback=None,job_deadline=None,provenance=None,inference_profile=None,related_append_only=False):
    run_dir=Path(run_dir);browser=None
    try:
        if run_dir.exists():raise ValueError('Fresh destination required')
        if checkpoint_callback is not None and not callable(checkpoint_callback):raise ValueError('Invalid checkpoint callback')
        if resume_from is not None:
            if related_append_only or any(v is not None for v in (records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,inference_profile)):raise ValueError('Resume overrides prohibited')
            if run_dir.resolve().is_relative_to(Path(resume_from).resolve().parent.parent):raise ValueError('Resume must not modify parent run')
            data,browser=load_checkpoint(resume_from);settings=data['settings.json']
            if data['state.json']['completed_rounds']==settings['question_count']:return {'status':'already_complete','output_created':False}
            records=data['dataset.json'];topic=settings['topic'];pages=data['pages.json']
            _,_,tasks,editable=build_paired_settings(records,topic,settings['selectors'],settings['question_ids'],settings['access_manifest'],settings['visible_labels'],settings['round_leaders'],settings.get('inference_profile'),settings.get('related_append_only',False),settings.get('question_pairing_policy'))
            histories=data['histories.json'];results=data['results.json'];transitions=data['transitions.json'];state=data['state.json']
            settings={**settings,'resume':{'parent':str(Path(resume_from).resolve()),'manifest_sha256':hashlib.sha256((Path(resume_from)/'paired-view-checkpoint.json').read_bytes()).hexdigest()}}
            for row in results:row['log_path']=str(Path(resume_from).resolve().parent.parent/row['log_path'])
        else:
            settings,pages,tasks,editable=build_paired_settings(records,topic,selectors,question_ids,access_manifest,visible_labels,round_leaders,inference_profile,related_append_only)
            histories={a:reset_base_history(settings['system_prompts'][a],topic) for a in AGENTS}
            results=[];transitions=[];state={'completed_rounds':0,'next_phase_index':0}
        client.notebook_tools_enabled=True;client.notebook_tool_schemas=settings["notebook_tool_schemas"]
        settings['provenance']={**(provenance or {}),'model':(validate_hf_client(client) if settings.get('inference_profile')=='hf-fp8-v1' else validate_client(client))}
        policy=TimedPolicy(**settings['policy'])
        split_policy=replace(policy,preparation_generated_tokens=1024,preparation_browser_calls=2,preparation_seconds=policy.preparation_seconds/2)
        run_dir.mkdir(parents=True)
        for name,value in (('settings.json',settings),('dataset.json',records),('pages.json',pages),('results.json',results),('transitions.json',transitions)):write_json(run_dir/name,value)
        if browser is None:browser=browser_for(settings,pages,editable,run_dir/'wiki.sqlite3')
        else:
            with closing(sqlite3.connect(run_dir/'wiki.sqlite3')) as db:browser.db.backup(db)
            browser.db.close();browser.db=sqlite3.connect(run_dir/'wiki.sqlite3',check_same_thread=False)
        client=BoundedContextClient(client,policy,run_dir)
        status={'status':'in_progress',**state};before=None
        try:
            saved=save_checkpoint(run_dir,browser,histories,state)
            if checkpoint_callback:checkpoint_callback(saved)
            for index in range(state['next_phase_index'],settings['maximum_phases']):
                agent,role,qid,slot,writer,reader=sequence(settings)[index]
                if index%7==0 and job_deadline is not None and time.monotonic()+2*(policy.preparation_seconds+policy.answer_seconds+NOTE_SECONDS)+7*policy.initial_readiness_timeout_seconds>=job_deadline:
                    status['status']='job_safety_stop';break
                view=None
                if role=='research1':
                    client.begin_question(reader,histories[reader])
                    before=show_view(browser,histories[reader],run_dir,index,slot,reader,writer,settings,'before');view=before
                elif role=='research':client.begin_question(writer,histories[writer])
                elif role=='research2':
                    view=show_view(browser,histories[reader],run_dir,index,slot,reader,writer,settings,'after',before,results[-1])
                phase='answer' if role=='answer' else 'preparation'
                active_policy=split_policy if role in ('research1','research2') else policy
                exposure=None
                if role!='note':
                    client.before_forced_exposure(agent,histories[agent])
                    exposure=expose_log(browser,histories[agent],run_dir,index,agent,phase,visible_label=settings['visible_labels'][agent])
                    phase_tasks={qid:{**tasks[qid],'collection_url':'https://docs.test/'}} if settings['access_plan'] else tasks
                    prompt=phase_prompt(phase,qid,slot,topic,phase_tasks,active_policy)
                    row=run_phase_with_readiness(browser,client,histories[agent],prompt,phase,getattr(active_policy,phase+'_seconds'),active_policy,
                        run_dir,index,qid,transitions,results,policy.initial_readiness_timeout_seconds if index==state['next_phase_index'] else policy.readiness_timeout_seconds,
                        f'[{index+1}/{settings["maximum_phases"]}] {agent} {role} round={slot}',agent=agent)
                else:
                    locked_answer=next(r['answer'] for r in reversed(results) if r['agent']==agent and r['question_id']==qid and r['phase_role']=='answer')
                    row=run_mandatory_note(HostAppendAdapter(browser) if settings.get("related_append_only") else browser,client,histories[agent],run_dir,index,agent,qid,settings['notebooks'][agent],locked_answer,results,transitions,
                        policy.readiness_timeout_seconds,append_notes=True,neutral_notebook=True)
                row.update(agent=agent,phase_role=role);transitions[-1].update(agent=agent,phase_role=role)
                if exposure:row['forced_log_exposure']=exposure
                if view:row['host_paired_notebook_view']=view
                if row['status'] not in NORMAL or row.get('safety_timeout_hit'):raise RuntimeError(f'Incomplete {agent} {role}: {row["status"]}')
                if role=='note' and not row.get('note_preservation',{}).get('persistence_verified'):raise RuntimeError('Unverified mandatory append')
                status['next_phase_index']=index+1
                if index%7==6:
                    state={'completed_rounds':slot,'next_phase_index':index+1};status.update(state)
                    for name,value in (('results.json',results),('transitions.json',transitions)):write_json(run_dir/name,value)
                    saved=save_checkpoint(run_dir,browser,histories,state)
                    if checkpoint_callback:checkpoint_callback(saved)
                write_json(run_dir/'histories.json',histories);write_json(run_dir/'manifest.json',status)
            else:status['status']='complete'
        except BaseException as error:
            status.update(status='failed',error=f'{type(error).__name__}: {error}');raise
        finally:
            for name,value in (('histories.json',histories),('results.json',results),('transitions.json',transitions),('manifest.json',status)):write_json(run_dir/name,value)
        return status
    finally:
        if browser is not None:browser.close()
