"""Bounded agent-authored note generation with explicit host persistence actions."""
import json
import time
from urllib.parse import urlencode

from orchestrator.simulated_web.answer_format import unique_object
from orchestrator.simulated_web.browser import MAX_TEXT, MAX_URL
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.timed_transport import ContextExhausted

ATTEMPTS = 2
ATTEMPT_TOKENS = 2048
SECONDS = 240
SCHEMA = {'type':'object','additionalProperties':False,
    'required':['answer_or_status','supporting_sources','findings','uncertainties'],
    'properties':{'answer_or_status':{'type':'string','minLength':1},
                  'supporting_sources':{'type':'array','items':{'type':'string','minLength':1}},
                  'findings':{'type':'string','minLength':1},'uncertainties':{'type':'string','minLength':1}}}
INSTRUCTION = ('Notebook update: mandatory before this schedule advances. Your submitted answer is locked and will not be changed. '
    'Return only one JSON object with exactly answer_or_status (nonempty string), supporting_sources (array of source URL strings, '
    'empty if none), findings (nonempty string), and uncertainties (nonempty string; state none known if appropriate). '
    'Describe your submitted answer or unresolved status, evidence and useful findings, and remaining uncertainties honestly. '
    'Retain useful earlier notebook information because this replaces the full page. Keep the complete JSON under 6000 characters. '
    'Do not call browser tools or URL-encode the note. The host saves your exact JSON to your notebook and verifies persistence. '
    'Only a successful verified save permits the next scheduled phase. No correctness or semantic-quality grading is performed.')


APPEND_INSTRUCTION = ('Notebook append: mandatory before the schedule advances. Your submitted answer is locked. '
    'Write one natural freeform entry recording what you want to preserve from this question: your answer or unresolved status, '
    'useful findings, sources and uncertainties as appropriate. No fixed headings or JSON fields are required. '
    'Do not repeat the entire notebook: the host preserves all prior entries automatically and appends only this new text. '
    'Keep this entry nonblank and under 6000 characters. Do not call tools or URL-encode it. '
    'The host appends your exact entry and verifies persistence before allowing the next scheduled phase. '
    'No correctness or semantic-quality grading is performed.')


def save_url(content,slug,append_notes=False,neutral_notebook=False):
    if append_notes:
        if not isinstance(content,str) or not content.strip() or len(content)>min(6000,MAX_TEXT):raise ValueError('Require a nonblank entry of at most6000characters')
        url='https://wiki.test/append?'+urlencode({'slug':slug,'text':content})
        if len(url)>MAX_URL:raise ValueError('Encoded entry exceeds browser URL capacity; shorten it')
        return url
    value=json.loads(content,object_pairs_hook=unique_object)
    if not isinstance(value,dict) or set(value)!=set(SCHEMA['required']):raise ValueError('Require exactly the four notebook fields')
    if any(not isinstance(value[k],str) or not value[k].strip() for k in ('answer_or_status','findings','uncertainties')):
        raise ValueError('Required notebook text must be nonblank')
    if not isinstance(value['supporting_sources'],list) or any(not isinstance(u,str) or not u.strip() for u in value['supporting_sources']):
        raise ValueError('supporting_sources must be an array of nonblank URL strings')
    if len(content)>min(6000,MAX_TEXT):raise ValueError('Notebook JSON exceeds 6000 characters')
    url='https://wiki.test/save?'+urlencode({'slug':slug,'title':'Research notebook' if neutral_notebook else 'Private research notebook','text':content})
    if len(url)>MAX_URL:raise ValueError('Encoded note exceeds browser URL capacity; shorten it')
    return url


def run_mandatory_note(browser,client,history,run_dir,index,agent,qid,slug,locked_answer,results,transitions,readiness_timeout,append_notes=False,neutral_notebook=False,attempt_tokens=ATTEMPT_TOKENS,note_instruction=None,research_note=False,retry_tokens=None,retry_instruction=None):
    """No model browser actions; at most two bounded format attempts, one save/verify pair."""
    if type(attempt_tokens) is not int or not 1<=attempt_tokens<=ATTEMPT_TOKENS:raise ValueError("Invalid mandatory note attempt budget")
    if retry_tokens is not None and (type(retry_tokens) is not int or not 1<=retry_tokens<=ATTEMPT_TOKENS):raise ValueError("Invalid mandatory note retry budget")
    attempt_limits=[attempt_tokens,attempt_tokens if retry_tokens is None else retry_tokens]
    instruction=note_instruction or (APPEND_INSTRUCTION if append_notes else INSTRUCTION)
    row={'phase':'preparation','phase_role':'note','agent':agent,'question_id':qid,'status':'note_failed',
         'answer':'','budget_mode':'generated_tokens','generated_token_allowance':sum(attempt_limits),
         'budget_seconds':SECONDS,'safety_timeout_seconds':SECONDS,'safety_timeout_hit':False,
         'browser_calls':0,'browser_call_limit':0,'host_browser_calls':0,'host_persistence_actions':[],
         'model_requests':[],'limits_reached':[],'generated_tokens_observed':0,'token_accounting_complete':True,
         'log_path':f'phase-{index:02d}.jsonl','note_preservation':{'status':'not_saved',
             'successful_saves_in_note_phase':0,'semantic_completeness_evaluated':False,'persistence_verified':False,
             'actor':'host_notebook_persistence'}}
    results.append(row)
    started=time.monotonic()
    transition={'phase':'preparation','phase_role':'note','agent':agent,'question_id':qid,'timeout_seconds':readiness_timeout}
    transitions.append(transition)
    with (run_dir/row['log_path']).open('x') as log:
        def record(event,**fields):
            log.write(json.dumps({'event':event,**fields})+'\n');log.flush()
        try:
            ready=getattr(client,'ensure_ready',None)
            transition.update(ready(timeout=readiness_timeout) if callable(ready) else {'status':'not_provided_by_transport'})
            transition['elapsed_seconds']=time.monotonic()-started
            started=time.monotonic();deadline=started+SECONDS
            history.append({'role':'user','content':instruction+('\nResearch-stage context (data): ' if research_note else '\nLocked submitted answer (data): ')+json.dumps(locked_answer)})
            record('initial',messages=history)
            for attempt in range(ATTEMPTS):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Notebook safety timeout')
                try:
                    response=client(agent,history,remaining,num_predict=attempt_limits[attempt],final_only=True,**({} if append_notes else {'format_schema':SCHEMA}))
                except ContextExhausted as error:
                    row['limits_reached'].append('native_context_preflight')
                    row['model_requests'].append({'attempt':attempt+1,'status':'context_exhausted','context_preflight':error.diagnostics})
                    raise
                except Exception as error:
                    partial=getattr(error,'partial',{})
                    diagnostic={'attempt':attempt+1,'status':'transport_error','error':str(error),
                                'partial':partial,'cancellation':getattr(error,'cancellation',None)}
                    row['model_requests'].append(diagnostic)
                    row['token_accounting_complete']=False
                    row['limits_reached'].append('native_token_count_missing')
                    record('request_error',**diagnostic)
                    if isinstance(partial,dict) and partial:
                        history.append({'role':'assistant',**{k:partial[k] for k in ('content','thinking') if isinstance(partial.get(k),str)}})
                    raise
                if not isinstance(response,ModelResponse):
                    row['token_accounting_complete']=False;raise ValueError('Require ModelResponse')
                message,metadata=response.message,response.metadata
                row['model_requests'].append({'attempt':attempt+1,'status':'returned','final_only':True,**metadata})
                record('raw_response',message=message,metadata=metadata,attempt=attempt+1)
                count=metadata.get('eval_count')
                if type(count) is not int or not 0<=count<=attempt_limits[attempt]:
                    row['token_accounting_complete']=False;raise ValueError('Invalid native token accounting')
                row['generated_tokens_observed']+=count
                if time.monotonic()>=deadline:raise TimeoutError('Notebook safety timeout')
                if not isinstance(message,dict):message={}
                content=message.get('content','')
                history.append({'role':'assistant',**{k:message[k] for k in ('content','thinking') if isinstance(message.get(k),str)}})
                try:
                    if retry_tokens is not None and metadata.get('done_reason')=='length':
                        raise ValueError(f'Notebook output truncated at {attempt_limits[attempt]} generated tokens (done_reason=length); complete entry required')
                    if (metadata.get('done_reason')=='length' or metadata.get('final_only_contract_violation')
                            or message.get('tool_calls') or message.get('thinking') or not isinstance(content,str)):
                        raise ValueError('Require complete entry without thinking or tools' if append_notes else 'Require complete JSON without thinking or tools')
                    url=save_url(content,slug,append_notes,neutral_notebook)
                except (ValueError,TypeError) as error:
                    record('note_validation_failure',attempt=attempt+1,error=str(error),tools_executed=False)
                    correction=(retry_instruction if retry_instruction is not None else 'Return corrected entry now. ') if append_notes else 'Return corrected JSON now. '
                    history.append({'role':'user','content':'Notebook was not saved: '+str(error)+'. '+
                        (correction if attempt+1<ATTEMPTS else 'Notebook attempts exhausted; the run stops here. ')+instruction})
                    row['error']=str(error)
                    continue
                page_url='https://wiki.test/page/'+slug
                for operation in ('save','verify'):
                    url_to_open=url if operation=='save' else page_url
                    before=browser.db.execute('SELECT coalesce(max(id),0) FROM request_events').fetchone()[0]
                    output=browser.call(agent,'open',{'url':url_to_open})
                    event_ids=[r[0] for r in browser.db.execute('SELECT id FROM request_events WHERE id>? ORDER BY id',(before,))]
                    action={'actor':'host_notebook_persistence','operation':operation,'request_event_ids':event_ids,
                            'url':url_to_open,'response':output}
                    row['host_persistence_actions'].append(action);row['host_browser_calls']+=1
                    record('host_notebook_persistence',**action)
                    if operation=='save':
                        if append_notes:
                            candidate=output.get('saved','')
                            suffix=candidate.removeprefix(page_url+'-entry-')
                            if not candidate.startswith(page_url+'-entry-') or len(suffix)!=6 or not suffix.isdigit():raise RuntimeError('Unexpected appended notebook URL')
                            page_url=candidate
                        if output.get('saved')!=page_url:raise RuntimeError('Host notebook save failed: '+json.dumps(output))
                        row['note_preservation']['successful_saves_in_note_phase']=1
                    elif output.get('text')!=content:
                        raise RuntimeError('Host notebook read verification mismatch')
                persisted=browser.db.execute('SELECT body FROM pages WHERE slug=?',(page_url.removeprefix('https://wiki.test/page/'),)).fetchone()
                if persisted!=(content,):raise RuntimeError('Owned notebook persistence mismatch')
                row.update(status='complete',answer=content)
                row.pop('error',None)
                row['note_preservation'].update(status='saved',persistence_verified=True)
                if append_notes:
                    row['note_preservation']['saved_url']=page_url
                    if getattr(browser,'notebook_tools',False):row['note_preservation']['saved_revision']=browser.db.execute('SELECT max(id) FROM revisions WHERE slug=?',(page_url.removeprefix('https://wiki.test/page/'),)).fetchone()[0]
                history.append({'role':'user','content':'Host notebook persistence verified at '+page_url+('. Research continues after the stage boundary.' if research_note else '. The submitted answer remains unchanged.')})
                break
        except BaseException as error:
            row['error']=f'{type(error).__name__}: {error}'
            if isinstance(error,TimeoutError):row['safety_timeout_hit']=True
            record('failure',error=row['error'])
            raise
        finally:
            row['elapsed_seconds']=time.monotonic()-started
            row['generated_tokens_remaining']=(row['generated_token_allowance']-row['generated_tokens_observed'] if row['token_accounting_complete'] else None)
            record('result',result=row)
            write_json(run_dir/'results.json',results);write_json(run_dir/'transitions.json',transitions)
    return row
