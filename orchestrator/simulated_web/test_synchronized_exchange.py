"""Mock-only 9d stage isolation, crossappend, reset and complete-round recovery."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
from orchestrator.simulated_web.synchronized_exchange import build_settings,run_exchange,load_checkpoint,schedule
from orchestrator.simulated_web.synchronized_notebooks import make_browser,fork_stage,publish_stage,self_memory
from orchestrator.simulated_web.test_related_notebooks import related_inputs
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient
from orchestrator.simulated_web.bounded_context import BoundedContextClient
from orchestrator.simulated_web.runner import ModelResponse


def inputs():
    value=related_inputs();value.pop('related_append_only');return value


class Client(NeutralClient):
    def __call__(self,agent,history,timeout,**kwargs):
        if kwargs.get('final_only'):
            return ModelResponse({'content':'Natural note '+agent},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)
    def count_text(self,text,timeout):return {'tokens':len(text)}


def test_stage_barrier_crossappend_provenance_immutable_access_and_logs(tmp_path):
    settings,pages,_=build_settings(**inputs());central=make_browser(settings,pages,':memory:')
    snapshots=fork_stage(central,settings,pages,tmp_path,1,1)
    try:
        a,b=snapshots.values();root='https://wiki.test/page/'+settings['notebooks']['agent-2']
        first=a.call('agent-1','append_notebook',{'text':'Pending request & space\nUnicode 漢字','notebook':root})
        assert first['author']=='90' and first['question_round']==1 and first['created_at']
        assert 'error' in b.call('agent-2','read_notebook',{'url':first['saved'],'revision':''})
        assert 'Pending request' not in b.call('agent-2','open',{'url':'https://docs.test/request-history'})['text']
        assert b.db.execute("SELECT count(*) FROM request_events WHERE owner='agent-1'").fetchone()[0]==0
        assert central.db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==0
        second=b.call('agent-2','append_notebook',{'text':'Other current-stage finding','notebook':root})
        assert first['saved']!=second['saved'] and first['revision']!=second['revision']
        assert 'error' in a.call('agent-1','open',{'url':second['saved']})
        for agent,local in [('agent-1',a),('agent-2',b)]:
            peer=next(x for x in settings['question_ids'] if x!=agent)
            denied=set(settings['access_plan']['allowed_urls'][peer])-set(settings['access_plan']['allowed_urls'][agent])
            for url in denied:assert local.call(agent,'open',{'url':url})=={'error':'Access denied'}
            assert 'error' in local.call(agent,'edit_notebook',{'url':first['saved'],'expected_revision':first['revision'],'text':'erase'})
            for route in ('save','append','edit'):
                assert 'error' in local.call(agent,'open',{'url':'https://wiki.test/'+route+'?slug=notes-maple&text=erase'})
            assert 'error' in local.call(agent,'append_notebook',{'text':'bad','notebook':'https://docs.test/'})
        mapping=publish_stage(central,snapshots)
        assert all(mapping.values())
        for entry in (first,second):
            read=central.call('agent-2','read_notebook',{'url':entry['saved'],'revision':''})
            assert read['author']==entry['author'] and read['question_round']==1 and read['created_at']==entry['created_at']
        assert central.db.execute('SELECT count(*) FROM revisions').fetchone()[0]==2
        payload,_=self_memory(central,'agent-1',Client(),2048)
        assert len(payload['entries'])==1 and payload['entries'][0]['url']==first['saved']
        assert payload['entries'][0]['author']=='90'
        log=central.call('agent-2','open',{'url':'https://docs.test/request-history'})['text']
        assert first['saved'] in log and 'Pending request' not in log and 'Unicode' not in log
    finally:
        for b in snapshots.values():b.close()
        central.close()


def test_memory_whole_entry_cap_is_nondestructive(tmp_path):
    settings,pages,_=build_settings(**inputs());b=make_browser(settings,pages,':memory:',1,1)
    try:
        for text in ('old lesson','x'*3000):b.call('agent-1','append_notebook',{'text':text})
        payload,tokens=self_memory(b,'agent-1',Client(),2048)
        assert tokens<=2048 and payload['entries']==[] and payload['omitted_entries']==2
        assert len(payload['all_entries_remain_available_at'])==2
        assert b.db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==2
        with pytest.raises(ValueError,match='metadata'):self_memory(b,'agent-1',Client(),1)
    finally:b.close()


def test_schedule_reset_notes_protection_and_resume(tmp_path):
    path=tmp_path/'first';client=Client();seen=[]
    original=BoundedContextClient.fit
    def fit(wrapper,agent,history,*args,**kwargs):
        protected=[i for i,m in enumerate(history) if m.get('tool_call_id','').startswith('host-stage-view')]
        assert not set(protected)&set(wrapper.candidates(agent,history))
        seen.append(len(protected));return original(wrapper,agent,history,*args,**kwargs)
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('mock stop')
    with patch.object(BoundedContextClient,'fit',fit),pytest.raises(RuntimeError,match='mock stop'):
        run_exchange(path,client,**inputs(),checkpoint_callback=stop)
    data,b,tasks=load_checkpoint(path/'checkpoints/rounds-001');b.close()
    assert len(data['results.json'])==16
    assert len([r for r in data['results.json'] if r['phase_role'].endswith('note')])==8
    assert all(len(h)==2 for h in data['histories.json'].values())
    assert max(seen)>2
    rows=data['results.json']
    assert all(r['generated_token_allowance']==1024 for r in rows)
    assert all(r['browser_call_limit']==4 for r in rows if not r['phase_role'].endswith('note'))
    stage1=json.loads((path/'round-01-stage-1/agent-2-after-views.json').read_text())
    stage3=json.loads((path/'round-01-stage-3/agent-2-after-views.json').read_text())
    assert set(r['url'] for r in stage1)<=set(r['url'] for r in stage3)
    assert run_exchange(tmp_path/'resume',Client(),resume_from=path/'checkpoints/rounds-001')['status']=='complete'
    final,b,_=load_checkpoint(tmp_path/'resume/checkpoints/rounds-002');b.close()
    assert len(final['results.json'])==32
    memory=json.loads((tmp_path/'resume/memory-02-agent-1.json').read_text())
    assert all(e['author']=='90' for e in memory['payload']['entries'])
    assert memory['native_tokens']<=2048
    assert run_exchange(tmp_path/'done',Client(),resume_from=tmp_path/'resume/checkpoints/rounds-002')['status']=='already_complete'
    assert not (tmp_path/'done').exists()


def test_failed_note_does_not_publish_or_advance(tmp_path):
    class Blank(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            response=super().__call__(agent,history,timeout,**kwargs)
            if kwargs.get('final_only'):response.message['content']=''
            return response
    with pytest.raises(RuntimeError):run_exchange(tmp_path/'bad',Blank(),**inputs())
    manifest=json.loads((tmp_path/'bad/manifest.json').read_text());assert manifest['completed_rounds']==0
    assert not (tmp_path/'bad/round-01-stage-1/barrier.json').exists()
    rows=json.loads((tmp_path/'bad/results.json').read_text());assert len(rows)==2
    assert rows[-1]['generated_token_allowance']==1024
    assert (tmp_path/'bad/round-01-stage-1/agent-1.sqlite3').exists()


def test_invalid_inputs_create_no_run(tmp_path):
    value=inputs();value['round_leaders']=[]
    with pytest.raises(ValueError):run_exchange(tmp_path/'invalid',Client(),**value)
    assert not (tmp_path/'invalid').exists()


def test_note_attempts_count_full_512_allowance_and_keep_locked_answer(tmp_path):
    class Length(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            if kwargs.get('final_only'):
                assert kwargs['num_predict']==512
                return ModelResponse({'content':'incomplete'}, {'eval_count':512,'prompt_eval_count':100,'done_reason':'length'})
            return super().__call__(agent,history,timeout,**kwargs)
    with pytest.raises(RuntimeError):run_exchange(tmp_path/'length',Length(),**inputs())
    note=json.loads((tmp_path/'length/results.json').read_text())[-1]
    assert note['generated_tokens_observed']==1024 and len(note['model_requests'])==2
    assert not note['note_preservation']['persistence_verified']


def test_initial_checkpoint_callback_and_tokenizer_prerequisite(tmp_path):
    checkpoints=[]
    def stop(cp):
        checkpoints.append(cp.name);raise RuntimeError('initial checkpoint persisted')
    with pytest.raises(RuntimeError,match='initial checkpoint persisted'):
        run_exchange(tmp_path/'initial',Client(),**inputs(),checkpoint_callback=stop)
    assert checkpoints==['rounds-000']
    client=Client();client.count_text=None
    with pytest.raises(ValueError,match='Native memory'):run_exchange(tmp_path/'invalid-client',client,**inputs())
    assert not (tmp_path/'invalid-client').exists()


def test_evidence_index_references_exact_data_without_agent_leak(tmp_path):
    class AppendingClient(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            if not kwargs.get('final_only') and history[-1]['role']=='user':
                self.calls.append((agent,json.loads(json.dumps(history))))
                return ModelResponse({'content':'','tool_calls':[{'function':{'name':'append_notebook','arguments':{'text':'Voluntary finding','notebook':'https://wiki.test/page/notes-maple'}}}]},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
            return super().__call__(agent,history,timeout,**kwargs)
    client=AppendingClient();path=tmp_path/'evidence'
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('stop after evidence')
    with pytest.raises(RuntimeError,match='stop after evidence'):run_exchange(path,client,**inputs(),checkpoint_callback=stop)
    events=[json.loads(line) for line in (path/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert [e['event_order'] for e in events]==list(range(1,len(events)+1))
    appends=[e for e in events if e['event']=='append_attempt']
    assert len(appends)==16 and all(e['success'] for e in appends)
    assert len([e for e in appends if e['actor']=='host_mandatory_note'])==8
    assert len([e for e in appends if e['actor']=='model_voluntary_tool'])==8
    assert events[-1]['event']=='run_finished'
    assert all(e['captured_at_utc'] and e['capture_elapsed_seconds']>=0 for e in events)
    for event in appends:
        assert event['author'] in ('90','91') and event['destination'] and event['revision']
        assert event['database'].endswith('.sqlite3') and event['audit_id']>0
    phase_events=[e for e in events if e['event']=='phase_finished']
    assert len(phase_events)==16 and all(e['generated_token_allowance']==1024 for e in phase_events)
    assert all('text' not in e for e in events)
    assert len([e for e in events if e['event']=='stage_published'])==4
    assert len([e for e in events if e['event']=='question_context_reset'])==1
    assert all('evidence-index.host-only' not in json.dumps(history) and 'annotation-schema.host-only' not in json.dumps(history) for _,history in client.calls)
    data,b,_=load_checkpoint(path/'checkpoints/rounds-001')
    try:
        for host_name in ('evidence-index.host-only.jsonl','annotation-schema.host-only.json'):
            assert 'error' in b.call('agent-1','open',{'url':'https://docs.test/'+host_name})
    finally:b.close()


def test_host_index_does_not_change_protocol_or_phase_costs(tmp_path):
    class NoIndex:
        def __init__(self,path):self.indexed_phases=set()
        def emit(self,*args,**kwargs):pass
        def phase(self,row,browser,baseline,q,stage,index,database):self.indexed_phases.add(index)
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('parity boundary')
    outputs=[]
    for enabled in (True,False):
        path=tmp_path/str(enabled);client=Client()
        if enabled:
            with pytest.raises(RuntimeError,match='parity boundary'):run_exchange(path,client,**inputs(),checkpoint_callback=stop)
        else:
            with patch('orchestrator.simulated_web.synchronized_exchange.EvidenceIndex',NoIndex),pytest.raises(RuntimeError,match='parity boundary'):
                run_exchange(path,client,**inputs(),checkpoint_callback=stop)
        rows=json.loads((path/'results.json').read_text())
        outputs.append(([(r['agent'],r['phase_role'],r['question_id'],r['answer'],r['generated_token_allowance'],r['generated_tokens_observed'],r['browser_calls']) for r in rows],
                        [history[0] for _,history in client.calls]))
    assert outputs[0]==outputs[1]
