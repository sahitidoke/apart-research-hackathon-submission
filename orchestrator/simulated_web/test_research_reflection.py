"""Finite mock-only 10a phase/reset/persistence contracts."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web.research_reflection import build_settings,run_research_reflection,load_checkpoint,RetainedContextClient
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client
from orchestrator.simulated_web.timed_transport import ContextExhausted
from orchestrator.simulated_web.timed_policy import TimedPolicy


class Client(FP8Client):
    def __init__(self):super().__init__();self.seen=[]
    def __call__(self,agent,history,timeout,**kwargs):
        self.seen.append((agent,json.loads(json.dumps(history)),kwargs))
        prompt=history[-1].get('content','')
        if prompt.startswith('Answer phase:') or kwargs.get('format_schema'):
            content=json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})
        elif prompt.startswith('Reflection:'):content='FINAL_REFLECTION_'+agent
        elif kwargs.get('final_only'):content='INITIAL_NOTE_'+agent
        else:content='INITIAL_RESEARCH_'+agent
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})


def test_schedule_locks_exact_save_reset_and_budget(tmp_path):
    client=Client();values=paired_options();settings,pages,tasks=build_settings(**values)
    result=run_research_reflection(tmp_path/'run',client,**values)
    assert result['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    assert [r['phase_role'] for r in rows]==['initial_research','initial_note']*2+(['answer']*2+['reflection','reflection_note']*2)*2
    assert all(r['generated_token_allowance']==0 and r['reflection_persistence']=='exact_final_entry' for r in rows if r['phase_role']=='reflection_note')
    assert len(client.seen)==12
    assert settings['maximum_generated_tokens']==50176 and settings['maximum_combined_generated_tokens']==54272
    for agent,history,kwargs in client.seen:
        text=json.dumps(history);prompt=history[-1].get('content','')
        if prompt.startswith('Initial research:'):
            assert all(task['question'] not in text for task in tasks.values())
        assert 'accessible research notebook' not in text
    first=json.loads((tmp_path/'run/round-01-histories-before-reset.json').read_text())
    second=json.loads((tmp_path/'run/round-02-histories-before-reset.json').read_text())
    assert all('INITIAL_RESEARCH_'+a in json.dumps(first[a]) for a in first)
    assert all('INITIAL_RESEARCH_' not in json.dumps(h) and 'INITIAL_NOTE_' not in json.dumps(h) for h in second.values())
    events=[json.loads(line) for line in (tmp_path/'run/evidence-index.host-only.jsonl').read_text().splitlines()]
    for q in (1,2):
        locks=[e['event_order'] for e in events if e['event']=='answer_locked' and e['question_round']==q]
        reflect=next(e['event_order'] for e in events if e['event']=='stage_snapshots_frozen' and e['question_round']==q and e['stage']==3)
        assert len(locks)==2 and max(locks)<reflect
    data,browser,_=load_checkpoint(tmp_path/'run/checkpoints/rounds-002')
    try:
        assert all(len(h)==2 for h in data['histories.json'].values())
        bodies=[r[0] for r in browser.db.execute('SELECT p.body FROM pages p JOIN entry_provenance USING(slug)')]
        assert bodies.count('FINAL_REFLECTION_agent-1')==2 and bodies.count('FINAL_REFLECTION_agent-2')==2
        assert len(bodies)==6
    finally:browser.close()


def test_partial_checkpoint_resume_and_legacy_schema_rejected(tmp_path):
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('mock stop')
    with pytest.raises(RuntimeError,match='mock stop'):
        run_research_reflection(tmp_path/'first',Client(),**paired_options(),checkpoint_callback=stop)
    assert run_research_reflection(tmp_path/'resume',Client(),resume_from=tmp_path/'first/checkpoints/rounds-001')['status']=='complete'
    with pytest.raises(ValueError,match='Resume overrides'):
        run_research_reflection(tmp_path/'override',Client(),resume_from=tmp_path/'first/checkpoints/rounds-001',memory_policy='question-reset-v1')
    with pytest.raises(ValueError):run_research_reflection(tmp_path/'bad',Client(),**{**paired_options(),'memory_policy':'strict-retain-v1'})
    assert not (tmp_path/'bad').exists()


def test_no_automatic_body_masking_and_native_guard_preserves_history(tmp_path):
    settings,_,_=build_settings(**paired_options());client=Client()
    client.count_context=lambda history,timeout:{'prompt_tokens':60000}
    wrapper=RetainedContextClient(client,TimedPolicy(**settings['policy']),tmp_path)
    history=[{'role':'tool','tool_name':'open','content':'FULL_SOURCE_BODY'}]
    with pytest.raises(ContextExhausted):wrapper('agent-1',history,10,num_predict=2048)
    assert history[0]['content']=='FULL_SOURCE_BODY'
    assert list(tmp_path.glob('*-failure-history.json'))


def test_reflection_invalid_final_repairs_bounded_and_persists(tmp_path):
    class Blank(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            response=super().__call__(agent,history,timeout,**kwargs)
            if history[-1].get('content','').startswith('Reflection:'):response.message['content']=''
            return response
    result=run_research_reflection(tmp_path/'run',Blank(),**paired_options())
    assert result['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    notes=[r for r in rows if r['phase_role']=='reflection_note']
    assert all(r['reflection_persistence']=='repaired_final_entry' and r['generated_token_allowance']==1536 for r in notes)
    assert all(r['note_preservation']['persistence_verified'] for r in notes)


def test_failed_persistence_preserves_answers_and_does_not_reset(tmp_path):
    with patch('orchestrator.simulated_web.research_reflection.persist_reflection',side_effect=RuntimeError('mock save failure')):
        with pytest.raises(RuntimeError,match='mock save failure'):run_research_reflection(tmp_path/'run',Client(),**paired_options())
    rows=json.loads((tmp_path/'run/results.json').read_text())
    assert len([r for r in rows if r['phase_role']=='answer'])==2
    assert not (tmp_path/'run/round-01-histories-before-reset.json').exists()
    assert (tmp_path/'run/round-01-stage-3/agent-1.sqlite3').exists()
    assert len(json.loads((tmp_path/'run/histories.json').read_text())['agent-1'])>2


def test_reflection_tools_future_question_hidden_and_only_final_text_saved(tmp_path):
    values=paired_options();settings,_,tasks=build_settings(**values)
    next_question=tasks[settings['question_ids']['agent-1'][1]]['question']
    class ToolReflection(Client):
        def __init__(self):super().__init__();self.reflections={a:0 for a in ('agent-1','agent-2')}
        def __call__(self,agent,history,timeout,**kwargs):
            prompt=history[-1].get('content','')
            if prompt.startswith('Reflection:'):
                self.reflections[agent]+=1
                if self.reflections[agent]==1:assert next_question not in json.dumps(history)
                assert kwargs['final_only'] is False
                return ModelResponse({'content':'','thinking':'PRIVATE_REFLECTION_REASONING','tool_calls':[
                    {'function':{'name':'open','arguments':{'url':'https://docs.test/'}}},
                    {'function':{'name':'read_notebook','arguments':{'url':'https://wiki.test/page/'+settings['notebooks'][agent],'revision':''}}}]},
                    {'eval_count':10,'prompt_eval_count':100,'done_reason':'stop'})
            if history[-1]['role']=='tool':
                return ModelResponse({'content':'EXACT_SELECTED_ENTRY_'+agent,'thinking':'NOT_FOR_NOTEBOOK'},
                                     {'eval_count':8,'prompt_eval_count':100,'done_reason':'stop'})
            return super().__call__(agent,history,timeout,**kwargs)
    assert run_research_reflection(tmp_path/'run',ToolReflection(),**values)['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    assert all(r['source_browser_calls']==1 and r['notebook_calls']==1 for r in rows if r['phase_role']=='reflection')
    data,browser,_=load_checkpoint(tmp_path/'run/checkpoints/rounds-002')
    try:
        body='\n'.join(r[0] for r in browser.db.execute('SELECT body FROM pages'))
        assert 'EXACT_SELECTED_ENTRY_' in body and 'PRIVATE_REFLECTION_REASONING' not in body and 'NOT_FOR_NOTEBOOK' not in body
    finally:browser.close()
