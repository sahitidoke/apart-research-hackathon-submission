"""Finite synthetic 11a contracts; scripted reads are plumbing, not behavior evidence."""
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import collaboration_pilot as runner
from orchestrator.simulated_web import modal_hf_collaboration_pilot as cli
from orchestrator.simulated_web.research_reflection import build_settings as baseline_settings
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client
from orchestrator.simulated_web import test_hf_transport as transport_tests
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.timed import run_phase_with_readiness

AGENTS=('agent-1','agent-2')
RUN_ID='germanwiki-2-011a-fp8-test-collaboration-pilot-11a'
VARIANT=Path(__file__).resolve().parents[2]/'research-log/mlb-tokenpair-2026-09-12/variant-011a'


class Client(FP8Client):
    def __init__(self):super().__init__();self.seen=[]
    def __call__(self,agent,history,timeout,**kwargs):
        self.seen.append((agent,json.loads(json.dumps(history)),kwargs))
        prompt=history[-1].get('content','')
        if prompt.startswith('Answer phase:') or kwargs.get('format_schema'):
            content=json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})
        elif prompt.startswith('Reflection:'):content='FINAL_REFLECTION_'+agent
        elif prompt.startswith('Research stage '):content='FINAL_'+prompt.split(':')[0]+'_'+agent+'\nExact entry with space & 漢字'
        elif kwargs.get('final_only'):content='INITIAL_NOTE_'+agent
        else:content='INITIAL_RESEARCH_'+agent
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})


def test_schedule_exact_save_reset_budget_and_question_hidden(tmp_path):
    client=Client();values=paired_options();settings,pages,tasks=runner.build_settings(**values)
    assert runner.run_collaboration_pilot(tmp_path/'run',client,**values)['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    expected=['initial_research','initial_note']*2+(['research1','research1_note']*2+['research2','research2_note']*2+['answer']*2+['reflection','reflection_note']*2)*2
    assert [r['phase_role'] for r in rows]==expected
    assert len(client.seen)==20 and len(rows)==32
    assert settings['maximum_generated_tokens']==50176 and settings['maximum_combined_generated_tokens']==54272
    assert len(runner.schedule(settings))==32
    notes=[r for r in rows if r['phase_role'].endswith('_note') and r['phase_role']!='initial_note']
    assert all(r['generated_token_allowance']==0 and r['final_entry_persistence']=='exact_final_entry' for r in notes)
    for i,row in enumerate(rows):
        if row in notes:assert row['answer']==rows[i-1]['answer'] and row['note_preservation']['persistence_verified']
        if row['phase_role'] in ('research1','research2','reflection'):
            assert row['generated_token_allowance']==(1024 if row['phase_role']=='reflection' else 1536)
            assert row['final_reserve_tokens']==256 and row['source_browser_call_limit']==4
            assert row['timing_engine_phase']=='answer' and row['phase']!='answer'
    for agent,history,kwargs in client.seen:
        text=json.dumps(history);prompt=history[-1].get('content','')
        if prompt.startswith('Initial research:'):assert all(t['question'] not in text for t in tasks.values())
        assert 'accessible research notebook' not in text
        peer=next(a for a in AGENTS if a!=agent)
        assert 'FINAL_REFLECTION_'+peer not in text and 'INITIAL_NOTE_'+peer not in text
        if prompt.startswith('Research stage '):assert 'format_schema' not in kwargs
    first=json.loads((tmp_path/'run/round-01-histories-before-reset.json').read_text())
    second=json.loads((tmp_path/'run/round-02-histories-before-reset.json').read_text())
    assert all('INITIAL_RESEARCH_'+a in json.dumps(first[a]) for a in AGENTS)
    assert all('INITIAL_RESEARCH_' not in json.dumps(h) and 'INITIAL_NOTE_' not in json.dumps(h) for h in second.values())
    events=[json.loads(line) for line in (tmp_path/'run/evidence-index.host-only.jsonl').read_text().splitlines()]
    for q in (1,2):
        locks=[e['event_order'] for e in events if e['event']=='answer_locked' and e['question_round']==q]
        reflect=next(e['event_order'] for e in events if e['event']=='stage_snapshots_frozen' and e['question_round']==q and e['stage']==5)
        assert len(locks)==2 and max(locks)<reflect
    data,browser,_=runner.load_checkpoint(tmp_path/'run/checkpoints/rounds-002')
    try:
        assert all(len(h)==2 for h in data['histories.json'].values())
        bodies=[r[0] for r in browser.db.execute('SELECT p.body FROM pages p JOIN entry_provenance USING(slug)')]
        assert len(bodies)==14 and bodies.count('FINAL_REFLECTION_agent-1')==2
    finally:browser.close()


def test_symmetric_publication_request_reply_answer_opportunity(tmp_path):
    values=paired_options();settings=runner.build_settings(**values)[0]
    seen=[]
    def inspect(browser,client,history,prompt,phase,seconds,policy,path,index,qid,*args,**kwargs):
        agent=kwargs['agent'];q=browser.round_index;stage=browser.stage_index
        entries=browser.db.execute('SELECT author,question_round,stage FROM entry_provenance').fetchall()
        # No current-stage publication by the other actor can enter its frozen replica.
        assert not any(a!=agent and r==q and s==stage for a,r,s in entries)
        if q and stage in (3,4):
            previous=stage-1
            assert {a for a,r,s in entries if r==q and s==previous}==set(AGENTS)
            seen.append((q,stage,agent))
        return run_phase_with_readiness(browser,client,history,prompt,phase,seconds,policy,path,index,qid,*args,**kwargs)
    class Exchange(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            prompt=history[-1].get('content','');peer=next(a for a in AGENTS if a!=agent)
            if prompt.startswith(('Research stage 2:','Answer phase:')):
                # These are deterministic test fixture URLs, not host-injected model context.
                serial=(3 if peer=='agent-1' else 4) if prompt.startswith('Research stage') else (5 if peer=='agent-1' else 6)
                url='https://wiki.test/page/'+settings['notebooks'][peer]+f'-entry-{serial:06d}'
                return ModelResponse({'content':'','thinking':'PRIVATE_NOT_FOR_PUBLICATION','tool_calls':[{'function':{'name':'read_notebook','arguments':{'url':url,'revision':''}}}]},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})
            if history[-1]['role']=='tool':
                body=json.loads(prompt)['text']
                assert 'FINAL_Research stage 1' in body or 'REPLY_TO_' in body
                active=next(m['content'] for m in reversed(history) if m['role']=='user' and m['content'].startswith(('Research stage 2:','Answer phase:')))
                if active.startswith('Answer phase:'):
                    assert 'REPLY_TO_' in body
                    return ModelResponse({'content':json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})
                return ModelResponse({'content':'REPLY_TO_'+body,'thinking':'PRIVATE_NOT_FOR_PUBLICATION'},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})
            return super().__call__(agent,history,timeout,**kwargs)
    with patch.object(runner,'run_phase_with_readiness',side_effect=inspect):
        assert runner.run_collaboration_pilot(tmp_path/'run',Exchange(),**values)['status']=='complete'
    assert len(seen)==8
    bodies='\n'.join(r[0] for r in sqlite3.connect(tmp_path/'run/wiki.sqlite3').execute('SELECT body FROM pages'))
    assert 'REPLY_TO_FINAL_Research stage 1' in bodies and 'PRIVATE_NOT_FOR_PUBLICATION' not in bodies
    rows=json.loads((tmp_path/'run/results.json').read_text())
    assert all(r['notebook_calls']==1 and r['source_browser_calls']==0 for r in rows if r['phase_role'] in ('research2','answer'))


@pytest.mark.parametrize('phase_prefix,tokens',[('Research stage ',1536),('Reflection:',1024)])
def test_final_reserve_inside_allowance_neutral_no_schema_exact_save(tmp_path,phase_prefix,tokens):
    class Exhaust(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            prompt=history[-1].get('content','')
            if prompt.startswith(phase_prefix):
                assert kwargs['num_predict']==tokens-256
                return ModelResponse({'content':'','thinking':'PRIVATE_LENGTH','tool_calls':[]},{'eval_count':tokens-256,'prompt_eval_count':100,'done_reason':'length'})
            if prompt.startswith('Finalize the notebook entry now'):
                assert kwargs['num_predict']==256 and kwargs['final_only'] and 'format_schema' not in kwargs
                assert 'shortest complete answer' not in prompt
                return ModelResponse({'content':'RESERVED_FINAL_'+agent},{'eval_count':42,'prompt_eval_count':100,'done_reason':'stop'})
            return super().__call__(agent,history,timeout,**kwargs)
    assert runner.run_collaboration_pilot(tmp_path/'run',Exhaust(),**paired_options())['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    roles=('reflection',) if phase_prefix=='Reflection:' else ('research1','research2')
    for i,row in enumerate(rows):
        if row['phase_role'] in roles:
            assert row['generated_tokens_observed']==tokens-256+42
            assert row['generated_token_allowance']==tokens
            assert rows[i+1]['answer']==row['answer']=='RESERVED_FINAL_'+row['agent']
            assert rows[i+1]['generated_tokens_observed']==0


def test_actual_outgoing_transport_browser_notebooks_and_final_only(tmp_path):
    helper=transport_tests.HFTransportTests();settings=runner.build_settings(**paired_options())[0]
    for final in (False,True):
        raw,connection=helper.client(),helper.connection()
        raw.notebook_tools_enabled=True;raw.notebook_tool_schemas=settings['notebook_tool_schemas']
        client=runner.FinalEntryClient(raw)
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
            client('agent-1',[{'role':'user','content':'Research stage 1:'}],5,num_predict=256,final_only=final)
        tokenize,chat=[json.loads(c.args[2]) for c in connection.request.call_args_list]
        assert tokenize=={k:chat[k] for k in tokenize}
        assert [t['function']['name'] for t in chat.get('tools',[])]==([] if final else ['search','open','click','read_notebook','append_notebook'])
        assert 'response_format' not in chat
        if final:
            assert chat['messages'][-1]['content'].startswith('Finalize the notebook entry now')
            assert chat['chat_template_kwargs']['enable_thinking'] is False


def test_resume_rejects_other_protocol_invalid_inputs_and_failure_retained(tmp_path):
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('mock stop')
    with pytest.raises(RuntimeError,match='mock stop'):runner.run_collaboration_pilot(tmp_path/'first',Client(),**paired_options(),checkpoint_callback=stop)
    cp=tmp_path/'first/checkpoints/rounds-001'
    assert runner.run_collaboration_pilot(tmp_path/'resume',Client(),resume_from=cp)['status']=='complete'
    p=cp/'collaboration-pilot-11a-checkpoint.json';manifest=json.loads(p.read_text());manifest['schema']='research-reflection-v1';p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Invalid collaboration'):runner.run_collaboration_pilot(tmp_path/'bad',Client(),resume_from=cp)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError):runner.run_collaboration_pilot(tmp_path/'invalid',Client(),**{**paired_options(),'memory_policy':'invalid'})
    assert not (tmp_path/'invalid').exists()
    with patch.object(runner,'persist_final_entry',side_effect=RuntimeError('mock persistence failure')):
        with pytest.raises(RuntimeError,match='mock persistence failure'):runner.run_collaboration_pilot(tmp_path/'failure',Client(),**paired_options())
    assert (tmp_path/'failure/round-01-stage-2/agent-1.sqlite3').exists()
    assert not (tmp_path/'failure/round-01-histories-before-reset.json').exists()
    assert json.loads((tmp_path/'failure/manifest.json').read_text())['status']=='failed'


def test_invalid_research_final_has_bounded_exceptional_repair(tmp_path):
    class Blank(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            result=super().__call__(agent,history,timeout,**kwargs)
            if history[-1].get('content','').startswith('Research stage '):result.message['content']=''
            return result
    assert runner.run_collaboration_pilot(tmp_path/'run',Blank(),**paired_options())['status']=='complete'
    rows=json.loads((tmp_path/'run/results.json').read_text())
    notes=[r for r in rows if r['phase_role'] in ('research1_note','research2_note')]
    assert all(r['final_entry_persistence']=='repaired_final_entry' and r['generated_token_allowance']==1536 for r in notes)
    assert all(r['note_preservation']['persistence_verified'] for r in notes)


def test_cli_count_identity_and_no_cloud_preflight(tmp_path,capsys):
    inputs=paired_options()
    for invalid in (True,1,3,6,'2'):
        with pytest.raises(ValueError,match='Question count'):cli.validate_fresh(RUN_ID,inputs,invalid)
    with pytest.raises(ValueError,match='Fresh run ID'):cli.validate_fresh('germanwiki-2-010a-fp8-research-reflection',inputs)
    argv=['--run-id',RUN_ID,'--validate-only','--short-note-retry','--notebook-context-policy','9e-v1','--notebook-quota-policy','notebook-exempt-v1','--question-pairing-policy','same-question-v1','--source-access-policy','discovery-only-v1']
    for flag,key in [('dataset','records'),('topic-file','topic'),('editable-sources','selectors'),('question-ids','question_ids'),('access-manifest','access_manifest'),('visible-labels','visible_labels'),('round-leaders','round_leaders')]:
        p=tmp_path/flag;value=inputs[key]
        p.write_text('\n'.join(json.dumps(r) for r in value) if key=='records' else value if key=='topic' else json.dumps(value));argv.extend(['--'+flag,str(p)])
    with patch.object(cli.execute,'remote',side_effect=AssertionError('No remote calls')),patch.object(cli.subprocess,'run',side_effect=AssertionError('No download')):cli.main(argv)
    preflight=json.loads(capsys.readouterr().out)
    assert preflight['status']=='validated_no_cloud_actions'
    assert preflight['resources']['maximum_combined_generated_tokens']==54272
    assert preflight['resources']['phase_records']==32 and preflight['resources']['normal_model_phases']==20
    assert preflight['resources']['hard_job_seconds']==21600 and preflight['resources']['gpu_count']==1


def test_solver_completion_never_precedes_verification_status(tmp_path):
    saved=[];original=runner.write_json
    def capture(path,value):
        if path.name=='manifest.json':saved.append(json.loads(json.dumps(value)))
        return original(path,value)
    def interrupted(*args,**kwargs):raise KeyboardInterrupt('mock verifier interruption')
    with patch.object(runner,'write_json',side_effect=capture),patch.object(runner,'judge_answers',side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):runner.run_collaboration_pilot(tmp_path/'run',Client(),**paired_options())
    assert not any(s['status']=='complete' for s in saved)
    assert any(s['status']=='verification_pending' and s['solver_status']=='complete' for s in saved)
    assert saved[-1]['status']=='verification_interrupted'


def test_actual_package_same_topic_overlap_partition_and_foreign_get():
    root=VARIANT.parent
    records=[json.loads(line) for line in (root/'variant-009d/dataset.host-only.jsonl').read_text().splitlines()]
    values={'records':records,'topic':(VARIANT/'topic-research.txt').read_text().strip(),
            'inference_profile':'hf-fp8-v1','note_retry_policy':'short-note-retry-v1',
            'notebook_context_policy':'9e-v1','notebook_quota_policy':'notebook-exempt-v1',
            'question_pairing_policy':'same-question-v1','source_access_policy':'discovery-only-v1'}
    for key,file in [('selectors','editable-sources'),('access_manifest','access-manifest'),('visible_labels','visible-labels')]:
        values[key]=json.loads((root/'variant-009d'/f'{file}.host-only.json').read_text())
    values['question_ids']=json.loads((VARIANT/'question-ids-same-2.host-only.json').read_text())
    values['round_leaders']=json.loads((VARIANT/'round-leaders-2.host-only.json').read_text())
    values['note_retry_policy']=cli.SHORT_NOTE_POLICY
    original=json.loads((root/'variant-010a/question-ids-same-6.host-only.json').read_text())
    assert all(values['question_ids'][a]==[original[a][0],original[a][2]] for a in AGENTS)
    selected=[next(r for r in records if r['id']==qid) for qid in values['question_ids']['agent-1']]
    paragraphs=[{(p['title'],p['paragraph_text']) for p in r['paragraphs']} for r in selected]
    shared=paragraphs[0]&paragraphs[1]
    assert len(shared)==6 and any(title=='Major League Baseball Most Valuable Player Award' for title,_ in shared)
    settings,pages,tasks=runner.build_settings(**values)
    baseline_inputs={**values,'question_ids':original,'round_leaders':json.loads((root/'variant-009d/round-leaders.host-only.json').read_text())[:6]}
    # Existing 10a uses same corpus/search partition; compare plans, not supplied gold fields.
    baseline=baseline_settings(**baseline_inputs)[0]
    assert settings['discovery_plan']==baseline['discovery_plan'] and settings['access_plan']==baseline['access_plan']
    browser=runner.make_browser(settings,pages,':memory:')
    try:
        for agent in AGENTS:
            foreign=set(settings['access_plan']['allowed_urls'][next(a for a in AGENTS if a!=agent)])-set(settings['access_plan']['allowed_urls'][agent])
            assert not foreign & {r['url'] for r in browser.search('Major League Baseball',agent)['results']}
            paragraph=next(url for url in foreign if '/p/' in url)
            assert 'error' not in browser.call(agent,'open',{'url':paragraph})
    finally:browser.close()
