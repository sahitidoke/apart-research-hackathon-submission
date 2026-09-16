"""Finite synthetic 9e isolation, support-gating and budget contracts."""
import json
import time
from unittest.mock import patch

import pytest

from orchestrator.simulated_web.answer_support import PROTOCOL,JUDGE_SCHEMA,JUDGE_SYSTEM,judge_answers
from orchestrator.simulated_web.synchronized_exchange import build_settings,run_exchange,load_checkpoint,SHORT_NOTE_POLICY,DISCOVERY_ACCESS_POLICY
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client
from orchestrator.simulated_web.test_synchronized_exchange import inputs


def options():
    return {**inputs(),'inference_profile':'hf-fp8-v1','source_access_policy':DISCOVERY_ACCESS_POLICY,
            'note_retry_policy':SHORT_NOTE_POLICY,'notebook_context_policy':PROTOCOL}


class NineClient(FP8Client):
    def __init__(self,answer):
        super().__init__();self.answer=answer;self.judges=[]
    def __call__(self,agent,history,timeout,**kwargs):
        if kwargs.get('format_schema')==JUDGE_SCHEMA:
            self.judges.append(json.loads(json.dumps(history)))
            assert kwargs['final_only'] and kwargs['num_predict']==1024
            return ModelResponse({'content':'{"verdict":"supported","reason":"Mock support"}'},{'eval_count':7,'done_reason':'stop'})
        self.calls.append((agent,json.loads(json.dumps(history))))
        if history[-1].get('content','').startswith('Answer phase:') or kwargs.get('format_schema'):
            return ModelResponse({'content':json.dumps(self.answer)},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        if kwargs.get('final_only'):
            assert 'format_schema' not in kwargs
            return ModelResponse({'content':'GENERATED_NOTE_'+agent},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


def test_no_automatic_content_valid_answers_reach_isolated_judge(tmp_path):
    settings,pages,_=build_settings(**options())
    source=next(p for p in pages if '/p/' in p['url'])
    answer={'status':'answered','answer':'Submitted result','citations':[{'url':source['url'],'quote':source['text']}]}
    client=NineClient(answer)
    with patch('orchestrator.simulated_web.synchronized_exchange.self_memory',side_effect=AssertionError('automatic memory forbidden')),patch('orchestrator.simulated_web.synchronized_exchange.expose_entries',side_effect=AssertionError('automatic views forbidden')):
        result=run_exchange(tmp_path/'run',client,**options())
    assert result['status']=='complete' and len(client.judges)==4
    assert not list((tmp_path/'run').glob('memory-*-agent-*.json'))
    assert not list((tmp_path/'run').glob('round-*-stage-*/*-views.json'))
    for agent,history in client.calls:
        text=json.dumps(history)
        peer='agent-2' if agent=='agent-1' else 'agent-1'
        assert 'GENERATED_NOTE_'+peer not in text
        assert JUDGE_SYSTEM not in text
    history=json.loads((tmp_path/'run/round-01-histories-before-reset.json').read_text())
    for agent,messages in history.items():
        assert any(m.get('role')=='assistant' and m.get('content')=='GENERATED_NOTE_'+agent for m in messages)
        assert any('request-history' in m.get('content','') for m in messages)
    for history in client.judges:
        assert len(history)==2 and history[0]['content']==JUDGE_SYSTEM
        payload=json.loads(history[1]['content'])
        assert set(payload)=={'question','answer','evidence'}
        assert 'GENERATED_NOTE' not in json.dumps(history)
    summary=json.loads((tmp_path/'run/answer-support/summary.json').read_text())
    assert summary['judge_calls']==4 and summary['generated_tokens_observed']==28
    assert summary['generated_token_allowance_consumed']==4096
    saved,browser,_=load_checkpoint(tmp_path/'run/checkpoints/rounds-002')
    try:
        assert saved['settings.json']['notebook_context_policy']==PROTOCOL
        slug=browser.db.execute("SELECT slug FROM entry_provenance WHERE author='agent-2' LIMIT 1").fetchone()[0]
        assert 'GENERATED_NOTE_agent-2' in browser.call('agent-1','read_notebook',{'url':'https://wiki.test/page/'+slug,'revision':''})['text']
    finally:browser.close()
    assert all('shortest complete' not in p and 'bounded selection' not in p for p in settings['system_prompts'].values())
    assert settings['maximum_combined_generated_tokens']==45056


def judge_case(tmp_path,response=None,answer=None,deadline=None):
    settings=build_settings(**options())[0];settings['provenance']={'model':{'name':'pinned-model'}}
    answer=answer or {'status':'answered','answer':'A','citations':[{'url':'https://docs.test/q/p/1','quote':'unrelated authentic quote'}]}
    rows=[{'phase_role':'answer','agent':'agent-1','question_id':'q','answer':answer['answer'],'structured_answer':answer}]
    seen=[]
    def client(agent,history,timeout,**kwargs):
        seen.append(history)
        if isinstance(response,Exception):raise response
        return response or ModelResponse({'content':'{"verdict":"unsupported","reason":"Unrelated evidence"}'},{'eval_count':9,'done_reason':'stop'})
    tasks={'q':{'question':'Question','answer':'GOLD_SECRET','supports':'GOLD_SUPPORT'}}
    judge_answers(tmp_path,client,rows,tasks,[{'url':'https://docs.test/q/p/1','text':'unrelated authentic quote'}],settings,deadline)
    return json.loads((tmp_path/'answer-support/summary.json').read_text()),seen


def test_authentic_unrelated_quote_is_not_support_and_gold_isolated(tmp_path):
    summary,seen=judge_case(tmp_path)
    assert summary['answers'][0]['quote_validation']['verified']
    assert summary['answers'][0]['verdict']=='unsupported'
    assert 'GOLD_' not in json.dumps(seen)


@pytest.mark.parametrize('url,quote',[('https://wiki.test/page/entry','unrelated authentic quote'),('https://docs.test/q/p/1','fabricated')])
def test_invalid_quotes_preclude_judge(tmp_path,url,quote):
    summary,seen=judge_case(tmp_path,answer={'status':'answered','answer':'A','citations':[{'url':url,'quote':quote}]})
    assert not seen and summary['answers'][0]['verdict']=='uncertain'


@pytest.mark.parametrize('response',[
    ModelResponse({'content':'not json'},{'eval_count':2,'done_reason':'stop'}),
    ModelResponse({'content':'{"verdict":"supported","reason":"yes"}'},{'eval_count':1024,'done_reason':'length'}),
    ModelResponse({'content':'{"verdict":"supported","reason":"yes"}'},{'done_reason':'stop'}),
    ModelResponse({'content':'{"verdict":"supported","reason":"yes"}','thinking':'hidden'},{'eval_count':5,'done_reason':'stop'}),
    RuntimeError('transport failure'),
])
def test_judge_failures_are_failclosed_and_preserved(tmp_path,response):
    summary,seen=judge_case(tmp_path,response)
    assert seen and summary['answers'][0]['verdict']=='uncertain'
    assert summary['generated_token_allowance_consumed']==1024
    assert (tmp_path/'answer-support/0000-request.json').is_file()
    if isinstance(response,Exception) or 'eval_count' not in response.metadata:
        assert summary['calls_with_unknown_usage']==1
        assert summary['answers'][0]['generated_tokens_observed'] is None


def test_deadline_precludes_judge(tmp_path):
    summary,seen=judge_case(tmp_path,deadline=time.monotonic()-1)
    assert not seen and summary['judge_calls']==0
    assert summary['answers'][0]['verdict']=='uncertain'


def test_invalid_protocol_before_artifacts(tmp_path):
    values=options();values['source_access_policy']=None
    with pytest.raises(ValueError,match='9e requires'):run_exchange(tmp_path/'bad',NineClient({}),**values)
    assert not (tmp_path/'bad').exists()


def test_invalid_final_json_never_reaches_judge(tmp_path):
    settings=build_settings(**options())[0];settings['provenance']={'model':{}}
    answer={'status':'answered','answer':'A','citations':[{'url':'https://docs.test/q/p/1','quote':'yes'}]}
    row={'phase_role':'answer','agent':'agent-1','question_id':'q','answer':'A','structured_answer':answer,
         'status':'invalid_final_response','final_response_valid':False}
    with patch('orchestrator.simulated_web.test_9e.NineClient.__call__',side_effect=AssertionError('must not call')):
        judge_answers(tmp_path,NineClient(answer),[row],{'q':{'question':'Q'}},[{'url':'https://docs.test/q/p/1','text':'yes'}],settings)
    summary=json.loads((tmp_path/'answer-support/summary.json').read_text())
    assert summary['judge_calls']==0 and summary['answers'][0]['verdict']=='uncertain'


def test_interrupted_verifier_preserves_solver_and_marks_manifest(tmp_path):
    settings,pages,_=build_settings(**options());source=next(p for p in pages if '/p/' in p['url'])
    answer={'status':'answered','answer':'A','citations':[{'url':source['url'],'quote':source['text']}]}
    with patch('orchestrator.simulated_web.synchronized_exchange.judge_answers',side_effect=KeyboardInterrupt('mock interrupted')):
        with pytest.raises(KeyboardInterrupt):run_exchange(tmp_path/'run',NineClient(answer),**options())
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['status']=='verification_interrupted' and manifest['solver_status']=='complete'
    assert (tmp_path/'run/results.json').is_file() and (tmp_path/'run/checkpoints/rounds-002/synchronized-checkpoint.json').is_file()


def test_zero_answers_has_truthful_summary(tmp_path):
    settings=build_settings(**options())[0];settings['provenance']={'model':{}}
    result=judge_answers(tmp_path,None,[],{},[],settings)
    assert result['status']=='no_answers_to_assess'
    summary=json.loads((tmp_path/result['artifact']).read_text())
    assert summary['status']=='no_answers_to_assess' and summary['answers']==[]
