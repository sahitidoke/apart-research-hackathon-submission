"""Opt-in early-answer drafts must pass one explicit constrained final call."""
from dataclasses import replace
import json

import pytest

from orchestrator.simulated_web.answer_format import AnswerFormatClient,EVIDENCE_SCHEMA
from orchestrator.simulated_web.collaboration_pilot import build_settings
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.timed import run_phase
from orchestrator.simulated_web.timed_policy import TimedPolicy

FINAL=json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})


def phase(tmp_path,client,required=True,reserve=256):
    settings,pages,_=build_settings(**paired_options());browser=make_browser(settings,pages,':memory:')
    history=[{'role':'system','content':'Use evidence.'}]
    try:
        row=run_phase(browser,AnswerFormatClient(client,require_explicit_finalization=required),history,'Answer phase: 180 seconds.','answer',180,replace(TimedPolicy(**settings['policy']),final_reserve_tokens=reserve),tmp_path/'phase.jsonl',notebook_quota_exempt=True)
        return row,history
    finally:browser.close()


@pytest.mark.parametrize('draft',[FINAL,'An explanatory paragraph.\n'+FINAL])
def test_early_prose_and_json_are_accounted_drafts_then_schema_final(tmp_path,draft):
    requests=[]
    def client(agent,history,timeout,**kwargs):
        requests.append((json.loads(json.dumps(history)),kwargs))
        if len(requests)==1:
            assert not kwargs['final_only'] and 'format_schema' not in kwargs
            return ModelResponse({'content':draft},{'eval_count':80,'done_reason':'stop'})
        assert kwargs['final_only'] and kwargs['format_schema']==EVIDENCE_SCHEMA and kwargs['num_predict']==256
        assert any(m.get('content')==draft for m in history)
        return ModelResponse({'content':FINAL},{'eval_count':25,'done_reason':'stop'})
    row,history=phase(tmp_path,client)
    assert len(requests)==2 and row['status']=='complete' and row['final_attempted']
    assert row['generated_tokens_observed']==105 and row['generated_token_allowance']==2048
    assert row['early_answer_draft']==draft and row['answer_format_enforcement']=='schema_and_host'
    assert row['budget_seconds']==180 and row['final_reserve_tokens']==256


@pytest.mark.parametrize('message,reason',[
    ({'content':'','tool_calls':[]},'stop'),
    ({'content':FINAL},'length'),
    ({'content':FINAL,'thinking':'hidden'},'stop'),
    ({'content':FINAL,'tool_calls':[{'function':{'name':'open','arguments':{'url':'https://docs.test/'}}}]},'stop'),
])
def test_bad_explicit_final_is_not_repaired_or_submitted(tmp_path,message,reason):
    requests=[]
    def client(agent,history,timeout,**kwargs):
        requests.append(kwargs)
        return ModelResponse({'content':FINAL} if len(requests)==1 else message,{'eval_count':9,'done_reason':'stop' if len(requests)==1 else reason})
    row,_=phase(tmp_path,client)
    assert len(requests)==2 and row['status']=='invalid_final_response'
    assert row['answer']=='' and row['browser_calls']==0 and row['generated_tokens_observed']==18


def test_historical_early_json_path_remains_default(tmp_path):
    requests=[]
    def client(*args,**kwargs):
        requests.append(kwargs);return ModelResponse({'content':FINAL},{'eval_count':9,'done_reason':'stop'})
    row,_=phase(tmp_path,client,required=False)
    assert len(requests)==1 and row['status']=='complete' and not row['final_attempted']
    assert row['answer_format_enforcement']=='prompt_and_host'


@pytest.mark.parametrize('required',[False,True])
def test_observed31token_tool_fragment_uses_reserved512_only_when_explicit(tmp_path,required):
    requests=[];counts=[633,99,181,592,31]
    fragment='\n\n<tool_call>\n<'
    def client(agent,history,timeout,**kwargs):
        requests.append(kwargs)
        if len(requests)<=5:
            assert not kwargs['final_only']
            if len(requests)==5:assert kwargs['num_predict']==31
            return ModelResponse({'content':fragment if len(requests)==5 else '', 'thinking':'research' if len(requests)<5 else ''},{'eval_count':counts[len(requests)-1],'done_reason':'length'})
        assert kwargs['final_only'] and kwargs['num_predict']==512 and kwargs['format_schema']==EVIDENCE_SCHEMA
        assert any(m.get('content')==fragment for m in history)
        return ModelResponse({'content':FINAL},{'eval_count':40,'done_reason':'stop'})
    row,history=phase(tmp_path,client,required=required,reserve=512)
    assert row['generated_token_allowance']==2048 and row['browser_calls']==0
    assert row['generated_tokens_observed']==(1576 if required else 1536)
    assert row['final_attempted'] is required
    assert row['status']==('complete' if required else 'invalid_final_response')
    assert len(requests)==(6 if required else 5)
    assert not any(m.get('tool_calls') for m in history)


def test_owner_cancellation_keeps_unknown_usage_without_timeout_flags(tmp_path):
    def client(*args,**kwargs):
        error=RuntimeError('HF request cancelled by shared owner')
        error.native_usage=None
        error.transport_failure={'reason':'owner_cancellation','generation_dispatched':True,'completion_usage_available':False}
        raise error
    result,_=phase(tmp_path,client,reserve=512)
    assert result['status']=='error'
    events=[json.loads(line) for line in (tmp_path/'phase.jsonl').read_text().splitlines()]
    row=next(e['result'] for e in events if e['event']=='result')
    assert not row['safety_timeout_hit'] and not row['token_accounting_complete']
    assert 'native_token_count_missing' in row['limits_reached']
    assert 'wall_time' not in row['limits_reached'] and 'safety_timeout' not in row['limits_reached']
    assert row['model_requests'][0]['transport_failure']['reason']=='owner_cancellation'
