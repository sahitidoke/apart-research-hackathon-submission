"""Mock-only JSON contracts, natural finals and actual transport request payloads."""
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.simulated_web.answer_format import AnswerFormatClient, EVIDENCE_SCHEMA, parse_evidence_answer
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.timed import run_phase
from orchestrator.simulated_web import test_timed
from orchestrator.simulated_web.test_timed import Clock
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import pair_policy, run_token_pair, load_pair_checkpoint

VALID={'status':'answered','answer':'x','citations':[{'url':'https://docs.test/p/1','quote':'exact words'}]}


@pytest.mark.parametrize('value',[{}, {'status':'answered','answer':'x','citations':[]},
 {'status':'insufficient_evidence','answer':'x','citations':[]},
 {'status':'insufficient_evidence','answer':'','citations':[{'url':'u','quote':'q'}]},
 {'status':'answered','answer':'x','citations':[{'url':'','quote':'q'}]}])
def test_semantic_rejections(value):
    with pytest.raises(ValueError):parse_evidence_answer(json.dumps(value))


def test_abstention_and_duplicate_keys():
    assert parse_evidence_answer('{"status":"insufficient_evidence","answer":"","citations":[]}')['answer']==''
    with pytest.raises(ValueError):parse_evidence_answer('{"status":"answered","status":"insufficient_evidence","answer":"","citations":[]}')


@pytest.mark.parametrize('forced',[False,True])
@pytest.mark.parametrize('content,valid',[(json.dumps(VALID),True),('not JSON',False)])
def test_natural_and_forced_final_no_retries_usage_retained(forced,content,valid):
    requests=[]
    def client(agent,history,timeout,**kwargs):
        requests.append(kwargs)
        if forced and len(requests)==1:
            return ModelResponse({'content':'','tool_calls':[{'function':{'name':'open','arguments':{'url':'https://docs.test/'}}}]},{'eval_count':3,'prompt_eval_count':10,'done_reason':'stop'})
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':10,'done_reason':'stop'})
    with tempfile.TemporaryDirectory() as tmp:
        log=Path(tmp)/'phase.jsonl';history=[]
        row=run_phase(MagicMock(call=MagicMock(return_value={'text':'source'})),AnswerFormatClient(client),history,
                      'Answer phase: 180 seconds.','answer',180,pair_policy(compaction_enabled=False,answer_browser_calls=1),log,clock=Clock())
        assert row['status']==('complete' if valid else 'invalid_final_response')
        assert row['answer']==('x' if valid else '')
        assert row['raw_answer_json']==content
        assert row['generated_tokens_observed']==(10 if forced else 7)
        assert len(requests)==(2 if forced else 1)
        assert ('format_schema' in requests[-1])==forced
        assert 'format_schema' not in requests[0]
        assert 'raw_response' in log.read_text()


def test_transport_schema_only_final_payload():
    client=test_timed.TransportTests().client();payloads=[]
    def factory(*args,**kwargs):
        c=MagicMock();r=io.BytesIO(b'{"message":{"content":"{}"},"done":true,"prompt_eval_count":100}\n');r.status=200
        c.getresponse.return_value=r;c.request.side_effect=lambda *args:payloads.append(json.loads(args[2]));return c
    with patch.object(client,'_start'),patch.object(client,'_idle_baseline',return_value=((42,'100',1234),7)),patch.object(client,'_count_prompt',return_value={'prompt_tokens':100}),patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection',side_effect=factory):
        client('agent-1',[],5,num_predict=1)
        client('agent-1',[],5,num_predict=1,final_only=True,format_schema=EVIDENCE_SCHEMA)
    assert 'format' not in payloads[0] and payloads[0]['tools']
    assert payloads[1]['format']==EVIDENCE_SCHEMA and 'tools' not in payloads[1]
    assert payloads[1]['think'] is False


def test_pair_checkpoint_option_and_legacy_default():
    records,selectors=inputs();ids=[r['id'] for r in records[:3]]
    class JsonClient(Client):
        def __call__(self,*args,**kwargs):return ModelResponse({'content':json.dumps(VALID)},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        run_token_pair(root/'run',records,'topic',JsonClient(),pair_policy(compaction_enabled=False),selectors,pair_protocol='answers_only',question_ids=ids,answer_format='json_evidence')
        cp=load_pair_checkpoint(root/'run/checkpoints/rounds-003')
        try:assert cp.data['settings.json']['answer_format_contract']['schema']==EVIDENCE_SCHEMA
        finally:cp.close()
        with pytest.raises(ValueError):run_token_pair(root/'bad',None,None,JsonClient(),resume_from=root/'run/checkpoints/rounds-003',answer_format='text')
        assert not (root/'bad').exists()


def test_truncated_natural_final_fails_without_retry():
    calls=[]
    def client(*args,**kwargs):
        calls.append(kwargs)
        return ModelResponse({'content':'{"status":"answered"'}, {'eval_count':7,'prompt_eval_count':10,'done_reason':'length'})
    with tempfile.TemporaryDirectory() as tmp:
        row=run_phase(MagicMock(),AnswerFormatClient(client),[], 'Answer phase: 180 seconds.','answer',180,
                      pair_policy(compaction_enabled=False),Path(tmp)/'log',clock=Clock())
        assert row['status']=='invalid_final_response' and row['generated_tokens_observed']==7
        assert len(calls)==1


def test_wrapper_delegates_readiness_and_leaves_administration_unwrapped():
    client=MagicMock()
    wrapped=AnswerFormatClient(client)
    wrapped.ensure_ready(timeout=2)
    client.ensure_ready.assert_called_once_with(timeout=2)
    wrapped.count_context([],timeout=2)
    client.count_context.assert_called_once_with([],timeout=2)
    wrapped('a',[],2,num_predict=20,final_only=False)
    assert 'format_schema' not in client.call_args.kwargs


@pytest.mark.parametrize('json_mode',[False,True])
def test_actual_cold_readiness_path_never_receives_schema(json_mode):
    client=test_timed.TransportTests().client()
    client.ready=False
    payloads=[]
    def factory(*args,**kwargs):
        connection=MagicMock()
        response=io.BytesIO(b'{"message":{"content":"OK"},"done":true,"eval_count":1,"prompt_eval_count":100}\n')
        response.status=200
        connection.getresponse.return_value=response
        connection.request.side_effect=lambda *args:payloads.append(json.loads(args[2]))
        return connection
    target=AnswerFormatClient(client) if json_mode else client
    with patch.object(client,'_start'),patch.object(client,'_idle_baseline',return_value=((42,'100',1234),7)),patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection',side_effect=factory):
        result=target.ensure_ready(timeout=5)
    assert result['status']=='warmed' and client.ready
    assert len(payloads)==1
    assert payloads[0]['messages']==[{'role':'user','content':'Readiness check. Reply OK.'}]
    assert 'format' not in payloads[0] and 'tools' not in payloads[0]
    assert payloads[0]['options']['num_predict']==1


def test_thinking_only_length_preserves_reserved_json_final():
    requests=[]
    def client(*args,**kwargs):
        requests.append(kwargs)
        if len(requests)==1:
            return ModelResponse({'content':'','thinking':'continued research reasoning'},
                                 {'eval_count':1792,'prompt_eval_count':10,'done_reason':'length'})
        return ModelResponse({'content':json.dumps(VALID)}, {'eval_count':7,'prompt_eval_count':10,'done_reason':'stop'})
    with tempfile.TemporaryDirectory() as tmp:
        row=run_phase(MagicMock(),AnswerFormatClient(client),[], 'Answer phase: 180 seconds.','answer',180,
                      pair_policy(compaction_enabled=False),Path(tmp)/'log',clock=Clock())
    assert len(requests)==2
    assert requests[0]['num_predict']==1792 and not requests[0]['final_only']
    assert requests[1]['num_predict']==256 and requests[1]['final_only']
    assert requests[1]['format_schema']==EVIDENCE_SCHEMA
    assert row['status']=='complete' and row['answer']=='x'
    assert row['generated_tokens_observed']==1799
