"""Finite mocked concurrent11a ownership, barriers, accounting and failure contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from orchestrator.simulated_web import concurrent_collaboration_pilot as runner
from orchestrator.simulated_web import concurrent_hf_transport as transport
from orchestrator.simulated_web import modal_hf_concurrent_collaboration_pilot as cli
from orchestrator.simulated_web.answer_format import EVIDENCE_SCHEMA
from orchestrator.simulated_web.hf_fp8 import server_command,MODEL,VLLM_VERSION
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.timed_policy import TimedPolicy

FINAL=json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})


class MockEndpoint(FP8Client):
    def __init__(self,owner,agent):
        super().__init__();self.owner=owner;self.agent=agent;self.events=[]
    def __call__(self,agent,history,timeout,**kwargs):
        assert agent==self.agent
        prompt=history[-1].get('content','')
        text=json.dumps(history)
        peer='agent-2' if agent=='agent-1' else 'agent-1'
        assert 'PRIVATE_'+peer not in text
        self.owner.calls.append((agent,json.loads(json.dumps(history)),kwargs))
        if prompt.startswith(('Initial research:','Research stage ','Answer phase:','Reflection:')):
            with self.owner.lock:
                self.owner.active+=1;self.owner.peak=max(self.owner.peak,self.owner.active)
            try:
                self.owner.overlap.wait(timeout=5)
                if self.owner.fail and prompt.startswith('Research stage 1:'):
                    if agent=='agent-1':
                        error=ValueError('synthetic malformed response with known usage')
                        error.native_usage={'eval_count':19,'prompt_eval_count':100}
                        error.transport_failure={'generation_dispatched':True,'completion_usage_available':True}
                        raise error
                    assert self.owner.cancelled.wait(timeout=5)
                    error=RuntimeError('peer cancelled with unknown usage')
                    error.native_usage=None
                    error.transport_failure={'generation_dispatched':True,'completion_usage_available':False}
                    raise error
            finally:
                with self.owner.lock:self.owner.active-=1
        if kwargs.get('format_schema'):
            assert kwargs['format_schema']==EVIDENCE_SCHEMA and kwargs['final_only'] and kwargs['num_predict']==256
            content=FINAL
        elif prompt.startswith('Answer phase:'):content='Early explanatory prose.\n'+FINAL
        else:content='PRIVATE_'+agent+' '+('NOTE' if kwargs.get('final_only') else 'FINDING')
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})


class MockOwner(FP8Client):
    concurrent_stage_supported=True
    def __init__(self,fail=False):
        super().__init__();self.cancelled=threading.Event();self.endpoints={};self.events=[]
        self.overlap=threading.Barrier(2);self.lock=threading.Lock();self.active=0;self.peak=0;self.fail=fail;self.calls=[];self.ready_checks=0
    def endpoint(self,agent):
        if agent not in self.endpoints:self.endpoints[agent]=MockEndpoint(self,agent)
        return self.endpoints[agent]
    def cancel(self):self.cancelled.set()
    def ensure_ready(self,timeout=120):
        assert self.active==0 and not self.cancelled.is_set();self.ready_checks+=1
        return {'status':'mock_shared_ready'}


def test_overlapping_workers_private_histories_barriers_exact_persistence(tmp_path):
    owner=MockOwner();values=paired_options();run=tmp_path/'run'
    seen=[];original=runner.worker
    def spy(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,shared,gate):
        # Both snapshots already exist and have the same published state before workers open them.
        with sqlite3.connect(folder/(agent+'.sqlite3')) as db:
            records=db.execute('SELECT author,question_round,stage FROM entry_provenance').fetchall()
        assert not any(r==q and s==index for a,r,s in records)
        if q and index in (3,4):assert {a for a,r,s in records if r==q and s==index-1}=={'agent-1','agent-2'}
        seen.append((q,index,agent))
        return original(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,shared,gate)
    with patch.object(runner,'worker',side_effect=spy):assert runner.run_concurrent_collaboration_pilot(run,owner,**values)['status']=='complete'
    assert owner.peak==2 and owner.active==0 and owner.ready_checks==9 and len(seen)==18
    rows=json.loads((run/'results.json').read_text());assert len(rows)==32
    assert [r['global_phase_index'] for r in rows]==list(range(32))
    assert all((run/r['log_path']).is_file() for r in rows)
    for i,row in enumerate(rows):
        if row['phase_role']=='answer':assert row['final_attempted'] and row['answer_format_enforcement']=='schema_and_host' and len(row['model_requests'])==2
        if row['phase_role'].endswith('_note') and row['phase_role']!='initial_note':
            assert row['answer']==rows[i-1]['answer'] and row['final_entry_result_index']==i-1
    assert len(owner.calls)==24 #20normal phases plus4explicit finalization requests
    assert all(len(h)==2 for h in json.loads((run/'histories.json').read_text()).values())
    events=[json.loads(line) for line in (run/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event']=='stage_published'])==9
    assert len([e for e in events if e['event']=='answer_locked'])==4
    # This scripted client requests no browser tools; host injection cannot count as voluntary.
    assert not any(e.get('actor')=='model_voluntary_tool' for e in events)
    exposures=[e for e in events if e['event']=='host_metadata_log_exposure']
    assert len(exposures)==18
    assert all((run/e['artifact']['artifact']).is_file() for e in exposures)
    for e in events:
        if e['event']=='append_attempt':assert e['actor']=='host_mandatory_note' and e['phase_role'].endswith('_note')
        if e['event']=='phase_finished':assert rows[e['results_index']]['global_phase_index']==e['phase_index']
    assert len([e for e in events if e['event']=='append_attempt'])==14
    settings=json.loads((run/'settings.json').read_text())
    assert settings['maximum_combined_generated_tokens']==54272 and settings['maximum_generated_tokens']==50176
    assert settings['policy']['answer_seconds']==180 and settings['policy']['final_reserve_tokens']==256
    assert (run/'checkpoints/rounds-002/concurrent-collaboration-pilot-checkpoint.json').is_file()


def test_peer_failure_cancels_settles_and_preserves_known_and_unknown_usage(tmp_path):
    owner=MockOwner(fail=True);run=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='Concurrent stage failed'):runner.run_concurrent_collaboration_pilot(run,owner,**paired_options())
    assert owner.cancelled.is_set() and owner.active==0
    barrier=json.loads((run/'round-01-stage-2/barrier.json').read_text())
    assert barrier['published'] is False and barrier['workers_settled']
    assert not (run/'checkpoints/rounds-001').exists()
    for agent in ('agent-1','agent-2'):
        assert (run/f'round-01-stage-2/{agent}-work/failure.json').is_file()
        assert (run/f'round-01-stage-2/{agent}-work/worker-status.json').is_file()
    rows=json.loads((run/'results.json').read_text())
    a=next(r for r in rows if r['phase_role']=='research1' and r['agent']=='agent-1')
    b=next(r for r in rows if r['phase_role']=='research1' and r['agent']=='agent-2')
    assert a['generated_tokens_observed']==19 and a['token_accounting_complete']
    assert not b['token_accounting_complete'] and 'native_token_count_missing' in b['limits_reached']
    with sqlite3.connect(run/'wiki.sqlite3') as db:assert db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==2
    assert json.loads((run/'manifest.json').read_text())['status']=='failed'
    events=[json.loads(line) for line in (run/'evidence-index.host-only.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']=='phase_finished':assert rows[event['results_index']]['global_phase_index']==event['phase_index']


def owned(tmp_path):
    with patch('orchestrator.simulated_web.hf_transport.sys.platform','linux'),patch('orchestrator.simulated_web.hf_transport.shutil.which',return_value='/mock/vllm'),patch('orchestrator.simulated_web.hf_transport.importlib.metadata.version',return_value=VLLM_VERSION):
        owner=transport.ConcurrentOwnedVllm(TimedPolicy(),tmp_path/'model',tmp_path/'server.log')
    owner.metadata={'checked':True};owner.process=Mock();owner.process.poll.return_value=None;owner.ready=True
    return owner


def test_endpoint_requests_overlap_actual_serialization_and_no_history_mix(tmp_path):
    owner=owned(tmp_path);owner.notebook_tools_enabled=False
    overlap=threading.Barrier(2);captured=[];lock=threading.Lock()
    class Connection:
        def __init__(self,*args,**kwargs):self.sock=Mock()
        def connect(self):pass
        def close(self):pass
        def request(self,method,path,raw,headers):
            self.path=path;self.body=json.loads(raw)
            with lock:captured.append((threading.current_thread().name,path,self.body))
        def getresponse(self):
            if self.path=='/tokenize':value={'tokens':[1,2],'count':2,'max_model_len':65536}
            else:
                overlap.wait(timeout=5)
                value={'choices':[{'finish_reason':'stop','message':{'content':self.body['messages'][0]['content'],'tool_calls':[]}}],'usage':{'completion_tokens':3,'prompt_tokens':2}}
            return SimpleNamespace(status=200,read=lambda limit:json.dumps(value).encode())
    with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',Connection):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(owner.endpoint(a),a,[{'role':'user','content':a}],5,num_predict=8) for a in ('agent-1','agent-2')]
            assert [f.result().message['content'] for f in futures]==['agent-1','agent-2']
    assert len(captured)==4 and owner.active_aborts=={}
    assert owner.endpoint('agent-1').owner is owner.endpoint('agent-2').owner
    assert owner.endpoints['agent-1'].events is not owner.endpoints['agent-2'].events
    for endpoint in owner.endpoints.values():assert len(endpoint.events)==1
    with pytest.raises(ValueError,match='Cross-agent'):owner.endpoint('agent-1')('agent-2',[],1,num_predict=1)


def test_shared_owner_start_once_two_sequences_and_cancel_both_sockets(tmp_path):
    owner=owned(tmp_path);owner.process=None;owner.ready=False
    process=Mock();process.poll.return_value=None
    connection=Mock();connection.getresponse.return_value=SimpleNamespace(status=200,read=lambda limit:json.dumps({'data':[{'id':MODEL}]}).encode())
    with patch('orchestrator.simulated_web.hf_transport.check_loopback_port_available'),patch('orchestrator.simulated_web.hf_transport.subprocess.Popen',return_value=process) as start,patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(owner._start,10**20) for _ in range(2)]
            for future in futures:future.result()
        assert start.call_count==1
        command=start.call_args.args[0]
        assert command[command.index('--max-num-seqs')+1]=='2'
        assert command[command.index('--tensor-parallel-size')+1]=='1'
    callbacks=[Mock(),Mock()]
    for agent,abort in zip(('agent-1','agent-2'),callbacks):owner._add_abort(agent,abort)
    with patch.object(owner,'_stop_locked') as kill:owner.cancel()
    assert kill.call_count==1 and owner.cancelled.is_set()
    for abort in callbacks:abort.assert_called_once()
    with pytest.raises(RuntimeError,match='cancelled'):owner.endpoint('agent-1')('agent-1',[],5,num_predict=8)
    owner.server_log.close()


def test_preflight_h100_single_model_budget_and_historical_server_default():
    settings=cli.validate_fresh('germanwiki-2-011a-fp8-h100-concurrent-11a',paired_options())
    resource=cli.resources(settings)
    assert resource['gpu']=='H100' and resource['gpu_count']==1 and resource['gpu_memory_gb']==80
    assert resource['weight_copies']==resource['model_processes']==1 and resource['max_num_seqs']==2
    assert resource['maximum_combined_generated_tokens']==54272
    assert resource['answer_seconds']==180 and resource['answer_final_reserve_tokens']==256
    default=server_command('/model',65536,8000)
    assert default[default.index('--max-num-seqs')+1]=='1'
    with pytest.raises(ValueError):server_command('/model',65536,8000,max_num_seqs=3)
