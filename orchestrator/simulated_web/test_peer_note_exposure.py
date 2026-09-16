"""Bounded deterministic forced delivery; native-counting calls are mocked, never generation."""
from copy import deepcopy
import json
import sqlite3
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web.peer_note_exposure import POLICY,deliver,validate_ledger
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.reference_resume import prepare_resume,load_checkpoint
from orchestrator.simulated_web.test_reference_12a import six,PARENT
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner,Final512Endpoint


class Counter:
    def __init__(self,count=100):self.calls=[];self.count=count
    def count_context(self,history,timeout=15):
        self.calls.append(deepcopy(history));assert timeout==15
        return {'method':'mock-native-chat-tokenize','prompt_tokens':self.count+len(self.calls)}


def options(seed=1):return {**six(seed),'peer_note_exposure_policy':POLICY}


def test_packet_cap_true_author_order_repeat_and_empty(tmp_path):
    settings,pages,_=runner.build_settings(**options());b=make_browser(settings,pages,tmp_path/'wiki.sqlite3');history=[];client=Counter()
    try:
        empty=deliver(b.db,client,history,settings,'agent-1',1,2,tmp_path)
        assert not empty['delivered'] and not client.calls and empty['input_token_cost']['marginal_input_tokens']==0
        # Peer crosswrites into recipient directory; author, not destination, selects it.
        body='# Long descriptive peer note\n'+'é🙂'*1600
        saved=b.call('agent-2','append_notebook',{'notebook':'https://wiki.test/page/'+settings['notebooks']['agent-1'],'text':body})
        b.call('agent-1','append_notebook',{'text':'# Later self note'})
        audit_count=b.db.execute('SELECT count(*) FROM audit').fetchone()[0]
        a=deliver(b.db,client,history,settings,'agent-1',1,2,tmp_path)
        assert a['sender']=='agent-2' and a['document']['url']==saved['saved']
        assert a['document']['delivered_characters']==2000 and a['document']['truncated']
        assert history[-1]['content'].endswith(body[:2000]) and a['native_context']['marginal_input_tokens']==1
        assert b.db.execute('SELECT count(*) FROM audit').fetchone()[0]==audit_count
        repeat=deliver(b.db,client,history,settings,'agent-1',1,3,tmp_path)
        assert repeat['packet']==a['packet'] and len(history)==2
        # Future/current publication excluded even if a fixture contains it.
        b.round_index=1;b.stage_index=3;b.call('agent-2','append_notebook',{'text':'# Future peer note'})
        same=deliver(b.db,client,[],settings,'agent-1',1,3,tmp_path)
        assert same['packet']==a['packet']
        ledger=[a,repeat,deliver(b.db,Counter(),[],settings,'agent-2',1,2,tmp_path),deliver(b.db,Counter(),[],settings,'agent-2',1,3,tmp_path)]
        validate_ledger(b.db,ledger,settings,1)
        for field in ('packet','candidate_order','native_context'):
            damaged=deepcopy(ledger)
            if field=='packet':damaged[0][field]['content']+='changed'
            elif field=='candidate_order':damaged[0][field]=[]
            else:damaged[0][field]['after']['prompt_tokens']=-1
            with pytest.raises(ValueError):validate_ledger(b.db,damaged,settings,1)
    finally:b.close()


def test_native_context_matches_phase_capacity_and_failure_no_delivery(tmp_path):
    settings,pages,_=runner.build_settings(**options());b=make_browser(settings,pages,tmp_path/'wiki.sqlite3')
    try:
        b.call('agent-2','append_notebook',{'text':'# Peer note'})
        history=[]
        assert deliver(b.db,Counter(40000),history,settings,'agent-1',1,2,tmp_path)['delivered']
        history=[]
        with pytest.raises(ValueError,match='capacity'):deliver(b.db,Counter(60000),history,settings,'agent-1',1,2,tmp_path)
        failed=json.loads((tmp_path/'peer-note-delivery.json').read_text())
        assert not history and not failed['delivered'] and failed['added_packets']==0
    finally:b.close()


class ExposureEndpoint(Final512Endpoint):
    def __call__(self,*args,**kwargs):
        result=super().__call__(*args,**kwargs)
        result.message['content']=result.message.get('content','').replace('PRIVATE_','PUBLIC_NOTE_')
        return result
    def count_context(self,history,timeout=15):
        self.owner.counts.append((self.agent,deepcopy(history)))
        return {'method':'mock-native-chat-tokenize','prompt_tokens':100+sum(len(m.get('content','')) for m in history)//10}


class ExposureOwner(Final512Owner):
    def __init__(self):super().__init__();self.counts=[]
    def endpoint(self,agent):
        if agent not in self.endpoints:self.endpoints[agent]=ExposureEndpoint(self,agent)
        return self.endpoints[agent]


def test_trajectory_ledger_barrier_reset_resume_and_baseline_parity(tmp_path):
    base,pages,_=runner.build_settings(**six(1));settings,other,_=runner.build_settings(**options())
    assert {k for k in settings.keys()|base.keys() if settings.get(k)!=base.get(k)}=={'peer_note_exposure_policy','peer_note_exposure','system_prompts','automatic_notebook_insertion','bounded_context'}
    for a in base['system_prompts']:
        assert 'Notebook bodies are never automatically inserted or restored.' in base['system_prompts'][a]
        assert settings['system_prompts'][a]==base['system_prompts'][a].replace('Notebook bodies are never automatically inserted or restored. ','')
    assert settings['policy']['answer_seconds']==180 and pages==other
    with pytest.raises(ValueError):runner.build_settings(**options(),answer_safety_seconds=600)
    with pytest.raises(ValueError):runner.build_settings(**options(),ablation_policy='generic-titles-v1')
    with pytest.raises(ValueError):prepare_resume(PARENT,options(0),runner.build_settings)
    dest=tmp_path/'partial';owner=ExposureOwner()
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('bounded stop')
    with pytest.raises(RuntimeError,match='bounded stop'):
        runner.run_concurrent_research_reflection(dest,owner,**options(),checkpoint_callback=stop)
    cp=dest/'checkpoints/rounds-001';data=load_checkpoint(cp,runner.build_settings);ledger=data['state.json']['peer_note_deliveries']
    assert len(ledger)==4 and all(r['delivered'] for r in ledger)
    assert sum(r['tokenizer_requests'] for r in ledger)==8 and owner.ready_checks==3
    events=[json.loads(line) for line in (dest/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event']=='host_peer_note_exposure'])==4
    assert len({r['boundary_id'] for r in ledger})==4
    assert all(not any(m.get('content','').startswith('Research document\n') for m in h) for h in data['histories.json'].values())
    for actor,history,kwargs in owner.calls:
        prompt=history[-1].get('content','')
        packets=[m for m in history if m.get('content','').startswith('Research document\n')]
        if prompt.startswith('Initial research:'):assert not packets
        elif prompt.startswith('Answer phase:'):assert len(packets)==1
        elif prompt.startswith('Reflection:'):assert len(packets)==2
    assert prepare_resume(cp,options(),runner.build_settings)['extension'] is False
    with pytest.raises(ValueError):prepare_resume(cp,six(1),runner.build_settings)
    next_run=tmp_path/'continued'
    def stop2(cp):
        if cp.name=='rounds-002':raise RuntimeError('bounded stop')
    with pytest.raises(RuntimeError,match='bounded stop'):
        runner.run_concurrent_research_reflection(next_run,ExposureOwner(),**options(),resume_from=cp,checkpoint_callback=stop2)
    resumed=load_checkpoint(next_run/'checkpoints/rounds-002',runner.build_settings)['state.json']['peer_note_deliveries']
    assert len(resumed)==8 and resumed[:4]==ledger
    assert len({r['boundary_id'] for r in resumed})==8


def test_count_failure_settles_pair_before_answer_generation(tmp_path):
    owner=ExposureOwner();original=ExposureEndpoint.count_context
    def fail(self,history,timeout=15):
        if self.agent=='agent-1' and any(m.get('content','').startswith('Research document\n') for m in history):raise ValueError('mock tokenizer failure')
        return original(self,history,timeout)
    with patch.object(ExposureEndpoint,'count_context',fail):
        with pytest.raises(RuntimeError,match='both workers settled'):
            runner.run_concurrent_research_reflection(tmp_path/'failure',owner,**options())
    assert not any(h[-1].get('content','').startswith('Answer phase:') for a,h,k in owner.calls)
    barrier=json.loads((tmp_path/'failure/round-01-stage-2/barrier.json').read_text())
    assert not barrier['published'] and barrier['workers_settled']
    assert (tmp_path/'failure/checkpoints/rounds-000').is_dir()
