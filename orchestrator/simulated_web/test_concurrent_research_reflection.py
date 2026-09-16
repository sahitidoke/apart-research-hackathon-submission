"""Finite concurrent10b reference tests; no models or services."""
import json
import sqlite3
import threading
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web import modal_hf_concurrent_research_reflection as cli
from orchestrator.simulated_web.answer_format import EVIDENCE_SCHEMA
from orchestrator.simulated_web.research_reflection import build_settings as serial_settings
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client
from orchestrator.simulated_web.test_same_question_pairing import paired_options

FINAL=json.dumps({'status':'insufficient_evidence','answer':'','citations':[]})


class Endpoint(FP8Client):
    def __init__(self,owner,agent):super().__init__();self.owner=owner;self.agent=agent;self.events=[]
    def __call__(self,agent,history,timeout,**kwargs):
        assert agent==self.agent
        peer='agent-2' if agent=='agent-1' else 'agent-1'
        assert 'PRIVATE_'+peer not in json.dumps(history)
        self.owner.calls.append((agent,json.loads(json.dumps(history)),kwargs))
        prompt=history[-1].get('content','')
        if prompt.startswith(('Initial research:','Answer phase:','Reflection:')):
            with self.owner.lock:self.owner.active+=1;self.owner.peak=max(self.owner.peak,self.owner.active)
            try:
                self.owner.overlap.wait(timeout=5)
                if self.owner.fail and prompt.startswith('Answer phase:'):
                    if agent=='agent-1':
                        error=ValueError('synthetic known-usage failure');error.native_usage={'eval_count':19,'prompt_eval_count':100}
                        error.transport_failure={'generation_dispatched':True,'completion_usage_available':True};raise error
                    assert self.owner.cancelled.wait(timeout=5)
                    error=RuntimeError('cancelled unknown-usage peer');error.native_usage=None
                    error.transport_failure={'generation_dispatched':True,'completion_usage_available':False};raise error
            finally:
                with self.owner.lock:self.owner.active-=1
        if kwargs.get('format_schema'):
            assert kwargs['format_schema']==EVIDENCE_SCHEMA and kwargs['final_only'] and kwargs['num_predict']==256
            content=FINAL
        elif prompt.startswith('Answer phase:'):content='Unconstrained early explanation. '+FINAL
        else:content='PRIVATE_'+agent+(' NOTE' if kwargs.get('final_only') else ' FINDING')
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})


class Owner(FP8Client):
    concurrent_stage_supported=True
    def __init__(self,fail=False):
        super().__init__();self.cancelled=threading.Event();self.endpoints={};self.events=[];self.fail=fail
        self.overlap=threading.Barrier(2);self.lock=threading.Lock();self.active=0;self.peak=0;self.calls=[];self.ready_checks=0
    def endpoint(self,agent):
        if agent not in self.endpoints:self.endpoints[agent]=Endpoint(self,agent)
        return self.endpoints[agent]
    def cancel(self):self.cancelled.set()
    def ensure_ready(self,timeout=120):
        assert self.active==0 and not self.cancelled.is_set();self.ready_checks+=1
        return {'status':'mock_shared_ready'}


def test_overlap_private_histories_reference_schedule_and_exact_persistence(tmp_path):
    owner=Owner();run=tmp_path/'run';original=runner.worker;seen=[]
    def spy(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,shared,gate):
        with sqlite3.connect(folder/(agent+'.sqlite3')) as db:records=db.execute('SELECT author,question_round,stage FROM entry_provenance').fetchall()
        assert not any(r==q and s==index for a,r,s in records)
        seen.append((q,index,agent))
        return original(local,client,history,settings,tasks,policy,folder,path,agent,q,index,roles,first_phase,shared,gate)
    with patch.object(runner,'worker',side_effect=spy):assert runner.run_concurrent_research_reflection(run,owner,**paired_options())['status']=='complete'
    assert owner.peak==2 and owner.active==0 and owner.ready_checks==5 and len(seen)==10
    rows=json.loads((run/'results.json').read_text())
    assert [r['phase_role'] for r in rows]==['initial_research','initial_note']*2+(['answer']*2+['reflection','reflection_note']*2)*2
    assert [r['global_phase_index'] for r in rows]==list(range(16)) and len(owner.calls)==16
    for i,row in enumerate(rows):
        assert (run/row['log_path']).is_file()
        if row['phase_role']=='answer':
            assert row['answer_format_enforcement']=='schema_and_host' and row['final_attempted']
            assert row['generated_token_allowance']==2048 and len(row['model_requests'])==2
            assert sum(r['eval_count'] for r in row['model_requests'])==14
        if row['phase_role']=='reflection':assert row['generated_token_allowance']==4096
        if row['phase_role']=='reflection_note':
            assert row['answer']==rows[i-1]['answer'] and row['reflection_result_index']==i-1
            assert row['generated_token_allowance']==0 and row['reflection_persistence']=='exact_final_entry'
    assert all(len(h)==2 for h in json.loads((run/'histories.json').read_text()).values())
    events=[json.loads(line) for line in (run/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event']=='stage_published'])==5
    for q in (1,2):
        locks=[e['event_order'] for e in events if e['event']=='answer_locked' and e['question_round']==q]
        reflection=next(e['event_order'] for e in events if e['event']=='stage_snapshots_frozen' and e['question_round']==q and e['stage']==3)
        assert len(locks)==2 and max(locks)<reflection
    assert not any(e.get('actor')=='model_voluntary_tool' for e in events)
    exposures=[e for e in events if e['event']=='host_metadata_log_exposure']
    assert len(exposures)==10 and all((run/e['artifact']['artifact']).is_file() for e in exposures)
    appends=[e for e in events if e['event']=='append_attempt']
    assert len(appends)==6 and all(e['actor']=='host_mandatory_note' for e in appends)
    for e in events:
        if e['event']=='phase_finished':assert rows[e['results_index']]['global_phase_index']==e['phase_index']
    assert (run/'checkpoints/rounds-002/concurrent-research-reflection-checkpoint.json').is_file()


def test_failure_settles_both_without_publication_or_reflection(tmp_path):
    owner=Owner(fail=True);run=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='Concurrent stage failed'):runner.run_concurrent_research_reflection(run,owner,**paired_options())
    assert owner.cancelled.is_set() and owner.active==0
    barrier=json.loads((run/'round-01-stage-2/barrier.json').read_text())
    assert barrier['published'] is False and barrier['workers_settled']
    assert not (run/'round-01-stage-3').exists() and not (run/'checkpoints/rounds-001').exists()
    for agent in ('agent-1','agent-2'):
        assert (run/f'round-01-stage-2/{agent}-work/failure.json').is_file()
        assert (run/f'round-01-stage-2/{agent}-work/worker-status.json').is_file()
    rows=json.loads((run/'results.json').read_text());a,b=rows[-2:]
    assert a['generated_tokens_observed']==19 and a['token_accounting_complete']
    assert not b['token_accounting_complete'] and 'native_token_count_missing' in b['limits_reached']
    with sqlite3.connect(run/'wiki.sqlite3') as db:assert db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==2


def test_reference_parity_and_h100_resources():
    inputs=paired_options();settings=cli.validate_fresh('germanwiki-2-010b-fp8-h100-concurrent-10b',inputs)
    serial,_,_=serial_settings(**inputs)
    for key in ('policy','question_ids','system_prompts','research_reflection_policy','notebook_tool_schemas','access_plan','discovery_plan',
                'maximum_generated_tokens','maximum_combined_generated_tokens','answer_support_verifier','reset'):
        assert settings[key]==serial[key]
    resource=cli.resources(settings)
    assert resource['gpu']=='H100' and resource['gpu_memory_gb']==80 and resource['gpu_count']==1
    assert resource['weight_copies']==resource['model_processes']==1 and resource['max_num_seqs']==2
    assert resource['maximum_combined_generated_tokens']==54272
    assert resource['answer_seconds']==180 and resource['answer_final_reserve_tokens']==256
    assert resource['phase_records']==16 and resource['normal_model_phases']==12


def test_cold_readiness_600_only_coordinator_and_phase_limits_unchanged(tmp_path):
    owner=Owner();run=tmp_path/'startup'
    with patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready:
        assert runner.run_concurrent_research_reflection(run,owner,**paired_options())['status']=='complete'
    assert [call.kwargs['timeout'] for call in ready.call_args_list]==[600,300,300,300,300]
    settings=json.loads((run/'settings.json').read_text())
    assert settings['policy']['initial_readiness_timeout_seconds']==300
    assert settings['policy']['readiness_timeout_seconds']==120
    assert settings['policy']['answer_seconds']==180
    assert settings['policy']['preparation_seconds']==settings['policy']['reflection_seconds']==600
    assert settings['maximum_combined_generated_tokens']==54272
    assert json.loads((run/'round-00-stage-1/shared-readiness.json').read_text())['timeout_seconds']==600
    assert json.loads((run/'round-01-stage-2/shared-readiness.json').read_text())['timeout_seconds']==300


def test_launcher_explicit_run_id_preflight_and_required_suffix():
    launcher=Path(__file__).resolve().parents[2]/'research-log/mlb-tokenpair-2026-09-12/variant-010b/concurrent-h100/run-tokenpair-2-010b-fp8-concurrent-h100.sh'
    run_id='germanwiki-2-010b-fp8-explicit-mock-concurrent-10b'
    result=subprocess.run(['bash',str(launcher),'--validate-only','--run-id',run_id],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['run_id']==run_id and report['status']=='validated_no_cloud_actions'
    assert report['resources']['coordinator_readiness']['initial_cold_seconds']==600
    assert report['resources']['hard_job_seconds']==21600 and report['resources']['answer_seconds']==180
    invalid=subprocess.run(['bash',str(launcher),'--validate-only','--run-id','germanwiki-fp8-wrong-suffix'],text=True,capture_output=True,timeout=30)
    assert invalid.returncode!=0 and 'end in -concurrent-10b' in invalid.stderr
    missing=subprocess.run(['bash',str(launcher),'--run-id'],text=True,capture_output=True,timeout=30)
    assert missing.returncode==2 and 'requires one nonempty value' in missing.stderr
