"""Mocked asymmetric retry and explicit immutable 9d checkpoint migration."""
import hashlib
import json
from pathlib import Path

import pytest
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.synchronized_exchange import run_exchange,load_checkpoint,migrated_settings,SHORT_NOTE_POLICY
from orchestrator.simulated_web.test_synchronized_exchange import inputs,Client


class RetryClient(Client):
    def __init__(self):
        super().__init__();self.limits=[];self.pending=False
    def __call__(self,agent,history,timeout,**kwargs):
        if kwargs.get('final_only'):
            self.limits.append(kwargs['num_predict'])
            assert '150–200 words' in history[-1]['content']
            if not self.pending:
                self.pending=True
                return ModelResponse({'content':'unfinished draft'}, {'eval_count':512,'prompt_eval_count':100,'done_reason':'length'})
            assert 'Shorten the failed draft' in history[-1]['content']
            self.pending=False
            return ModelResponse({'content':'Concise complete entry'}, {'eval_count':20,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


def test_asymmetric_retry_and_immutable_checkpoint_migration(tmp_path):
    original=tmp_path/'original'
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('checkpoint boundary')
    with pytest.raises(RuntimeError,match='checkpoint boundary'):run_exchange(original,Client(),**inputs(),checkpoint_callback=stop)
    cp=original/'checkpoints/rounds-001';hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir()}
    data,b,_=load_checkpoint(cp);original_entries=b.db.execute('SELECT * FROM pages ORDER BY slug').fetchall();b.close()
    migrated=migrated_settings(data,cp)
    assert migrated['note_policy_migration']['legacy_prefix_generated_allowance']==16384
    assert migrated['note_policy_migration']['remaining_generated_allowance']==20480
    assert data['settings.json'].get('note_retry_policy') is None
    client=RetryClient();dest=tmp_path/'new'
    assert run_exchange(dest,client,resume_from=cp,migrate_note_retry=True)['status']=='complete'
    assert client.limits==[512,1024]*8
    new,b,_=load_checkpoint(dest/'checkpoints/rounds-002')
    try:
        assert all(b.db.execute('SELECT * FROM pages WHERE slug=?',(row[0],)).fetchone()==row for row in original_entries)
    finally:b.close()
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir()}
    assert new['settings.json']['note_retry_policy']==SHORT_NOTE_POLICY
    notes=[r for r in new['results.json'][16:] if r['phase_role'].endswith('note')]
    assert all(r['generated_token_allowance']==1536 and r['generated_tokens_observed']==532 and r['note_preservation']['persistence_verified'] for r in notes)
    assert all(r['answer']=='Concise complete entry' for r in notes)
    with pytest.raises(ValueError,match='original'):migrated_settings(new,dest/'checkpoints/rounds-002')


def test_both_length_rejections_report_real_caps_and_no_save(tmp_path):
    class Length(Client):
        def __call__(self,agent,history,timeout,**kwargs):
            if kwargs.get('final_only'):return ModelResponse({'content':'truncated'}, {'eval_count':kwargs['num_predict'],'prompt_eval_count':100,'done_reason':'length'})
            return super().__call__(agent,history,timeout,**kwargs)
    with pytest.raises(RuntimeError,match='truncated at 1024 generated tokens'):
        run_exchange(tmp_path/'fail',Length(),**inputs(),note_retry_policy=SHORT_NOTE_POLICY)
    note=json.loads((tmp_path/'fail/results.json').read_text())[-1]
    assert note['generated_tokens_observed']==1536 and note['host_persistence_actions']==[]
    assert [r['eval_count'] for r in note['model_requests']]==[512,1024]


def test_migration_requires_resume_before_artifacts(tmp_path):
    with pytest.raises(ValueError,match='requires resume'):run_exchange(tmp_path/'invalid',Client(),**inputs(),migrate_note_retry=True)
    assert not (tmp_path/'invalid').exists()
