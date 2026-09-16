"""No models: completed parent continuation, citation checks, retries and durable restore."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from orchestrator.simulated_web.evidence_feedback import BOUNDARY, SUCCESS, FAILURE, verify_citations
from orchestrator.simulated_web.pair_extension import BOUNDARY_SEPARATOR, extension_settings, load_extension_checkpoint, run_extension
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.source_discovery import dataset_digest
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair


@pytest.fixture
def parent(tmp_path):
    records,selectors=inputs()
    extra=json.loads(json.dumps(records))
    for index,r in enumerate(extra):
        r['id']+='n';r['question']=f'Nanjing question {index}?'
        for p in r['paragraphs']:p['title']='Nanjing '+p['title'];p['paragraph_text']='Nanjing '+p['paragraph_text']
    data=records+extra
    manifest={'schema':'source-access-v1','dataset_sha256':dataset_digest(data),
              'corpus_questions':{'agent-1':[r['id'] for r in records],'agent-2':[r['id'] for r in extra]}}
    ids={'agent-1':[r['id'] for r in extra[:3]],'agent-2':[r['id'] for r in records[:3]]}
    run=tmp_path/'parent'
    run_token_pair(run,data,'topic',Client(),pair_policy(compaction_enabled=False,browser_retention='question_boundary',preparation_generated_tokens=2048,answer_generated_tokens=2048,preparation_browser_calls=4,answer_browser_calls=4),selectors,
                   pair_protocol='question_research',question_ids=ids,access_manifest=manifest,log_exposure='forced',prompt_condition='reward_persistence')
    return run/'checkpoints/rounds-003'


class EvidenceClient(Client):
    def __init__(self,citation,mode='pass'):
        super().__init__();self.citation=citation;self.mode=mode;self.answers={}
    def __call__(self,agent,history,timeout,**kwargs):
        self.calls.append((agent,json.loads(json.dumps(history))))
        if history[-1]['content'].startswith('Question research:'):
            content='I investigated the current question.'
        else:
            question=re.search(r'Question: (.*?)\n',history[-1]['content']).group(1)
            key=(agent,question);self.answers[key]=self.answers.get(key,0)+1
            fail=self.mode=='fail' or (self.mode=='mixed' and agent=='agent-1' and self.answers[key]==1)
            value={'status':'insufficient_evidence','answer':'','citations':[]} if fail else {'status':'answered','answer':'ungraded answer','citations':[self.citation]}
            content=json.dumps(value)
        return ModelResponse({'content':content},{'eval_count':7,'prompt_eval_count':100,'done_reason':'stop'})


def source_citation(parent):
    pages=json.loads((parent/'pages.json').read_text())
    page=next(p for p in pages if '/p/' in p['url'])
    return {'url':page['url'],'quote':page['text']},pages


def test_verifier_all_citations_original_not_current_edits(parent):
    citation,pages=source_citation(parent)
    good={'status':'answered','answer':'deliberately ungraded','citations':[citation]}
    assert verify_citations(good,pages)['verified']
    normalized={**citation,'quote':' \n '+citation['quote'].replace(' ',' \t ')+'  '}
    assert verify_citations({**good,'citations':[normalized]},pages)['verified']
    for bad in ({**citation,'url':'https://example.com/source'}, {**citation,'url':citation['url']+'?x=1'}, {**citation,'quote':'forged self-edit text'}, {'url':'https://docs.test/','quote':'Browse'}):
        assert not verify_citations({**good,'citations':[citation,bad]},pages)['verified']
    checkpoint=load_pair_checkpoint(parent)
    try:
        # A database edit does not alter the archived source payload used by verifier.
        checkpoint.browser.db.execute('UPDATE source_pages SET body=?',('forged self-edit text',))
        assert not verify_citations({**good,'citations':[{**citation,'quote':'forged self-edit text'}]},checkpoint.data['pages.json'])['verified']
        assert citation['url'] not in checkpoint.browser.access_allowed_urls['agent-2']
        assert verify_citations(good,checkpoint.data['pages.json'])['verified']
    finally:checkpoint.close()


@pytest.mark.parametrize('mode,expected_phases',[('pass',28),('mixed',42),('fail',56)])
def test_extension_unused_questions_attempt_bounds_feedback_and_parent_unchanged(parent,tmp_path,mode,expected_phases):
    before={p.name:p.read_bytes() for p in parent.iterdir()}
    citation,_=source_citation(parent);client=EvidenceClient(citation,mode)
    root=tmp_path/'child'
    result=run_extension(root,client,parent_from=parent)
    assert result['status']=='complete' and result['completed_rounds']==7
    assert len(client.calls)==expected_phases
    rows=json.loads((root/'results.json').read_text())
    assert len(rows)==expected_phases and max(r['attempt'] for r in rows)<=1
    settings=json.loads((root/'settings.json').read_text())
    old=json.loads((parent/'settings.json').read_text())
    for agent in ('agent-1','agent-2'):
        assert len(settings['orders'][agent])==7 and not set(settings['orders'][agent]) & set(old['schedule']['orders'][agent])
        assert set(r['question_id'] for r in rows if r['agent']==agent)==set(settings['orders'][agent])
    assert len(list((root/'checkpoints').glob('rounds-*')))==8
    assert len(list(root.glob('forced-log-*')))==expected_phases
    assert len(list(root.glob('browser-retention-*')))==14
    histories=json.loads((root/'histories.json').read_text())
    for history in histories.values():
        feedback=[m['content'] for m in history if m.get('content') in (SUCCESS,FAILURE)]
        assert feedback and all(citation['url'] not in text and citation['quote'] not in text for text in feedback)
        assert history[0]['role']=='system' and history[0]['content'].endswith(BOUNDARY_SEPARATOR+BOUNDARY)
    loaded,data=load_extension_checkpoint(root/'checkpoints/rounds-007')
    loaded.close()
    assert data['state.json']['completed_rounds']==7
    assert before=={p.name:p.read_bytes() for p in parent.iterdir()}
    assert run_extension(tmp_path/'unused',client,resume_from=root/'checkpoints/rounds-007')['status']=='already_complete'
    assert not (tmp_path/'unused').exists()


def test_interrupted_resume_no_completed_question_replay(parent,tmp_path):
    citation,_=source_citation(parent)
    def stop(path):
        if path.name=='rounds-001':raise RuntimeError('simulated interruption')
    first=tmp_path/'first'
    with pytest.raises(RuntimeError,match='simulated interruption'):
        run_extension(first,EvidenceClient(citation),parent_from=parent,checkpoint_callback=stop)
    checkpoint=first/'checkpoints/rounds-001'
    before={str(p.relative_to(checkpoint)):p.read_bytes() for p in checkpoint.rglob('*') if p.is_file()}
    client=EvidenceClient(citation);child=tmp_path/'resumed'
    result=run_extension(child,client,resume_from=checkpoint)
    assert result['completed_rounds']==7 and len(client.calls)==24
    firstrows=json.loads((first/'results.json').read_text())
    rows=json.loads((child/'results.json').read_text())
    assert rows[:4]==[{**r,'log_path':str(first/r['log_path'])} for r in firstrows]
    assert before=={str(p.relative_to(checkpoint)):p.read_bytes() for p in checkpoint.rglob('*') if p.is_file()}


def test_initial_checkpoint_safety_stop_and_invalid_parent_no_artifacts(parent,tmp_path):
    citation,_=source_citation(parent);client=EvidenceClient(citation)
    root=tmp_path/'stopped'
    assert run_extension(root,client,parent_from=parent,job_deadline=0)['status']=='job_safety_stop'
    assert not client.calls
    loaded,data=load_extension_checkpoint(root/'checkpoints/rounds-000');loaded.close()
    assert data['state.json']['completed_rounds']==0
    original=json.loads((parent/'histories.json').read_text())
    for agent in original:
        assert data['histories.json'][agent][1:]==original[agent][1:]
        assert data['histories.json'][agent][0]['content']==original[agent][0]['content']+BOUNDARY_SEPARATOR+BOUNDARY
    with pytest.raises(ValueError):run_extension(tmp_path/'bad',client,parent_from=parent,resume_from=root/'checkpoints/rounds-000')
    assert not (tmp_path/'bad').exists()


def test_resume_rejects_changed_settings_and_hashes(parent,tmp_path):
    citation,_=source_citation(parent);root=tmp_path/'run'
    run_extension(root,EvidenceClient(citation),parent_from=parent,job_deadline=0)
    checkpoint=root/'checkpoints/rounds-000'
    settingpath=checkpoint/'settings.json';settings=json.loads(settingpath.read_text());settings['maximum_attempts']=3
    settingpath.write_text(json.dumps(settings))
    with pytest.raises(ValueError,match='hash mismatch'):load_extension_checkpoint(checkpoint)
    manifestpath=checkpoint/'extension-checkpoint.json';manifest=json.loads(manifestpath.read_text())
    manifest['files_sha256']['settings.json']=hashlib.sha256(settingpath.read_bytes()).hexdigest();manifestpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='settings mismatch'):load_extension_checkpoint(checkpoint)


def test_partial_round_artifacts_preserved_and_resume_restarts_only_unfinished_round(parent,tmp_path):
    citation,_=source_citation(parent)
    class PartialFailure(EvidenceClient):
        def __call__(self,*args,**kwargs):
            if len(self.calls)==5:raise RuntimeError('partial round transport failure')
            return super().__call__(*args,**kwargs)
    first=tmp_path/'partial'
    with pytest.raises(RuntimeError,match='partial round transport failure'):
        run_extension(first,PartialFailure(citation),parent_from=parent)
    assert (first/'phase-05.jsonl').exists()
    assert json.loads((first/'state.json').read_text())['status']=='failed'
    assert len(json.loads((first/'results.json').read_text()))==6
    client=EvidenceClient(citation)
    run_extension(tmp_path/'continued',client,resume_from=first/'checkpoints/rounds-001')
    assert len(client.calls)==24
    assert (first/'phase-05.jsonl').exists()
