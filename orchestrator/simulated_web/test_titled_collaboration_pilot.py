"""Finite opt-in notebook-title tests; no generation or external service."""
import json
import sqlite3

import pytest

from orchestrator.simulated_web import concurrent_collaboration_pilot as runner
from orchestrator.simulated_web import modal_hf_titled_collaboration_pilot as cli
from orchestrator.simulated_web.notebook_titles import entry_title, INSTRUCTION
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_no_log_collaboration_pilot import inputs
from orchestrator.simulated_web.test_collaboration_pilot_11b2 import Final512Owner, Final512Endpoint
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.collaboration_pilot import persist_final_entry
from orchestrator.simulated_web.runner import ModelResponse


def options():return {**inputs(),'agent_log_policy':'no-agent-history-v1','notebook_title_policy':'agent-first-line-v1'}


def test_policy_actual_input_parity_default_isolation_and_title_edges():
    data=options();after=cli.validate_fresh('pilot-fp8-concurrent-11d',data)
    old={k:v for k,v in data.items() if k!='notebook_title_policy'};before,pages,_=runner.build_settings(**old)
    assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)}=={'notebook_title_policy','system_prompts','notebook_tool_schemas'}
    for agent,prompt in before['system_prompts'].items():assert after['system_prompts'][agent]==prompt+'\n\n'+INSTRUCTION
    assert cli.resources(after)['maximum_combined_generated_tokens']==54272
    assert after['policy']['request_history_mode']=='disabled'
    assert entry_title('\n ## Yankees World Series counts\nEvidence unchanged')=='Yankees World Series counts'
    assert entry_title('No heading supplied.\nStill valid')=='No heading supplied.'
    assert entry_title('X'*130+'\nbody')=='X'*120
    assert entry_title('#\nbody')=='#'
    with pytest.raises(ValueError,match='first-line'):cli.validate_fresh('pilot-fp8-concurrent-11d',old)
    with pytest.raises(ValueError,match='Invalid notebook title'):runner.build_settings(**old,notebook_title_policy='bad')


def test_native_append_exact_body_search_title_provenance_and_legacy(tmp_path):
    data=options();settings,pages,_=runner.build_settings(**data)
    for policy in (None,'agent-first-line-v1'):
        config=dict(settings)
        if policy is None:config.pop('notebook_title_policy')
        browser=make_browser(config,pages,tmp_path/str(policy))
        try:
            body='\n# Yankees appearances evidence\nLiteral body with quotes " and symbols &.\n'
            response=browser.call('agent-1','append_notebook',{'text':body,'notebook':'https://wiki.test/page/'+settings['notebooks']['agent-2']})
            assert 'error' not in response
            slug=response['saved'].split('/')[-1]
            title,saved=browser.db.execute('SELECT title,body FROM pages WHERE slug=?',(slug,)).fetchone()
            assert saved==body
            assert title==('Yankees appearances evidence' if policy else f"Research notebook entry by {settings['visible_labels']['agent-1']}, round 0, {response['created_at']}")
            row=browser.call('agent-2','read_notebook',{'url':response['saved'],'revision':''})
            assert row['author']==settings['visible_labels']['agent-1'] and row['created_at']==response['created_at']
            results=browser.call('agent-2','search',{'query':'Yankees appearances evidence'})['results']
            assert any(r['url']==response['saved'] and r['title']==title for r in results)
            assert browser.history_search_candidates('agent-2')==[]
            assert browser.db.execute('SELECT count(*) FROM audit').fetchone()[0]>=3
        finally:browser.close()


def test_complete_phases_exact_saves_same_generation_and_title_instructions(tmp_path):
    owner=Final512Owner();path=tmp_path/'run'
    opts={**paired_options(),'answer_final_reserve_tokens':512,'agent_log_policy':'no-agent-history-v1','notebook_title_policy':'agent-first-line-v1'}
    assert runner.run_concurrent_collaboration_pilot(path,owner,**opts)['status']=='complete'
    results=json.loads((path/'results.json').read_text())
    assert len(results)==32 and len(owner.finals)==4 and len(owner.calls)==20
    with sqlite3.connect(path/'wiki.sqlite3') as db:
        for title,body in db.execute('SELECT p.title,p.body FROM pages p JOIN entry_provenance e USING(slug)'):assert title==entry_title(body)
    for row in results:
        if row['phase_role'] in ('research1_note','research2_note','reflection_note'):assert row['generated_token_allowance']==0 and row['final_entry_persistence']=='exact_final_entry'
    for _,history,kwargs in owner.calls:
        assert INSTRUCTION in history[0]['content']
        last=history[-1].get('content','')
        if last.startswith(('Research stage ','Reflection:')):assert INSTRUCTION in last
    assert not list(path.rglob('*forced-log*'))
    assert all(len(h)==2 for h in json.loads((path/'histories.json').read_text()).values())


@pytest.mark.parametrize('role',['research1','research2','reflection'])
def test_invalid_final_repair_reuses_title_path_without_body_rewrite(tmp_path,role):
    settings,pages,_=runner.build_settings(**options());browser=make_browser(settings,pages,tmp_path/'db')
    body='# Repair evidence heading\nRecovered exact evidence body.'
    class Client:
        def __call__(self,*args,**kwargs):return ModelResponse({'content':body},{'eval_count':12,'prompt_eval_count':100,'done_reason':'stop'})
    try:
        rows=[];transitions=[];history=[{'role':'system','content':INSTRUCTION}]
        row=persist_final_entry(browser,Client(),history,tmp_path,0,'agent-1','q',settings['notebooks']['agent-1'],{'status':'empty_response','answer':''},rows,transitions,120,role=role)
        assert row['final_entry_persistence']=='repaired_final_entry'
        assert browser.db.execute('SELECT title,body FROM pages WHERE slug LIKE ?',('%-entry-%',)).fetchone()==('Repair evidence heading',body)
        assert len(row['model_requests'])==1 and row['generated_tokens_observed']==12
    finally:browser.close()


def test_agent_heading_changes_existing_bm25_ranking_not_only_display(tmp_path):
    settings,pages,_=runner.build_settings(**options());winners=[]
    for policy in (None,'agent-first-line-v1'):
        config=dict(settings)
        if policy is None:config.pop('notebook_title_policy')
        browser=make_browser(config,pages,tmp_path/str(policy))
        try:
            heading=browser.call('agent-1','append_notebook',{'text':'# rareheadingtoken\nfiller'})['saved']
            body=browser.call('agent-1','append_notebook',{'text':'Other heading\nrareheadingtoken rareheadingtoken'})['saved']
            first=browser.call('agent-2','search',{'query':'rareheadingtoken'})['results'][0]['url']
            winners.append(first==heading)
        finally:browser.close()
    assert winners==[False,True]


class RepairEndpoint(Final512Endpoint):
    def __call__(self,agent,history,timeout,**kwargs):
        result=super().__call__(agent,history,timeout,**kwargs)
        if history[-1].get('content','').startswith('Research stage '):result.message['content']=''
        return result


class RepairOwner(Final512Owner):
    def endpoint(self,agent):
        if agent not in self.endpoints:self.endpoints[agent]=RepairEndpoint(self,agent)
        return self.endpoints[agent]


def test_full32phase_research_repairs_share_exact_title_path(tmp_path):
    owner=RepairOwner();path=tmp_path/'repairs'
    opts={**paired_options(),'answer_final_reserve_tokens':512,'agent_log_policy':'no-agent-history-v1','notebook_title_policy':'agent-first-line-v1'}
    assert runner.run_concurrent_collaboration_pilot(path,owner,**opts)['status']=='complete'
    rows=json.loads((path/'results.json').read_text());assert len(rows)==32
    notes=[r for r in rows if r['phase_role'] in ('research1_note','research2_note')]
    assert len(notes)==8 and all(r['final_entry_persistence']=='repaired_final_entry' for r in notes)
    assert all(r['note_preservation']['persistence_verified'] for r in notes)
    with sqlite3.connect(path/'wiki.sqlite3') as db:
        for title,body in db.execute('SELECT p.title,p.body FROM pages p JOIN entry_provenance e USING(slug)'):assert title==entry_title(body)
    assert not list(path.rglob('*forced-log*'))
