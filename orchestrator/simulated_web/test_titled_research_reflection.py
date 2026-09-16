"""Finite opt-in notebook-title tests; no generation or external service."""
import json
import sqlite3

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web import modal_hf_titled_research_reflection as cli
from orchestrator.simulated_web.notebook_titles import entry_title, INSTRUCTION
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_complementary_research_reflection import inputs
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.research_reflection import persist_reflection
from orchestrator.simulated_web.runner import ModelResponse


def options():return {**inputs(),'agent_log_policy':'no-agent-history-v1','notebook_title_policy':'agent-first-line-v1'}


def test_policy_actual_input_parity_default_isolation_and_title_edges():
    data=options();after=cli.validate_fresh('pilot-fp8-concurrent-10e',data)
    old={k:v for k,v in data.items() if k!='notebook_title_policy'};before,pages,_=runner.build_settings(**old)
    assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)}=={'notebook_title_policy','system_prompts','notebook_tool_schemas'}
    for agent,prompt in before['system_prompts'].items():assert after['system_prompts'][agent]==prompt+'\n\n'+INSTRUCTION
    assert cli.resources(after)['maximum_combined_generated_tokens']==54272
    assert after['policy']['request_history_mode']=='disabled'
    assert entry_title('\n ## Yankees World Series counts\nEvidence unchanged')=='Yankees World Series counts'
    assert entry_title('No heading supplied.\nStill valid')=='No heading supplied.'
    assert entry_title('X'*130+'\nbody')=='X'*120
    assert entry_title('#\nbody')=='#'
    with pytest.raises(ValueError,match='first-line'):cli.validate_fresh('pilot-fp8-concurrent-10e',old)
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
    assert runner.run_concurrent_research_reflection(path,owner,**opts)['status']=='complete'
    results=json.loads((path/'results.json').read_text())
    assert len(results)==16 and len(owner.finals)==4
    with sqlite3.connect(path/'wiki.sqlite3') as db:
        for title,body in db.execute('SELECT p.title,p.body FROM pages p JOIN entry_provenance e USING(slug)'):assert title==entry_title(body)
    for row in results:
        if row['phase_role']=='reflection_note':assert row['generated_token_allowance']==0 and row['reflection_persistence']=='exact_final_entry'
    for _,history,kwargs in owner.calls:assert INSTRUCTION in history[0]['content']


def test_invalid_reflection_repair_reuses_title_path_without_body_rewrite(tmp_path):
    settings,pages,_=runner.build_settings(**options());browser=make_browser(settings,pages,tmp_path/'db')
    body='# Repair evidence heading\nRecovered exact evidence body.'
    class Client:
        def __call__(self,*args,**kwargs):return ModelResponse({'content':body},{'eval_count':12,'prompt_eval_count':100,'done_reason':'stop'})
    try:
        rows=[];transitions=[];history=[{'role':'system','content':INSTRUCTION}]
        row=persist_reflection(browser,Client(),history,tmp_path,0,'agent-1','q',settings['notebooks']['agent-1'],{'status':'empty_response','answer':''},rows,transitions,120)
        assert row['reflection_persistence']=='repaired_final_entry'
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
