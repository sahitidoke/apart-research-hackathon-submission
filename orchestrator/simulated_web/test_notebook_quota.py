"""Finite synthetic checks for opt-in notebook quota exemption; no live model."""
from dataclasses import replace
import json

import pytest

from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.synchronized_exchange import build_settings, run_exchange, load_checkpoint
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_9e import options, NineClient
from orchestrator.simulated_web.timed import run_phase
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.test_timed import Clock


def call(name, **arguments):
    return {'function': {'name': name, 'arguments': arguments}}


def fixture():
    settings,pages,_=build_settings(**options(),notebook_quota_policy='notebook-exempt-v1')
    browser=make_browser(settings,pages,':memory:',1,1)
    root='https://wiki.test/page/'+settings['notebooks']['agent-2']
    entry=browser.call('agent-2','append_notebook',{'text':'Peer evidence'})['saved']
    return settings,browser,root,entry


def phase_run(tmp_path, browser, policy, batches, phase='preparation', enabled=True):
    seen=[]
    def client(agent, history, timeout, **kwargs):
        seen.append(kwargs)
        batch=batches[len(seen)-1] if len(seen)<=len(batches) else []
        return ModelResponse({'content':'' if batch else 'Done','tool_calls':batch},
                             {'eval_count':5,'done_reason':'stop'})
    history=[]
    row=run_phase(browser,client,history,'Phase: 600 seconds.',phase,600,policy,tmp_path/'phase.jsonl',notebook_quota_exempt=enabled)
    return row,history,seen


@pytest.mark.parametrize('phase',['preparation','answer'])
def test_postquota_mixed_batch_and_later_turn_keep_notebooks(tmp_path,phase):
    settings,browser,root,entry=fixture()
    try:
        sources=[call('open',url='https://docs.test/')]*4
        batch=sources+[call('search',query='blocked'),call('read_notebook',url=entry,revision=''),call('append_notebook',text='Cross append',notebook=root)]
        row,history,seen=phase_run(tmp_path,browser,TimedPolicy(**settings['policy']),[batch,[call('open',url=entry)]],phase)
        assert row['source_browser_calls']==4 and row['notebook_calls']==3
        assert row['browser_calls']==7 and row['source_quota_rejections']==1
        assert row['status']=='complete' and len(seen)==3
        assert not any(r['final_only'] for r in seen)
        assert 'browser_limit' not in row['limits_reached']
        responses=[json.loads(m['content']) for m in history if m['role']=='tool']
        assert responses[4]=={'error':'Non-notebook browser allowance exhausted for this phase.'}
        assert responses[5]['text']=='Peer evidence' and responses[6]['author']==settings['visible_labels']['agent-1']
        assert 'non-notebook browser calls' in history[0]['content']
        assert browser.db.execute("SELECT count(*) FROM audit WHERE operation='search'").fetchone()[0]==0
    finally:browser.close()


def test_routes_require_actual_notebooks_and_agent_owned_clicks(tmp_path):
    settings,browser,root,entry=fixture()
    try:
        view=browser.call('agent-1','open',{'url':root})
        page=view['page_id']
        links=browser.views['agent-1'][page]
        link=next(i+1 for i,row in enumerate(links) if row['url']==entry)
        assert browser.is_notebook_action('agent-1','click',{'page_id':page,'link_id':link})
        assert not browser.is_notebook_action('agent-2','click',{'page_id':page,'link_id':link})
        for url in ['https://docs.test/','https://docs.test/request-history',root+'?x=1',entry+'#x',root+'-entry-999999','https://wiki.test.evil/page/x']:
            assert not browser.is_notebook_action('agent-1','open',{'url':url})
        for args in [{'page_id':page,'link_id':True},{'page_id':page,'link_id':999},{'page_id':[],'link_id':1}]:
            assert not browser.is_notebook_action('agent-1','click',args)
        assert not browser.is_notebook_action('agent-1','open',{'url':root,'extra':1})
        row,history,_=phase_run(tmp_path,browser,TimedPolicy(**settings['policy']),[[call('open',url='https://docs.test/')]*4+[call('click',page_id=page,link_id=link)]])
        assert row['source_browser_calls']==4 and row['notebook_calls']==1
        assert any(m.get('tool_name')=='click' and 'Peer evidence' in m['content'] for m in history)
    finally:browser.close()


def test_native_errors_exempt_without_source_bypass_or_size_bypass(tmp_path):
    settings,browser,root,entry=fixture()
    try:
        batch=[call('read_notebook',url='https://docs.test/',revision=''),call('append_notebook',text='x'*6001),call('append_notebook',text='bad',notebook='https://docs.test/')]
        row,history,_=phase_run(tmp_path,browser,TimedPolicy(**settings['policy']),[batch])
        assert row['source_browser_calls']==0 and row['notebook_calls']==3
        responses=[json.loads(m['content']) for m in history if m['role']=='tool']
        assert all('error' in r for r in responses)
        assert browser.db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==1
    finally:browser.close()


def test_turn_cap_still_stops_exempt_actions(tmp_path):
    settings,browser,root,entry=fixture()
    try:
        policy=replace(TimedPolicy(**settings['policy']),max_steps=2)
        row,_,seen=phase_run(tmp_path,browser,policy,[[call('read_notebook',url=entry,revision='')]]*3)
        assert len(seen)==2 and row['notebook_calls']==2 and row['status']=='step_limit'
        assert 'steps' in row['limits_reached']
    finally:browser.close()


def test_legacy_9e_counting_unchanged(tmp_path):
    settings,browser,root,entry=fixture()
    try:
        row,_,seen=phase_run(tmp_path,browser,TimedPolicy(**settings['policy']),[[call('read_notebook',url=entry,revision='')]*5],enabled=False)
        assert row['browser_calls']==4 and row['status']=='budget_exhausted' and len(seen)==1
        assert 'notebook_calls' not in row
        assert 'notebook_quota_policy' not in build_settings(**options())[0]
    finally:browser.close()


def test_policy_persisted_checkpoint_and_no_resume_override(tmp_path):
    values={**options(),'notebook_quota_policy':'notebook-exempt-v1'}
    run_exchange(tmp_path/'run',NineClient({'status':'abstained','answer':'','citations':[]}),**values)
    checkpoint=tmp_path/'run/checkpoints/rounds-002'
    data,browser,_=load_checkpoint(checkpoint);browser.close()
    assert data['settings.json']['notebook_quota_policy']=='notebook-exempt-v1'
    phases=[r for r in data['results.json'] if r['phase_role'] in ('research','answer')]
    assert all(r['notebook_quota_policy']=='notebook-exempt-v1' for r in phases)
    with pytest.raises(ValueError,match='Resume overrides'):
        run_exchange(tmp_path/'bad',NineClient({}),resume_from=checkpoint,notebook_quota_policy='notebook-exempt-v1')
    for policy in ['bad',True]:
        with pytest.raises(ValueError,match='quota policy'):
            build_settings(**options(),notebook_quota_policy=policy)
    legacy=options();legacy['notebook_context_policy']=None
    with pytest.raises(ValueError,match='quota policy'):
        build_settings(**legacy,notebook_quota_policy='notebook-exempt-v1')


@pytest.mark.parametrize('guard',['tokens','time'])
def test_notebook_exemption_retains_token_and_time_caps(tmp_path,guard):
    settings,browser,root,entry=fixture()
    clock=Clock();requests=[]
    def client(agent,history,timeout,**kwargs):
        requests.append(kwargs)
        if guard=='time':clock.now=601
        return ModelResponse({'content':'','tool_calls':[call('read_notebook',url=entry,revision='')]},
                             {'eval_count':1024 if guard=='tokens' else 1,'done_reason':'length' if guard=='tokens' else 'stop'})
    try:
        row=run_phase(browser,client,[],'Research','preparation',600,TimedPolicy(**settings['policy']),tmp_path/'guard.jsonl',clock=clock,notebook_quota_exempt=True)
        assert len(requests)==1 and row['browser_calls']==0
        assert ('generated_tokens' if guard=='tokens' else 'safety_timeout') in row['limits_reached']
        assert row['status']==('budget_exhausted' if guard=='tokens' else 'deadline_reached')
    finally:browser.close()


def test_directory_pagination_remains_readable_after_quota(tmp_path):
    settings,browser,root,entry=fixture()
    try:
        for index in range(6):browser.call('agent-2','append_notebook',{'text':f'Entry {index}'})
        view=browser.call('agent-1','open',{'url':root})
        links=browser.views['agent-1'][view['page_id']]
        link=next(i+1 for i,row in enumerate(links) if '?offset=5' in row['url'])
        for url in [root+'?offset=-1',root+'?offset=999',root+'?offset=0&offset=1',root+'?offset=5&x=1',root+'?offset=']:
            assert not browser.is_notebook_action('agent-1','open',{'url':url})
        row,history,_=phase_run(tmp_path,browser,TimedPolicy(**settings['policy']),[[call('open',url='https://docs.test/')]*4+
            [call('click',page_id=view['page_id'],link_id=link),call('open',url=root+'?offset=0')]])
        assert row['notebook_calls']==2 and row['source_browser_calls']==4
        responses=[json.loads(m['content']) for m in history if m['role']=='tool']
        assert all('error' not in response for response in responses)
        assert responses[-2]['url']==root+'?offset=5'
    finally:browser.close()
