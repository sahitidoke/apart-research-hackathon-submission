"""Finite populated-fixture and mocked trajectory controls for 12b/c/d."""
from contextlib import closing
import sqlite3
import json
from pathlib import Path

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web.reference_ablations import ARMS
from orchestrator.simulated_web.reference_resume import prepare_resume
from orchestrator.simulated_web.synchronized_notebooks import make_browser,fork_stage,publish_stage
from orchestrator.simulated_web.test_reference_12a import six,PARENT
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner


@pytest.mark.parametrize('policy',ARMS)
def test_contract_parity_and_cross_arm_rejection(policy,tmp_path):
    baseline,pages,_=runner.build_settings(**six(1))
    changed,other,_=runner.build_settings(**six(1),ablation_policy=policy)
    expected={'ablation_policy'}|({'notebook_title_render_policy'} if policy=='generic-titles-v1' else {'notebook_visibility_policy'} if policy=='own-only-notebooks-v1' else {'access_plan','original_access_plan'})
    assert {k for k in set(baseline)|set(changed) if baseline.get(k)!=changed.get(k)}==expected
    assert pages==other
    assert changed['system_prompts']==baseline['system_prompts']
    assert changed['notebook_tool_schemas']==baseline['notebook_tool_schemas']
    with pytest.raises(ValueError,match='override'):
        prepare_resume(PARENT,{**six(), 'ablation_policy':policy},runner.build_settings)


def test_generic_titles_preserve_exact_body_and_all_publication_paths(tmp_path):
    settings,pages,_=runner.build_settings(**six(),ablation_policy='generic-titles-v1')
    browser=make_browser(settings,pages,tmp_path/'wiki.sqlite3')
    try:
        body='# Distinctive Agent Heading\nLiteral useful content'
        saved=browser.call('agent-1','append_notebook',{'text':body})
        read=browser.call('agent-1','read_notebook',{'url':saved['saved'],'revision':''})
        assert read['text']==body and read['title'].startswith('Research notebook entry by ')
        assert 'Distinctive' not in read['title']
        hit=browser.call('agent-2','search',{'query':'Distinctive Agent Heading'})
        assert saved['saved'] in json.dumps(hit) and 'Research notebook entry by ' in json.dumps(hit)
    finally:browser.close()
    dest=tmp_path/'trajectory'
    runner.run_concurrent_research_reflection(dest,Final512Owner(),**six(1),ablation_policy='generic-titles-v1')
    # All voluntary, initial mandatory and exact reflection saves share StageBrowser.
    with closing(sqlite3.connect(dest/'wiki.sqlite3')) as db:
        rows=db.execute('SELECT p.title,p.body,r.body FROM pages p JOIN entry_provenance e USING(slug) JOIN revisions r USING(slug)').fetchall()
        assert rows and all(t.startswith('Research notebook entry by ') and b==r for t,b,r in rows)



def test_private_populated_routes_native_history_and_host_audit(tmp_path):
    settings,pages,_=runner.build_settings(**six(),ablation_policy='own-only-notebooks-v1')
    central=make_browser(settings,pages,tmp_path/'wiki.sqlite3')
    snapshots={}
    try:
        own=[];peer=[]
        for i in range(7):
            own.append(central.call('agent-1','append_notebook',{'text':f'# Own topic {i}\nEvidence own.'})['saved'])
            peer.append(central.call('agent-2','append_notebook',{'text':f'# Secret peer {i}\nPeer-only-secret-body.'})['saved'])
        # A host-seeded foreign-authored entry in the owned directory is still own memory.
        central.notebook_visibility_policy=None
        cross=central.call('agent-2','append_notebook',{'text':'# Cross authored own memory','notebook':'https://wiki.test/page/'+settings['notebooks']['agent-1']})['saved']
        central.notebook_visibility_policy='own-only-v1'
        folder=tmp_path/'stage';folder.mkdir();snapshots=fork_stage(central,settings,pages,folder,1,2)
        a=snapshots['agent-1'];peerroot='https://wiki.test/page/'+settings['notebooks']['agent-2'];ownroot='https://wiki.test/page/'+settings['notebooks']['agent-1']
        for operation,args in [('open',{'url':peerroot}),('open',{'url':peerroot+'?offset=5'}),('open',{'url':peer[0]}),('read_notebook',{'url':peer[0],'revision':''}),('read_notebook',{'url':peer[0],'revision':'r-2'}),('read_notebook',{'url':peerroot,'revision':''}),('append_notebook',{'notebook':peerroot,'text':'Crosswrite'})]:
            assert 'error' in a.call('agent-1',operation,args)
        a.views={'agent-1':{'guess':[{'url':peer[0]}]}}
        assert 'error' in a.call('agent-1','click',{'page_id':'guess','link_id':1})
        for result in [a.call('agent-1','search',{'query':'Secret peer'}),a.call('agent-1','open',{'url':'https://wiki.test/'})]:
            assert peerroot not in json.dumps(result) and 'Secret peer' not in json.dumps(result)
        assert a.call('agent-1','read_notebook',{'url':cross,'revision':''})['text']=='# Cross authored own memory'
        assert 'error' not in a.call('agent-1','open',{'url':ownroot+'?offset=5'})
        assert a.call('agent-1','read_notebook',{'url':own[0],'revision':''})['text'].startswith('# Own')
        assert 'saved' in a.call('agent-1','append_notebook',{'text':'# New own\nContent'})
        assert 'error' in a.call('agent-1','open',{'url':'https://wiki.test/history'})
        assert 'saved' in snapshots['agent-2'].call('agent-2','append_notebook',{'text':'# Next own serial'})
        publish_stage(central,snapshots)
        assert central.db.execute('SELECT count(*) FROM entry_provenance').fetchone()[0]==17
        assert central.db.execute('SELECT count(*) FROM audit').fetchone()[0]>14
        assert snapshots['agent-2'].db.execute("SELECT count(*) FROM entry_provenance WHERE author='agent-1'").fetchone()[0]==0
    finally:
        for b in snapshots.values():b.close()
        central.close()


def test_union_source_discovery_exact_fixed_corpus(tmp_path):
    settings,pages,_=runner.build_settings(**six(),ablation_policy='union-discovery-v1')
    browser=make_browser(settings,pages,tmp_path/'wiki.sqlite3')
    try:
        assert settings['access_plan']['allowed_urls']['agent-1']==settings['access_plan']['allowed_urls']['agent-2']
        assert settings['access_plan']['groups']['agent-1']==sorted(settings['discovery_plan']['groups'])
        assert not browser.hidden_urls('agent-1') and not browser.hidden_urls('agent-2')
        for agent in ('agent-1','agent-2'):
            assert 'error' not in browser.call(agent,'open',{'url':next(iter(settings['discovery_plan']['groups'].values()))[0]})
    finally:browser.close()


@pytest.mark.parametrize('policy',ARMS)
def test_within_arm_resume_and_other_arm_denied(policy,tmp_path):
    inputs={**six(1),'ablation_policy':policy};partial=tmp_path/'partial'
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('bounded interruption')
    with pytest.raises(RuntimeError,match='bounded interruption'):
        runner.run_concurrent_research_reflection(partial,Final512Owner(),**inputs,checkpoint_callback=stop)
    cp=partial/'checkpoints/rounds-001'
    assert prepare_resume(cp,inputs,runner.build_settings)['verification_start_index']==0
    other={**inputs,'ablation_policy':next(p for p in ARMS if p!=policy)}
    with pytest.raises(ValueError,match='override'):prepare_resume(cp,other,runner.build_settings)
