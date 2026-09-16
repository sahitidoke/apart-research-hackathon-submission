"""Source-discovery-only 9d: route semantics, isolation and recovery without models."""
import json

import pytest

from orchestrator.simulated_web.synchronized_exchange import (
    build_settings, run_exchange, load_checkpoint, SHORT_NOTE_POLICY,
    DISCOVERY_ACCESS_POLICY, SOURCE_DENIAL_POLICY,
)
from orchestrator.simulated_web.synchronized_notebooks import make_browser, fork_stage
from orchestrator.simulated_web.test_synchronized_exchange import inputs
from orchestrator.simulated_web.test_hf_synchronized_exchange import FP8Client


def test_only_source_access_contract_changes():
    hard=build_settings(**inputs(), inference_profile='hf-fp8-v1',note_retry_policy=SHORT_NOTE_POLICY)[0]
    discovery=build_settings(**inputs(), inference_profile='hf-fp8-v1',note_retry_policy=SHORT_NOTE_POLICY,
                             source_access_policy=DISCOVERY_ACCESS_POLICY)[0]
    assert discovery.pop('source_access_policy')==DISCOVERY_ACCESS_POLICY
    assert discovery.pop('source_access')=='discovery_only'
    assert hard.pop('source_access')=='hard'
    assert discovery['access_plan'].pop('source_access_mode')=='discovery_only'
    assert discovery==hard


@pytest.mark.parametrize('no_peer', [False,True])
def test_foreign_sources_hidden_from_discovery_but_known_open_and_click_allowed(tmp_path,no_peer):
    settings,pages,_=build_settings(**inputs(),source_access_policy=DISCOVERY_ACCESS_POLICY,
                                   no_peer_information=no_peer)
    central=make_browser(settings,pages,':memory:',1,1)
    snapshots={}
    try:
        peer_note=central.call('agent-2','append_notebook',{'text':'PEER_BODY_SENTINEL'})
        own_note=central.call('agent-1','append_notebook',{'text':'OWN_BODY_SENTINEL'})
        snapshots=fork_stage(central,settings,pages,tmp_path,1,1)
        for agent,peer in [('agent-1','agent-2'),('agent-2','agent-1')]:
            browser=snapshots[agent]
            foreign=set(settings['access_plan']['allowed_urls'][peer])-set(settings['access_plan']['allowed_urls'][agent])
            assert foreign
            # Index/listing responses cannot reveal foreign links; ordinary
            # document URLs already known from another channel remain readable.
            for listing in settings['discovery_plan']['listing_urls']:
                result=browser.call(agent,'open',{'url':listing})
                assert not foreign & {link['url'] for link in result.get('links',[])}
            for query in ('Evidence','Document','source',''):
                assert not foreign & {row['url'] for row in browser.search(query,agent)['results']}
            for url in foreign:
                assert 'error' not in browser.call(agent,'open',{'url':url})
                browser.views.setdefault(agent,{})['p999']=[{'label':'known source','url':url}]
                assert 'error' not in browser.call(agent,'click',{'page_id':'p999','link_id':1})
        a=snapshots['agent-1']
        initial_history=a.call('agent-1','open',{'url':'https://docs.test/request-history'})
        if no_peer:
            assert peer_note['saved'] not in initial_history['text']
            assert not a.db.execute("SELECT count(*) FROM request_events WHERE owner='agent-2'").fetchone()[0]
        routes=[a.call('agent-1','open',{'url':peer_note['saved']}),
                a.call('agent-1','read_notebook',{'url':peer_note['saved'],'revision':''})]
        a.views.setdefault('agent-1',{})['p999']=[{'label':'peer note','url':peer_note['saved']}]
        routes.append(a.call('agent-1','click',{'page_id':'p999','link_id':1}))
        history=a.call('agent-1','open',{'url':'https://docs.test/request-history'})
        if no_peer:
            assert all('error' in row for row in routes)
            # Our explicit failed reads are our own history, not foreign exposure.
            assert peer_note['saved'] in history['text']
            assert 'PEER_BODY_SENTINEL' not in json.dumps(routes)
            assert 'PEER_BODY_SENTINEL' not in history['text']
            assert not a.db.execute("SELECT count(*) FROM request_events WHERE owner='agent-2'").fetchone()[0]
        else:
            assert all('error' not in row for row in routes)
            assert peer_note['saved'] in history['text']
        assert 'error' not in a.call('agent-1','read_notebook',{'url':own_note['saved'],'revision':''})
    finally:
        for browser in snapshots.values():browser.close()
        central.close()


def test_checkpoint_inherits_mode_rejects_override_and_tampered_mode(tmp_path):
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('mock stop')
    with pytest.raises(RuntimeError,match='mock stop'):
        run_exchange(tmp_path/'first',FP8Client(),**inputs(),source_access_policy=DISCOVERY_ACCESS_POLICY,
                     inference_profile='hf-fp8-v1',checkpoint_callback=stop)
    checkpoint=tmp_path/'first/checkpoints/rounds-001'
    saved,browser,_=load_checkpoint(checkpoint)
    try:assert browser.source_access_mode=='discovery_only'
    finally:browser.close()
    with pytest.raises(ValueError,match='Resume overrides prohibited'):
        run_exchange(tmp_path/'invalid',FP8Client(),resume_from=checkpoint,source_access_policy=DISCOVERY_ACCESS_POLICY)
    assert not (tmp_path/'invalid').exists()
    assert run_exchange(tmp_path/'resume',FP8Client(),resume_from=checkpoint)['status']=='complete'
    final=json.loads((tmp_path/'resume/settings.json').read_text())
    assert final['source_access_policy']==DISCOVERY_ACCESS_POLICY
    assert final['access_plan']['source_access_mode']=='discovery_only'
    with pytest.raises(ValueError,match='Unknown source access'):
        run_exchange(tmp_path/'bad',FP8Client(),**inputs(),source_access_policy='typo')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='hard-denial wording'):
        build_settings(**inputs(),source_access_policy=DISCOVERY_ACCESS_POLICY,source_denial_policy=SOURCE_DENIAL_POLICY)
