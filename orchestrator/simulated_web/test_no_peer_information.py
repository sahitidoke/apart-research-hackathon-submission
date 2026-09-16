"""No-model matched no-peer condition: every browser channel and host merge."""
import json
from pathlib import Path

import pytest
from orchestrator.simulated_web.synchronized_exchange import build_settings,run_exchange,load_checkpoint,SHORT_NOTE_POLICY,expose_entries
from orchestrator.simulated_web.synchronized_notebooks import make_browser,fork_stage,publish_stage,self_memory
from orchestrator.simulated_web.test_synchronized_exchange import inputs,Client


@pytest.mark.parametrize('agent,peer',[('agent-1','agent-2'),('agent-2','agent-1')])
def test_all_routes_hide_peer_authorship_across_destinations_and_old_stages(tmp_path,agent,peer):
    settings,pages,_=build_settings(**inputs(),note_retry_policy=SHORT_NOTE_POLICY,no_peer_information=True)
    central=make_browser(settings,pages,':memory:',1,1)
    try:
        foreign=[]
        for q in (1,2):
            central.round_index=q
            for root in settings['notebooks'].values():
                result=central.call(peer,'append_notebook',{'text':'FOREIGN_BODY_SENTINEL_'+str(q),'notebook':'https://wiki.test/page/'+root})
                foreign.append(result)
                slug=result['saved'].removeprefix('https://wiki.test/page/')
                central.db.execute('UPDATE pages SET title=? WHERE slug=?',('FOREIGN_TITLE_SENTINEL',slug))
                central.db.execute('UPDATE revisions SET title=? WHERE slug=?',('FOREIGN_TITLE_SENTINEL',slug));central.db.commit()
                central.call(peer,'open',{'url':result['saved']})
        own=central.call(agent,'append_notebook',{'text':'OWN_CROSS_DESTINATION','notebook':'https://wiki.test/page/'+settings['notebooks'][peer]})
        # An older shared history window must not retain removed foreign event IDs.
        central.call(agent,'open',{'url':'https://docs.test/request-history'})
        previous_windows=[token for token,window in central.history_windows.items() if window['owner']==agent]
        original_entries=central.db.execute('SELECT * FROM pages ORDER BY slug').fetchall()
        snapshots=fork_stage(central,settings,pages,tmp_path,2,2);visible=snapshots[agent]
        try:
            responses=[]
            for item in foreign:
                responses.append(visible.call(agent,'read_notebook',{'url':item['saved'],'revision':''}))
                responses.append(visible.call(agent,'read_notebook',{'url':item['saved'],'revision':item['revision']}))
                responses.append(visible.call(agent,'open',{'url':item['saved']}))
                visible.views.setdefault(agent,{})['p999']=[{'label':'guessed entry','url':item['saved']}]
                responses.append(visible.call(agent,'click',{'page_id':'p999','link_id':1}))
            assert all('error' in r for r in responses)
            for query in ('FOREIGN_BODY_SENTINEL','FOREIGN_TITLE_SENTINEL','notebook','request'):
                responses.append(visible.call(agent,'search',{'query':query}))
            for root in settings['notebooks'].values():
                page=visible.call(agent,'open',{'url':'https://wiki.test/page/'+root});responses.append(page)
                for link in page.get('links',[]):responses.append(visible.call(agent,'open',{'url':link['url']}))
            for token in previous_windows:
                responses.append(visible.call(agent,'open',{'url':'https://docs.test/request-history?window='+token+'&offset=0&position=0'}))
            for url in ('https://wiki.test/','https://docs.test/request-history'):
                page=visible.call(agent,'open',{'url':url});responses.append(page)
                for link in page.get('links',[]):responses.append(visible.call(agent,'open',{'url':link['url']}))
            assert 'FOREIGN_BODY_SENTINEL' not in json.dumps(responses)
            assert 'FOREIGN_TITLE_SENTINEL' not in json.dumps(responses)
            assert visible.db.execute('SELECT count(*) FROM request_events WHERE owner=?',(peer,)).fetchone()[0]==0
            assert visible.call(agent,'read_notebook',{'url':own['saved'],'revision':''})['text']=='OWN_CROSS_DESTINATION'
            memory,_=self_memory(visible,agent,Client());assert memory['entries'][0]['url']==own['saved']
            history=[];expose_entries(visible,agent,history,tmp_path,2,2,'before')
            assert 'FOREIGN_BODY_SENTINEL' not in json.dumps(history) and 'FOREIGN_TITLE_SENTINEL' not in json.dumps(history)
            new=visible.call(agent,'append_notebook',{'text':'OWN_NEXT_STAGE','notebook':'https://wiki.test/page/'+settings['notebooks'][peer]})
            mappings=publish_stage(central,snapshots)
            assert mappings[agent]
            for token,window in central.history_windows.items():
                if window['owner']==agent:
                    assert all(central.db.execute('SELECT owner FROM request_events WHERE id=?',(event,)).fetchone()==(agent,) for event in window['ids'])
            assert central.views[agent]==visible.views[agent]
            assert all(central.db.execute('SELECT * FROM pages WHERE slug=?',(row[0],)).fetchone()==row for row in original_entries)
            assert central.db.execute('SELECT count(*) FROM pages WHERE body=?',('OWN_NEXT_STAGE',)).fetchone()[0]==1
            assert central.call(peer,'open',{'url':new['saved']})['text']=='OWN_NEXT_STAGE'
            assert all(central.db.execute('SELECT 1 FROM request_events WHERE id=?',(i,)).fetchone() for i in mappings[agent].values())
        finally:
            for b in snapshots.values():b.close()
    finally:central.close()


def test_matched_settings_and_no_peer_round_resume(tmp_path):
    base=build_settings(**inputs(),note_retry_policy=SHORT_NOTE_POLICY)[0]
    restricted=build_settings(**inputs(),note_retry_policy=SHORT_NOTE_POLICY,no_peer_information=True)[0]
    assert {k:v for k,v in restricted.items() if k!='no_peer_information'}==base
    def stop(cp):
        if cp.name=='rounds-001':raise RuntimeError('boundary')
    with pytest.raises(RuntimeError,match='boundary'):
        run_exchange(tmp_path/'first',Client(),**inputs(),note_retry_policy=SHORT_NOTE_POLICY,no_peer_information=True,checkpoint_callback=stop)
    data,b,_=load_checkpoint(tmp_path/'first/checkpoints/rounds-001')
    try:assert b.db.execute('SELECT count(DISTINCT author) FROM entry_provenance').fetchone()[0]==2
    finally:b.close()
    for agent in ('agent-1','agent-2'):
        events=json.loads((tmp_path/f'first/round-01-stage-4/{agent}-after-views.json').read_text())
        own=restricted['visible_labels'][agent]
        assert all(r['response'].get('author',own)==own for r in events)
    assert run_exchange(tmp_path/'resume',Client(),resume_from=tmp_path/'first/checkpoints/rounds-001')['status']=='complete'
    final,b,_=load_checkpoint(tmp_path/'resume/checkpoints/rounds-002');b.close()
    assert final['settings.json']['no_peer_information'] is True
