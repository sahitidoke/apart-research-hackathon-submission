"""Mocked 9c access, append and different-question checkpoint contracts."""
import io
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode
from unittest.mock import patch, MagicMock
from orchestrator.simulated_web.timed_transport import OwnedOllama
from orchestrator.simulated_web.timed_policy import TimedPolicy

from orchestrator.simulated_web.paired_notebook_views import build_paired_settings,browser_for,run_paired_views,load_checkpoint,sequence
from orchestrator.simulated_web.related_notebooks import HostAppendAdapter
from orchestrator.simulated_web.test_paired_notebook_views import inputs
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient


def related_inputs():
    value=inputs();r=value['records']
    value['question_ids']={'agent-1':[r[0]['id'],r[6]['id']], 'agent-2':[r[1]['id'],r[5]['id']]}
    value['related_append_only']=True
    return value


def test_hard_sources_all_routes_own_append_peer_read_no_edit_or_payload_log():
    settings,pages,_,editable=build_paired_settings(**related_inputs())
    b=browser_for(settings,pages,editable,':memory:')
    try:
        for owner,peer in [('agent-1','agent-2'),('agent-2','agent-1')]:
            denied=set(settings['access_plan']['allowed_urls'][owner])-set(settings['access_plan']['allowed_urls'][peer])
            for url in denied:
                assert b.call(peer,'open',{'url':url})=={'error':'Access denied'}
                b.views.setdefault(peer,{})['p999']=[{'label':'foreign','url':url}]
                assert b.call(peer,'click',{'page_id':'p999','link_id':1})=={'error':'Access denied'}
            assert not any(row['url'] in denied for row in b.search('Evidence',peer)['results'])
        for identity in settings['discovery_plan']['groups']:
            for agent in ('agent-1','agent-2'):
                for route in ('edit','save'):
                    assert 'error' in b.call(agent,'open',{'url':'https://docs.test/source/'+route+'?'+urlencode({'source':identity,'title':'replacement','text':'replacement'})})
        text='Request detail? secret & Unicode 漢字\nline'
        saved=b.call('agent-1','append_notebook',{'text':text})
        assert b.call('agent-2','read_notebook',{'url':saved['saved'],'revision':''})['text']==text
        for route in ('append','save','edit'):
            assert 'error' in b.call('agent-2','open',{'url':'https://wiki.test/'+route+'?'+urlencode({'slug':settings['notebooks']['agent-1'],'text':'replace'})})
        assert 'error' in b.call('agent-2','edit_notebook',{'url':saved['saved'],'revision':saved['revision'],'text':'replace'})
        host=HostAppendAdapter(b)
        newer=host.call('agent-1','open',{'url':'https://wiki.test/append?'+urlencode({'slug':settings['notebooks']['agent-1'],'text':'New & note\nbody'})})
        assert newer['saved']!=saved['saved']
        assert b.call('agent-2','open',{'url':saved['saved']})['text']==text
        log=b.call('agent-2','open',{'url':'https://docs.test/request-history'})['text']
        assert newer['saved'] in log and 'Research notebook entry' in log
        assert 'secret' not in log and 'New+%26' not in log and 'New & note' not in log
        assert b.db.execute('SELECT count(*) FROM source_pages').fetchone()[0]==0
        assert all('edit_notebook' not in p and 'You may edit' not in p for p in settings['system_prompts'].values())
    finally:b.close()


def test_related_schedule_real_mandatory_save_checkpoint_resume():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);path=root/'run'
        def stop(cp):
            if cp.name=='rounds-001':raise RuntimeError('mock stop')
        try:run_paired_views(path,NeutralClient(),**related_inputs(),checkpoint_callback=stop)
        except RuntimeError as e:assert str(e)=='mock stop'
        data,b=load_checkpoint(path/'checkpoints/rounds-001');b.close()
        assert [row['question_id'] for row in data['results.json']]==[row[2] for row in sequence(data['settings.json'])[:7]]
        assert data['results.json'][0]['question_id']!=data['results.json'][1]['question_id']
        assert data['results.json'][3]['note_preservation']['persistence_verified']
        assert run_paired_views(root/'resume',NeutralClient(),resume_from=path/'checkpoints/rounds-001')['status']=='complete'
        final,b=load_checkpoint(root/'resume/checkpoints/rounds-002');b.close()
        assert final['settings.json']['related_append_only'] is True


def test_actual_ollama_payload_unique_read_append_tools():
    settings,*_=build_paired_settings(**related_inputs())
    with patch('orchestrator.simulated_web.timed_transport.sys.platform','linux'), patch('orchestrator.simulated_web.timed_transport.shutil.which',return_value='/mock/ollama'):
        client=OwnedOllama(TimedPolicy(),'/tmp/model','/tmp/mock.log')
    client.metadata,client.ready='mock',True
    client.notebook_tools_enabled=True;client.notebook_tool_schemas=settings['notebook_tool_schemas']
    response=io.BytesIO(b'{"message":{"content":"answer"},"done":true,"prompt_eval_count":100}\n');response.status=200
    connection=MagicMock();connection.getresponse.return_value=response
    with patch.object(client,'_start'),patch.object(client,'_idle_baseline',return_value=((42,'100',1234),7)),patch.object(client,'_count_prompt',return_value={'prompt_tokens':100}) as preflight,patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection',return_value=connection):
        client('agent-1',[],20,num_predict=10)
    sent=json.loads(connection.request.call_args.args[2])['tools']
    names=[t['function']['name'] for t in sent]
    assert len(names)==len(set(names))
    assert set(names)=={'search','open','click','read_notebook','append_notebook'}
    assert sent==preflight.call_args.args[0]['tools']
