"""Identity ablation, relevant leader schedule and versioned cross-edits; mocks only."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from urllib.parse import urlencode

from orchestrator.simulated_web import test_timed
from orchestrator.simulated_web.private_notes import build_settings,create_browser,load_checkpoint,run_private_notes,steps,LABELED_NOTEBOOKS
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient


class EditingClient(NeutralClient):
    edited=False
    def __call__(self,agent,history,timeout,**kwargs):
        message=history[-1]
        if agent=='agent-2' and not self.edited and message['role']=='user' and message['content'].startswith('Question research:'):
            return ModelResponse({'content':'','tool_calls':[{'function':{'name':'read_notebook','arguments':{'url':'https://wiki.test/page/notes-cedar-entry-000001','revision':''}}}]},{'eval_count':5,'prompt_eval_count':100,'done_reason':'stop'})
        if message.get('tool_name')=='read_notebook':
            read=json.loads(message['content']);self.edited=True
            return ModelResponse({'content':'','tool_calls':[{'function':{'name':'edit_notebook','arguments':{'url':read['url'],'expected_revision':read['revision'],'text':'Corrected entry from the evidence.'}}}]},{'eval_count':5,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


class NotebookIdentityTests(unittest.TestCase):
    def config(self,shared=False,count=3):
        r,s,m,old=CrossedAccessTests().setup_inputs()
        ids={a:[x['id'] for x in r[:count]] for a in old}
        opts={'sequence_mode':'agent_serial','mandatory_notes':True,'append_notes':True,'retain_context':True,
              'neutral_notebook':True,'notebook_tools':True,'bounded_context':True,'question_count':count,
              'visible_labels':{'agent-1':'8i0','agent-2':'8i0'} if shared else {'agent-1':'8h0','agent-2':'8h1'}}
        return r,s,m,ids,opts

    def test_matched_settings_leaders_and_self_only_labels(self):
        r,s,m,ids,opts=self.config(count=10);opts['round_leaders']=['agent-1']*5+['agent-2']*5
        h=build_settings(r,'topic',s,ids,m,**opts)[0]
        opts['visible_labels']={'agent-1':'8i0','agent-2':'8i0'}
        i=build_settings(r,'topic',s,ids,m,**opts)[0]
        self.assertEqual({k for k in h if h[k]!=i[k]},{'visible_labels','system_prompts'})
        sequence=steps(h);self.assertEqual(sequence[0][0],'agent-1');self.assertEqual(sequence[30][0],'agent-2')
        self.assertIn('Your identifier is 8h0.',h['system_prompts']['agent-1'])
        self.assertNotIn('8h1',h['system_prompts']['agent-1'])
        self.assertNotIn('8h0',h['system_prompts']['agent-2'])
        self.assertEqual(h['notebooks'],LABELED_NOTEBOOKS)

    def test_authentic_visible_authors_conflicts_old_revisions_and_source_denial(self):
        for shared in (False,True):
            r,s,m,ids,opts=self.config(shared)
            settings,pages,_,editable=build_settings(r,'topic',s,ids,m,**opts)
            b=create_browser(settings,pages,editable,':memory:')
            try:
                result=b.call('agent-1','open',{'url':'https://wiki.test/append?'+urlencode({'slug':'notes-cedar','text':'Original note'})})
                url=result['saved'];first=b.call('agent-2','read_notebook',{'url':url,'revision':''})
                self.assertEqual(first['author'],'8i0' if shared else '8h0')
                edit=b.call('agent-2','edit_notebook',{'url':url,'expected_revision':first['revision'],'text':'Corrected note'})
                self.assertEqual(edit['author'],'8i0' if shared else '8h1')
                conflict=b.call('agent-1','edit_notebook',{'url':url,'expected_revision':first['revision'],'text':'Stale overwrite'})
                self.assertIn('conflict',conflict['error'])
                old=b.call('agent-1','read_notebook',{'url':url,'revision':first['revision']})
                self.assertEqual(old['text'],'Original note')
                self.assertEqual(b.call('agent-1','read_notebook',{'url':url,'revision':''})['text'],'Corrected note')
                self.assertEqual(b.db.execute('SELECT agent FROM revisions ORDER BY id').fetchall(),[('agent-1',),('agent-2',)])
                self.assertIn('error',b.call('agent-1','edit_notebook',{'url':'https://docs.test/p/example','expected_revision':edit['revision'],'text':'intrusion'}))
                self.assertIn('error',b.call('agent-1','read_notebook',{'url':'https://docs.test/p/example','revision':''}))
                self.assertEqual(b.db.execute('SELECT count(*) FROM revisions').fetchone()[0],2)
            finally:b.close()

    def test_checkpoint_preserves_original_append_and_latest_cross_edit(self):
        r,s,m,ids,opts=self.config()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'first'
            def stop(cp):
                if cp.name=='rounds-001':raise RuntimeError('mock stop')
            with self.assertRaisesRegex(RuntimeError,'mock stop'):
                run_private_notes(path,EditingClient(),r,'topic',s,ids,m,checkpoint_callback=stop,**opts)
            cp=path/'checkpoints/rounds-001';data,b=load_checkpoint(cp)
            try:
                row=data['results.json'][2]
                self.assertNotEqual(row['answer'],'Corrected entry from the evidence.')
                self.assertEqual(b.call('agent-1','read_notebook',{'url':row['note_preservation']['saved_url'],'revision':''})['text'],'Corrected entry from the evidence.')
                self.assertIn('saved_revision',row['note_preservation'])
            finally:b.close()
            second=Path(tmp)/'second';result=run_private_notes(second,NeutralClient(),resume_from=cp)
            self.assertEqual(result['status'],'complete')
            data,b=load_checkpoint(second/'checkpoints/rounds-003');b.close()
            self.assertEqual(data['settings.json']['visible_labels'],opts['visible_labels'])

    def test_edit_route_matches_cross_edit_capability_and_legacy_append(self):
        for enabled in (False,True):
            r,s,m,ids,opts=self.config();opts['notebook_tools']=enabled
            settings,pages,_,editable=build_settings(r,'topic',s,ids,m,**opts)
            b=create_browser(settings,pages,editable,':memory:')
            try:
                text=b.call('agent-1','open',{'url':'https://wiki.test/edit'})['text']
                if enabled:
                    self.assertIn('read_notebook',text);self.assertIn('edit_notebook',text)
                    self.assertIn('Earlier versions are preserved',text)
                    self.assertNotIn('cannot be replaced',text)
                else:
                    self.assertIn('Existing entries cannot be replaced',text)
                    self.assertNotIn('edit_notebook',text)
            finally:b.close()

    def test_transport_adds_only_notebook_tools_when_enabled(self):
        client=test_timed.TransportTests().client();payloads=[]
        def factory(*args,**kwargs):
            c=MagicMock();r=io.BytesIO(b'{"message":{"content":"ok"},"done":true,"prompt_eval_count":100}\n');r.status=200
            c.getresponse.return_value=r;c.request.side_effect=lambda *args:payloads.append(json.loads(args[2]));return c
        with patch.object(client,'_start'),patch.object(client,'_idle_baseline',return_value=((42,'100',1234),7)),patch.object(client,'_count_prompt',return_value={'prompt_tokens':100}),patch('orchestrator.simulated_web.timed_transport.http.client.HTTPConnection',side_effect=factory):
            client('agent-1',[],5,num_predict=1)
            client.notebook_tools_enabled=True
            client('agent-1',[],5,num_predict=1)
            client('agent-1',[],5,num_predict=1,final_only=True)
        names=lambda p:{t['function']['name'] for t in p['tools']}
        self.assertEqual(names(payloads[1])-names(payloads[0]),{'read_notebook','edit_notebook'})
        self.assertNotIn('tools',payloads[2])
