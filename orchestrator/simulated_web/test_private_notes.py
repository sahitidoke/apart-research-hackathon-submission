"""Owned writes, public log exposure, context boundaries and checkpoints; mocks only."""
import hashlib
from html import unescape
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.private_notes import NOTEBOOKS, build_settings, create_browser, load_checkpoint, notebook_url, run_private_notes
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_token_pair import Client
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair


class NoteClient(Client):
    def __init__(self,fail_save=False):
        super().__init__();self.fail_save=fail_save

    def __call__(self,agent,history,timeout,**kwargs):
        self.calls.append((agent,json.loads(json.dumps(history))))
        if history[-1]['role']=='user':
            note=history[-1]['content'].startswith('Notebook update:')
            url=notebook_url(agent)
            if note:url='https://wiki.test/save?'+urlencode({'slug':'unknown' if self.fail_save else NOTEBOOKS[agent],'title':'Private research notebook','text':agent+' saved evidence'})
            message={'content':'','tool_calls':[{'function':{'name':'open','arguments':{'url':url}}}]}
        else:message={'content':agent+' private result'}
        return ModelResponse(message,{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})


class PrivateNotesTests(unittest.TestCase):
    def inputs(self):return CrossedAccessTests().setup_inputs()

    def test_public_read_log_search_owned_writes_and_discovery_read(self):
        records,selectors,manifest,ids=self.inputs()
        settings,pages,_,editable=build_settings(records,'topic',selectors,ids,manifest)
        b=create_browser(settings,pages,editable,':memory:')
        try:
            self.assertEqual(b.db.execute('SELECT count(*) FROM pages').fetchone()[0],2)
            body='Only authored note, source https://docs.test/example; secretword'
            save='https://wiki.test/save?'+urlencode({'slug':NOTEBOOKS['agent-1'],'title':'Private notebook','text':body})
            self.assertIn('saved',b.call('agent-1','open',{'url':save}))
            self.assertEqual(b.call('agent-2','open',{'url':notebook_url('agent-1')})['text'],body)
            self.assertEqual(b.call('agent-2','open',{'url':notebook_url('agent-1')})['links'],[])
            self.assertEqual(b.call('agent-2','open',{'url':save})['error'],'Notebook write access denied')
            for slug in ('new-page',NOTEBOOKS['agent-2']):
                result=b.call('agent-1','open',{'url':'https://wiki.test/save?'+urlencode({'slug':slug,'title':'x','text':'bad'})})
                self.assertIn('error',result)
            self.assertEqual(b.db.execute('SELECT count(*) FROM revisions').fetchone()[0],1)
            log=unescape(b.call('agent-2','open',{'url':'https://docs.test/request-history'})['text'])
            self.assertIn(save,log);self.assertIn('secretword',log)
            self.assertIn(notebook_url('agent-1'),[r['url'] for r in b.search('secretword','agent-2')['results']])
            hidden=set(b.pages)-b.access_allowed_urls['agent-2']
            source=next(url for url in hidden if '/p/' in url)
            self.assertNotIn(source,[r['url'] for r in b.search(b.pages[source]['title'],'agent-2')['results']])
            self.assertNotIn('error',b.call('agent-2','open',{'url':source}))
            root=b.call('agent-2','open',{'url':'https://docs.test/'})
            self.assertFalse(any(link['url'] in hidden for link in root['links']))
            identity=next(iter(b.source_urls))
            self.assertEqual(b.call('agent-2','open',{'url':'https://docs.test/source/save?'+urlencode({'source':identity,'title':'intrusion','text':'bad'})}),{'error':'Access denied'})
            self.assertNotIn('PRIVATE_GOLD',json.dumps([log,root,b.search('PRIVATE_GOLD','agent-2')]))
        finally:b.close()

    def test_schedule_context_reset_note_outcome_and_resume(self):
        records,selectors,manifest,ids=self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'first';client=NoteClient()
            def stop(path):
                if path.name=='rounds-001':raise RuntimeError('mock interrupted after checkpoint')
            with self.assertRaises(RuntimeError):run_private_notes(run,client,records,'topic',selectors,ids,manifest,checkpoint_callback=stop)
            cp=run/'checkpoints/rounds-001'
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir() if p.is_file()}
            data,b=load_checkpoint(cp)
            try:
                self.assertEqual(len(data['results.json']),6)
                self.assertEqual([r['phase_role'] for r in data['results.json']],['research','research','answer','answer','note','note'])
                self.assertTrue(all(len(h)==2 for h in data['histories.json'].values()))
            finally:b.close()
            resumed=Path(tmp)/'resumed';resumed_client=NoteClient();result=run_private_notes(resumed,resumed_client,resume_from=cp)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['note_updates_not_saved'],0)
            first_context=resumed_client.calls[0][1]
            self.assertNotIn('agent-1 private result',json.dumps(first_context))
            self.assertIn('agent-1 saved evidence',json.dumps(resumed_client.calls[1][1]))
            data,b=load_checkpoint(resumed/'checkpoints/rounds-003')
            try:
                self.assertEqual(len(data['results.json']),18)
                self.assertEqual(b.db.execute('SELECT count(*) FROM revisions').fetchone()[0],6)
                self.assertEqual(b.db.execute('SELECT count(*) FROM request_events').fetchone()[0],30)
                self.assertTrue(all(notebook_url(a) in data['histories.json'][a][0]['content'] for a in NOTEBOOKS))
            finally:b.close()
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir() if p.is_file()})
            starts=[(a,h) for a,h in client.calls if h[-1]['role']=='user']
            for a,h in starts:
                if h[-1]['content'].startswith('Notebook update:'):
                    self.assertIn(a+' private result',json.dumps(h))
                    self.assertIn('512 generated tokens',h[-1]['content'])
                    self.assertNotIn('agent-2 private result' if a=='agent-1' else 'agent-1 private result',json.dumps(h))
            with self.assertRaises(ValueError):run_private_notes(Path(tmp)/'bad',Client(),resume_from=cp,source_access_mode='hard')
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_failed_save_recorded_and_invalid_input_before_output(self):
        records,selectors,manifest,ids=self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed-saves'
            status=run_private_notes(path,NoteClient(True),records,'topic',selectors,ids,manifest,source_access_mode='hard')
            self.assertEqual(status['note_updates_not_saved'],6)
            hard_data,hard_browser=load_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertEqual(hard_data['settings.json']['source_access'],'hard')
                self.assertEqual(hard_browser.source_access_mode,'hard')
                self.assertNotIn('error',hard_browser.call('agent-2','open',{'url':notebook_url('agent-1')}))
            finally:hard_browser.close()
            self.assertEqual(len(list(path.glob('context-reset-*'))),6)
            self.assertTrue(all(len(h)==2 for h in json.loads((path/'histories.json').read_text()).values()))
            bad=Path(tmp)/'bad'
            with self.assertRaises(ValueError):run_private_notes(bad,Client(),records,'topic',selectors,ids,None)
            self.assertFalse(bad.exists())
        settings,pages,_,editable=build_settings(records,'topic',selectors,ids,manifest,'hard')
        browser=create_browser(settings,pages,editable,':memory:')
        try:
            foreign=next(u for u in set(browser.pages)-browser.access_allowed_urls['agent-2'] if '/p/' in u)
            self.assertEqual(browser.call('agent-2','open',{'url':foreign}),{'error':'Access denied'})
            self.assertNotIn('error',browser.call('agent-2','open',{'url':notebook_url('agent-1')}))
        finally:browser.close()

    def test_legacy_and_discovery_pair_checkpoint_modes(self):
        records,selectors,manifest,ids=self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'discovery'
            run_token_pair(path,records,'topic',Client(),pair_policy(compaction_enabled=False),selectors,
                           pair_protocol='question_research',question_ids=ids,access_manifest=manifest,source_access_mode='discovery_only')
            cp=load_pair_checkpoint(path/'checkpoints/rounds-003')
            try:
                b=cp.browser;foreign=next(u for u in set(b.pages)-b.access_allowed_urls['agent-2'] if '/p/' in u)
                self.assertNotIn('error',b.call('agent-2','open',{'url':foreign}))
                self.assertEqual(b.call('agent-2','open',{'url':'https://wiki.test/'}),{'error':'Access denied'})
                self.assertEqual(cp.data['settings.json']['source_access_mode'],'discovery_only')
            finally:cp.close()
