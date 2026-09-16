"""Append-only freeform notes survive pagination, peer reads, and resume."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode,parse_qs,urlsplit

from orchestrator.simulated_web.private_notes import build_settings,create_browser,load_checkpoint,run_private_notes,NOTEBOOKS
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_private_notes import NoteClient


class AppendClient(NoteClient):
    def __init__(self):super().__init__();self.entries=[]
    def __call__(self,agent,history,timeout,**kwargs):
        if history[-1]['role']=='user' and 'Notebook append:' in history[-1]['content']:
            self.calls.append((agent,json.loads(json.dumps(history))))
            assert 'format_schema' not in kwargs
            body=f'{agent} note {len(self.entries)+1}\nA & B remain uncertain; source https://docs.test/example'
            self.entries.append(body)
            return ModelResponse({'content':body},{'eval_count':15,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


class AppendNotebookTests(unittest.TestCase):
    def test_old_entries_readable_searchable_paginated_and_never_replaced(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        settings,pages,_,editable=build_settings(records,'topic',selectors,ids,manifest,sequence_mode='agent_serial',mandatory_notes=True,append_notes=True)
        browser=create_browser(settings,pages,editable,':memory:')
        try:
            bodies=[]
            for i in range(7):
                body=f'unique{i} spaces & newline\n'+'evidence '*300;bodies.append(body)
                result=browser.call('agent-1','open',{'url':'https://wiki.test/append?'+urlencode({'slug':NOTEBOOKS['agent-1'],'text':body})})
                self.assertIn('saved',result)
            root='https://wiki.test/page/'+NOTEBOOKS['agent-1']
            listing=browser.call('agent-2','open',{'url':root})
            more=next(x['url'] for x in listing['links'] if x['label']=='More entries')
            self.assertEqual(len(browser.call('agent-2','open',{'url':more})['links']),3)
            for i,body in enumerate(bodies):
                url=root+f'-entry-{i+1:06d}'
                self.assertEqual(browser.call('agent-2','open',{'url':url})['text'],body)
                self.assertTrue(any(r['url']==url for r in browser.search(f'unique{i}','agent-2')['results']))
            bad=browser.call('agent-2','open',{'url':'https://wiki.test/append?'+urlencode({'slug':NOTEBOOKS['agent-1'],'text':'intrusion'})})
            self.assertIn('error',bad)
            overwrite=browser.call('agent-1','open',{'url':'https://wiki.test/save?'+urlencode({'slug':NOTEBOOKS['agent-1'],'title':'x','text':'overwrite'})})
            self.assertIn('error',overwrite)
            self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0],7)
        finally:browser.close()

    def test_blank_freeform_attempt_retries_without_blank_entry(self):
        class BlankFirst(AppendClient):
            def __init__(self):super().__init__();self.blank=False
            def __call__(self,agent,history,timeout,**kwargs):
                if history[-1]['role']=='user' and 'Notebook append:' in history[-1]['content'] and not self.blank:
                    self.blank=True
                    return ModelResponse({'content':'  '},{'eval_count':5,'prompt_eval_count':100,'done_reason':'stop'})
                return super().__call__(agent,history,timeout,**kwargs)
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            result=run_private_notes(path,BlankFirst(),records,'topic',selectors,ids,manifest,
                sequence_mode='agent_serial',retain_context=True,mandatory_notes=True,append_notes=True)
            self.assertEqual(result['status'],'complete')
            data,browser=load_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0],6)
                self.assertEqual(len(data['results.json'][2]['model_requests']),2)
            finally:browser.close()

    def test_freeform_new_entry_only_in_public_request_and_checkpoint_resume(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'first';client=AppendClient()
            def stop(cp):
                if cp.name=='rounds-001':raise RuntimeError('mock stop')
            with self.assertRaisesRegex(RuntimeError,'mock stop'):
                run_private_notes(path,client,records,'topic',selectors,ids,manifest,sequence_mode='agent_serial',retain_context=True,
                    mandatory_notes=True,append_notes=True,bounded_context=True,checkpoint_callback=stop)
            cp=path/'checkpoints/rounds-001';data,browser=load_checkpoint(cp)
            first_body=client.entries[0]
            try:
                for row in data['results.json']:
                    if row['phase_role']=='note':
                        action=row['host_persistence_actions'][0]
                        self.assertEqual(urlsplit(action['url']).path,'/append')
                        self.assertEqual(parse_qs(urlsplit(action['url']).query)['text'],[row['answer']])
            finally:browser.close()
            second=Path(tmp)/'second';result=run_private_notes(second,AppendClient(),resume_from=cp)
            self.assertEqual(result['status'],'complete')
            final,browser=load_checkpoint(second/'checkpoints/rounds-003')
            try:
                self.assertEqual(browser.db.execute('SELECT count(*) FROM pages').fetchone()[0],8)
                self.assertEqual(browser.call('agent-2','open',{'url':'https://wiki.test/page/'+NOTEBOOKS['agent-1']+'-entry-000001'})['text'],first_body)
                self.assertTrue(final['settings.json']['append_notes'])
            finally:browser.close()
