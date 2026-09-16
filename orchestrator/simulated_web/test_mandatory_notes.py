"""Mandatory notebook persistence and failure gates; no real model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from orchestrator.simulated_web.mandatory_notes import SCHEMA, save_url
from orchestrator.simulated_web.private_notes import NOTEBOOKS, PersonalNotebookBrowser, load_checkpoint, run_private_notes
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_private_notes import NoteClient

NOTE=json.dumps({'answer_or_status':'answer with spaces & symbols',
    'supporting_sources':['https://docs.test/example?a=1&b=2'],
    'findings':'line one\nline two & more', 'uncertainties':'Not verified'},indent=2)


class MandatoryClient(NoteClient):
    def __init__(self,outputs=None):
        super().__init__();self.outputs=list(outputs or []);self.note_requests=[]

    def __call__(self,agent,history,timeout,**kwargs):
        if kwargs.get('format_schema')==SCHEMA:
            self.calls.append((agent,json.loads(json.dumps(history))))
            self.note_requests.append((agent,kwargs))
            return ModelResponse({'content':self.outputs.pop(0) if self.outputs else NOTE},
                                 {'eval_count':15,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


class MandatoryNotesTests(unittest.TestCase):
    def run_case(self,path,client,**kwargs):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        return run_private_notes(path,client,records,'topic',selectors,ids,manifest,
            sequence_mode='agent_serial',retain_context=True,mandatory_notes=True,history_search=True,**kwargs)

    def test_exact_encoding_host_attribution_locked_answer_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'first';client=MandatoryClient()
            def stop(cp):
                if cp.name=='rounds-001':raise RuntimeError('mock stop')
            with self.assertRaisesRegex(RuntimeError,'mock stop'):
                self.run_case(path,client,checkpoint_callback=stop)
            cp=path/'checkpoints/rounds-001';data,browser=load_checkpoint(cp)
            try:
                rows=data['results.json'];self.assertEqual(len(rows),6)
                for row in rows:
                    if row['phase_role']=='answer':self.assertEqual(row['answer'],row['agent']+' private result')
                    if row['phase_role']=='note':
                        self.assertEqual(row['browser_calls'],0);self.assertEqual(row['host_browser_calls'],2)
                        self.assertTrue(row['note_preservation']['persistence_verified'])
                        save,verify=row['host_persistence_actions']
                        self.assertEqual(save['actor'],'host_notebook_persistence')
                        self.assertEqual(parse_qs(urlsplit(save['url']).query)['text'],[NOTE])
                        self.assertNotIn(' ',save['url']);self.assertNotIn('\n',save['url'])
                        self.assertEqual(verify['response']['text'],NOTE)
                        event=browser.db.execute('SELECT requested FROM request_events WHERE id=?',(save['request_event_ids'][0],)).fetchone()
                        self.assertIn('text=',str(event))
                self.assertEqual(browser.db.execute('SELECT body FROM pages WHERE slug=?',(NOTEBOOKS['agent-1'],)).fetchone(),(NOTE,))
                prior=data['histories.json']
            finally:browser.close()
            second=Path(tmp)/'resumed';resumed=MandatoryClient()
            result=run_private_notes(second,resumed,resume_from=cp)
            self.assertEqual(result['status'],'complete')
            agent,history=resumed.calls[0];self.assertEqual(history[:len(prior[agent])],prior[agent])
            final,browser=load_checkpoint(second/'checkpoints/rounds-003');browser.close()
            self.assertEqual(final['settings.json']['maximum_generated_tokens'],49152)
            self.assertEqual(len(resumed.note_requests),4)

    def test_blank_retry_then_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ok';client=MandatoryClient(['',NOTE])
            self.assertEqual(self.run_case(path,client)['status'],'complete')
            note=json.loads((path/'results.json').read_text())[2]
            self.assertEqual(len(note['model_requests']),2)
            self.assertEqual(note['generated_tokens_observed'],30)
            self.assertEqual(note['host_browser_calls'],2)

    def test_malformed_exhaustion_does_not_admit_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad';client=MandatoryClient(['{}','{"findings":'])
            with self.assertRaisesRegex(RuntimeError,'Incomplete agent-1 note'):
                self.run_case(path,client)
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual(len(rows),3);self.assertTrue(all(r['agent']=='agent-1' for r in rows))
            self.assertEqual(rows[1]['answer'],'agent-1 private result')
            self.assertEqual(rows[2]['status'],'note_failed');self.assertEqual(rows[2]['host_browser_calls'],0)
            self.assertEqual(rows[2]['generated_tokens_observed'],30)
            self.assertEqual(json.loads((path/'manifest.json').read_text())['next_phase_index'],2)
            self.assertFalse((path/'checkpoints/rounds-001').exists())

    def test_verification_failure_preserves_save_and_stops(self):
        original=PersonalNotebookBrowser.call
        def broken(browser,agent,name,args):
            result=original(browser,agent,name,args)
            if args.get('url','').startswith('https://wiki.test/page/') and result.get('text')==NOTE:
                return {**result,'text':'not the saved text'}
            return result
        with tempfile.TemporaryDirectory() as tmp, patch.object(PersonalNotebookBrowser,'call',broken):
            path=Path(tmp)/'bad'
            with self.assertRaisesRegex(RuntimeError,'verification mismatch'):self.run_case(path,MandatoryClient())
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual(len(rows),3);self.assertEqual(rows[2]['note_preservation']['successful_saves_in_note_phase'],1)
            self.assertFalse(rows[2]['note_preservation']['persistence_verified'])

    def test_transport_timeout_preserves_partial_and_unknown_accounting(self):
        original=MandatoryClient.__call__
        def timeout(client,agent,history,seconds,**kwargs):
            if kwargs.get('format_schema')==SCHEMA:
                error=TimeoutError('mock deadline');error.partial={'content':'partial note'}
                raise error
            return original(client,agent,history,seconds,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(MandatoryClient,'__call__',timeout):
            path=Path(tmp)/'bad'
            with self.assertRaisesRegex(TimeoutError,'mock deadline'):self.run_case(path,MandatoryClient())
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual(len(rows),3)
            self.assertFalse(rows[-1]['token_accounting_complete'])
            self.assertIsNone(rows[-1]['generated_tokens_remaining'])
            self.assertEqual(rows[-1]['model_requests'][0]['partial']['content'],'partial note')
            self.assertIn('partial note',(path/'histories.json').read_text())

    def test_shape_rejection(self):
        for bad in ('','{}','null',json.dumps({'answer_or_status':' ', 'supporting_sources':[], 'findings':'x','uncertainties':'x'})):
            with self.subTest(bad=bad),self.assertRaises(ValueError):save_url(bad,'private-research-1')
