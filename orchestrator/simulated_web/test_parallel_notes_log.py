"""Mandatory note futures and metadata-only public history; mocked inference only."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.async_notebooks import EventLog
from orchestrator.simulated_web.metadata_request_log import make_metadata_browser
from orchestrator.simulated_web.parallel_notebooks import build_parallel_settings, run_parallel_notebooks
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_async_notebooks import inputs, tool
from orchestrator.simulated_web.test_parallel_notebooks import Client


class NotesClient(Client):
    def __init__(self, agent, callback=None):
        super().__init__();self.agent=agent;self.callback=callback;self.note_calls=0;self.qa_calls=0
    def __call__(self, agent, history, timeout, *, num_predict, final_only=False):
        self.assert_agent = agent
        if final_only:self.note_calls+=1
        else:self.qa_calls+=1
        if self.callback:return self.callback(self,history,num_predict,final_only)
        return ModelResponse({'content': ('早期 & facts\nsource evidence' if final_only else 'Locked answer '+agent)},
                             {'eval_count': min(9,num_predict), 'done_reason': 'stop'})


class NotesLogTests(unittest.TestCase):
    def test_note_inference_overlaps_peer_research_and_locks_answer(self):
        note_started=threading.Event();peer_research=threading.Event()
        def a(c,h,n,note):
            if note and c.note_calls==1:
                note_started.set();self.assertTrue(peer_research.wait(2))
            return ModelResponse({'content':'A authored note' if note else 'A locked answer'},{'eval_count':7,'done_reason':'stop'})
        def b(c,h,n,note):
            if not note and c.qa_calls==1:
                self.assertTrue(note_started.wait(2));peer_research.set()
                return ModelResponse(tool('open',url='https://docs.test/'),{'eval_count':3,'done_reason':'stop'})
            return ModelResponse({'content':'B note' if note else 'B answer'},{'eval_count':5,'done_reason':'stop'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run';clients={'agent-1':NotesClient('agent-1',a),'agent-2':NotesClient('agent-2',b)}
            result=run_parallel_notebooks(path,clients,**inputs(),mandatory_post_answer=True,shared_request_log=True)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['results']),4)
            for row in result['results']:
                self.assertEqual(row['mandatory_note']['status'],'saved')
                self.assertEqual(row['answer'],row['mandatory_note']['locked_answer'])
                self.assertEqual(len(row['mandatory_note']['host_persistence_actions']),2)
                self.assertTrue(all(action['request_event_ids'] for action in row['mandatory_note']['host_persistence_actions']))
            events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            verified=next(e['sequence'] for e in events if e['event']=='mandatory_note_verified' and e['agent']=='agent-1')
            next_question=next(e['sequence'] for e in events if e['event']=='question_available' and e['agent']=='agent-1' and e['question_index']==1)
            self.assertLess(verified,next_question)
            pubs=[e for e in events if e['event']=='notebook_published_available']
            self.assertTrue(all(e['actor']=='host_notebook_persistence' for e in pubs))
            self.assertFalse(any(e['event']=='tool_returned' and e.get('operation')=='append_notebook' for e in events))

    def test_blank_retry_exhaustion_stops_only_that_agent(self):
        def blank(c,h,n,note):return ModelResponse({'content':'' if note else 'Locked final'}, {'eval_count':11,'done_reason':'stop'})
        with tempfile.TemporaryDirectory() as tmp:
            clients={'agent-1':NotesClient('agent-1',blank),'agent-2':NotesClient('agent-2')}
            result=run_parallel_notebooks(Path(tmp)/'run',clients,**inputs(),mandatory_post_answer=True)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(clients['agent-1'].qa_calls,1);self.assertEqual(clients['agent-1'].note_calls,2)
            self.assertEqual(result['states']['agent-2']['status'],'complete')
            row=next(r for r in result['results'] if r['agent']=='agent-1')
            self.assertEqual(row['answer'],'Locked final');self.assertEqual(row['mandatory_note']['generated_tokens'],22)
            self.assertFalse(row['mandatory_note']['persistence_verified'])

    def test_verification_failure_preserves_entry_and_stops_only_writer(self):
        original=make_metadata_browser
        def factory(*args,**kwargs):
            b=original(*args,**kwargs);call=b.call
            def mismatch(agent,operation,arguments):
                result=call(agent,operation,arguments)
                if agent=='agent-1' and operation=='read_notebook':
                    result={**result,'text':'wrong returned body'}
                return result
            b.call=mismatch;return b
        with tempfile.TemporaryDirectory() as tmp,patch('orchestrator.simulated_web.parallel_notebooks.make_metadata_browser',factory):
            clients={a:NotesClient(a) for a in ('agent-1','agent-2')}
            path=Path(tmp)/'run'
            result=run_parallel_notebooks(path,clients,**inputs(),mandatory_post_answer=True,shared_request_log=True)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['states']['agent-2']['status'],'complete')
            self.assertEqual(clients['agent-1'].qa_calls,1)
            row=next(r for r in result['results'] if r['agent']=='agent-1')
            self.assertEqual(row['mandatory_note']['status'],'failed')
            self.assertTrue(row['mandatory_note']['saved_url'])
            self.assertEqual(len(row['mandatory_note']['host_persistence_actions']),2)

    def test_unanswered_length_slots_have_no_note_or_fabricated_answer(self):
        def length(c,h,n,note):
            self.assertFalse(note)
            return ModelResponse({'content':'partial'}, {'eval_count':n,'done_reason':'length'})
        with tempfile.TemporaryDirectory() as tmp:
            clients={a:NotesClient(a,length) for a in ('agent-1','agent-2')}
            result=run_parallel_notebooks(Path(tmp)/'run',clients,**inputs(),mandatory_post_answer=True)
            self.assertTrue(all(row['answer'] is None and 'mandatory_note' not in row for row in result['results']))
            self.assertTrue(all(c.note_calls==0 for c in clients.values()))

    def test_log_has_real_titles_entry_urls_but_never_bodies_or_encoded_save_text(self):
        settings,pages,_=build_parallel_settings(**inputs(),shared_request_log=True)
        with tempfile.TemporaryDirectory() as tmp:
            b=make_metadata_browser(settings,pages,':memory:',EventLog(Path(tmp)/'events.jsonl'))
            try:
                url=next(u for u in settings['access_plan']['allowed_urls']['agent-1'] if u in b.pages and u not in settings['discovery_plan']['listing_urls'])
                returned=b.call('agent-1','open',{'url':url})
                note='Secret正文 only in notebook\nA & B'
                saved=b.call('agent-1','append_notebook',{'text':note})
                b.call('agent-1','open',{'url':'https://wiki.test/append?'+urlencode({'slug':settings['notebooks']['agent-1'],'text':note})})
                b.call('agent-1','search',{'query':note})
                candidates=b.history_search_candidates('agent-2')
                indexed=''.join(c['text'] for c in candidates)
                self.assertNotIn('Evidence 0',indexed)
                self.assertIn(returned['title'],indexed);self.assertIn(saved['saved'],indexed)
                self.assertNotIn(note,indexed);self.assertNotIn('Secret',indexed);self.assertNotIn('%E6%AD%A3',indexed)
                log=b.call('agent-2','open',{'url':'https://docs.test/request-history'})
                self.assertNotIn('Secret',log['text']);self.assertIn('entry-000001',log['text'])
                self.assertTrue(any('/request-history?' in row['url'] for row in b.search('entry 000001','agent-2')['results']))
                self.assertEqual(b.call('agent-2','read_notebook',{'url':saved['saved'],'revision':''})['text'],note)
            finally:b.close()

    def test_opt_in_settings_and_budget(self):
        legacy,*_=build_parallel_settings(**inputs())
        new,*_=build_parallel_settings(**inputs(),mandatory_post_answer=True,shared_request_log=True)
        self.assertNotIn('mandatory_post_answer',legacy);self.assertNotIn('shared_request_log',legacy)
        self.assertEqual(legacy['maximum_generated_tokens'],16384)
        self.assertEqual(new['maximum_generated_tokens'],32768)
        for prompt in new['system_prompts'].values():
            self.assertNotIn('There is no separate notebook-update phase',prompt)
            self.assertIn('After a finalized answer',prompt)
