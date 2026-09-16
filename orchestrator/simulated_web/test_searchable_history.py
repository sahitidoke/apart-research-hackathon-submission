"""Searchable genuine history, bounded matching snapshots, and serial notebook mocks."""
import hashlib
from html import unescape
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.private_notes import load_checkpoint, run_private_notes
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_private_notes import NoteClient


PAGES=[{'url':'https://docs.test/','title':'Collection','text':'Ordinary documents','links':[]}]


class SearchNoteClient(NoteClient):
    def __call__(self,agent,history,timeout,**kwargs):
        if agent=='agent-2' and history[-1]['role']=='user' and history[-1]['content'].startswith('Question research:'):
            self.calls.append((agent,json.loads(json.dumps(history))))
            return ModelResponse({'content':'','tool_calls':[{'function':{'name':'search','arguments':{'query':'saved evidence'}}}]},
                                 {'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent,history,timeout,**kwargs)


class SearchHistoryTests(unittest.TestCase):
    def browser(self,mode='shared',enabled=True):
        return Browser(PAGES,':memory:',request_history_mode=mode,history_search=enabled,search_policy='distinct_sources_5',search_snippets=False)

    def test_shared_matching_snapshot_long_save_and_owner_bound(self):
        b=self.browser()
        try:
            body='ordinary '*700+' finallogneedle & < >'
            url='https://wiki.test/save?'+urlencode({'slug':'notes','title':'Personal','text':body})
            self.assertIn('saved',b.call('agent-1','open',{'url':url}))
            # Newer entries must not move the matching old event out of its search target.
            for i in range(6):b.call('agent-1','search',{'query':'irrelevant'+str(i)})
            result=b.call('agent-2','search',{'query':'finallogneedle'})
            logs=[r for r in result['results'] if r['title']=='Request history']
            self.assertEqual(len(logs),1);self.assertLessEqual(len(result['results']),5)
            target=logs[0]['url']
            response=b.call('agent-2','open',{'url':target})
            self.assertIn('finallogneedle',unescape(response['text']))
            self.assertLessEqual(len(response['text']),8000)
            self.assertEqual(b.call('agent-1','open',{'url':target}),{'error':'Unknown request history page'})
            self.assertEqual(len(b.history_windows),1)
            self.assertNotIn('snippet',logs[0])
            before=len(b.history_windows)
            self.assertEqual(b.call('agent-2','search',{'query':'neverpreviouslyrequested'})['results'],[])
            self.assertEqual(len(b.history_windows),before)
        finally:b.close()

    def test_isolated_default_bounds_and_no_response_body_index(self):
        b=self.browser('isolated')
        try:
            b.call('agent-1','search',{'query':'foreignneedle'})
            self.assertEqual(b.call('agent-2','search',{'query':'foreignneedle'})['results'],[])
            self.assertFalse(b.history_windows)
            self.assertEqual(len(b.call('agent-1','search',{'query':'foreignneedle'})['results']),1)
            for i in range(101):b.call('agent-1','search',{'query':'recent'+str(i)})
            self.assertEqual(b.search('foreignneedle','agent-1')['results'],[])
            b.history_windows={str(i):{'owner':'agent-1','ids':[]} for i in range(2000)}
            self.assertEqual(b.search('recent100','agent-1')['results'],[])
        finally:b.close()
        b=self.browser(enabled=False)
        try:
            b.call('agent-1','search',{'query':'foreignneedle'})
            self.assertEqual(b.search('foreignneedle','agent-2')['results'],[])
        finally:b.close()
        b=Browser([{'url':'https://docs.test/hidden','title':'Secret','text':'responsebodyneedle','links':[]}],':memory:',request_history_mode='shared',history_search=True,access_allowed_urls={'agent-1':['https://docs.test/hidden'],'agent-2':[]},search_policy='distinct_sources_5')
        try:
            b.call('agent-1','open',{'url':'https://docs.test/hidden'})
            self.assertEqual(b.call('agent-2','search',{'query':'responsebodyneedle'})['results'],[])
        finally:b.close()

    def test_serial_same_questions_no_forced_and_checkpoint_modes(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        ids={a:list(ids['agent-2']) for a in ids}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'serial'
            status=run_private_notes(path,SearchNoteClient(),records,'topic',selectors,ids,manifest,
                sequence_mode='agent_serial',history_search=True,log_exposure='spontaneous')
            self.assertEqual(status['status'],'complete')
            self.assertEqual(list(path.glob('forced-log-*')),[])
            data,b=load_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertEqual([(r['agent'],r['phase_role']) for r in data['results.json'][:6]],
                    [(a,p) for a in ids for p in ('research','answer','note')])
                self.assertTrue(b.history_search)
                saved_search=json.loads(b.db.execute("SELECT result FROM audit WHERE agent='agent-2' AND operation='search' ORDER BY id LIMIT 1").fetchone()[0])
                saved_hit=next(r for r in saved_search['results'] if r['title']=='Request history')
                self.assertIn('saved',unescape(b.call('agent-2','open',{'url':saved_hit['url']})['text']))
                self.assertEqual(b.db.execute('SELECT count(*) FROM request_events').fetchone()[0],19)
                self.assertEqual(data['settings.json']['question_ids']['agent-1'],data['settings.json']['question_ids']['agent-2'])
                result=b.call('agent-2','search',{'query':'saved evidence'})
                hit=next(r for r in result['results'] if r['title']=='Request history')
                self.assertIn('saved',unescape(b.call('agent-2','open',{'url':hit['url']})['text']))
            finally:b.close()
            with self.assertRaises(ValueError):run_private_notes(Path(tmp)/'bad',NoteClient(),resume_from=path/'checkpoints/rounds-003',history_search=False)
            self.assertFalse((Path(tmp)/'bad').exists())
            # Missing new settings are the unchanged original condition, not an invalid schema.
            original=Path(tmp)/'original'
            run_private_notes(original,NoteClient(),records,'topic',selectors,ids,manifest)
            data,b=load_checkpoint(original/'checkpoints/rounds-003')
            try:
                self.assertFalse(b.history_search)
                self.assertNotIn('sequence_mode',data['settings.json'])
                self.assertNotIn('history_search',data['settings.json'])
                self.assertEqual(data['settings.json']['log_exposure'],'forced_before_research_and_answer')
            finally:b.close()
