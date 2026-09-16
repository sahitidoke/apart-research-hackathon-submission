"""008e history lifetime and persistent shared wiki contracts, with a mock client."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_token_pair import Client
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, reset_base_history, run_token_pair, RESET_PROMPT


class WikiClient(Client):
    def __call__(self, agent, history, timeout, **kwargs):
        self.calls.append((agent, json.loads(json.dumps(history))))
        if history[-1]['role'] == 'user':
            url = 'https://wiki.test/page/notes-1'
            if agent == 'agent-1':
                url = 'https://wiki.test/save?' + urlencode({'slug':'notes-1', 'title':'Notes 1', 'text':'Persistent observation'})
            message = {'content':'', 'tool_calls':[{'function':{'name':'open','arguments':{'url':url}}}]}
        else:
            message = {'content': 'PRIVATE_RESULT_' + agent}
        return ModelResponse(message, {'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})


class ResetWikiTests(unittest.TestCase):
    def test_reset_is_owner_only_preserves_research_wiki_and_checkpoint(self):
        records, selectors, manifest, ids = CrossedAccessTests().setup_inputs()
        policy = pair_policy(compaction_enabled=False, preparation_browser_calls=4, answer_browser_calls=4,
                             preparation_generated_tokens=2048, browser_retention='question_boundary')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            client = WikiClient()
            result = run_token_pair(path, records, 'topic', client, policy, selectors,
                pair_protocol='question_research', question_ids=ids, access_manifest=manifest,
                prompt_condition='reward_persistence', context_reset='after_answer', shared_wiki=True, log_exposure='forced')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(list(path.glob('context-reset-*'))), 6)
            self.assertEqual(len(list(path.glob('forced-log-*'))), 12)
            starts = [(a,h) for a,h in client.calls if h[-1]['role']=='user']
            self.assertEqual(len(starts), 12)
            for i, (agent, history) in enumerate(starts):
                text = json.dumps(history)
                self.assertIn(RESET_PROMPT, history[0]['content'])
                self.assertNotIn('PRIVATE_RESULT_' + ('agent-2' if agent=='agent-1' else 'agent-1'), text)
                if i % 4 < 2:
                    self.assertNotIn('PRIVATE_RESULT_', text)
                else:
                    self.assertIn('PRIVATE_RESULT_' + agent, text)
            cp = load_pair_checkpoint(path/'checkpoints/rounds-003')
            try:
                settings=cp.data['settings.json']
                self.assertEqual(settings['access_plan']['wiki_access'], 'shared')
                self.assertEqual(settings['maximum_phase_generated_tokens'], 24576)
                for history in cp.data['histories.json'].values():
                    self.assertEqual(history, reset_base_history(settings['system_prompt'],'topic'))
                b=cp.browser
                self.assertEqual(b.db.execute('SELECT count(*) FROM pages').fetchone()[0],3)
                self.assertEqual(b.db.execute('SELECT count(*) FROM revisions').fetchone()[0],6)
                self.assertEqual(b.db.execute('SELECT count(*) FROM request_events').fetchone()[0],24)
                for agent in ids:
                    self.assertEqual(b.call(agent,'open',{'url':'https://wiki.test/page/notes-1'})['text'],'Persistent observation')
                    root=b.call(agent,'open',{'url':'https://docs.test/'})
                    self.assertIn('https://wiki.test/',[link['url'] for link in root['links']])
                    self.assertIn('https://wiki.test/page/notes-1',[r['url'] for r in b.search('Persistent observation',agent)['results']])
                    self.assertNotIn('PRIVATE_RESULT',json.dumps(b.call(agent,'open',{'url':'https://wiki.test/'})))
                    denied=set(b.pages)-b.access_allowed_urls[agent]
                    self.assertTrue(denied)
                    self.assertEqual(b.call(agent,'open',{'url':next(iter(denied))}),{'error':'Access denied'})
                saved=b.call('agent-2','open',{'url':'https://wiki.test/save?' + urlencode({'slug':'notes-2','title':'Notes 2','text':'B observation'})})
                self.assertIn('saved',saved)
                self.assertEqual(b.call('agent-1','open',{'url':saved['saved']})['text'],'B observation')
            finally:
                cp.close()
            for override in ({'shared_wiki':False},{'context_reset':'none'}):
                with self.assertRaises(ValueError):
                    run_token_pair(Path(tmp)/'bad',None,None,Client(),resume_from=path/'checkpoints/rounds-003',**override)
                self.assertFalse((Path(tmp)/'bad').exists())
            self.assertEqual(run_token_pair(Path(tmp)/'done',None,None,Client(),resume_from=path/'checkpoints/rounds-003')['status'],'already_complete')

    def test_invalid_reset_or_wiki_before_artifacts(self):
        records, selectors, manifest, ids = CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad'
            for protocol,reset,wiki in [('standard','after_answer',True),('question_research','after_reflection',True),('question_research','after_answer','yes')]:
                with self.assertRaises(ValueError):
                    run_token_pair(path,records,'topic',Client(),pair_policy(),selectors,pair_protocol=protocol,
                                   question_ids=ids,context_reset=reset,shared_wiki=wiki)
                self.assertFalse(path.exists())
