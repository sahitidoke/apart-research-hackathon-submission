"""Hard crossed corpus access and research-before-answer mock contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.source_access import access_plan, access_browser_options
from orchestrator.simulated_web.source_discovery import discovery_plan, browser_discovery, dataset_digest
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair, validate_pair


class CrossedAccessTests(unittest.TestCase):
    def setup_inputs(self):
        records, selectors = inputs()
        manifest={'schema':'source-access-v1','dataset_sha256':dataset_digest(records),
                  'corpus_questions':{'agent-1':[r['id'] for r in records[:5]],'agent-2':[r['id'] for r in records[5:]]}}
        ids={'agent-1':[r['id'] for r in records[5:8]],'agent-2':[r['id'] for r in records[:3]]}
        return records,selectors,manifest,ids

    def test_hard_routes_edits_clicks_and_shared_denied_history(self):
        records,selectors,manifest,ids=self.setup_inputs()
        policy=pair_policy()
        pages,_,_,editable=validate_pair(records,'topic',policy,selectors,'question_research',ids)
        discovery=discovery_plan(records,pages,editable)
        access=access_plan(records,pages,discovery,manifest)
        b=Browser(pages,':memory:',editable_sources=editable,request_history_mode='shared',
                  **browser_discovery(discovery),**access_browser_options(access))
        try:
            denied=set(access['allowed_urls']['agent-1'])-set(access['allowed_urls']['agent-2'])
            for url in denied:
                self.assertEqual(b.call('agent-2','open',{'url':url}),{'error':'Access denied'})
            url=next(iter(b.source_urls.values()))[0]
            view=b.call('agent-1','open',{'url':url})
            # Guessing an owned view id cannot open its foreign source through click.
            self.assertIn('error',b.call('agent-2','click',{'page_id':view['page_id'],'link_id':1}))
            b.views.setdefault('agent-2', {})['p999']=[{'label':'foreign', 'url':url}]
            self.assertEqual(b.call('agent-2','click',{'page_id':'p999','link_id':1}),{'error':'Access denied'})
            identity=next(iter(b.source_urls))
            for route,fields in [('edit',{'source':identity}),('save',{'source':identity,'title':'intrusion','text':'bad'})]:
                self.assertEqual(b.call('agent-2','open',{'url':'https://docs.test/source/'+route+'?'+urlencode(fields)}),{'error':'Access denied'})
            self.assertEqual(b.db.execute('SELECT count(*) FROM revisions').fetchone()[0],0)
            self.assertEqual(b.call('agent-2','open',{'url':'https://wiki.test/'}),{'error':'Access denied'})
            root=b.call('agent-2','open',{'url':'https://docs.test/'})
            self.assertFalse(any(link['url'] in denied for link in root['links']))
            for visible_url in set(access['allowed_urls']['agent-2']) & set(discovery['listing_urls']):
                listing=b.call('agent-2','open',{'url':visible_url})
                self.assertTrue(all(f'Document {i}' not in listing['text'] for i in range(5)))
            self.assertFalse(any(row['url'] in denied for row in b.search('source','agent-2')['results']))
            log=b.call('agent-1','open',{'url':'https://docs.test/request-history'})
            self.assertIn('error',log['text'])
            self.assertIn(url,log['text'])
            self.assertNotIn('Evidence 0',log['text'])
            own_save=b.call('agent-1','open',{'url':'https://docs.test/source/save?'+urlencode({'source':identity,'title':'own edit','text':'private edited body'})})
            self.assertIn('saved',own_save)
            self.assertEqual(b.call('agent-1','open',{'url':own_save['saved']})['text'],'private edited body')
            self.assertEqual(b.call('agent-2','open',{'url':own_save['saved']}),{'error':'Access denied'})
        finally:b.close()

    def test_research_order_retention_access_checkpoint_and_no_future_question(self):
        records,selectors,manifest,ids=self.setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run';client=Client()
            result=run_token_pair(path,records,'topic',client,pair_policy(compaction_enabled=False,browser_retention='question_boundary',preparation_generated_tokens=2048,preparation_browser_calls=4),selectors,
                                  pair_protocol='question_research',question_ids=ids,access_manifest=manifest,log_exposure='forced')
            self.assertEqual(result['completed_rounds'],3)
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual([(r['agent'],r['phase']) for r in rows],[(a,p) for _ in range(3) for p in ('preparation','answer') for a in ids])
            self.assertTrue(all(r.get('phase_role')=='question_research' for r in rows if r['phase']=='preparation'))
            self.assertEqual(len(list(path.glob('forced-log-*'))),12)
            self.assertEqual(len(list(path.glob('browser-retention-*'))),6)
            prompts=[h[-1]['content'] for _,h in client.calls if h[-1]['role']=='user']
            self.assertEqual(len(prompts),12)
            self.assertTrue(all('Document collection: https://docs.test/\n' in p for p in prompts))
            questions={r['id']:r['question'] for r in records}
            for i,prompt in enumerate(prompts):
                agent=('agent-1','agent-2')[i%2];slot=i//4
                self.assertIn(questions[ids[agent][slot]],prompt)
                self.assertTrue(all(questions[q] not in prompt for q in ids[agent][slot+1:]))
            # Research tool observations survive until that owner's answer begins.
            answer_histories=[h for _,h in client.calls if h[-1]['role']=='user' and h[-1]['content'].startswith('Answer phase:')]
            self.assertTrue(all(any(m['role']=='tool' and 'Document collection' in m.get('content','') for m in h) for h in answer_histories))
            cp=load_pair_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertTrue(cp.complete)
                self.assertEqual(cp.manifest['next_phase_index'],12)
                self.assertIsNotNone(cp.browser.access_allowed_urls)
                self.assertEqual(cp.data['settings.json']['maximum_phase_generated_tokens'],24576)
            finally:cp.close()
            with self.assertRaises(ValueError):
                run_token_pair(Path(tmp)/'bad',None,None,Client(),resume_from=path/'checkpoints/rounds-003',access_manifest=manifest)
            self.assertFalse((Path(tmp)/'bad').exists())
