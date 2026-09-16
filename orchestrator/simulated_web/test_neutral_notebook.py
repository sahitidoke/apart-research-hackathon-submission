"""Neutral labels, honest prompts, disabled reward, and legacy-safe checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.private_notes import build_settings,load_checkpoint,run_private_notes,NEUTRAL_NOTEBOOKS
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_append_notebooks import AppendClient
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests


class NeutralClient(AppendClient):
    def __call__(self,agent,history,timeout,**kwargs):
        if history[-1]['role']=='user' and 'Notebook append:' in history[-1]['content']:
            return super().__call__(agent,history,timeout,**kwargs)
        self.calls.append((agent,json.loads(json.dumps(history))))
        message=({'content':'','tool_calls':[{'function':{'name':'open','arguments':{'url':'https://docs.test/'}}}]}
                 if history[-1]['role']=='user' else {'content':agent+' result'})
        return ModelResponse(message,{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})


class NeutralNotebookTests(unittest.TestCase):
    def test_neutral_surfaces_reward_disabled_and_resume(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        kwargs={'sequence_mode':'agent_serial','mandatory_notes':True,'append_notes':True,'retain_context':True,
                'bounded_context':True,'neutral_notebook':True}
        settings,*_=build_settings(records,'topic',selectors,ids,manifest,**kwargs)
        self.assertEqual(settings['notebooks'],NEUTRAL_NOTEBOOKS)
        self.assertFalse(settings['reward_enabled']);self.assertNotIn('privacy_claim',settings)
        for prompt in settings['system_prompts'].values():
            self.assertIn('Use it to record findings for later questions.',prompt)
            for forbidden in ('private','other agent','peer','alone','reward','1 - 0.1','search incentive'):
                self.assertNotIn(forbidden,prompt.lower())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'first'
            def stop(cp):
                if cp.name=='rounds-001':raise RuntimeError('mock stop')
            with self.assertRaisesRegex(RuntimeError,'mock stop'):
                run_private_notes(path,NeutralClient(),records,'topic',selectors,ids,manifest,checkpoint_callback=stop,**kwargs)
            cp=path/'checkpoints/rounds-001';data,browser=load_checkpoint(cp)
            try:
                self.assertTrue(all('private' not in title.lower() and 'agent-' not in title for title, in browser.db.execute('SELECT title FROM pages')))
                self.assertTrue(all('private' not in slug for slug, in browser.db.execute('SELECT slug FROM pages')))
                body=browser.call('agent-2','open',{'url':'https://wiki.test/page/research-1-entry-000001'})
                self.assertNotIn('error',body);self.assertEqual(body['title'],'Research notebook entry 1')
                self.assertTrue(all('stated_reward_accounting' not in row for row in data['results.json']))
                requests=' '.join(x[0] for x in browser.db.execute('SELECT requested FROM request_events'))
                self.assertNotIn('private-research',requests)
            finally:browser.close()
            second=Path(tmp)/'second';result=run_private_notes(second,NeutralClient(),resume_from=cp)
            self.assertEqual(result['status'],'complete')
            data,browser=load_checkpoint(second/'checkpoints/rounds-003');browser.close()
            self.assertEqual(data['settings.json']['notebooks'],NEUTRAL_NOTEBOOKS)
            self.assertTrue(all('stated_reward_accounting' not in row for row in data['results.json']))
