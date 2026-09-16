"""Mock-only paired schedule, real notebook snapshots, and full-round resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web.bounded_context import BoundedContextClient,MARKER
from orchestrator.simulated_web.notebook_tools import TOOLS as LEGACY_TOOLS
from orchestrator.simulated_web.paired_notebook_views import build_paired_settings,browser_for,load_checkpoint,run_paired_views,sequence,TOOLS
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient


def inputs():
    records,selectors,access,_=CrossedAccessTests().setup_inputs()
    ids=[records[0]['id'],records[5]['id']]
    return dict(records=records,topic='topic',selectors=selectors,question_ids={a:ids for a in ('agent-1','agent-2')},
                access_manifest=access,visible_labels={'agent-1':'90','agent-2':'91'},round_leaders=['agent-1','agent-2'])


class PairedTests(unittest.TestCase):
    def test_schedule_views_budget_protection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'first';client=NeutralClient();observed=[]
            original=BoundedContextClient.fit
            def check(wrapper,agent,history,*args,**kwargs):
                protected=[i for i,m in enumerate(history) if m.get('tool_call_id','').startswith('host-notebook-view-') and i>=wrapper.starts.get(agent,0)]
                self.assertTrue(all(i not in wrapper.candidates(agent,history) for i in protected))
                self.assertTrue(all(history[i]['content']!=MARKER for i in protected));observed.append(len(protected))
                return original(wrapper,agent,history,*args,**kwargs)
            def stop(cp):
                if cp.name=='rounds-001':raise RuntimeError('mock boundary stop')
            with patch.object(BoundedContextClient,'fit',check),self.assertRaisesRegex(RuntimeError,'mock boundary stop'):
                run_paired_views(path,client,**inputs(),checkpoint_callback=stop)
            cp=path/'checkpoints/rounds-001';data,browser=load_checkpoint(cp);browser.close()
            self.assertEqual(data['state.json'],{'completed_rounds':1,'next_phase_index':7})
            rows=data['results.json'];self.assertEqual([r['phase_role'] for r in rows],['research1','research','answer','note','research2','answer','note'])
            self.assertEqual([r['agent'] for r in rows],['agent-2','agent-1','agent-1','agent-1','agent-2','agent-2','agent-2'])
            before=json.loads((path/'paired-view-01-before.json').read_text());after=json.loads((path/'paired-view-01-after.json').read_text())
            self.assertEqual(before['entry_urls_before'],[])
            saved=rows[3]['note_preservation'];self.assertEqual(after['host_actions'][-1]['url'],saved['saved_url'])
            self.assertEqual(after['host_actions'][-1]['response']['text'],rows[3]['answer'])
            self.assertEqual(after['host_actions'][-1]['response']['author'],'90')
            self.assertEqual(after['host_actions'][-1]['response']['revision'],f'r-{saved["saved_revision"]}')
            self.assertTrue(all(a['request_event_ids'] for a in after['host_actions']))
            self.assertGreaterEqual(max(observed),3)
            settings=data['settings.json'];self.assertEqual(settings['maximum_phases'],14);self.assertEqual(settings['maximum_generated_tokens'],32768)
            for i in (0,4):
                self.assertEqual(rows[i]['generated_token_allowance'],1024)
                self.assertEqual(rows[i]['browser_call_limit'],2)
            resumed=Path(tmp)/'resumed'
            self.assertEqual(run_paired_views(resumed,NeutralClient(),resume_from=cp)['status'],'complete')
            final,browser=load_checkpoint(resumed/'checkpoints/rounds-002');browser.close()
            self.assertEqual([r['agent'] for r in final['results.json'][7:]],['agent-1','agent-2','agent-2','agent-2','agent-1','agent-1','agent-1'])
            second_before=json.loads((resumed/'paired-view-02-before.json').read_text());second_after=json.loads((resumed/'paired-view-02-after.json').read_text())
            self.assertEqual(second_before['prior_entry'],second_after['prior_entry'])
            self.assertEqual(len(second_before['host_actions']),2);self.assertEqual(len(second_after['host_actions']),3)
            self.assertEqual(second_before['host_actions'][1]['response'],second_after['host_actions'][1]['response'])
            self.assertEqual(run_paired_views(Path(tmp)/'complete',NeutralClient(),resume_from=resumed/'checkpoints/rounds-002')['status'],'already_complete')
            self.assertFalse((Path(tmp)/'complete').exists())
            transition_path=cp/'transitions.json';bad=json.loads(transition_path.read_text());bad[0]['phase_role']='research'
            transition_path.write_text(json.dumps(bad));manifest_path=cp/'paired-view-checkpoint.json';manifest=json.loads(manifest_path.read_text())
            manifest['files_sha256']['transitions.json']=hashlib.sha256(transition_path.read_bytes()).hexdigest();manifest_path.write_text(json.dumps(manifest))
            invalid=Path(tmp)/'invalid'
            with self.assertRaisesRegex(ValueError,'Invalid paired phase'):run_paired_views(invalid,NeutralClient(),resume_from=cp)
            self.assertFalse(invalid.exists())

    def test_note_failure_preserves_answer_without_after_or_admission(self):
        class Blank(NeutralClient):
            def __call__(self,agent,history,timeout,**kwargs):
                if kwargs.get('final_only') and 'Notebook append:' in history[-1].get('content',''):
                    return ModelResponse({'content':''},{'eval_count':10,'prompt_eval_count':100,'done_reason':'stop'})
                return super().__call__(agent,history,timeout,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed'
            with self.assertRaises(RuntimeError):run_paired_views(path,Blank(),**inputs())
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual(len(rows),4);self.assertEqual(rows[2]['phase_role'],'answer')
            self.assertTrue(rows[2]['answer']);self.assertFalse(rows[3]['note_preservation']['persistence_verified'])
            self.assertFalse((path/'paired-view-01-after.json').exists());self.assertFalse((path/'checkpoints/rounds-001').exists())

    def test_permission_is_opt_in_and_directory_error_actionable(self):
        self.assertNotEqual(TOOLS,LEGACY_TOOLS)
        self.assertFalse(any('You may edit any accessible' in t['function']['description'] for t in LEGACY_TOOLS))
        settings,pages,_,editable=build_paired_settings(**inputs());browser=browser_for(settings,pages,editable,':memory:')
        try:
            root='https://wiki.test/page/'+settings['notebooks']['agent-1']
            result=browser.call('agent-2','edit_notebook',{'url':root,'expected_revision':'r-1','text':'x'})
            self.assertIn('directory',result['error']);self.assertEqual(result['entries'],[])
            saved=browser.call('agent-1','open',{'url':'https://wiki.test/append?'+urlencode({'slug':settings['notebooks']['agent-1'],'text':'Actual note with spaces & newline\nDetails'})})
            result=browser.call('agent-2','edit_notebook',{'url':root,'expected_revision':'r-1','text':'x'})
            link=result['entries'][0];self.assertEqual(link['url'],saved['saved'])
            read=browser.call('agent-2','read_notebook',{'url':link['url'],'revision':''});self.assertEqual(link['revision'],read['revision'])
            edited=browser.call('agent-2','edit_notebook',{'url':link['url'],'expected_revision':link['revision'],'text':'Reviewed note'})
            self.assertNotIn('error',edited)
        finally:browser.close()
