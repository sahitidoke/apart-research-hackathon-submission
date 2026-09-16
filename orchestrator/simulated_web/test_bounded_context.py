"""Native fitting mocks: protected messages, current evidence and raw provenance."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.bounded_context import BoundedContextClient, MARKER
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.timed_transport import ContextExhausted
from orchestrator.simulated_web.private_notes import private_notes_policy, run_private_notes, load_checkpoint
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_mandatory_notes import MandatoryClient


class CountingClient:
    def __init__(self):self.requests=[]
    def count_context(self,history,timeout):return {'prompt_tokens':sum(len(m.get('content','')) for m in history)}
    def __call__(self,agent,history,timeout,**kwargs):
        self.requests.append(json.loads(json.dumps(history)))
        return ModelResponse({'content':'ok'},{'eval_count':1})


class BoundedContextTests(unittest.TestCase):
    def test_only_old_browser_payload_masked_native_fit_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=CountingClient();wrapped=BoundedContextClient(client,replace(private_notes_policy(),context_length=4096),Path(tmp))
            history=[{'role':'system','content':'system'}, {'role':'assistant','content':'own reasoning'},
                     {'role':'tool','tool_name':'open','content':'x'*3000}]
            wrapped.begin_question('agent-1',history)
            history.append({'role':'tool','tool_name':'open','content':'current source'*40})
            wrapped('agent-1',history,15,num_predict=512)
            self.assertEqual(history[2]['content'],MARKER)
            self.assertEqual(history[1]['content'],'own reasoning')
            self.assertEqual(history[3]['content'],'current source'*40)
            snapshot=next(Path(tmp).glob('*-before.json'))
            self.assertEqual(json.loads(snapshot.read_text())[2]['content'],'x'*3000)
            self.assertEqual(client.requests[0],history)

    def test_superseded_log_before_insertion_latest_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapped=BoundedContextClient(CountingClient(),private_notes_policy(),Path(tmp))
            history=[{'role':'tool','tool_name':'open','tool_call_id':'host-log-01-agent-1','content':'old note URL payload'}]
            wrapped.before_forced_exposure('agent-1',history)
            history.append({'role':'tool','tool_name':'open','tool_call_id':'host-log-02-agent-1','content':'new note URL payload'})
            wrapped('agent-1',history,15,num_predict=2048)
            self.assertEqual(history[0]['content'],MARKER)
            self.assertEqual(history[1]['content'],'new note URL payload')

    def test_protected_oversize_fails_without_model_request_or_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=CountingClient();wrapped=BoundedContextClient(client,replace(private_notes_policy(),context_length=4096),Path(tmp))
            history=[{'role':'assistant','content':'x'*5000}]
            with self.assertRaises(ContextExhausted):wrapped('agent-1',history,15,num_predict=512)
            self.assertEqual(client.requests,[])
            self.assertEqual(history[0]['content'],'x'*5000)
            self.assertTrue(list(Path(tmp).glob('*-failure.json')))

    def test_exact_template_preflight_retry_masks_without_duplicate_generation(self):
        class ExactClient(CountingClient):
            def __init__(self):super().__init__();self.attempts=0
            def count_context(self,history,timeout):return {'prompt_tokens':100}
            def __call__(self,agent,history,timeout,**kwargs):
                self.attempts+=1
                if self.attempts==1:raise ContextExhausted({'prompt_tokens':70000})
                return super().__call__(agent,history,timeout,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            client=ExactClient();wrapped=BoundedContextClient(client,private_notes_policy(),Path(tmp))
            history=[{'role':'tool','tool_name':'open','content':'older source'}]
            wrapped.begin_question('agent-1',history)
            wrapped('agent-1',history,15,num_predict=2048)
            self.assertEqual(client.attempts,2);self.assertEqual(len(client.requests),1)
            self.assertEqual(history[0]['content'],MARKER)
            self.assertEqual(len(list(Path(tmp).glob('*-request.json'))),2)

    def test_bounded_run_checkpoints_and_notebooks_preserved(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            status=run_private_notes(path,MandatoryClient(),records,'topic',selectors,ids,manifest,
                sequence_mode='agent_serial',retain_context=True,mandatory_notes=True,bounded_context=True)
            self.assertEqual(status['status'],'complete')
            data,browser=load_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertEqual(browser.db.execute('SELECT count(*) FROM revisions').fetchone()[0],6)
                self.assertTrue(data['settings.json']['bounded_context'])
                self.assertTrue(list(path.glob('bounded-context-*-before.json')))
            finally:browser.close()
