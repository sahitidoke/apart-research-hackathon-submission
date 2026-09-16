"""Mock probes preserve parent state and isolate spontaneous writes between agents."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from orchestrator.simulated_web.notebook_understanding_probe import QUESTION,run_understanding_probe
from orchestrator.simulated_web.private_notes import run_private_notes
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient
from orchestrator.simulated_web.test_token_pair import Client


class ProbeClient(Client):
    def __init__(self):super().__init__();self.second_read=None
    def __call__(self,agent,history,timeout,**kwargs):
        self.calls.append((agent,json.loads(json.dumps(history))))
        last=history[-1]
        if last['role']=='user':
            message={'content':'','tool_calls':[{'function':{'name':'read_notebook','arguments':{'url':'https://wiki.test/page/notes-cedar-entry-000001','revision':''}}}]}
        elif last.get('tool_name')=='read_notebook' and agent=='agent-1':
            data=json.loads(last['content'])
            message={'content':'','tool_calls':[{'function':{'name':'edit_notebook','arguments':{'url':data['url'],'expected_revision':data['revision'],'text':'probe-only-edit'}}}]}
        else:
            if agent=='agent-2':self.second_read=json.loads(last['content'])['text']
            message={'content':'I can read an entry and use edit_notebook with its current revision.'}
        return ModelResponse(message,{'eval_count':5,'prompt_eval_count':100,'done_reason':'stop'})


class UnderstandingProbeTests(unittest.TestCase):
    def test_exact_start_isolation_original_immutable_and_outputs_preserved(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent'
            run_private_notes(parent,NeutralClient(),records,'topic',selectors,ids,manifest,sequence_mode='agent_serial',
                retain_context=True,append_notes=True,mandatory_notes=True,neutral_notebook=True,notebook_tools=True,
                visible_labels={'agent-1':'8h0','agent-2':'8h1'})
            cp=parent/'checkpoints/rounds-003'
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir()}
            histories=json.loads((cp/'histories.json').read_text())
            client=ProbeClient();output=root/'probe'
            result=run_understanding_probe(output,client,cp)
            self.assertEqual(result['status'],'complete')
            for agent in histories:
                first=next(h for a,h in client.calls if a==agent)
                self.assertEqual(first[:len(histories[agent])],histories[agent])
                self.assertTrue(first[len(histories[agent])]['content'].startswith(QUESTION))
                self.assertEqual(json.loads((output/agent/'initial-history.json').read_text()),histories[agent])
                self.assertTrue((output/agent/'phase-00.jsonl').is_file())
            self.assertNotEqual(client.second_read,'probe-only-edit')
            for agent,expected in [('agent-1','probe-only-edit'),('agent-2',client.second_read)]:
                with sqlite3.connect(output/agent/'wiki.sqlite3') as db:
                    self.assertEqual(db.execute("SELECT body FROM pages WHERE slug='notes-cedar-entry-000001'").fetchone()[0],expected)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir()})
            with self.assertRaises(ValueError):run_understanding_probe(parent/'illegal',client,cp)
            self.assertFalse((parent/'illegal').exists())
