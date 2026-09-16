"""Deterministic mocked concurrency and central-authority checks; no model launches."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

from orchestrator.simulated_web.async_notebooks import AsyncBudget, make_browser
from orchestrator.simulated_web.parallel_notebooks import run_parallel_notebooks, build_parallel_settings
from orchestrator.simulated_web.test_async_notebooks import ScriptedClient, inputs, tool
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.hf_transport import OwnedVllm
from orchestrator.simulated_web.hf_fp8 import VLLM_VERSION, MODEL
from orchestrator.simulated_web.timed_policy import TimedPolicy


class Client(ScriptedClient):
    def __init__(self, fn=None):
        super().__init__();self.fn=fn;self.cancelled=threading.Event()
    def cancel(self):self.cancelled.set()
    def __call__(self,agent,history,timeout,*,num_predict):
        if self.fn:return self.fn(self,agent,history,timeout,num_predict)
        return super().__call__(agent,history,timeout,num_predict=num_predict)


class ParallelTests(unittest.TestCase):
    def test_actual_overlap_immediate_publication_and_private_histories(self):
        overlap=threading.Barrier(2);published=threading.Event();read=threading.Event();mutated=threading.Event();persisted=threading.Event()
        count={'agent-1':0,'agent-2':0};settings,*_=build_parallel_settings(**inputs())
        entry='https://wiki.test/page/'+settings['notebooks']['agent-1']+'-entry-000001'
        original=make_browser
        def browser(*args,**kwargs):
            b=original(*args,**kwargs);call=b.call
            def observed(agent,name,args):
                value=call(agent,name,args)
                if name=='append_notebook':published.set()
                if name=='read_notebook':
                    self.assertEqual(value['text'],'早期 evidence & 🧪\nexact');read.set()
                return value
            b.call=observed;return b
        def generate(client,agent,history,timeout,allowance):
            count[agent]+=1
            if count[agent]==1:overlap.wait(timeout=2)
            if agent=='agent-1' and count[agent]==1:
                self.assertTrue(mutated.wait(2))
                message=tool('append_notebook',text='早期 evidence & 🧪\nexact')
            elif agent=='agent-2' and count[agent]==1:
                history.append({'role':'assistant','content':'worker_private_mutation'})
                mutated.set();self.assertTrue(persisted.wait(2))
                self.assertTrue(published.wait(2));message=tool('read_notebook',url=entry,revision='')
            else:
                if agent=='agent-1':self.assertTrue(read.wait(2))
                message={'content':'final'}
            client.allowances.append(allowance)
            return ModelResponse(message,{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        clients={a:Client(generate) for a in count}
        with tempfile.TemporaryDirectory() as temp,patch('orchestrator.simulated_web.parallel_notebooks.make_browser',browser):
            path=Path(temp)/'run'
            def checkpoint(path):
                if published.is_set() and not persisted.is_set():
                    self.assertNotIn('worker_private_mutation',(path/'histories.json').read_text());persisted.set()
            result=run_parallel_notebooks(path,clients,**inputs(),checkpoint_callback=checkpoint)
            self.assertEqual(result['status'],'complete')
            events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
            publication=next(e['sequence'] for e in events if e['event']=='notebook_published_available')
            delivery=next(e['sequence'] for e in events if e['event']=='notebook_read_returned')
            answer=next(e['sequence'] for e in events if e['event']=='question_terminated' and e['agent']=='agent-1')
            self.assertLess(publication,delivery);self.assertLess(delivery,answer)
            self.assertEqual(clients['agent-1'].allowances[:2],[4096,4093])
            with sqlite3.connect(path/'wiki.sqlite3') as db:self.assertEqual(db.execute('select count(*) from revisions').fetchone()[0],1)

    def test_worker_failure_does_not_cancel_peer(self):
        barrier=threading.Barrier(2)
        def fail(c,a,h,t,n):barrier.wait(2);raise TimeoutError('only A')
        first=[True]
        def succeed(c,a,h,t,n):
            if first[0]:first[0]=False;barrier.wait(2)
            self.assertFalse(c.cancelled.is_set())
            return ModelResponse({'content':'B answer'},{'eval_count':3,'done_reason':'stop'})
        with tempfile.TemporaryDirectory() as temp:
            result=run_parallel_notebooks(Path(temp)/'run',{'agent-1':Client(fail),'agent-2':Client(succeed)},**inputs())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['states']['agent-2']['status'],'complete')
            self.assertEqual(len(result['results']),2)

    def test_authority_failure_cancels_both_and_preserves_inflight(self):
        barrier=threading.Barrier(2)
        def generate(c,a,h,t,n):
            barrier.wait(2)
            if a=='agent-1':return ModelResponse(tool('open',url='https://docs.test/'),{'eval_count':3,'done_reason':'stop'})
            self.assertTrue(c.cancelled.wait(2));raise TimeoutError('cancelled peer')
        clients={a:Client(generate) for a in ('agent-1','agent-2')};original=make_browser
        def broken(*args,**kwargs):
            b=original(*args,**kwargs)
            def call(*args):raise RuntimeError('authority failed')
            b.call=call;return b
        with tempfile.TemporaryDirectory() as temp,patch('orchestrator.simulated_web.parallel_notebooks.make_browser',broken):
            path=Path(temp)/'run'
            with self.assertRaisesRegex(RuntimeError,'authority failed'):run_parallel_notebooks(path,clients,**inputs())
            self.assertTrue(all(c.cancelled.is_set() for c in clients.values()))
            events=(path/'events.jsonl').read_text();self.assertIn('inflight_outcome_preserved',events)
            self.assertEqual(json.loads((path/'manifest.json').read_text())['status'],'failed')

    def test_cleanup_failure_preserves_original_authority_error_and_state(self):
        class BrokenCancel(Client):
            def cancel(self):
                super().cancel()
                raise RuntimeError('cleanup unconfirmed')
        clients={'agent-1':BrokenCancel(),'agent-2':Client()}
        first=[True]
        def checkpoint(path):
            if first[0]:
                first[0]=False
                raise RuntimeError('original authority error')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            with self.assertRaisesRegex(RuntimeError,'original authority error'):
                run_parallel_notebooks(path,clients,**inputs(),checkpoint_callback=checkpoint)
            self.assertTrue(clients['agent-2'].cancelled.is_set())
            self.assertIn('original authority error',(path/'failure.json').read_text())
            self.assertTrue((path/'cleanup-failure.json').exists())
            self.assertEqual(json.loads((path/'manifest.json').read_text())['status'],'failed')
            self.assertTrue(all(s['status']=='interrupted' for s in json.loads((path/'state.json').read_text()).values()))

    def test_length_terminal_never_executes_tools(self):
        def generate(c,a,h,t,n):return ModelResponse(tool('append_notebook',text='discard'),{'eval_count':n,'done_reason':'length'})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run';result=run_parallel_notebooks(path,{a:Client(generate) for a in ('agent-1','agent-2')},**inputs())
            self.assertTrue(all(r['status']=='generation_limit_reached' for r in result['results']))
            with sqlite3.connect(path/'wiki.sqlite3') as db:self.assertEqual(db.execute('select count(*) from revisions').fetchone()[0],0)

    def test_device_and_port_isolation_without_environment_mutation(self):
        with tempfile.TemporaryDirectory() as temp,patch('orchestrator.simulated_web.hf_transport.sys.platform','linux'),patch('orchestrator.simulated_web.hf_transport.shutil.which',return_value='vllm'),patch('orchestrator.simulated_web.hf_transport.importlib.metadata.version',return_value=VLLM_VERSION),patch('orchestrator.simulated_web.hf_transport.check_loopback_port_available'),patch('orchestrator.simulated_web.hf_transport.subprocess.Popen') as popen,patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection') as http:
            environment_before=dict(os.environ)
            http.return_value.getresponse.return_value.status=200
            http.return_value.getresponse.return_value.read.return_value=json.dumps({'data':[{'id':MODEL}]}).encode()
            popen.return_value.poll.return_value=None
            clients=[OwnedVllm(TimedPolicy(),Path(temp),Path(temp)/f'{i}.log',port=8000+i,cuda_visible_devices=str(i)) for i in range(2)]
            for c in clients:c.metadata={};c._start(10**12)
            self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in popen.call_args_list],['0','1'])
            self.assertEqual([c.port for c in clients],[8000,8001])
            self.assertTrue(all(call.kwargs['start_new_session'] for call in popen.call_args_list))
            self.assertEqual(dict(os.environ),environment_before)
            with patch.object(clients[0], '_stop_locked') as stop:
                clients[0].cancel();stop.assert_called_once()
            self.assertFalse(clients[1].cancelled.is_set())
            with self.assertRaisesRegex(RuntimeError,'cancelled'):clients[0]._start(10**12)
            self.assertEqual(popen.call_count,2)
            for c in clients:c.server_log.close()
