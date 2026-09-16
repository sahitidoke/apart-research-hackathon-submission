import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from orchestrator.simulated_web.hf_fp8 import server_command, validate_snapshot
from orchestrator.simulated_web.hf_transport import OwnedVllm, chat_body, hf_messages, native_count, effective_tools
from orchestrator.simulated_web.browser import TOOLS
from orchestrator.simulated_web.research_reflection import build_settings as reflection_settings
from orchestrator.simulated_web.test_same_question_pairing import paired_options
from orchestrator.simulated_web.timed_transport import ContextExhausted, DeadlineExpired
from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, VLLM_VERSION, KV_CACHE_DTYPE
from orchestrator.simulated_web.paired_notebook_views import build_paired_settings, run_paired_views, load_checkpoint
from orchestrator.simulated_web.test_paired_notebook_views import inputs
from orchestrator.simulated_web.test_neutral_notebook import NeutralClient


class HFTransportTests(unittest.TestCase):
    def client(self):
        client = OwnedVllm.__new__(OwnedVllm)
        client.metadata = {'checked': True}
        client.ready = True
        client.policy = SimpleNamespace(context_length=4096)
        client.seed, client.port, client.events = 0, 8000, []
        client.request_lock = threading.Lock()
        client._start, client.stop = Mock(), Mock()
        return client

    def connection(self, prompt_count=2, completion_count=1, error=None):
        connection = Mock()
        def request(method, path, raw, headers):
            connection.path, connection.body = path, json.loads(raw)
        connection.request.side_effect = request
        def response():
            if connection.path == '/tokenize':
                value = {'tokens': [10, 11], 'count': 2, 'max_model_len': 4096}
            else:
                if error:
                    raise error
                value = {'choices': [{'finish_reason': 'stop', 'message': {
                    'content': 'done', 'reasoning': '', 'tool_calls': []}}],
                         'usage': {'prompt_tokens': prompt_count, 'completion_tokens': completion_count}}
            return SimpleNamespace(status=200, read=lambda limit: json.dumps(value).encode())
        connection.getresponse.side_effect = response
        return connection

    def test_history_retains_reasoning_and_pairs_tool_ids(self):
        mapped = hf_messages([{'role': 'assistant', 'thinking': 'remember evidence', 'content': '',
                              'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://wiki.test/'}}}]},
                             {'role': 'tool', 'tool_name': 'open', 'content': 'evidence'}])
        self.assertEqual(mapped[0]['reasoning'], 'remember evidence')
        self.assertEqual(mapped[0]['tool_calls'][0]['id'], mapped[1]['tool_call_id'])
        self.assertEqual(json.loads(mapped[0]['tool_calls'][0]['function']['arguments']), {'url': 'https://wiki.test/'})

    def test_unpaired_history_fails(self):
        with self.assertRaises(ValueError):
            hf_messages([{'role': 'tool', 'tool_name': 'open', 'content': 'orphan'}])

    def test_final_mode_and_preflight_share_render_fields(self):
        final = chat_body([{'role': 'user', 'content': 'answer'}], [{'function': {}}], True)
        self.assertNotIn('tools', final)
        self.assertEqual(final['chat_template_kwargs'], {'enable_thinking': False, 'reasoning_effort': 'medium', 'preserve_thinking': True})
        client, connection = self.client(), self.connection()
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection', return_value=connection):
            result = client('a', [{'role': 'user', 'content': 'answer'}], 5, num_predict=20, final_only=True)
        calls = connection.request.call_args_list
        tokenize, generate = [json.loads(call.args[2]) for call in calls]
        self.assertEqual(tokenize, {k: generate[k] for k in tokenize})
        self.assertEqual(result.metadata['eval_count'], 1)
        client.stop.assert_not_called()

    def test_10a_actual_tokenize_and_chat_advertise_browser_and_notebooks(self):
        settings,_,_=reflection_settings(**paired_options())
        for final_only in (False,True):
            client,connection=self.client(),self.connection()
            client.notebook_tools_enabled=True
            client.notebook_tool_schemas=settings['notebook_tool_schemas']
            with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
                result=client('agent-1',[{'role':'user','content':'Research'}],5,num_predict=20,final_only=final_only)
            tokenize,chat=[json.loads(c.args[2]) for c in connection.request.call_args_list]
            self.assertEqual(tokenize,{k:chat[k] for k in tokenize})
            expected=[] if final_only else ['search','open','click','read_notebook','append_notebook']
            self.assertEqual([t['function']['name'] for t in chat.get('tools',[])],expected)
            self.assertEqual(result.metadata['context_preflight']['tool_names'],expected)
            if not final_only:
                self.assertEqual(chat['tools'][:3],TOOLS)
                self.assertEqual(chat['tools'][3:],settings['notebook_tool_schemas'])
                self.assertNotIn('accessible research notebook',json.dumps(chat['tools']))
            else:self.assertNotIn('tools',chat)

    def test_count_context_uses_same_effective_tools_and_duplicates_are_safe(self):
        client,connection=self.client(),self.connection()
        client.notebook_tools_enabled=True
        client.notebook_tool_schemas=[*TOOLS]
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
            diagnostics=client.count_context([],timeout=5)
        self.assertEqual(len(connection.request.call_args_list),1)
        self.assertEqual(connection.body['tools'],TOOLS)
        self.assertEqual(diagnostics['tool_names'],['search','open','click'])
        client.notebook_tool_schemas=[{'type':'function','function':{'name':'open','description':'conflict'}}]
        with self.assertRaisesRegex(ValueError,'Conflicting'):effective_tools(client)
        client.notebook_tools_enabled=False
        self.assertEqual(effective_tools(client),TOOLS)

    def test_timeout_reports_dispatch_stage_without_guessing_native_usage(self):
        for during_tokenize in (False,True):
            client,connection=self.client(),self.connection(error=TimeoutError('expired'))
            if during_tokenize:connection.getresponse.side_effect=TimeoutError('tokenize expired')
            with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
                with self.assertRaises(DeadlineExpired) as caught:client('a',[],5,num_predict=20)
            detail=caught.exception.transport_timeout
            self.assertEqual(detail['generation_dispatched'],not during_tokenize)
            self.assertEqual(detail['request_path'],'/tokenize' if during_tokenize else '/v1/chat/completions')
            self.assertEqual(detail['generation_tokens_if_not_dispatched'],0 if during_tokenize else None)
            self.assertNotIn('eval_count',detail)
            self.assertFalse(detail['completion_usage_available'])
            client.stop.assert_called_once()

    def test_native_mismatch_kills_and_refuses_response(self):
        client, connection = self.client(), self.connection(prompt_count=3)
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, 'native token'):
                client('a', [], 5, num_predict=20)
        client.stop.assert_called_once()

    def test_context_exhaustion_never_dispatches_generation(self):
        client, connection = self.client(), self.connection()
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection', return_value=connection), \
             patch('orchestrator.simulated_web.hf_transport.native_count', return_value=4096):
            with self.assertRaises(ContextExhausted):
                client('a', [], 5, num_predict=20)
        self.assertEqual([call.args[1] for call in connection.request.call_args_list], ['/tokenize'])
        client.stop.assert_not_called()

    def test_deadline_kills_owned_group(self):
        client, connection = self.client(), self.connection(error=TimeoutError('expired'))
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection', return_value=connection):
            with self.assertRaises(DeadlineExpired):
                client('a', [], 5, num_predict=20)
        client.stop.assert_called_once()
        self.assertFalse(client.events[-1]['partial_text_available'])

    def test_invalid_native_count_refuses_fallback(self):
        for value in ({'tokens': [1], 'count': 2, 'max_model_len': 4096},
                      {'tokens': [1], 'count': 1, 'max_model_len': 8192}):
            with self.assertRaises(RuntimeError):
                native_count(value, 4096)

    def test_command_separates_fp8_weights_from_bf16_kv(self):
        command = server_command('/cached', 65536, 8000)
        self.assertEqual(command[command.index('--quantization') + 1], 'fp8')
        self.assertEqual(command[command.index('--kv-cache-dtype') + 1], 'auto')
        self.assertEqual(command[command.index('--dtype') + 1], 'bfloat16')
        self.assertNotIn('--trust-remote-code', command)
        with self.assertRaises(ValueError):
            server_command('/cached', 131072, 8000)

    def test_missing_and_mutated_snapshot_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = b'{}'
            row = {'rfilename': 'config.json', 'size': len(data),
                   'blobId': hashlib.sha1(b'blob 2\0' + data).hexdigest()}
            with patch('orchestrator.simulated_web.hf_fp8.artifact_lock', return_value={'files': [row]}):
                with self.assertRaisesRegex(ValueError, 'Missing'):
                    validate_snapshot(root)
                self.assertEqual(list(root.iterdir()), [])
                (root / 'config.json').write_bytes(b'[]')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    validate_snapshot(root)

    def test_profile_is_opt_in_and_legacy_model_is_rejected_before_output(self):
        legacy = build_paired_settings(**inputs())[0]
        self.assertNotIn('inference_profile', legacy)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'fp8'
            client = NeutralClient()
            client.count_text = Mock(return_value={'tokens': 1})
            with self.assertRaisesRegex(ValueError, 'HF FP8 model contract'):
                run_paired_views(destination, client, **inputs(), inference_profile='hf-fp8-v1')
            self.assertFalse(destination.exists())

    def test_fp8_mock_schedule_checkpoint_and_resume(self):
        metadata = {'name': MODEL, 'revision': REVISION, 'digest': 'hf:' + REVISION,
                    'backend': 'vllm', 'backend_version': VLLM_VERSION,
                    'kv_cache_dtype': KV_CACHE_DTYPE, 'all_artifact_checksums_verified': True}
        def boundary(path):
            if path.name == 'rounds-001':
                raise RuntimeError('mock stop')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = NeutralClient()
            client.count_text = Mock(return_value={'tokens': 1})
            with patch.object(client, 'inspect_model', return_value=metadata):
                with self.assertRaisesRegex(RuntimeError, 'mock stop'):
                    run_paired_views(root / 'first', client, **inputs(), inference_profile='hf-fp8-v1', checkpoint_callback=boundary)
                checkpoint = root / 'first/checkpoints/rounds-001'
                data, browser = load_checkpoint(checkpoint)
                browser.close()
                self.assertEqual(data['settings.json']['inference_profile'], 'hf-fp8-v1')
                with self.assertRaisesRegex(ValueError, 'pinned model digest'):
                    run_paired_views(root / 'legacy', client, **inputs())
                run_paired_views(root / 'continued', client, resume_from=checkpoint)
                self.assertEqual(json.loads((root / 'continued/manifest.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()


class ExternalCancellationTests(unittest.TestCase):
    client=HFTransportTests.client
    connection=HFTransportTests.connection
    def test_owner_abort_is_not_deadline_and_native_count_stays_unknown(self):
        client,connection=self.client(),self.connection()
        callbacks=[]
        client._register_abort=lambda callback:callbacks.append(callback) or 1
        client._unregister_abort=Mock()
        original=connection.getresponse.side_effect
        def response():
            if connection.path=='/v1/chat/completions':
                callbacks[0]()
                raise OSError('socket closed by owner')
            return original()
        connection.getresponse.side_effect=response
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection):
            with self.assertRaises(RuntimeError) as caught:
                client('agent-1',[{'role':'user','content':'research'}],60,num_predict=20)
        self.assertNotIsInstance(caught.exception,TimeoutError)
        self.assertIsNone(caught.exception.native_usage)
        self.assertEqual(caught.exception.transport_failure['reason'],'owner_cancellation')
        self.assertTrue(caught.exception.transport_failure['generation_dispatched'])
        self.assertFalse(any(e['event']=='request_timeout' for e in client.events))
        client.stop.assert_called_once()

    def test_deadline_wins_then_late_owner_abort_cannot_relabel_it(self):
        client,connection=self.client(),self.connection();callbacks=[];timers=[]
        client._register_abort=lambda callback:callbacks.append(callback) or 1
        client._unregister_abort=Mock()
        def timer(delay,callback):
            timers.append(callback)
            return Mock()
        original=connection.getresponse.side_effect
        def response():
            if connection.path=='/v1/chat/completions':
                timers[0]()
                callbacks[0]()
                raise OSError('deadline closed socket')
            return original()
        connection.getresponse.side_effect=response
        with patch('orchestrator.simulated_web.hf_transport.http.client.HTTPConnection',return_value=connection), patch('orchestrator.simulated_web.hf_transport.threading.Timer',side_effect=timer):
            with self.assertRaises(DeadlineExpired):client('agent-1',[{'role':'user','content':'research'}],60,num_predict=20)
        self.assertTrue(any(e['event']=='request_timeout' for e in client.events))
        self.assertFalse(any(e['event']=='request_external_cancellation' for e in client.events))
