"""Prepared mocked endpoint/preflight checks; never launch Ollama or use a GPU."""
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse
from orchestrator.simulated_web.session import (
    AgentOllama, endpoint_ports, gpu_configuration, main, make_schedule, model_metadata, run_session,
)
from orchestrator.simulated_web.test_session import records


def connection_for(models=None, status=200, body=None):
    connection = MagicMock()
    response = connection.getresponse.return_value
    response.status = status
    response.read.return_value = body if body is not None else json.dumps({'models': models}).encode()
    return connection


def installed(digest='sha256:same'):
    return [{'name': 'qwen3.5:9b', 'digest': digest, 'details': {'family': 'qwen'}}]


class EndpointTests(unittest.TestCase):
    def test_shared_compatibility_and_invalid_assignments(self):
        self.assertEqual(endpoint_ports(2), {'agent-1': 11434, 'agent-2': 11434})
        self.assertEqual(endpoint_ports(2, port=12000), {'agent-1': 12000, 'agent-2': 12000})
        for ports in ([], [12001], [12001, 12001], [0, 12001], [12001, 65536]):
            with self.subTest(ports=ports), self.assertRaises(ValueError):
                endpoint_ports(2, agent_ports=ports)
        with self.assertRaises(ValueError):
            endpoint_ports(2, port=12000, agent_ports=[12001, 12002])

    def test_routing_preserves_identity_limits_and_final_only_transport(self):
        connections = [connection_for(body=json.dumps({'message': {'content': 'answer'},
                        'eval_count': 3, 'prompt_eval_count': 7}).encode()) for _ in range(2)]
        client = AgentOllama('qwen3.5:9b', endpoint_ports(2, agent_ports=[12001, 12002]),
                             19, 262144, 8192)
        messages = [{'role': 'user', 'content': 'question'}]
        with patch('orchestrator.simulated_web.runner.http.client.HTTPConnection',
                   side_effect=connections) as connect:
            result = client('agent-2', messages, 12.5, num_predict=256, final_only=True)
            client('agent-1', messages, 8, num_predict=9000, final_only=False)
        self.assertEqual([call.args for call in connect.call_args_list],
                         [('127.0.0.1', 12002), ('127.0.0.1', 12001)])
        self.assertEqual(connect.call_args_list[0].kwargs, {'timeout': 12.5})
        final = json.loads(connections[0].request.call_args.args[2])
        normal = json.loads(connections[1].request.call_args.args[2])
        self.assertEqual(final['options'], {'seed': 20, 'num_predict': 256, 'num_ctx': 262144})
        self.assertEqual(normal['options'], {'seed': 19, 'num_predict': 8192, 'num_ctx': 262144})
        self.assertFalse(final['think'])
        self.assertNotIn('tools', final)
        self.assertIn('tools', normal)
        self.assertEqual(final['messages'], messages)
        self.assertEqual(client.max_output_tokens, 8192)
        self.assertEqual(result.metadata['eval_count'], 3)
        self.assertTrue(result.metadata['final_only'])
        for connection in connections:
            connection.close.assert_called_once()
        with self.assertRaises(KeyError):
            client('agent-3', messages, 8)

    def test_separate_endpoints_share_peer_saves_during_same_slot(self):
        saved, read = Event(), Event()
        source = records(2)
        first = {agent: order[0] for agent, order in make_schedule(['q0', 'q1'], 2, 0)['orders'].items()}
        ports = endpoint_ports(2, agent_ports=[12001, 12002])
        router = AgentOllama('qwen3.5:9b', ports, 0, 262144, 8192)

        def model_call(transport, agent, messages, timeout, **options):
            self.assertEqual(transport.port, ports[agent])
            qid = 'q' + messages[1]['content'].split()[1].rstrip('?')
            tools = [message for message in messages if message['role'] == 'tool']
            if qid == first[agent]:
                if agent == 'agent-1':
                    if not tools:
                        url = 'https://wiki.test/save?' + urlencode({
                            'slug': 'live-peer', 'title': 'Live peer note', 'text': 'Same-slot evidence'})
                        return {'content': '', 'tool_calls': [{'function': {
                            'name': 'open', 'arguments': {'url': url}}}]}
                    saved.set()
                elif not tools:
                    self.assertTrue(saved.wait(5), 'Peer save did not complete within the same slot')
                    return {'content': '', 'tool_calls': [{'function': {
                        'name': 'open', 'arguments': {'url': 'https://wiki.test/page/live-peer'}}}]}
                else:
                    self.assertIn('Same-slot evidence', json.loads(tools[-1]['content'])['text'])
                    read.set()
            return ModelResponse({'content': 'answer'}, {'eval_count': 1, 'prompt_eval_count': 2})

        with tempfile.TemporaryDirectory() as temporary, patch(
                'orchestrator.simulated_web.runner.Ollama.__call__', new=model_call):
            outcomes = run_session(Path(temporary) / 'run', source, router, steps=3, history_mode='reset')
        self.assertTrue(read.is_set())
        self.assertTrue(all(row['status'] == 'complete' for row in outcomes))
        self.assertEqual(len(outcomes), 4)

    def test_model_preflight_checks_every_endpoint_and_deduplicates_shared(self):
        for ports in (endpoint_ports(2), endpoint_ports(2, agent_ports=[12001, 12002])):
            connections = [connection_for(installed()) for _ in set(ports.values())]
            with patch('orchestrator.simulated_web.session.http.client.HTTPConnection',
                       side_effect=connections) as connect:
                metadata = model_metadata('qwen3.5:9b', ports)
            self.assertEqual(set(metadata), set(ports.values()))
            self.assertEqual(connect.call_count, len(set(ports.values())))
            for connection in connections:
                connection.request.assert_called_once_with('GET', '/api/tags')
                connection.getresponse.return_value.read.assert_called_once_with(MAX_RESPONSE + 1)
                connection.close.assert_called_once()

    def test_cli_rejects_bad_second_endpoint_before_run_creation(self):
        failures = [connection_for(installed('sha256:different')), connection_for([]),
                    connection_for(installed(None)), connection_for(installed('')),
                    connection_for(installed(), status=503),
                    connection_for(body=b'x' * (MAX_RESPONSE + 1)),
                    connection_for(body=b'not json')]
        for failed in failures:
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / 'dataset.jsonl'
                dataset.write_text(''.join(json.dumps(row) + '\n' for row in records(2)))
                argv = ['session', '--dataset', str(dataset), '--run-dir', str(root / 'run'),
                        '--agent-ports', '12001', '12002']
                first = connection_for(installed())
                with patch('sys.argv', argv), patch(
                        'orchestrator.simulated_web.session.http.client.HTTPConnection',
                        side_effect=[first, failed]), self.assertRaises(ValueError):
                    main()
                self.assertFalse((root / 'run').exists())
                first.close.assert_called_once()
                failed.close.assert_called_once()

    def test_cli_connection_failure_leaves_no_run_and_closes_connection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset.jsonl'
            dataset.write_text(''.join(json.dumps(row) + '\n' for row in records(2)))
            first, second = connection_for(installed()), MagicMock()
            second.request.side_effect = ConnectionRefusedError('second endpoint offline')
            argv = ['session', '--dataset', str(dataset), '--run-dir', str(root / 'run'),
                    '--agent-ports', '12001', '12002']
            with patch('sys.argv', argv), patch(
                    'orchestrator.simulated_web.session.http.client.HTTPConnection',
                    side_effect=[first, second]), self.assertRaises(ConnectionRefusedError):
                main()
            self.assertFalse((root / 'run').exists())
            second.close.assert_called_once()

    def test_invalid_routing_does_not_read_dataset_or_connect(self):
        for extra in (['--agent-ports', '12001'], ['--agent-ports', '12001', '12001'],
                      ['--agent-ports', '12001', '12002', '--gpu-devices', '3']):
            argv = ['session', '--dataset', '/missing/dataset', '--run-dir', '/unused/run'] + extra
            with patch('sys.argv', argv), patch('pathlib.Path.read_bytes') as read, patch(
                    'orchestrator.simulated_web.session.http.client.HTTPConnection') as connect:
                with self.assertRaises(ValueError):
                    main()
                read.assert_not_called()
                connect.assert_not_called()

    def test_cli_records_mapping_before_coordinator_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset.jsonl'
            dataset.write_text(''.join(json.dumps(row) + '\n' for row in records(2)))
            argv = ['session', '--dataset', str(dataset), '--run-dir', str(root / 'run'),
                    '--agent-ports', '12001', '12002', '--gpu-devices', '3,7']
            with patch('sys.argv', argv), patch(
                    'orchestrator.simulated_web.session.http.client.HTTPConnection',
                    side_effect=[connection_for(installed()), connection_for(installed())]), patch(
                    'orchestrator.simulated_web.session.run_session', return_value=[]) as run:
                self.assertEqual(main(), 0)
            settings = run.call_args.args[8]
            self.assertEqual(settings['server_mode'], 'per-agent')
            self.assertEqual(settings['cuda_visible_devices'], '3,7')
            self.assertEqual(settings['agent_endpoints']['agent-1']['port'], 12001)
            self.assertEqual(settings['agent_endpoints']['agent-2']['cuda_visible_devices'], '7')
            self.assertEqual(settings['agent_endpoints']['agent-2']['model_digest'], 'sha256:same')


class GpuConfigurationTests(unittest.TestCase):
    def test_preserves_slurm_indices_and_full_uuids_in_order(self):
        self.assertEqual(gpu_configuration('3,7', 2), ('per-agent', ['3', '7']))
        self.assertEqual(gpu_configuration('7,3', 2, 'per-agent'), ('per-agent', ['7', '3']))
        uuids = ['GPU-8932f937-d72c-4106-c12f-20bd9faed9f6',
                 'GPU-a132f937-d72c-4106-c12f-20bd9faed9f6']
        self.assertEqual(gpu_configuration(','.join(uuids), 2), ('per-agent', uuids))
        self.assertEqual(gpu_configuration('7', 2), ('shared', ['7']))
        self.assertEqual(gpu_configuration('7', 1), ('shared', ['7']))
        self.assertEqual(gpu_configuration('3,7', 2, 'shared'), ('shared', ['3', '7']))

    def test_rejects_bad_gpu_count_and_ambiguous_mapping(self):
        for devices, agents, mode in [('', 2, 'auto'), ('-1', 2, 'auto'), ('0,0', 2, 'auto'),
                ('0,', 2, 'auto'), ('0, 1', 2, 'auto'), ('0,1,2', 2, 'auto'),
                ('7', 2, 'per-agent'), ('3,7', 1, 'per-agent'), ('3', 1, 'unknown'),
                ('0,GPU-8932f937-d72c-4106-c12f-20bd9faed9f6', 2, 'per-agent')]:
            with self.subTest(devices=devices, agents=agents, mode=mode), self.assertRaises(ValueError):
                gpu_configuration(devices, agents, mode)


if __name__ == '__main__':
    unittest.main()
