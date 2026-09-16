"""Local prerequisite/request-bound checks; no Modal hydration or model execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web.modal_cancellation_diagnostic import LimitedClient, replay_input
from orchestrator.simulated_web.timed import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama


class DiagnosticTests(unittest.TestCase):
    def test_replay_preserves_exact_history_including_thinking_and_tools(self):
        suffix = '\nNew phase: prior final-only instructions have ended. This phase has 20 seconds of elapsed time; early completion is allowed.'
        messages = [{'role': 'assistant', 'thinking': 'retained', 'tool_calls': [{'id': 'x'}]},
                    {'role': 'user', 'content': 'Exact question?' + suffix}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'phase.jsonl'
            path.write_text(json.dumps({'event': 'initial', 'messages': messages}) + '\n')
            saved, prompt = replay_input(path)
            self.assertEqual(saved, messages)
            self.assertEqual(prompt + suffix, messages[-1]['content'])
            path.write_text(json.dumps({'event': 'initial', 'messages': [{'role': 'user', 'content': 'wrong suffix'}]}))
            with self.assertRaises(ValueError):
                replay_input(path)

    def test_total_request_limit_includes_internal_readiness_calls(self):
        with patch('orchestrator.simulated_web.timed_transport.sys.platform', 'linux'), patch('orchestrator.simulated_web.timed_transport.shutil.which', return_value='/fake/ollama'):
            client = LimitedClient(TimedPolicy(), '/fake/cache', '/fake/log')
        with patch.object(OwnedOllama, '__call__', return_value=None) as request:
            for _ in range(16):
                client('agent-1', [], 1, num_predict=1, _readiness=True)
            with self.assertRaisesRegex(RuntimeError, '16request limit'):
                client('agent-1', [], 1, num_predict=1)
        self.assertEqual(request.call_count, 16)


if __name__ == '__main__':
    unittest.main()
