"""Mock-only generated-token phase allowances and legacy policy compatibility."""
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.timed import run_phase, session_prompt, run_timed_session, resume_timed_session, PREPARATION_PROMPT, ANSWER_PROMPT, REFLECTION_PROMPT
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint
from orchestrator.simulated_web.test_session import records
from orchestrator.simulated_web.test_timed_resume import ResumeClient, MODEL
from orchestrator.simulated_web import test_modal_timed
from orchestrator.simulated_web.test_modal_timed import modal_timed
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.test_timed import Clock


def response(count, *, calls=0, reason='stop', content='answer', thinking='reasoning'):
    return ModelResponse({'content': content, 'thinking': thinking,
                          'tool_calls': [{'function': {'name': 'open', 'arguments': {'url': 'https://docs.test/'}}}
                                         for _ in range(calls)]},
                         {'eval_count': count, 'done_reason': reason, 'prompt_eval_count': 10})


def execute(responses, phase='answer', clock=None, **overrides):
    policy = TimedPolicy(budget_mode='generated_tokens', compaction_enabled=False, **overrides)
    requests = []
    browser = Mock()
    browser.call.return_value = {'text': 'evidence'}
    history = []
    iterator = iter(responses)
    def client(agent, messages, timeout, **kwargs):
        requests.append((agent, timeout, kwargs))
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value
    with tempfile.TemporaryDirectory() as folder:
        result = run_phase(browser, client, history, 'Phase: 600 seconds.', phase, 600, policy,
                           Path(folder) / 'phase.jsonl', clock=clock or Clock(), agent='agent-2')
    return result, requests, browser, history


def test_reasoning_tools_final_accounting_and_reserve():
    row, requests, browser, history = execute([response(1000, calls=1), response(792, calls=1), response(12, thinking='')])
    assert [r[2]['num_predict'] for r in requests] == [1792, 792, 256]
    assert [r[2]['final_only'] for r in requests] == [False, False, True]
    assert all(r[0] == 'agent-2' for r in requests)
    assert browser.call.call_args.args[0] == 'agent-2'
    assert row['generated_tokens_observed'] == 1804
    assert row['generated_tokens_remaining'] == 244
    assert row['status'] == 'complete' and row['token_accounting_complete']
    assert row['final_available_seconds'] == 600  # Token reserve does not impose a five-second window.
    assert 'seconds.' not in history[0]['content']


@pytest.mark.parametrize('phase', ['preparation', 'reflection'])
def test_truncated_calls_never_execute_and_normal_allowance_exhaustion(phase):
    row, requests, browser, _ = execute([response(4096, calls=1, reason='length'),
                                         response(4096, calls=1, reason='length')], phase,
                                        max_output_tokens=4096)
    assert row['status'] == 'budget_exhausted'
    assert row['generated_tokens_observed'] == 8192
    assert row['generated_tokens_remaining'] == 0
    assert browser.call.call_count == 0
    assert len(requests) == 2


def test_final_truncated_tools_not_executed_or_accepted():
    row, requests, browser, _ = execute([response(1792, calls=1, reason='length'),
                                         response(256, calls=1, reason='length')])
    assert row['status'] == 'invalid_final_response' and row['answer'] == ''
    assert row['generated_tokens_observed'] == 2048
    assert browser.call.call_count == 0
    assert len(requests) == 2


def test_browser_ceiling_with_multiple_calls_finalizes_early():
    row, requests, browser, history = execute([response(20, calls=3), response(5, thinking='')], answer_browser_calls=2)
    assert browser.call.call_count == 2
    assert requests[-1][2]['final_only']
    assert requests[-1][2]['num_predict'] == 2028
    assert 'browser_actions' in row['limits_reached']
    assert len(history[1]['tool_calls']) == 2


def test_no_rollover_and_early_completion():
    first, *_ = execute([response(2)], 'preparation')
    second, requests, *_ = execute([response(3)])
    assert first['status'] == second['status'] == 'complete'
    assert requests[0][2]['num_predict'] == 1792
    assert second['generated_tokens_remaining'] == 2045


@pytest.mark.parametrize('count', [None, -1, True])
def test_missing_or_invalid_native_counts_fail_visibly(count):
    row, requests, browser, _ = execute([response(count, calls=1)])
    assert row['status'] == 'error'
    assert not row['token_accounting_complete']
    assert 'native_token_count_missing' in row['limits_reached']
    assert browser.call.call_count == 0


def test_timeout_does_not_guess_native_count_or_continue():
    row, requests, *_ = execute([TimeoutError('stopped')])
    assert row['status'] == 'error' and row['safety_timeout_hit']
    assert not row['token_accounting_complete'] and len(requests) == 1


def test_late_complete_response_is_charged_before_discard():
    clock = Clock()
    class Late:
        def __iter__(self):
            clock.now = 601
            yield response(30, calls=1)
    row, requests, browser, _ = execute(Late(), 'preparation', clock=clock)
    assert row['status'] == 'deadline_reached' and row['safety_timeout_hit']
    assert row['generated_tokens_observed'] == 30
    assert browser.call.call_count == 0


def test_backend_cap_violation_fails_without_executing_tools():
    row, _, browser, _ = execute([response(1793, calls=1)])
    assert row['status'] == 'error' and row['generated_tokens_observed'] == 1793
    assert 'native_output_cap_violation' in row['limits_reached']
    assert browser.call.call_count == 0


def test_legacy_policy_and_serialized_token_policy():
    policy = TimedPolicy()
    assert policy.budget_mode == 'elapsed_time'
    assert (policy.preparation_seconds, policy.answer_seconds, policy.reflection_seconds) == (90, 20, 20)
    token = TimedPolicy(budget_mode='generated_tokens')
    assert TimedPolicy(**json.loads(json.dumps(asdict(token)))) == token
    assert 'time window' not in session_prompt(token)
    assert 'answer deadline' not in session_prompt(token)
    assert 'time window' in session_prompt(policy)
    for field, value in [('budget_mode', 'bogus'), ('answer_generated_tokens', 256),
                         ('final_reserve_tokens', 0), ('reflection_browser_calls', True)]:
        with pytest.raises(ValueError):
            TimedPolicy(**{field: value})


def test_checkpoint_resume_token_policy_and_old_missing_fields():
    with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
        root = Path(folder)
        policy = TimedPolicy(budget_mode='generated_tokens', compaction_enabled=False)
        run_timed_session(root / 'parent', records(10), 'topic', ResumeClient(), policy, provenance={'model': MODEL})
        checkpoint = root / 'parent/checkpoints/questions-005'
        resume_timed_session(root / 'child', checkpoint, ResumeClient())
        resumed = json.loads((root / 'child/results.json').read_text())
        assert all(row['generated_token_allowance'] == (2048 if row['phase'] == 'answer' else 8192)
                   for row in resumed if row.get('budget_mode') == 'generated_tokens')
        saved = json.loads((checkpoint / 'settings.json').read_text())
        for key in ('budget_mode', 'preparation_generated_tokens', 'answer_generated_tokens',
                    'reflection_generated_tokens', 'final_reserve_tokens', 'preparation_browser_calls',
                    'answer_browser_calls', 'reflection_browser_calls'):
            del saved['policy'][key]
        (checkpoint / 'settings.json').write_text(json.dumps(saved))
        manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
        manifest['files_sha256']['settings.json'] = hashlib.sha256((checkpoint / 'settings.json').read_bytes()).hexdigest()
        (checkpoint / 'checkpoint.json').write_text(json.dumps(manifest))
        loaded = load_checkpoint(checkpoint)
        assert TimedPolicy(**loaded.data['settings.json']['policy']).budget_mode == 'elapsed_time'
        loaded.browser.close()


@pytest.mark.skipif(modal_timed is None, reason='Optional Modal SDK not installed')
def test_cli_token_policy_and_no_resume_overrides():
    with tempfile.TemporaryDirectory() as folder:
        args = test_modal_timed.ModalValidationTests().inputs(folder)
        config = Path(folder) / 'modal.toml'
        config.write_text('[research-profile]\n')
        output = io.StringIO()
        with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(config)}), \
                patch.object(modal_timed.app, 'run') as run, redirect_stdout(output):
            modal_timed.main(args + ['--budget-mode', 'generated_tokens', '--reflection-generated-tokens', '8192'])
        run.assert_not_called()
        saved = json.loads(output.getvalue())['policy']
        assert saved['budget_mode'] == 'generated_tokens' and saved['reflection_generated_tokens'] == 8192
        with pytest.raises(ValueError, match='does not accept'):
            modal_timed.main(['--resume-from', 'parent/checkpoints/questions-005', '--run-id', 'child',
                              '--budget-mode', 'generated_tokens', '--validate-only'])


@pytest.mark.parametrize('phase,template', [('preparation', PREPARATION_PROMPT), ('answer', ANSWER_PROMPT),
                                            ('reflection', REFLECTION_PROMPT)])
def test_formatted_float_safety_cap_not_advertised_as_phase_budget(phase, template):
    seconds = 600.0
    text = template.format(seconds=seconds, topic='Topic', question='Question?', collection_url='https://docs.test/')
    history = []
    with tempfile.TemporaryDirectory() as folder:
        run_phase(Mock(), lambda *args, **kwargs: response(1), history, text, phase, seconds,
                  TimedPolicy(budget_mode='generated_tokens'), Path(folder) / 'phase.jsonl', clock=Clock())
    assert 'seconds.' not in history[0]['content']
    assert 'generated tokens;' in history[0]['content']
