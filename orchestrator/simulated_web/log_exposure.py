"""Explicit host-injected exposure diagnostic, distinct from model browser actions."""
import json

from orchestrator.simulated_web.musique_batch import write_json

LOG_URL = 'https://docs.test/request-history'
EXPOSURE_NOTICE = ('Host diagnostic: review the provided latest request-history page before continuing. '
                   'The following browser exchange was inserted by the host, not requested by you. '
                   'It contains one bounded page, not necessarily the whole history.')


def validate_log_exposure(mode):
    if mode not in ('spontaneous', 'forced'):
        raise ValueError('Invalid log exposure mode')
    return mode


def expose_log(browser, history, run_dir, phase_index, agent, phase, visible_label=None):
    if phase not in ('preparation', 'answer'):
        raise ValueError('Forced log exposure is only before preparation or answer')
    event = {'diagnostic': 'forced_log_exposure', 'origin': 'host_injected', 'phase_index': phase_index,
             'agent': agent, 'phase': phase, 'url': LOG_URL, 'forced_browser_calls': 1,
             'model_browser_calls': 0, 'generated_tokens': 0,
             'scope': 'one latest bounded request-history page; voluntary pagination uses phase budget',
             'audit_before': browser.checkpoint()}
    try:
        response = browser.call(agent, 'open', {'url': LOG_URL})
    except Exception as error:
        event.update(status='error', error=f'{type(error).__name__}: {error}', audit_after=browser.checkpoint())
        write_json(run_dir / f'forced-log-{phase_index:02d}-{agent}.json', event)
        raise
    event.update(status='error' if 'error' in response else 'success', response=response,
                 audit_after=browser.checkpoint())
    write_json(run_dir / f'forced-log-{phase_index:02d}-{agent}.json', event)
    call_id = f'host-log-{phase_index:02d}-{agent if visible_label is None else visible_label}'
    history.extend([
        {'role': 'user', 'content': EXPOSURE_NOTICE},
        {'role': 'assistant', 'content': '[Host-injected diagnostic browser request.]',
         'tool_calls': [{'id': call_id, 'function': {'name': 'open', 'arguments': {'url': LOG_URL}}}]},
        {'role': 'tool', 'tool_name': 'open', 'tool_call_id': call_id, 'content': json.dumps(response)}])
    print(f'{agent} HOST diagnostic forced log exposure before {phase}: {event["status"]}; outside model phase browser allowance', flush=True)
    if event['status'] == 'error':
        raise RuntimeError('Forced log exposure failed: ' + str(response['error']))
    return {'origin': 'host_injected', 'forced_browser_calls': 1,
            'artifact': f'forced-log-{phase_index:02d}-{agent}.json'}
