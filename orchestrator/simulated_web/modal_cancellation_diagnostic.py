"""One cached L40S cancellation/continuation check; hard600s, no retries/downloads."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time

import modal

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.modal_startup_diagnostic import observe, stamp
from orchestrator.simulated_web.modal_timed import image, models, runs, PROFILE
from orchestrator.simulated_web.timed import TimedPolicy, run_phase_with_readiness
from orchestrator.simulated_web.timed_transport import OwnedOllama

app = modal.App('germanwiki-cancellation-diagnostic')
SOURCE_RUN = 'germanwiki-timed-mlb-20-004'


def replay_input(path):
    with path.open() as stream:
        first = json.loads(stream.readline())
    history = first.get('messages')
    suffix = '\nNew phase: prior final-only instructions have ended. This phase has 20 seconds of elapsed time; early completion is allowed.'
    if (first.get('event') != 'initial' or not isinstance(history, list) or not history
            or history[-1].get('role') != 'user' or not history[-1].get('content', '').endswith(suffix)):
        raise ValueError('Cannot reconstruct exact saved answer phase input')
    prompt = history[-1]['content'][:-len(suffix)]
    return history, prompt


def prepare_browser(source, folder):
    """Initialize fresh schema, then restore a read-only snapshot into that connection."""
    pages = json.loads((source / 'pages.json').read_text())
    settings = json.loads((source / 'settings.json').read_text())
    browser = Browser(pages, folder / 'wiki.sqlite3', editable_sources=settings['editable_source'], editable_title_marker=False)
    try:
        snapshot = sqlite3.connect((source / 'wiki.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        try:
            snapshot.backup(browser.db)
        finally:
            snapshot.close()
    except Exception:
        browser.close()
        raise
    return browser


class LimitedClient(OwnedOllama):
    """Bound all inference calls, including neutral readiness, without changing prompts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_count = 0

    def __call__(self, *args, **kwargs):
        self.request_count += 1
        if self.request_count > 16:
            raise RuntimeError('Diagnostic administrative16request limit reached')
        return super().__call__(*args, **kwargs)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768, timeout=600, retries=0, max_containers=1,
              volumes={'/models': models, '/runs': runs})
def diagnose(run_id, expected_history_sha256):
    if re.fullmatch(r'cancellation-check-[0-9TZ]+', run_id) is None:
        raise ValueError('Invalid diagnostic ID')
    source = Path('/runs') / SOURCE_RUN
    saved, replay_prompt = replay_input(source / 'phase-21.jsonl')
    source_hash = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
    if source_hash != expected_history_sha256:
        raise ValueError('Saved replay history differs from locally validated input')
    json.loads((source / 'pages.json').read_text())
    json.loads((source / 'settings.json').read_text())
    if not (source / 'wiki.sqlite3').is_file():
        raise ValueError('Missing source wiki snapshot')
    if not list(Path('/models/manifests').rglob('q4_K_M')) and not any(Path('/models/manifests').rglob('*q4*')):
        # Identity inspection below is authoritative; this merely catches empty caches.
        if not Path('/models/manifests').is_dir():
            raise ValueError('Cached model manifests missing; downloads prohibited')
    folder = Path('/runs') / run_id
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + 570  # Reserve30s for stop, artifacts and volume commit.
    policy = TimedPolicy(context_length=131072)
    result = {'started_at': stamp(), 'scope': 'one prefill replay, one generation interruption, two following phases',
              'source_run': SOURCE_RUN, 'source_phase': 'phase-21.jsonl', 'input_sha256': source_hash,
              'policy': asdict(policy), 'gpu': 'L40S', 'requests_max': 16, 'status': 'started'}
    (folder / 'input.json').write_text(json.dumps(saved) + '\n')
    (folder / 'source.py').write_text(Path(__file__).read_text())
    client = LimitedClient(policy, '/models', folder / 'ollama.log')
    stop = threading.Event()
    observer = threading.Thread(target=observe, args=(folder, stop))
    observer.start()
    browser = None
    rows, transitions = [], []
    try:
        browser = prepare_browser(source, folder)
        print(f'{run_id}: checking cached model; no downloads', flush=True)
        result['model'] = client.inspect_model(timeout=min(30, deadline-time.monotonic()))
        histories = [saved[:-1], [{'role': 'system', 'content': 'This is a neutral transport diagnostic. Use no tools.'}]]
        cases = [('prefill-replay', 'answer', 20, replay_prompt, histories[0]),
                 ('after-prefill', 'reflection', 20, 'Reply with only OK.', histories[0]),
                 ('generation-interrupt', 'reflection', 5, 'Write the integers from1to100000, one integer per line. Continue until you have written all of them.', histories[1]),
                 ('after-generation', 'reflection', 20, 'Stop counting. Reply with only OK.', histories[1])]
        for index, (label, phase, seconds, prompt, history) in enumerate(cases):
            remaining = deadline - time.monotonic()
            # Reserve the complete phase window and bounded cleanup; never extend it.
            readiness_timeout = min(300 if index == 0 else 120, remaining-seconds-7)
            if readiness_timeout <= 0:
                raise RuntimeError('Diagnostic total-time bound exhausted before next stage')
            row = run_phase_with_readiness(browser, client, history, prompt, phase, seconds, policy,
                                           folder, index, None, transitions, rows, readiness_timeout, label)
            if index == 0:
                with (folder / 'phase-00.jsonl').open() as stream:
                    initial = json.loads(stream.readline())['messages']
                result['replay_exact'] = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest() == source_hash
                if not result['replay_exact']:
                    raise RuntimeError('Diagnostic did not reproduce exact saved input')
            result[label] = {'status': row['status'], 'elapsed_seconds': row['elapsed_seconds'],
                             'final_attempted': row['final_attempted'], 'limits_reached': row['limits_reached']}
            (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        result['status'] = 'completed_check'
        result['following_phases_complete'] = all(rows[i]['status'] == 'complete' for i in (1, 3))
        result['interruption_coverage'] = 'Inspect phase partials and server prefill/generation logs; natural completion is inconclusive, no retry.'
    except Exception as error:
        result.update(status='failed_check', error=f'{type(error).__name__}: {error}')
        print(result['error'], flush=True)
    finally:
        try:
            client.close()
        finally:
            if browser is not None:
                browser.close()
            stop.set()
            observer.join(timeout=5)
            result.update(finished_at=stamp(), wall_seconds=time.monotonic()-started,
                          requests=client.request_count, transport_events=client.events)
            (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            runs.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError(f'Require MODAL_PROFILE={PROFILE}')
    saved, _ = replay_input(Path('/private/tmp/germanwiki-004-phase21.jsonl'))
    expected = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
    run_id = 'cancellation-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'Artifact directory: {run_id}', flush=True)
    with modal.enable_output(), app.run():
        diagnose.remote(run_id, expected)
