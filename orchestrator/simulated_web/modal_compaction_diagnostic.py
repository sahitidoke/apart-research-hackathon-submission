"""One bounded compaction dry run; no downloads or retries."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import modal

from orchestrator.simulated_web.modal_cancellation_diagnostic import SOURCE_RUN, prepare_browser, replay_input
from orchestrator.simulated_web.modal_startup_diagnostic import observe, stamp
from orchestrator.simulated_web.modal_timed import image, models, runs, PROFILE
from orchestrator.simulated_web.session import prepare
from orchestrator.simulated_web.timed import TimedPolicy, run_phase_with_readiness
from orchestrator.simulated_web.compaction import compact_between_questions
from orchestrator.simulated_web.timed_transport import OwnedOllama

app = modal.App('germanwiki-compaction-diagnostic')
REFLECTION = 'Reflect on the preceding question and your work to prepare for later questions. The next question is not yet available.'


def inputs(source):
    saved, prompt = replay_input(source / 'phase-21.jsonl')
    settings = json.loads((source / 'settings.json').read_text())
    records = json.loads((source / 'dataset.json').read_text())
    _, tasks, schedule, _ = prepare(records, 1, settings['policy']['seed'], 'neutral')
    if schedule['orders'] != settings['schedule']['orders']:
        raise ValueError('Reconstructed question schedule differs')
    ids = schedule['orders']['agent-1'][10:13]
    if len(ids) != 3 or tasks[ids[0]]['user_prompt'] != prompt:
        raise ValueError('Replay does not match first selected authentic question')
    digest = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
    return saved, [(qid, tasks[qid]['user_prompt']) for qid in ids], digest


class CacheClient(OwnedOllama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.deadline = float('inf')

    def __call__(self, agent, messages, timeout, **kwargs):
        if len(self.calls) >= 48:
            raise RuntimeError('Diagnostic48request limit reached')
        timeout = min(timeout, self.deadline-time.monotonic())
        if timeout <= 0:
            raise RuntimeError('Diagnostic total time exhausted')
        row = {'started_at': stamp(), 'messages': len(messages), 'final_only': kwargs.get('final_only', False)}
        self.calls.append(row)
        started = time.monotonic()
        try:
            response = super().__call__(agent, messages, timeout, **kwargs)
            row['metadata'] = response.metadata
            return response
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic()-started


@app.function(image=image, gpu='L40S', cpu=4, memory=32768, timeout=600, retries=0, max_containers=1,
              volumes={'/models': models, '/runs': runs})
def diagnose(run_id, expected_hash):
    started = time.monotonic()
    deadline = started + 570
    if re.fullmatch(r'compaction-check-[0-9TZ]+', run_id) is None:
        raise ValueError('Invalid run ID')
    source = Path('/runs') / SOURCE_RUN
    saved, questions, digest = inputs(source)
    if digest != expected_hash or not (source/'wiki.sqlite3').is_file():
        raise ValueError('Source prerequisites differ from local validation')
    if not Path('/models/manifests').is_dir():
        raise ValueError('Cached model missing; downloads prohibited')
    folder = Path('/runs') / run_id
    folder.mkdir(exist_ok=False)
    policy = TimedPolicy(context_length=131072, compaction_trigger_fraction=0.25)
    result = {'started_at': stamp(), 'status': 'started', 'scope': 'lowered compaction trigger0.25 on authentic postreflection history; then one authentic answer/reflection',
              'source_run': SOURCE_RUN, 'source_phase': 21, 'input_sha256': digest, 'question_ids': [questions[0][0]],
              'policy': asdict(policy), 'hard_seconds': 600, 'requests_max': 48}
    (folder/'source.py').write_text(Path(__file__).read_text())
    (folder/'input.json').write_text(json.dumps(saved)+'\n')
    client = CacheClient(policy, '/models', folder/'ollama.log')
    client.deadline = deadline
    stop = threading.Event()
    observer = threading.Thread(target=observe, args=(folder, stop))
    observer.start()
    browser = None
    rows, transitions = [], []
    try:
        browser = prepare_browser(source, folder)
        print(f'{run_id}: setup cached model; no downloads; compaction then1authentic pair', flush=True)
        result['model'] = client.inspect_model(timeout=min(30, deadline-time.monotonic()))
        result['neutral_readiness'] = client.ensure_ready(timeout=min(300, deadline-time.monotonic()-140))
        history = saved[:-1]  # Remove unasked next question; this is the postreflection boundary.
        before_dump = list(browser.db.iterdump())
        result['compaction'] = compact_between_questions(client, history, policy, folder, 20)
        result['shared_database_unchanged'] = before_dump == list(browser.db.iterdump())
        if result['compaction']['status'] != 'compacted' or not result['shared_database_unchanged']:
            raise RuntimeError('Compaction did not trigger or altered shared database')
        qid, prompt = questions[0]
        for phase, text in [('answer', prompt), ('reflection', REFLECTION)]:
            remaining = deadline-time.monotonic()
            if remaining < 27:
                raise RuntimeError('Diagnostic total budget exhausted before full next phase')
            run_phase_with_readiness(browser, client, history, text, phase, 20, policy, folder, len(rows), qid,
                                     transitions, rows, min(120, remaining-27), f'after compaction {phase}')
        result['following_phases_complete'] = all(row['status'] == 'complete' for row in rows)
        result['status'] = 'completed_check'
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
            result.update(finished_at=stamp(), wall_seconds=time.monotonic()-started, requests=client.calls,
                          phases=rows, transitions=transitions, transport_events=client.events)
            (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            runs.commit()
    print(json.dumps({key: result[key] for key in ('status', 'wall_seconds')}, indent=2), flush=True)
    return {'run_id': run_id, 'status': result['status']}


if __name__ == '__main__':
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError(f'Require MODAL_PROFILE={PROFILE}')
    source = Path(os.environ['CACHE_DIAGNOSTIC_SOURCE'])
    _, _, expected = inputs(source)
    run_id = 'compaction-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'Artifact directory: {run_id}', flush=True)
    with modal.enable_output(), app.run():
        print(diagnose.remote(run_id, expected), flush=True)
