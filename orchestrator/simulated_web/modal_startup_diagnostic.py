"""One authorized startup diagnostic, at most600s; no dataset or weight download."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import modal

from orchestrator.simulated_web.modal_timed import image, models, runs, PROFILE
from orchestrator.simulated_web.timed import TimedPolicy
from orchestrator.simulated_web.timed_transport import OwnedOllama

app = modal.App('germanwiki-startup-diagnostic')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def observe(folder, stop):
    offset = 0
    pending = b''
    with (folder / 'observations.jsonl').open('x') as output, (folder / 'ollama-timestamped.jsonl').open('x') as timestamped:
        while not stop.is_set():
            row = {'timestamp': stamp(), 'monotonic': time.monotonic()}
            for name in ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective'):
                path = Path('/sys/fs/cgroup') / name
                row[name] = path.read_text() if path.exists() else None
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                row['gpu'] = gpu.stdout.strip()
                row['gpu_error'] = gpu.stderr.strip()
            except Exception as error:
                row['gpu_error'] = str(error)
            output.write(json.dumps(row) + '\n')
            output.flush()
            source = folder / 'ollama.log'
            if source.exists():
                with source.open('rb') as log:
                    log.seek(offset)
                    data = log.read()
                    offset = log.tell()
                lines = (pending + data).split(b'\n')
                pending = lines.pop()
                for line in lines:
                    timestamped.write(json.dumps({'observed_at': stamp(), 'line': line.decode(errors='replace')}) + '\n')
                timestamped.flush()
            stop.wait(1)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768, timeout=600, retries=0, max_containers=1,
              volumes={'/models': models, '/runs': runs})
def diagnose(run_id):
    folder = Path('/runs') / run_id
    folder.mkdir(exist_ok=False)
    policy = TimedPolicy(context_length=131072)
    client = OwnedOllama(policy, '/models', folder / 'ollama.log')
    stop = threading.Event()
    observer = threading.Thread(target=observe, args=(folder, stop))
    result = {'started_at': stamp(), 'gpu': 'L40S', 'requested_cpu': 4, 'context':131072,
              'scope': 'cold readiness plus2neutral32token requests; no downloads', 'requests': []}
    observer.start()
    started = time.monotonic()
    try:
        print('Diagnostic: checking cached identity', flush=True)
        result['model'] = client.inspect_model(timeout=30)
        print('Diagnostic: cold readiness,300s maximum', flush=True)
        before = time.monotonic()
        result['cold_readiness'] = client.ensure_ready(timeout=300)
        result['cold_readiness']['wall_seconds'] = time.monotonic() - before
        for index in range(2):
            print(f'Diagnostic: warm request{index + 1}/2,60s maximum', flush=True)
            before = time.monotonic()
            try:
                reply = client('agent-1', [{'role':'user', 'content':'Write the integers from1to10, separated by spaces.'}], 60, num_predict=32, final_only=True)
                result['requests'].append({'index':index, 'wall_seconds':time.monotonic()-before,
                                           'metadata':reply.metadata, 'message':reply.message})
            except Exception as error:
                result['requests'].append({'index':index, 'wall_seconds':time.monotonic()-before, 'error':str(error)})
                raise
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            client.close()
        finally:
            stop.set()
            observer.join(timeout=5)
            result.update(finished_at=stamp(), wall_seconds=time.monotonic()-started, transport_events=client.events)
            (folder / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
            runs.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError(f'Require MODAL_PROFILE={PROFILE}')
    run_id = 'startup-diagnostic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'Artifact directory: {run_id}', flush=True)
    with modal.enable_output(), app.run():
        diagnose.remote(run_id)
