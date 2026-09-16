"""Explicit Modal SDK launch; validate local inputs BEFORE creating cloud resources.

Run only with separate experiment authorization. `--validate-only` is read-only.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import time

import modal

from orchestrator.simulated_web.timed import TimedPolicy, run_timed_session, validate_inputs, resume_timed_session, validate_resume_checkpoint
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint, checkpoint_model_digest
from orchestrator.simulated_web.timed_transport import MODEL, OwnedOllama

PROFILE = 'research-profile'
MODEL_VOLUME = 'germanwiki-qwen38-ollama-models'
RUN_VOLUME = 'germanwiki-timed-runs'
IMAGE = 'ollama/ollama:0.33.3'
GPU = 'L40S'
JOB_TIMEOUT_SECONDS = 10800
app = modal.App('germanwiki-timed')
models = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=True)
image = (modal.Image.from_registry(IMAGE, add_python='3.12').entrypoint([])
         .env({'PYTHONPATH': '/root/project'})
         .add_local_dir(Path(__file__).parent, '/root/project/orchestrator/simulated_web',
                        ignore=['__pycache__', '*.pyc']))


@app.function(image=image, gpu=GPU, cpu=4, memory=32768,
              volumes={'/models': models, '/runs': runs}, timeout=JOB_TIMEOUT_SECONDS, max_containers=1, retries=0)
def execute(records, topic, selectors, policy_values, run_id, expected_digest, download_model, resume_from=None):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id) is None:
        raise ValueError('Invalid run ID')
    destination = Path('/runs') / run_id
    if destination.exists():
        raise ValueError('Run ID already exists; preserve previous artifacts')
    if (Path('/runs') / (run_id + '-setup')).exists():
        raise ValueError('Setup directory already exists; use a fresh run ID')
    if resume_from is not None:
        validate_resume_selector(resume_from, run_id)
        if download_model or records is not None or topic is not None or selectors is not None or policy_values:
            raise ValueError('Resume uses checkpoint inputs/policy and cached model only')
        checkpoint_path = Path('/runs') / resume_from
        if not checkpoint_path.resolve().is_relative_to(Path('/runs').resolve()):
            raise ValueError('Checkpoint must stay within the run volume')
        checkpoint = load_checkpoint(checkpoint_path)
        try:
            policy, _ = validate_resume_checkpoint(checkpoint)
            if checkpoint.complete:
                return {'status': 'already_complete', 'checkpoint': resume_from, 'output_created': False}
            digest = checkpoint_model_digest(checkpoint)
            if expected_digest is not None and expected_digest.removeprefix('sha256:') != digest.removeprefix('sha256:'):
                raise ValueError('Requested model digest differs from checkpoint')
            expected_digest = digest
            validate_remote_budget(policy, False, checkpoint.manifest['completed_questions'])
        finally:
            checkpoint.close()
    else:
        policy = TimedPolicy(**policy_values)
        validate_inputs(records, topic, policy, selectors)
        validate_remote_budget(policy, download_model)
    # Useful startup/download diagnostics survive failures, but malformed inputs do
    # not create a run directory. Run setup has begun once this attempt is recorded.
    attempt = Path('/runs') / (run_id + '-setup')
    attempt.mkdir(exist_ok=False)
    setup_started = time.monotonic()
    client = OwnedOllama(policy, '/models', attempt / 'ollama.log', expected_digest=expected_digest, seed=policy.seed)
    try:
        print(f'[{run_id}] setup: GPU={GPU}, context={policy.context_length}, questions={policy.question_count}', flush=True)
        if download_model:
            client._start(time.monotonic() + 30)
            pull_model(client, attempt / 'pull.log')
            models.commit()
            print(f'[{run_id}] model cache committed', flush=True)
        print(f'[{run_id}] inspecting installed model identity', flush=True)
        metadata = client.inspect_model()
        print(f'[{run_id}] model identity verified: {metadata["name"]}, digest={metadata["digest"]}; warmup will finish before the next task phase', flush=True)
        setup_seconds = time.monotonic() - setup_started
        (attempt / 'setup.json').write_text(json.dumps({'status': 'identity_checked', 'elapsed_seconds': setup_seconds,
            'download_requested': download_model, 'gpu_request': GPU, 'model': metadata}, indent=2) + '\n')
        provenance = {'setup_seconds': setup_seconds, 'backend': 'Modal', 'requested_profile': PROFILE, 'model': metadata,
                      'requested_model': MODEL, 'expected_digest': expected_digest,
                      'image': IMAGE, 'modal_version': modal.__version__,
                      'gpu_request': GPU, 'server_parallelism': 1,
                      'kv_cache_type': 'q8_0', 'flash_attention': True,
                      'context_strategy': 'ollama-debug-render-only-and-owned-tokenize; full history; no truncation/shift',
                      'final_strategy': 'stable-tools-and-think-settings; closed-think-assistant-prefill; host rejects tools/thinking',
                      'deadline_strategy': 'stream-disconnect-quiet1.1s-sparse-polls-fresh-single-slot-idle-ack-2s; owned-group-kill fallback',
                      'run_volume': RUN_VOLUME, 'model_volume': MODEL_VOLUME}
        def commit_checkpoint(checkpoint):
            runs.commit()
            print(f'[{run_id}] checkpoint committed to {RUN_VOLUME}: {checkpoint.relative_to(Path("/runs"))}', flush=True)

        if resume_from is None:
            run_timed_session(destination, records, topic, client, policy, selectors, provenance,
                              checkpoint_callback=commit_checkpoint)
        else:
            resume_timed_session(destination, checkpoint_path, client, provenance,
                                 checkpoint_callback=commit_checkpoint)
        return {'status': 'complete', 'volume': RUN_VOLUME, 'path': run_id}
    except BaseException as error:
        print(f'[{run_id}] failed: {type(error).__name__}: {error}', flush=True)
        (attempt / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        try:
            client.close()
        finally:
            (attempt / 'transport-events.json').write_text(json.dumps(client.events, indent=2) + '\n')
            runs.commit()
            print(f'[{run_id}] artifacts committed to {RUN_VOLUME}: {run_id}/ and {run_id}-setup/', flush=True)


def pull_model(client, log_path, timeout=1800, clock=time.monotonic):
    """Keep complete CLI diagnostics and announce bounded download progress."""
    started = clock()
    print(f'[setup] pulling {MODEL}; full progress log: {log_path}', flush=True)
    with log_path.open('xb') as log:
        process = subprocess.Popen(['ollama', 'pull', MODEL],
            env={**os.environ, 'OLLAMA_HOST': f'127.0.0.1:{client.port}'},
            stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                remaining = timeout - (clock() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                try:
                    code = process.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    print(f'[setup] model pull running: {clock() - started:.0f}s elapsed', flush=True)
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
                print(f'[setup] model pull complete: {clock() - started:.1f}s elapsed', flush=True)
                return
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def validate_remote_budget(policy, download_model, completed_questions=0):
    setup = 1830 if download_model else 30
    remaining = policy.question_count - completed_questions
    readiness = policy.initial_readiness_timeout_seconds + 2 * remaining * policy.readiness_timeout_seconds
    phase_time = (0 if completed_questions else policy.preparation_seconds) + remaining * (policy.answer_seconds + policy.reflection_seconds)
    compaction = max(0, remaining - 1) * policy.compaction_timeout_seconds if policy.memory_mode == 'in_context' and policy.compaction_enabled else 0
    if setup + readiness + phase_time + compaction > JOB_TIMEOUT_SECONDS - 300:
        raise ValueError('Requested phase windows, readiness bounds and setup exceed the bounded three-hour Modal job (300s safety margin)')


def validate_resume_selector(value, run_id):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}/checkpoints/questions-[0-9]{3}', value) is None:
        raise ValueError('Resume selector must be RUN/checkpoints/questions-NNN')
    if value.split('/')[0] == run_id:
        raise ValueError('Resume requires a fresh run ID, different from the parent')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--topic-file', type=Path)
    parser.add_argument('--editable-sources', type=Path,
                        help='JSON list of exact {title,text_sha256} source paragraph selectors')
    parser.add_argument('--resume-from', help='Existing volume checkpoint RUN/checkpoints/questions-NNN; inputs and policy come from it')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-model-digest',
                        help='Optional previously recorded model digest; omit to record the first-run identity')
    parser.add_argument('--budget-mode', choices=('elapsed_time', 'generated_tokens'), default='elapsed_time')
    parser.add_argument('--preparation-generated-tokens', type=int, default=8192)
    parser.add_argument('--answer-generated-tokens', type=int, default=2048)
    parser.add_argument('--reflection-generated-tokens', type=int, default=8192)
    parser.add_argument('--final-reserve-tokens', type=int, default=256)
    parser.add_argument('--preparation-browser-calls', type=int, default=40)
    parser.add_argument('--answer-browser-calls', type=int, default=10)
    parser.add_argument('--reflection-browser-calls', type=int, default=80)
    parser.add_argument('--question-count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preparation-seconds', type=float, default=90)
    parser.add_argument('--answer-seconds', type=float, default=20)
    parser.add_argument('--reflection-seconds', type=float, default=20)
    parser.add_argument('--final-reserve-seconds', type=float, default=5)
    parser.add_argument('--initial-readiness-timeout-seconds', type=float, default=300)
    parser.add_argument('--readiness-timeout-seconds', type=float, default=120)
    parser.add_argument('--context-length', type=int, default=65536)
    parser.add_argument('--max-output-tokens', type=int, default=32768)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--request-history-mode', choices=('disabled', 'shared', 'isolated'), default='disabled')
    parser.add_argument('--browser-retention', choices=('full', 'question_boundary'), default='full')
    parser.add_argument('--memory-mode', choices=('in_context', 'private_scratchpad'), default='in_context')
    parser.add_argument('--scratchpad-tokens', type=int, default=4096)
    parser.add_argument('--compaction-enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compaction-trigger-fraction', type=float, default=0.75)
    parser.add_argument('--compaction-retained-tokens', type=int, default=16384)
    parser.add_argument('--compaction-output-tokens', type=int, default=4096)
    parser.add_argument('--compaction-timeout-seconds', type=float, default=180)
    parser.add_argument('--download-model', action='store_true', help='Explicitly allow an uncached model pull during setup')
    parser.add_argument('--validate-only', action='store_true')
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.resume_from is not None:
        validate_resume_selector(args.resume_from, args.run_id)
        policy_flags = {'--' + key.replace('_', '-') for key in TimedPolicy.__dataclass_fields__}
        policy_flags.add('--no-compaction-enabled')
        if (args.dataset is not None or args.topic_file is not None or args.editable_sources is not None
                or args.download_model or any(arg.split('=', 1)[0] in policy_flags for arg in argv)):
            raise ValueError('Resume does not accept dataset/topic/selectors, policy overrides, or model download')
        records = topic = selectors = policy = None
    else:
        if args.dataset is None or args.topic_file is None or args.editable_sources is None:
            raise ValueError('Fresh runs require --dataset, --topic-file, and --editable-sources')
        records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
        topic = args.topic_file.read_text().strip()
        selectors = json.loads(args.editable_sources.read_text())
        if not isinstance(selectors, list) or len(selectors) != 5:
            raise ValueError('Current pilot requires exactly five editable canonical source paragraphs')
        policy = TimedPolicy(**{k: getattr(args, k) for k in TimedPolicy.__dataclass_fields__})
        validate_inputs(records, topic, policy, selectors)
        validate_remote_budget(policy, args.download_model)
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run_id) is None:
        raise ValueError('Run ID must be 1..80 letters, digits, underscores or hyphens, starting alphanumeric')
    if args.expected_model_digest is not None and re.fullmatch(r'(sha256:)?[0-9a-f]{64}', args.expected_model_digest) is None:
        raise ValueError('Expected model digest must be a SHA256 digest')
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError(f'Set MODAL_PROFILE={PROFILE}; this launcher never switches the default profile')
    config_path = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    profiles = tomllib.loads(config_path.read_text()) if config_path.is_file() else {}
    if PROFILE not in profiles:
        raise ValueError(f'Modal profile {PROFILE} is not configured at {config_path}; configure it before launching')
    if args.validate_only and args.resume_from is not None:
        print(json.dumps({'status': 'validated_resume_selector_only_no_cloud_actions',
                          'resume_from': args.resume_from, 'run_id': args.run_id,
                          'checkpoint_contents_validated': False,
                          'note': 'Remote checkpoint hashes, policy, progress and model identity are checked on the worker before setup.'}, indent=2))
        return
    if args.validate_only:
        print(json.dumps({'status': 'validated_no_cloud_actions', 'profile': PROFILE, 'policy': asdict(policy),
                          'questions': [r['id'] for r in records], 'editable_sources': selectors}, indent=2))
        return
    # First cloud action; all locally checkable inputs have now passed.
    with modal.enable_output(), app.run():
        print(json.dumps(execute.remote(records, topic, selectors, asdict(policy) if policy else {}, args.run_id,
                                        args.expected_model_digest, args.download_model, args.resume_from), indent=2))


if __name__ == '__main__':
    main()
