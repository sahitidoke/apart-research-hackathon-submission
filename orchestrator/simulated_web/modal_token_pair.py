"""Explicit six-hour maximum, single-L40S two-agent pilot; default is validation only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time
import tomllib

import modal

from orchestrator.simulated_web.modal_timed import models, runs, image, PROFILE, RUN_VOLUME, MODEL_VOLUME, IMAGE
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.answer_format import format_settings
from orchestrator.simulated_web.source_access import access_plan
from orchestrator.simulated_web.source_discovery import discovery_plan
from orchestrator.simulated_web.log_exposure import validate_log_exposure
from orchestrator.simulated_web.timed_transport import MODEL, OwnedOllama
from orchestrator.simulated_web.token_pair import DIGEST, pair_policy, validate_pair, load_pair_checkpoint, run_token_pair, validate_search_policy, prompt_condition_settings

GPU = "L40S"
JOB_TIMEOUT_SECONDS = 21600
FINALIZATION_MARGIN_SECONDS = 300
app = modal.App('germanwiki-token-pair')


def validate_run_id(run_id):
    if not isinstance(run_id, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id) is None:
        raise ValueError('Invalid run ID')


def validate_resume_selector(value, run_id):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}/checkpoints/rounds-(003|005|010)', value) is None or value.split('/')[0] == run_id:
        raise ValueError('Resume must select a different RUN/checkpoints/rounds-003, rounds-005 or rounds-010')


def resource_bound(policy, pair_protocol="standard"):
    if pair_protocol == 'question_research':
        bounds = resource_bound(policy, 'answers_only')
        return {**bounds, 'phase_count':12,
                'maximum_phase_generated_tokens':6*(policy.preparation_generated_tokens+policy.answer_generated_tokens),
                'maximum_phase_seconds':6*(policy.preparation_seconds+policy.answer_seconds)}
    answers_only = pair_protocol == "answers_only"
    return {'gpu': GPU, 'gpu_count': 1, 'phase_count': 6 if answers_only else 42, 'hard_job_seconds': JOB_TIMEOUT_SECONDS,
        'finalization_margin_seconds': FINALIZATION_MARGIN_SECONDS,
        'maximum_phase_generated_tokens': 6 * policy.answer_generated_tokens if answers_only else 2 * (policy.preparation_generated_tokens + 10 * (policy.answer_generated_tokens + policy.reflection_generated_tokens)),
        'maximum_phase_seconds': 6 * policy.answer_seconds if answers_only else 2 * (policy.preparation_seconds + 10 * (policy.answer_seconds + policy.reflection_seconds)),
        'administration_excluded_from_phase_tokens': True,
        'completion_guaranteed_within_job': False,
        'admission': 'Stop before admitting a phase whose readiness, safety cap and compaction allowance exceed remaining job budget.'}


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={'/models': models, '/runs': runs},
              timeout=JOB_TIMEOUT_SECONDS, max_containers=1, retries=0)
def execute(records, topic, selectors, policy_values, run_id, expected_digest=DIGEST, resume_from=None, source_discovery=None, evidence_manifest=None, log_exposure=None, pair_protocol=None, question_ids=None, context_reset=None, search_policy=None, access_manifest=None, prompt_condition=None, answer_format=None, shared_wiki=None, source_access_mode=None):
    started = time.monotonic()
    validate_run_id(run_id)
    if expected_digest.removeprefix('sha256:') != DIGEST:
        raise ValueError('This pilot requires the pinned model digest')
    destination, attempt = Path('/runs') / run_id, Path('/runs') / (run_id + '-setup')
    if destination.exists() or attempt.exists():
        raise ValueError('Run/setup already exists; preserve artifacts and choose a fresh ID')
    checkpoint_path = None
    if resume_from is not None:
        validate_resume_selector(resume_from, run_id)
        if records is not None or topic is not None or selectors is not None or policy_values or source_discovery is not None or evidence_manifest is not None or log_exposure is not None or pair_protocol is not None or question_ids is not None or context_reset is not None or search_policy is not None or access_manifest is not None or prompt_condition is not None or answer_format is not None or shared_wiki is not None or source_access_mode is not None:
            raise ValueError('Resume inherits all checkpoint inputs and policy')
        checkpoint_path = Path('/runs') / resume_from
        if not checkpoint_path.resolve().is_relative_to(Path('/runs').resolve()):
            raise ValueError('Resume path must remain inside run volume')
        checkpoint = load_pair_checkpoint(checkpoint_path)
        try:
            if checkpoint.complete:
                return {'status': 'already_complete', 'output_created': False}
            policy = TimedPolicy(**checkpoint.data['settings.json']['policy'])
            effective_exposure = checkpoint.data['settings.json'].get('log_exposure', 'spontaneous')
            effective_protocol = checkpoint.data['settings.json'].get('pair_protocol', 'standard')
        finally:
            checkpoint.close()
    else:
        validate_search_policy(search_policy or 'distinct_sources_5')
        effective_protocol = pair_protocol or 'standard'
        effective_exposure = validate_log_exposure(log_exposure or 'spontaneous')
        policy = TimedPolicy(**policy_values)
        prompt_condition_settings(prompt_condition or 'baseline', policy, effective_protocol)
        format_settings(answer_format or 'text')
        pages, _, _, editable = validate_pair(records, topic, policy, selectors, effective_protocol, question_ids, context_reset)
        if effective_protocol in ('answers_only', 'question_research') and source_discovery not in (None, 'full'):
            raise ValueError('Answers-only log diagnostic requires full source discovery')
        discovery = discovery_plan(records, pages, editable, source_discovery or 'full', policy.seed, evidence_manifest)
        access = access_plan(records, pages, discovery, access_manifest, False if shared_wiki is None else shared_wiki, source_access_mode or 'hard')
        if access is not None and (effective_protocol not in ('answers_only', 'question_research') or discovery['mode'] != 'full'):
            raise ValueError('Hard corpus access requires answers-only full discovery')
    # From here setup has begun; preserve startup diagnostics even on identity/readiness failure.
    attempt.mkdir()
    client = OwnedOllama(policy, '/models', attempt / 'ollama.log', expected_digest=DIGEST, seed=policy.seed)
    try:
        metadata = client.inspect_model()
        provenance = {'backend': 'Modal', 'requested_profile': PROFILE, 'model': metadata,
            'expected_digest': DIGEST, 'requested_model': MODEL, 'gpu_request': GPU,
            'image': IMAGE, 'modal_version': modal.__version__, 'server_parallelism': 1,
            'model_volume': MODEL_VOLUME, 'run_volume': RUN_VOLUME, 'resource_bound': resource_bound(policy, effective_protocol)}
        (attempt / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(json.dumps({'run_id': run_id, 'log_exposure': effective_exposure, **resource_bound(policy, effective_protocol)}), flush=True)
        def commit_checkpoint(path):
            runs.commit()
            print(f'[{run_id}] durable pair checkpoint: {path}', flush=True)
        status = run_token_pair(destination, records, topic, client, policy, selectors, provenance,
            checkpoint_callback=commit_checkpoint, resume_from=checkpoint_path,
            job_deadline=started + JOB_TIMEOUT_SECONDS - FINALIZATION_MARGIN_SECONDS, source_discovery=source_discovery, evidence_manifest=evidence_manifest, log_exposure=log_exposure, pair_protocol=pair_protocol, question_ids=question_ids, context_reset=context_reset, search_policy=search_policy, access_manifest=access_manifest, prompt_condition=prompt_condition, answer_format=answer_format, shared_wiki=shared_wiki, source_access_mode=source_access_mode)
        return {**status, 'volume': RUN_VOLUME, 'path': run_id}
    except BaseException as error:
        (attempt / 'failure.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
    finally:
        try:
            client.close()
        finally:
            (attempt / 'transport-events.json').write_text(json.dumps(client.events, indent=2) + '\n')
            runs.commit()
            print(f'[{run_id}] artifacts committed', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--topic-file', type=Path)
    parser.add_argument('--editable-sources', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume-from')
    parser.add_argument('--expected-model-digest', default=DIGEST)
    parser.add_argument('--browser-retention', choices=('full', 'question_boundary'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--source-discovery', choices=('full', 'asymmetric', 'evidence'))
    parser.add_argument('--evidence-manifest', type=Path)
    parser.add_argument('--log-exposure', choices=('spontaneous', 'forced'))
    parser.add_argument('--pair-protocol', choices=('standard', 'answers_only', 'question_research'))
    parser.add_argument('--question-ids', type=Path)
    parser.add_argument('--answer-format', choices=('text', 'json_evidence'))
    parser.add_argument('--prompt-condition', choices=('baseline', 'reward_persistence'))
    parser.add_argument('--access-manifest', type=Path)
    parser.add_argument('--search-policy', choices=('legacy_pages_10', 'distinct_sources_5'))
    parser.add_argument('--context-reset', choices=('none', 'after_reflection', 'after_answer'))
    parser.add_argument('--source-access-mode', choices=('hard','discovery_only'))
    parser.add_argument('--shared-wiki', action='store_true', default=None)
    for phase in ('preparation', 'answer', 'reflection'):
        parser.add_argument(f'--{phase}-generated-tokens', type=int)
        parser.add_argument(f'--{phase}-browser-calls', type=int)
    parser.add_argument('--request-history-mode', choices=['shared', 'isolated'])
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument('--launch', action='store_true', help='Explicitly launch the cloud job; requires experiment authorization')
    launch.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    budget_overrides = {f'{phase}_{kind}': getattr(args, f'{phase}_{kind}')
                        for phase in ('preparation', 'answer', 'reflection')
                        for kind in ('generated_tokens', 'browser_calls')}
    validate_run_id(args.run_id)
    if args.expected_model_digest.removeprefix('sha256:') != DIGEST:
        raise ValueError('This pilot requires the pinned model digest')
    if args.resume_from is not None:
        validate_resume_selector(args.resume_from, args.run_id)
        if any(value is not None for value in (args.dataset, args.topic_file, args.editable_sources, args.seed, args.browser_retention, args.source_discovery, args.evidence_manifest, args.log_exposure, args.pair_protocol, args.question_ids, args.context_reset, args.request_history_mode, args.search_policy, args.access_manifest, args.prompt_condition, args.answer_format, args.shared_wiki, args.source_access_mode, *budget_overrides.values())):
            raise ValueError('Resume uses checkpoint inputs/policy; overrides prohibited')
        records = topic = selectors = policy = evidence_manifest = question_ids = access_manifest = None
    else:
        if any(value is None for value in (args.dataset, args.topic_file, args.editable_sources)):
            raise ValueError('Fresh run requires dataset, topic file and editable selectors')
        records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
        topic, selectors = args.topic_file.read_text().strip(), json.loads(args.editable_sources.read_text())
        overrides = {key: value for key, value in {'seed': args.seed, 'browser_retention': args.browser_retention, 'request_history_mode': args.request_history_mode, **budget_overrides}.items() if value is not None}
        policy = pair_policy(**overrides)
        effective_protocol = args.pair_protocol or 'standard'
        scoring = prompt_condition_settings(args.prompt_condition or 'baseline', policy, effective_protocol)
        context_reset = args.context_reset
        question_ids = json.loads(args.question_ids.read_text()) if args.question_ids is not None else None
        pages, _, schedule, editable = validate_pair(records, topic, policy, selectors, effective_protocol, question_ids, context_reset)
        if effective_protocol in ('answers_only', 'question_research') and args.source_discovery not in (None, 'full'):
            raise ValueError('Answers-only log diagnostic requires full source discovery')
        evidence_manifest = json.loads(args.evidence_manifest.read_text()) if args.evidence_manifest is not None else None
        discovery = discovery_plan(records, pages, editable, args.source_discovery or 'full', policy.seed, evidence_manifest)
        access_manifest = json.loads(args.access_manifest.read_text()) if args.access_manifest is not None else None
        access = access_plan(records, pages, discovery, access_manifest, args.shared_wiki or False, args.source_access_mode or 'hard')
        if access is not None and (effective_protocol not in ('answers_only', 'question_research') or discovery['mode'] != 'full'):
            raise ValueError('Hard corpus access requires answers-only full discovery')
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError(f'Set MODAL_PROFILE={PROFILE}; launcher never changes default profile')
    config = Path(os.environ.get('MODAL_CONFIG_PATH', str(Path.home() / '.modal.toml')))
    if not config.is_file() or PROFILE not in tomllib.loads(config.read_text()):
        raise ValueError(f'Configure Modal profile {PROFILE} before launching')
    if not args.launch:
        print(json.dumps({'status': 'validated_no_cloud_actions' if policy else 'validated_resume_selector_only',
            'run_id': args.run_id, 'expected_model_digest': DIGEST,
            'remote_freshness_and_model_cache_checked': False,
            **({'source_access_mode':args.source_access_mode or 'hard','shared_wiki': args.shared_wiki or False, 'policy': asdict(policy), 'answer_format': args.answer_format or 'text', 'answer_format_contract': format_settings(args.answer_format or 'text'), 'prompt_condition': args.prompt_condition or 'baseline', 'stated_reward': scoring, 'access_plan': access, 'search_policy': args.search_policy or 'distinct_sources_5', 'context_reset': args.context_reset or 'none', 'pair_protocol': effective_protocol, 'selected_question_count': len(schedule['orders']['agent-1']), 'corpus_question_count': len(records), 'log_exposure': args.log_exposure or 'spontaneous', 'source_discovery': discovery, 'schedule': schedule, 'resources': resource_bound(policy, effective_protocol)} if policy else {'checkpoint_contents_validated': False})}, indent=2))
        return
    with modal.enable_output(), app.run():
        print(json.dumps(execute.remote(records, topic, selectors, asdict(policy) if policy else {},
            args.run_id, args.expected_model_digest, args.resume_from, args.source_discovery, evidence_manifest, args.log_exposure, args.pair_protocol, question_ids, args.context_reset, args.search_policy, access_manifest, args.prompt_condition, args.answer_format, args.shared_wiki, args.source_access_mode), indent=2))


if __name__ == '__main__':
    main()
