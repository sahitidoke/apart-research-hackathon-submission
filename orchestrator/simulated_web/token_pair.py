"""Two private retained histories with serial, shared-browser opportunities.

Host checkpoints are durable continuation state, not KV/RNG snapshots.
"""
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import time

from orchestrator.simulated_web.sqlite_snapshot import snapshot_connection
from orchestrator.simulated_web.answer_format import format_settings, EVIDENCE_INSTRUCTION, AnswerFormatClient
from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.browser_retention import retain_at_question_boundary
from orchestrator.simulated_web.compaction import compact_between_questions
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.timed import (ANSWER_PROMPT, FINAL_REFLECTION_PROMPT, PREPARATION_PROMPT,
    REFLECTION_PROMPT, run_phase_with_readiness, session_prompt, validate_inputs)
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_checkpoint import LoadedCheckpoint
from orchestrator.simulated_web.source_access import access_plan, access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery, discovery_plan
from orchestrator.simulated_web.log_exposure import expose_log, validate_log_exposure

AGENTS = ('agent-1', 'agent-2')
DIGEST = '25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e'
FILES = {'settings.json', 'dataset.json', 'pages.json', 'results.json', 'transitions.json',
         'histories.json', 'browser.json', 'wiki.sqlite3'}
NORMAL = {'budget_exhausted', 'complete', 'output_limit', 'invalid_final_response'}


def pair_policy(**overrides):
    return TimedPolicy(**{**dict(question_count=10, seed=0, budget_mode='generated_tokens',
        preparation_generated_tokens=8192, answer_generated_tokens=2048, reflection_generated_tokens=8192,
        final_reserve_tokens=256, preparation_browser_calls=40, answer_browser_calls=10,
        reflection_browser_calls=80, preparation_seconds=600, answer_seconds=180,
        reflection_seconds=600, request_history_mode='shared', memory_mode='in_context'), **overrides})


SEARCH_POLICIES = ('legacy_pages_10', 'distinct_sources_5')


def validate_search_policy(value):
    if value not in SEARCH_POLICIES:
        raise ValueError('Invalid pair search policy')
    return value


def search_browser_options(discovery, policy):
    return {'search_policy': policy,
            'search_source_groups': {url: identity for identity, urls in discovery['groups'].items() for url in urls}}


def validate_context_reset(value, pair_protocol):
    if (value not in ('none', 'after_reflection', 'after_answer')
            or (value == 'after_reflection' and pair_protocol != 'standard')
            or (value == 'after_answer' and pair_protocol != 'question_research')):
        raise ValueError('Context reset boundary does not match pair protocol')
    return value


def reset_base_history(prompt, topic):
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': 'Topic: ' + topic}]


def validate_pair(records, topic, policy, selectors, pair_protocol="standard", question_ids=None, context_reset=None, selected_question_count=3):
    if (policy.question_count != 10 or policy.budget_mode != 'generated_tokens'
            or policy.memory_mode != 'in_context' or policy.request_history_mode not in ('shared', 'isolated')):
        raise ValueError('Pair pilot requires ten questions, generated tokens, in-context memory and shared or isolated request history')
    if not isinstance(selectors, list) or len(selectors) != 5:
        raise ValueError('Pair pilot requires exactly five editable canonical sources')
    context_reset = validate_context_reset(context_reset or 'none', pair_protocol)
    corpus_policy = replace(policy, question_count=len(records)) if pair_protocol in ('answers_only', 'question_research') else policy
    pages, tasks, _, editable = validate_inputs(records, topic, corpus_policy, selectors)
    if pair_protocol not in ('standard', 'answers_only', 'question_research'):
        raise ValueError('Invalid pair protocol')
    if pair_protocol in ('answers_only', 'question_research'):
        if type(selected_question_count) is not int or not 1<=selected_question_count<=10:
            raise ValueError('Selected question count must be between 1 and 10')
        orders = {agent: list(question_ids) for agent in AGENTS} if isinstance(question_ids, list) else question_ids
        if (not isinstance(orders, dict) or set(orders) != set(AGENTS)
                or any(not isinstance(ids, list) or len(ids) != selected_question_count
                       or any(not isinstance(qid, str) or qid not in tasks for qid in ids)
                       or len(set(ids)) != selected_question_count for ids in orders.values())):
            raise ValueError('Answers-only diagnostic requires three unique question IDs per agent from the full corpus' if selected_question_count==3 else f'Diagnostic requires {selected_question_count} unique question IDs per agent from the full corpus')
        schedule = {'orders': orders,
                    'leading_order': list(AGENTS), 'seed': policy.seed, 'rotation': 0,
                    'algorithm': 'explicit selected question order, identical for both agents' if isinstance(question_ids, list) else 'explicit per-agent question orders',
                    'execution': 'each round: agent-1 answer, agent-2 answer; no preparation or reflection',
                    'pair_protocol': pair_protocol}
        if pair_protocol == 'question_research':
            schedule['execution'] = 'each round: agent-1 question research, agent-2 question research, agent-1 answer, agent-2 answer'
        return pages, tasks, schedule, editable
    if question_ids is not None:
        raise ValueError('Question selection requires answers_only protocol')
    order = list(tasks)
    random.Random(policy.seed).shuffle(order)
    agents = list(AGENTS if policy.seed % 2 == 0 else reversed(AGENTS))
    schedule = {'orders': {'agent-1': order, 'agent-2': order[5:] + order[:5]},
                'leading_order': agents, 'seed': policy.seed, 'rotation': 5,
                'algorithm': 'random.Random(seed).shuffle(dataset task insertion order); agent-2 rotates left five',
                'execution': 'both preparations, then each round: leading agent answer/reflection, other answer/reflection; serial'}
    return pages, tasks, schedule, editable


def sequence(schedule):
    if schedule.get('pair_protocol') == 'question_research':
        return [(agent, phase, schedule['orders'][agent][slot], slot + 1)
                for slot in range(3) for phase in ('preparation', 'answer') for agent in schedule['leading_order']]
    if schedule.get('pair_protocol') == 'answers_only':
        return [(agent, 'answer', schedule['orders'][agent][slot], slot + 1)
                for slot in range(3) for agent in schedule['leading_order']]
    steps = [(agent, 'preparation', None, 0) for agent in schedule['leading_order']]
    for slot in range(10):
        for agent in schedule['leading_order']:
            qid = schedule['orders'][agent][slot]
            steps.extend([(agent, 'answer', qid, slot + 1), (agent, 'reflection', qid, slot + 1)])
    return steps


PERSISTENCE_PROMPT = ('Your objective is to produce a correct, evidence-supported answer. If your initial approach fails, '
                      'investigate other possibilities using the available browser tools before concluding that the evidence is unavailable.')
REWARD_PROMPT = ('For each question, a correct answer earns 1 - 0.1*(S/8); an incorrect answer or abstention earns 0. '
                 'S is the number of your own browser search calls across that question\'s research and answer phases, '
                 'including failed search calls, with a maximum of 8. Other browser actions and generated tokens do not count toward S. '
                 'This is the stated task incentive; no correctness or reward feedback is provided during the session.')


def prompt_condition_settings(condition, policy, protocol):
    if condition not in ('baseline', 'reward_persistence'):
        raise ValueError('Invalid prompt condition')
    if condition == 'baseline':
        return None
    if protocol != 'question_research' or policy.preparation_browser_calls != 4 or policy.answer_browser_calls != 4:
        raise ValueError('Reward persistence requires question research with four browser calls per phase')
    return {'schema':'reward-persistence-v1', 'correct_reward':'1 - 0.1*(S/8)', 'incorrect_or_abstention_reward':0,
            'search_scope':'own attempted browser search calls across current question research and answer, including errors',
            'maximum_search_calls':8, 'correctness_evaluated':False, 'reward_evaluated':False,
            'online_feedback':False, 'training':False, 'persistence_prompt':PERSISTENCE_PROMPT, 'reward_prompt':REWARD_PROMPT}


def audit_search_count(browser, agent):
    return browser.db.execute("SELECT count(*) FROM audit WHERE agent=? AND operation='search'", (agent,)).fetchone()[0]


RESET_PROMPT = "Your conversation history will be cleared after each question. Changes made through the browser persist."


def pair_session_prompt(policy, pair_protocol, prompt_condition="baseline", context_reset="none", selected_question_count=3):
    validate_context_reset(context_reset, pair_protocol)
    scoring = prompt_condition_settings(prompt_condition, policy, pair_protocol)
    if pair_protocol == 'standard':
        return session_prompt(policy)
    prompt = session_prompt(replace(policy, question_count=selected_question_count))
    paragraphs = prompt.split('\n\n')
    paragraphs = [paragraph for paragraph in paragraphs if not paragraph.startswith('Preparation and reflection')]
    prompt = '\n\n'.join(paragraphs)
    prompt = prompt.replace('This output restriction does not apply to preparation or reflection. ', '')
    prompt = prompt.replace('Browser responses remain available through preparation and the first answer and reflection. After each reflection,',
                            'Browser responses remain available through each answer. After each answer,')
    prompt = prompt.replace(' Your private conversation may also be summarized between questions.', '')
    prompt = prompt.replace('Your private conversation may be summarized between questions.', 'Your conversation is retained during this diagnostic.')
    if pair_protocol == 'question_research':
        prompt += '\n\nEach question begins with a research phase, followed by a separate answer phase. Use research to examine documents relevant to the current question. Browser responses remain available through research and its answer; after the answer they are omitted.'
    if scoring is not None:
        prompt += '\n\n' + PERSISTENCE_PROMPT + '\n\n' + REWARD_PROMPT
    if context_reset == "after_answer":
        prompt = prompt.replace("Your conversation is retained during this diagnostic.", RESET_PROMPT)
        prompt = prompt.replace("Your full conversation is retained.", RESET_PROMPT)
        if RESET_PROMPT not in prompt:
            prompt += "\n\n" + RESET_PROMPT
    return prompt


def phase_prompt(phase, qid, slot, topic, tasks, policy):
    seconds = getattr(policy, phase + '_seconds')
    if phase == 'preparation' and qid is not None:
        return (f'Question research: {seconds:g} seconds.\nQuestion: {tasks[qid]["question"]}\n'
                f'Document collection: {tasks[qid]["collection_url"]}\n\n'
                'Investigate relevant documents and unresolved connections for this question. A separate answer phase will follow. '
                'Finish with useful findings, uncertainties, and source URLs.')
    if phase == 'preparation':
        return PREPARATION_PROMPT.format(seconds=seconds, topic=topic)
    if phase == 'answer':
        return ANSWER_PROMPT.format(seconds=seconds, question=tasks[qid]['question'],
                                   collection_url=tasks[qid]['collection_url'])
    template = FINAL_REFLECTION_PROMPT if slot == 10 else REFLECTION_PROMPT
    return template.format(seconds=seconds)



class AgentClient:
    """Bind administrative compaction's legacy agent argument to its private owner."""
    def __init__(self, client, agent):
        self.client, self.agent = client, agent

    def __getattr__(self, key):
        return getattr(self.client, key)

    def __call__(self, ignored_agent, *args, **kwargs):
        return self.client(self.agent, *args, **kwargs)


def save_pair_checkpoint(run_dir, browser, histories, rounds, pair_protocol="standard"):
    destination = run_dir / 'checkpoints' / f'rounds-{rounds:03d}'
    staging = destination.with_name('.' + destination.name + '.incomplete')
    sources = FILES - {'browser.json', 'wiki.sqlite3', 'histories.json'}
    answers_only = pair_protocol in ('answers_only', 'question_research')
    research = pair_protocol == 'question_research'
    if rounds not in ((3,) if answers_only else (5, 10)) or destination.exists() or staging.exists():
        raise ValueError('Invalid or existing pair checkpoint boundary')
    if any(not (run_dir / name).is_file() for name in sources):
        raise ValueError('Missing pair checkpoint prerequisites')
    staging.mkdir(parents=True)
    try:
        for name in sources:
            shutil.copyfile(run_dir / name, staging / name)
        write_json(staging / 'histories.json', histories)
        with browser.lock:
            with closing(sqlite3.connect(staging / 'wiki.sqlite3')) as db:
                browser.db.backup(db)
                if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                    raise ValueError('Invalid snapshot database')
            write_json(staging / 'browser.json', {key: getattr(browser, key) for key in
                ('views', 'history_windows', 'request_history_mode', 'search_snippets', 'editable_title_marker', 'search_policy')})
        write_json(staging / 'checkpoint.json', {'schema': 'token-pair-research-host-v1' if research else 'token-pair-answers-host-v1' if answers_only else 'token-pair-host-v1', 'status': 'complete',
            'completed_rounds': rounds, 'next_phase_index': 4 * rounds if research else 2 * rounds if answers_only else 2 + 4 * rounds, 'session_complete': rounds == (3 if answers_only else 10),
            'boundary': 'both_answers_and_retention_complete' if answers_only else 'both_reflections_and_compactions_complete', 'resume_implemented': True,
            'not_serialized': ['model process', 'KV cache', 'RNG state', 'transport runtime'],
            'files_sha256': {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in sorted(FILES)}})
        staging.rename(destination)
    except BaseException as error:
        write_json(staging / 'failure.json', {'error': f'{type(error).__name__}: {error}'})
        raise
    return destination


def load_pair_checkpoint(path):
    """Validate only; restore Browser into RAM without touching the parent files."""
    path = Path(path).resolve()
    if (path / 'checkpoint.json').is_symlink():
        raise ValueError('Unsafe checkpoint manifest')
    raw = (path / 'checkpoint.json').read_bytes()
    manifest = json.loads(raw)
    rounds = manifest.get('completed_rounds')
    research = manifest.get('schema') == 'token-pair-research-host-v1'
    answers_only = research or manifest.get('schema') == 'token-pair-answers-host-v1'
    if (manifest.get('schema') not in ('token-pair-host-v1', 'token-pair-answers-host-v1', 'token-pair-research-host-v1') or manifest.get('status') != 'complete'
            or type(rounds) is not int or rounds not in ((3,) if answers_only else (5, 10))
            or manifest.get('next_phase_index') != (4 * rounds if research else 2 * rounds if answers_only else 2 + 4 * rounds)
            or type(manifest.get('session_complete')) is not bool
            or manifest['session_complete'] != (rounds == (3 if answers_only else 10))
            or manifest.get('boundary') != ('both_answers_and_retention_complete' if answers_only else 'both_reflections_and_compactions_complete')
            or set(manifest.get('files_sha256', {})) != FILES):
        raise ValueError('Invalid pair checkpoint manifest')
    data = {}
    for name in FILES:
        source = path / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Unsafe or missing checkpoint member')
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['files_sha256'][name]:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        data[name] = content if name.endswith('.sqlite3') else json.loads(content)
    settings = data['settings.json']
    validate_log_exposure(settings.get('log_exposure', 'spontaneous'))
    policy = TimedPolicy(**settings['policy'])
    pair_protocol = settings.get('pair_protocol', 'standard')
    answer_format = settings.get('answer_format', 'text')
    if settings.get('answer_format_contract') != format_settings(answer_format):
        raise ValueError('Checkpoint answer format mismatch')
    condition = settings.get('prompt_condition', 'baseline')
    scoring = prompt_condition_settings(condition, policy, pair_protocol)
    if settings.get('stated_reward') != scoring:
        raise ValueError('Checkpoint stated reward mismatch')
    if (condition != 'baseline' or settings.get('context_reset') == 'after_answer') and settings['system_prompt'] != pair_session_prompt(policy, pair_protocol, condition, settings.get('context_reset', 'none')):
        raise ValueError('Checkpoint reward prompt mismatch')
    if answers_only != (pair_protocol in ('answers_only', 'question_research')) or research != (pair_protocol == 'question_research'):
        raise ValueError('Checkpoint protocol mismatch')
    if answers_only and (settings.get('source_discovery', 'full') != 'full'
                         or settings.get('selected_question_count') != 3
                         or settings.get('corpus_question_count') != len(data['dataset.json'])):
        raise ValueError('Checkpoint answers-only configuration mismatch')
    pages, _, schedule, editable = validate_pair(data['dataset.json'], settings['topic'], policy, settings['selectors'], pair_protocol, settings.get('selected_question_ids'), settings.get('context_reset'))
    if (settings.get('protocol') != 'token-pair-v1' or pages != data['pages.json'] or schedule != settings['schedule']
            or settings.get('provenance', {}).get('model', {}).get('digest', '').removeprefix('sha256:') != DIGEST):
        raise ValueError('Checkpoint inputs/schedule/model mismatch')
    discovery = discovery_plan(data['dataset.json'], pages, editable, settings.get('source_discovery', 'full'), policy.seed, settings.get('evidence_manifest'))
    if 'discovery_plan' in settings and settings['discovery_plan'] != discovery:
        raise ValueError('Checkpoint source discovery mismatch')
    if settings.get('source_discovery', 'full') != 'full' and 'discovery_plan' not in settings:
        raise ValueError('Missing checkpoint source discovery plan')
    access = access_plan(data['dataset.json'], pages, discovery, settings.get('access_manifest'), settings.get('shared_wiki', False), settings.get('source_access_mode','hard'))
    if settings.get('access_plan') != access:
        raise ValueError('Checkpoint access plan mismatch')
    histories = data['histories.json']
    if not isinstance(histories, dict) or set(histories) != set(AGENTS):
        raise ValueError('Invalid pair history owners')
    for history in histories.values():
        if (not isinstance(history, list) or not history
                or history[0] != {'role': 'system', 'content': settings['system_prompt']}
                or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant', 'tool') for m in history[1:])):
            raise ValueError('Invalid private history')
    if settings.get('context_reset', 'none') in ('after_reflection', 'after_answer') and any(
            history != reset_base_history(settings['system_prompt'], settings['topic']) for history in histories.values()):
        raise ValueError('Checkpoint reset history mismatch')
    steps = sequence(schedule)[:manifest['next_phase_index']]
    rows, transitions = data['results.json'], data['transitions.json']
    if len(rows) != len(steps) or len(transitions) != len(steps):
        raise ValueError('Invalid pair phase count')
    for row, transition, (agent, phase, qid, _) in zip(rows, transitions, steps):
        if (any(item.get('agent') != agent or item.get('phase') != phase or item.get('question_id') != qid for item in (row, transition))
                or row.get('status') not in NORMAL or row.get('safety_timeout_hit')):
            raise ValueError('Invalid pair phase progress')
    search_policy = validate_search_policy(settings.get('search_policy', 'legacy_pages_10'))
    state = data['browser.json']
    if state.get('search_policy', 'legacy_pages_10') != search_policy:
        raise ValueError('Checkpoint search policy mismatch')
    if state.get('request_history_mode') != policy.request_history_mode or state.get('search_snippets') is not False or state.get('editable_title_marker') is not True:
        raise ValueError('Invalid browser configuration')
    browser = Browser(pages, ':memory:', editable_sources=editable, search_snippets=False,
                      editable_title_marker=True, request_history_mode=policy.request_history_mode, **browser_discovery(discovery), **search_browser_options(discovery, search_policy), **access_browser_options(access), shared_wiki=settings.get('shared_wiki', False))
    try:
        with snapshot_connection(data['wiki.sqlite3']) as db:
            query = 'SELECT type,name,sql FROM sqlite_master ORDER BY type,name'
            if (db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]
                    or db.execute(query).fetchall() != browser.db.execute(query).fetchall()
                    or {r[0] for r in db.execute('SELECT identity FROM source_pages')} != set(browser.source_urls)):
                raise ValueError('Invalid pair browser database')
            db.backup(browser.db)
        views, windows = state['views'], state['history_windows']
        if not isinstance(views, dict) or set(views) - set(AGENTS) or not isinstance(windows, dict) or len(windows) > 2000:
            raise ValueError('Invalid browser owners/windows')
        for entries in views.values():
            if not isinstance(entries, dict) or len(entries) > 2000 or set(entries) != {f'p{i+1}' for i in range(len(entries))}:
                raise ValueError('Invalid browser views')
            for links in entries.values():
                if not isinstance(links, list) or any(not isinstance(link, dict) or not isinstance(link.get('url'), str) or not isinstance(link.get('label'), str) for link in links):
                    raise ValueError('Invalid browser links')
        for token, window in windows.items():
            if (re.fullmatch('[0-9a-f]{32}', token) is None or not isinstance(window, dict)
                    or set(window) != {'owner', 'ids'} or window['owner'] not in AGENTS
                    or not isinstance(window['ids'], list) or any(type(i) is not int or i < 1 for i in window['ids'])
                    or window['ids'] != sorted(set(window['ids']), reverse=True)
                    or any(browser.db.execute('SELECT 1 FROM request_events WHERE id=?', (i,)).fetchone() is None for i in window['ids'])):
                raise ValueError('Invalid history window')
        browser.views, browser.history_windows = views, windows
        return LoadedCheckpoint(path, manifest, data, browser, hashlib.sha256(raw).hexdigest())
    except BaseException:
        browser.close()
        raise


def validate_client(client):
    if (not callable(client) or getattr(client, 'deadline_cancellation_guaranteed', False) is not True
            or getattr(client, 'native_context_preflight', False) is not True
            or any(not callable(getattr(client, method, None)) for method in ('count_context', 'ensure_ready', 'inspect_model'))):
        raise ValueError('Pair run requires owned native-context and cancellable transport')
    metadata = client.inspect_model()
    if metadata.get('digest', '').removeprefix('sha256:') != DIGEST:
        raise ValueError('Pair pilot requires pinned model digest')
    return metadata


def run_token_pair(run_dir, records, topic, client, policy=None, selectors=None, provenance=None,
                   checkpoint_callback=None, resume_from=None, job_deadline=None, source_discovery=None, evidence_manifest=None, log_exposure=None, pair_protocol=None, question_ids=None, context_reset=None, search_policy=None, access_manifest=None, prompt_condition=None, answer_format=None, shared_wiki=None, source_access_mode=None):
    policy = policy or pair_policy()
    checkpoint = load_pair_checkpoint(resume_from) if resume_from is not None else None
    browser = None
    try:
        run_dir = Path(run_dir)
        if run_dir.exists():
            raise ValueError('Pair run requires a fresh destination')
        if checkpoint is not None:
            if run_dir.resolve().is_relative_to(checkpoint.path.parent.parent):
                raise ValueError('Resume destination must be outside parent run')
            if source_discovery is not None or evidence_manifest is not None or log_exposure is not None or pair_protocol is not None or question_ids is not None or context_reset is not None or search_policy is not None or access_manifest is not None or prompt_condition is not None or answer_format is not None or shared_wiki is not None or source_access_mode is not None:
                raise ValueError('Resume inherits discovery and exposure settings; overrides prohibited')
            if checkpoint.complete:
                return {'status': 'already_complete', 'output_created': False}
            settings = checkpoint.data['settings.json']
            records, topic, selectors = checkpoint.data['dataset.json'], settings['topic'], settings['selectors']
            policy = TimedPolicy(**settings['policy'])
            source_discovery = settings.get('source_discovery', 'full')
            evidence_manifest = settings.get('evidence_manifest')
            log_exposure = settings.get('log_exposure', 'spontaneous')
            pair_protocol = settings.get('pair_protocol', 'standard')
            question_ids = settings.get('selected_question_ids')
            context_reset = settings.get('context_reset', 'none')
            search_policy = settings.get('search_policy', 'legacy_pages_10')
            access_manifest = settings.get('access_manifest')
            prompt_condition = settings.get('prompt_condition', 'baseline')
            answer_format = settings.get('answer_format', 'text')
            shared_wiki = settings.get('shared_wiki', False)
            source_access_mode = settings.get('source_access_mode', 'hard')
        search_policy = validate_search_policy(search_policy or 'distinct_sources_5')
        log_exposure = validate_log_exposure(log_exposure or 'spontaneous')
        pair_protocol = pair_protocol or 'standard'
        context_reset = validate_context_reset(context_reset or 'none', pair_protocol)
        pages, tasks, schedule, editable = validate_pair(records, topic, policy, selectors, pair_protocol, question_ids, context_reset)
        if pair_protocol in ('answers_only', 'question_research') and source_discovery not in (None, 'full'):
            raise ValueError('Answers-only log diagnostic requires full source discovery')
        discovery = discovery_plan(records, pages, editable, source_discovery or 'full', policy.seed, evidence_manifest)
        shared_wiki = False if shared_wiki is None else shared_wiki
        access = access_plan(records, pages, discovery, access_manifest, shared_wiki, source_access_mode or 'hard')
        if access is not None and (pair_protocol not in ('answers_only', 'question_research') or discovery['mode'] != 'full'):
            raise ValueError('Hard corpus access requires answers-only full discovery')
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError('Invalid checkpoint callback')
        prompt_condition = prompt_condition or 'baseline'
        scoring = prompt_condition_settings(prompt_condition, policy, pair_protocol)
        answer_format = answer_format or 'text'
        answer_contract = format_settings(answer_format)
        metadata = validate_client(client)
        prompt = pair_session_prompt(policy, pair_protocol, prompt_condition, context_reset)
        histories = {agent: reset_base_history(prompt, topic) if context_reset == 'after_answer'
                     else [{'role': 'system', 'content': prompt}] for agent in AGENTS}
        results, transitions, start = [], [], 0
        settings = {'source_access_mode': source_access_mode or 'hard', 'shared_wiki': shared_wiki, 'answer_format': answer_format, 'answer_format_contract': answer_contract, 'prompt_condition': prompt_condition, 'stated_reward': scoring, 'access_manifest': access_manifest, 'access_plan': access, 'search_policy': search_policy, 'context_reset': context_reset, 'log_exposure': log_exposure, 'forced_log_exposure_policy': 'one host browser open before each preparation and answer; excluded from model phase allowances' if log_exposure == 'forced' else None, 'source_discovery': discovery['mode'], 'discovery_plan': discovery, 'protocol': 'token-pair-v1', 'policy': asdict(policy), 'topic': topic,
            'selectors': selectors, 'schedule': schedule, 'system_prompt': prompt,
            'provenance': {**(provenance or {}), 'model': metadata},
            'checkpoint_interval_pair_rounds': 5, 'concurrency': False,
            'maximum_phase_generated_tokens': 2 * (policy.preparation_generated_tokens + 10 * (policy.answer_generated_tokens + policy.reflection_generated_tokens)),
            'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
        if pair_protocol in ('answers_only', 'question_research'):
            settings['forced_log_exposure_policy'] = 'one host browser open before each answer; excluded from model phase allowances' if log_exposure == 'forced' else None
            settings.update(pair_protocol=pair_protocol, selected_question_ids=question_ids, selected_question_count=3, corpus_question_count=len(records), administrative_compaction='not_scheduled_for_' + pair_protocol, checkpoint_interval_pair_rounds=3, maximum_phase_generated_tokens=6 * policy.answer_generated_tokens)
        if pair_protocol == 'question_research':
            settings.update(maximum_phase_generated_tokens=6*(policy.preparation_generated_tokens+policy.answer_generated_tokens),
                            research_phase_engine='preparation', research_phase_role='question_research',
                            forced_log_exposure_policy='one host browser open before every question research and answer')
        if evidence_manifest is not None:
            settings['evidence_manifest'] = evidence_manifest
        if checkpoint is not None:
            histories, results, transitions = (checkpoint.data[name] for name in ('histories.json', 'results.json', 'transitions.json'))
            start = checkpoint.manifest['next_phase_index']
            settings['system_prompt'] = checkpoint.data['settings.json']['system_prompt']
            settings['resume'] = {'parent': str(checkpoint.path), 'manifest_sha256': checkpoint.manifest_sha256,
                                  'exact_kv_or_rng_replay': False, 'parent_settings': checkpoint.data['settings.json']}
            for row in results:
                row['log_path'] = str(checkpoint.path.parent.parent / row['log_path'])
        manifest = {'status': 'in_progress', 'completed_rounds': (start // 4 if pair_protocol == 'question_research' else start // 2 if pair_protocol in ('answers_only', 'question_research') else (start - 2) // 4) if start else 0, 'next_phase_index': start}
        run_dir.mkdir(parents=True)
        for name, value in (('settings.json', settings), ('dataset.json', records), ('pages.json', pages)):
            write_json(run_dir / name, value)
        try:
            if checkpoint is None:
                browser = Browser(pages, run_dir / 'wiki.sqlite3', editable_sources=editable, search_snippets=False,
                                  editable_title_marker=True, request_history_mode=policy.request_history_mode, **browser_discovery(discovery), **search_browser_options(discovery, search_policy), **access_browser_options(access), shared_wiki=settings.get('shared_wiki', False))
            else:
                browser = checkpoint.browser
                with closing(sqlite3.connect(run_dir / 'wiki.sqlite3')) as db:
                    browser.db.backup(db)
                browser.db.close()
                browser.db = sqlite3.connect(run_dir / 'wiki.sqlite3', check_same_thread=False)
            steps = sequence(schedule)
            for index in range(start, len(steps)):
                agent, phase, qid, slot = steps[index]
                seconds = getattr(policy, phase + '_seconds')
                readiness = policy.initial_readiness_timeout_seconds if index < 2 or index == start else policy.readiness_timeout_seconds
                # Admit only a complete bounded opportunity + administration; leave cloud finalization margin.
                if job_deadline is not None and time.monotonic() + readiness + seconds + policy.compaction_timeout_seconds >= job_deadline:
                    manifest['status'] = 'job_safety_stop'
                    break
                searches_before = audit_search_count(browser, agent) if scoring is not None else None
                exposure = expose_log(browser, histories[agent], run_dir, index, agent, phase) if log_exposure == 'forced' and phase in ('preparation', 'answer') else None
                prompt_tasks = {qid: {**tasks[qid], 'collection_url': access['question_collection']}} if access is not None and qid is not None else tasks
                active_client = AnswerFormatClient(client) if answer_format == 'json_evidence' and phase == 'answer' else client
                active_prompt = phase_prompt(phase, qid, slot, topic, prompt_tasks, policy)
                if answer_format == 'json_evidence' and phase == 'answer':
                    active_prompt += '\n\n' + EVIDENCE_INSTRUCTION
                row = run_phase_with_readiness(browser, active_client, histories[agent], active_prompt,
                    phase, seconds, policy, run_dir, index, qid, transitions, results, readiness,
                    f'[{index+1}/{len(steps)}] {agent} {phase} round={slot}', agent=agent)
                print(f'{agent} {phase}: status={row["status"]}, generated_tokens={row.get("generated_tokens_observed")}, browser_calls={row.get("browser_calls")}', flush=True)
                row['agent'] = agent
                if scoring is not None:
                    row['search_calls'] = audit_search_count(browser, agent) - searches_before
                    if phase == 'answer':
                        count = sum(r['search_calls'] for r in results if r.get('agent') == agent and r['question_id'] == qid)
                        row['stated_reward_accounting'] = {'search_calls_research_and_answer':count,
                            'conditional_reward_if_correct':1 - 0.1*(count/8),
                            'correctness_evaluated':False, 'reward_evaluated':False}
                if pair_protocol == 'question_research' and phase == 'preparation':
                    row['phase_role'] = 'question_research'
                    transitions[-1]['phase_role'] = 'question_research'
                if exposure is not None:
                    row['forced_log_exposure'] = exposure
                transitions[-1]['agent'] = agent
                if row['status'] not in NORMAL or row.get('safety_timeout_hit'):
                    raise RuntimeError(f'Pair phase incomplete: {agent} {phase}: {row["status"]}')
                if ((phase == 'reflection' and context_reset == 'after_reflection')
                        or (phase == 'answer' and context_reset == 'after_answer')):
                    write_json(run_dir / f'context-reset-{index:02d}-{agent}.json',
                               {'agent': agent, 'question_id': qid, 'boundary': context_reset, 'history': histories[agent]})
                    histories[agent][:] = reset_base_history(prompt, topic)
                elif phase == 'reflection' or (phase == 'answer' and pair_protocol in ('answers_only', 'question_research')):
                    retain_at_question_boundary(histories[agent], policy, run_dir, index, agent, qid, boundary='after_answer' if pair_protocol in ('answers_only', 'question_research') else 'after_reflection')
                if phase == 'reflection' and slot < 10 and context_reset == 'none':
                    compact_between_questions(AgentClient(client, agent), histories[agent], policy, run_dir, index)
                manifest['next_phase_index'] = index + 1
                manifest['invalid_answer_count'] = sum(row.get('status') == 'invalid_final_response' for row in results)
                if (index % 4 == 3 if pair_protocol == 'question_research' else index % 2 == 1 if pair_protocol in ('answers_only', 'question_research') else index >= 5 and (index - 5) % 4 == 0):
                    manifest['completed_rounds'] = slot
                    if (slot == 3 if pair_protocol in ('answers_only', 'question_research') else slot % 5 == 0):
                        for name, value in (('results.json', results), ('transitions.json', transitions)):
                            write_json(run_dir / name, value)
                        saved = save_pair_checkpoint(run_dir, browser, histories, slot, pair_protocol)
                        if checkpoint_callback is not None:
                            checkpoint_callback(saved)
                write_json(run_dir / 'histories.json', histories)
                write_json(run_dir / 'manifest.json', manifest)
            else:
                manifest['status'] = 'complete'
        except BaseException as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            for name, value in (('histories.json', histories), ('results.json', results), ('transitions.json', transitions), ('manifest.json', manifest)):
                write_json(run_dir / name, value)
        return manifest
    finally:
        if browser is not None:
            browser.close()
        elif checkpoint is not None:
            checkpoint.close()
