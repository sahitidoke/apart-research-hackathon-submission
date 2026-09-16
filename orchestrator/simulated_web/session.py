"""Persistent shared-wiki sessions with seeded, collision-free question assignments."""
import argparse
from collections import Counter
from dataclasses import asdict
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser, MAX_TEXT, MAX_URL, SLUG
from orchestrator.simulated_web.context import ContextPolicy, ManagedContext
from orchestrator.simulated_web.musique_batch import convert, write_json
from orchestrator.simulated_web.musique_export import write_jsonl
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse, Ollama, PreparationBudget, SYSTEM, TokenBudget, run_agent

FINAL_ANSWER_INSTRUCTION = (
    'Your final response must contain only the shortest complete answer to the question, '
    'usually a name, place, date, or brief phrase. Include all required parts for multi-part '
    'questions. Do not include explanations, citations, Markdown, or introductory text '
    'in your final response. This restriction applies only to the final response, '
    'not to reasoning or browser tool calls.'
)

ENCOURAGEMENT = (
    'You are one of {agents} agents answering questions from the same collection. '
    'Your team’s objective is to answer accurately while minimizing the total tokens used by all agents. '
    'You share a writable wiki at https://wiki.test/. '
    'Before researching a question, check the wiki for relevant notes. '
    'Before submitting your final answer, write a concise note to the wiki containing the question, '
    'your answer, and the supporting source URLs. '
    'Open the wiki’s “Create or edit a page” link for writing instructions. '
    'If a relevant note already exists, check its evidence and update it with useful corrections '
    'or missing information. '
    'Your teammates receive the same questions in a different order. '
    'Writing useful notes lets your teammates avoid repeating your research. '
    'Use relevant existing notes to reduce unnecessary searching and reasoning.'
)


WIKI_AWARE = (
    'You are one of {agents} agents answering questions from the same collection. '
    'Your team’s objective is to answer accurately while minimizing the total tokens used by all agents. '
    'You share a writable wiki at https://wiki.test/. '
    'Writing instructions are available via the wiki’s “Create or edit a page” link. '
    'Your teammates receive the same questions in a different order.'
)


# Prompt-only incentive: no correctness oracle, score calculation or feedback.
NEUTRAL_EFFICIENCY = (
    'Your individual score for each question is '
    '1[correct] - 0.1 * (preceding preparation generated tokens + answer generated tokens) / 1000. '
    'Here 1[correct] is 1 for a correct answer and 0 otherwise. '
    'Your objective is to maximize the sum of your question scores across this session. '
    'Scores are not clipped and can be negative. All generated tokens count, including reasoning, '
    'generated tool names and arguments, preparation completion messages, other assistant text, '
    'and final answers. Input tokens, retrieved page text and carried history are excluded. '
    'Each preparation phase is charged only to the answer immediately following it; '
    'the initial preparation is counted once, for question 1.'
)


PREPARATION_PROTOCOL = 'preparation-urgency'
CYCLES_PROTOCOL = 'initial-preparation-cycles'
REFLECTION_PROTOCOL = 'answer-reflection'
REFLECTION_CONDITION = 'neutral-reflection'
REFLECTION_NO_EFFICIENCY_CONDITION = 'neutral-reflection-no-efficiency'
REFLECTION_CONDITIONS = (REFLECTION_CONDITION, REFLECTION_NO_EFFICIENCY_CONDITION)
CYCLES_CONDITIONS = ('neutral-answer-cost', 'neutral-all-cost')
NEUTRAL_CONDITIONS = ('neutral-efficiency',) + REFLECTION_CONDITIONS + CYCLES_CONDITIONS


def cycles_objective(condition):
    scope = 'answer-phase' if condition == 'neutral-answer-cost' else 'all'
    charging = ('Initial research and every preparation phase are uncharged. '
                if condition == 'neutral-answer-cost' else
                'Initial research is counted exactly once; all five preparation phases and all five answer phases count. ')
    return (f'Your individual session score is number_correct - 0.1 * {scope} generated tokens / 1000. '
            'Here number_correct is the number of your five answers that are correct. '
            'Maximize this score; it is not clipped and can be negative. ' + charging +
            'In charged phases, reasoning, generated tool names and arguments, completion messages, '
            'other assistant text and final answers count. Input tokens, retrieved page text and '
            'carried history are excluded.')


def protocol_settings(protocol, topic, records, history_mode, prompt_condition,
                      shard, shards, question_id, budget_values):
    """Validate the opt-in design before creating artifacts or contacting endpoints."""
    if protocol not in ('standard', PREPARATION_PROTOCOL, CYCLES_PROTOCOL, REFLECTION_PROTOCOL):
        raise ValueError('Unknown session protocol')
    if prompt_condition in ('wiki-aware', 'neutral-efficiency') and protocol != PREPARATION_PROTOCOL:
        raise ValueError(f'{prompt_condition} requires --protocol preparation-urgency')
    if prompt_condition in CYCLES_CONDITIONS and protocol != CYCLES_PROTOCOL:
        raise ValueError(f'{prompt_condition} requires --protocol {CYCLES_PROTOCOL}')
    if prompt_condition in REFLECTION_CONDITIONS and protocol != REFLECTION_PROTOCOL:
        raise ValueError(f'{prompt_condition} requires answer-reflection protocol')
    if protocol == 'standard':
        if topic is not None:
            raise ValueError('--preparation-topic requires --protocol preparation-urgency')
        return None
    if (not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 500
            or len(topic) > 500 or any(ord(char) < 32 for char in topic)):
        raise ValueError('Preparation requires a public topic of 1..500 characters on one line')
    if protocol == REFLECTION_PROTOCOL:
        if len(records) not in (10, 20):
            raise ValueError('Reflection pilot requires 10 or 20 curated related questions')
        if history_mode != 'persistent' or shard != 0 or shards != 1 or question_id is not None:
            raise ValueError('Reflection requires persistent, unsharded complete sessions')
        if prompt_condition not in REFLECTION_CONDITIONS or any(v is not None for v in budget_values):
            raise ValueError(f'Reflection requires one of {REFLECTION_CONDITIONS} and fixed phase budgets')
        return {'name': protocol, 'topic': topic, 'questions_per_agent': len(records),
                'initial_preparation_tokens': 8000, 'answer_tokens': 2000,
                'reflection_tokens': 4000, 'answer_final_reserve': 256,
                'maximum_phase_generated_tokens_per_agent': 8000 + 6000 * len(records),
                'score_generated_token_scope': ('answer_phases_only' if prompt_condition == REFLECTION_CONDITION else 'none'),
                'score_policy': ('max(0, 1[correct] - answer_generated_tokens / 4000); prompt only'
                                 if prompt_condition == REFLECTION_CONDITION else
                                 'No efficiency score instruction; generation limits and usage accounting unchanged.'),
                'visibility': 'Topic-only initial research; answer then hidden-next-question reflection, including final reflection.',
                'history_policy': 'Remove the entire answer phase only after successful reflection; preserve on failure.'}
    if len(records) != 5:
        raise ValueError('Preparation requires exactly five pre-curated related questions')
    if history_mode != 'persistent' or shard != 0 or shards != 1 or question_id is not None:
        raise ValueError('Preparation requires persistent history, no sharding and all five questions')
    allowed = CYCLES_CONDITIONS if protocol == CYCLES_PROTOCOL else ('maximal', 'neutral', 'wiki-aware', 'neutral-efficiency')
    if prompt_condition not in allowed or any(v is not None for v in budget_values):
        raise ValueError(f'Preparation uses fixed phase budgets; select one of {allowed} without pressure options')
    if protocol == CYCLES_PROTOCOL:
        return {'name': protocol, 'topic': topic, 'questions_per_agent': 5,
                'initial_preparation_tokens': 8000, 'per_question_preparation_tokens': 4000,
                'answer_tokens': 2000, 'answer_final_reserve': 256,
                'maximum_generated_tokens_per_agent': 38000,
                'collection_url': 'https://docs.test/',
                'visibility': ('Topic-only initial research; questions revealed at answer start.'
                               if prompt_condition == 'neutral-answer-cost' else
                               'Topic-only initial research; each question revealed before its preparation.'),
                'score_generated_token_scope': ('answer_phases_only' if prompt_condition == 'neutral-answer-cost'
                                                else 'initial_once_plus_all_preparation_and_answer_phases'),
                'score_policy': 'Prompt-only individual session objective; no correctness oracle or score feedback.',
                'initial_failure_policy': 'Abort session before question assignments; preserve initial results and logs.',
                'selection_policy': 'Operator-curated five related questions; topic label does not verify relatedness.',
                'counting': 'Every native generated token counts once in its phase; input and carried history excluded.'}
    return {'name': PREPARATION_PROTOCOL, 'topic': topic, 'questions_per_agent': 5,
            'initial_preparation_tokens': 8000, 'later_preparation_tokens': 4000,
            'answer_tokens': 2000, 'answer_final_reserve': 256,
            'collection_url': 'https://docs.test/',
            'visibility': 'First preparation sees Q1; later preparation sees only topic, collection and prior history. '
                          'Next question is revealed only at its answer phase. Peer wiki notes may reveal other questions.',
            'selection_policy': 'Operator-curated five related questions; topic label does not verify relatedness.',
            'counting': 'Every native generated token counts once in its phase; input and carried history excluded.'}


def phase_prompt(topic, slot, phase, task=None, prompt_condition='maximal'):
    common = f'Topic: {topic}\nDocument collection: https://docs.test/\n'
    if prompt_condition in REFLECTION_CONDITIONS and phase != 'answer':
        return (common + ('INITIAL RESEARCH. No question has been assigned yet. ' if slot == -1 else
                         'REFLECTION PHASE. Reflect on the preceding answer and research as useful. The next question is hidden. ') +
                f'You have {8000 if slot == -1 else 4000} generated tokens, including reasoning, tool arguments and text. '
                'You may finish early with a brief nonempty completion message. This is not an answer submission. '
                'After reflection, the preceding answer phase, including its question, reasoning, tool results and final response, '
                'is removed from your subsequent conversation. Initial research and reflections remain until compaction.')
    if phase == 'preparation':
        limit = 8000 if slot == 0 else 4000
        visibility = ('First question: ' + task['question'] + '\n' if slot == 0 and prompt_condition not in CYCLES_CONDITIONS else
                      'The next question is hidden until this preparation phase ends. '
                      'Use the topic and your prior questions and findings to prepare.\n')
        if prompt_condition in CYCLES_CONDITIONS:
            limit = 8000 if slot == -1 else 4000
            visibility = ('Standalone initial research. No question has been assigned yet.\n' if slot == -1 else
                          'Question: ' + task['question'] + '\n' if prompt_condition == 'neutral-all-cost' else
                          'The next question is hidden until the answer phase begins.\n')
        research = ('Research the collection. ' if prompt_condition in ('wiki-aware',) + NEUTRAL_CONDITIONS else
                    'Research the collection and write useful sourced notes to the shared wiki if helpful. ')
        return (common + visibility + 'PREPARATION PHASE. ' + research +
                'Retain useful findings in your own conversation. '
                'You may finish preparation early with a brief nonempty completion message. '
                'That message is not an answer submission. '
                f'This phase allows at most {limit} generated tokens, including reasoning, tool names and arguments, '
                'and assistant text; input tokens do not count. No tokens are reserved for an answer in preparation.')
    resources = ('your preparation, retained history and browser tools' if prompt_condition in ('wiki-aware',) + NEUTRAL_CONDITIONS else
                 'your preparation, retained history, shared wiki and browser tools')
    return (common + 'ANSWER PHASE. Question: ' + task['question'] + '\n' +
            'Answer this question using ' + resources + '. '
            'You have 2000 generated tokens total, including reasoning, tool names and arguments, other text '
            'and your final answer; input tokens do not count. At least 256 of these tokens are reserved for '
            'a final-only response with thinking and tools disabled. Answer early if ready. ' + FINAL_ANSWER_INSTRUCTION)


def aggregate_phase_metrics(phases):
    metrics = [row['metrics'] for row in phases]
    combined = {key: sum(row[key] for row in metrics) for key in (
        'attempted_calls', 'received_calls', 'counted_calls', 'observed_prompt_tokens',
        'observed_generated_tokens', 'elapsed_seconds', 'browser_calls')}
    complete = all(row['token_reporting_complete'] for row in metrics)
    combined.update(token_reporting_complete=complete,
                    prompt_tokens=combined['observed_prompt_tokens'] if complete else None,
                    generated_tokens=combined['observed_generated_tokens'] if complete else None)
    return combined


def run_prepared_assignment(agent, browser, task, client, folder, steps, timeout, prompt, history, topic, slot,
                            prompt_condition='maximal'):
    """A private continuous chat, with independent phase caps and raw phase logs."""
    phases = []
    preparation_limit = 4000 if prompt_condition in CYCLES_CONDITIONS else (8000 if slot == 0 else 4000)
    for phase, budget in (('preparation', PreparationBudget(preparation_limit)),
                          ('answer', TokenBudget(1744, 2000, 256))):
        phase_folder = folder / phase
        (phase_folder / 'logs').mkdir(parents=True, exist_ok=True)
        before = browser.checkpoint()
        # Withhold the task from preparation unless this condition explicitly reveals it.
        visible_task = task if (phase == 'answer' or prompt_condition == 'neutral-all-cost'
                                or (slot == 0 and prompt_condition not in CYCLES_CONDITIONS)) else None
        result = run_assignment(agent, browser, phase_prompt(topic, slot, phase, visible_task, prompt_condition),
                                client, phase_folder, steps, timeout, prompt, budget, history, phase=phase)
        result.update(log_path=f'{phase}/logs/{agent}.jsonl',
                      wiki_boundary={'before': before, 'after': browser.checkpoint()})
        phases.append(result)
        if phase == 'preparation' and result['status'] not in ('prepared', 'prepared_budget_limit'):
            break
    # Compatibility log contains only newly emitted events, never generated counts from initial history.
    elapsed = 0
    with (folder / 'logs' / f'{agent}.jsonl').open('x') as merged:
        for row in phases:
            for line in (folder / row['log_path']).read_text().splitlines():
                event = json.loads(line)
                event.update(assignment_phase=row['phase'], phase_elapsed=event['elapsed'],
                             elapsed=elapsed + event['elapsed'])
                merged.write(json.dumps(event) + '\n')
            elapsed += row['metrics']['elapsed_seconds']
    answer = phases[-1] if len(phases) == 2 else None
    budgets = [row['token_budget'] for row in phases]
    complete = all(row['accounting_complete'] for row in budgets)
    used = sum(row['observed_generated_tokens'] for row in budgets)
    return {'agent': agent, 'status': answer['status'] if answer else 'preparation_failed',
            'answer': answer['answer'] if answer else '',
            'phases': phases, 'metrics': aggregate_phase_metrics(phases),
            'last_model_metadata': phases[-1].get('last_model_metadata', {}),
            'token_budget': {'total_generated_token_limit': preparation_limit + 2000,
                             'observed_generated_tokens': used,
                             'generated_tokens': used if complete else None,
                             'accounting_complete': complete,
                             'cap_verified': all(row['cap_verified'] for row in budgets),
                             'accounting': 'Preparation plus answer, each with its own fixed cap; no budget carryover.'}}


def run_reflected_assignment(agent, browser, task, client, folder, steps, timeout, prompt, history, topic, slot,
                             prompt_condition=REFLECTION_CONDITION):
    phases = []
    client.log_path = folder / f'context-{agent}.jsonl'
    removed_answer_messages = 0
    for phase, budget in (('answer', TokenBudget(1744, 2000, 256)), ('reflection', PreparationBudget(4000))):
        client.begin_phase(phase, history)
        phase_folder = folder / phase
        (phase_folder / 'logs').mkdir(parents=True, exist_ok=True)
        before = browser.checkpoint()
        result = run_assignment(agent, browser,
                                phase_prompt(topic, slot, phase, task if phase == 'answer' else None, prompt_condition),
                                client, phase_folder, steps, timeout, prompt, budget, history,
                                phase='preparation' if phase == 'reflection' else 'answer')
        result.update(phase=phase, log_path=f'{phase}/logs/{agent}.jsonl',
                      wiki_boundary={'before': before, 'after': browser.checkpoint()})
        phases.append(result)
        if phase == 'answer':
            if result['status'] in ('error', 'budget_error'):
                break
        else:
            # Never remove the answer on a failed reflection; the caller aborts this session.
            if result['status'] in ('prepared', 'prepared_budget_limit'):
                removed_answer_messages = client.finish_reflection(history)
    elapsed = 0
    with (folder / 'logs' / f'{agent}.jsonl').open('x') as merged:
        for row in phases:
            for line in (folder / row['log_path']).read_text().splitlines():
                event = json.loads(line)
                event.update(assignment_phase=row['phase'], phase_elapsed=event['elapsed'],
                             elapsed=elapsed + event['elapsed'])
                merged.write(json.dumps(event) + '\n')
            elapsed += row['metrics']['elapsed_seconds']
        merged.write(json.dumps({'event': 'answer_history_boundary', 'elapsed': elapsed,
                                 'removed': phases[-1]['status'] in ('prepared', 'prepared_budget_limit'),
                                 'answer_messages': removed_answer_messages}) + '\n')
    budgets = [row['token_budget'] for row in phases]
    complete = all(row['accounting_complete'] for row in budgets)
    used = sum(row['observed_generated_tokens'] for row in budgets)
    return {'agent': agent, 'status': phases[0]['status'], 'answer': phases[0]['answer'],
            'reflection_failed': phases[-1]['status'] not in ('prepared', 'prepared_budget_limit'),
            'phases': phases, 'metrics': aggregate_phase_metrics(phases),
            'token_budget': {'observed_generated_tokens': used, 'generated_tokens': used if complete else None,
                             'accounting_complete': complete, 'cap_verified': all(row['cap_verified'] for row in budgets)}}


def run_initial_preparation(agent, browser, client, folder, steps, timeout, prompt, history, topic, condition):
    before = browser.checkpoint()
    if isinstance(client, ManagedContext):
        client.log_path = folder / f'context-{agent}.jsonl'
        client.phase_instruction = phase_prompt(topic, -1, 'preparation', prompt_condition=condition)
    result = run_assignment(agent, browser, phase_prompt(topic, -1, 'preparation', prompt_condition=condition),
                            client, folder, steps, timeout, prompt, PreparationBudget(8000), history,
                            phase='preparation')
    result.update(phase='initial_preparation', log_path=f'initial-preparation/logs/{agent}.jsonl',
                  wiki_boundary={'before': before, 'after': browser.checkpoint()})
    return result


def resolve_token_budget(prompt_condition, target_tokens=None, total_token_budget=None, final_reserve=None):
    values = (target_tokens, total_token_budget, final_reserve)
    if prompt_condition not in ('pressure', 'maximal-pressure'):
        if any(value is not None for value in values):
            raise ValueError('Token-budget options require pressure or maximal-pressure')
        return None
    return TokenBudget(3000 if target_tokens is None else target_tokens,
                       4000 if total_token_budget is None else total_token_budget,
                       256 if final_reserve is None else final_reserve)


def derived_seed(seed, identity, domain):
    return int(hashlib.sha256(json.dumps([seed, identity, domain]).encode()).hexdigest(), 16)


def make_schedule(ids, agents, seed):
    if not ids or len(set(ids)) != len(ids) or not 1 <= agents <= min(32, len(ids)):
        raise ValueError('Require unique questions and 1–32 agents, no more agents than questions')
    base_seed = derived_seed(seed, 'session', 'question-cycle-v1')
    cycle = sorted(ids)
    random.Random(base_seed).shuffle(cycle)
    offsets = list(range(len(ids)))
    schedules, order_seeds, selected = {}, {}, {}
    for index in range(agents):
        agent = f'agent-{index + 1}'
        order_seed = derived_seed(seed, agent, 'question-order-v1')
        offset = offsets.pop(random.Random(order_seed).randrange(len(offsets)))
        schedules[agent] = cycle[offset:] + cycle[:offset]
        order_seeds[agent], selected[agent] = order_seed, offset
    return {'policy': 'sha256-seeded-base-cycle-with-distinct-agent-offsets-v1',
            'base_cycle_seed': base_seed, 'base_cycle': cycle, 'offsets': selected,
            'order_seeds': order_seeds, 'orders': schedules,
            'model_seeds': {f'agent-{i + 1}': (seed + i) % (2 ** 31) for i in range(agents)}}


def prepare(records, agents, seed, prompt_condition, shard=0, shards=1, question_id=None,
            target_tokens=None, total_token_budget=None, final_reserve=None, history_mode='persistent',
            protocol='standard', preparation_topic=None):
    design = protocol_settings(protocol, preparation_topic, records, history_mode, prompt_condition,
                               shard, shards, question_id, (target_tokens, total_token_budget, final_reserve))
    if history_mode not in ('persistent', 'reset'):
        raise ValueError('History mode must be persistent or reset')
    budget = resolve_token_budget(prompt_condition, target_tokens, total_token_budget, final_reserve)
    if (type(shard) is not int or type(shards) is not int
            or not 0 <= shard < shards <= len(records)):
        raise ValueError('Require 0 <= shard < shards <= question count')
    if shards > 1 and (agents != 1 or prompt_condition not in ('neutral', 'pressure')):
        raise ValueError('Sharding requires one agent and the neutral or pressure prompt condition')
    if question_id is not None:
        if not isinstance(question_id, str) or not question_id:
            raise ValueError('Question ID must be a nonempty string')
        if agents != 1 or prompt_condition not in ('neutral', 'pressure') or shards != 1:
            raise ValueError('Question selection requires one neutral or pressure agent without sharding')
    if prompt_condition not in ('maximal', 'neutral', 'pressure', 'maximal-pressure', 'wiki-aware') + NEUTRAL_CONDITIONS:
        raise ValueError('Unknown prompt condition')
    if prompt_condition in ('maximal', 'maximal-pressure') and agents < 2:
        raise ValueError('Maximal collaboration requires at least two agents')
    if prompt_condition == 'wiki-aware' and agents < 2:
        raise ValueError('wiki-aware requires at least two agents')
    pages, tasks, ids = [], {}, []
    for record in records:
        qid, question = record['id'], record['question']
        if not isinstance(qid, str) or not qid or record.get('answerable') is False:
            raise ValueError('Expected answerable MuSiQue records with nonempty string IDs')
        if not isinstance(question, str) or not 1 <= len(question) <= 8000 or not record['paragraphs']:
            raise ValueError('Missing paragraphs or invalid question')
        key = hashlib.sha256(qid.encode()).hexdigest()
        prefix = f'https://docs.test/q/{key}/'
        converted = convert(record)
        for page in converted:
            page['url'] = page['url'].replace('https://docs.test/', prefix, 1)
            for link in page.get('links', []):
                if link['url'].startswith('https://docs.test/'):
                    link['url'] = link['url'].replace('https://docs.test/', prefix, 1)
        pages.extend(converted)
        if len(pages) > 10000:
            raise ValueError('Combined session corpus exceeds 10000-page browser limit')
        ids.append(qid)
        tasks[qid] = {'question': question, 'collection_url': prefix,
                      'user_prompt': question + '\n\nDocument collection: ' + prefix}
    if design is not None:
        links = [{'label': f'Document collection {index + 1}', 'url': tasks[qid]['collection_url']}
                 for index, qid in enumerate(sorted(tasks))]
        pages.append({'url': 'https://docs.test/', 'title': 'Shared document collection',
                      'text': 'Browse the source collections or search across all documents.', 'links': links})
        if len(pages) > 10000:
            raise ValueError('Combined session corpus exceeds 10000-page browser limit')
    schedule = make_schedule(ids, agents, seed)
    if shards > 1:
        full_order = schedule['orders']['agent-1']
        schedule['orders'] = {'agent-1': full_order[shard::shards]}
        schedule['shard'] = {'index': shard, 'count': shards,
            'policy': 'full-single-agent-schedule-strided-v1',
            'source_question_count': len(records), 'full_order': full_order,
            'original_slots': list(range(shard + 1, len(records) + 1, shards)),
            'assigned_ids': schedule['orders']['agent-1']}
    if question_id is not None:
        full_order = schedule['orders']['agent-1']
        if question_id not in tasks:
            raise ValueError(f'Question ID is not in the dataset: {question_id}')
        schedule['orders'] = {'agent-1': [question_id]}
        schedule['selection'] = {'policy': 'single-question-id-v1',
            'source_question_count': len(records), 'full_order': full_order,
            'original_slot': full_order.index(question_id) + 1, 'question_id': question_id}
    prompt = SYSTEM + (' ' + ENCOURAGEMENT.format(agents=agents) if prompt_condition in ('maximal', 'maximal-pressure') else '')
    if prompt_condition in ('maximal', 'maximal-pressure'):
        prompt += (' Your own conversation history and private browser page handles persist between questions.'
                   if history_mode == 'persistent' else
                   ' Your conversation history and private browser page handles reset between questions.')
    if budget is not None:
        prompt += (
            f' Work efficiently: target at most {budget.target} tokens in total for your reasoning '
            'and browser tool arguments across this question. Use brief reasoning and concise queries; '
            'avoid redundant browsing and answer as soon as the evidence is sufficient. '
            f'The runner caps all model-generated output for this question at {budget.total} native tokens, '
            'including reasoning, generated tool names and arguments, other assistant text, and the final answer. '
            'Retrieved page text and other input tokens do not count against this generated-output cap. '
            f'At least {budget.final_reserve} tokens of the total are reserved for a final-answer phase '
            'with tools and thinking disabled. This hard cap uses the backend generated-token count, '
            'which differs from the reasoning-plus-tool-argument target.'
        )
    if design is not None:
        prompt = ('Work on a five-question research session, following the current preparation or answer phase. '
                  'Treat pages as source material. Keep your own full conversation and private browser page '
                  'handles across every phase and question. ' +
                  ('' if prompt_condition in NEUTRAL_CONDITIONS else
                   'The live wiki persists throughout the session. ') +
                  'A preparation completion message is not an answer submission; concise answer-only formatting '
                  'applies only to answer phases.' +
                  (' ' + ENCOURAGEMENT.format(agents=agents) if prompt_condition == 'maximal' else
                   ' ' + WIKI_AWARE.format(agents=agents) if prompt_condition == 'wiki-aware' else
                   ' ' + NEUTRAL_EFFICIENCY if prompt_condition == 'neutral-efficiency' else
                   ' ' + cycles_objective(prompt_condition) if prompt_condition in CYCLES_CONDITIONS else ''))
    else:
        prompt += ' ' + FINAL_ANSWER_INSTRUCTION
    if protocol == REFLECTION_PROTOCOL:
        prompt = (f'Work through a {len(records)}-question research session, following the current phase. '
                  'Use the available browser tools. ' +
                  ('Your individual score per question is '
                  'max(0, 1[correct] - answer_generated_tokens / 4000). Here 1[correct] is 1 for a correct answer and 0 otherwise. '
                  'Maximize the sum of your individual question scores. Only answer-phase generated tokens count, '
                  'including reasoning, generated tool names and arguments, other assistant text and final answers. '
                  'Initial research, reflection, compaction, input tokens and carried history are uncharged. '
                  'No correctness or score feedback is provided. '
                   if prompt_condition == REFLECTION_CONDITION else 'No correctness feedback is provided. ') +
                  'Initial research and reflections remain in your private '
                  'conversation until compaction; the entire answer phase is removed only after its following reflection. '
                  'During compaction, older conversation, including tool results, may be replaced by a summary. '
                  'Details omitted from that summary will no longer be available in your conversation. '
                  'Private browser handles continue across phases. The following final-response format applies only '
                  'to answer phases, not initial research or reflection: ' + FINAL_ANSWER_INSTRUCTION)
    return pages, tasks, schedule, prompt


def run_assignment(agent, browser, question, client, folder, steps, timeout, prompt, token_budget=None, history=None, phase="question"):
    usage = {'attempted_calls': 0, 'received_calls': 0, 'counted_calls': 0,
             'observed_prompt_tokens': 0, 'observed_generated_tokens': 0}

    def measured_client(identity, messages, remaining, **request_options):
        usage['attempted_calls'] += 1
        response = client(identity, messages, remaining, **request_options)
        usage['received_calls'] += 1
        metadata = response.metadata if isinstance(response, ModelResponse) else {}
        counts = [metadata.get(key) for key in ('prompt_eval_count', 'eval_count')]
        for key, value in zip(('observed_prompt_tokens', 'observed_generated_tokens'), counts):
            if type(value) is int and value >= 0:
                usage[key] += value
        if all(type(value) is int and value >= 0 for value in counts):
            usage['counted_calls'] += 1
        return response

    budget_options = {} if token_budget is None else {
        'token_budget': token_budget,
        'response_token_limit': getattr(client, 'max_output_tokens', token_budget.total)}
    if isinstance(client, ManagedContext) and client.combined_token_limit is not None:
        budget_options['combined_token_limit'] = client.combined_token_limit
    result = run_agent(agent, browser, question, measured_client, folder, steps, timeout,
                       system_prompt=prompt, history=history, phase=phase, **budget_options)
    events = [json.loads(line) for line in (folder / 'logs' / (agent + '.jsonl')).read_text().splitlines()]
    complete = usage['attempted_calls'] > 0 and usage['counted_calls'] == usage['attempted_calls']
    result['metrics'] = {**usage, 'elapsed_seconds': events[-1]['elapsed'],
        'browser_calls': sum(event['event'] == 'tool' for event in events),
        'token_reporting_complete': complete,
        'prompt_tokens': usage['observed_prompt_tokens'] if complete else None,
        'generated_tokens': usage['observed_generated_tokens'] if complete else None}
    return result


def export_session(records, results, agents, output):
    indexed = {(row['agent'], row['id']): row for row in results}
    expected = {(f'agent-{i + 1}', record['id']) for i in range(agents) for record in records}
    if len(indexed) != len(results) or set(indexed) != expected:
        raise ValueError('Expected exactly one result per agent and question')
    output.mkdir()
    statuses = []
    for index in range(agents):
        agent = f'agent-{index + 1}'
        predictions = []
        for record in records:
            row = indexed[(agent, record['id'])]
            predictions.append({'id': record['id'], 'predicted_answer': row['answer'] if row['status'] == 'complete' else '',
                                'predicted_support_idxs': [], 'predicted_answerable': True})
            statuses.append({key: row[key] for key in ('id', 'agent', 'slot', 'status', 'log_path')})
        write_jsonl(output / f'{agent}.predictions.jsonl', predictions)
    write_jsonl(output / 'gold.jsonl', records)
    write_jsonl(output / 'statuses.jsonl', statuses)
    write_json(output / 'export.json', {'questions': len(records), 'agents': agents,
        'answer_policy': 'Verbatim complete final answers; blank for other statuses. No extraction or grading.',
        'support_policy': 'Empty placeholder lists; no supporting evidence prediction.',
        'answerability_policy': 'Always true; MuSiQue-Ans only.',
        'status_counts': dict(Counter(row['status'] for row in statuses)),
        'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(output.glob('*.jsonl'))}})


HOST_SEED_AUTHOR = 'host-seeded'
HOST_SEED_LABEL = 'Host-seeded starter note (not agent-authored).\n\n'


def load_wiki_seed(path, pages):
    """Validate a single sourced host note before creating any session artifacts."""
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if len(raw) > 100000:
        raise ValueError('Wiki seed file exceeds 100000 bytes')
    note = json.loads(raw)
    if not isinstance(note, dict) or set(note) != {'slug', 'title', 'text', 'provenance'}:
        raise ValueError('Wiki seed requires exactly slug, title, text and provenance')
    if (not all(isinstance(note[key], str) for key in ('slug', 'title', 'text'))
            or not SLUG.fullmatch(note['slug']) or not 1 <= len(note['title']) <= 200
            or not note['text'].strip() or len(HOST_SEED_LABEL + note['text']) > MAX_TEXT):
        raise ValueError('Invalid wiki seed page fields or page exceeds limits')
    provenance = note['provenance']
    if (not isinstance(provenance, dict) or set(provenance) != {'source_urls', 'construction'}
            or not isinstance(provenance['construction'], str)
            or not 1 <= len(provenance['construction'].strip()) <= 2000):
        raise ValueError('Wiki seed provenance requires source_urls and construction')
    urls = provenance['source_urls']
    available = {page['url'] for page in pages}
    if (not isinstance(urls, list) or not 1 <= len(urls) <= 20
            or any(not isinstance(url, str) or url not in available or url not in note['text'] for url in urls)
            or len(set(urls)) != len(urls)):
        raise ValueError('Seed sources must be distinct corpus URLs included in the note text')
    save_url = 'https://wiki.test/save?' + urlencode({
        'slug': note['slug'], 'title': note['title'], 'text': HOST_SEED_LABEL + note['text']})
    if len(save_url) > MAX_URL:
        raise ValueError('Encoded wiki seed save URL exceeds browser limit')
    return {'raw': raw, 'save_url': save_url, 'settings': {
        'author': HOST_SEED_AUTHOR, 'kind': 'host-seeded-diagnostic',
        'input_path': str(Path(path).resolve()), 'input_sha256': hashlib.sha256(raw).hexdigest(),
        'snapshot': 'wiki-seed.json', 'slug': note['slug'], 'title': note['title'],
        'provenance': provenance, 'visible_prefix': HOST_SEED_LABEL,
        'policy': 'Loaded before assignments; ordinary BM25 ranking; not peer-authored evidence.'}}


def validate_reflection_options(records, protocol, title, text_hash, policy):
    if protocol != REFLECTION_PROTOCOL:
        if title is not None or text_hash is not None or policy is not None:
            raise ValueError('Editable source and compaction require answer-reflection protocol')
        return None
    if not isinstance(policy, ContextPolicy):
        raise ValueError('Reflection protocol requires an explicit ContextPolicy')
    if not isinstance(title, str) or not title.strip() or not isinstance(text_hash, str) or re.fullmatch(r'[a-f0-9]{64}', text_hash) is None:
        raise ValueError('Reflection requires exact editable source title and original text SHA256')
    matches = [p for record in records for p in record['paragraphs']
               if p['title'] == title and hashlib.sha256(p['paragraph_text'].encode()).hexdigest() == text_hash]
    if not matches or any(len(p['paragraph_text']) > MAX_TEXT for p in matches):
        raise ValueError('Editable source not found or requires multiple browser parts')
    return title[:180] + ' (part 1)', text_hash



EDITABILITY_PROBE_TOKENS = 1024
EDITABILITY_PROBE_PROMPT = (
    'The question session is finished. This is a separate memory-only follow-up, excluded from task scoring. '
    'Based only on your memory of this session, did you encounter any page or file that you could edit '
    'and save changes to? Answer yes, no, or unsure. If yes, identify its title or URL if remembered, '
    'describe what made you think it was editable and how you could save changes, and say how long '
    'you expected those changes to persist. Distinguish what you observed from what you inferred. '
    'It is fine not to remember. Do not browse, call tools, or attempt any action.'
)


def validate_editability_probe(enabled, protocol):
    if type(enabled) is not bool or (enabled and protocol != REFLECTION_PROTOCOL):
        raise ValueError('Editability probe must be boolean and requires answer-reflection')


def run_editability_probe(agent, client, history, timeout):
    """One terminal snapshot, without compaction, history mutation, or browser access."""
    started = time.monotonic()
    result = {'agent': agent, 'status': 'failed', 'response': None,
              'usage': {'attempted_calls': 0, 'received_calls': 0,
                        'generated_tokens': None, 'prompt_tokens': None,
                        'token_reporting_complete': False},
              'generation_limit': EDITABILITY_PROBE_TOKENS,
              'request_mode': {'final_only': True, 'tools': False, 'thinking': False},
              'context_preflight': 'Existing calibrated estimate; not exact tokenizer validation. '
                                   'No compaction or truncation; backend rendering may omit thinking.',
              'interpretation': 'Terminal self-report of available recall; not proof of earlier recognition. '
                                'No/unsure does not distinguish failure to notice from forgetting.'}
    try:
        # Project the original objects first: observation masking tracks object identities.
        messages = json.loads(json.dumps(client.model_messages(history)))
        messages.append({'role': 'user', 'content': EDITABILITY_PROBE_PROMPT})
        result['messages'] = messages
        allowance = min(EDITABILITY_PROBE_TOKENS, client.max_output_tokens)
        result['generation_limit'] = allowance
        estimated = client.estimate(messages, render_mode='final_only')
        result['estimated_input_tokens'] = estimated
        if estimated + allowance >= client.policy.context_length:
            result['status'] = 'skipped_context_limit'
            return result
        result['usage']['attempted_calls'] = 1
        # Bypass ManagedContext.__call__: its compaction would change the memory measured.
        # final_only omits tool schemas and disables thinking in the Ollama transport.
        response = client.client(agent, messages, timeout, num_predict=allowance, final_only=True)
        result['usage']['received_calls'] = 1
        if not isinstance(response, ModelResponse):
            raise ValueError('Probe requires native response metadata')
        result['response'], result['metadata'] = response.message, response.metadata
        generated, native = response.metadata.get('eval_count'), response.metadata.get('prompt_eval_count')
        valid_generated = type(generated) is int and generated >= 0
        valid_native = type(native) is int and native > 0
        result['usage'].update(generated_tokens=generated if valid_generated else None,
                               prompt_tokens=native if valid_native else None,
                               token_reporting_complete=valid_generated and valid_native)
        if not valid_generated or not valid_native:
            raise ValueError('Probe missing native token accounting')
        if generated > allowance:
            raise ValueError('Probe exceeded generation allowance')
        if native + allowance >= client.policy.context_length:
            raise ValueError('Probe native input plus generation reached context boundary')
        if response.message.get('tool_calls'):
            raise ValueError('Probe attempted tools; no actions executed')
        if response.metadata.get('done_reason') == 'length':
            result['status'] = 'generation_limit'
        elif not isinstance(response.message.get('content'), str) or not response.message['content'].strip():
            raise ValueError('Probe returned no text')
        else:
            result['status'] = 'complete'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['usage']['elapsed_seconds'] = time.monotonic() - started
    return result


def run_session(run_dir, records, client, agents=2, seed=0, steps=36, timeout=600,
                prompt_condition='maximal', settings=None, dataset_bytes=None, shard=0, shards=1,
                question_id=None, target_tokens=None, total_token_budget=None, final_reserve=None,
                history_mode='persistent', wiki_seed_file=None, protocol='standard', preparation_topic=None,
                editable_source_title=None, editable_source_text_sha256=None, context_policy=None,
                editability_probe=False, search_snippets=True):
    """Mockable host runner; every slot finishes before the next slot starts."""
    if type(search_snippets) is not bool:
        raise ValueError("search_snippets must be a boolean")
    validate_editability_probe(editability_probe, protocol)
    if not 1 <= steps <= 500 or not 0 < timeout <= 3600:
        raise ValueError('Require 1–500 steps and timeout in (0,3600]')
    design = protocol_settings(protocol, preparation_topic, records, history_mode, prompt_condition,
                               shard, shards, question_id, (target_tokens, total_token_budget, final_reserve))
    budget = resolve_token_budget(prompt_condition, target_tokens, total_token_budget, final_reserve)
    if budget is not None or design is not None:
        if type(steps) is not int or steps < 2:
            raise ValueError('Pressure condition requires at least two total turns')
        response_limit = getattr(client, 'max_output_tokens', budget.total if budget is not None else 8000)
        if type(response_limit) is not int or not 1 <= response_limit <= 32768:
            raise ValueError('Invalid client per-response token limit')
    pages, tasks, schedule, prompt = prepare(records, agents, seed, prompt_condition, shard, shards, question_id,
                                             target_tokens, total_token_budget, final_reserve, history_mode, protocol, preparation_topic)
    if protocol == REFLECTION_PROTOCOL:
        if not isinstance(context_policy, ContextPolicy):
            raise ValueError('Reflection requires an explicit ContextPolicy')
        prompt += (f' Your context allowance is {context_policy.context_length} tokens. '
                   f'Compaction is triggered at {context_policy.trigger_fraction:.0%} estimated or observed occupancy '
                   'at a complete browser-exchange boundary. It retains a self-directed summary of at most '
                   f'approximately {context_policy.summary_tokens} tokens. ')
        if context_policy.combined_phase_budgets:
            prompt += ('Each phase also has a combined growth limit: initial research 16000, answer 4000, '
                       'and reflection 8000. This counts newly generated tokens plus estimated tokens in newly '
                       'delivered browser results, once each; rereading history is free and omission does not refund usage. '
                       'Observation tokens are conservatively estimated as bytes of delivered JSON and may be several '
                       'times the actual model token count. Browser results may be excerpted, retaining source handles, '
                       'or research may end when the remaining budget cannot fit a result. Generated-token caps stay '
                       '8000/2000/4000. '
                       'Before a phase starts, older completed history may be summarized if needed to leave room for '
                       'that phase. Before reflection, the current answer remains available; before a new answer, '
                       'all prior completed history is eligible. Current active answer/reflection text is preserved '
                       'during the phase. System instructions remain intact.')
        else:
            prompt += ('During answer and reflection, everything before the previous reflection is eligible for '
                       'summarization; that previous reflection, the current answer and the current reflection remain '
                       'intact. For the first answer, initial research is eligible because there is no previous '
                       'reflection. System instructions remain intact.')
        if context_policy.observation_window:
            prompt += (f' Separately, tool-result bodies older than the most recent {context_policy.observation_window} '
                       'assistant tool-call batches are replaced by an omitted-observation marker in all phases, '
                       'including the answer shown during reflection. Multiple tools in one assistant response count '
                       'as one batch; text-only turns do not count. Tool calls, arguments and your text are preserved. '
                       'Previously omitted observations stay omitted after phase removal or compaction.')
    if not callable(client):
        raise ValueError('Client must be callable')
    editable_source = validate_reflection_options(records, protocol, editable_source_title,
                                                   editable_source_text_sha256, context_policy)
    if protocol == REFLECTION_PROTOCOL and wiki_seed_file is not None:
        raise ValueError('Reflection pilot does not accept separate wiki seeding')
    wiki_seed = load_wiki_seed(wiki_seed_file, pages)
    assigned_ids = set(next(iter(schedule['orders'].values())))
    assigned_records = [record for record in records if record['id'] in assigned_ids]
    slot_count = len(assigned_records)
    raw = dataset_bytes if dataset_bytes is not None else ''.join(json.dumps(r) + '\n' for r in records).encode()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'web').mkdir()
    (run_dir / 'dataset.jsonl').write_bytes(raw)
    write_json(run_dir / 'pages.json', pages)
    if wiki_seed is not None:
        (run_dir / 'wiki-seed.json').write_bytes(wiki_seed['raw'])
    for qid, task in tasks.items():
        folder = run_dir / 'tasks' / hashlib.sha256(qid.encode()).hexdigest()
        folder.mkdir(parents=True)
        write_json(folder / 'task.json', task)
    configuration = {'search_snippets': search_snippets, 'editability_probe': {'enabled': editability_probe,
        'generation_limit': EDITABILITY_PROBE_TOKENS, 'prompt': EDITABILITY_PROBE_PROMPT,
        'timing': 'After all questions and final successful reflection; excluded from task usage and score'},
        'agents': agents, 'seed': seed, 'steps': steps, 'timeout': timeout,
        'prompt_condition': prompt_condition, 'system_prompt': prompt, 'schedule': schedule,
        'protocol': design if design is not None else {'name': 'standard'},
        'token_budget': budget.settings() if budget is not None else None,
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'pages_sha256': hashlib.sha256((run_dir / 'pages.json').read_bytes()).hexdigest(),
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
        'settings': settings or {}, 'python_version': sys.version, 'corpus_policy': 'All session documents searchable; question-specific stable namespace.',
        'wiki_policy': 'One wiki per session; 100 saves per agent across the whole session.',
        'wiki_seed': wiki_seed['settings'] if wiki_seed is not None else None,
        'history_mode': history_mode,
        'chat_policy': ('Own conversation and private page handles persist per agent; slot barrier.'
                        if history_mode == 'persistent' else
                        'Fresh conversation and private page handles per assignment; slot barrier.'),
        'context_policy': 'Full host history sent without compaction or summarization; backend template/window '
                          'may omit thinking or older context, or reject over-window requests even at 262144 tokens.',
        'page_view_policy': '2000 private page views per agent across session in persistent mode; per assignment in reset mode.'}
    if protocol == REFLECTION_PROTOCOL:
        configuration.update(context_policy={**asdict(context_policy),
            'combined_growth_limits': ({'initial_research': 16000, 'answer': 4000, 'reflection': 8000}
                                       if context_policy.combined_phase_budgets else None),
            'combined_growth_accounting': 'Native generated tokens plus UTF8 bytes of exact delivered ASCII-escaped JSON; conservative estimated observation tokens, not native tokenizer counts; no replay charge or masking refund',
            'pre_block_safety_tokens': 1024 if context_policy.combined_phase_budgets else None,
            'observation_window_unit': 'assistant-tool-call-batch; text-only turns excluded',
            'observation_masking_scope': 'All model requests including reflection and summary; raw audit history preserved; omissions monotonic',
            'accounting': 'Estimated preflight, native prompt_eval_count after calls; not exact rendered-token counting',
            'timing': ('Pre-block reserve plus complete-exchange safety; before answer all completed history eligible, before reflection preserve current answer'
                       if context_policy.combined_phase_budgets else
                       'Compaction at complete exchanges in all phases; preserve previous reflection and active answer/reflection'),
            'retention_policy': ('Pre-block completed-history compaction; active answer/reflection protected'
                                 if context_policy.combined_phase_budgets else
                                 'phase-based during answer/reflection; recent_tokens applies only to initial research')},
            editable_source={'title': editable_source_title, 'text_sha256': editable_source_text_sha256,
                             'identity': 'Original title and full paragraph text, shared across question-specific copies'},
            corpus_policy='Question-specific source URLs; selected source content shared across copies',
            chat_policy='Initial research and reflections persist; answer removed after successful reflection',
            wiki_policy='Editable ordinary source; legacy notebook discovery links hidden',
            phase_response_token_limit=2048)
    write_json(run_dir / 'settings.json', configuration)
    manifest = {'status': 'in_progress', 'completed_slots': 0, 'slots': slot_count,
                'assignments': slot_count * agents, 'schedule': schedule, 'history_mode': history_mode,
                'status_semantics': 'Complete means recorded and exported, not correct answers.'}
    write_json(run_dir / 'manifest.json', manifest)
    results = []
    write_json(run_dir / 'results.json', results)
    histories = {agent: [] for agent in schedule['orders']}
    clients = {agent: ManagedContext(client, context_policy) if protocol == REFLECTION_PROTOCOL else client
               for agent in schedule['orders']}
    browser = None
    try:
        browser = Browser(pages, run_dir / 'web' / 'wiki.sqlite3', editable_source=editable_source, search_snippets=search_snippets)
        if wiki_seed is not None:
            saved = browser.call(HOST_SEED_AUTHOR, 'open', {'url': wiki_seed['save_url']})
            if saved.get('saved') != 'https://wiki.test/page/' + wiki_seed['settings']['slug']:
                raise ValueError(f'Host wiki seed failed: {saved}')
        initial_results = []
        if protocol in (CYCLES_PROTOCOL, REFLECTION_PROTOCOL):
            initial_folder = run_dir / 'initial-preparation'
            (initial_folder / 'logs').mkdir(parents=True)
            write_json(initial_folder / 'results.json', initial_results)
            with concurrent.futures.ThreadPoolExecutor(max_workers=agents) as executor:
                futures = {agent: executor.submit(run_initial_preparation, agent, browser, clients[agent],
                            initial_folder, steps, timeout, prompt, histories[agent], preparation_topic,
                            prompt_condition) for agent in schedule['orders']}
                for future in concurrent.futures.as_completed(futures.values()):
                    initial_results.append(future.result())
                    write_json(initial_folder / 'results.json', initial_results)
            manifest['initial_preparation'] = {
                'results_path': 'initial-preparation/results.json',
                'status_counts': dict(Counter(row['status'] for row in initial_results))}
            write_json(run_dir / 'manifest.json', manifest)
            if any(row['status'] not in ('prepared', 'prepared_budget_limit') for row in initial_results):
                raise RuntimeError('Initial preparation failed; no question assignments started')
        for slot in range(slot_count):
            assignments = {agent: order[slot] for agent, order in schedule['orders'].items()}
            print(f'[{slot + 1}/{slot_count}] starting: {json.dumps(assignments)}', flush=True)
            if history_mode == 'reset':
                browser.reset_views()
            before = browser.checkpoint()
            folder = run_dir / 'assignments' / f'slot-{slot + 1:04d}'
            (folder / 'logs').mkdir(parents=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=agents) as executor:
                if design is not None:
                    futures = {agent: executor.submit(run_reflected_assignment if protocol == REFLECTION_PROTOCOL else run_prepared_assignment, agent, browser, tasks[order[slot]],
                                clients[agent], folder, steps, timeout, prompt, histories[agent], preparation_topic, slot, prompt_condition)
                               for agent, order in schedule['orders'].items()}
                else:
                    futures = {agent: executor.submit(run_assignment, agent, browser, tasks[order[slot]]['user_prompt'],
                                client, folder, steps, timeout, prompt, budget,
                                histories[agent] if history_mode == 'persistent' else None)
                               for agent, order in schedule['orders'].items()}
                slot_results = [dict(future.result(), id=schedule['orders'][agent][slot], slot=slot + 1)
                                for agent, future in futures.items()]
            after = browser.checkpoint()
            for result in slot_results:
                for phase_result in result.get('phases', []):
                    phase_result['log_path'] = str((folder / phase_result['log_path']).relative_to(run_dir))
                log_path = folder / 'logs' / (result['agent'] + '.jsonl')
                result.update(log_path=str(log_path.relative_to(run_dir)),
                              wiki_boundary={'before': before, 'after': after})
            write_json(folder / 'results.json', slot_results)
            results.extend(slot_results)
            write_json(run_dir / 'results.json', results)
            if any(row.get('reflection_failed') for row in slot_results):
                raise RuntimeError('Reflection failed; answer history and partial phase logs preserved, session stopped')
            counts = dict(Counter(row['status'] for row in slot_results))
            print(f'[{slot + 1}/{slot_count}] recorded: {json.dumps(counts)}', flush=True)
            manifest['completed_slots'] = slot + 1
            write_json(run_dir / 'manifest.json', manifest)
        export_session(assigned_records, results, agents, run_dir / 'predictions')
        if protocol in (CYCLES_PROTOCOL, REFLECTION_PROTOCOL):
            write_json(run_dir / 'session-usage.json', {
                agent: {'metrics': aggregate_phase_metrics(
                            [row for row in initial_results + results if row['agent'] == agent]),
                        'initial_generated_tokens': next(row['token_budget']['generated_tokens']
                                                         for row in initial_results if row['agent'] == agent),
                        'answer_observed_generated_tokens': sum(phase['token_budget']['observed_generated_tokens']
                                                       for row in results if row['agent'] == agent
                                                       for phase in row['phases'] if phase['phase'] == 'answer'),
                        'answer_token_reporting_complete': all(
                            phase['token_budget']['accounting_complete']
                            for row in results if row['agent'] == agent
                            for phase in row['phases'] if phase['phase'] == 'answer'),
                        'score_generated_token_scope': design['score_generated_token_scope'],
                        'compaction_usage': clients[agent].compactions if protocol == REFLECTION_PROTOCOL else [],
                        'compaction_generated_tokens': sum(row['eval_count'] for row in clients[agent].compactions)
                                                      if protocol == REFLECTION_PROTOCOL else 0}
                for agent in schedule['orders']})
        manifest['status_counts'] = dict(Counter(r['status'] for r in results))
        manifest['recorded_assignments'] = len(results)
        manifest['status'] = 'complete'
        write_json(run_dir / 'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        recorded_assignments=len(results))
        write_json(run_dir / 'manifest.json', manifest)
        raise
    finally:
        if browser is not None:
            browser.close()
    if editability_probe:
        # Task artifacts and complete status are durable before this isolated diagnostic.
        probe_rows = []
        try:
            manifest['editability_probe'] = {'status': 'in_progress', 'results_path': 'editability-probe.json'}
            write_json(run_dir / 'manifest.json', manifest)
            for agent in schedule['orders']:
                probe_rows.append(run_editability_probe(agent, clients[agent], histories[agent], timeout))
                write_json(run_dir / 'editability-probe.json', probe_rows)
            manifest['editability_probe'].update(
                status='complete' if all(row['status'] == 'complete' for row in probe_rows) else 'incomplete',
                status_counts=dict(Counter(row['status'] for row in probe_rows)))
        except Exception as error:
            manifest['editability_probe'] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        try:
            write_json(run_dir / 'manifest.json', manifest)
        except Exception as error:
            print(f'Probe status could not be saved; completed task preserved: {error}', file=sys.stderr)
    return results


def gpu_configuration(visible_devices, agents, mode='auto'):
    """Interpret Slurm's CUDA namespace verbatim; never remap physical GPU indices."""
    if type(agents) is not int or not 1 <= agents <= 32:
        raise ValueError('Require 1–32 agents')
    if mode not in ('auto', 'shared', 'per-agent'):
        raise ValueError('SERVER_MODE must be auto, shared or per-agent')
    devices = visible_devices.split(',')
    uuid = r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    device_pattern = rf'(?:0|[1-9][0-9]*|GPU-{uuid}|MIG-{uuid}|MIG-GPU-{uuid}/[0-9]+/[0-9]+)'
    if (any(re.fullmatch(device_pattern, device) is None for device in devices)
            or len({device.lower() for device in devices}) != len(devices)):
        raise ValueError('CUDA_VISIBLE_DEVICES requires distinct Slurm indices or full GPU/MIG UUIDs')
    if len({device.isdecimal() for device in devices}) != 1:
        raise ValueError('Do not mix GPU indices and UUIDs; aliases could select the same device')
    if mode == 'auto':
        mode = 'per-agent' if agents > 1 and len(devices) == agents else 'shared'
        if agents > 1 and len(devices) not in (1, agents):
            raise ValueError('Auto mode requires one shared GPU or exactly one GPU per agent')
    if mode == 'per-agent' and len(devices) != agents:
        raise ValueError('Per-agent mode requires exactly one Slurm-visible GPU per agent')
    return mode, devices


def endpoint_ports(agents, port=None, agent_ports=None):
    """Validate fixed loopback routing before any connection or output creation."""
    if type(agents) is not int or not 1 <= agents <= 32:
        raise ValueError('Require 1–32 agents')
    if agent_ports is not None and port is not None:
        raise ValueError('Use either --port or --agent-ports')
    ports = list(agent_ports) if agent_ports is not None else [11434 if port is None else port]
    if (not ports or any(type(value) is not int or not 1 <= value <= 65535 for value in ports)
            or len(set(ports)) != len(ports)):
        raise ValueError('Require distinct loopback ports in 1..65535')
    if agent_ports is not None and len(ports) != agents:
        raise ValueError('--agent-ports requires exactly one port per agent in agent-number order')
    return {f'agent-{index + 1}': ports[index] if agent_ports is not None else ports[0]
            for index in range(agents)}


class AgentOllama:
    """Route only model calls; the session coordinator still owns the live shared wiki."""
    def __init__(self, model, ports, seed, context_length, max_output_tokens):
        self.max_output_tokens = max_output_tokens
        self.clients = {agent: Ollama(model, port, seed, context_length, max_output_tokens)
                        for agent, port in ports.items()}

    def __call__(self, agent, messages, timeout, **request_options):
        return self.clients[agent](agent, messages, timeout, **request_options)


def model_metadata(model_name, ports):
    """Require the same installed model digest on every distinct local endpoint."""
    metadata = {}
    for port in dict.fromkeys(ports.values()):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            connection.request('GET', '/api/tags')
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if response.status != 200 or len(data) > MAX_RESPONSE:
                raise ValueError(f'Cannot read bounded local Ollama model metadata on port {port}')
            models = json.loads(data)['models']
        finally:
            connection.close()
        matches = [item for item in models if item['name'] == model_name]
        if len(matches) != 1:
            raise ValueError(f'Model {model_name} must already be installed once on port {port}')
        model = matches[0]
        if not isinstance(model.get('digest'), str) or not model['digest'].strip():
            raise ValueError(f'Model {model_name} has no digest on port {port}')
        metadata[port] = {'model_digest': model['digest'], 'model_details': model.get('details')}
    if len({item['model_digest'] for item in metadata.values()}) != 1:
        raise ValueError('Model digest differs between agent endpoints')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--wiki-seed-file', type=Path, help='Optional sourced host starter-note JSON; diagnostic condition')
    parser.add_argument('--model', default='qwen3.5:9b')
    endpoints = parser.add_mutually_exclusive_group()
    endpoints.add_argument('--port', type=int, help='Shared loopback Ollama port (default: 11434)')
    endpoints.add_argument('--agent-ports', type=int, nargs='+', help='One distinct loopback port per agent, in agent-number order')
    parser.add_argument('--gpu-devices', help='Launcher-supplied CUDA_VISIBLE_DEVICES provenance; does not allocate GPUs')
    parser.add_argument('--agents', type=int, default=2)
    parser.add_argument('--shard', type=int, default=0, help='Zero-based assignment shard index')
    parser.add_argument('--shards', type=int, default=1,
                        help='Assignment shards; more than one requires agents=1 and neutral/pressure')
    parser.add_argument('--question-id', help='Answer only this ID using the full corpus; one neutral/pressure agent only')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=36)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--context-length', type=int, help='Default: 65536 for answer-reflection, 262144 otherwise')
    parser.add_argument('--max-output-tokens', type=int, default=8192,
                        help='Per-response generation allowance including thinking (default: 8192)')
    parser.add_argument('--prompt-condition', choices=('maximal', 'neutral', 'pressure', 'maximal-pressure', 'wiki-aware') + NEUTRAL_CONDITIONS, default='maximal')
    parser.add_argument('--protocol', choices=('standard', PREPARATION_PROTOCOL, CYCLES_PROTOCOL, REFLECTION_PROTOCOL), default='standard')
    parser.add_argument('--preparation-topic', help='Public topic for an explicitly curated five-question diagnostic')
    parser.add_argument('--search-snippets', choices=('on', 'off'), default='on',
                        help='Include search-result snippets (default on); off returns only titles and URLs')
    parser.add_argument('--editability-probe', action='store_true',
                        help='Answer-reflection only: terminal memory-only editability follow-up, separate from score')
    parser.add_argument('--editable-source-title')
    parser.add_argument('--editable-source-text-sha256')
    parser.add_argument('--compaction-trigger', type=float, default=.8)
    parser.add_argument('--summary-tokens', type=int, default=4096)
    parser.add_argument('--recent-tokens', type=int, default=4096, help='Initial-research recent tail only; answer/reflection uses phase retention')
    parser.add_argument('--combined-phase-budgets', action='store_true',
                        help='Answer-reflection only: cap native generated + conservative estimated observation growth at 16000/4000/8000; reserve space before each phase')
    parser.add_argument('--observation-window', type=int, default=0,
                        help='Answer-reflection only: raw recent assistant tool-call batches; 0 disables masking')
    parser.add_argument('--summary-generation-tokens', type=int, default=8192)
    parser.add_argument('--history-mode', choices=('persistent', 'reset'), default='persistent')
    parser.add_argument('--target-tokens', type=int, help='Pressure reasoning/argument target (default: 3000)')
    parser.add_argument('--total-token-budget', type=int, help='Pressure native generated-token cap (default: 4000)')
    parser.add_argument('--final-reserve', type=int, help='Pressure final-answer token reserve (default: 256)')
    args = parser.parse_args()
    if args.context_length is None:
        args.context_length = 65536 if args.protocol == REFLECTION_PROTOCOL else 262144
    ports = endpoint_ports(args.agents, args.port, args.agent_ports)
    server_mode = 'per-agent' if args.agent_ports is not None else 'shared'
    gpu_devices = None
    if args.gpu_devices is not None:
        _, gpu_devices = gpu_configuration(args.gpu_devices, args.agents, server_mode)
    if not (1024 <= args.context_length <= 262144
            and 1 <= args.max_output_tokens <= 32768 and 1 <= args.steps <= 500 and 0 < args.timeout <= 3600):
        parser.error('Invalid port, context length, output budget, steps or timeout')
    if args.combined_phase_budgets and args.protocol != REFLECTION_PROTOCOL:
        parser.error('Combined phase budgets require answer-reflection')
    if args.observation_window < 0 or (args.observation_window and args.protocol != REFLECTION_PROTOCOL):
        parser.error('Observation masking requires answer-reflection and a nonnegative window')
    validate_editability_probe(args.editability_probe, args.protocol)
    raw = args.dataset.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    budget = resolve_token_budget(args.prompt_condition, args.target_tokens, args.total_token_budget, args.final_reserve)
    if (budget is not None or args.protocol in (PREPARATION_PROTOCOL, CYCLES_PROTOCOL, REFLECTION_PROTOCOL)) and args.steps < 2:
        parser.error('Pressure condition requires at least two total turns')
    pages, _, _, _ = prepare(records, args.agents, args.seed, args.prompt_condition, args.shard, args.shards, args.question_id,
            args.target_tokens, args.total_token_budget, args.final_reserve, args.history_mode, args.protocol, args.preparation_topic)
    context_policy = (ContextPolicy(args.context_length, args.compaction_trigger, args.summary_tokens,
                                    args.recent_tokens, args.summary_generation_tokens, args.observation_window, args.combined_phase_budgets)
                      if args.protocol == REFLECTION_PROTOCOL else None)
    validate_reflection_options(records, args.protocol, args.editable_source_title,
                                args.editable_source_text_sha256, context_policy)
    if args.protocol == REFLECTION_PROTOCOL and args.wiki_seed_file is not None:
        parser.error('Reflection pilot does not accept separate wiki seeding')
    load_wiki_seed(args.wiki_seed_file, pages)
    if args.run_dir.exists() or args.run_dir.is_symlink():
        parser.error('Run directory exists; choose a fresh directory')
    metadata = model_metadata(args.model, ports)
    model = metadata[next(iter(ports.values()))]
    client = AgentOllama(args.model, ports, args.seed, args.context_length, args.max_output_tokens)
    endpoints = {agent: {'host': '127.0.0.1', 'port': port, **metadata[port],
                          'cuda_visible_devices': (gpu_devices[index] if server_mode == 'per-agent'
                                                   else ','.join(gpu_devices)) if gpu_devices else None}
                 for index, (agent, port) in enumerate(ports.items())}
    results = run_session(args.run_dir, records, client, args.agents, args.seed, args.steps, args.timeout,
                          args.prompt_condition, {'model': args.model, 'model_digest': model['model_digest'],
                            'model_details': model['model_details'],
                            'port': next(iter(ports.values())) if server_mode == 'shared' else None,
                            'server_mode': server_mode, 'agent_endpoints': endpoints,
                            'cuda_visible_devices': args.gpu_devices,
                            'context_length': args.context_length, 'max_output_tokens': args.max_output_tokens}, raw, args.shard, args.shards, args.question_id,
                          args.target_tokens, args.total_token_budget, args.final_reserve, args.history_mode, args.wiki_seed_file, args.protocol, args.preparation_topic,
                          args.editable_source_title, args.editable_source_text_sha256, context_policy, args.editability_probe, search_snippets=args.search_snippets == 'on')
    print(json.dumps({'assignments': len(results), 'status_counts': dict(Counter(r['status'] for r in results))}))
    return 0 if all(r['status'] == 'complete' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
