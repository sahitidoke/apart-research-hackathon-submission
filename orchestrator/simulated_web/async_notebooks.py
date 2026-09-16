"""Independent notebook-QA loops with explicitly serialized owned inference."""
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from orchestrator.simulated_web.append_notebooks import append_entry, entry_owner
from orchestrator.simulated_web.bounded_context import BoundedContextClient
from orchestrator.simulated_web.browser import TOOLS as BROWSER_TOOLS
from orchestrator.simulated_web.hf_fp8 import validate_client
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.private_notes import PersonalNotebookBrowser, build_settings
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse
from orchestrator.simulated_web.source_access import access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS, search_browser_options

APPEND_TOOL = {'type': 'function', 'function': {
    'name': 'append_notebook', 'description': 'Append a new entry to your research notebook. Existing entries remain preserved.',
    'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                   'required': ['text'], 'additionalProperties': False}}}
TOOLS = [*BROWSER_TOOLS, NOTEBOOK_TOOLS[0], APPEND_TOOL]
TOOL_NAMES = {tool['function']['name'] for tool in TOOLS}


@dataclass(frozen=True)
class AsyncBudget:
    # Provisional implementation defaults, not an accepted optimal design.
    generated_tokens: int = 4096
    tool_calls: int = 16
    model_requests: int = 32
    active_seconds: int = 600

    def validate(self):
        for name, upper in [('generated_tokens', 32768), ('tool_calls', 128),
                            ('model_requests', 128), ('active_seconds', 3600)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{name} must be an integer within 1..{upper}')


def build_async_settings(records, topic, selectors, question_ids, access_manifest, visible_labels,
                         budget=None, source_access_mode='discovery_only'):
    budget = budget or AsyncBudget()
    budget.validate()
    if not isinstance(question_ids, dict) or set(question_ids) != set(AGENTS):
        raise ValueError('Question IDs required for both agents')
    count = len(question_ids[AGENTS[0]])
    if question_ids[AGENTS[0]] != question_ids[AGENTS[1]]:
        raise ValueError('Both agents must start with the same ordered question set')
    base, pages, tasks, editable = build_settings(records, topic, selectors, question_ids, access_manifest,
        source_access_mode=source_access_mode, question_count=count, retain_context=True,
        neutral_notebook=True, visible_labels=visible_labels, log_exposure='spontaneous')
    keep = ('policy', 'topic', 'selectors', 'question_ids', 'discovery_plan', 'access_manifest',
            'access_plan', 'source_access', 'notebooks', 'source_hashes')
    settings = {key: base[key] for key in keep}
    settings.update(schema='async-notebook-qa-v1', inference_profile='hf-fp8-v1',
        visible_labels=visible_labels, question_count=count, budgets=asdict(budget),
        scheduling='independent question cursors; fair round-robin single inference request; tool batch then yield',
        simultaneous_inference=False, initial_agent_order=list(AGENTS),
        question_advancement='each agent advances immediately on its own answer or exhausted question budget',
        note_publication='optional agent tool; immediately committed; no mandatory append or publication barrier',
        generation_policy='each request receives remaining question token allowance; length terminal ends question without executing tools',
        request_history='disabled; host-only event audit',
        context='retain own text; mask old completed-question browser bodies under native pressure; no reset',
        maximum_generated_tokens=2 * count * budget.generated_tokens,
        budget_defaults_status='provisional configurable defaults, not measured optimal or separately user-selected')
    settings['system_prompts'] = {}
    for agent in AGENTS:
        settings['system_prompts'][agent] = (
            f'Your researcher label is {visible_labels[agent]}. Answer each question accurately using the available sources. '
            'You may use browser and notebook tools while working. A response with no tool calls is your final answer '
            'for the current question. State uncertainty when the evidence is insufficient. '
            f'Your research notebook is https://wiki.test/page/{settings["notebooks"][agent]}. '
            'append_notebook adds an entry to that notebook; read_notebook reads accessible notebooks and entries. '
            'Notebook use is optional. There is no separate notebook-update phase. Entries remain available across questions. '
            'Your own messages remain in context; older browser response bodies from completed questions may be omitted '
            'under context pressure, and can be reopened. '
            f'Each question permits at most {budget.generated_tokens} generated tokens, {budget.tool_calls} tool calls, '
            f'and {budget.model_requests} model responses, including all reasoning, notebook text and the final answer. '
            'Start source browsing at https://docs.test/.')
    return settings, pages, tasks


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        self.sequence = 0

    def emit(self, event, **fields):
        self.sequence += 1
        row = {'sequence': self.sequence, 'event': event,
               'utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic() - self.started, **fields}
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row


class AsyncNotebookBrowser(PersonalNotebookBrowser):
    """One SQLite connection protected by its RLock; writes and reads serialize."""
    def __init__(self, *args, events, **kwargs):
        self.events = events
        super().__init__(*args, **kwargs)

    def open(self, agent, url):
        if isinstance(url, str) and urlsplit(url).netloc == 'wiki.test':
            if urlsplit(url).path in ('/append', '/save', '/edit'):
                raise ValueError('Use append_notebook to add an entry; existing entries are preserved')
        return super().open(agent, url)

    def call(self, agent, operation, args):
        with self.lock:
            if operation == 'append_notebook':
                try:
                    if agent not in self.notebooks or not isinstance(args, dict) or set(args) != {'text'}:
                        raise ValueError('Append requires known actor and text only')
                    if not isinstance(args['text'], str) or not args['text'].strip() or len(args['text']) > 6000:
                        raise ValueError('Entry must contain 1..6000 characters')
                    with self.db:
                        result = append_entry(self, agent, self.notebooks[agent], args['text'], self.notebooks, True)
                        revision = self.db.execute('SELECT max(id) FROM revisions').fetchone()[0]
                        result = {**result, 'author': self.visible_labels[agent], 'revision': f'r-{revision}'}
                        self.db.execute('INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)',
                                        (agent, operation, json.dumps(args), json.dumps(result)))
                    # Emitted only after the transaction commits, while the lock remains held.
                    self.events.emit('notebook_published_available', agent=agent, url=result['saved'],
                        revision=result['revision'], author=result['author'],
                        text_sha256=hashlib.sha256(args['text'].encode()).hexdigest(),
                        available_to=list(AGENTS), availability_semantics='committed; retrievable by subsequent tool call')
                    return result
                except (ValueError, TypeError, KeyError) as error:
                    result = {'error': str(error)}
                    self.db.execute('INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)',
                                    (agent, operation, json.dumps(args)[:31000], json.dumps(result)))
                    self.db.commit()
                    return result
            if operation not in TOOL_NAMES:
                return {'error': 'Unknown tool'}
            result = super().call(agent, operation, args)
            url = result.get('url', '')
            slug = urlsplit(url).path.removeprefix('/page/') if isinstance(url, str) else ''
            if 'error' not in result and (entry_owner(slug, self.notebooks) or slug in self.notebooks.values()):
                latest = self.db.execute('SELECT max(id) FROM revisions WHERE slug=?', (slug,)).fetchone()[0]
                self.events.emit('notebook_read_returned', agent=agent, operation=operation, url=url,
                    revision=result.get('revision', f'r-{latest}' if latest else None),
                    entry_body_returned=entry_owner(slug, self.notebooks) is not None,
                    text_sha256=hashlib.sha256(result.get('text', '').encode()).hexdigest(),
                    interpretation='read delivery only; awareness and causal uptake not classified')
            return result


def make_browser(settings, pages, path, events):
    return AsyncNotebookBrowser(pages, path, events=events, append_notes=True,
        notebooks=settings['notebooks'], neutral_notebook=True, visible_labels=settings['visible_labels'],
        notebook_tools=True, editable_sources=[], editable_title_marker=False, search_snippets=False,
        request_history_mode='disabled', history_search=False, **browser_discovery(settings['discovery_plan']),
        **search_browser_options(settings['discovery_plan'], 'distinct_sources_5'),
        **access_browser_options(settings['access_plan']))


def write_state(path, data):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def run_async_notebooks(run_dir, client, *, records, topic, selectors, question_ids, access_manifest,
                        visible_labels, budget=None, source_access_mode='discovery_only',
                        job_deadline=None, checkpoint_callback=None, provenance=None):
    budget = budget or AsyncBudget()
    settings, pages, tasks = build_async_settings(records, topic, selectors, question_ids, access_manifest,
                                                 visible_labels, budget, source_access_mode)
    path = Path(run_dir)
    if path.exists():
        raise ValueError('Fresh async run directory required')
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError('Invalid checkpoint callback')
    metadata = validate_client(client)
    settings['provenance'] = {**(provenance or {}), 'model': metadata}
    client.notebook_tools_enabled, client.notebook_tool_schemas = True, TOOLS
    path.mkdir(parents=True)
    events = EventLog(path / 'events.jsonl')
    browser = None
    histories = {a: [{'role': 'system', 'content': settings['system_prompts'][a]}] for a in AGENTS}
    states = {a: {'question_index': 0, 'status': 'active', 'generated_tokens': 0, 'tool_calls': 0,
                  'model_requests': 0, 'active_seconds': 0.0} for a in AGENTS}
    results = []
    status = 'in_progress'
    wrapper = BoundedContextClient(client, TimedPolicy(**settings['policy']), path)
    queued = {}

    def save():
        for name, data in [('histories.json', histories), ('state.json', states), ('results.json', results),
                           ('manifest.json', {'status': status, 'scheduling': settings['scheduling'],
                             'terminated_questions': len(results),
                             'answered_questions': sum(row['status'] == 'answered' for row in results)})]:
            write_state(path / name, data)
        if checkpoint_callback:
            checkpoint_callback(path)

    def assign(agent):
        state = states[agent]
        if state['question_index'] >= settings['question_count']:
            state['status'] = 'complete'
            events.emit('agent_complete', agent=agent)
            return
        qid = question_ids[agent][state['question_index']]
        wrapper.begin_question(agent, histories[agent])
        histories[agent].append({'role': 'user', 'content': 'Question: ' + tasks[qid]['question']})
        queued[agent] = time.monotonic()
        events.emit('question_available', agent=agent, question_id=qid, question_index=state['question_index'])

    def finish(agent, reason, answer=None, termination_detail=None):
        state = states[agent]
        result = {'agent': agent, 'question_id': question_ids[agent][state['question_index']],
                  'question_index': state['question_index'], 'status': reason, 'answer': answer,
                  **{k: state[k] for k in ('generated_tokens', 'tool_calls', 'model_requests', 'active_seconds')}}
        if termination_detail is not None:
            result['termination_detail'] = termination_detail
        results.append(result)
        events.emit('question_terminated', **result)
        state.update(question_index=state['question_index'] + 1, generated_tokens=0, tool_calls=0,
                     model_requests=0, active_seconds=0.0)
        assign(agent)

    try:
        for name, data in [('settings.json', settings), ('dataset.json', records), ('pages.json', pages)]:
            write_state(path / name, data)
        browser = make_browser(settings, pages, path / 'wiki.sqlite3', events)
        setup_started = time.monotonic()
        setup_timeout = settings['policy']['initial_readiness_timeout_seconds']
        if job_deadline is not None:
            setup_timeout = min(setup_timeout, max(0, job_deadline - time.monotonic()))
        client.ensure_ready(timeout=setup_timeout)
        events.emit('shared_backend_ready', setup_seconds=time.monotonic() - setup_started,
                    charged_to_agent_question=False)
        for agent in AGENTS:
            assign(agent)
        ready = deque(AGENTS)
        save()
        while ready:
            agent = ready.popleft()
            state = states[agent]
            if state['status'] != 'active':
                continue
            if job_deadline is not None and time.monotonic() >= job_deadline:
                status = 'job_safety_stop'
                break
            if (state['generated_tokens'] >= budget.generated_tokens or state['model_requests'] >= budget.model_requests
                    or state['active_seconds'] >= budget.active_seconds):
                finish(agent, 'budget_exhausted')
            else:
                allowance = budget.generated_tokens - state['generated_tokens']
                active_remaining = budget.active_seconds - state['active_seconds']
                timeout = min(active_remaining, max(0, job_deadline - time.monotonic())) if job_deadline else active_remaining
                events.emit('inference_slot_granted', agent=agent, question_index=state['question_index'],
                            queue_seconds=time.monotonic() - queued[agent], allowance=allowance,
                            concurrent_inference_requests=1)
                started = time.monotonic()
                try:
                    client.ensure_ready(timeout=min(timeout, settings['policy']['initial_readiness_timeout_seconds']))
                    timeout -= time.monotonic() - started
                    if timeout <= 0:
                        raise TimeoutError('Readiness exhausted active request allowance')
                    state['model_requests'] += 1
                    response = wrapper(agent, histories[agent], timeout, num_predict=allowance)
                finally:
                    state['active_seconds'] += time.monotonic() - started
                if not isinstance(response, ModelResponse):
                    raise ValueError('Native ModelResponse required')
                count = response.metadata.get('eval_count')
                if type(count) is not int or not 0 <= count <= allowance:
                    raise ValueError('Invalid native generation count')
                state['generated_tokens'] += count
                events.emit('model_response', agent=agent, question_index=state['question_index'],
                            message=response.message, metadata=response.metadata, charged_generated_tokens=count)
                message = response.message
                if not isinstance(message, dict) or len(json.dumps(message)) > MAX_RESPONSE:
                    raise ValueError('Invalid model message')
                calls = message.get('tool_calls', [])
                if not isinstance(calls, list) or len(calls) > 8:
                    raise ValueError('Invalid tool-call collection')
                content, thinking = message.get('content', ''), message.get('thinking', '')
                if not isinstance(content, str) or not isinstance(thinking, str):
                    raise ValueError('Invalid response text')
                if response.metadata.get('done_reason') == 'length':
                    histories[agent].append({'role': 'assistant', 'content': content, 'thinking': thinking})
                    events.emit('truncated_response_not_executed', agent=agent,
                                question_index=state['question_index'], tool_calls_discarded=len(calls))
                    finish(agent, 'generation_limit_reached', termination_detail={
                        'backend_done_reason': 'length', 'requested_generation_tokens': allowance,
                        'backend_requested_num_predict': response.metadata.get('requested_num_predict'),
                        'native_generated_tokens': count,
                        'question_tokens_remaining': budget.generated_tokens - state['generated_tokens'],
                        'partial_response_preserved': True, 'continuation_attempted': False})
                elif calls:
                    accepted = {'role': 'assistant', 'content': content, 'thinking': thinking, 'tool_calls': []}
                    histories[agent].append(accepted)
                    for call in calls:
                        function = call.get('function', {}) if isinstance(call, dict) else {}
                        name, args = function.get('name'), function.get('arguments')
                        if name not in TOOL_NAMES or not isinstance(args, dict):
                            raise ValueError('Unsupported model tool request')
                        accepted['tool_calls'].append(call)
                        if state['tool_calls'] >= budget.tool_calls:
                            output = {'error': 'Question tool-call budget exhausted; respond without tool calls to finalize'}
                        else:
                            state['tool_calls'] += 1
                            output = browser.call(agent, name, args)
                        histories[agent].append({'role': 'tool', 'tool_name': name, 'content': json.dumps(output)})
                        events.emit('tool_returned', agent=agent, question_index=state['question_index'],
                                    operation=name, args=args, result=output, charged_tool_calls=state['tool_calls'])
                else:
                    histories[agent].append({'role': 'assistant', 'content': content, 'thinking': thinking})
                    if content.strip() and response.metadata.get('done_reason') != 'length':
                        finish(agent, 'answered', content)
                queued[agent] = time.monotonic()
            if state['status'] == 'active':
                ready.append(agent)
            save()
        else:
            status = 'complete'
        if status != 'complete':
            for agent in AGENTS:
                if states[agent]['status'] == 'active':
                    states[agent]['status'] = 'interrupted'
                    events.emit('agent_interrupted', agent=agent, reason=status)
    except BaseException as error:
        status = 'failed'
        events.emit('run_failed', error=f'{type(error).__name__}: {error}', partial_native_counts_unavailable=True)
        for agent in AGENTS:
            if states[agent]['status'] == 'active':
                states[agent]['status'] = 'interrupted'
                events.emit('agent_interrupted', agent=agent, reason='global_fail_closed_backend_or_host_failure')
        raise
    finally:
        try:
            save()
        finally:
            if browser:
                browser.close()
    return {'status': status, 'results': results, 'states': states}
