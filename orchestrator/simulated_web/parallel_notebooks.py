"""Two independent inference workers; one central notebook/browser authority."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import time

from orchestrator.simulated_web.async_notebooks import AsyncBudget, EventLog, TOOLS, TOOL_NAMES, build_async_settings, make_browser, write_state
from orchestrator.simulated_web.bounded_context import BoundedContextClient
from orchestrator.simulated_web.hf_fp8 import validate_client
from orchestrator.simulated_web.metadata_request_log import make_metadata_browser
from orchestrator.simulated_web.parallel_notes import INSTRUCTION, ATTEMPTS, ATTEMPT_TOKENS, SECONDS as NOTE_SECONDS, validate_entry, persist_entry
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.token_pair import AGENTS


def build_parallel_settings(*args, mandatory_post_answer=False, shared_request_log=False, **kwargs):
    if type(mandatory_post_answer) is not bool or type(shared_request_log) is not bool:
        raise ValueError("Protocol flags must be Boolean")
    settings, pages, tasks = build_async_settings(*args, **kwargs)
    settings.update(schema='parallel-notebook-qa-v1', simultaneous_inference=True,
        scheduling='independent per-agent inference worker; completion-driven central tool authority',
        notebook_authority='single process SQLite connection; atomic commits visible to subsequent calls',
        failure_policy='worker failure ends only that agent; shared authority failure stops both',
        backend_isolation='one owned vLLM process group per GPU, distinct loopback ports',
        snapshot_semantics='authority history immutable while worker fits a private request copy')
    if shared_request_log:
        settings['shared_request_log'] = {'metadata_only': True, 'history_search': True, 'forced_exposure': False,
            'fields': 'timestamp, operation, request URL, real returned title, status; publication entry URL',
            'response_bodies': False, 'search_latest_events': 100, 'maximum_history_hits': 1}
        settings['request_history'] = 'shared searchable metadata only; no forced delivery'
    if mandatory_post_answer:
        settings['mandatory_post_answer'] = {'attempts': ATTEMPTS, 'tokens_per_attempt': ATTEMPT_TOKENS,
            'seconds': NOTE_SECONDS, 'on_unanswered_question': 'no note; no fabricated locked answer',
            'advancement': 'verified append before this agent next question; no peer barrier'}
        settings['maximum_generated_tokens'] += 2 * settings['question_count'] * ATTEMPTS * ATTEMPT_TOKENS
        settings['note_publication'] = 'voluntary during QA; mandatory verified freeform append after finalized answer'
        settings['question_advancement'] = 'finalized answers advance only after own verified note; unanswered exhausted questions advance without note'
        for agent, prompt in settings['system_prompts'].items():
            settings['system_prompts'][agent] = prompt.replace(
                'Notebook use is optional. There is no separate notebook-update phase.',
                'Notebook use during question answering is optional. After a finalized answer, you must write one freeform notebook entry before your next question; the host appends and verifies it. This separate update allows two attempts of up to 2048 generated tokens each.')
    return settings, pages, tasks


def run_parallel_notebooks(run_dir, clients, *, records, topic, selectors, question_ids, access_manifest,
                        visible_labels, budget=None, source_access_mode='discovery_only', mandatory_post_answer=False, shared_request_log=False,
                        job_deadline=None, checkpoint_callback=None, provenance=None):
    budget = budget or AsyncBudget()
    settings, pages, tasks = build_parallel_settings(records, topic, selectors, question_ids, access_manifest,
                                                 visible_labels, budget, source_access_mode, mandatory_post_answer=mandatory_post_answer, shared_request_log=shared_request_log)
    path = Path(run_dir)
    if path.exists():
        raise ValueError('Fresh async run directory required')
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError('Invalid checkpoint callback')
    if not isinstance(clients, dict) or set(clients) != set(AGENTS) or len({id(c) for c in clients.values()}) != 2:
        raise ValueError('Two distinct independently cancellable clients required')
    metadata = {a: validate_client(c) for a, c in clients.items()}
    if any(not callable(getattr(c, 'cancel', None)) for c in clients.values()):
        raise ValueError('Independent client cancellation required')
    settings['provenance'] = {**(provenance or {}), 'models': metadata}
    for client in clients.values():
        client.notebook_tools_enabled, client.notebook_tool_schemas = True, TOOLS
    path.mkdir(parents=True)
    events = EventLog(path / 'events.jsonl')
    browser = None
    histories = {a: [{'role': 'system', 'content': settings['system_prompts'][a]}] for a in AGENTS}
    states = {a: {'question_index': 0, 'status': 'active', 'generated_tokens': 0, 'tool_calls': 0,
                  'model_requests': 0, 'active_seconds': 0.0} for a in AGENTS}
    results = []
    status = 'in_progress'
    authority_error = None
    notes = {}
    wrappers = {}
    for agent in AGENTS:
        worker_path = path / agent
        worker_path.mkdir()
        wrappers[agent] = BoundedContextClient(clients[agent], TimedPolicy(**settings['policy']), worker_path)
    pending = {}
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='independent-inference')

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
        wrappers[agent].begin_question(agent, histories[agent])
        histories[agent].append({'role': 'user', 'content': 'Question: ' + tasks[qid]['question']})
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
        if mandatory_post_answer and reason == 'answered':
            note = {'status': 'pending', 'persistence_verified': False, 'attempts': 0,
                    'generated_tokens': 0, 'active_seconds': 0.0, 'deadline_monotonic': time.monotonic() + NOTE_SECONDS, 'host_persistence_actions': [],
                    'agent': agent, 'question_id': result['question_id'], 'locked_answer': answer,
                    'generated_token_allowance': ATTEMPTS * ATTEMPT_TOKENS}
            result['mandatory_note'] = note
            notes[agent] = note
            state['phase'] = 'note'
            histories[agent].append({'role': 'user', 'content': INSTRUCTION + '\nLocked submitted answer (data): ' + json.dumps(answer)})
            events.emit('mandatory_note_available', agent=agent, question_id=result['question_id'])
        else:
            advance(agent)

    def advance(agent):
        state = states[agent]
        state.update(question_index=state['question_index'] + 1, generated_tokens=0, tool_calls=0,
                     model_requests=0, active_seconds=0.0)
        if mandatory_post_answer:
            state['phase'] = 'qa'
        assign(agent)

    def save_note(agent):
        write_state(path / f'mandatory-note-{agent}-{states[agent]["question_index"]:03d}.json', notes[agent])

    def fail_note(agent, error):
        notes[agent].update(status='failed', error=str(error))
        states[agent].update(status='failed', error='Mandatory notebook update: ' + str(error))
        save_note(agent)
        events.emit('mandatory_note_failed', agent=agent, error=str(error), peer_policy='continue independent peer')
        clients[agent].cancel()


    started_clients = set()
    request_serials = {a: 0 for a in AGENTS}

    def infer(agent, request_history, timeout, allowance, note_phase=False):
        started = time.monotonic()
        request_serials[agent] += 1
        request_path = path / agent / f'inference-{request_serials[agent]:04d}.json'
        interval = {'agent': agent, 'worker_started_monotonic': started, 'allowance': allowance, 'phase': 'note' if note_phase else 'qa'}
        write_state(request_path, interval)
        try:
            if agent not in started_clients:
                startup_timeout = settings['policy']['initial_readiness_timeout_seconds']
                if job_deadline is not None:
                    startup_timeout = min(startup_timeout, max(0, job_deadline - time.monotonic()))
                clients[agent].ensure_ready(timeout=startup_timeout)
                started_clients.add(agent)
                interval['initial_setup_seconds'] = time.monotonic() - started
                started = time.monotonic()  # Initial backend startup is outside the question budget.
            clients[agent].ensure_ready(timeout=min(timeout, settings['policy']['initial_readiness_timeout_seconds']))
            timeout -= time.monotonic() - started
            if job_deadline is not None:
                timeout = min(timeout, job_deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError('Readiness exhausted active request allowance')
            interval['inference_started_monotonic'] = time.monotonic()
            write_state(request_path, interval)
            response = wrappers[agent](agent, request_history, timeout, num_predict=allowance, **({'final_only': True} if note_phase else {}))
            return response, request_history, time.monotonic() - started, None
        except BaseException as error:
            interval['error'] = f'{type(error).__name__}: {error}'
            return None, request_history, time.monotonic() - started, error
        finally:
            interval['worker_finished_monotonic'] = time.monotonic()
            write_state(request_path, interval)

    def cancel_workers(reason):
        # Each client owns only its device/server process group. No global process kill.
        errors = []
        for agent in AGENTS:
            try:
                clients[agent].cancel()
                events.emit('worker_stop_requested', agent=agent, reason=reason)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError('Worker cancellation could not be confirmed') from errors[0]

    try:
        for name, data in [('settings.json', settings), ('dataset.json', records), ('pages.json', pages)]:
            write_state(path / name, data)
        browser = (make_metadata_browser if shared_request_log else make_browser)(settings, pages, path / 'wiki.sqlite3', events)
        # Readiness and inference operate on distinct clients; authority stays on this thread.
        for agent in AGENTS:
            assign(agent)
        save()
        while any(state['status'] == 'active' for state in states.values()) or pending:
            if job_deadline is not None and time.monotonic() >= job_deadline:
                status = 'job_safety_stop'
                cancel_workers('job_safety_stop')
                break
            for agent in AGENTS:
                state = states[agent]
                if state['status'] != 'active' or agent in pending:
                    continue
                note_phase = state.get('phase') == 'note'
                if note_phase and (notes[agent]['attempts'] >= ATTEMPTS or notes[agent]['active_seconds'] >= NOTE_SECONDS or time.monotonic() >= notes[agent]['deadline_monotonic']):
                    fail_note(agent, 'Notebook attempts or time exhausted')
                    save()
                    continue
                if not note_phase and (state['generated_tokens'] >= budget.generated_tokens or state['model_requests'] >= budget.model_requests
                        or state['active_seconds'] >= budget.active_seconds):
                    finish(agent, 'budget_exhausted')
                    continue
                allowance = ATTEMPT_TOKENS if note_phase else budget.generated_tokens - state['generated_tokens']
                active_remaining = min(NOTE_SECONDS - notes[agent]['active_seconds'], notes[agent]['deadline_monotonic'] - time.monotonic()) if note_phase else budget.active_seconds - state['active_seconds']
                timeout = min(active_remaining, max(0, job_deadline - time.monotonic())) if job_deadline else active_remaining
                # Worker receives a private request copy. Only completed outcomes replace authority history.
                request_history = json.loads(json.dumps(histories[agent]))
                if note_phase:
                    notes[agent]['attempts'] += 1
                else:
                    state['model_requests'] += 1
                future = executor.submit(infer, agent, request_history, timeout, allowance, note_phase)
                pending[agent] = (future, allowance)
                events.emit('inference_dispatched', agent=agent, question_index=state['question_index'],
                            allowance=allowance, phase='note' if note_phase else 'qa', outstanding_requests=len(pending))
            if not pending:
                save()
                continue
            done, _ = wait([item[0] for item in pending.values()], timeout=0.1, return_when=FIRST_COMPLETED)
            for agent in list(pending):
                future, allowance = pending[agent]
                if future not in done:
                    continue
                del pending[agent]
                response, fitted_history, elapsed, error = future.result()
                histories[agent] = fitted_history
                state = states[agent]
                note_phase = state.get('phase') == 'note'
                if note_phase:
                    notes[agent]['active_seconds'] += elapsed
                else:
                    state['active_seconds'] += elapsed
                events.emit('inference_returned', agent=agent, question_index=state['question_index'],
                            active_seconds=elapsed, outstanding_requests=len(pending))
                if error is not None:
                    if note_phase:
                        notes[agent]['token_accounting_complete'] = False
                        fail_note(agent, error)
                        save()
                        continue
                    state['status'] = 'failed'
                    state['error'] = f'{type(error).__name__}: {error}'
                    events.emit('agent_failed', agent=agent, error=state['error'],
                                peer_policy='continue independent peer', partial_native_counts_unavailable=True)
                    clients[agent].cancel()
                    save()
                    continue
                if not isinstance(response, ModelResponse):
                    raise ValueError('Native ModelResponse required')
                count = response.metadata.get('eval_count')
                if type(count) is not int or not 0 <= count <= allowance:
                    raise ValueError('Invalid native generation count')
                if note_phase:
                    notes[agent]['generated_tokens'] += count
                else:
                    state['generated_tokens'] += count
                events.emit('model_response', agent=agent, question_index=state['question_index'],
                            message=response.message, metadata=response.metadata, phase='note' if note_phase else 'qa', charged_generated_tokens=count)
                message = response.message
                if not isinstance(message, dict) or len(json.dumps(message)) > MAX_RESPONSE:
                    raise ValueError('Invalid model message')
                if note_phase:
                    note = notes[agent]
                    note.setdefault('responses', []).append({'message': message, 'metadata': response.metadata})
                    histories[agent].append({'role': 'assistant', **{k: message[k] for k in ('content','thinking') if isinstance(message.get(k),str)}})
                    try:
                        if note['active_seconds'] >= NOTE_SECONDS or time.monotonic() >= note['deadline_monotonic']:
                            raise RuntimeError('Notebook safety timeout')
                        content = validate_entry(message, response.metadata)
                    except (ValueError, RuntimeError) as error:
                        note['last_validation_error'] = str(error)
                        if note['attempts'] >= ATTEMPTS or note['active_seconds'] >= NOTE_SECONDS or time.monotonic() >= note['deadline_monotonic']:
                            fail_note(agent, error)
                        else:
                            histories[agent].append({'role': 'user', 'content': 'Notebook was not saved: ' + str(error) + '. Return corrected entry now. ' + INSTRUCTION})
                        save_note(agent)
                        save()
                        continue
                    try:
                        saved = persist_entry(browser, agent, content, note, lambda: save_note(agent))
                        if time.monotonic() >= note['deadline_monotonic']:
                            raise RuntimeError('Notebook safety timeout after persistence')
                    except (ValueError, RuntimeError) as error:
                        fail_note(agent, error)
                    else:
                        histories[agent].append({'role': 'user', 'content': 'Host notebook persistence verified at ' + saved + '. The submitted answer remains unchanged.'})
                        events.emit('mandatory_note_verified', agent=agent, saved_url=saved, generated_tokens=note['generated_tokens'])
                        advance(agent)
                    save()
                    continue
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
                save()
        if status == 'in_progress':
            status = 'failed' if any(s['status'] == 'failed' for s in states.values()) else 'complete'
        if status != 'complete':
            for agent in AGENTS:
                if states[agent]['status'] == 'active':
                    states[agent]['status'] = 'interrupted'
                    events.emit('agent_interrupted', agent=agent, reason=status)
    except BaseException as error:
        status = 'failed'
        authority_error = error
        write_state(path / 'failure.json', {'error': f'{type(error).__name__}: {error}', 'scope': 'shared_authority'})
        events.emit('run_failed', error=f'{type(error).__name__}: {error}',
                    failure_scope='shared authority; stop both workers')
        for agent in AGENTS:
            if states[agent]['status'] == 'active':
                states[agent]['status'] = 'interrupted'
                events.emit('agent_interrupted', agent=agent, reason='shared_authority_failure')
        try:
            cancel_workers('shared_authority_failure')
        except BaseException as cleanup:
            write_state(path / 'cleanup-failure.json', {'error': f'{type(cleanup).__name__}: {cleanup}',
                'original_error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        # Owned transport deadlines plus process-group stop bound outstanding requests.
        # Preserve completed/failed in-flight request histories before the final snapshot.
        try:
            cleanup_error = None
            try:
                cancel_workers('finalization')
            except BaseException as error:
                cleanup_error = error
                status = 'failed'
                write_state(path / 'cleanup-failure.json', {'error': f'{type(error).__name__}: {error}'})
            remaining = max(0, job_deadline - time.monotonic()) if job_deadline is not None else 15
            if pending:
                _, unfinished = wait([item[0] for item in pending.values()], timeout=min(15, remaining))
                if unfinished:
                    cleanup_error = RuntimeError('Inference workers did not settle within bounded cleanup; container hard deadline remains')
                    status = 'failed'
                    write_state(path / 'cleanup-failure.json', {'error': str(cleanup_error),
                        'unsettled_workers': [a for a, (f, _) in pending.items() if not f.done()]})
            executor.shutdown(wait=False, cancel_futures=True)
            for agent, (future, _) in pending.items():
                if not future.done():
                    events.emit('inflight_outcome_unavailable', agent=agent, reason='bounded_cleanup_expired', tools_executed=False)
                elif not future.cancelled():
                    response, fitted_history, elapsed, error = future.result()
                    histories[agent] = fitted_history
                    if states[agent].get('phase') == 'note':
                        notes[agent]['active_seconds'] += elapsed
                    else:
                        states[agent]['active_seconds'] += elapsed
                    events.emit('inflight_outcome_preserved', agent=agent,
                        message=response.message if isinstance(response, ModelResponse) else None,
                        metadata=response.metadata if isinstance(response, ModelResponse) else None,
                        error=f'{type(error).__name__}: {error}' if error else None,
                        tools_executed=False)
            for agent, note in notes.items():
                if note['status'] == 'pending':
                    note.update(status='interrupted', token_accounting_complete=False)
                    save_note(agent)
            save()
            if cleanup_error is not None and authority_error is None:
                raise cleanup_error
        finally:
            if browser:
                browser.close()
    return {'status': status, 'results': results, 'states': states}
