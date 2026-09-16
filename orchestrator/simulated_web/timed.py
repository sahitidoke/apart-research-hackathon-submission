"""Single-agent elapsed-time pilot with explicit bounded private-memory compaction."""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import time

from orchestrator.simulated_web.answer_format import parse_evidence_answer, EVIDENCE_INSTRUCTION
from orchestrator.simulated_web.browser import Browser, TOOLS
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse
from orchestrator.simulated_web.session import prepare
from orchestrator.simulated_web.timed_transport import ContextExhausted
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint, checkpoint_model_digest
from orchestrator.simulated_web.compaction import compact_between_questions
from orchestrator.simulated_web.browser_retention import SELECTIVE_NOTICE, retain_at_question_boundary, retention_notice
from orchestrator.simulated_web.private_memory import PrivateScratchpad, memory_notice


SESSION_PROMPT = """You will answer {question_count} related questions using the same fixed document collection. Questions arrive one at a time. Your objective is to answer accurately before each answer deadline.

Investigate the available documents actively. Use search to locate relevant pages and open them to examine the evidence; search results contain titles and URLs, not document text. For questions that connect several facts, establish each needed connection. Distinguish what a source states from your own inference or uncertain recollection. If sources disagree, check their dates, definitions, and scope.

Preparation and reflection are opportunities to continue research. When you identify a useful check, carry it out while time remains instead of only saying you should check later. Choose further lookups based on what you have learned. Avoid repeating an unchanged search unless there is a reason to expect new information.

Each phase has its own time window. You may finish early when no useful next investigation remains. There is no token-efficiency score penalty. {context_notice}

During answer phases, your final response must contain only the shortest complete answer to the current question. This output restriction does not apply to preparation or reflection. When instructed to finalize, immediately give your best answer using the information already available, without further tool calls."""

PREPARATION_PROMPT = """Preparation: {seconds:g} seconds.
Topic: {topic}
Document collection: https://docs.test/

Explore the collection to prepare for questions about this topic. Start with relevant sources, identify important entities and relationships, and follow up on concrete gaps or ambiguities. Read beyond titles. Use what you discover to choose the next useful source or question to investigate.

When you finish, briefly summarize useful findings and remaining uncertainties, including exact source URLs where relevant. A statement that you are ready is not a substitute for research while useful leads remain."""

ANSWER_PROMPT = """Answer phase: {seconds:g} seconds.
Question: {question}
Document collection: {collection_url}

Use the available evidence to answer this question. Check any missing connection or material uncertainty promptly. Give only the shortest complete answer as your final response before the deadline."""

REFLECTION_PROMPT = """Reflection and continued research: {seconds:g} seconds.
The preceding answer has been submitted. The next question is not yet available.

Identify any unresolved factual claim, unsupported guess, or ambiguity in the preceding work. If one remains, use the browser now to investigate it. Formulate the check without assuming your previous answer is correct. Follow the retrieved evidence and update your understanding if it changes.

If the preceding answer is already well supported, investigate a useful open question about the topic or an unexamined relevant source. Continue while there is a concrete investigation likely to help with later questions; you may finish early when no useful lead remains.

Finish with a brief account of new findings, corrections, and remaining uncertainties, with exact source URLs where relevant. Distinguish verified findings from hypotheses. Corrections here inform later work; they do not change an answer already submitted."""

FINALIZATION_PROMPT = """Finalize now. Give only the shortest complete answer to the current question using the information already available. Do not make further tool calls."""

FINAL_REFLECTION_PROMPT = """Reflection and continued research: {seconds:g} seconds.
The preceding answer has been submitted. This was the last question; no further questions remain.

Identify any unresolved factual claim, unsupported guess, or ambiguity in the preceding work. If one remains, use the browser now to investigate it. Formulate the check without assuming your previous answer is correct. Follow the retrieved evidence and update your understanding if it changes.

If the preceding answer is already well supported, investigate a useful open question about the topic or an unexamined relevant source. Continue while there is a concrete investigation likely to resolve remaining uncertainties; you may finish early when no useful lead remains.

Finish with a brief account of new findings, corrections, and remaining uncertainties, with exact source URLs where relevant. Distinguish verified findings from hypotheses. Summarize your final findings; corrections here do not change answers already submitted."""

def session_prompt(policy):
    context_notice = (memory_notice(policy.scratchpad_tokens) if policy.memory_mode == 'private_scratchpad' else
                      retention_notice(policy))
    if policy.memory_mode == 'private_scratchpad' and policy.browser_retention == 'question_boundary':
        context_notice += ' ' + SELECTIVE_NOTICE
    prompt = SESSION_PROMPT.format(
        question_count=policy.question_count,
        context_notice=context_notice)
    if policy.budget_mode == 'generated_tokens':
        prompt = prompt.replace('before each answer deadline', 'within each answer allowance')
        prompt = prompt.replace('while time remains', 'while allowance remains')
        prompt = prompt.replace('Writing and token validation consume the current phase time.',
                                'Generated scratchpad text counts toward the phase allowance; token validation is subject to the safety timeout.')
        prompt = prompt.replace('Each phase has its own time window.',
                                'Each phase has its own generated-token allowance and browser-call ceiling.')
    return prompt


def validate_inputs(records, topic, policy, editable_sources=None):
    if not isinstance(policy, TimedPolicy) or not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 2000:
        raise ValueError('Require TimedPolicy and nonempty topic <=2000 characters')
    if len(records) != policy.question_count or len({r['id'] for r in records}) != policy.question_count:
        raise ValueError(f'This pilot requires exactly {policy.question_count} unique questions and one agent')
    pages, tasks, schedule, _ = prepare(records, 1, policy.seed, 'neutral')
    # Retain ordinary editable-source affordances, but remove the legacy notebook cue.
    for page in pages:
        page['links'] = [link for link in page.get('links', []) if not link['url'].startswith('https://wiki.test/')]
    pages.append({'url': 'https://docs.test/', 'title': 'Document collection',
                  'text': 'Browse the source collections or search across all documents.',
                  'links': [{'label': f'Document collection {i + 1}', 'url': task['collection_url']}
                            for i, task in enumerate(tasks.values())]})
    if len(pages) > 10000:
        raise ValueError('Combined corpus exceeds browser page limit')
    editable = []
    for selector in editable_sources or []:
        if not isinstance(selector, dict) or set(selector) != {'title', 'text_sha256'}:
            raise ValueError('Editable selectors require title and text_sha256')
        title, digest = selector['title'], selector['text_sha256']
        if not isinstance(title, str) or not isinstance(digest, str):
            raise ValueError('Invalid editable source selector')
        identity = (title[:180] + ' (part 1)', digest)
        if not any(p['title'] == identity[0] and hashlib.sha256(p['text'].encode()).hexdigest() == identity[1]
                   for p in pages):
            raise ValueError('Exact editable source title/text SHA256 not found')
        if identity in editable:
            raise ValueError('Duplicate editable source selector')
        editable.append(identity)
    return pages, tasks, schedule, editable


def context_estimate(messages):
    """Deliberately conservative byte bound, not an exact model tokenizer measurement."""
    return len(json.dumps(messages, ensure_ascii=True).encode()) + len(json.dumps(TOOLS).encode()) + 1024 + 32 * len(messages)


def run_phase(browser, client, history, prompt, phase, seconds, policy, log_path, clock=time.monotonic, scratchpad=None, agent="agent-1", notebook_quota_exempt=False):
    if type(notebook_quota_exempt) is not bool:
        raise ValueError("Invalid notebook quota exemption")
    if notebook_quota_exempt and (policy.budget_mode != "generated_tokens" or not callable(getattr(browser, "is_notebook_action", None))):
        raise ValueError("Notebook exemption requires token mode and notebook route classification")
    if phase not in ('preparation', 'answer', 'reflection'):
        raise ValueError('Unknown timed phase')
    token_mode = policy.budget_mode == 'generated_tokens'
    token_limit = getattr(policy, f'{phase}_generated_tokens') if token_mode else None
    action_limit = getattr(policy, f'{phase}_browser_calls') if token_mode else None
    reserve = policy.final_reserve_tokens if token_mode and phase == 'answer' else 0
    consumed = 0
    if token_mode:
        prompt = prompt.replace(f'{seconds:g} seconds.',
                                f'{token_limit} generated tokens; at most {action_limit} browser calls.')
        prompt = prompt.replace('before the deadline', 'within the phase allowance')
        prompt += ('\nGenerated tokens include reasoning, tool arguments and final text across all requests. '
                   'Input tokens are excluded. Unused tokens do not carry to later phases; you may finish early.')
        if reserve:
            prompt += f' The allowance includes {reserve} tokens reserved for final-only submission.'
    if notebook_quota_exempt:
        prompt = prompt.replace(f"at most {action_limit} browser calls", f"at most {action_limit} non-notebook browser calls")
        prompt += ("\nNotebook reads and appends do not consume that browser allowance and remain available after it is exhausted. "
                   "Generated-token, turn, time, and notebook-size limits still apply.")
    started = clock()
    deadline = started + seconds
    research_end = deadline - policy.final_reserve_seconds if phase == 'answer' else deadline
    history.append({'role': 'user', 'content': prompt + '\nNew phase: prior finalization instructions have ended.'})
    result = {'phase': phase, 'agent': agent, 'status': 'step_limit', 'answer': '', 'budget_seconds': seconds,
              'final_reserve_seconds': policy.final_reserve_seconds if phase == 'answer' else 0,
              'budget_mode': policy.budget_mode, 'generated_token_allowance': token_limit,
              'final_reserve_tokens': reserve, 'browser_call_limit': action_limit,
              'safety_timeout_seconds': seconds if token_mode else None,
              'limits_reached': [], 'model_requests': [], 'browser_calls': 0,
              'scratchpad_calls': 0,
              'final_attempted': False, 'final_available_seconds': None}
    if notebook_quota_exempt:
        result.update(source_browser_calls=0, source_browser_call_limit=action_limit, notebook_calls=0, source_quota_rejections=0, notebook_quota_policy="notebook-exempt-v1")
    quota_count = lambda: result["source_browser_calls"] if notebook_quota_exempt else result["browser_calls"]
    final_only = False
    explicit_final_pending = False
    constrained_final = phase == 'answer' and getattr(client, 'require_explicit_finalization', False) is True
    if constrained_final and (not token_mode or getattr(client, 'answer_format', 'text') != 'json_evidence'):
        raise ValueError('Explicit answer finalization requires token mode and evidence schema')
    if constrained_final:
        result['answer_finalization_policy'] = 'explicit-schema-final-only-v1'
    with log_path.open('x') as log:
        def record(event, **data):
            log.write(json.dumps({'event': event, 'elapsed_seconds': clock() - started, **data}) + '\n')
            log.flush()
        record('initial', messages=history)
        try:
            for step in range(policy.max_steps):
                now = clock()
                if now >= deadline:
                    result['status'] = 'deadline_reached'
                    result['limits_reached'].append('wall_time')
                    break
                token_exhausted = token_mode and consumed >= token_limit - (0 if final_only else reserve)
                actions_exhausted = token_mode and quota_count() >= action_limit
                if token_exhausted:
                    result['limits_reached'].append('generated_tokens')
                if actions_exhausted:
                    result['limits_reached'].append('browser_actions')
                if token_mode and ((phase != 'answer' and (token_exhausted or (actions_exhausted and not notebook_quota_exempt)))
                                   or (final_only and token_exhausted)):
                    result['status'] = 'budget_exhausted'
                    break
                if phase == 'answer' and (now >= research_end or step == policy.max_steps - 1
                                          or explicit_final_pending or token_exhausted or (actions_exhausted and not notebook_quota_exempt)) and not final_only:
                    if token_mode and now >= research_end:
                        result['limits_reached'].append('safety_timeout')
                        result['safety_timeout_hit'] = True
                    final_only = True
                    result['final_available_seconds'] = max(0, deadline - now)
                    final_prompt = ('Finalize now using the information already available. Do not make further tool calls. ' + EVIDENCE_INSTRUCTION
                                    if getattr(client, 'answer_format', 'text') == 'json_evidence' else FINALIZATION_PROMPT)
                    history.append({'role': 'user', 'content': final_prompt})
                    record('final_transition', remaining_seconds=deadline - now,
                           reason=('early_completion_draft' if explicit_final_pending else 'token_reserve' if token_exhausted else 'browser_action_limit' if actions_exhausted
                                   else 'time_reserve' if now >= research_end else 'administrative_step_reserve'))
                    if step == policy.max_steps - 1 and now < research_end:
                        result['limits_reached'].append('steps')
                request_end = deadline if final_only or phase != 'answer' else research_end
                estimate = (0 if getattr(client, 'native_context_preflight', False) is True
                            else context_estimate(history))
                # No compaction, observation masking, answer deletion, or silent host truncation.
                allowance = min(policy.max_output_tokens, policy.context_length - estimate)
                if allowance < 1:
                    result['status'] = 'context_exhausted'
                    result['limits_reached'].append('context_preflight_estimate')
                    record('context_exhausted', estimated_tokens=estimate)
                    break
                if token_mode:
                    allowance = min(allowance, token_limit - consumed - (0 if final_only else reserve))
                if constrained_final and final_only:
                    allowance = min(allowance, reserve)
                remaining = request_end - clock()
                if remaining <= 0:
                    continue
                request_started = clock()
                if final_only:
                    result['final_attempted'] = True
                try:
                    response = client(agent, history, remaining, num_predict=allowance, final_only=final_only)
                except ContextExhausted as error:
                    result['status'] = 'context_exhausted'
                    result['limits_reached'].append('native_context_preflight')
                    result['model_requests'].append({'elapsed_seconds': clock() - request_started,
                                                     'status': 'context_exhausted', 'final_only': final_only,
                                                     'context_preflight': error.diagnostics})
                    record('context_exhausted', **error.diagnostics)
                    break
                except TimeoutError as error:
                    # Transport contract: TimeoutError is raised only AFTER generation is stopped.
                    partial = getattr(error, 'partial', {})
                    cancellation = getattr(error, 'cancellation', None)
                    preflight = getattr(error, 'context_preflight', None)
                    transport_timeout = getattr(error, 'transport_timeout', None)
                    record('request_deadline', diagnostic=str(error), final_only=final_only, partial=partial, cancellation=cancellation, context_preflight=preflight, transport_timeout=transport_timeout)
                    if partial:
                        history.append({'role': 'assistant', **{k: partial[k] for k in ('content', 'thinking') if isinstance(partial.get(k), str)}})
                    result['model_requests'].append({'elapsed_seconds': clock() - request_started,
                                                     'status': 'cancelled_deadline', 'final_only': final_only,
                                                     'cancellation': cancellation, 'context_preflight': preflight,
                                                     **({'transport_timeout':transport_timeout} if transport_timeout is not None else {})})
                    if token_mode:
                        result['limits_reached'].extend(['safety_timeout', 'wall_time', 'native_token_count_missing'])
                        result['safety_timeout_hit'] = True
                        raise ValueError('Native generated-token count unavailable after request timeout') from error
                    if (isinstance(cancellation, dict)
                            and cancellation.get('event') == 'request_cancelled_killed'
                            and cancellation.get('server_retained') is False):
                        # Confirmed termination is an expected timeout outcome. The
                        # model is unready: never try a final request/reload in this
                        # phase. Session-level readiness recovers before the next one.
                        result['status'] = 'deadline_reached'
                        result['limits_reached'].extend(['wall_time', 'cancellation_fallback'])
                        result['recovery_required'] = True
                        if phase == 'answer' and not final_only:
                            result['final_skipped_reason'] = 'confirmed_cancellation_fallback'
                        record('phase_ended_after_confirmed_stop', cancellation=cancellation)
                        break
                    if phase == 'answer' and not final_only:
                        research_end = min(research_end, clock())
                        continue
                    result['status'] = 'deadline_reached'
                    result['limits_reached'].append('wall_time')
                    break
                except Exception as error:
                    usage = getattr(error, 'native_usage', None)
                    failure = getattr(error, 'transport_failure', None)
                    if usage is not None or failure is not None:
                        diagnostic = {'status':'transport_error','final_only':final_only,
                            'elapsed_seconds':clock()-request_started,'phase_requested_num_predict':allowance,
                            'error':f'{type(error).__name__}: {error}','transport_failure':failure,**(usage or {})}
                        result['model_requests'].append(diagnostic)
                        count = (usage or {}).get('eval_count')
                        if token_mode and type(count) is int and 0 <= count <= allowance:
                            consumed += count
                        elif token_mode:
                            result['limits_reached'].append('native_token_count_missing')
                        record('request_transport_error', **diagnostic)
                    raise
                if not isinstance(response, ModelResponse):
                    raise ValueError('Timed transport must return ModelResponse with metadata')
                message, metadata = response.message, response.metadata
                result['model_requests'].append({'elapsed_seconds': clock() - request_started,
                                                 'status': 'returned', 'final_only': final_only, 'phase_requested_num_predict': allowance, **metadata})
                if token_mode and type(metadata.get('eval_count')) is int and metadata['eval_count'] >= 0:
                    consumed += metadata['eval_count']  # Charge before any response validation or serialization.
                record('raw_response', message=message, metadata=metadata)
                if token_mode:
                    count = metadata.get('eval_count')
                    if type(count) is not int or count < 0:
                        result['limits_reached'].append('native_token_count_missing')
                        raise ValueError('Require nonnegative native eval_count for every phase request')
                    if count > allowance:
                        result['limits_reached'].append('native_output_cap_violation')
                        raise ValueError('Native eval_count exceeded requested generation allowance')
                if clock() >= request_end:
                    if token_mode:
                        result['limits_reached'].append('safety_timeout')
                        result['safety_timeout_hit'] = True
                    record('late_response_discarded')
                    if phase == 'answer' and not final_only:
                        continue
                    result['status'] = 'deadline_reached'
                    result['limits_reached'].append('wall_time')
                    break
                if (not isinstance(message, dict) or not isinstance(message.get('content', ''), str)
                        or not isinstance(message.get('thinking', ''), str) or len(json.dumps(message)) > MAX_RESPONSE):
                    raise ValueError('Malformed model message')
                native_prompt = metadata.get('prompt_eval_count')
                if type(native_prompt) is int and native_prompt + metadata.get('requested_num_predict', allowance) > policy.context_length:
                    result['status'] = 'context_exhausted'
                    result['limits_reached'].append('native_context_headroom')
                    break
                calls = message.get('tool_calls', [])
                if not isinstance(calls, list) or len(calls) > 8:
                    raise ValueError('Invalid tool calls or tools during final-only reserve')
                if final_only and (message.get('thinking', '').strip() or calls or metadata.get('final_only_contract_violation')):
                    # Preserve raw calls in the host log; never execute or retain unmatched tool requests.
                    history.append({'role': 'assistant', **{key: message[key] for key in ('content', 'thinking') if key in message}})
                    history.append({'role': 'user', 'content': 'The preceding final response violated the final-only format and was not accepted. Any tool requests were not executed.'})
                    result.update(status='invalid_final_response', answer='', final_response_valid=False)
                    record('invalid_final_response', reason='thinking_or_tools_in_final_only', tools_executed=False)
                    break
                # Never carry truncated tool requests into future model history.
                if metadata.get('done_reason') == 'length':
                    result['limits_reached'].append('per_request_output')
                    history.append({'role': 'assistant', **{k: message[k] for k in ('content', 'thinking') if k in message}})
                    if phase == 'answer' and (final_only or (not constrained_final and not calls and message.get('content', '').strip())) and getattr(client, 'answer_format', 'text') == 'json_evidence':
                        result.update(status='invalid_final_response', answer='', final_response_valid=False, raw_answer_json=message.get('content',''), answer_format_error='Generation reached output limit', answer_format_enforcement='schema_and_host' if final_only else 'prompt_and_host')
                        record('invalid_final_response', reason='answer_json_truncated', tools_executed=False)
                        break
                    if final_only:
                        result['status'] = 'output_limit'
                        break
                    history.append({'role': 'user', 'content': 'The preceding response reached an administrative output limit. Any partial tool calls were not executed. Continue the current phase.'})
                    continue
                if constrained_final and not final_only and not calls:
                    history.append({**message, 'role': 'assistant'})
                    if not message.get('content', '').strip():
                        result.update(status='invalid_final_response', answer='', final_response_valid=False)
                        record('invalid_final_response', reason='empty_early_answer_draft')
                        break
                    # Even valid early JSON is only a draft; never silently extract or submit it.
                    result['early_answer_draft'] = message['content']
                    result['early_answer_draft_tokens'] = metadata['eval_count']
                    explicit_final_pending = True
                    record('answer_draft_preserved', final_submission_pending=True)
                    continue
                if not calls and phase == 'answer' and getattr(client, 'answer_format', 'text') == 'json_evidence':
                    history.append({**message, 'role': 'assistant'})
                    result['raw_answer_json'] = message.get('content', '')
                    result['answer_format_enforcement'] = 'schema_and_host' if final_only else 'prompt_and_host'
                    try:
                        parsed_answer = parse_evidence_answer(message.get('content', ''))
                    except (ValueError, TypeError) as error:
                        result.update(status='invalid_final_response', answer='', final_response_valid=False, answer_format_error=str(error))
                        record('invalid_final_response', reason='answer_json_contract', diagnostic=str(error), tools_executed=False)
                    else:
                        result.update(status='complete', answer=parsed_answer['answer'], structured_answer=parsed_answer, final_response_valid=True)
                    break
                if not calls:
                    history.append({**message, 'role': 'assistant'})
                    result.update(status='complete' if message.get('content', '').strip() else 'empty_response',
                                  answer=message.get('content', ''))
                    break
                # Commit only complete tool exchanges, retaining useful executed tools if time ends.
                accepted = {**message, 'role': 'assistant', 'tool_calls': []}
                history.append(accepted)
                for call in calls:
                    if clock() >= request_end:
                        break
                    function = call.get('function', {}) if isinstance(call, dict) else {}
                    name, args = function.get('name', ''), function.get('arguments', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            if name != 'private_scratchpad_update' or scratchpad is None:
                                raise  # Preserve the browser action contract.
                            # The private action rejects malformed arguments atomically below.
                            pass
                    if name == 'private_scratchpad_update' and scratchpad is not None:
                        result['scratchpad_calls'] += 1
                        record('scratchpad_update_attempt', agent=agent, arguments=args,
                               previous_tokens=scratchpad.tokens)
                        try:
                            output = scratchpad.update(agent, args, client, request_end, clock)
                        except Exception as error:
                            record('scratchpad_update_failed', error=f'{type(error).__name__}: {error}',
                                   retained=scratchpad.snapshot())
                            raise
                        record('scratchpad_update', response=output, retained=scratchpad.snapshot())
                    else:
                        exempt = notebook_quota_exempt and browser.is_notebook_action(agent, name, args)
                        if token_mode and quota_count() >= action_limit and not exempt:
                            result['limits_reached'].append('browser_actions')
                            if not notebook_quota_exempt:
                                break
                            output = {'error': 'Non-notebook browser allowance exhausted for this phase.'}
                            result['source_quota_rejections'] += 1
                            record('source_quota_rejection', name=name, arguments=args)
                        else:
                            output = browser.call(agent, name, args)
                            result['browser_calls'] += 1
                            if notebook_quota_exempt:
                                result['notebook_calls' if exempt else 'source_browser_calls'] += 1
                    if name != 'private_scratchpad_update' and isinstance(output, dict) and 'limit' in str(output.get('error', '')).lower():
                        result['limits_reached'].append('browser_limit')
                    accepted['tool_calls'].append(call)
                    history.append({'role': 'tool', 'tool_name': name, 'content': json.dumps(output),
                                    **({'tool_call_id': call['id']} if isinstance(call.get('id'), str) else {})})
                    record('tool', name=name, arguments=args, response=output)
                if not accepted['tool_calls']:
                    accepted.pop('tool_calls')
            else:
                result['limits_reached'].append('steps')
        except Exception as error:
            result.update(status='error', error=f'{type(error).__name__}: {error}')
            record('error', error=result['error'])
        result['elapsed_seconds'] = clock() - started
        result['deadline_overrun_seconds'] = max(0, clock() - deadline)
        if token_mode:
            if consumed >= token_limit:
                result['limits_reached'].append('generated_tokens')
            if quota_count() >= action_limit:
                result['limits_reached'].append('browser_actions')
            if 'wall_time' in result['limits_reached']:
                result['limits_reached'].append('safety_timeout')
                result['safety_timeout_hit'] = True
            result.setdefault('safety_timeout_hit', False)
            result['generated_tokens_remaining'] = token_limit - consumed
        result['limits_reached'] = sorted(set(result['limits_reached']))
        result['generated_tokens_observed'] = sum(r['eval_count'] for r in result['model_requests'] if type(r.get('eval_count')) is int and r['eval_count'] >= 0)
        result['token_accounting_complete'] = all(type(r.get('eval_count')) is int and r['eval_count'] >= 0 for r in result['model_requests'])
        record('result', result=result)
    return result


def run_phase_with_readiness(browser, client, history, text, phase, seconds, policy, run_dir,
                             index, qid, transitions, results, readiness_timeout, label, scratchpad=None, agent="agent-1", notebook_quota_exempt=False):
    """Shared session/diagnostic transition; recovery occurs before the phase clock."""
    print(f'{label}: readiness check (up to {readiness_timeout:g}s, outside phase window)', flush=True)
    transition_started = time.monotonic()
    readiness = getattr(client, 'ensure_ready', None)
    try:
        transition = (readiness(timeout=readiness_timeout) if callable(readiness)
                      else {'status': 'not_provided_by_transport'})
    except Exception as error:
        transitions.append({'phase': phase, 'agent': agent, 'question_id': qid, 'status': 'failed', 'timeout_seconds': readiness_timeout,
                            'elapsed_seconds': time.monotonic() - transition_started,
                            'error': f'{type(error).__name__}: {error}'})
        write_json(run_dir / 'transitions.json', transitions)
        raise
    transitions.append({**transition, 'phase': phase, 'agent': agent, 'question_id': qid, 'timeout_seconds': readiness_timeout,
                        'elapsed_seconds': time.monotonic() - transition_started})
    write_json(run_dir / 'transitions.json', transitions)
    window = (f"{getattr(policy, phase + '_generated_tokens')} generated tokens with {seconds:g}s safety cap"
              if policy.budget_mode == 'generated_tokens' else f'{seconds:g}s window')
    print(f'{label}: ready after {transitions[-1]["elapsed_seconds"]:.3f}s; starting {window}', flush=True)
    row = run_phase(browser, client, history, text, phase, seconds, policy,
                    run_dir / f'phase-{index:02d}.jsonl',
                    **({'agent': agent} if agent != 'agent-1' else {}),
                    **({'scratchpad': scratchpad} if scratchpad is not None else {}),
                    **({'notebook_quota_exempt': True} if notebook_quota_exempt else {}))
    if scratchpad is not None:
        write_json(run_dir / 'private-scratchpad.json', scratchpad.snapshot())
    row.update(question_id=qid, log_path=f'phase-{index:02d}.jsonl')
    results.append(row)
    write_json(run_dir / 'results.json', results)
    write_json(run_dir / 'history.json', history)
    print(f'{label}: {row["status"]}; elapsed={row.get("elapsed_seconds", 0):.3f}s, '
          f'browser_calls={row.get("browser_calls", 0)}, limits={row.get("limits_reached", [])}', flush=True)
    if row['status'] in ('error', 'context_exhausted'):
        detail = row.get('error') or ', '.join(row.get('limits_reached', [])) or row['status']
        raise RuntimeError(f'{phase} stopped: {detail} (phase log: {run_dir / row["log_path"]})')
    return row


def save_checkpoint(run_dir, browser, history, completed_questions, next_phase_index, next_question_id, scratchpad=None):
    """Publish a consistent host snapshot, including opt-in private memory."""
    run_dir = Path(run_dir)
    name = f'questions-{completed_questions:03d}'
    destination = run_dir / 'checkpoints' / name
    staging = run_dir / 'checkpoints' / ('.' + name + '.incomplete')
    source_names = ('settings.json', 'dataset.json', 'pages.json', 'results.json', 'transitions.json')
    # Fail before creating avoidable artifacts for checkable input problems.
    if destination.exists() or staging.exists():
        raise ValueError('Checkpoint already exists; preserve prior snapshot or failure diagnostics')
    for filename in source_names:
        if not (run_dir / filename).is_file():
            raise ValueError(f'Missing checkpoint prerequisite: {filename}')
    staging.mkdir(parents=True)
    try:
        for filename in source_names:
            shutil.copyfile(run_dir / filename, staging / filename)
        write_json(staging / 'history.json', history)
        if scratchpad is not None:
            write_json(staging / 'private-scratchpad.json', scratchpad.snapshot())
        with browser.lock:
            # SQLite backup includes audit/revisions and current shared source bodies.
            # views is also essential: click refers to old per-agent page IDs/link lists.
            with closing(sqlite3.connect(staging / 'wiki.sqlite3')) as snapshot:
                browser.db.backup(snapshot)
                if snapshot.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                    raise ValueError('Checkpoint SQLite integrity check failed')
            write_json(staging / 'browser.json', {
                'views': browser.views,
                'request_history_mode': browser.request_history_mode,
                'history_windows': browser.history_windows,
                'search_snippets': browser.search_snippets,
                'editable_title_marker': browser.editable_title_marker,
                'editable_urls': sorted(browser.editable_urls),
                'source_identities': browser.source_identities,
                'source_urls': browser.source_urls,
                'source_identity': browser.source_identity,
                'source_editor_revisions': browser.source_editor_revisions,
            })
        manifest = {
            'schema': 'timed-host-checkpoint-v2' if scratchpad is not None else 'timed-host-checkpoint-v1', 'status': 'complete',
            'completed_questions': completed_questions,
            'boundary': 'after_reflection_and_private_reset' if scratchpad is not None else 'after_reflection_and_any_scheduled_compaction',
            'next_phase_index': next_phase_index,
            'next_phase': 'answer' if next_question_id is not None else None,
            'next_question_id': next_question_id,
            'session_complete': next_question_id is None,
            'resume_implemented': True,
            'model_identity': 'settings.json:provenance.model',
            'schedule_and_configuration': 'settings.json',
            'external_raw_history': '../../phase-*.jsonl, ../../compaction-*.json, ../../history-*-compaction-*.json, ../../history-before-reset-*.json, ../../memory-reset-*.json',
            'not_serialized': ['model process', 'KV cache', 'in-flight generation', 'transport runtime'],
            'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(staging.iterdir()) if path.is_file()},
        }
        write_json(staging / 'checkpoint.json', manifest)
        staging.rename(destination)
    except BaseException as error:
        if staging.exists():
            write_json(staging / 'failure.json', {'status': 'incomplete', 'error': f'{type(error).__name__}: {error}'})
        raise
    return destination


def run_remaining_phases(run_dir, browser, client, history, tasks, schedule, topic, policy,
                         results, transitions, manifest, checkpoint_callback, start_index=0, scratchpad=None):
    sequence = [('preparation', policy.preparation_seconds,
                 PREPARATION_PROMPT.format(seconds=policy.preparation_seconds, topic=topic), None)]
    order = schedule['orders']['agent-1']
    for slot, qid in enumerate(order):
        reflection_prompt = FINAL_REFLECTION_PROMPT if slot == len(order) - 1 else REFLECTION_PROMPT
        sequence.extend([('answer', policy.answer_seconds,
                          ANSWER_PROMPT.format(seconds=policy.answer_seconds, question=tasks[qid]['question'],
                                               collection_url=tasks[qid]['collection_url']), qid),
                         ('reflection', policy.reflection_seconds,
                          reflection_prompt.format(seconds=policy.reflection_seconds), qid)])
    for index, (phase, seconds, text, qid) in enumerate(sequence):
        if index < start_index:
            continue
        label = f'[{index + 1}/{len(sequence)}] {phase}' + (f' question={qid}' if qid else '')
        readiness_timeout = (policy.initial_readiness_timeout_seconds if index == 0
                             else policy.readiness_timeout_seconds)
        row = run_phase_with_readiness(browser, client, history, text, phase, seconds, policy, run_dir,
                                       index, qid, transitions, results, readiness_timeout, label, scratchpad)
        if phase == 'answer':
            manifest['completed_questions'] += 1
        write_json(run_dir / 'manifest.json', manifest)
        if phase == 'reflection':
            retain_at_question_boundary(history, policy, run_dir, index, 'agent-1', qid)
            write_json(run_dir / 'history.json', history)
        if phase == 'reflection' and scratchpad is not None:
            before = json.dumps(history, sort_keys=True)
            write_json(run_dir / f'history-before-reset-{index:02d}.json', history)
            history[:] = scratchpad.reset_history(history[0], topic)
            write_json(run_dir / f'memory-reset-{index:02d}.json', {
                'boundary': 'after_reflection', 'phase_index': index, 'question_id': qid,
                'agent': scratchpad.agent, 'memory_mode': policy.memory_mode,
                'previous_history_sha256': hashlib.sha256(before.encode()).hexdigest(),
                'previous_history_path': f'history-before-reset-{index:02d}.json',
                'scratchpad': scratchpad.snapshot(), 'generation_performed': False,
                'active_history_sha256': hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()})
            write_json(run_dir / 'history.json', history)
        elif phase == 'reflection' and index + 1 < len(sequence):
            compact_between_questions(client, history, policy, run_dir, index)
            write_json(run_dir / 'history.json', history)
        if phase == 'reflection' and manifest['completed_questions'] % 5 == 0:
            next_index = index + 1
            next_qid = sequence[next_index][3] if next_index < len(sequence) else None
            checkpoint = save_checkpoint(run_dir, browser, history, manifest['completed_questions'],
                                         next_index, next_qid, scratchpad)
            # Publishing/remote durability happen outside question-phase clocks.
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint)
            print(f'Checkpoint saved: {checkpoint}', flush=True)


def run_timed_session(run_dir, records, topic, client, policy=TimedPolicy(), editable_sources=None, provenance=None, checkpoint_callback=None):
    pages, tasks, schedule, editable = validate_inputs(records, topic, policy, editable_sources)
    if not callable(client) or getattr(client, 'deadline_cancellation_guaranteed', False) is not True:
        raise ValueError('Timed runner requires a transport that stops generation before raising TimeoutError')
    if policy.memory_mode == 'in_context' and policy.compaction_enabled and not callable(getattr(client, 'count_context', None)):
        raise ValueError('Compaction requires native count_context transport')
    if policy.memory_mode == 'private_scratchpad' and (getattr(client, 'native_context_preflight', False) is not True
            or not callable(getattr(client, 'count_text', None))):
        raise ValueError('Private scratchpad requires native text tokenization and context preflight')
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError('Checkpoint callback must be callable')
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError('Run directory already exists')
    native_preflight = getattr(client, 'native_context_preflight', False) is True
    context_check = 'native rendered-prompt tokenization' if native_preflight else 'conservative estimated preflight'
    prompt = session_prompt(policy)
    history = [{'role': 'system', 'content': prompt}]
    scratchpad = PrivateScratchpad('agent-1', policy.scratchpad_tokens) if policy.memory_mode == 'private_scratchpad' else None
    if not native_preflight and context_estimate(history) >= policy.context_length:
        raise ValueError('Context cannot fit initial instructions')
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'settings.json', {'protocol': 'germanwiki-timed-v3', 'policy': asdict(policy),
               'system_prompt': prompt, 'topic': topic, 'schedule': schedule, 'provenance': provenance or {},
               'search_snippets': False, 'editable_title_marker': True,
               'checkpoint_interval_questions': 5,
               'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
               'dataset_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
               'context_policy': f'Raw host history preserved; memory mode {policy.memory_mode}; compaction applies only to in_context mode. Admission: {context_check}.',
               'editable_source': editable,
               'score': ('Accuracy within phase allowance; no token penalty' if policy.budget_mode == 'generated_tokens'
                         else 'Accuracy before deadline; no token penalty'),
               'generation_accounting': 'Native eval_count includes reasoning, tool arguments and final text. No input-token charge or rollover. Administrative compaction and warmup excluded.',
               'timing': 'Neutral readiness/recovery before each phase is timed separately. Normal prefill/generation and any within-answer reload count toward phase time. Cleanup may overrun; no overlap with later phases.'})
    write_json(run_dir / 'dataset.json', records)
    write_json(run_dir / 'pages.json', pages)
    results = []
    transitions = []
    manifest = {'status': 'in_progress', 'completed_questions': 0}
    write_json(run_dir / 'manifest.json', manifest)
    browser = None
    try:
        browser = Browser(pages, run_dir / 'wiki.sqlite3', editable_sources=editable, search_snippets=False, editable_title_marker=True,
                          request_history_mode=policy.request_history_mode)
        run_remaining_phases(run_dir, browser, client, history, tasks, schedule, topic, policy,
                             results, transitions, manifest, checkpoint_callback, scratchpad=scratchpad)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if browser is not None:
            browser.close()
        write_json(run_dir / 'manifest.json', manifest)
    return results


def validate_resume_checkpoint(checkpoint):
    """Validate configuration/corpus against the current timed runner before output."""
    settings = checkpoint.data['settings.json']
    policy = TimedPolicy(**settings['policy'])
    records = checkpoint.data['dataset.json']
    pages, tasks, schedule, _ = validate_inputs(records, 'Restored session', policy)
    if pages != checkpoint.data['pages.json'] or schedule != settings['schedule']:
        raise ValueError('Checkpoint corpus or schedule differs from the supported runner')
    return policy, tasks


def resume_timed_session(run_dir, checkpoint_path, client=None, provenance=None, checkpoint_callback=None):
    """Resume in a fresh directory. Complete checkpoints return a no-op without output."""
    checkpoint = load_checkpoint(checkpoint_path)
    try:
        policy, tasks = validate_resume_checkpoint(checkpoint)
        run_dir = Path(run_dir)
        if run_dir.exists():
            raise ValueError('Run directory already exists; use a fresh resume run ID')
        parent_run = checkpoint.path.parent.parent if checkpoint.path.parent.name == 'checkpoints' else checkpoint.path
        if run_dir.resolve().is_relative_to(parent_run):
            raise ValueError('Resume destination must be outside the parent run')
        if checkpoint.complete:
            return {'status': 'already_complete', 'completed_questions': policy.question_count,
                    'checkpoint': str(checkpoint.path), 'output_created': False}
        scratchpad = (PrivateScratchpad.restore(checkpoint.data['private-scratchpad.json'], policy.scratchpad_tokens)
                      if policy.memory_mode == 'private_scratchpad' else None)
        if scratchpad is not None and (getattr(client, 'native_context_preflight', False) is not True
                or not callable(getattr(client, 'count_text', None))):
            raise ValueError('Private scratchpad resume requires native text tokenization and context preflight')
        digest = checkpoint_model_digest(checkpoint)
        if (not callable(client) or getattr(client, 'deadline_cancellation_guaranteed', False) is not True
                or any(not callable(getattr(client, method, None))
                       for method in ('inspect_model', 'ensure_ready', 'count_context'))):
            raise ValueError('Resume requires model identity, readiness and native context counting transport')
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError('Checkpoint callback must be callable')
        metadata = client.inspect_model()
        if metadata.get('digest', '').removeprefix('sha256:') != digest.removeprefix('sha256:'):
            raise ValueError('Resume model digest differs from checkpoint')
        settings = json.loads(json.dumps(checkpoint.data['settings.json']))
        resume = {'parent_checkpoint': str(checkpoint.path),
                  'parent_checkpoint_sha256': checkpoint.manifest_sha256,
                  'parent_files_sha256': checkpoint.manifest['files_sha256'],
                  'completed_questions': checkpoint.manifest['completed_questions'],
                  'next_phase_index': checkpoint.manifest['next_phase_index'],
                  'next_question_id': checkpoint.manifest['next_question_id'],
                  'parent_source_hashes': settings.get('source_hashes'),
                  'parent_provenance': settings.get('provenance'),
                  'model_digest_verified': metadata,
                  'exact_kv_or_rng_replay': False}
        resume['parent_checkpoint_interval_questions'] = settings.get('checkpoint_interval_questions', 10)
        settings['checkpoint_interval_questions'] = 5
        settings['resume'] = resume
        settings['provenance'] = {**(provenance or settings.get('provenance', {})), 'model': metadata}
        settings['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(__file__).parent.glob('*.py')}
        history = checkpoint.data['history.json']
        results = checkpoint.data['results.json']
        transitions = checkpoint.data['transitions.json']
        for row in results:
            if row.get('log_path') and not Path(row['log_path']).is_absolute():
                row['log_path'] = str(parent_run / row['log_path'])
        manifest = {'status': 'in_progress', 'completed_questions': checkpoint.manifest['completed_questions'],
                    'resumed_from': str(checkpoint.path)}
        run_dir.mkdir(parents=True)
        try:
            write_json(run_dir / 'settings.json', settings)
            write_json(run_dir / 'resume.json', resume)
            for name, value in [('dataset.json', checkpoint.data['dataset.json']),
                                ('pages.json', checkpoint.data['pages.json']), ('history.json', history),
                                ('results.json', results), ('transitions.json', transitions)]:
                write_json(run_dir / name, value)
            with closing(sqlite3.connect(run_dir / 'wiki.sqlite3')) as output_db:
                checkpoint.browser.db.backup(output_db)
            checkpoint.browser.db.close()
            checkpoint.browser.db = sqlite3.connect(run_dir / 'wiki.sqlite3', check_same_thread=False)
            # Rebuild private context before starting Q11's original answer clock.
            started = time.monotonic()
            deadline = started + policy.initial_readiness_timeout_seconds
            warmup = {'status': 'in_progress', 'output_discarded': True}
            def remaining():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('Resume context warmup exceeded readiness allowance')
                return left
            try:
                warmup['readiness'] = client.ensure_ready(timeout=remaining())
                if scratchpad is not None:
                    info = client.count_text(scratchpad.text, timeout=remaining())
                    scratchpad.validate_count(info, scratchpad.text)
                    if info['tokens'] != scratchpad.tokens:
                        raise ValueError('Restored scratchpad native token count mismatch')
                    warmup['scratchpad_tokenization'] = info
                warmup['context'] = client.count_context(history, timeout=remaining())
                count = warmup['context'].get('prompt_tokens')
                if type(count) is not int or not 0 <= count < policy.context_length:
                    raise ValueError('Restored history cannot fit native context plus warmup token')
                warmed = client('agent-1', history, remaining(), num_predict=1, final_only=False)
                warmup['metadata'] = warmed.metadata
                warmup['status'] = 'complete'
            except BaseException as error:
                warmup.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                warmup['elapsed_seconds'] = time.monotonic() - started
                write_json(run_dir / 'resume-warmup.json', warmup)
            run_remaining_phases(run_dir, checkpoint.browser, client, history, tasks, settings['schedule'],
                                 settings.get('topic', 'Restored session'), policy, results, transitions, manifest,
                                 checkpoint_callback, checkpoint.manifest['next_phase_index'], scratchpad)
            manifest['status'] = 'complete'
        except BaseException as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            write_json(run_dir / 'manifest.json', manifest)
        return {'status': 'complete', 'completed_questions': manifest['completed_questions'],
                'checkpoint': str(checkpoint.path), 'output_created': True}
    finally:
        checkpoint.close()
