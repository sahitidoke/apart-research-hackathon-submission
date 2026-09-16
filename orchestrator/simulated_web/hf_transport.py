"""Owned vLLM HF FP8 transport. Live GPU compatibility remains unverified.

Every incomplete generation kills and verifies the owned process group. There
is no claim that HTTP disconnect alone cancels inference. Nonstream responses
have no recoverable partial text when interrupted; the deadline event records it.
"""
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

from orchestrator.simulated_web.browser import TOOLS
from orchestrator.simulated_web.hf_fp8 import MODEL, VLLM_VERSION, server_command, validate_snapshot
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse
from orchestrator.simulated_web.timed_transport import OwnedOllama, ContextExhausted, DeadlineExpired, check_loopback_port_available


def hf_messages(messages):
    """Map legacy histories to OpenAI IDs, preserving reasoning and tool order."""
    output, pending = [], []
    for index, message in enumerate(messages):
        role = message['role']
        row = {'role': role, 'content': message.get('content', '')}
        if role == 'assistant':
            if pending:
                raise ValueError('Unmatched tool history before assistant message')
            row['reasoning'] = message.get('thinking', '')
            # vLLM chat and tokenize paths both consume `reasoning` and pass it
            # to HF as reasoning_content (v0.24.0 chat_utils.py).
            calls = []
            for number, call in enumerate(message.get('tool_calls', [])):
                function = call['function']
                identifier = f'call_{index}_{number}'
                calls.append({'id': identifier, 'type': 'function', 'function': {
                    'name': function['name'], 'arguments': json.dumps(function['arguments'])}})
                pending.append((function['name'], identifier))
            if calls:
                row['tool_calls'] = calls
        elif role == 'tool':
            if not pending or pending[0][0] != message.get('tool_name'):
                raise ValueError('Tool history cannot be mapped without reordering')
            row['tool_call_id'] = pending.pop(0)[1]
        elif role not in ('user', 'system'):
            raise ValueError('Unsupported history role')
        elif pending:
            raise ValueError('Unmatched tool history before user/system message')
        output.append(row)
    if pending:
        raise ValueError('Unmatched tool calls at history boundary')
    return output


def chat_body(messages, tools, final_only):
    body = {'model': MODEL, 'messages': hf_messages(messages),
            'add_generation_prompt': True, 'continue_final_message': False,
            'add_special_tokens': False,
            'chat_template_kwargs': {'enable_thinking': not final_only,
                                     'reasoning_effort': 'medium', 'preserve_thinking': True}}
    if not final_only:
        body['tools'] = tools
    return body



def effective_tools(client):
    """Notebook schemas extend browser tools, matching the shared runner contract."""
    extras=getattr(client,'notebook_tool_schemas',[]) if getattr(client,'notebook_tools_enabled',False) else []
    if not isinstance(extras,list):raise ValueError('Notebook tool schemas must be a list')
    combined=[];by_name={}
    for schema in [*TOOLS,*extras]:
        if not isinstance(schema,dict) or not isinstance(schema.get('function'),dict) or not isinstance(schema['function'].get('name'),str):
            raise ValueError('Invalid tool schema')
        name=schema['function']['name']
        if name in by_name:
            if by_name[name]!=schema:raise ValueError('Conflicting tool schema: '+name)
            continue
        by_name[name]=schema;combined.append(schema)
    return combined

def native_count(value, context_length):
    tokens = value.get('tokens')
    if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
            or type(value.get('count')) is not int or value['count'] != len(tokens)
            or value.get('max_model_len') != context_length):
        raise RuntimeError('Unsupported native tokenization or context configuration')
    return len(tokens)


class OwnedVllm(OwnedOllama):
    """Reuse only process-group cleanup, readiness and serialization contracts."""
    def __init__(self, policy, snapshot, log_path, port=8000, seed=0, cuda_visible_devices=None):
        if sys.platform != 'linux' or shutil.which('vllm') is None:
            raise ValueError('HF transport requires Linux and installed vLLM')
        if importlib.metadata.version('vllm') != VLLM_VERSION:
            raise ValueError('HF transport requires pinned vLLM ' + VLLM_VERSION)
        if policy.memory_mode != 'in_context':
            raise ValueError('HF pilot currently supports notebook protocol without private scratchpad')
        server_command(snapshot, policy.context_length, port)
        if cuda_visible_devices is not None and (not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.isdecimal()):
            raise ValueError("A single numeric CUDA device ordinal is required")
        self.cuda_visible_devices = cuda_visible_devices
        self.policy, self.port, self.seed = policy, port, seed
        self.snapshot, self.log_path = Path(snapshot), Path(log_path)
        self.process = self.server_log = self.metadata = None
        self.ready = False
        self.events = []
        self.cancelled = threading.Event()
        self.stop_lock = threading.Lock()
        self.request_lock = threading.Lock()

    def inspect_model(self, timeout=30):
        if self.metadata is None:
            self.metadata = validate_snapshot(self.snapshot)
        return self.metadata

    def cancel(self):
        """Permanent worker cancellation; serialized against process creation."""
        with self.stop_lock:
            self.cancelled.set()
            self._stop_locked()

    def _start(self, deadline):
        with self.stop_lock:
            if getattr(self, 'cancelled', None) is not None and self.cancelled.is_set():
                raise RuntimeError('Owned vLLM worker cancelled')
            if self.process is not None and self.process.poll() is None:
                return
            if self.metadata is None:
                raise ValueError('Validate cached HF snapshot before server start')
            check_loopback_port_available(self.port)
            if self.server_log is None:
                self.server_log = self.log_path.open('ab', buffering=0)
            self.process = subprocess.Popen(server_command(self.snapshot, self.policy.context_length, self.port, **({'max_num_seqs':self.max_num_seqs} if hasattr(self,'max_num_seqs') else {})),
                env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                     **({'CUDA_VISIBLE_DEVICES': self.cuda_visible_devices} if getattr(self, 'cuda_visible_devices', None) is not None else {})},
                stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True)
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError('Owned vLLM exited or cancelled; see preserved server log')
            conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=0.2)
            try:
                conn.request('GET', '/v1/models')
                response = conn.getresponse()
                if response.status == 200:
                    models = json.loads(response.read(MAX_RESPONSE))['data']
                    if len(models) != 1 or models[0]['id'] != MODEL:
                        raise ValueError('Owned vLLM served-model identity mismatch')
                    self.events.append({'event': 'server_started', 'backend': 'vllm'})
                    return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                conn.close()
            time.sleep(0.02)
        raise TimeoutError('Owned vLLM startup deadline exhausted')

    def _request(self, agent, messages, timeout, *, num_predict, final_only=False,
                 _readiness=False, _count_only=False, _text_to_count=None, format_schema=None):
        if type(num_predict) is not int or num_predict < 1 or timeout <= 0:
            raise ValueError('Positive generation allowance and timeout required')
        if self.metadata is None or (not self.ready and not _readiness):
            raise ValueError('Identity inspection and between-phase readiness required')
        if _readiness and (messages != [{'role': 'user', 'content': 'Readiness check. Reply OK.'}]
                           or num_predict != 1 or not final_only):
            raise ValueError('Only fixed neutral readiness permitted')
        if format_schema is not None and (not final_only or _readiness or _count_only):
            raise ValueError('Schema requires ordinary final-only generation')
        deadline = time.monotonic() + timeout
        connection = None
        expired = threading.Event()
        externally_cancelled = threading.Event()
        socket_lock = threading.Lock()
        published_socket = None
        dispatched = acknowledged = False
        preflight = None
        active_path = None
        partial = {'content': '', 'thinking': ''}
        known_usage = None

        def abort(deadline_expired=False):
            nonlocal published_socket
            with socket_lock:
                if not expired.is_set() and not externally_cancelled.is_set():
                    (expired if deadline_expired or time.monotonic()>=deadline else externally_cancelled).set()
                if published_socket:
                    try:
                        published_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        def post(path, body):
            nonlocal connection, published_socket, active_path, dispatched
            active_path = path
            if connection:
                connection.close()
            left = deadline - time.monotonic()
            if externally_cancelled.is_set():raise RuntimeError('HF request externally cancelled')
            if left <= 0 or expired.is_set():
                abort(True)
                raise TimeoutError('HF request deadline reached')
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=left)
            connection.connect()
            with socket_lock:
                published_socket = connection.sock
                if externally_cancelled.is_set():raise RuntimeError('HF request externally cancelled')
                if expired.is_set():
                    raise TimeoutError('HF request publication expired')
            if path == '/v1/chat/completions':
                dispatched = True  # A failed/partial send is conservatively treated as possible generation.
            connection.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            limit = max(MAX_RESPONSE, self.policy.context_length * 128)
            raw = response.read(limit + 1)
            if externally_cancelled.is_set():raise RuntimeError('HF request externally cancelled')
            if expired.is_set() or time.monotonic() >= deadline:
                abort(True)
                raise TimeoutError('HF response deadline reached')
            if response.status != 200 or len(raw) > limit:
                raise RuntimeError(f'vLLM {path}: HTTP {response.status}: {raw[:512]!r}')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError('vLLM returned non-object response')
            return value

        timer = None
        registration = None
        try:
            register = getattr(self, '_register_abort', None)
            if callable(register):
                registration = register(abort)
            self._start(deadline)
            timer = threading.Timer(max(0, deadline - time.monotonic()), lambda: abort(True))
            timer.start()
            if _text_to_count is not None:
                value = post('/tokenize', {'model': MODEL, 'prompt': _text_to_count, 'add_special_tokens': False})
                count = native_count(value, self.policy.context_length)
                return ModelResponse({}, {'text_tokenization': {'tokens': count, 'method': 'vllm-native-tokenize',
                    'text_sha256': hashlib.sha256(_text_to_count.encode()).hexdigest()}})
            body = chat_body(messages, [] if final_only else effective_tools(self), final_only)
            count = native_count(post('/tokenize', body), self.policy.context_length)
            preflight = {'method': 'vllm-native-chat-tokenize', 'prompt_tokens': count,
                         'boundary_reserve_tokens': 1, 'tool_schema_policy': 'browser-plus-notebook-v1',
                         'tool_names': [tool['function']['name'] for tool in body.get('tools',[])], 'chat_request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
            num_predict = min(num_predict, self.policy.context_length - count - 1)
            preflight['requested_num_predict'] = num_predict
            self.events.append({'event': 'context_preflight', **preflight})
            if num_predict < 1:
                raise ContextExhausted(preflight)
            if _count_only:
                return ModelResponse({}, {'context_preflight': preflight})
            request = {**body, 'max_tokens': num_predict, 'seed': self.seed, 'stream': False,
                       'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'reasoning_effort': 'medium'}
            if format_schema is not None:
                request['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'answer', 'strict': True, 'schema': format_schema}}
            result = post('/v1/chat/completions', request)
            choices, usage = result.get('choices', []), result.get('usage', {})
            if (type(usage.get('completion_tokens')) is int and 0 <= usage['completion_tokens'] <= num_predict
                    and usage.get('prompt_tokens') == count):
                known_usage = {'eval_count':usage['completion_tokens'], 'prompt_eval_count':count,
                    'context_preflight':preflight, 'requested_num_predict':num_predict, 'backend':'vllm'}
            if (len(choices) != 1 or choices[0].get('finish_reason') not in ('stop', 'length', 'tool_calls')
                    or type(usage.get('completion_tokens')) is not int or not 0 <= usage['completion_tokens'] <= num_predict
                    or usage.get('prompt_tokens') != count):
                raise RuntimeError('vLLM native token/terminal contract mismatch; history preserved')
            message = choices[0]['message']
            partial = {'content': message.get('content') or '',
                       'thinking': message.get('reasoning') or message.get('reasoning_content') or ''}
            calls = message.get('tool_calls') or []
            if len(calls) > 8 or any(not isinstance(value, str) for value in partial.values()) or len(json.dumps(partial)) > MAX_RESPONSE:
                raise RuntimeError('vLLM response exceeds message bounds')
            converted = []
            for call in calls:
                function = call['function']
                arguments = json.loads(function['arguments'])
                if not isinstance(arguments, dict):
                    raise RuntimeError('Tool arguments must be an object')
                converted.append({'function': {'name': function['name'], 'arguments': arguments}})
            acknowledged = True
            return ModelResponse({**partial, 'tool_calls': converted}, {
                'done': True, 'done_reason': choices[0]['finish_reason'], 'eval_count': usage['completion_tokens'],
                'prompt_eval_count': count, 'requested_num_predict': num_predict,
                'context_preflight': preflight, 'backend': 'vllm',
                'final_only_contract_violation': bool(final_only and (partial['thinking'].strip() or converted))})
        except (TimeoutError, socket.timeout):
            abort(True)
            raise
        except Exception as error:
            error.native_usage = known_usage
            error.transport_failure = {'backend':'vllm','request_path':active_path,'generation_dispatched':dispatched,
                'completion_usage_available':known_usage is not None,'response_mode':'nonstream'}
            raise
        finally:
            if registration is not None:
                self._unregister_abort(registration)
            if timer:
                timer.cancel()
                timer.join()
            if connection:
                connection.close()
            if (dispatched and not acknowledged) or expired.is_set() or externally_cancelled.is_set():
                self.stop()
                self.events.append({'event': 'request_cancelled_killed', 'server_retained': False,
                                    'partial_text_available': False,'reason':'owner_cancellation' if externally_cancelled.is_set() else 'deadline' if expired.is_set() else 'incomplete_response'})
            if externally_cancelled.is_set():
                error=RuntimeError('HF request cancelled by shared owner; not a deadline expiry')
                error.native_usage=known_usage
                error.transport_failure={'backend':'vllm','request_path':active_path,'generation_dispatched':dispatched,
                    'completion_usage_available':known_usage is not None,'response_mode':'nonstream','reason':'owner_cancellation',
                    'partial_text_available':False}
                self.events.append({'event':'request_external_cancellation',**error.transport_failure})
                raise error
            if expired.is_set():
                error = DeadlineExpired(partial, max(0, time.monotonic() - deadline))
                error.context_preflight = preflight
                error.transport_timeout = {'backend':'vllm','request_path':active_path,'generation_dispatched':dispatched,
                    'completion_usage_available':False,'response_mode':'nonstream','requested_timeout_seconds':timeout,
                    'partial_text_available':False,'generation_tokens_if_not_dispatched':0 if not dispatched else None}
                self.events.append({'event':'request_timeout',**error.transport_timeout})
                raise error
