"""Owned Linux Ollama transport with deadline watchdog and fail-closed cleanup.

Routine deadlines disconnect the request and require a fresh completed task on
the owned single-slot llama-server. Unconfirmed cancellation kills the owned group.
The pinned backend contract still requires live latency/compatibility validation.
"""
import http.client
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from orchestrator.simulated_web.browser import TOOLS
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.private_memory import UPDATE_TOOL
from orchestrator.simulated_web.runner import MAX_RESPONSE, ModelResponse

MODEL = 'qwen3.8:27b-q4_K_M'
FINAL_PREFILL = {'role': 'assistant', 'content': ' ', 'thinking': ''}
FINAL_RENDER_SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


class DeadlineExpired(TimeoutError):
    def __init__(self, partial, cleanup_seconds):
        super().__init__(f'Owned inference stopped; cleanup took {cleanup_seconds:.3f}s; partial output has no complete native token count')
        self.partial = partial
        self.cleanup_seconds = cleanup_seconds


class ContextExhausted(ValueError):
    def __init__(self, diagnostics):
        super().__init__('Rendered native prompt leaves no generation capacity; full history preserved')
        self.diagnostics = diagnostics


def validate_model_metadata(models, expected_digest=None):
    matches = [row for row in models if row.get('name') == MODEL]
    if len(matches) != 1:
        raise ValueError(f'Exact requested model {MODEL} is not installed; no substitute is accepted')
    row = matches[0]
    details = row.get('details', {})
    if (not isinstance(row.get('digest'), str) or not row['digest']
            or details.get('quantization_level', '').upper() != 'Q4_K_M'
            or not str(details.get('parameter_size', '')).upper().startswith('27')):
        raise ValueError('Model must have a digest, 27B parameter size and Q4_K_M quantization')
    if expected_digest is not None and row['digest'].removeprefix('sha256:') != expected_digest.removeprefix('sha256:'):
        raise ValueError('Requested model digest mismatch')
    return row


def check_loopback_port_available(port):
    # Match the server's SO_REUSEADDR behavior: terminated connections in
    # TIME_WAIT must not look like a still-running listener. Do not set
    # SO_REUSEPORT, which would allow sharing an active listener's address.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


class OwnedOllama:
    deadline_cancellation_guaranteed = True
    native_context_preflight = True

    def __init__(self, policy, cache_path, log_path, port=11434, seed=0, expected_digest=None):
        if sys.platform != 'linux' or shutil.which('ollama') is None:
            raise ValueError('Owned timed transport requires Linux and an installed Ollama executable')
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('Invalid loopback port')
        self.policy, self.port, self.seed = policy, port, seed
        self.cache_path, self.log_path = Path(cache_path), Path(log_path)
        self.expected_digest = expected_digest
        self.process = None
        self.server_log = None
        self.metadata = None
        self.ready = False
        self.events = []
        self.stop_lock = threading.Lock()
        self.request_lock = threading.Lock()

    def _start(self, deadline):
        if self.process is not None and self.process.poll() is None:
            return
        # Never attach to or kill an operator's preexisting listener.
        check_loopback_port_available(self.port)
        environment = {**os.environ, 'OLLAMA_HOST': f'127.0.0.1:{self.port}',
                       'OLLAMA_MODELS': str(self.cache_path), 'OLLAMA_NUM_PARALLEL': '1',
                       'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_KEEP_ALIVE': '-1',
                       'OLLAMA_CONTEXT_LENGTH': str(self.policy.context_length),
                       'OLLAMA_FLASH_ATTENTION': '1', 'OLLAMA_KV_CACHE_TYPE': 'q8_0'}
        if self.server_log is None:
            self.server_log = self.log_path.open('ab', buffering=0)
        started = time.monotonic()
        self.process = subprocess.Popen(['ollama', 'serve'], env=environment,
                                        stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Owned Ollama exited during startup; see server log')
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=min(0.2, max(0.001, deadline - time.monotonic())))
            try:
                connection.request('GET', '/api/version')
                response = connection.getresponse()
                if response.status == 200:
                    version = json.loads(response.read(4096)).get('version')
                    if version != '0.33.3':
                        raise ValueError(f'Timed cancellation transport requires audited Ollama0.33.3, got {version!r}')
                    self.events.append({'event': 'server_started', 'elapsed_seconds': time.monotonic() - started})
                    return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        raise TimeoutError('Server startup exhausted request window')

    def stop(self):
        with self.stop_lock:
            self._stop_locked()

    def _stop_locked(self):
        process = self.process
        if process is None:
            return
        started = time.monotonic()
        # SIGKILL is deliberately used for hard containment, not graceful HTTP cancellation.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        # Linux descendants may be adopted by PID1. Zombies cannot compute; require
        # no live group member, rather than assuming reaping the parent is sufficient.
        until = time.monotonic() + 5
        while True:
            live = []
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                        live.append(int(entry.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            if not live:
                break
            if time.monotonic() >= until:
                raise RuntimeError('Cannot confirm inference process group is stopped; session must abort')
            time.sleep(0.01)
        self.process = None
        self.ready = False
        self.events.append({'event': 'server_group_stopped', 'cleanup_seconds': time.monotonic() - started})

    def close(self):
        try:
            self.stop()
        finally:
            if self.server_log is not None:
                self.server_log.close()
                self.server_log = None

    def inspect_model(self, timeout=30):
        self._start(time.monotonic() + timeout)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        try:
            connection.request('GET', '/api/tags')
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if response.status != 200 or len(data) > MAX_RESPONSE:
                raise ValueError('Cannot inspect local model metadata')
            row = validate_model_metadata(json.loads(data)['models'], self.expected_digest)
            self.expected_digest = row['digest']
            self.metadata = row
            return row
        finally:
            connection.close()

    def ensure_ready(self, timeout=120):
        """Neutral setup/recovery only; never receives an agent's task or history."""
        started = time.monotonic()
        if self.ready and self.process is not None and self.process.poll() is None:
            return {'status': 'already_ready', 'elapsed_seconds': time.monotonic() - started}
        if self.metadata is None:
            raise ValueError('Inspect model identity before readiness warmup')
        try:
            response = self('agent-1', [{'role': 'user', 'content': 'Readiness check. Reply OK.'}],
                            timeout, num_predict=1, final_only=True, _readiness=True)
        except Exception as error:
            event = {'event': 'readiness_failed', 'elapsed_seconds': time.monotonic() - started,
                     'error': f'{type(error).__name__}: {error}'}
            self.events.append(event)
            raise RuntimeError(f'Neutral readiness failed within {timeout:g}s: {error}') from error
        self.ready = True
        event = {'event': 'readiness_complete', 'status': 'warmed',
                 'elapsed_seconds': time.monotonic() - started, 'metadata': response.metadata}
        self.events.append(event)
        return event

    def _runner_identity(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('Owned Ollama is not running')
        matches = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) != self.process.pid or fields[0] in ('Z', 'X'):
                    continue
                argv = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
                if Path(argv[0]).name != 'llama-server':
                    continue
                host = argv[argv.index('--host') + 1]
                port = int(argv[argv.index('--port') + 1])
                parallel = int(argv[argv.index('-np') + 1])
                if host != '127.0.0.1' or not 1024 <= port <= 65535 or parallel != 1:
                    raise RuntimeError('Runner must use owned loopback and exactly one slot')
                matches.append((int(entry.name), fields[19], port))  # PID,starttime,port
            except (FileNotFoundError, ProcessLookupError):
                continue
        if len(matches) != 1:
            raise RuntimeError('Cannot identify exactly one owned llama-server')
        return matches[0]

    def _slot(self, identity, timeout):
        if self._runner_identity() != identity:
            raise RuntimeError('Runner identity changed during cancellation')
        connection = http.client.HTTPConnection('127.0.0.1', identity[2], timeout=timeout)
        try:
            connection.request('GET', '/slots')
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE + 1)
            if response.status != 200 or len(raw) > MAX_RESPONSE:
                raise RuntimeError('Owned runner /slots unavailable')
            slots = json.loads(raw)
            if (not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict)
                    or type(slots[0].get('is_processing')) is not bool
                    or type(slots[0].get('id_task')) is not int):
                raise RuntimeError('Owned runner returned unsupported slot schema')
            return {k: slots[0][k] for k in ('id_task', 'is_processing')}
        finally:
            connection.close()

    def _idle_baseline(self, deadline):
        identity = self._runner_identity()
        while time.monotonic() < deadline:
            slot = self._slot(identity, min(0.25, max(0.001, deadline - time.monotonic())))
            if not slot['is_processing']:
                return identity, slot['id_task']
            time.sleep(0.01)
        raise RuntimeError('Runner did not acknowledge idle before request')

    def _confirm_cancelled(self, baseline, timeout=2):
        started = time.monotonic()
        last = None
        reason = 'No pre-request idle baseline (cold initialization)'
        if baseline is not None:
            identity, previous_task = baseline
            try:
                # b10760 checks disconnect after a1s result wait. /slots results
                # wake that shared wait, resetting its relative timeout. Leave a
                # quiet interval, then make only sparse bounded observations.
                for offset in (1.1, 1.7):
                    if offset >= timeout:
                        break
                    time.sleep(max(0, started + offset - time.monotonic()))
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        break
                    try:
                        last = self._slot(identity, min(0.2, remaining))
                    except (TimeoutError, ConnectionError):
                        continue
                    if last['id_task'] > previous_task and not last['is_processing']:
                        event = {'event': 'request_cancelled_idle', 'elapsed_seconds': time.monotonic()-started,
                                 'runner_pid': identity[0], 'previous_task': previous_task, 'slot': last,
                                 'server_retained': True, 'quiet_seconds': 1.1,
                                 'poll_schedule_seconds': [1.1, 1.7]}
                        self.events.append(event)
                        return event
                time.sleep(max(0, started + timeout - time.monotonic()))
                reason = 'Fresh task idle acknowledgement not received within cancellation bound'
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
        self.events.append({'event': 'cancellation_fallback', 'reason': reason, 'last_slot': last,
                            'elapsed_seconds': time.monotonic()-started})
        self.stop()  # Raises on unconfirmed termination; no further request is safe.
        event = {'event': 'request_cancelled_killed', 'server_retained': False,
                 'elapsed_seconds': time.monotonic()-started, 'reason': reason}
        self.events.append(event)
        return event

    def _count_prompt(self, request, post_json, identity):
        started = time.monotonic()
        # Same Ollama renderer and message/tools/thinking configuration as inference.
        # Debug rendering runs no generation and truncate:false preserves the input.
        rendered = post_json(self.port, '/api/chat', {**request, 'stream': False,
                                                       '_debug_render_only': True})
        info = rendered.get('_debug_info')
        if (not isinstance(info, dict) or not isinstance(info.get('rendered_template'), str)
                or info.get('image_count', 0) != 0):
            raise RuntimeError('Native text-only render preflight unavailable; refusing heuristic fallback')
        prompt = info['rendered_template']
        final_prefill = request.get('messages', [])[-1:] == [FINAL_PREFILL]
        if final_prefill and not prompt.endswith(FINAL_RENDER_SUFFIX):
            raise RuntimeError('Final prefill renderer contract mismatch; refusing generation')
        system_prefix = prompt.split('<|im_end|>', 1)[0]
        if self._runner_identity() != identity:
            raise RuntimeError('Runner identity changed during prompt preflight')
        tokenized = post_json(identity[2], '/tokenize', {'content': prompt,
                                                       'add_special': True, 'parse_special': True})
        tokens = tokenized.get('tokens')
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise RuntimeError('Native tokenizer returned unsupported tokens')
        if self._runner_identity() != identity:
            raise RuntimeError('Runner identity changed during prompt tokenization')
        return {'method': 'ollama-debug-render-owned-tokenize', 'prompt_tokens': len(tokens),
                'add_special': True, 'parse_special': True, 'boundary_reserve_tokens': 1,
                'rendered_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'rendered_bytes': len(prompt.encode()),
                'system_prefix_sha256': hashlib.sha256(system_prefix.encode()).hexdigest(),
                'final_prefill': final_prefill, 'elapsed_seconds': time.monotonic() - started}

    def count_context(self, messages, timeout=15):
        """Native render/tokenize only; no generation or agent-visible endpoint."""
        return self('agent-1', messages, timeout, num_predict=1, _count_only=True).metadata['context_preflight']

    def count_text(self, text, timeout=15):
        """Count literal notes with the owned model tokenizer; no template/special tokens."""
        if not isinstance(text, str):
            raise ValueError('Native text counting requires a string')
        return self('agent-1', [], timeout, num_predict=1, _text_to_count=text).metadata['text_tokenization']

    def __call__(self, agent, messages, timeout, *, num_predict, final_only=False, _readiness=False, _count_only=False, _text_to_count=None, format_schema=None):
        if not self.request_lock.acquire(blocking=False):
            raise RuntimeError('Owned transport permits only one active request')
        try:
            return self._request(agent, messages, timeout, num_predict=num_predict, final_only=final_only, _readiness=_readiness, _count_only=_count_only, _text_to_count=_text_to_count, format_schema=format_schema)
        finally:
            self.request_lock.release()

    def _request(self, agent, messages, timeout, *, num_predict, final_only=False, _readiness=False, _count_only=False, _text_to_count=None, format_schema=None):
        if format_schema is not None and (not final_only or _readiness or _count_only or _text_to_count is not None):
            raise ValueError('Answer schema requires ordinary final-only generation')
        if self.metadata is None:
            raise ValueError('Model identity must be checked before phase execution')
        if _readiness and (messages != [{'role': 'user', 'content': 'Readiness check. Reply OK.'}]
                           or num_predict != 1 or not final_only):
            raise ValueError('Preflight bypass permits only the fixed neutral readiness request')
        if not self.ready and not _readiness:
            raise RuntimeError('Owned backend is not ready for native preflight; between-phase readiness required')
        started = time.monotonic()
        deadline = started + timeout
        expired = threading.Event()
        publication_lock = threading.Lock()
        request_socket = None
        acknowledged = False
        generation_dispatched = False
        preflight = None
        baseline = None
        partial = {'content': '', 'thinking': ''}
        connection = None
        request_events = len(self.events)

        def abort():
            expired.set()
            with publication_lock:
                if request_socket is not None:
                    try:
                        request_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        # Startup is bounded by its deadline and cannot race a new server against the watchdog.
        try:
            self._start(deadline)
        except TimeoutError:
            self.stop()
            raise DeadlineExpired(partial, time.monotonic() - deadline)
        if self.ready:
            try:
                baseline = self._idle_baseline(min(deadline, time.monotonic() + 0.5))
            except Exception:
                self.stop()
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self.stop()
            raise DeadlineExpired(partial, max(0, time.monotonic() - deadline))
        watchdog = threading.Timer(remaining, abort)
        watchdog.start()
        def open_connection(port):
            nonlocal connection, request_socket
            if connection is not None:
                connection.close()
            with publication_lock:
                request_socket = None
            left = deadline - time.monotonic()
            if left <= 0 or expired.is_set():
                expired.set()
                raise TimeoutError('Deadline reached before request publication')
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=left)
            connection.connect()
            with publication_lock:
                request_socket = connection.sock
                if expired.is_set():
                    request_socket.shutdown(socket.SHUT_RDWR)
                    raise TimeoutError('Deadline reached before request publication')
            return connection

        def post_json(port, path, body):
            conn = open_connection(port)
            conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            # Separate administrative wire bound for full-context rendered text/token IDs.
            wire_limit = max(MAX_RESPONSE, self.policy.context_length * 128)
            raw = response.read(wire_limit + 1)
            if expired.is_set() or time.monotonic() >= deadline:
                expired.set()
                raise TimeoutError('Deadline reached during native prompt preflight')
            if response.status != 200 or len(raw) > wire_limit:
                raise RuntimeError(f'Native prompt preflight {path}: HTTP {response.status}, '
                                   f'bytes={len(raw)}, wire_limit={wire_limit}; {raw[:512]!r}')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError('Native prompt preflight returned non-object JSON')
            return value

        try:
            if _text_to_count is not None:
                identity = baseline[0]
                if self._runner_identity() != identity:
                    raise RuntimeError('Runner identity changed before text tokenization')
                tokenized = post_json(identity[2], '/tokenize', {'content': _text_to_count,
                                                               'add_special': False, 'parse_special': False})
                tokens = tokenized.get('tokens')
                if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
                        or (_text_to_count and not tokens)):
                    raise RuntimeError('Native tokenizer returned unsupported text tokens')
                if self._runner_identity() != identity:
                    raise RuntimeError('Runner identity changed during text tokenization')
                info = {'method': 'owned-native-text-tokenize', 'tokens': len(tokens),
                        'add_special': False, 'parse_special': False,
                        'text_sha256': hashlib.sha256(_text_to_count.encode()).hexdigest(),
                        'elapsed_seconds': time.monotonic() - started}
                self.events.append({'event': 'text_tokenization', **info})
                return ModelResponse({}, {'text_tokenization': info})
            request = {'model': MODEL, 'messages': messages, 'stream': True, 'shift': False, 'truncate': False,
                       'options': {'seed': self.seed, 'num_ctx': self.policy.context_length,
                                   'num_predict': num_predict}}
            if format_schema is not None:
                request['format'] = format_schema
            if _readiness:
                request['think'] = False
            elif final_only:
                request['think'] = False
                # Nonempty raw whitespace selects parser content-mode; the
                # pinned renderer trims it and emits a closed empty think block.
                request['messages'] = [*messages, dict(FINAL_PREFILL)]
            else:
                request['tools'] = [*TOOLS, UPDATE_TOOL] if self.policy.memory_mode == 'private_scratchpad' else [*TOOLS,*getattr(self,'notebook_tool_schemas',NOTEBOOK_TOOLS)] if getattr(self,'notebook_tools_enabled',False) else TOOLS
            if self.ready:
                preflight = self._count_prompt(request, post_json, baseline[0])
                num_predict = min(num_predict, self.policy.context_length - preflight['prompt_tokens'] - 1)
                preflight['requested_num_predict'] = num_predict
                self.events.append({'event': 'context_preflight', **preflight})
                if num_predict < 1:
                    raise ContextExhausted(preflight)
                request['options']['num_predict'] = num_predict
            if _count_only:
                return ModelResponse({}, {'context_preflight': preflight})
            # Initial readiness alone uses the fixed neutral one-token request before
            # a loaded backend exists. Every actual phase request has native preflight.
            connection = open_connection(self.port)
            generation_dispatched = True
            connection.request('POST', '/api/chat', json.dumps(request), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f'Ollama HTTP {response.status}: {response.read(4096)!r}')
            received = 0
            calls = []
            metadata = None
            while True:
                line = response.readline(MAX_RESPONSE + 1)
                if expired.is_set() or time.monotonic() >= deadline:
                    expired.set()
                    raise TimeoutError('Deadline reached while receiving stream')
                if not line:
                    raise ValueError('Inference stream ended without terminal acknowledgement')
                received += len(line)
                if received > 16 * MAX_RESPONSE:
                    raise ValueError('Stream exceeds administrative wire limit')
                chunk = json.loads(line)
                if 'error' in chunk:
                    raise ValueError(f"Ollama generation error: {chunk['error']}")
                message = chunk.get('message', {})
                for key in partial:
                    partial[key] += message.get(key, '')
                calls.extend(message.get('tool_calls', []))
                if len(json.dumps(partial)) > MAX_RESPONSE or len(calls) > 8:
                    raise ValueError('Response exceeds administrative message limit')
                if chunk.get('done') is True:
                    metadata = {k: v for k, v in chunk.items() if k != 'message'}
                    acknowledged = True
                    break
            final_violation = bool(final_only and not _readiness and (partial['thinking'].strip() or calls))
            if preflight is not None:
                native_prompt = metadata.get('prompt_eval_count')
                if type(native_prompt) is not int or native_prompt > preflight['prompt_tokens']:
                    raise RuntimeError(f'Native prompt count violates preflight contract: {native_prompt!r} > '
                                       f"{preflight['prompt_tokens']}; history preserved")
            return ModelResponse({**partial, 'tool_calls': calls},
                                 {**metadata, 'requested_num_predict': num_predict,
                                  'context_preflight': preflight,
                                  'final_only_contract_violation': final_violation,
                                  'transport_events': self.events[request_events:]})
        except TimeoutError:
            expired.set()
            raise
        finally:
            watchdog.cancel()
            watchdog.join()
            if request_socket is not None:
                try:
                    request_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if connection is not None:
                connection.close()
            cancellation = None
            if generation_dispatched and (expired.is_set() or not acknowledged):
                try:
                    cancellation = self._confirm_cancelled(baseline)
                except Exception as error:
                    raise RuntimeError('Deadline cleanup failed; no further inference is safe') from error
            if expired.is_set():
                error = DeadlineExpired(partial, max(0, time.monotonic() - deadline))
                error.cancellation = cancellation or {'event': 'preflight_deadline_no_generation',
                                                       'server_retained': True}
                error.context_preflight = preflight
                raise error
