"""Explicit, estimated preflight context management with native post-call accounting."""
from dataclasses import dataclass
import json
import math

from orchestrator.simulated_web.browser import TOOLS
from orchestrator.simulated_web.runner import ModelResponse


OMITTED_OBSERVATION = '[Observation omitted: older tool result outside the retained observation window.]'


class ContextError(ValueError):
    pass


def serialized_bound(messages):
    # UTF-8 bytes deliberately overestimate ordinary prose; template overhead is reserved separately.
    return len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))


@dataclass(frozen=True)
class ContextPolicy:
    context_length: int = 65536
    trigger_fraction: float = .8
    summary_tokens: int = 4096
    recent_tokens: int = 4096
    summary_generation_tokens: int = 8192
    observation_window: int = 0
    combined_phase_budgets: bool = False

    def __post_init__(self):
        if type(self.combined_phase_budgets) is not bool:
            raise ValueError('Combined phase budgets must be boolean')
        if type(self.observation_window) is not int or self.observation_window < 0:
            raise ValueError('Observation window must be a nonnegative integer (0 disables masking)')
        if (type(self.context_length) is not int or not 16384 <= self.context_length <= 262144
                or not math.isfinite(self.trigger_fraction) or not .5 <= self.trigger_fraction <= .9
                or any(type(v) is not int or not 1 <= v <= 8192 for v in
                       (self.summary_tokens, self.recent_tokens, self.summary_generation_tokens))
                or self.summary_tokens + self.recent_tokens + 2048 >= self.context_length * self.trigger_fraction):
            raise ValueError('Invalid compaction policy or insufficient retained-context headroom')


class ManagedContext:
    """One agent: summarize eligible history while preserving complete protected phases."""
    def __init__(self, client, policy):
        self.client, self.policy = client, policy
        self.max_output_tokens = min(getattr(client, 'max_output_tokens', 2048), 2048)
        self.last_native = None
        self.last_bytes = None
        self.last_messages = None
        self.last_render_mode = None
        self.pending_compaction = False
        self.bytes_per_token = 3.0
        self.allow_compaction = True
        self.log_path = None
        self.compactions = []
        self.phase_instruction = None
        self.phase = 'initial_research'
        self.pending_phase_start = None
        self.previous_reflection_anchor = None
        self.answer_anchor = None
        self.reflection_anchor = None
        self.summary_anchor = None
        # Keep object references so removed observations cannot be unmasked or IDs reused.
        self.masked_observations = {}
        self.block_pending = self.policy.combined_phase_budgets

    def model_messages(self, history):
        """Project raw history; one exchange is an assistant tool-call batch and all results.

        Masking is monotonic for surviving messages, including after phase removal or
        compaction. Text-only turns do not advance the window. Raw objects stay intact.
        """
        window = self.policy.observation_window
        if not window:
            return history
        starts = [i for i, message in enumerate(history)
                  if message.get('role') == 'assistant' and message.get('tool_calls')]
        cutoff = starts[-window] if len(starts) > window else 0
        projected = []
        for index, message in enumerate(history):
            if message.get('role') == 'tool' and message.get('content') != OMITTED_OBSERVATION:
                if index < cutoff:
                    self.masked_observations[id(message)] = message
                if id(message) in self.masked_observations:
                    message = {**message, 'content': OMITTED_OBSERVATION}
            projected.append(message)
        return projected

    @property
    def combined_token_limit(self):
        if not self.policy.combined_phase_budgets:
            return None
        return {'initial_research': 16000, 'answer': 4000, 'reflection': 8000}[self.phase]

    def prepare_block(self, agent, history, timeout):
        """Reserve next phase growth before generation; summarize only completed history."""
        if not self.block_pending:
            return
        reserve = self.combined_token_limit + 1024
        before = self.estimate(history)
        compacted = False
        if before + reserve >= self.policy.context_length:
            if self.log_path is None:
                raise ContextError('Pre-block compaction requires an audit log; history preserved')
            if self.phase == 'reflection':
                split = self.anchor_index(history, self.answer_anchor)
            elif self.phase == 'answer':
                split = self.anchor_index(history, self.answer_anchor)
            else:
                split = 1  # Initial task instruction is active, not completed history.
            prior_compactions = len(self.compactions)
            self.compact(agent, history, timeout, self.log_path,
                         split_override=split, reserved_tokens=reserve)
            compacted = len(self.compactions) > prior_compactions
        after = self.estimate(history)
        if self.log_path is not None:
            with self.log_path.open('a') as log:
                log.write(json.dumps({'event': 'pre_block_context', 'phase': self.phase,
                                      'estimated_input_before': before, 'estimated_input_after': after,
                                      'combined_growth_allowance': self.combined_token_limit,
                                      'instruction_and_template_safety': 1024, 'compacted': compacted,
                                      'fits': after + reserve < self.policy.context_length,
                                      'exact_token_limit': False}) + '\n')
        if after + reserve >= self.policy.context_length:
            raise ContextError('Pre-block context plus growth allowance cannot fit; phase not started, history preserved')
        self.block_pending = False

    @staticmethod
    def anchor_index(history, anchor):
        for index, message in enumerate(history):
            if message is anchor:
                return index
        raise ContextError('Protected phase boundary missing; history preserved')

    def begin_phase(self, phase, history):
        if phase not in ('answer', 'reflection'):
            raise ValueError('Expected answer or reflection phase')
        self.phase = phase
        self.block_pending = self.policy.combined_phase_budgets
        self.pending_phase_start = len(history)
        self.phase_instruction = None
        self.allow_compaction = True
        if phase == 'answer':
            self.answer_anchor = self.reflection_anchor = None

    def capture_phase_boundary(self, messages):
        if self.pending_phase_start is None:
            return
        if self.pending_phase_start >= len(messages) or messages[self.pending_phase_start].get('role') != 'user':
            raise ContextError('New phase is missing its user instruction; history preserved')
        anchor = messages[self.pending_phase_start]
        if self.phase == 'answer':
            self.answer_anchor = anchor
        else:
            self.reflection_anchor = anchor
        self.pending_phase_start = None

    def protected_start(self, history):
        anchor = (self.answer_anchor if self.policy.combined_phase_budgets else
                  self.previous_reflection_anchor or self.answer_anchor)
        if anchor is None:
            raise ContextError('Active answer/reflection boundary missing; history preserved')
        return self.anchor_index(history, anchor)

    def finish_reflection(self, history):
        start = self.anchor_index(history, self.answer_anchor)
        end = self.anchor_index(history, self.reflection_anchor)
        if start >= end:
            raise ContextError('Answer/reflection boundaries out of order; history preserved')
        del history[start:end]
        self.previous_reflection_anchor = self.reflection_anchor
        self.answer_anchor = self.reflection_anchor = None
        self.last_native = self.last_bytes = self.last_messages = None
        self.pending_compaction = False
        return end - start

    def estimate(self, messages, render_mode=None):
        messages = self.model_messages(messages)
        size = serialized_bound(messages)
        overhead = serialized_bound(TOOLS) + 1024
        if (self.last_messages is not None and messages[:len(self.last_messages)] == self.last_messages
                and (render_mode is None or render_mode == self.last_render_mode)):
            return self.last_native + math.ceil(max(0, size - self.last_bytes) / min(self.bytes_per_token, 3.0)) + 256
        return math.ceil(size / self.bytes_per_token) + math.ceil(overhead / 3)

    def __call__(self, agent, messages, timeout, **options):
        self.capture_phase_boundary(messages)
        render_mode = ('final_only' if options.get('final_only') else
                       'summary_only' if options.get('summary_only') else 'tools')
        if render_mode != self.last_render_mode:
            # Tool schemas and thinking/template switches change native input counts.
            # Keep calibration but never compare append counts across render modes.
            self.last_native = self.last_bytes = self.last_messages = None
        self.last_render_mode = render_mode
        self.prepare_block(agent, messages, timeout)
        if self.allow_compaction and self.log_path is not None:
            self.compact(agent, messages, timeout, self.log_path)
        messages = self.model_messages(messages)
        estimate = self.estimate(messages)
        allowance = min(options.get('num_predict', self.max_output_tokens), self.max_output_tokens)
        if estimate + allowance >= self.policy.context_length:
            detail = ('Protected previous reflection/current answer/current reflection exceeds available context'
                      if self.phase in ('answer', 'reflection') else 'Estimated input plus generation exceeds context')
            raise ContextError(detail + '; history preserved without truncation')
        response = self.client(agent, messages, timeout, **{**options, 'num_predict': allowance})
        if not isinstance(response, ModelResponse):
            raise ContextError('Context-managed calls require native metadata')
        if self.log_path is not None:
            with self.log_path.open('a') as log:
                log.write(json.dumps({'event': 'context_response', 'message': response.message,
                                      'metadata': response.metadata, 'estimated_input_tokens': estimate,
                                      'render_mode': render_mode}) + '\n')
        native = response.metadata.get('prompt_eval_count')
        if type(native) is not int or native < 1:
            raise ContextError('Missing native prompt token count; history preserved')
        response.metadata['context_accounting'] = {
            'estimated_input_tokens': estimate, 'actual_prompt_eval_count': native,
            'render_mode': render_mode,
            'observation_window': self.policy.observation_window,
            'observation_window_unit': 'assistant-tool-call-batch',
            'masked_tool_results': sum(m.get('role') == 'tool' and m.get('content') ==
                OMITTED_OBSERVATION for m in messages),
            'preflight_method': 'prior-native-plus-calibrated-UTF8-delta-or-calibrated-full-UTF8; template estimate',
            'exact_preflight': False}
        # A native count lower than the previous prompt on a strict append is evidence of lost input.
        if (self.last_messages is not None and messages[:len(self.last_messages)] == self.last_messages
                and native + 32 < self.last_native):
            raise ContextError('Native prompt count decreased on appended history; possible backend truncation')
        self.bytes_per_token = max(1.0, min(4.0, serialized_bound(messages) / native))
        self.last_native, self.last_bytes = native, serialized_bound(messages)
        self.last_messages = json.loads(json.dumps(messages))
        self.pending_compaction |= native >= math.ceil(self.policy.context_length * self.policy.trigger_fraction)
        if native + allowance >= self.policy.context_length:
            raise ContextError('Native input plus requested generation reached context boundary')
        return response

    def compact(self, agent, history, timeout, log_path, *, split_override=None, reserved_tokens=None):
        if not history:
            return
        estimate = self.estimate(history)
        if split_override is None and not self.pending_compaction and estimate < math.ceil(self.policy.context_length * self.policy.trigger_fraction):
            return
        reserve = self.max_output_tokens if reserved_tokens is None else reserved_tokens
        phase_retention = self.phase in ('answer', 'reflection') or split_override is not None
        if split_override is not None:
            split = split_override
        elif phase_retention:
            split = self.protected_start(history)
        else:
            # Initial research retains a bounded tail of complete user-led exchanges.
            starts = [i for i, message in enumerate(history) if i > 0 and message['role'] == 'user']
            split = len(history)
            for start in reversed(starts):
                if math.ceil(serialized_bound(self.model_messages(history)[start:]) / self.bytes_per_token) <= self.policy.recent_tokens:
                    split = start
                else:
                    break
        prefix, recent = history[1:split], history[split:]
        if not prefix or (len(prefix) == 1 and prefix[0] is self.summary_anchor):
            # Rewriting the same summary cannot release protected phase inputs.
            # Continue below the hard limit; the request guard reports overflow clearly.
            with log_path.open('a') as log:
                log.write(json.dumps({'event': 'compaction_deferred', 'phase': self.phase,
                                      'reason': 'No new eligible history beyond retained summary',
                                      'estimated_input_tokens': estimate,
                                      'protected_messages': len(recent)}) + '\n')
            return
        instruction = (
            'You are continuing the same task. Summarize the preceding conversation for your future self. '
            'Retain useful findings, sources, uncertainties, task constraints and discovered capabilities as you judge useful. '
            f'Return only the retained summary, at most {self.policy.summary_tokens} tokens. '
            'Do not answer a new question or call tools.')
        model_history = self.model_messages(history)
        request = [model_history[0], *model_history[1:split], {'role': 'user', 'content': instruction}]
        # A failed summary never mutates caller-owned history. Raw attempted summary is preserved.
        with log_path.open('a') as log:
            log.write(json.dumps({'event': 'compaction_initial', 'messages': request,
                                  'retained_recent': recent, 'phase': self.phase,
                                  'observation_window': self.policy.observation_window,
                                  'retention_policy': ('pre-block-completed-history' if split_override is not None else
                                      'active-answer-reflection' if self.policy.combined_phase_budgets and phase_retention else
                                      'previous-reflection-and-active-phases' if phase_retention else 'initial-recent-exchanges'),
                                  'estimated_input_tokens': self.estimate(request, render_mode='summary_only')}) + '\n')
            log.flush()
            if phase_retention and self.estimate([history[0], *recent]) + reserve >= self.policy.context_length:
                raise ContextError('Protected previous reflection/current answer/current reflection cannot fit; history preserved')
            summary_allowance = min(self.policy.summary_generation_tokens,
                                    self.policy.context_length - self.estimate(request, render_mode='summary_only') - 1024)
            if summary_allowance < 256:
                raise ContextError('Summary cannot fit full prefix plus allowance; history preserved')
            response = self.client(agent, request, timeout,
                                   num_predict=summary_allowance, summary_only=True)
            log.write(json.dumps({'event': 'compaction_response', 'message': response.message,
                                  'metadata': response.metadata}) + '\n')
            log.flush()
            text = response.message.get('content')
            count = response.metadata.get('eval_count')
            native = response.metadata.get('prompt_eval_count')
            if (response.metadata.get('done_reason') == 'length' or response.message.get('tool_calls')
                    or not isinstance(text, str) or not text.strip()
                    or type(count) is not int or not 0 <= count <= summary_allowance
                    or type(native) is not int or not 0 < native < self.policy.context_length
                    or native + count >= self.policy.context_length
                    or math.ceil(len(text.encode('utf-8')) / self.bytes_per_token) > self.policy.summary_tokens):
                raise ContextError('Summary failed validation or estimated retained-token limit; original history preserved')
            summary_message = {'role': 'user', 'content': 'Retained context from the preceding conversation:\n' + text}
            replacement = [history[0], summary_message, *recent]
            if self.phase_instruction is not None and not any(
                    message.get('content') == self.phase_instruction for message in recent):
                replacement.append({'role': 'user', 'content': self.phase_instruction})
            safe_limit = (self.policy.context_length if phase_retention else
                          self.policy.context_length * self.policy.trigger_fraction)
            if self.estimate(replacement) + reserve >= safe_limit:
                raise ContextError('Protected phases plus summary cannot fit after compaction; original history preserved')
            log.write(json.dumps({'event': 'compaction_commit', 'messages_before': len(history),
                                  'messages_after': len(replacement), 'phase': self.phase,
                                  'protected_messages': len(recent) if phase_retention else 0,
                                  'estimated_summary_tokens': math.ceil(len(text.encode('utf-8')) / self.bytes_per_token),
                                  'estimated_recent_tokens': math.ceil(serialized_bound(recent) / self.bytes_per_token),
                                  'retained_recent_is_phase_protected': phase_retention,
                                  'retained_token_limits_exact': False,
                                  'summary_native_generated_tokens': count, 'actual_prompt_eval_count': native}) + '\n')
            self.compactions.append({'prompt_eval_count': native, 'eval_count': count, 'status': 'complete'})
            history[:] = replacement
            self.summary_anchor = summary_message
            self.last_native = self.last_bytes = self.last_messages = None
            self.pending_compaction = False
