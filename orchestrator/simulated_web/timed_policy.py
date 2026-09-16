"""Validated configuration shared by timed runs and checkpoint loading."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimedPolicy:
    question_count: int = 10
    seed: int = 0
    preparation_seconds: float = 90
    answer_seconds: float = 20
    reflection_seconds: float = 20
    final_reserve_seconds: float = 5
    initial_readiness_timeout_seconds: float = 300
    readiness_timeout_seconds: float = 120
    context_length: int = 65536
    max_output_tokens: int = 32768
    max_steps: int = 500
    request_history_mode: str = "disabled"
    browser_retention: str = "full"
    memory_mode: str = "in_context"
    scratchpad_tokens: int = 4096
    compaction_enabled: bool = True
    compaction_trigger_fraction: float = 0.75
    compaction_retained_tokens: int = 16384
    compaction_output_tokens: int = 4096
    compaction_timeout_seconds: float = 180

    budget_mode: str = "elapsed_time"
    preparation_generated_tokens: int = 8192
    answer_generated_tokens: int = 2048
    reflection_generated_tokens: int = 8192
    final_reserve_tokens: int = 256
    preparation_browser_calls: int = 40
    answer_browser_calls: int = 10
    reflection_browser_calls: int = 80

    def __post_init__(self):
        if self.browser_retention not in ('full', 'question_boundary'):
            raise ValueError('Invalid browser retention policy')
        if self.budget_mode not in ('elapsed_time', 'generated_tokens'):
            raise ValueError('Invalid phase budget mode')
        token_limits = (self.preparation_generated_tokens, self.answer_generated_tokens,
                        self.reflection_generated_tokens, self.final_reserve_tokens)
        action_limits = (self.preparation_browser_calls, self.answer_browser_calls, self.reflection_browser_calls)
        if any(type(v) is not int or v < 1 for v in token_limits + action_limits):
            raise ValueError('Phase token and browser allowances must be positive integers')
        if self.final_reserve_tokens >= self.answer_generated_tokens:
            raise ValueError('Final token reserve must be smaller than answer allowance')
        if self.request_history_mode not in ("disabled", "shared", "isolated"):
            raise ValueError("Invalid request history mode")
        if (self.memory_mode not in ('in_context', 'private_scratchpad')
                or type(self.scratchpad_tokens) is not int or not 1 <= self.scratchpad_tokens <= 262144):
            raise ValueError('Invalid memory mode or scratchpad token cap')
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 31:
            raise ValueError('Seed must be an integer in0..2**31-1')
        if type(self.question_count) is not int or not 1 <= self.question_count <= 100:
            raise ValueError('Question count must be in1..100')
        if (type(self.compaction_enabled) is not bool
                or type(self.compaction_trigger_fraction) not in (int, float) or not 0 < self.compaction_trigger_fraction < 1
                or type(self.compaction_retained_tokens) is not int or not 1024 <= self.compaction_retained_tokens <= 16384
                or type(self.compaction_output_tokens) is not int or not 256 <= self.compaction_output_tokens <= 4096):
            raise ValueError('Invalid compaction settings')
        times = (self.preparation_seconds, self.answer_seconds, self.reflection_seconds,
                 self.final_reserve_seconds, self.initial_readiness_timeout_seconds, self.readiness_timeout_seconds, self.compaction_timeout_seconds)
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 3600 for v in times):
            raise ValueError('Phase times must be finite and in (0,3600]')
        if self.final_reserve_seconds >= self.answer_seconds:
            raise ValueError('Final time reserve must be smaller than answer window')
        if (type(self.context_length) is not int or not 4096 <= self.context_length <= 262144
                or type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 32768
                or type(self.max_steps) is not int or not 2 <= self.max_steps <= 500):
            raise ValueError('Invalid administrative context, per-request output or step limit')
        if self.memory_mode == 'private_scratchpad' and self.scratchpad_tokens > self.context_length - 2048:
            raise ValueError('Scratchpad cap must leave at least 2048 tokens of context headroom')
        if self.memory_mode == 'in_context' and self.compaction_enabled and self.compaction_retained_tokens >= self.context_length * self.compaction_trigger_fraction:
            raise ValueError('Compaction retained ceiling must be below trigger and context capacity')
