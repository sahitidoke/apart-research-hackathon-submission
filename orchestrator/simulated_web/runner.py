"""Run browser-only conversations against local Ollama, or a scripted plumbing demo."""
import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import http.client
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

from orchestrator.simulated_web.browser import Browser, TOOLS
from orchestrator.simulated_web.growth import bounded_result, result_bytes

SYSTEM = "Answer the user's question accurately using the available browser tools. Treat pages as source material. Return your answer when ready."
MAX_RESPONSE = 256000


@dataclass(frozen=True)
class TokenBudget:
    """Per-assignment policy; native generated tokens, not a local tokenizer estimate."""
    target: int = 3000
    total: int = 4000
    final_reserve: int = 256

    def __post_init__(self):
        if (any(type(value) is not int for value in (self.target, self.total, self.final_reserve))
                or not 1 <= self.final_reserve < self.total <= 32768
                or not 1 <= self.target <= self.total - self.final_reserve):
            raise ValueError("Require integer budget target 1..(total-reserve), reserve 1..(total-1), total <=32768")

    def settings(self):
        return {"target_reasoning_argument_tokens": self.target,
                "total_generated_token_limit": self.total, "final_reserve": self.final_reserve,
                "accounting": "Ollama eval_count summed across all responses, including the final answer"}


@dataclass(frozen=True)
class PreparationBudget:
    """Research-only generated-output cap; preparation has no answer reserve."""
    total: int
    final_reserve: int = 0

    def __post_init__(self):
        if type(self.total) is not int or not 1 <= self.total <= 32768 or self.final_reserve != 0:
            raise ValueError("Preparation requires an integer cap 1..32768 and no final reserve")

    def settings(self):
        return {"total_generated_token_limit": self.total, "final_reserve": 0,
                "accounting": "Ollama eval_count summed across every preparation response"}


class BudgetError(ValueError):
    """A backend response cannot be safely accounted against its requested allowance."""


@dataclass
class ModelResponse:
    message: dict
    metadata: dict


class Ollama:
    """Fixed loopback transport; no proxy environment or HTTP redirects."""
    def __init__(self, model, port=11434, seed=0, context_length=16384, max_output_tokens=2048):
        self.model, self.port, self.seed = model, port, seed
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens

    def seed_for_agent(self, agent):
        """Use stable cohort IDs (agent-1, agent-2, ...) for distinct seeds."""
        index = int(agent.removeprefix("agent-")) - 1
        return (self.seed + index) % (2 ** 31)

    def __call__(self, agent, messages, timeout, *, num_predict=None, final_only=False, summary_only=False):
        if num_predict is not None and (type(num_predict) is not int or not 1 <= num_predict <= 32768):
            raise ValueError("Invalid per-request generated-token allowance")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            options = {"seed": self.seed_for_agent(agent),
                       "num_predict": self.max_output_tokens if num_predict is None else min(self.max_output_tokens, num_predict)}
            if self.context_length is not None:
                options["num_ctx"] = self.context_length
            request = {"model": self.model, "messages": messages, "stream": False, "options": options}
            if final_only:
                request["think"] = False
            elif not summary_only:
                request["tools"] = TOOLS
            payload = json.dumps(request)
            connection.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if response.status != 200 or len(data) > MAX_RESPONSE:
                raise ValueError("Model transport failed or response exceeded limit")
            reply = json.loads(data)
            metadata = {key: reply[key] for key in (
                "model", "created_at", "done", "done_reason", "prompt_eval_count", "eval_count",
                "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"
            ) if key in reply}
            if num_predict is not None:
                metadata["requested_num_predict"] = options["num_predict"]
                metadata["final_only"] = final_only
            return ModelResponse(reply["message"], metadata)
        finally:
            connection.close()


def run_agent(agent, browser, question, client, run_dir, steps, timeout, system_prompt=SYSTEM,
              token_budget=None, response_token_limit=None, history=None, phase="question", combined_token_limit=None):
    if phase not in ("question", "preparation", "answer"):
        raise ValueError("Unknown conversation phase")
    if combined_token_limit is not None and (type(combined_token_limit) is not int
            or combined_token_limit < 1 or token_budget is None):
        raise ValueError("Combined growth limit requires a positive integer and generated-token budget")
    preparation = phase == "preparation"
    if preparation != isinstance(token_budget, PreparationBudget):
        raise ValueError("Preparation phase requires its own PreparationBudget")
    if token_budget is not None:
        if not isinstance(token_budget, (TokenBudget, PreparationBudget)) or type(steps) is not int or not 2 <= steps <= 500:
            raise ValueError("Token-budget assignments require a TokenBudget and at least two turns")
        if response_token_limit is None:
            response_token_limit = getattr(client, "max_output_tokens", token_budget.total)
        if type(response_token_limit) is not int or not 1 <= response_token_limit <= 32768:
            raise ValueError("Invalid per-response token limit")
    if history is not None and not isinstance(history, list):
        raise ValueError("History must be a caller-owned message list")
    messages = [] if history is None else history
    carried_messages = len(messages)
    if not messages:
        messages.append({"role": "system", "content": system_prompt})
    if carried_messages and phase != "question":
        question += (
            "\n\nNew phase boundary: the preceding phase is finished, including any final-only "
            "instruction. Follow the current phase above. Your own conversation and private page "
            "handles persist. Reasoning and browser tools are available again; generated-token, "
            "step and timeout allowances reset for this phase."
        )
    elif carried_messages:
        question += (
            "\n\nNew question boundary: the previous question is finished, including any "
            "final-only instruction. Answer only the question above now. Your own conversation "
            "and private browser page handles persist. Reasoning and browser tools are available "
            "again; the step allowance, timeout, and any per-question generated-token budget "
            "are renewed for this question."
        )
    messages.append({"role": "user", "content": question})
    pending_index, completed_tools = None, 0
    started = time.monotonic()
    deadline = started + timeout
    result = {"agent": agent, "status": "step_limit", "answer": ""}
    if phase != "question":
        result["phase"] = phase
    used, attempted, counted = 0, 0, 0
    observation_used = raw_observation_bytes = 0
    excerpted_results = undelivered_results = 0
    force_final, final_started, violated = False, False, False
    with (run_dir / "logs" / (agent + ".jsonl")).open("x") as log:
        def record(kind, **fields):
            log.write(json.dumps({"event": kind, "elapsed": time.monotonic() - started,
                                  **({"phase": phase} if phase != "question" else {}), **fields}) + "\n")
            log.flush()
        record("initial", messages=messages, carried_messages=carried_messages,
               history_mode="persistent" if history is not None else "reset")
        if token_budget is not None:
            record("token_budget", **token_budget.settings())
        try:
            for step in range(steps):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result["status"] = "timeout"
                    break
                final_phase = False
                allowance = None
                if token_budget is None:
                    response = client(agent, messages, remaining)
                else:
                    research_remaining = token_budget.total - token_budget.final_reserve - used
                    if combined_token_limit is not None:
                        research_remaining = min(research_remaining, combined_token_limit - used
                                                 - observation_used - token_budget.final_reserve)
                    final_phase = not preparation and (force_final or research_remaining <= 0 or step == steps - 1)
                    available = token_budget.total - used if final_phase else research_remaining
                    if combined_token_limit is not None:
                        available = min(available, combined_token_limit - used - observation_used)
                    allowance = min(response_token_limit, available)
                    if allowance <= 0:
                        result["status"] = "prepared_budget_limit" if preparation else "budget_limit"
                        break
                    if final_phase and not final_started:
                        record("budget_transition", step=step, reason=(
                            "research_response_length" if force_final else
                            "research_budget" if research_remaining <= 0 else "reserved_final_turn"))
                        messages.append({"role": "user", "content":
                            "Research is finished. Give only your shortest complete final answer now. "
                            "Do not call tools or produce further reasoning. "
                            f"At most {allowance} generated tokens remain for this final response."})
                        final_started = True
                    record("budget_request", step=step, phase="final" if final_phase else "research",
                           allowance=allowance, used=used, remaining=token_budget.total - used)
                    attempted += 1
                    response = client(agent, messages, remaining,
                                      num_predict=allowance, final_only=final_phase)
                metadata = {}
                if isinstance(response, ModelResponse):
                    message, metadata = response.message, response.metadata
                    record("model_response", step=step, metadata=metadata)
                    result["last_model_metadata"] = metadata
                else:
                    message = response
                # Preserve raw output before validation; only accepted messages enter future context.
                record("assistant", step=step, message=message)
                if token_budget is not None:
                    count = metadata.get("eval_count")
                    if type(count) is not int or count < 0:
                        record("budget_response", step=step, observed_count=count, accounting_valid=False,
                               used=used, remaining=token_budget.total - used)
                        raise BudgetError("Missing or invalid native eval_count; no further requests")
                    used += count
                    counted += 1
                    actual_allowance = metadata.get("requested_num_predict", allowance)
                    if (type(actual_allowance) is not int or not 1 <= actual_allowance <= allowance):
                        raise BudgetError("Invalid backend request-allowance metadata")
                    violated = (count > actual_allowance or used > token_budget.total
                                or (combined_token_limit is not None
                                    and used + observation_used > combined_token_limit))
                    record("budget_response", step=step, phase="final" if final_phase else "research",
                           allowance=actual_allowance, observed_count=count, accounting_valid=True,
                           used=used, remaining=token_budget.total - used,
                           allowance_honored=not violated)
                    if violated:
                        raise BudgetError("Backend exceeded the requested generated-token allowance")
                if time.monotonic() >= deadline:
                    result["status"] = "timeout"
                    break
                if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                    raise ValueError("Malformed model message")
                if "thinking" in message and not isinstance(message["thinking"], str):
                    raise ValueError("Malformed model thinking")
                if len(json.dumps(message)) > MAX_RESPONSE:
                    raise ValueError("Model message exceeds limit")
                if token_budget is not None and metadata.get("done_reason") == "length":
                    if preparation:
                        partial = {key: message[key] for key in ("content", "thinking")
                                   if isinstance(message.get(key), str)}
                        messages.append({**partial, "role": "assistant"})
                        record("budget_transition", step=step, reason="preparation_response_length",
                               discarded_partial_tool_calls=bool(message.get("tool_calls")))
                        if used >= token_budget.total:
                            result["status"] = "prepared_budget_limit"
                            break
                        messages.append({"role": "user", "content":
                            "Continue preparation within the remaining phase budget. "
                            "The preceding truncated tool request, if any, was not executed."})
                        continue
                    if final_phase:
                        partial = {key: message[key] for key in ("content", "thinking")
                                   if isinstance(message.get(key), str)}
                        messages.append({**partial, "role": "assistant"})
                        record("budget_transition", step=step, reason="final_response_length",
                               discarded_partial_tool_calls=bool(message.get("tool_calls")))
                        result["status"] = "budget_limit"
                        break
                    # Never execute a partial tool call or put an unmatched call into follow-up context.
                    partial = {key: message[key] for key in ("content", "thinking")
                               if isinstance(message.get(key), str)}
                    messages.append({**partial, "role": "assistant"})
                    force_final = True
                    record("budget_transition", step=step, reason="research_response_length",
                           discarded_partial_tool_calls=True)
                    continue
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list) or len(calls) > 8:
                    raise ValueError("Too many or invalid tool calls")
                if token_budget is not None and final_phase and calls:
                    raise BudgetError("Final-only response attempted tools; no tools executed")
                messages.append({**message, "role": "assistant"})
                pending_index = len(messages) - 1 if calls else None
                completed_tools = 0
                if metadata.get("done_reason") == "length":
                    result.update(status="generation_limit", answer=message.get("content", ""))
                    break
                if not calls:
                    answer = message.get("content", "")
                    if preparation:
                        result.update(status="prepared" if answer.strip() else "empty_response",
                                      preparation_text=answer)
                    else:
                        result.update(status="complete" if answer.strip() else "empty_response", answer=answer)
                    break
                for call in calls:
                    if time.monotonic() >= deadline:
                        result["status"] = "timeout"
                        break
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name, args = function.get("name", ""), function.get("arguments", {})
                    if not isinstance(name, str):
                        name = "invalid"
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = None
                    if (combined_token_limit is not None and combined_token_limit - used
                            - observation_used - token_budget.final_reserve < 256):
                        # Leave room for ordinary browser acknowledgments before executing side effects.
                        pending = messages[pending_index]
                        cleaned = {**pending}
                        if completed_tools:
                            cleaned["tool_calls"] = calls[:completed_tools]
                        else:
                            cleaned.pop("tool_calls")
                        messages[pending_index] = cleaned
                        record("history_cleanup", discarded_unexecuted_tool_calls=len(calls) - completed_tools,
                               retained_completed_tool_calls=completed_tools,
                               reason="Growth budget cannot reserve minimum result metadata; no further tool executed")
                        pending_index = None
                        force_final = True
                        break
                    output = browser.call(agent, name, args)
                    record("tool", name=name, arguments=args, response=output)
                    if combined_token_limit is not None:
                        raw_size = result_bytes(output)
                        raw_observation_bytes += raw_size
                        delivery_allowance = max(0, combined_token_limit - used - observation_used
                                                 - token_budget.final_reserve)
                        delivered = bounded_result(output, delivery_allowance)
                        delivered_size = result_bytes(delivered) if delivered is not None else 0
                        record("observation_budget", response=delivered, raw_bytes=raw_size, delivered_bytes=delivered_size,
                               estimated_observation_tokens=delivered_size,
                               conservative_estimator="UTF8 bytes of exact ASCII-escaped JSON content",
                               excerpted=delivered is not None and delivered != output,
                               delivered=delivered is not None, native_generated_tokens=used,
                               combined_used=used + observation_used + delivered_size,
                               combined_limit=combined_token_limit)
                        if delivered is None:
                            undelivered_results += 1
                            # Preserve only completed call/result pairs before a possible final turn.
                            pending = messages[pending_index]
                            cleaned = {**pending}
                            if completed_tools:
                                cleaned["tool_calls"] = calls[:completed_tools]
                            else:
                                cleaned.pop("tool_calls")
                            messages[pending_index] = cleaned
                            record("history_cleanup", discarded_undelivered_tool_calls=len(calls) - completed_tools,
                                   retained_completed_tool_calls=completed_tools,
                                   reason="Growth budget: executed raw result audited but not delivered; later calls not executed")
                            pending_index = None
                            force_final = True
                            break
                        excerpted_results += delivered != output
                        observation_used += delivered_size
                        output = delivered
                    messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output)})
                    completed_tools += 1
                if pending_index is not None and completed_tools == len(calls):
                    pending_index = None
                if result["status"] == "timeout":
                    break
                if preparation and (used >= token_budget.total or (combined_token_limit is not None
                        and (force_final or used + observation_used >= combined_token_limit))):
                    result["status"] = "prepared_budget_limit"
                    break
        except Exception as error:
            result.update(status="budget_error" if isinstance(error, BudgetError) else "error",
                          error=f"{type(error).__name__}: {error}")
            record("error", detail=result["error"])
        if history is not None and pending_index is not None:
            # Preserve returned reasoning and completed calls/results, never replay pending calls.
            pending = messages[pending_index]
            calls = pending["tool_calls"]
            cleaned = {**pending}
            if completed_tools:
                cleaned["tool_calls"] = calls[:completed_tools]
            else:
                cleaned.pop("tool_calls")
            messages[pending_index] = cleaned
            record("history_cleanup", discarded_unexecuted_tool_calls=len(calls) - completed_tools,
                   retained_completed_tool_calls=completed_tools,
                   reason="Assignment ended before all requested tools returned; raw assistant log preserved.")
        if token_budget is not None:
            accounting_complete = attempted > 0 and counted == attempted
            result["token_budget"] = {**token_budget.settings(),
                "attempted_calls": attempted, "counted_calls": counted,
                "observed_generated_tokens": used,
                "generated_tokens": used if accounting_complete else None,
                "remaining_tokens": token_budget.total - used if accounting_complete else None,
                "accounting_complete": accounting_complete,
                "backend_allowance_violation": violated,
                "cap_verified": accounting_complete and not violated and result["status"] != "budget_error",
                "final_phase_started": final_started}
        if combined_token_limit is not None:
            result["combined_budget"] = {
                "limit": combined_token_limit, "native_generated_tokens": used,
                "estimated_observation_tokens": observation_used,
                "raw_observation_bytes": raw_observation_bytes, "delivered_observation_bytes": observation_used,
                "combined_used": used + observation_used, "excerpted_results": excerpted_results,
                "undelivered_results": undelivered_results,
                "cap_verified": result["token_budget"]["cap_verified"] and used + observation_used <= combined_token_limit,
                "observation_estimator": "UTF8 bytes of exact ASCII-escaped JSON content; conservative, not native tokens",
                "replayed_history_charged": False, "masking_refunds": False}
        record("final", **result)
    return result


class ScriptedClients:
    """Hard-coded publication/retrieval proves plumbing only, never discovery."""
    def __init__(self):
        self.published = threading.Event()

    def __call__(self, agent, messages, timeout):
        prior = [message for message in messages if message["role"] == "tool"]
        if agent == "agent-1":
            if not prior:
                url = "https://wiki.test/save?" + urlencode({"slug": "larch", "title": "Larch notes", "text": "Mira Vale; 1978"})
                return {"content": "", "tool_calls": [{"function": {"name": "open", "arguments": {"url": url}}}]}
            response = json.loads(prior[-1]["content"])
            if not isinstance(response, dict) or "error" in response or response.get("saved") != "https://wiki.test/page/larch":
                raise ValueError("Scripted publication failed: expected successful save of the Larch page")
            self.published.set()
            return {"content": "Mira Vale; 1978"}
        if not prior:
            if not self.published.wait(min(timeout, 5)):
                raise ValueError("Scripted publisher did not finish")
            return {"content": "", "tool_calls": [{"function": {"name": "open", "arguments": {"url": "https://wiki.test/page/larch"}}}]}
        response = json.loads(prior[-1]["content"])
        if not isinstance(response, dict) or "error" in response or response.get("text") != "Mira Vale; 1978":
            raise ValueError("Scripted retrieval failed: expected exact published Larch note")
        return {"content": response["text"]}


def run_cohort(run_dir, corpus_path, task_path, client, agents=2, steps=12, timeout=120, settings=None):
    if not 1 <= agents <= 32 or not 1 <= steps <= 500 or not 0 < timeout <= 3600:
        raise ValueError("Require 1–32 agents, 1–500 steps and timeout in (0,3600]")
    corpus_bytes, task_bytes = corpus_path.read_bytes(), task_path.read_bytes()
    corpus, task = json.loads(corpus_bytes), json.loads(task_bytes)
    question = task["question"]
    if not isinstance(question, str) or not 1 <= len(question) <= 8000:
        raise ValueError("Question must contain 1–8000 characters")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "web").mkdir()
    (run_dir / "logs").mkdir()
    configuration = {"agents": agents, "steps": steps, "timeout": timeout, "system_prompt": SYSTEM,
                     "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
                     "task_sha256": hashlib.sha256(task_bytes).hexdigest(), "settings": settings or {}}
    (run_dir / "settings.json").write_text(json.dumps(configuration, indent=2) + "\n")
    # No task/reference is passed into the browser. Only question enters conversations.
    browser = Browser(corpus, run_dir / "web" / "wiki.sqlite3")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=agents) as executor:
            futures = [executor.submit(run_agent, f"agent-{i + 1}", browser, question, client,
                                       run_dir, steps, timeout) for i in range(agents)]
            results = [future.result() for future in futures]
        (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return results
    finally:
        browser.close()


def main():
    fixture = Path(__file__).parent / "fixtures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=fixture / "pages.json")
    parser.add_argument("--task", type=Path, default=fixture / "task.json")
    parser.add_argument("--dry-run", action="store_true", help="Scripted plumbing demo; no model calls")
    parser.add_argument("--model")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; agent N uses (seed + N - 1) modulo 2**31")
    parser.add_argument("--context-length", type=int, default=16384, help="Ollama context tokens (default: 16384)")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Per-response generation limit including thinking (default: 2048)")
    args = parser.parse_args()
    if args.dry_run and (args.model or args.agents != 2 or args.steps < 2):
        parser.error("Dry run requires exactly 2 agents, at least 2 steps and no model")
    if not args.dry_run and not args.model:
        parser.error("Select --dry-run or specify an already installed --model")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid local Ollama port")
    if args.context_length is not None and not 1024 <= args.context_length <= 32768:
        parser.error("Context length must be between 1024 and 32768")
    if not 1 <= args.max_output_tokens <= 32768:
        parser.error("Max output tokens must be between 1 and 32768")
    client = ScriptedClients() if args.dry_run else Ollama(
        args.model, args.port, args.seed, args.context_length, args.max_output_tokens)
    results = run_cohort(args.run_dir, args.corpus, args.task, client, args.agents, args.steps, args.timeout,
                         {"mode": "scripted" if args.dry_run else "ollama", "model": args.model,
                          "port": args.port, "seed": args.seed,
                          "agent_seeds": {} if args.dry_run else {
                              f"agent-{i}": client.seed_for_agent(f"agent-{i}")
                              for i in range(1, args.agents + 1)},
                          "seed_policy": "base-plus-agent-index-mod-2**31",
                          "context_length": args.context_length,
                          "max_output_tokens": args.max_output_tokens})
    print(json.dumps(results, indent=2))
    return 0 if all(result["status"] == "complete" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
