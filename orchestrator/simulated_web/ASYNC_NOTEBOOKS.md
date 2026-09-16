> Update September 13: the main 9b launcher now selects the separate [true parallel runner](PARALLEL_NOTEBOOKS.md), with two GPUs and one notebook authority. This document describes the preserved serialized implementation; its original launcher is now `run-serialized-fp8-notebooks.sh`.

# Experiment 9b: independent voluntary notebook loops

`modal_async_notebooks` is an opt-in protocol on the pinned HF FP8 transport.
It preserves existing runners. It is implemented and mock-tested; no live model
execution or cloud image build has verified this protocol or its new backend.

Each agent independently receives the same ordered questions, keeps its own
conversation and has its own question cursor and resource ceilings. Both first
questions are available before the first model request. Each agent may browse,
read a notebook, or append to its own notebook whenever it chooses while
working. A completed nonempty response without tool calls finalizes that
agent's answer. Neither notebook writing nor peer reading is required.
Budget exhaustion records an unanswered question and advances that agent.
There is no automatic post-answer publication phase or notebook write.

The scheduler runs one model request, executes that response's bounded tool
batch, then yields to the next ready agent. It starts with agent-1 then agent-2
and follows fair round-robin order. There is exactly one outstanding backend
request. This is independent, cooperatively scheduled progress with serialized
inference, **not simultaneous GPU inference**. A fast agent can reach its next
question while the peer remains on its previous question; no shared question
completion or publication barrier is imposed. Completed agents leave the queue.
Each request receives the agent's remaining question token allowance rather
than an artificial intermediate chunk limit. A long response can therefore
hold the inference slot longer; fairness is between completed requests.

This serialization is deliberate: the owned backend enforces deadlines by
terminating its entire process group. There is no concurrently generating peer
to kill silently. Any backend/host exception fails the run closed, preserves
artifacts and explicitly marks every unfinished agent interrupted. The CLI
does not automatically restart, extend budgets or resume a failed run.

## Notebook and source behavior

`append_notebook(text)` writes only the caller's notebook. A transaction creates
a new immutable entry and revision plus its audit record under the SQLite
connection's reentrant lock. It commits before returning the URL, visible author
and revision. Later tool calls can read it immediately. Existing entries remain
preserved. Malformed appends are rejected; interrupted transactions roll back.
The direct tool accepts up to 6,000 Unicode characters without URL encoding;
Chinese text and emoji have the same character limit as ASCII. The shared
literal-text helper also serves legacy append URLs, whose existing URL-length
validation remains unchanged.
The notebook directory retains ordinary pagination and links to individual
entries. `read_notebook` can read either notebook and explicit historical
revisions. Browser open/search remain available. This protocol does not expose
entry editing or source-editing tools; legacy `/append`, `/save`, and `/edit`
URLs direct callers to the atomic append tool.

There are no forced notebook views, shared request-history exposure or scheduled
notes. The request-history surface is disabled, avoiding an alternative forced
delivery channel. The initial prompt gives only the caller's notebook URL;
ordinary wiki navigation/search and supplied tools provide notebook affordances.
This is a voluntary-use experiment with advertised notebook capability, not a
claim of uncued discovery.

The same validated corpus/access manifest is used. `--source-access-mode` keeps
the existing `discovery_only` default or explicitly selects `hard` access. The
prepared launcher retains the previous five-baseball/five-Nanjing question set
and topic-based access manifest. That split does **not** establish that either
agent must cooperate to answer. No complementary-evidence corpus construction,
question correction, or matched no-sharing control is included.

## Budgets and persistence

All limits are per agent per question and configurable:

| Flag | Provisional default | Accounting |
| --- | ---: | --- |
| `--generated-tokens` | 4096 | All native generated tokens: reasoning, tool arguments, note text and final answer |
| `--tool-calls` | 16 | Executed calls including reads, appends and errors; excess requests receive a budget error |
| `--model-requests` | 32 | Request attempts; bounds empty/zero-token or repeatedly refused responses |
| `--active-seconds` | 600 | Readiness checks, native preflight and generation while holding the inference slot; excludes queue waiting |

These are provisional implementation defaults, not a user-selected optimum or
performance result. Initial shared backend startup is logged separately and
does not consume only the first agent's question budget. The six-hour cloud
job cap still applies to setup, all agents and persistence combined.
For the three-question launcher, the maximum generated-token ceiling is 24,576.
No separate uncharged note-generation call exists. Every request uses the
remaining question allowance; the former `--tokens-per-request` option is
removed. Normal completed tool calls still yield to the peer before another
request. A length-terminal response is charged, its partial text is preserved,
and its tool calls are not executed or finalized. That question ends with
`generation_limit_reached`; the agent proceeds to its next question rather
than restarting truncated reasoning as a new assistant turn. The result records
the requested allowance, backend-reported ceiling when available, native
generated count and remaining question allowance. This distinction matters if
the backend limit binds before the full question budget is consumed. Literal
continuation of partial reasoning/tool output is not attempted.

Native context handling preserves own text and current-question evidence. Older
browser response bodies from completed questions can be masked under pressure,
with original snapshots and omission records preserved. There is no full
conversation reset or model summary. If protected input cannot fit, execution
fails visibly. Notebook storage is independent of conversation masking.

`settings.json`, input/pages snapshots, atomic per-step `state.json`,
`histories.json`, `results.json`, `manifest.json`, SQLite, and `events.jsonl`
are retained. Step snapshots are diagnostic persistence, not a validated
restart/checkpoint format. The Modal worker commits them after each scheduling
step and on finalization; a container crash between commits can lose that
uncommitted interval. Previously committed artifacts are never deleted.

Host-only event sequence/UTC/monotonic timestamps distinguish:

- question availability and independent termination;
- queue duration and inference-slot ownership;
- raw model responses/native token charges and tool results;
- committed notebook publication/availability and actually returned notebook
  reads, including URL, revision and body hash;
- shared setup, truncated responses and explicit interruptions/failures.

A read event records delivery, not awareness or causal use. Publication timing
is logged just after commit while holding the lock; the exact commit instant
is not separately instrumented. No automatic uptake classifier is applied.
Event timing and host principals are not injected into notebook text.

## Prepared local validation

The descriptive launcher is
`research-log/async-fp8-notebooks-2026-09-13/run-async-fp8-notebooks.sh`.
With no arguments or `--validate-only`, it only validates local inputs and
prints settings. It uses distinct visible labels `researcher-a`/`researcher-b`
and fresh run ID `germanwiki-9b-async-fp8-3mlb-20260913`.
Trailing budget flags can override the provisional defaults. Use a different
`--run-id` for any subsequent attempt. Only after explicit execution
authorization, `--launch` spends compute; an empty HF cache additionally needs
`--download-model`. The current work did neither.

The backend restrictions, immutable checkpoint, intended BF16 KV, nonstream
partial-output limitation and live validation gaps are in [HF_FP8.md](HF_FP8.md).
The original experiment9 and paired-view FP8 entrypoints remain separate.
