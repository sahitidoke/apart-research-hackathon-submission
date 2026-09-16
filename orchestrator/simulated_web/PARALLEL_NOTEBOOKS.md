# Parallel 9b (implemented, not launched)

`modal_parallel_notebooks` requests **one Modal container with two H100 GPUs**,
8 CPU cores and 128 GiB host memory. Two independent model replicas run on
CUDA devices 0 and 1, ports 8000 and 8001, in separately owned process groups.
Environment overrides are passed only to each child process; host environment
and peer device assignment are not mutated. Each replica uses the same pinned
HF FP8 model, context, tokenizer, tool schema and seed 0. Cache checks precede
output creation and model startup. The six-hour job bound begins before any
optional explicit download, checksum checks or server startup. Maximum allocated
GPU time is 12 GPU-hours, not an expected cost/runtime. Memory fit and concurrent
live operation remain unverified.

The coordinator owns the sole browser, SQLite connection, event sequence and
agent state. It dispatches independent inference futures without waiting for
the peer's request. Completion order determines tool processing; tool batches
run on the authority thread. Notebook appends commit atomically before returning,
and subsequent reads see them immediately. There is no replicated notebook DB,
remote-volume synchronization channel, or SQLite access from inference workers.
A peer already generating does not receive mid-generation updates; it can see
published content on its next voluntary read. When both futures are ready in the
same scheduler poll, the stable agent order breaks the tie. Initial question
availability is shared; subsequent answers/questions have no peer barrier.

Each in-flight request gets a private history copy and per-worker bounded-context
artifacts. Persistence reads only coordinator-owned histories, which are replaced
with the fitted history when that future completes. Original observations remain
in worker artifacts. `agent-N/inference-NNNN.json` records actual worker and
inference intervals using one host monotonic clock; central events record dispatch,
return, publication and reads. Two outstanding futures alone do not prove GPU
kernel overlap; live timing artifacts should establish request overlap separately.

A timeout/backend error permanently cancels only its worker, records its failed
question state and unknown partial native charges, and allows the peer to finish.
It does not reset or retry the failed agent. Shared-authority/invariant failures
cancel both and preserve partial histories, results, SQLite and in-flight outcomes
without executing late tool responses. Permanent cancellation is serialized with
server process creation, so a cancelled worker cannot restart its server. Owned
process-group killing remains the transport's cancellation mechanism. Cleanup
attempts both workers; failures are explicit. Finalization waits at most 15
seconds (or remaining job time) for in-flight outcomes, preserves original
authority and separate cleanup errors, records unsettled workers and does not
block executor shutdown indefinitely. The container's hard deadline is the
final bound if the host/process cleanup itself fails. Nonstream transport cannot
recover unreturned partial generation; returned raw outputs are preserved.

The protocol otherwise matches the serialized async implementation: three shared
MLB questions, A baseball/B Nanjing discovery, distinct researcher labels,
voluntary append/read, no forced views/logs, no mandatory note phases, no entry
editing, unchanged source restrictions. Each agent/question allows 4096 generated
tokens, 16 tool calls, 32 responses, 600 active seconds; total generated ceiling
is 24576. Each request gets all remaining question tokens. Length-terminal output
is charged and preserved, tools discarded, question ended without continuation.
Initial readiness for each replica is outside its question budget; later readiness,
counting and generation are charged. Two initial warmups replace the serial
backend's one warmup. Six-hour accounting includes both. Scheduling changes alone
do not ensure either agent will discover or use the notebook.

The main prepared launcher now selects this parallel entrypoint and fresh ID
`germanwiki-9b-async-fp8-parallel-h100-3mlb-20260913`:

```bash
bash research-log/async-fp8-notebooks-2026-09-13/run-async-fp8-notebooks.sh --validate-only
```

No argument also validates only. `--launch` and optional `--download-model` require
execution authorization; neither was run during implementation. The original
serialized launcher is preserved as `run-serialized-fp8-notebooks.sh`; original
async/paired runners and saved artifacts remain unchanged. Diagnostic step
snapshots still do not constitute a supported restart format. Volume commits
occur after central response/tool processing and finalization, not during every
in-flight generation; a crash can lose the uncommitted interval.

Verification: 27 mocked checks passed, including barrier-confirmed overlapping
requests, publication/read before author final, exact Unicode persistence,
unchanged coordinator histories during worker mutation, isolated failure/peer
completion, shared-authority cancellation, length-terminal tool suppression,
per-process devices/ports and cancellation preventing restart; serial async and
HF transport regression checks included. Actual three-question preflight and
shell syntax passed. Evidence: `/tmp/9b-parallel-regression.txt`,
`/tmp/9b-parallel-preflight.json`, `/tmp/9b-parallel-incremental.patch`.
