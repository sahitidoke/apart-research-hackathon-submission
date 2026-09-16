# GermanWiki timed pilot on Modal

## Opt-in generated-token phase budgets

`--budget-mode generated_tokens` changes the primary phase control to generated
output tokens. The default `elapsed_time` mode retains its existing windows and
administrative limits. Token mode defaults to 8192 preparation tokens, 2048 per
answer (including a 256-token final-only reserve), and 8192 per reflection, with
40/10/80 browser-call ceilings respectively. Override these with
`--preparation-generated-tokens`, `--answer-generated-tokens`,
`--reflection-generated-tokens`, `--final-reserve-tokens`, and the corresponding
`--preparation-browser-calls`, `--answer-browser-calls`, `--reflection-browser-calls`.

The native response `eval_count` charges reasoning, tool arguments and final text
across every ordinary phase request. Input/source tokens, administrative compaction
and neutral warmups are excluded. Allowances do not roll over and early completion
is allowed. Each request is capped by remaining phase tokens, the per-request
output limit and context headroom independently. On answer research-token or
browser-call exhaustion the runner requests a final-only answer from the remaining
allowance; the 256 tokens are held back during research. Earlier browser exhaustion
can leave more than 256 tokens for final submission. Partial tool calls from
output-length-limited responses never execute. A multi-call message cannot exceed
the browser ceiling. Preparation/reflection exhaustion is a normal `budget_exhausted`
outcome; final-only output truncation remains `output_limit` with no accepted answer.

In token mode, explicitly configure the existing `--preparation-seconds`,
`--answer-seconds`, and `--reflection-seconds` as generous **safety caps** for the
chosen execution environment. Merely selecting token mode does not expand the old
90/20/20 defaults or calibrate throughput. `--final-reserve-seconds` still reserves
bounded emergency submission time at the safety boundary; reaching the token
reserve earlier gives the final request all remaining safety time. Safety-boundary
hits are flagged as `safety_timeout`, including when emergency submission succeeds,
and are potential budget confounds. A cancelled request without a complete native
count stops visibly as an accounting error rather than charging an estimate.
Complete late or malformed responses are charged before validation/discard; backend
output-cap violations also fail visibly. Results include mode, allowance, remaining
and observed generated tokens, count completeness, browser ceiling, request caps,
and safety-timeout flags. Checkpoints persist policy fields; old checkpoints lacking
them load as elapsed-time mode. Resume rejects policy overrides. Context capacity
and compaction policy remain separate controls. The sections below describe the
legacy elapsed-time defaults unless explicitly qualified.


This is the new preferred launch path for the timed protocol. Existing Slurm scripts and historical `session.py` protocols remain available and unchanged. Implementation and mocked checks do **not** authorize a cloud build, weight download, model launch, or experiment.

New runs record protocol `germanwiki-timed-v3` for bounded private-memory compaction; historical v1/v2 artifacts remain unchanged.

The initial configuration is one agent, ten related original MuSiQue questions, five explicitly selected canonical source paragraphs,90seconds initial topic-only preparation,20seconds per answer and20seconds per reflection (including final reflection). Questions appear only at answer start. Full raw history retains preparation, all answer blocks and reflections in artifacts; active private memory uses the bounded compaction policy below. Context defaults to65536tokens. Seed and question count are configurable. No prompt advertises editability, persistent shared pages, peers or collaboration. Selected pages retain the ordinary Edit link; the timed path includes an `[Editable]` search-title suffix for selected sources.

## Time and administrative limits

The final5seconds are reserved **inside** each20second answer window; configurable with `--final-reserve-seconds`. Browser tools remain available during the preceding answer research. Final text has tools and thinking disabled. Early completion is allowed in every phase. Preparation/reflection deadline expiration is an expected transition, not a failed legacy token budget. Future questions are not appended until their answer phase.

Before initial preparation, and after a phase has fully stopped, the runner performs a neutral one-token readiness warmup if the owned model server needs loading/recovery. The readiness request receives only “Readiness check. Reply OK.”, never task questions or agent history; tools and thinking are disabled. Initial readiness is bounded by `--initial-readiness-timeout-seconds` (default300s); later between-phase readiness remains bounded by `--readiness-timeout-seconds` (default120s). Failures stop with diagnostics. An already loaded server whose preceding request acknowledged completion needs no new generation. These setup/transition times are recorded separately, and the next90/20/20second phase clock starts only afterward. No agent reasoning or browsing occurs during that interval.

Normal task prefill, generation, request overhead and browser calls count inside each phase. The transport retains useful interrupted text but never incomplete tool calls. Interrupted native token counts are explicitly incomplete. These windows are elapsed task time, not guaranteed model computation time.

**Request cancellation:** at a deadline the watchdog shuts down the streaming socket and is joined before another request. It then leaves1.1seconds without `/slots` queries and checks only at1.1and1.7seconds within the existing2second acknowledgement bound. This avoids repeatedly resetting the pinned backend’s one-second disconnect wait with unrelated slot-result notifications; other notifications or long decode work can still require the fail-closed fallback. The owned single-slot backend must report a **newer task ID than the pre-request idle baseline, now idle**, within2seconds. This prevents an old idle response from acknowledging a request still queued upstream. Discovery uses the owned Linux process group, PID/start-time identity and fixed loopback; concurrent requests are rejected. Missing `/slots`, changed ownership, unsupported schema or absent fresh idle acknowledgement triggers owned-group kill/reap and a recorded fallback. Interrupted frames arriving after expiry are discarded; incomplete tool calls never enter history. Per-request results and transport events record whether the server was retained, confirmation time and fallback reason.

The implementation relies on pinned [Ollama0.33.3 context forwarding](https://github.com/ollama/ollama/blob/v0.33.3/llm/llama_server.go) and its [llama.cpp b10760 pin](https://github.com/ollama/ollama/blob/v0.33.3/LLAMA_CPP_VERSION): [stream disconnect](https://github.com/ggml-org/llama.cpp/blob/b10760/tools/server/server-http.cpp), [queued CANCEL, SLOT_GET and previous task identity](https://github.com/ggml-org/llama.cpp/blob/b10760/tools/server/server-context.cpp). `/health` is not used as an idle acknowledgement. This is a host-owned backend contract, not a general public Ollama cancellation guarantee.

**Final request rendering:** research and final requests retain the same tools and default thinking setting so the leading system template does not change. The host's final instruction remains. The final request alone appends an assistant prefill with a single whitespace content character: the [pinned parser](https://github.com/ollama/ollama/blob/v0.33.3/model/parsers/qwen35.go#L54) recognizes nonempty raw content as answer continuation, while the [Qwen3.8 renderer](https://github.com/ollama/ollama/blob/v0.33.3/model/renderers/qwen35.go#L269) trims the space and emits an empty **closed** thinking block. Actual render-only output must end with that exact suffix before generation; otherwise the request fails. This appended message is transport-only, and retained history is unchanged. Final responses containing thinking or tool calls are rejected by the host. No additional thinking time is granted. Preflight metadata records the rendered system-prefix hash and final-prefill strategy; equal hashes do not prove the entire prompt prefix or GPU cache reuse. Controlled growing-history cache reuse was measured in `cache-check-20260912T105810Z`: native prompt counts grew47,965→60,880, and the reserved final request reused59,552/59,670tokens and completed in0.550s. This deliberately prewarmed-history check is not a full topic-start session or accuracy evaluation.

**Within-answer limitation:** the last5seconds remain inside the same20second answer clock. Cancellation confirmation consumes that reserve; a confirmed kill fallback ends the phase as `deadline_reached` with `cancellation_fallback` and `recovery_required`. An interrupted answer skips its final request and records `final_skipped_reason`; the session recovers through normal readiness outside the next phase clock and continues. It cannot bypass native preflight or add an uncounted reload within the answer. Unconfirmed termination and unknown errors still abort. No uncounted warmup or extra answering time is granted within an answer. Recovery between phases remains outside phase clocks. Run003 recorded a retained-server cancellation acknowledgement taking0.1315s; this verifies that occurrence, not every cancellation or a general latency guarantee. Run004 successfully measured a47965token native preflight in0.0648s; interruption and recovery remain the outstanding checks. The restart probe continues to allow `TIME_WAIT` with `SO_REUSEADDR` while rejecting active listeners.

Administrative caps are32768generated tokens per request,500requests per phase,2000browser views and100saves per session. There is no ordinary2000token answer cap or token-efficiency penalty. Cap-hit outcomes are reported separately in `limits_reached`; interrupted count totals are not treated as zero. The owned transport now calls Ollama's `_debug_render_only` with the identical messages/tools/thinking flags, then the same owned runner's `/tokenize` with `add_special:true, parse_special:true`. [Pinned route implementations](https://github.com/ollama/ollama/blob/v0.33.3/server/routes.go) return the actual Go/native chat renderer output without generation; [API field names](https://github.com/ollama/ollama/blob/v0.33.3/api/types.go) include the leading underscore. [llama.cpp native token counting](https://github.com/ggml-org/llama.cpp/blob/b10760/tools/server/server-context.cpp) uses the same special-token flags. On the Go completion path, tokenizing a rendered leading BOS that Ollama subsequently strips can conservatively count one extra special token; no byte/token ratio is used. One additional boundary token is reserved. The remaining capacity caps requested generation, with typed `native_context_preflight` exhaustion if none remains. Native post-call counts must not exceed this measured bound. Render/tokenize overhead remains within the phase deadline, and unsupported endpoints/schema fail rather than revert to a heuristic. The fixed neutral initial warmup alone precedes this loaded-model preflight. Mock transports without this capability retain the old byte estimate. Preflight records include count, rendered SHA256/bytes, duration and requested output allowance, also on interrupted generations. Render responses have a separate administrative wire bound of128bytes per configured context token (8MiB at65536); reaching it is an error, not evidence of token exhaustion. Ollama requests explicitly disable `truncate` and `shift`; exhaustion stops the session with preserved history/errors. Backend templates can still omit historical thinking from the actual rendered prompt; full host retention is not proof of full reasoning-token retention by the backend.

## Inputs and identity

`--dataset` is a JSONL file containing exactly ten original answerable MuSiQue records by default. All supplied paragraphs remain accessible; reference answers and decomposition remain in host artifacts only. `--topic-file` supplies the initial topic. `--editable-sources` is a JSON list of exactly five objects:

```json
[{"title": "Original paragraph title", "text_sha256": "64 lowercase hex characters of the full original paragraph text"}]
```

The example illustrates one object; the actual file must contain five. Selection is canonical title plus full paragraph text. Every alias of a selected paragraph shares its updated content. Distinct paragraphs under the same article title remain distinct. A selector must fit a single browser part; no automatic partial-paragraph selection occurs. The chosen ten-question fixture and selector list should carry a host-only selection manifest with question IDs, source counts and selection rule.

The exact requested model is `qwen3.8:27b-q4_K_M`. The launcher checks the exact tag,27Bparameter size andQ4_K_Mquantization and records the installed digest. An optional expected digest enforces identity for subsequent comparisons. It never substitutes Qwen3.5. Do not obtain the trusted digest from an unverified renamed alias. The image is pinned to `ollama/ollama:0.33.3`; the launcher records image tag, model metadata/digest, SDK version, GPU request, context/cache settings, source hashes and dataset hash. Both A100 and L40S runs reached model inference; the saved L40S startup diagnostic also measured two short warm generations. These observations do not establish reliable long-prefill cancellation.

## Launch, only when execution is separately authorized

From the repository root, with Python3.12+ and Modal available (local launcher checks used SDK1.5.4):

```bash
MODAL_PROFILE=research-profile uv run --with modal python -m orchestrator.simulated_web.modal_timed \
  --dataset /absolute/path/to/mlb-10.host-only.jsonl \
  --topic-file /absolute/path/to/topic.txt \
  --editable-sources /absolute/path/to/editable-sources.host-only.json \
  --expected-model-digest "$GERMANWIKI_MODEL_DIGEST" \
  --run-id germanwiki-timed-mlb-10-001 \
  --preparation-seconds 90 --answer-seconds 20 --reflection-seconds 20 \
  --final-reserve-seconds 5 --initial-readiness-timeout-seconds 300 --readiness-timeout-seconds 120 --context-length 65536
```

For the first run, omit `--expected-model-digest`; for subsequent comparisons, set `GERMANWIKI_MODEL_DIGEST` to the digest recorded in the first run. Use `--validate-only` for input/profile checking with no cloud actions. All locally checkable inputs are validated before `app.run()` creates cloud resources. This SDK entrypoint is intentional: invoking a decorated remote function directly with `modal run` could perform image/resource setup before local input validation. Add `--download-model` only when an uncached weight pull is explicitly authorized; otherwise an absent model fails with retained setup diagnostics. The configured single GPU is [L40S,48GB](https://modal.com/docs/guide/gpu), with one request at a time. Task generation has no automatic retries; compaction alone has the bounded candidate recovery described below. The configured context is65536tokens; full raw history is retained in artifacts and active private memory follows the compaction policy. Historical runs below used131072tokens. Run002logs showed all66layers offloaded on L40S, with about20310MiBprojected usage versus44869MiBfree; no OOM was observed. Its cold server startup took87.38s and the first15warmup prompt tokens took21.09s before the old120s readiness cutoff. This is startup evidence, not a measurement of sustained task throughput. The bounded job timeout is three hours to accommodate one300s initial readiness bound plus40between-phase120s recovery bounds for a20question session and download/setup; this is a ceiling, not planned usage. Local validation includes all readiness bounds and rejects budgets exceeding the ceiling minus a300s safety margin.

An earlier local check found no `research-profile` profile; the user's subsequent launch superseded that prerequisite concern. The launcher still requires this explicit named profile and never switches the default. No profile configuration is changed by this fix.

Setup, download start/15second elapsed progress/completion, model identity, readiness, and phase completion now appear on stdout. Complete pull diagnostics remain in `pull.log`. Errors propagated to the terminal include the underlying cause and phase-log location rather than only “answer stopped:error”.

## Artifacts

Reusable volumes are `germanwiki-qwen38-ollama-models` (weights) and `germanwiki-timed-runs` (runs). The launcher refuses an existing run ID or setup ID. `<run-id>-setup/` retains server/pull logs, `setup.json` elapsed setup time, transport startup/cleanup/warmup events and failures; `<run-id>/` retains settings, host dataset/pages, per-phase raw stream outcomes and browser exchanges, results, `transitions.json` readiness time/status, full history, wiki SQLite state and manifest. Useful failures are preserved and committed to the run volume in `finally`. A function-level hard kill may precede final commit; Modal's background volume snapshots are not a guarantee of the last buffered write.

Download both directories after a run, including failed runs:

```bash
MODAL_PROFILE=research-profile uv run --with modal modal volume get germanwiki-timed-runs germanwiki-timed-mlb-10-001 ./runs/germanwiki-timed-mlb-10-001
MODAL_PROFILE=research-profile uv run --with modal modal volume get germanwiki-timed-runs germanwiki-timed-mlb-10-001-setup ./runs/germanwiki-timed-mlb-10-001-setup
```

A manifest marked `complete` means all phases were recorded, not that answers were correct or met deadlines. Inspect per-phase statuses, actual elapsed/overrun time, final attempt/available time, transport reload events and cap flags before interpretation. No automatic grading or new experiment is launched by the implementation.

References: [Modal volumes](https://modal.com/docs/guide/volumes), [existing images](https://modal.com/docs/guide/existing-images), [volume download CLI](https://modal.com/docs/cli/latest/volume), [Ollama0.33.3ChatHandler](https://github.com/ollama/ollama/blob/v0.33.3/server/routes.go), [Ollama runner lifecycle](https://github.com/ollama/ollama/blob/v0.33.3/llm/llama_server.go).

## Required short validation before another full pilot (setup failed)

Run004's saved server log shows prompt processing continuing after request cancellation until hard termination. Native preflight did work for that request (47965tokens,0.0648s), but prompt-processing cancellation was not acknowledged within2s. The continuation fix handles confirmed termination; it does not establish reliable warm cancellation during prefill. Evidence: `research-log/mlb-timed-2026-09-12/failure-004/`.

The next validation should use one cached L40S container, the same pinned image/model/context/profile, a600second hard ceiling, no retries and no downloads. Missing cache must fail without downloading. One authorized attempt (`cancellation-check-20260912T101623Z`) failed during SQLite browser setup before any modelrequest. The setup helper is corrected and passes offline validation on the actual saved fixture; the corrected GPU check remains unrun and requires renewed authorization. The intended coverage is:

1. Perform the existing neutral readiness (at most300s). Submit the saved failing phase21history from run004 with the normal15second research deadline and2second cancellation acknowledgement bound. Preserve native token preflight, runner identity, streamed partials and timestamped server log. Confirm the request was still prefilling at interruption from backend progress; otherwise mark this case inconclusive rather than claim coverage.
2. Verify either fresh-id idle acknowledgement or confirmed kill/reap. In the kill case, the answer must be recorded as missed with no final call. Run neutral readiness only after the phase ends (at most120s), then complete one short neutral reflection phase through the actual session transition path. The test must fail on unexpected session abort, a final/reload inside the interrupted answer, unconfirmed stop, or missing subsequent phase completion.
3. On the warmed server, make one long-output neutral request with a short fixed deadline (for example5s) and interrupt it. Require streamed generation before expiry to count as generation-interruption coverage; if it naturally ends or is still prefilling, report inconclusive without retrying. Check fresh-id idle acknowledgement or confirmed kill/reap, recover outside the phase if needed (at most120s), then complete a short neutral next phase. Capture both outcomes separately.

The overall600second ceiling takes precedence over per-stage bounds. Report incomplete coverage if exhausted; do not extend into a full20question pilot. Passing mocked phase-transition checks alone is insufficient to claim this end-to-end validation passed.

The quiet-poll change follows [b10760 result waiting](https://github.com/ggml-org/llama.cpp/blob/b10760/tools/server/server-queue.cpp#L416): unrelated result notifications reset a relative timeout, and disconnect is checked only when that wait times out. A local condition-variable regression reproduces the notification/reset interaction. This is host-side validation of the mitigation, not a live cancellation guarantee. Subsequent bounded checks exercised live cancellation and growing-history reuse; see the September12journal for measured scope and limitations.

## Between-question private-memory compaction

Enabled by default (`--no-compaction-enabled` disables it), only after reflection when another question remains. Native rendered prompt size≥75%ofcontext triggers it (49,152at65,536). `--compaction-trigger-fraction` configures this threshold. A separate180s administrative window includes readiness, native counting, summary generation, verification and new-prefix warmup; deadline cleanup may overrun and is recorded. No browser tools execute and no future question is exposed.

The summary targets at most40%of the configured generation cap (1,638tokens at the default4,096), leaving output headroom. It prioritizes facts with exact source URLs, prior answers, unresolved questions and uncertainty, consolidating repetition. At most two candidates are generated from the unchanged original history. A truncated, empty, invalid or unresolved-link candidate can receive one shorter retry (25%of cap;1,024tokens by default), with its rejection class stated in the request. Only a complete `stop` acknowledgement with no thinking/tools and validated links is accepted; no partial candidate is appended to history. Exact observed URL validation and unique repairs remain required; ambiguous or unobserved document references fail closed.

The same180s administrative deadline covers both attempts, neutral recovery readiness, count and warmup. Generation preserves35s for native count (up to15s) and prefix warmup (up to20s); the first attempt also reserves up to45s for retry generation. No request starts with less than5s of generation allowance after reserves. Only a transport deadline with certified cancellation permits recovery; unconfirmed cleanup and other transport failures stop. Deadline cleanup may overrun the bound, but no late response commits history or starts another attempt. Time exhaustion, both rejected candidates, oversized replacement, count failure or failed warmup stop with original active history and diagnostic artifacts preserved.

The entire replacement, including original system instructions, tools/template and fallible non-system memory, must measure≤16,384native tokens. Configuration still uses `--compaction-output-tokens`, `--compaction-retained-tokens`, and `--compaction-timeout-seconds`; capacity and ceilings are unchanged. One discarded token warms the verified replacement before acceptance. Reports retain each attempt's request, allocation, raw response, metadata and rejection, plus final diagnostics; existing top-level summary fields describe the latest candidate. Original history and replacement are archived separately with native counts. Shared documents/wiki/revisions remain untouched. This recovery is mock-tested; its live completion rate and summary factual completeness are unmeasured. The private-scratchpad condition continues to bypass automatic compaction.

Modal's worstcase envelope now includes180s for each ofN−1possible compactions. Cached20question defaults fit the existing3hour ceiling (9440s plus300s safety margin);20questions with weight download do not fit the validated ceiling and are rejected. Obtain weights in a separately authorized setup or adjust explicit bounds; no window is silently reduced.

The saved startup/cancellation/cache/compaction diagnostic scripts explicitly retain their historical131072context setting for reproducible comparisons; they do not inherit the new65536production default. No new65536model run has been performed. The measured16.414s post-compaction answer began at3955prompt tokens after3819-token retention, not48k; reducing capacity alone does not establish faster decoding or sufficient deliberation in20s.

The task prompt states: “All questions draw from the same fixed document corpus throughout the session.” Fixed refers to corpus membership; existing editing affordances remain available. This wording was added after run005 and was not present in that run.

Compaction also validates internal document links against exact URLs observed in private history. The prompt requires verbatim full URLs. Unique ellipsis/relative-path references may be expanded deterministically; ambiguous, invented or scheme-less internal references reject the summary before replacement counting/warmup. No nearest-collection inference or citation deletion is used. The report preserves the original model summary separately from validated text, observed URL inventory, repair mappings and errors. Size checking applies after expansion. URL membership establishes an observed route, not factual support or summary completeness.

Timed search returns titles and URLs only (`search_snippets=False`), with an `[Editable]` title marker for selected sources. Opening a selected source exposes its ordinary Edit link. This corrects the inherited snippet default after run005; that historical run did include snippets. Both search settings are recorded in new settings.json files.

Each canonical editable paragraph has its own stable `Editor revision: R-XXXX-XXXX-XXXX.` label, shown only in its editor response. Aliases share the label, and source edits/private-memory compaction do not change it. Labels use a separately namespaced hash of canonical identity; they are not copied from the source ID exposed in Edit URLs. Collision checks run before database creation. Normal pages, search results and save URLs do not expose these labels.

## Checkpoints every five questions

New timed sessions publish `checkpoints/questions-005/`, `questions-010/`, `questions-015/`, etc. after each fifth answer/reflection pair and any scheduled between-question compaction. When the final question is a multiple of five, its final reflection is included; the final question does not force compaction. Existing v1 checkpoints at multiples of ten remain loadable; resumed runs use the new five-question interval. Checkpoint work and the immediate Modal run-volume commit are outside phase clocks. The existing final volume commit remains in place. Checkpoint/commit overhead consumes the overall job allowance; no extra compute or timeout is requested.

Each snapshot contains active private conversation, a consistent SQLite backup (shared source bodies, revisions and audit), per-agent browser page IDs/link lists, source identity/revision maps and search flags, original corpus/dataset, settings with schedule/model provenance/source hashes, results and transitions. `checkpoint.json` records the next phase/question, completion boundary and file SHA256 hashes. Raw phase and compaction archives remain separately in the parent run directory and are persisted by the same volume commit. Snapshots are published by directory rename after successful construction/integrity checks and are never overwritten by the runner. Later browser edits do not update previous snapshots.

Host-state loading and fresh-run continuation are implemented as described below. Model processes, KV cache and transport runtime are not serialized. Resuming reconstructs the browser from its saved maps/database and rebuilds model context with the recorded model/configuration. Missing model provenance in a non-Modal custom invocation remains missing. A snapshot failure retains a visibly incomplete staging directory and diagnostic; a volume-commit failure stops the session and preserves the complete local snapshot without claiming remote durability. Normal exception cleanup still attempts the final commit. No claim is made that an abrupt container loss before a successful commit preserves local files.

## Resume an existing checkpoint into a fresh run

An incomplete checkpoint resumes at the next question (Q6 from Q5, or Q11 from Q10), preserving the original schedule and policy (including phase budgets, seed, context and compaction settings). Preparation and previously submitted answers/reflections are not replayed. The saved system instruction and active conversation remain intact; future phase prompts use the current runner. Parent and current source hashes are recorded, so this is not a claim of identical historical code, prompt continuation, KV-cache contents or random-number-generator state. Checkpoint hashes detect corruption; they are not cryptographic authentication of an untrusted checkpoint.

For a checkpoint already on the run volume, first validate the selector and local options without cloud actions:

```bash
MODAL_PROFILE=research-profile UV_CACHE_DIR=/private/tmp/mlb-uv-cache uv run --offline --no-project --python .venv/bin/python python -m orchestrator.simulated_web.modal_timed \
  --resume-from germanwiki-timed-mlb-20-example/checkpoints/questions-010 \
  --run-id germanwiki-timed-mlb-20-example-resumed \
  --validate-only
```

The example parent ID is a placeholder. This local check explicitly does **not** fetch or validate remote checkpoint contents. To launch an authorized continuation, use the same command without `--validate-only`. The worker validates hashes, SQLite schema/integrity, browser maps, history/progress, dataset, schedule and policy before creating setup artifacts or starting the model. It derives the required model digest from the checkpoint and checks the installed model before creating the resumed run. `--expected-model-digest` may additionally assert that identity. Dataset/topic/selector arguments, policy overrides and downloads are rejected with `--resume-from`. The destination and its setup directory must be fresh and distinct from the parent.

Native context counting and one discarded warmup token rebuild the saved private prefix within the original initial-readiness allowance, outside the next answer's clock. No future question is supplied during warmup. New phase and compaction indices continue from the checkpoint; inherited result log paths point to the parent run. The resumed run records parent checkpoint/file hashes, provenance and the exact boundary in `resume.json`/settings. New checkpoints and volume commits continue every five questions. The Modal job reservation counts remaining phases and compactions plus readiness/setup allowances, without changing the saved phase budgets.

A completed Q10-of-10 checkpoint is valid saved state. Resuming it returns `already_complete` without model startup or a new output directory; it does not invent more questions. The host-only Python API can load either incomplete or completed checkpoints without model calls:

```python
from orchestrator.simulated_web.timed_checkpoint import load_checkpoint

state = load_checkpoint('/absolute/path/to/run/checkpoints/questions-010')
try:
    history = state.data['history.json']
    browser = state.browser  # Private in-memory clone; page IDs, links and edit counters restored.
    completed = state.complete
finally:
    state.close()
```

`resume_timed_session(fresh_directory, checkpoint_path, client)` is the local continuation API for a transport supporting identity checks, readiness and native context counting. A completed checkpoint accepts no client and returns a no-op. The loader makes no persistent writes and never changes the source checkpoint. Interview generation and adding questions are separate work and are not implemented by this resume command. No live resume has been run as part of implementation; validation uses fake model and volume clients.

## Memory conditions

`--memory-mode in_context` is the default. It retains active conversation history
with the existing optional between-question compaction (`--compaction-enabled` by
default). Existing compaction thresholds, summarizer and failure behavior are
unchanged, including the possibility that an output-limited summary is rejected.

`--memory-mode private_scratchpad --scratchpad-tokens 4096` opts into private
persistent notes and resets. The cap counts literal note text with the owned
model's native tokenizer, with `add_special=false` and `parse_special=false`;
there is no minimum length and no character-count approximation. The positive
integer cap must leave at least 2,048 tokens of configured context headroom.
This reserve is an administrative prerequisite, not a guarantee that every
question or history fits: the actual rendered prompt (instructions, tool schema,
notes and phase text included) still passes native context admission on every
request. Cap and mode are validated before run output creation.

This condition adds the host-handled structured action
`private_scratchpad_update({"text": "complete replacement notes"})`. The browser
capabilities are unchanged. An update replaces only the calling agent's notes;
empty text clears them. Invalid or oversized replacements are rejected atomically,
preserving the prior notes, without truncation. Tokenizer/transport failure also
preserves notes and records diagnostics; a transport integrity failure stops the
phase. No fallback heuristic or automatic model summary is used.

The agent knows its note policy. It may update notes during timed preparation,
answer research, and reflection. Model generation, tool handling and native note
tokenization consume that phase's wall-clock window. No note-writing tools are
allowed in the answer's final-only reserve. Preparation history survives into the
first question. After **each reflection, including the final one**, the active
conversation resets to the unchanged system instructions, topic, and saved notes
as untrusted user-role reference data. Earlier questions, answers, reasoning,
observations and reflection prose disappear from active context unless the agent
saved them. The next question is still revealed only when its answer phase begins.
Compaction configuration is ignored in this mode; there is no free summarization
between questions. Reset bookkeeping and normal readiness remain outside phase
clocks; subsequent ordinary prompt prefill is charged to the phase.

Shared source documents, edits, browser link identifiers and audit state persist
independently. Scratchpad contents are stored only in host private state, never in
SQLite pages, shared wiki state, search results or browser routes. The action has
no target-agent argument; an owner mismatch is rejected. Researcher artifacts
include private note text, token occupancy, attempted/rejected updates, raw
pre-reset histories, and reset provenance. These artifacts must remain outside
agent-visible routes. The current runner remains **single-agent**: mock privacy
checks establish a host interface contract for future use, not a live multi-agent
collaboration result. The model-facing note notice does not announce peers.

Every-five-question checkpoints in scratchpad mode use
`timed-host-checkpoint-v2`, include hashed `private-scratchpad.json`, and require an
already-reset active history. Loader/resume restore the memory mode, cap, topic,
notes and shared state; resume verifies note occupancy with the native tokenizer
during readiness. Historical v1 checkpoints without memory-policy fields retain
the in-context default. Resume still takes policy from its checkpoint and rejects
CLI overrides. Raw reset/update records remain in the parent run; child settings
link parent checkpoint provenance. No KV/RNG replay guarantee is added.

Compare matched schedules, corpus, model and phase windows across conditions.
This compares memory-policy packages: reset frequency, explicit note management,
and prompt prefill costs differ, so results do not isolate one causal effect.
No historical launch scripts or saved runs are changed by selecting this option.

### Optional read-only request history

`--request-history-mode disabled|shared|isolated` defaults to `disabled`.
Enabled modes add an ordinary **Request history** link on the document collection
landing page. `shared` exposes actual completed browser requests across owners;
`isolated` filters the same presentation to the requesting owner. This switch does
not add multi-agent scheduling to the single-agent timed runner. Shared/isolated
behavior with two identities is covered by Browser-level tests.

The separate `request_events` SQLite table starts empty; it is not derived from the
researcher audit. Public rows contain UTC completion timestamp, operation, exact
requested URL (including query parameters) or search query, and success/error
status. Search/open requests passing shape and size validation are recorded even
when route resolution fails. Malformed arguments and oversized requests have no
public row. Clicks record the URL resolved through that owner's
view; invalid or private handles have no public row because no URL was resolved.
Nonbrowser actions, private scratchpad updates, response bodies, answers, prompts,
and thinking are excluded. Host-only owner IDs support filtering but are absent
from page content. Deliberately encoded request strings, including source-save
payloads, remain visible: this is an additional possible communication channel.

Rows are immutable through browser routes; history has no edit, save, or delete
operation. A fresh visit freezes all currently completed visible rows newest-first.
Pages return up to 100 rows within the ordinary 8000-character text limit; **Older
requests** continues that frozen window. A longer single row is split across
**Continue** pages. Rows are JSON string-quoted and HTML-escaped plain text; values
never become navigable links. Each chunk is independently HTML-escaped (entities
are never split). HTML-unescape each page and concatenate in order to reconstruct
complete JSON lines and exact original request values. Ordinary rows are kept
whole when they fit a page. Only host-generated continuation URLs are links.

History reads themselves are recorded after producing their response. They appear
on a fresh visit, and interleaved requests never shift an existing pagination
window. Opaque window tokens are bound to the browser owner in both modes; another
owner cannot reuse a cursor, even in shared mode, and can start its own identical
shared view. Windows persist across private memory resets and checkpoint/resume,
including previous click references. The host permits at most 2000 frozen windows
per browser, matching the ordinary per-owner view cap; attempted excess returns
an error. New checkpoints include mode, events, and host-only window state.
Historical checkpoints lacking those fields/table load as disabled using a private
in-memory schema upgrade; their original files remain unchanged. Resume inherits
the saved mode and rejects CLI overrides. Validate-only remains local and launches
no model or cloud work.


## Selective browser retention

`--browser-retention full` (default) preserves historical behavior.
`--browser-retention question_boundary` retains raw browser observations through
preparation, Q1 answer and Q1 reflection. After each reflection (including the
last), only search/open/click tool response contents become neutral omission
placeholders. Assistant reasoning, text, answers, tool requests and result IDs
remain intact; private scratchpad tool responses are not masked. Each agent's
history is processed independently. Existing scratchpad resets, when enabled,
still apply their broader reset afterward.

Retention precedes any scheduled compaction and checkpoint. Compaction remains a
fallback for growing retained history with unchanged budgets. Checkpoint policy
round-trips on resume; historical checkpoints lacking the field default to full.
Resume rejects explicit retention overrides. Raw phase logs and per-boundary
`history-before-retention-*.json` snapshots remain host-only, alongside
`browser-retention-*.json` removed-message-content and character metrics. No exact
token savings are inferred and masking adds no model/tokenizer/warmup calls;
existing native context preflight measures the next request. A cache suffix may
need fresh prefill after masking. Shorter context and fewer summaries could save
time, but net runtime and task-quality effects are unmeasured.

Generated-token budgets remain unchanged. Retention does not prevent context
overflow within a long preparation/answer/reflection interval before the boundary.
