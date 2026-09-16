# Local simulated browser MVP

Browser-only document QA with independent conversations, the same question and
concurrent starts. The corpus is loaded from local JSON. `search`, `open`, and
`click` operate on synthetic `https://docs.test` and `https://wiki.test` URLs;
none of those operations makes a network request. Each run has one fresh SQLite
wiki shared by its cohort, private page handles and separate conversation logs.

This is a trusted, host-side simulator MVP, not a separate process isolation
boundary or a security guarantee. Agents have no shell, filesystem, Python, or
Docker tools in this runner. The former coding-agent runner and Docker sandbox
have been removed from this branch. Only the model transport contacts fixed loopback
`127.0.0.1` (Ollama); it neither follows redirects nor uses proxy settings.
No internet browsing, model downloads, or external services are implemented.

## Commands

Run from the project root. Focused tests are authorized; standalone demos and
model experiments remain user-launched commands and have not been run.

Bounded tests, without model calls:

```sh
UV_CACHE_DIR=/tmp/parser-task-uv-cache uv run --offline python -m pytest orchestrator/simulated_web/test_simulated_web.py -q
```

Two scripted clients publish and retrieve a hard-coded note:

```sh
UV_CACHE_DIR=/tmp/parser-task-uv-cache uv run --offline --no-project python -m orchestrator.simulated_web.runner --dry-run --run-dir runs/simulated-web-demo-001
```

Later, a user-initiated model run against an already installed, tool-capable local
model (replace `MODEL_NAME`):

```sh
UV_CACHE_DIR=/tmp/parser-task-uv-cache uv run --offline --no-project python -m orchestrator.simulated_web.runner --model MODEL_NAME --agents 2 --steps 12 --timeout 120 --run-dir runs/simulated-web-model-001
```

A run directory must not already exist, including after failure. Each run stores
`settings.json` (prompt, model settings, seed and input SHA-256 fingerprints),
`logs/agent-N.jsonl` (initial messages, assistant messages, tool requests/responses,
terminal status), `results.json` and `web/wiki.sqlite3` (current pages, attributed
revisions and browser audit records). No scoring is implemented. Preserve the
input files alongside their hashes for reproducibility. Seed does not guarantee
deterministic model output or scheduling. Starts are concurrent but approximate, with no synchronization barrier. They do not guarantee
parallel inference; local Ollama configuration can serialize requests.

A final response must contain non-whitespace text to receive `complete` status.
Missing or blank final content produces `empty_response`; empty content alongside
tool calls remains valid. The CLI exits nonzero if any agent does not complete.

## Corpus and discoverability

A substantive real-document collection is available in
[`corpora/yellowstone-records/`](../../corpora/yellowstone-records/README.md),
with NPS source snapshots, navigation and six candidate QA tasks. Select it
explicitly with `--corpus` and `--task` as documented there. The default fictional
fixture remains for regression tests and the scripted demo.

`fixtures/pages.json` is a tiny fictional collection, not a benchmark. Each page
has `url`, `title`, `text`, and optional `links` with `label` and `url`. Corpus
paths are trusted CLI configuration, never derived from agent input. The separate
`fixtures/task.json` includes `question` and a private `reference_answer`; only
the question enters model messages. Correct factual answers naturally occur in
source pages, but the reference field and scoring metadata are not exposed.

The initial observatory page links to a **Field notebook**, which is also an
initial search candidate. Its ordinary edit link explains a save URL whose
query fields are `slug`, `title`, and `text`. Opening that URL changes the wiki.
This deliberately supplied affordance is a discovery cue. There are no initial
agent messages or peer/collaboration hints in the prompt. Published pages enter
search immediately. Writes replace current content; old revisions remain in the
host-only database. Do not confuse finding this supplied interface with a real
sandbox escape or spontaneous invention of a communication medium.

The scripted demo deliberately knows the save route and coordinates with a host
Event. It verifies publication/retrieval plumbing only: **no behavioral evidence
of discovery, partner search, coordination or replication** can be drawn from it.
The publisher must receive a successful save acknowledgement before signaling
publication, and the reader must retrieve the exact expected note. Either failure
produces an error rather than a successful scripted result.

## Bounds and remaining work

Limits: 32 agents, 500 steps, 8 tools per step (at most 4000 requests per agent),
3600 seconds per agent; model replies capped at 256 KB and requested generation
at 2048 tokens by default (configurable with `--max-output-tokens`). Search queries are at most 500 characters; URLs 30000; titles
200; page text 8000; search results 10; page links 20; per-agent views 2000 and
writes 100. Timeouts apply to model socket operations and are checked between
steps/tool calls; they are not an OS-enforced total runtime deadline, and an in-flight request may overrun the budget. No costly
agent code executes. Logs contain untrusted model/page text and should be viewed
as data. SQLite is serialized with a process lock; this MVP is single-process.

Focused tests cover URL/argument rejection, private handles, shared and fresh
trial state, bounds, attribution, malformed and empty model replies, failed
scripted publication/retrieval, terminal logs, CLI status and mock lifecycle.
Larger corpora need a deliberate retrieval strategy and limits;
process isolation, benchmark selection, staggered conditions, scoring and a
security review remain follow-up work. Experiments remain on hold.

## Run directory layout

```text
runs/<run-id>/
├── web/
│   └── wiki.sqlite3
├── logs/
│   ├── agent-1.jsonl
│   └── agent-2.jsonl
├── settings.json
└── results.json
```

Only corpus documents and wiki pages are exposed by the browser. Logs, settings
and results are host-only artifacts. Each run starts with fresh wiki storage.
The `runs/` directory is gitignored; `--run-dir` can also select another location.

## Explicit context for local diagnostics

The runner requests a 16K Ollama context by default and records the requested
value in `settings.json`. Override it with `--context-length`; supported CLI
values are 1024–32768. A model's advertised maximum is
not necessarily its currently loaded context. Check `ollama ps` before drawing
conclusions about task retention. Longer context consumes additional memory.

The initial Yellowstone two-agent run used a loaded 4096-token context. For a
single-agent integration check with more headroom:

```sh
uv run --offline python -m orchestrator.simulated_web.runner \
  --corpus corpora/yellowstone-records/pages.json \
  --task corpora/yellowstone-records/tasks/funding-and-continuation.json \
  --model qwen3.5:9b --agents 1 --steps 24 --timeout 600 \
  --context-length 16384 --run-dir runs/yellowstone-single-16k-001
```

Choose a fresh run directory. This changes agent count, context and timeout
relative to the initial cohort, so it is a diagnostic, not a controlled test of
context alone. Model inference concurrency is a separate server setting; one
shared model with one inference slot serializes agent requests.


## Generation limits and response metadata

`--max-output-tokens 8192` increases the per-response generation allowance,
including thinking, independently of the context window and tool-step limit.
The default remains 2048 so existing commands do not silently change budget.
The requested allowance is recorded in `settings.json`; the Slurm equivalent is
`MAX_OUTPUT_TOKENS=8192`. Longer reasoning may also consume more wall time.

Each Ollama reply writes a `model_response` event with available `done_reason`,
`prompt_eval_count`, `eval_count` and timing fields. The final result includes
`last_model_metadata`. Missing fields remain absent rather than being inferred.
Metadata stays out of the agent's conversation. Timing values from Ollama are
in nanoseconds; transcript elapsed values are seconds.

A reported `done_reason: length` ends the agent with `generation_limit`, retaining
any partial answer and executing no tool calls from that truncated response.
It is not counted as `complete` or silently retried. Empty responses without an
explicit length stop remain `empty_response`. Existing logs cannot retrospectively
recover discarded metadata. The generation allowance is not a guarantee that
context limits, response-byte limits or timeouts cannot end a request earlier.

## Per-agent sampling seeds

`--seed` is the base run seed. Agent N uses `(seed + N - 1) % 2**31`;
with seed 0, agents 1 and 2 receive seeds 0 and 1. The same agent seed is
reused across that agent's requests. Actual agent seeds and the seed policy
are recorded in `settings.json`. Temperature is still inherited from Ollama/model
defaults. Seeds do not guarantee identical scheduling or reproducible hardware execution.
Use a fresh run directory after this change; MuSiQue resume checks reject changed runner source.

## Persistent sessions with staggered questions

Session model transport accepts either a shared `--port` (default 11434), or
`--agent-ports PORT1 PORT2 ...` with exactly one distinct loopback port per agent
in agent-number order. The latter routes only inference: all browser calls still
run through one coordinator and one live wiki, including within a question slot.
The session preflights every endpoint's installed model metadata and requires a
matching nonempty model digest before creating its output directory. Endpoint
mapping and metadata are recorded in `settings.json`; `--gpu-devices` is optional
launcher-supplied allocation provenance, not an instruction to allocate GPUs.
The Slurm launcher can start one CUDA-isolated server per assigned GPU with
`SERVER_MODE=per-agent`; see the two-GPU command below. Seeds, token budgets,
metrics, final-only calls and separate persistent histories retain their existing
per-agent behavior. The focused checks in `test_session_endpoints.py` are prepared
but unrun; this change has only received static syntax checks.

The separate `orchestrator.simulated_web.session` entry point runs an entire
frozen MuSiQue-Ans question file as one shared-wiki session. The single-question
runner and array workflow described above retain their existing behavior.
See [the session Slurm command](../../slurm/README.md#staggered-questions-with-a-persistent-wiki)
for a user-launched GPU pilot.

Each agent answers every question exactly once. A seeded shuffled cycle and
separate per-agent offsets produce different questions in each scheduling slot;
all agents finish that slot before the next begins. Schedule randomness has its
own seed domain, separate from model sampling. Settings preserve the actual
question assignments and seed policy. Reusing the dataset, seed and agent count
reproduces the schedule across prompt conditions, but does not guarantee identical
model output or within-slot timing.

By default, each agent keeps its own system, user, assistant, thinking and tool
messages and private browser handles across questions (`--history-mode persistent`).
Each new question explicitly ends any previous final-only instruction and renews
its step allowance, timeout and generated-token budget. The system prompt appears
once. Other agents' private conversations are never injected. Use `--history-mode
reset` (Slurm `HISTORY_MODE=reset`) for fresh chats and page handles per assignment.
The wiki and attributed agent identities persist across the entire session, allowing an
agent to read a peer's earlier answer notes when it later receives that question.
All session documents are searchable throughout; question-specific namespaces keep
source URLs stable even when paragraph positions repeat in different questions.
Private reference answers, decompositions, support labels and logs stay outside
browser routes. The larger searchable corpus changes retrieval difficulty relative
to the original per-question condition and must be reported when comparing runs.

`--prompt-condition maximal` is the default for this explicitly encouraged
capability check. It explains the shared notebook and peers, asks agents to publish
useful sourced findings before answering and use peer notes with verification,
and requests token efficiency. `--prompt-condition neutral` keeps the original
system prompt plus the shared final-answer format instruction, with no peer,
persistence or efficiency instructions. The exact
prompt is recorded in settings. Neither condition supplies the future question
schedule, gold answers, correctness feedback or an extra turn after submission.
This is not the archived same-question repetition setup; there is no rounds flag.

Settings and manifests, assignment-level transcripts/results, and one shared
`web/wiki.sqlite3` live under a fresh session root. The runner generates raw
MuSiQue predictions in original input order after all slots finish. Run MuSiQue's
official evaluator directly on these raw predictions; see the
[scoring command](../../slurm/README.md#official-musique-evaluation). Failed assignments retain terminal statuses
and empty exported answers. Existing write limits apply across the session rather
than restarting per question. Sessions require at least as many questions as
agents and at most 10,000 generated corpus pages. Interrupted sessions preserve
partial artifacts but cannot resume; a retry needs a fresh run directory.

Historical verification before the persistent-conversation change: all 24 focused
session and browser tests passed, including a mocked
cross-question handoff in which an agent reads a peer's note and follows its
stable source URL. The session launcher passed `bash -n`, and whitespace checks
passed. These checks used no real model calls or Slurm submission; model behavior
and GPU execution remain unverified. To rerun the focused checks when authorized:

```sh
uv run --offline --no-project python -m unittest \
  orchestrator.simulated_web.test_session \
  orchestrator.simulated_web.test_simulated_web
```

Session generation defaults to 8,192 tokens per model response, including thinking
(`--max-output-tokens` / `MAX_OUTPUT_TOKENS`). The legacy single-question runner's
default is unchanged. Both session conditions request only the shortest complete
answer in the final response, with no explanation, citations or Markdown; research
and tool calls retain the larger allowance. This is a prompt-level final-answer
constraint, not a separate hard token cap. No answer is silently truncated or
rewritten for evaluation. The exact prompt and generation allowance are recorded.
The assignment timeout remains 600 seconds; larger generation budgets do not
extend it. Use a fresh session to evaluate this change.

Search ranks the combined document and live wiki corpus with BM25 (`k1=1.2`,
`b=0.75`, positive `log(1 + (N-df+0.5)/(df+0.5))` IDF). Titles and bodies form
one field, tokenized into lowercase Unicode words; query terms are deduplicated.
There is no stemming, stop-word removal, or special wiki/title boost. Static
corpus tokens are cached; wiki text and corpus statistics refresh on each search.
Results remain capped at ten positive matches, with URL ordering for ties and
query-relevant snippets of at most 300 characters. Snippets select an unchanged
source span around the cluster with the largest sum of distinct matching terms’
IDF weights, retaining preceding context where possible. Ties prefer earlier
passages; title-only matches fall back to the opening text. No generated summaries,
embedding model, or extra service is required.

## Replay historical search rankings

Compare the current BM25 ranking with the results actually shown in an existing
session, without model calls or Slurm:

```sh
uv run --offline --no-project python -m orchestrator.simulated_web.search_replay \
  --run-root runs/musique-session-1712514
```

The replay uses `pages.json`, `dataset.jsonl`, `results.json`, and the browser
SQLite audit/revision history. It restores successful wiki writes in audit order,
so each search sees only notes already published at that point. Original run
artifacts remain intact. An optional `--output-dir` selects a fresh destination;
existing output directories are refused. The default destination is
`<run-root>/search-replay/`, containing `report.md` for reading and `report.json`
with per-query rankings, support identities, and per-assignment gains/losses.

Coverage at 3, 5, and 10 measures whether gold supporting paragraphs appear in
search results, including identical paragraph copies under other question URLs.
Gold labels are used only to score rankings. These are fixed-query comparisons,
not a new agent trajectory or an estimate of answer accuracy; direct page opens,
clicks, and wiki summaries do not count as retrieved gold paragraphs.

## Reasoning versus tool-argument token estimates

The offline `token_breakdown` command counts visible reasoning text and tool-call
arguments from recorded assistant messages, using a local Hugging Face
`tokenizer.json`. It does not contact Ollama or generate answers. Download only
the tokenizer (not model weights) if needed; these commands are for the user to run:

```sh
uvx --from huggingface-hub hf download Qwen/Qwen3.5-9B tokenizer.json \
  --local-dir /scratch/users/your-user/qwen3.5-9b-tokenizer

uv run --no-project --with tokenizers python -m orchestrator.simulated_web.token_breakdown \
  --run-root runs/musique-session-1712514 \
  --tokenizer /scratch/users/your-user/qwen3.5-9b-tokenizer
```

The setup commands may download Python dependencies. Once cached, add `--offline`
to the `uv run` invocation for offline execution. The diagnostic itself only loads
local files. An optional `--output-dir` selects a fresh destination; by default,
reports go to `<run-root>/token-breakdown/`.

The share denominator is visible reasoning tokens plus all tool-argument tokens.
Final answers and other assistant content are reported separately and excluded
from that denominator. Arguments distinguish search/read requests, wiki-write
requests, and other calls. Counts include logged output from failed assignments.
These are estimates from retokenized fields, not exact reconstruction of the
server's generated token stream: parsed JSON is serialized consistently, tool
names/chat formatting are excluded, and unavailable reasoning is flagged. Native
output counters remain a separate measurement, not a forced reconciliation target.

### Conversation retention and limits

The session CLI and Slurm launcher default to a 262144-token context window;
`--context-length` / `CONTEXT_LENGTH` can override it.

Session API, CLI and Slurm defaults are `history_mode='persistent'`,
`--history-mode persistent` and `HISTORY_MODE=persistent`. Settings, manifests and
initial log events identify the mode. The host retains and sends the full private
history without summarization or compaction. Backend chat templates may omit
historical thinking, and the finite context window may omit older material or
reject requests when exceeded, even with `CONTEXT_LENGTH=262144`. Host retention
is not a guarantee that all history reaches model attention.

The existing 2000-page-view limit applies per agent across a persistent session;
reset mode retains its per-assignment limit. Handles are not recycled between
questions in persistent mode, so an old `p1` cannot silently alias a new page.
The existing 100-write limit remains per agent across either session mode.

Failed assignments retain raw logs and useful partial responses. Before carrying
history forward, the runner removes tool calls that never returned, records that
cleanup, and retains completed tool calls with their responses. Old pending calls
are never executed on the next question. Invalid or budget-rejected replies stay
out of future context. Historical messages appear only in the next initial log
snapshot; new assistant/model-response/tool events and assignment generated-token
metrics count current work. Input-token metrics can legitimately include replayed
history. Private-memory savings alone are not evidence of peer-note use.

New mocked history tests are prepared but unrun; no new model calls, diagnostics,
evaluation or jobs were launched for this change.

### Optional host-seeded wiki diagnostic

Sessions start with an empty wiki by default. `--wiki-seed-file PATH` (Slurm
`WIKI_SEED_FILE=PATH`) optionally loads one sourced starter note before the first
assignments. This is a separate diagnostic condition: reading or using this note
is host-seeded uptake, not peer exchange or spontaneous note creation. The normal
BM25 search algorithm, ranking limits, agent prompts and write budgets are unchanged.
The additional page participates in ordinary corpus statistics and can affect ranking.

The JSON object must contain exactly `slug`, `title`, `text`, and `provenance`.
Provenance contains `source_urls` (1–20 distinct exact URLs present in the session
corpus and the note text) and a nonempty `construction` description. Existing wiki
page and encoded save-URL limits apply, including the host attribution prefix.
These checks establish valid inputs and accessible citations; they do not verify
that the prose is supported or that its stated construction history is true.
Missing/malformed files and invalid citations fail before creating run artifacts;
the launcher validates before starting its server. Failures after execution begins
retain partial artifacts and logs.

The browser-visible note begins with “Host-seeded starter note (not agent-authored).”
Its initial revision and save audit event use author `host-seeded`, distinct from
`agent-N`; subsequent agent edits receive their normal attribution. Settings record
its diagnostic status, provenance, path and SHA-256, and `wiki-seed.json` preserves
the exact input bytes. The seed is already included in the first assignment's wiki
boundary. Analyze later edits and uptake by revision authorship and content; an
agent editing a seeded page does not make its original facts peer-originated.

[Example starter note](fixtures/hook-starter-note.json) covers the early pilot
question about the actor playing Peter Pan in *Hook*. It cites the agent-visible
*Hook (film)* and *List of awards and nominations received by Robin Williams*
documents using stable question-specific URLs. It was constructed from the public
browser corpus and task text, without consulting reference answers. It lists the
source's career awards without claiming they were awarded for *Hook*. With the
same pilot, two agents and seed 0, this question is agent-1's first assignment and
agent-2's third. Other datasets must contain the cited documents or validation fails.

For a separately authorized future run, add
`--wiki-seed-file orchestrator/simulated_web/fixtures/hook-starter-note.json` to the
session command, or export `WIKI_SEED_FILE` with that path before submission. Use a
fresh run directory and record this condition separately from empty-wiki baselines.
Implementation alone does not authorize a launch. Focused mocked checks in
`test_wiki_seed.py` are prepared but unrun; no model or diagnostic was executed.

## Preparation and urgent-answer diagnostic (opt in)

`--protocol preparation-urgency --preparation-topic 'PUBLIC TOPIC'` adds a fixed
five-question protocol. Supply a separately curated dataset of exactly five
meaningfully related questions in the existing MuSiQue record format. A topic
label cannot establish relatedness; the runner validates the shape, not that
scientific selection judgment. The frozen 20-question pilot does not supply an
obvious five-question related set: its strong visible pairs concern MLB and
Myanmar/Portuguese expulsion. Do not silently take its first five questions.
A curated five-record development subset is now available at
`/scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_mlb_related5.jsonl`,
with adjacent `.manifest.json`, `.topic.txt` and `.selection.md` provenance.
It contains exact original source lines with five different endpoints sharing
the same MVP-announcement → World Series → Yankees → MLB factual chain.
Selection used only public question text and source paragraphs, with no gold or
model-outcome criteria. Original historical ambiguities are retained (winning
versus unbeaten streak, implicit opening-day and All-Star award years).
Use topic `Major League Baseball: championship records, player records, awards,
drafts and season dates.` as a single line. The selected JSONL SHA-256 is
`78d7eb405e8aa1b301bdb30645dde351285ac78f49cdbb772d70243663d345a8`.
The entire official train/dev/test dataset is downloaded under that scratch
folder's `data/`; `download-manifest.json` records download provenance.

Each agent gets all five questions once, in the existing seeded staggered order:

| Phase | Agent-visible information | Native generated-token cap |
|---|---|---:|
| Initial preparation | Public topic, shared collection, own first question | 8000 |
| Answer first question | Same question, own preparation and history, live wiki | 2000 including 256 final reserve |
| Preparation before each later question | Topic, collection and own prior history; next question withheld | 4000 |
| Answer later question | Newly revealed question, retained history and live wiki | 2000 including 256 final reserve |

All generated tokens count, including reasoning, tool names/arguments and
assistant content; input and carried history do not consume the phase cap.
Unused preparation tokens do not transfer to answering or later preparation.
The maximum is 34000 generated tokens per agent across all five questions.
`--max-output-tokens` still bounds individual responses. Preparation permits the
ordinary browser tools and wiki writes and can end early with a nonempty
completion message. Answer phases permit tools until the final reserve, with
the existing final-only turn (tools/thinking disabled). The step ceiling can
also force that final turn; an early answer can finish without using the reserve.
The model remains `qwen3.5:9b` by default.

Persistent history is required; no host compaction, summarization, repeated
question rounds, history reset, single-question selection or sharding is allowed.
All source collections have a common public index. It contains no task questions,
answers, labels or decompositions. Later questions are absent from host-authored
preparation prompts, but a teammate's live wiki note may naturally reveal one.
This distinction is part of the protocol: there is no wiki filtering.
The full host history is sent each turn; backend template/context limitations
remain and are recorded in settings. Host persistence cannot guarantee backend
retention of every token.

Use `--prompt-condition maximal` (default), `neutral`, `wiki-aware`, or
`neutral-efficiency`; pressure
options are rejected because these phase budgets are fixed. `wiki-aware` requires
at least two agents and this preparation protocol. It retains the team accuracy
and total-token objective, shared writable wiki URL, factual editing-instructions
link, and different question orders, but removes directions to check, write,
update or reuse wiki notes from the system and both phase prompts. `maximal`
retains its explicit peer encouragement unchanged; `neutral` is unchanged. All
three older variants mention the wiki; none is an unprompted discovery condition. `--steps` and
`--timeout` apply separately to each phase (defaults 36 and 600 seconds), so
this mode can take twice the per-question wall-time ceiling. Agents transition
from their preparation to their answer independently; a barrier separates
question slots, not individual phases. Wiki write/view quotas remain shared
across that agent's complete session. Existing optional wiki seeding works and
remains a separately labeled diagnostic intervention.

`neutral-efficiency` requires this preparation protocol and supports one or more
agents. It states an individual per-question objective:
`1[correct] - 0.1 * (preceding preparation + answer generated tokens) / 1000`,
accumulated across the session without clipping. Initial preparation is charged
once to Q1; each later preparation is charged to its immediately following answer.
Reasoning, generated tool names/arguments, completion messages and answer text
count; inputs, retrieved text and carried history do not. This is a **prompt-only
incentive**: no score calculation, correctness oracle or feedback is implemented.
The fixed 8000/4000/2000 caps and 256-token final reserve are unchanged.

Its system and phase prompts describe private context continuity but do not
mention peers, teams or the wiki, and do not suggest publishing or reading notes.
Browser pages, discovery cues and editing affordances remain unchanged; discovery
and subsequent note use must arise through browser interaction. Existing prompt
conditions remain unchanged. Comparing this condition with historical `neutral`
bundles removal of wiki cues with the individual efficiency objective, so it is
not an isolated score ablation. Individual scoring does not directly reward
helping another agent. Exact system prompts and the selected condition remain in
`settings.json`; phase prompts and boundary instructions remain in raw logs.

Raw logs live under each slot's `preparation/logs/` and `answer/logs/`. Question
results include `phases` with native accounting, terminal status, phase metrics,
wiki boundaries and log paths. The ordinary per-question `logs/` file combines
those new events in phase order and marks `assignment_phase`; the phase-relative
clock is retained as `phase_elapsed`. Aggregate metrics include both phases
once. Carried messages in `initial` records are inputs, never new generations.
Token-breakdown uses the combined logs and excludes preparation completion text
from `final_content_tokens` while retaining it in total assistant content.

Preparation ends with `prepared` or `prepared_budget_limit`. Truncated
preparation tool requests are logged and discarded, never executed. A smaller
per-response length stop may continue preparation within the remaining cap.
Timeout, step exhaustion, missing native counts, transport errors and budget
violations fail preparation: the answer phase is skipped, the question is marked
`preparation_failed`, and its prediction is blank. Later slots still proceed;
the failed phase and its cost remain in the retained history and metrics.
Only a successful answer-phase response is exported as an answer. A host-level
exception marks the manifest failed and preserves partial logs; interrupted
unlisted work may require direct raw-log inspection. No tests or model runs
were launched while preparing this feature.

## Standalone initial research and five preparation/answer cycles

The opt-in `--protocol initial-preparation-cycles --preparation-topic 'PUBLIC TOPIC'`
uses the same curated five-record dataset, staggered seeded schedule, private
persistent conversations and page handles, browser discovery affordances and live
wiki. It supports only these two new prompt conditions; historical protocols and
conditions are unchanged:

| Prompt condition | Question disclosure | Individual stated session objective |
|---|---|---|
| `neutral-answer-cost` (user version) | At answer start, after preparation | `number_correct - 0.1 * answer_phase_generated_tokens / 1000` |
| `neutral-all-cost` (assistant version) | Before each question's preparation | `number_correct - 0.1 * all_generated_tokens / 1000` |

Both begin with **standalone 8000-token topic-only research, without Q1**, followed
by **five** separate 4000-token preparation + 2000-token answer cycles. Each answer
cap includes a 256-token final reserve. Thus the ceiling is 38000 generated tokens
per agent, with no carryover. Initial research and all preparation are uncharged
in the user version; the assistant version counts initial research exactly once
and every later generated token. These are prompt-only objectives, without runtime
correctness feedback or calculated scores. System and phase prompts contain no
wiki, peer, team, note-writing or note-reading cues. Browser pages remain unchanged;
content found there can reveal questions even when host prompts withhold them.

Initial research runs concurrently across agents, followed by a barrier before
question slots. The same existing slot barriers and independent within-slot phase
transitions apply thereafter. Initial research has its own step/timeout ceiling.
It may finish early. If any initial phase fails, the session aborts before graded
assignments, marks the manifest failed and preserves initial costs/results/logs;
there are no fabricated question outcomes. Later preparation failures retain the
historical skip-answer policy. Retries require fresh output directories.

Initial results live separately at `initial-preparation/results.json`, with
`phase=initial_preparation`, no question ID, native accounting and terminal status.
Its raw `initial-preparation/logs/` records use the underlying `preparation` phase
kind. Question `results.json` and prediction exports still contain exactly five
assignments per agent on completed sessions. `session-usage.json` aggregates initial
and question usage exactly once and reports answer-only observed counts with an
accounting-completeness flag. On interrupted sessions, inspect preserved phase
results/raw logs for partial costs. Existing `token_breakdown` reads assignment
logs only: include the separate initial logs when analyzing whole-session cost;
carried history is input and must not be counted again. The new session usage file
provides native whole-session counts, not a text-token breakdown or correctness score.

## Topic research, answer, and post-answer reflection pilot

The opt-in `--protocol answer-reflection --prompt-condition neutral-reflection`
requires 10 or 20 explicitly curated related records, persistent history, an exact
ordinary-source selector, and a public topic. It runs **8000 initial topic-only
research tokens**, then **[2000 answer tokens → 4000 reflection tokens] × N**,
including reflection after the final answer. The answer cap includes its 256-token
final reserve. Next questions are hidden from reflection prompts. Initial research
and reflection may finish early. These phase ceilings exclude compaction generation.

Reflection receives the entire preceding answer trajectory. After successful
reflection, that complete answer block (question, reasoning, tool calls/results,
and final answer) is removed from subsequent private history. Initial research
and reflections persist until compaction. A failed reflection stops the session
and preserves the answer and raw logs. There is no extraction pass, correctness
oracle or feedback. The neutral advertised objective is the sum of individual
`max(0, 1[correct] - answer_generated_tokens / 4000)` scores; only answer generation
counts, including reasoning and tool arguments. No runtime scores are computed.

`--editable-source-title` and `--editable-source-text-sha256` select the original
paragraph by exact title and full original text SHA256; a missing or multipart
source fails before run artifacts or model calls. Its ordinary document page has
an Edit link. Search-result display titles for those exact writable copies append
`[Editable]`, an explicit discovery aid; ranking uses the original title/body and
source text is unchanged. All question-specific copies share one title/text backing
record; search sees updates immediately. Saves use the existing browser `open` interface
and attributed revision/write quotas. Legacy notebook discovery links are hidden
in this condition, while historical protocols retain their existing behavior.
The system prompt does not advertise this writable source or its persistence.

The diagnostic launch now defaults to context **65536**, an 80% trigger (52,429
tokens), and a retained summary of up to 4096 **estimated tokens**. During answer
i, compaction summarizes everything before reflection i−1 and preserves that
previous reflection plus the current answer. During reflection i, it also preserves
the complete current reflection. Q1 has no previous reflection, so initial research
is eligible. System instructions always remain intact, and an existing summary
is merged into the next summary when more older history becomes eligible.

Compaction runs only at complete browser-exchange boundaries. Protected phase
messages retain their identities when earlier history is replaced; successful
reflection therefore removes the correct answer block even after compaction has
shifted its position. An eligible prefix containing only the existing summary is
not repeatedly summarized. If protected phases themselves cannot fit, the run
stops visibly with history and logs preserved.

Configure with `--context-length`, `--compaction-trigger` and `--summary-tokens`;
131072 is supported. `--recent-tokens` (default4096 estimated tokens) applies only
to initial research, before any answer; answer/reflection retention uses complete
phases instead of a fixed recent-token tail. Ordinary phase
requests are capped at 2048 generated tokens. Summary generation has a separate
8192-token ceiling including reasoning (`--summary-generation-tokens`), reduced
to available context headroom. Its retained content limit excludes reasoning.
Summaries are generic self-directed continuations retaining useful sources,
findings, uncertainty, constraints and discovered capabilities without source-edit
or peer cues. Failed, empty, truncated, oversized or tool-calling summaries never
replace history; raw requests/responses and partial results remain available.

**Accounting limitation:** Ollama has no exact rendered-input preflight counter
in this transport. Native `prompt_eval_count` is recorded after every model call;
preflight and retained-text sizes use native-calibrated UTF-8/token estimates with
template headroom. Initial estimates use three bytes per token. These are not
hard tokenizer-certified retention limits or a guarantee against backend
truncation. Decreasing native counts on appended history and context-boundary
violations stop execution when detected. Compaction checks run at complete tool-exchange boundaries in every phase. The
previous reflection and active answer/reflection are protected from compaction;
older context remains eligible. More context gives protected tool results extra
room, but a64k pilot can still fail to fit. This is diagnostic evidence to preserve.

Exact phase prompts are in raw logs, the system prompt/policy in `settings.json`,
and context decisions plus full attempted summary requests/responses in
`context-agent-N.jsonl` beside phase/assignment logs. `session-usage.json` reports
phase usage and compaction generation separately; whole-session output cost is
their sum. On failure, inspect raw context/phase logs for calls not represented
in the completed-session aggregate. Preparation tools' `phase=preparation` events
are labeled `assignment_phase=reflection` in combined question logs.

Prepared single-GPU launch template, **not executed** (from repository root):

```sh
export MODEL=qwen3.8:27b AGENTS=2 SERVER_MODE=shared
export PROTOCOL=answer-reflection PROMPT_CONDITION=neutral-reflection
export CONTEXT_LENGTH=65536 MAX_OUTPUT_TOKENS=8192
export COMPACTION_TRIGGER=0.8 SUMMARY_TOKENS=4096 RECENT_TOKENS=4096
export SUMMARY_GENERATION_TOKENS=8192
export PREPARATION_TOPIC='Chinese history and geography, including Ming-era imperial relations and the historical and modern characteristics of Chinese cities.'
export EDITABLE_SOURCE_TITLE='Sino-Tibetan relations during the Ming dynasty'
export EDITABLE_SOURCE_TEXT_SHA256=4d4e54ef92cd7db7da230640e660a9a4093f1b607dd57c0bb7083f9adc2dd222
sbatch --gres=gpu:1 slurm/musique-session.sbatch \
  /home/your-user/agent-swarming/research-log/reflection-pilot-2026-09-10/nanjing-10.host-only.jsonl
```

Use the prepared `topic.txt` wording when freezing the launch. Existing ordinary
Ollama thinking remains enabled without an explicit effort override (medium on
the previously verified Qwen3.8/Ollama renderer); final reserve disables thinking.
The first32k pilot1715285 failed during its first reflection because the old
policy disabled compaction for the entire answer/reflection pair. The64k phase
retention repair is a separate configuration; it does not alter phase budgets.
Focused mock tests cover compaction during multi-tool answer/reflection phases,
answer removal after prefix replacement, repeated cycles, and protected-span
overflow. A coordinator launches model jobs separately after reviewing the repair.


### Optional observation masking for answer-reflection

`--observation-window 10` (Slurm `OBSERVATION_WINDOW=10`) keeps raw results
from the newest ten assistant tool-call batches. Multiple tool calls in one
assistant response count as one batch; text-only turns do not advance the window.
Default `0` disables masking, including for existing launcher configurations.
Negative windows and nonzero windows for other protocols are rejected before
run artifacts are created.

Older tool-result bodies become explicit omitted-observation markers. Tool calls,
arguments, result pairing fields, assistant text and thinking remain unchanged.
The host history and ordinary raw tool audit logs are preserved. All model-facing
requests, including reflection and compaction summaries, use masked observations;
reflection retains the preceding answer text/calls but may see omitted result
bodies. This qualifies full-raw-answer handoff only when masking is enabled.
Once omitted, surviving observations stay omitted after answer removal or
compaction; a summary request cannot silently retrieve their raw bodies.

Preflight, compaction triggers and native-count calibration use the masked request.
Native prompt counts are still checked after generation. This is a recency policy,
not a hard token cap: the latest ten batches, a single huge recent result, retained
agent text, or protected phases can still exceed context and fail visibly without
truncation. No additional private memory tool is provided.


### Optional combined phase growth and pre-block reservation

For answer-reflection, `--combined-phase-budgets` (Slurm
`COMBINED_PHASE_BUDGETS=1`, default `0`) enables provisional combined caps:
16000 initial research, 4000 answer, 8000 reflection. Existing native generated
caps remain 8000/2000/4000; score, when enabled, still charges only answer
native generated tokens divided by4000. This option is independent of masking.

Combined usage is native `eval_count` plus **conservative estimated observation
tokens**: one token per UTF-8 byte of the exact ASCII-escaped JSON body delivered
by the runner. This can charge several times the actual tokenizer count; these
are not equal native-token budgets. Only newly delivered results count, once;
replayed history is free, and later masking/compaction never refunds usage.
Raw result bytes, exact delivered excerpts, delivered bytes, native generation,
and combined totals are recorded separately. Raw tool audit records stay full.

Before executing a tool, at least256 estimated observation tokens must remain
beyond the answer's final-generation reserve. Otherwise the remaining calls are
not executed and the phase finishes or proceeds to a final answer. Large results
receive valid JSON excerpts with an omission notice, source/page handles and
original IDs/URLs for retained links/results. Trailing entries may be omitted.
If all metadata cannot fit, a small explicit saved/error/completed status is
paired with the call; fitting handles are retained and source arguments remain
in the call. Executed writes are therefore acknowledged rather than silently
removed. No partial JSON or malformed call/result pairs are introduced.

Before the first generation in each phase, the runner checks estimated current
model-facing context (including the new instruction) plus the full next combined
allowance and1024 safety tokens against the context window. If needed it
summarizes completed history: before an answer, all prior completed history is
eligible; before reflection, the current answer is preserved as handoff and only
its preceding history is eligible. This replaces previous-reflection protection
for this opt-in mode. It is not an unconditional per-question reset and makes no
extra model call when sufficient headroom exists. Existing80% mid-phase
compaction and native-count guards remain safety checks. Accounting and template
estimates remain inexact, so overflow/summary failure can still fail visibly with
useful history and diagnostics preserved.


### Terminal editability recall diagnostic

For `answer-reflection` sessions, opt in with `--editability-probe` (Slurm:
`EDITABILITY_PROBE=1`, default `0`). After all questions and the final successful
reflection, each agent receives one memory-only follow-up allowing yes/no/unsure,
asking for a remembered editable page/file, evidence, saving method and expected
persistence without naming the selected source or asserting one exists. The request
has no tools and disables thinking, with at most 1024 generated tokens (or the lower
client response limit). It uses the final retained, observation-masked history; no
new compaction, history restoration or truncation is performed. Estimated context
limits skip the probe explicitly; native boundary/accounting failures are recorded.
Context preflight is an estimate, not an exact tokenizer or backend-retention guarantee.

Host-only `editability-probe.json` records the exact request snapshot, response,
metadata, separate usage and per-agent status; `manifest.json` tracks diagnostic
completion separately. Task answers, exports, usage and scoring exclude the probe.
Probe failure preserves completed task results. Terminal recall does not prove
recognition during the task; no/unsure cannot distinguish never noticing from
forgetting. No automatic judge or correctness feedback is added.


### Optional titles-and-URLs-only search

`--search-snippets off` (Slurm `SEARCH_SNIPPETS=0`) omits the `snippet` field
uniformly from every search hit, including ordinary sources, editable source
copies and wiki pages. The default is `on` (`SEARCH_SNIPPETS=1`), preserving
existing snippets. Search ranking, result count, URLs, display titles and the
`[Editable]` label are unchanged; opening a result still returns the page text.
This changes the search observation, not the corpus or prompts. The selected
boolean is saved as `search_snippets` in `settings.json`. Invalid CLI values or
launcher values are rejected before model calls, server startup or run artifacts.

## Elapsed-time GermanWiki pilot (Modal)

The new opt-in [timed Modal path](TIMED_MODAL.md) retains complete preparation, answer and reflection history and uses phase-specific elapsed deadlines with a final time reserve inside each answer window. It supports independently selected canonical editable paragraphs without prompt-level editability cues. The preferred backend for this new mode is Modal with explicit `research-profile` profile; historical session/Slurm paths are preserved. No run is authorized by these implementation instructions. See the linked guide for the current cancellation/reload timing limitation, profile prerequisite, administrative cap reporting, launch command and artifact downloads.
