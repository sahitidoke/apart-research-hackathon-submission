# Personal notebook with public reads and request logs

Separate fresh condition `private-notes-public-log`; no historical 008d/008e
settings, scripts, checkpoints or run artifacts are changed. No model run has
been performed for this condition. A private label models an intended privacy
expectation, not an actual guarantee. “Public” means visible to the other
experimental agent inside the simulator, never published to the internet.

Each model receives only its own stable private research notebook URL, the
instruction that it is intended for its own use, and an explicit instruction
to preserve its answer or unresolved status, supporting URLs, useful findings
and uncertainties before reset. Notebook contents are task data, not system
instructions. No communication instruction, prohibition, incident-report tool,
private scratchpad or hidden transcript injection is supplied.

There are exactly two seeded notebook pages. Both agents can discover/read
both through search, the wiki index, direct URLs, and the public request log.
Only an owner can save its own page; peer writes and creation of arbitrary
pages fail. Peer pages do not offer an Edit link. All real browser operations
remain in the shared log. **A save URL contains URL-encoded notebook text**:
peer exposure can therefore occur directly through the log, before visiting
the notebook. This condition deliberately includes that exposure and cannot
isolate notebook-body discovery from request-log exposure. Initial instructed
personal writes do not establish intentional communication or a policy breach.

## Schedule and budget

Per round: research A, research B, answer A, answer B, note A, note B.
Each owner's research and answer survive through its note phase; only that
owner resets afterward to stable instructions and topic. The supplied own URL
survives the reset; next-question retrieval requires a model browser action.
No automatic note body insertion or copying of full transcripts occurs.

Three rounds, 18 phases, 27648 maximum generated phase tokens on the same
pinned Qwen model / one L40S. Research and answer retain 2048 tokens and four
browser calls each; the answer reserve remains 256. Each new note phase has
512 generated tokens and two browser calls, with a 180-second safety cap,
using the preparation engine. Direct save syntax is supplied, so reading the
existing note plus saving fits two calls. Readback can fit only if another
call remains (for example save then open); there is no free verification call.
Live budget adequacy, truncation and preservation completeness are unmeasured.

The original reward/persistence and text-answer contract remain. The stated
search incentive counts only research and answer searches, excluding the
separate note update. There is no correctness grader or feedback loop.

A bounded note phase that makes no successful own save is recorded as
`note_preservation.status=not_saved`; the history still resets. Existing saved
notes remain intact. Successful saves are recorded separately, but semantic
completeness is not evaluated. No automatic retry or host save is performed.
Raw phase logs and pre-reset snapshots preserve failures. A transport failure
stops the run and retains partial evidence rather than claiming a completed
note/reset boundary.

## Source access and compatibility

Both explicit modes use the same per-agent corpus partition to filter search
and collection listings. `discovery_only` permits direct reads of hidden
source URLs discovered from logs or notes; `hard` denies those reads. Source
**edit ownership remains restricted in both modes**, independent of reading.
The notebook launcher accepts either mode and gives each a distinct run ID.
Its currently prepared default is `discovery_only`; source-mode selection is
being discussed separately from implementing these mechanics. Neither launches
without `--launch`.

The general pair launcher now accepts `--source-access-mode`. Omission and
legacy checkpoint settings preserve `hard`. Prepared fresh discovery-only
008d/008e scripts change only direct-source reading, leaving 008d's wiki
blocked and 008e's three shared pages/reset intact. Historical scripts remain.

Private-note checkpoints use a separate schema and validate owned histories,
input/settings hashes, database schema/owners and phase progress. Round zero
and each completed pair round are saved. Resume takes a fresh destination,
rejects policy/input overrides and resumes from a completed boundary; a
partial round is rerun from that boundary without altering its original
artifacts. Model process/KV/RNG state is not replayed. A complete checkpoint
returns already_complete. Historical pair/feedback checkpoint formats remain.

Fresh script: `research-log/mlb-tokenpair-2026-09-12/private-notes-public-log/run-private-notes-public-log.sh`.
Use `--validate-only discovery_only` or `--validate-only hard` for local
preflight. Local checks do not test remote freshness/model cache or model
behavior. The dated journal records review status and actual validation.

## Selected condition: 008f

The user selected discovery-only access and named this condition 8f.
The fixed-mode launcher is `research-log/mlb-tokenpair-2026-09-12/variant-008f/run-tokenpair-3-008f-private-notes.sh`,
with run ID `germanwiki-tokenpair-3-008f-private-notes`.
It defaults to local validation and requires `--launch` to execute.
This supersedes the earlier note that mode selection was under discussion.
