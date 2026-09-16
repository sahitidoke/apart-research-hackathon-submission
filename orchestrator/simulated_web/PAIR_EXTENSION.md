# Explicit 8d feedback extension

This child continuation starts from a complete three-round8d checkpoint. Ordinary
pair resume still treats that parent as complete. The new entry point adds seven
unused questions per agent, selected in original opposite-corpus order, for ten
questions total. It preserves the parent source corpus, fixed hard access, shared
browser database and request log, independent browser views and private histories.
The actual local parent has68 messages for A and49 for B, and65536 context capacity.

The child archives an immutable copy of the entire original checkpoint. It preserves
the first system message's complete original text as a prefix and appends a clearly
separated superseding system instruction: seven additional questions, JSON evidence
answers, archived citation-text feedback, at most one retry, and no prior numerical
reward/search penalty for these questions. Persistence wording remains. All other
parent messages are unchanged at the fork. No old questions are replayed.

Each new round runs A research, B research, A answer, B answer. If citation checking
fails, only failing agents get one extra research+answer attempt: failed A research,
failed B research, failed A answer, failed B answer, omitting successful agents.
There are no further retries. The first attempt's observations remain available
for its retry; browser responses are masked after success or the final failed
attempt. Assistant text persists. Genuine latest request history is forced into
context before every phase, outside the model browser allowance.

Each research and answer remains2048 generated tokens/four browser calls, with
an unchanged256-token answer reserve. Across seven rounds:28–56 new phases and
at most114688 generated tokens. The worst-case phase safety ceilings total21840
seconds, exceeding the six-hour job limit even before readiness and finalization.
The runner only admits another whole round when its maximum retry/time allowance
fits; completed-round checkpoints support continuation in a fresh job. Completion
of all ten questions is not guaranteed. The inherited native context guard remains;
there is no new compaction, summary or reset. Context exhaustion preserves diagnostic
artifacts and may require a design decision before further continuation.

## What verification establishes

JSON shape is checked using the existing answer contract. For an answered object,
all citation URLs must exactly identify original archived source pages in the input
corpus; all nonempty quotes must occur in the corresponding original page after
whitespace normalization. Root/index/editor/history URLs and external URLs fail.
Archived pages come from the parent's hash-validated `pages.json`, not mutable
browser text. A self-edit cannot make a fabricated quote pass. Foreign-corpus
citations are allowed; the agent need not have opened the cited page itself.
Verification does not open browser routes, add public audit events or contact a
network. It never checks answer correctness, entailment or the usefulness of a
quote, and calculates no score. Irrelevant but exact quotes can pass.

Only these messages go to the agent:

- Success: “Citation text verified against the archived source. This does not establish answer correctness.”
- Failure: “Supporting evidence was not verified. The task remains unresolved. Reconsider your approach and continue investigating using the available browser tools.”

Host results retain attempt numbers, verifier outcome and detailed private reasons.
Feedback supplies no correct URL, quote, gold answer or peer hint. Behavioral
recognition/contact remains the research outcome. More rounds, JSON, feedback,
removed numerical reward and retries form a combined intervention; this continuation
does not isolate duration alone. Existing task ambiguities remain.

## Checkpoints and launch

All checkable parent/configuration prerequisites are validated before output. A
round0 checkpoint is written before the first new phase. Each completed new pair
round, including its retries, produces a self-contained checkpoint with immutable
parent files, source hashes, current database, private histories, results and browser
view state. Restore checks member hashes, parent identity, settings, completed dynamic
phase order and verification results. Ordinary old checkpoints remain compatible.

On failure, partial-round logs and state remain in the failed run. Resume uses the
last completed-round checkpoint in a new destination: it redoes the unfinished
round from that boundary, not its exact interrupted model state. Already completed
rounds are not replayed, and the original run/checkpoint is not modified. The final
round7 checkpoint is complete and resumes as a no-op. There is no model KV/RNG replay.

Prepared command from repository root:

```bash
bash research-log/mlb-tokenpair-2026-09-12/008d-feedback-extension/run-008d-feedback-extension.sh --launch
```

The default without `--launch` validates the downloaded local parent only; it does
not reserve a GPU or check remote run freshness. Launch revalidates the remote
checkpoint against the locally inspected manifest hash before creating setup output.
The user controls launch. For later resume, `modal_pair_extension` takes
`--resume-from RUN/checkpoints/rounds-00N`, `--local-checkpoint DOWNLOADED_CHECKPOINT`,
and a fresh `--run-id`, again defaulting to validation only. Parent extension and
extension resume are mutually exclusive. The pinned model and one L40S are unchanged.
