# Saved 9d evaluation

`evaluate_saved_runs.py` extracts mechanical evidence using only the Python standard library. It does not import the runner, model transport, cloud clients, or SQLite; it never launches a model. Evaluation is an analysis operation: run it on experimental data only when authorized. Development tests use synthetic artifacts only.

```bash
uv run python -m orchestrator.simulated_web.evaluate_saved_runs --help
uv run python -m orchestrator.simulated_web.evaluate_saved_runs \
  /absolute/path/to/run-a /absolute/path/to/run-b \
  --output /absolute/path/to/existing-parent/fresh-report
```

The output directory must be fresh, outside every input run, with an existing parent. All inputs are parsed and reports serialized before it is created. Existing reports and run artifacts are never overwritten. Invalid inputs leave no report directory. An actual write failure retains partial output for diagnosis. Keep outputs host-only and outside agent-visible routes. Evaluate a stable downloaded or completed run snapshot; this tool does not coordinate with an active writer.

Each input requires `settings.json` with `schema: synchronized-exchange-v1` and at least one of `results.json` or `evidence-index.host-only.jsonl`. Optional `manifest.json` and `round-NN-histories-before-reset.json` provide completion and body evidence. Malformed supplied artifacts fail validation; absent optional evidence stays unknown. SHA256 fingerprints identify every consumed artifact. References cannot escape the input directory through a symlink. The tool does not follow event artifact paths or transcript paths, including absolute paths copied into resumed results.

Outputs are `evaluation.host-only.json` (`schema: saved-9d-evaluation-v1`) and `evaluation.host-only.md`. JSON has one `runs` object per input, with provenance, coverage, phases, costs by agent/phase, append events/counts, peer-information evidence, source observations, memory/barrier events, failures, annotation templates, and consumed-artifact hashes. No cross-run totals or statistical estimates are produced. Exact duplicate event IDs are counted once with a warning; conflicting duplicates fail. Duplicate/nested input paths fail.

## Interpretation

- Configured/intended model fields, saved runtime inspection, and individual request identity metadata remain separate. Q4 and FP8 identities are never inferred from current code defaults or combined. Saved inspection is recorded evidence, not independent verification of loaded weights. Missing identity stays null.
- Costs describe the phases in the supplied `results.json`. Every total includes known/total row coverage; missing values stay null rather than zero. Native token-accounting flags are retained separately. A reported token subtotal may itself be incomplete when native accounting is incomplete.
- Resumed results can contain a copied prefix while local event indices and histories cover only the resumed suffix. The tool exposes local event coverage and recorded migration/resume metadata, but does **not** validate or reconstruct parent checkpoints. It does not combine parent and child runs. Abandoned attempts in a parent run are not included in a child's cost. Compare the separate reports manually; never sum overlapping trajectories.
- Append metrics are local indexed **persistence attempts**, divided by host mandatory versus voluntary model actor, outcome, and destination ownership. Note-generation retries rejected before persistence are not append attempts. Phase failures/tokens remain visible. Visible author labels are mapped through saved settings; destination and authorship are independent.
- Host exposure events are retained as delivery records, not automatically labeled peer-body receipt. Only actual nonempty peer text in reviewed round histories establishes body delivery. Receipt additionally requires a subsequent non-host assistant message in that same history. It establishes opportunity to process the body, not attention, useful uptake, or final-answer relevance. A terminal delivery has no such opportunity.
- Voluntary reads require a model-origin matched call and successful peer-body response. Directory reads, own entries, and host calls do not count. The metric is confined to the history files listed in coverage. `histories.json` is deliberately excluded: it may be a duplicate or an empty post-reset snapshot. Interrupted questions and external resumed-prefix histories may be absent. Tool event hashes alone do not prove nonempty text. Counts are observed lower bounds, with unmatched calls and unavailable responses reported explicitly; the unclassified-body counter also includes ordinary source and directory text where an author is not applicable.
- Eligibility denominators remain unknown. A saved no-peer condition has peer opportunity `not_applicable`, rather than a failed discovery. Observed trace counts are still retained so contradictory evidence is inspectable.
- Source returns are counted only when the recorded response provides a `docs.test` source URL. Denials often omit URLs; complete source-attempt/denial rates are unsupported. Returned text does not establish answer support or correctness.
- Memory restored/omitted entry lists and resets/barriers retain exact event references. Index `captured_at_utc` is capture time, not the action timestamp; `created_at` remains entry creation time. No response latency or cross-agent within-stage ordering is inferred.

The definitions reuse the earlier project-local `research-log/2026-09-13-evaluation/EVALUATION_CONTRACT.md` and lessons from its specialized `evaluate_saved.py`: body versus directory, host versus voluntary, later-model opportunity, and recovered-prefix caution. This reusable CLI does not replace that evaluator's explicit reference adjudication or recovered-prefix verification. The two tools have different declared coverage.

## Offline annotations and unsupported claims

JSON exports a review template for each peer-body observation and indexed append, plus a run-level blank template with exact history/message/call references. Recognition, directed request, reply relevance, uptake, and error propagation start `unassessed`. Reviewers may copy the template into a separate artifact, link further source/request/downstream references and exact quotes, and use `supported`, `contradicted`, `insufficient-evidence`, or `unassessed`. These are semantic claim states, separate from mechanical opportunity status. Annotation ingestion, automated semantic inference, reply linking, answer/reference correctness, entailment, and causal benefit scoring are intentionally unsupported. An append/read count cannot establish any of these claims.

## Synthetic validation

```bash
uv run python -m pytest orchestrator/simulated_web/test_evaluate_saved_runs.py -q
```

A mock-only runner integration test checks actual emitted artifacts; static fixtures also mirror `exchange_evidence.EvidenceIndex.phase`, `synchronized_exchange.expose_entries`, `synchronized_notebooks.call`, and result metadata emitted by `timed.run_phase`. Tests cover provenance separation, missing/partial/resume coverage, append versus delivery, body/terminal receipt, alias ownership, malformed inputs, duplicate events, safe outputs, no mutation, and CLI help. No real experimental run is evaluated by these tests.
