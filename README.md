# Agent swarming: browser-mediated communication experiments

Code accompanying an Apart Research hackathon project on whether independently prompted document-QA agents discover, read, and reuse one another's notebook entries. The environment is a host-side browser simulator: agents use local document and notebook routes; host code performs model calls, scheduling, accounting, checkpointing, and evaluation.

This snapshot includes the working implementation, including files that were uncommitted in the development repository. It is a code release, not a release of private transcripts or a claim of independent result reproduction. `SUBMISSION_MANIFEST.json` records source and released hashes. Personal paths and the Modal profile name have been replaced with public placeholders.

## Setup

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/), then run from this repository:

```sh
uv sync --frozen --group dev
```

The lock file includes the host-side Modal and test dependencies. Model weights, GPU inference software and cloud credentials are not bundled. Remote HF jobs declare their pinned vLLM image and model inventory in `orchestrator/simulated_web/hf_fp8.py`, `hf_fp8_lock.json`, and the Modal launcher modules. Legacy local runners expect a separately installed Ollama server and model.

## Code map

- `orchestrator/simulated_web/browser.py` and `runner.py`: document navigation, per-run shared storage, and the original local model loop.
- `timed.py`, `timed_policy.py`, `timed_checkpoint.py`, and `timed_transport.py`: bounded phases, accounting, failure handling, checkpoints, and transport.
- `concurrent_research_reflection.py`, `research_reflection.py`, and notebook modules: initial research, answers, reflection, publication barriers, and agent-specific views.
- `modal_hf_reference_12a.py` through `modal_hf_reference_12e.py`: the final reference and comparison conditions; portable wrappers live in `reproducibility/`.
- `reference_ablations.py`, `reference_repair.py`, `reference_resume.py`, and `peer_note_exposure.py`: intervention policies, versioned repairs, checkpoint compatibility, and bounded automatic exposure.
- `evaluate_saved_runs.py` and `EVALUATION.md`: saved-artifact evaluation tools and evidence definitions.
- `orchestrator/containment_diagnostic/`, `orchestrator/transfer_diagnostic/`, and `slurm/`: earlier diagnostic and cluster runners, retained as implementation context.
- `corpora/yellowstone-records/`: a small NPS corpus, offline builder, provenance and task files. This is an earlier demonstration corpus, not the final suite's dataset.

The module-level documentation describes several historical configurations. It is not a unified statement of the final protocol. `EXPERIMENT_PLAN.md` is an earlier research plan; proposals in it should not be read as completed work.

## Final condition family

| Condition | Main difference |
| --- | --- |
| 12a | Reference: descriptive agent-written notebook titles, complementary source discovery, no visible agent history |
| 12b | Generic notebook entry titles |
| 12c | Own-only notebook access |
| 12d | Same-union source discovery for both agents |
| 12e | Predetermined delivery of the latest published peer-note body, bounded to 2,000 Unicode codepoints |

The latest executable includes the reserved-final/cancellation repair. Current 12b–12d wrappers use a 600-second answer safety cap; 12a and 12e retain 180 seconds. The generated-token allowance remains 2,048 including a 512-token final reserve. Historical 12a seed 0 continued an earlier checkpoint; fresh seeds and conditions are not identical starting states. These differences must be retained when interpreting comparisons. Required notebook publication and forced exposure are distinct from voluntary discovery or causal collaboration benefit.

See [reproduction inputs and limitations](reproducibility/README.md) before using the final launchers. Missing historical inputs and checkpoints prevent an out-of-the-box exact rerun.

## Checks and execution boundaries

A focused, model-free check command for the included local fixture is:

```sh
uv run --frozen python -m pytest orchestrator/simulated_web/test_simulated_web.py tests/test_yellowstone_corpus.py -q
```

This command is provided for maintainers and was **not run while preparing this release**. Static Python parsing, JSON parsing, shell syntax, import-target, manifest, size, and targeted secret-pattern checks are recorded in `VALIDATION.md`. Some historical integration tests refer to excluded private checkpoints or research-log inputs; the complete test collection is not claimed to run unchanged without those artifacts.

Cloud launchers default to validation. `--launch` spends compute and `--download-model` permits weight downloads. Use unique run IDs, configure your own credentials, and inspect the selected module before launch. No model calls, test suite, services, or cloud jobs were run to prepare this snapshot.

## Data, rights, and interpretation

Original NPS source text is accompanied by its existing source URLs, retrieval metadata and rights notice in the corpus README and manifest. No images are included. Tiny fictional fixtures support software tests. A small host-only replay fixture contains explicitly labelled observed and synthetic snippets for regression testing; it is not a full run or independent result.

The MuSiQue-derived final dataset, private run artifacts, journals, checkpoints, model weights, and manuscript are not included. Dataset and checkpoint hashes describe omitted inputs without granting rights to them. No new license is assigned to the project's code by this snapshot; third-party source material retains its existing terms.

The simulator is not an operating-system sandbox. Model-generated text and browser pages are untrusted; answer keys, settings, private logs and evaluation artifacts must remain outside agent-visible routes. The code and fixtures alone do not establish reliable collaboration, performance improvement, harmful collusion, or successful containment.
