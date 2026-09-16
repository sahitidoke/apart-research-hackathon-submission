# Final-suite launch wrappers

These ten shell scripts preserve the selected arguments of the development 12a–12e seed 0/1 wrappers while replacing machine-specific paths and uv invocations. They default to local validation; they do not launch automatically.

Set `INPUT_DIR` to an absolute directory with the seven browser-suite files listed in `input-manifest.json` (the eighth entry belongs to the provenance audit below and is not used by these ten scripts). Those inputs are **not included**: the dataset contains third-party paragraphs and the remaining files describe the historical task selection, partition and discovery policies. The manifest records exact byte counts and SHA-256 hashes for matching copies. Use the existing dataset/export and partition implementation as reference when preparing a new study; different inputs constitute a new configuration, not an exact reproduction.

```sh
export INPUT_DIR=/absolute/path/to/authorized-inputs
export MODAL_PROFILE=research-profile
bash reproducibility/run-12a-seed1.sh --validate-only
```

The shared launcher currently validates the explicit profile name `research-profile`. Configure your own Modal profile under that name (or deliberately change `PROFILE` and corresponding test fixtures); no credentials are supplied. Validation may check local Modal configuration, and `uv` may need to download locked dependencies on first use.

**12a seed 0 is historical continuation.** Its script deliberately retains the original relative checkpoint path under `runs/downloads/10d2-20260913/` and parent run ID. That checkpoint and referenced transcripts are excluded, so this wrapper cannot validate or reproduce the continuation without them. Do not silently substitute a fresh start and describe it as the historical baseline. Other scripts start fresh unless explicit compatible resume arguments are supplied.

Current 12b–12d default IDs/timing represent the versioned 600-second answer safety setting. Not every completed historical job used that setting. Exact run-level comparison requires saved settings and lineage, which this code-only release does not contain. No frozen archive, analysis outputs, or result logs from this browser suite are bundled. (The provenance audit below is the one exception, and a narrow one: it bundles eight rendered figures and no run data.)

After reviewing configuration and costs, a maintainer can explicitly request `--launch`, optionally with `--detach` and `--download-model`. That is a paid execution step and was not performed for this release.

---

# Provenance audit

A separate experiment with its own three wrappers. It asks whether a planner-led
collective's answer was *derived from evidence* or *deferred to the planner*, by
replaying each episode with individual bus messages ablated, paraphrased or
swapped and watching whether the answer moves. The code is
`orchestrator/provenance/`; read its `README.md` before running anything, and
`FINDINGS.md` for what the two collected runs found.

```sh
export MODAL_PROFILE=research-profile
bash reproducibility/run-prov-ind-seed0.sh            # --validate-only is the default
bash reproducibility/run-prov-ind-seed1.sh            # the second question draw
bash reproducibility/run-prov-audit.sh                # CPU-only scoring of a finished run
```

**The validation mechanism differs from the ten scripts above.** Those pass
`--validate-only` through to a runner that implements it.
`orchestrator/provenance/modal_run.py` has a plain Modal entrypoint with no such
flag, so these three do the validating themselves: they check `MODAL_PROFILE`,
print the fully-resolved `modal run` command, and execute nothing. `--launch` is
required to actually run, and only then is `--detach` accepted. The guarantee is
the same — nothing launches by itself — but the mechanism is not, and a reader
comparing the scripts should know which they are looking at.

Collection is a billed GPU job (L40S, several hours). Scoring is CPU-only and
re-runnable: a fresh `--audit-name` writes a new audit beside the old one instead
of over it, which is what makes a re-score comparable with the score it replaced.

## Offline checks need nothing installed

```sh
python -m unittest discover -s orchestrator/provenance -p 'test_*.py'
```

346 tests, a few seconds, **no model, no GPU, no download, nothing installed** —
not even a virtualenv. Everything in the package is stdlib-only outside its vLLM
wrapper, deliberately: a plumbing test that needs a card stops being run. Every
one of these exercises the real control flow against scripted agents, including
two that drive collect → freeze → replay → audit end to end.

These tests are also collected by `python -m pytest` from the repository root,
via this repository's `testpaths`.

## What is bundled, and what is not

The eight figures embedded in `FINDINGS.md` live in
`orchestrator/provenance/figures/` (PNG at 300 dpi and PDF for each). **Their
inputs are not bundled.** Every plot script reads an `audit.json` produced by a
scoring pass, and no run artifacts — transcripts, `batch.jsonl`, `replays.jsonl`,
`audit.json` — are in this repository. The committed figures are the surviving
output of runs `prov-ind` (seed 0) and `prov-ind-s1` (seed 1); they cannot be
re-rendered until a collection and scoring pass regenerates their input.

Nor is the dataset. `input-manifest.json`'s eighth entry records the byte count
and SHA-256 of the MuSiQue-Ans export used, so a copy can be matched, but it is
third-party data and is not redistributed here. It is also **optional**:
`modal_run.py` defaults to `hf:bdsaglam/musique:answerable:validation` from the
Hub, which preserves the original AllenAI schema the package validates. A local
export is for a pinned or air-gapped copy.

## Plot scripts

`plots/` holds four, all of which read `audit.json` and nothing else. They never
import `orchestrator.provenance`, so matplotlib cannot enter that package's
dependency graph and **no figure can change a number** — everything they draw was
already computed by `audit.py`.

```sh
uv venv /tmp/plotenv && uv pip install --python /tmp/plotenv/bin/python matplotlib
/tmp/plotenv/bin/python reproducibility/plots/plot_seeds.py \
    --seed0 runs/prov-ind/audit-v2/audit.json \
    --seed1 runs/prov-ind-s1/audit/audit.json -o orchestrator/provenance/figures
```

| script | draws |
|---|---|
| `plot_seeds.py` | the three figures whose numbers depend on the question sample, both draws in one frame |
| `plot_thesis.py` | the screen artifact and the channel-influence measurement |
| `plot_findings.py` | the seed-0 observations in discovery order |
| `plot_roc.py` | ROC curves and Hanley–McNeil intervals behind the detection result |

**They generate more figures than this repository bundles**, which is deliberate:
only the eight the write-up embeds are committed, and the rest are reproducible
from the same commands rather than carried as unreferenced files.

## Seeds are question draws, not noise

Sampling is greedy, so a seed does not resample. It redraws *which* questions are
asked and *which* distractor is planted. A second seed is therefore a second
**question sample**, and running one is not a noise estimate — at temperature 0
identical prompts produce byte-identical output, which is what the 210/210
identity-replay result demonstrates. `APPENDIX.md` §I has the details.

This matters for reading the two runs: the deference rate moved from 40% to 64%
between draws (both far from zero, statistically consistent, p = 0.121), and one
earlier finding did not survive the second draw at all. Two runs at n = 30 give
two wide intervals; one run at n = 150 would give one useful one.

## Authorization

Same rule as the rest of this repository: **do not launch a run without being
asked to.** The bundled three-question fixture is a smoke test whose numbers are
not results — it has no default and must be requested with `--smoke`, which
stamps the manifest. A run with no `--dataset` is refused rather than quietly
given the fixture. Credentials are never in this repository, in an image, in a
config, or in a default argument.
