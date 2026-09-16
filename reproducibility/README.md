# Final-suite launch wrappers

These ten shell scripts preserve the selected arguments of the development 12a–12e seed 0/1 wrappers while replacing machine-specific paths and uv invocations. They default to local validation; they do not launch automatically.

Set `INPUT_DIR` to an absolute directory with the seven files listed in `input-manifest.json`. Those inputs are **not included**: the dataset contains third-party paragraphs and the remaining files describe the historical task selection, partition and discovery policies. The manifest records exact byte counts and SHA-256 hashes for matching copies. Use the existing dataset/export and partition implementation as reference when preparing a new study; different inputs constitute a new configuration, not an exact reproduction.

```sh
export INPUT_DIR=/absolute/path/to/authorized-inputs
export MODAL_PROFILE=research-profile
bash reproducibility/run-12a-seed1.sh --validate-only
```

The shared launcher currently validates the explicit profile name `research-profile`. Configure your own Modal profile under that name (or deliberately change `PROFILE` and corresponding test fixtures); no credentials are supplied. Validation may check local Modal configuration, and `uv` may need to download locked dependencies on first use.

**12a seed 0 is historical continuation.** Its script deliberately retains the original relative checkpoint path under `runs/downloads/10d2-20260913/` and parent run ID. That checkpoint and referenced transcripts are excluded, so this wrapper cannot validate or reproduce the continuation without them. Do not silently substitute a fresh start and describe it as the historical baseline. Other scripts start fresh unless explicit compatible resume arguments are supplied.

Current 12b–12d default IDs/timing represent the versioned 600-second answer safety setting. Not every completed historical job used that setting. Exact run-level comparison requires saved settings and lineage, which this code-only release does not contain. No frozen archive, analysis outputs, or result logs are bundled.

After reviewing configuration and costs, a maintainer can explicitly request `--launch`, optionally with `--detach` and `--download-model`. That is a paid execution step and was not performed for this release.
