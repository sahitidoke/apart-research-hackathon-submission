# Release preparation checks

Performed on the staged working-tree snapshot without importing runner modules or executing tests:

- Python AST parse: 186 files passed.
- JSON parse: 15 files passed (including the source comparison manifest).
- `bash -n`: 15 shell/Slurm scripts passed.
- Static absolute `orchestrator.*` import-target check: no missing local modules. This does not resolve every dynamic or relative import.
- Source comparison manifest: 241 copied/adapted files matched their recorded staged SHA-256 values.
- Targeted scan: no private-key blocks, common Hugging Face/OpenAI/GitHub/AWS key patterns, original personal usernames/profile, or macOS user-home paths detected. Pattern scans do not prove the absence of every possible secret.
- No symlinks, binary caches, virtual environments, full runs, research journals, model weights, or files exceeding 1 MB are included.

Not performed: pytest, dependency installation, importing cloud entry points, dataset rebuilding, model execution, GPU/backend readiness, cloud deployment, historical result reproduction, or license adjudication. Existing corpus provenance and rights statements are retained. Full test-suite portability is not established: some historical tests require omitted input/checkpoint artifacts.

`SUBMISSION_MANIFEST.json` compares copied files to the development working tree; added release documentation and the input-hash manifest have no source counterpart. Personal path/profile substitutions and portable launcher transformations are labelled. The snapshot is based on the recorded Git revision plus then-current uncommitted and untracked files, not solely that revision.
