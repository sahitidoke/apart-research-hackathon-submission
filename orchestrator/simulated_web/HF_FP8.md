# Future HF FP8 paired-view runs

This is an opt-in migration to official `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Existing Ollama launchers,
model cache, run directories and Q4 checkpoints retain their identity.
The new entry point is `python -m orchestrator.simulated_web.modal_hf_paired_views`.
It accepts the same seven fresh input files as `modal_paired_notebook_views`:
`--dataset`, `--topic-file`, `--editable-sources`, `--question-ids`,
`--access-manifest`, `--visible-labels`, and `--round-leaders`.
Use a fresh `--run-id` containing `fp8`. Local validation is the default;
`--validate-only` is explicit equivalent. Validation does not create a run,
download weights, build a cloud image, or demonstrate GPU readiness.

Only after execution authorization, add `--launch`. An empty remote cache
additionally requires `--download-model`; there is no implicit weight download.
The dedicated HF volume is `germanwiki-qwen38-hf-fp8-models`; results use the
existing `germanwiki-timed-runs` volume under fresh IDs. The CLI intentionally
does not accept old-checkpoint resume or model replacement. The common runner
can resume a new FP8 checkpoint only with the same FP8 client; its saved profile
cannot be overridden with the legacy model.

## What is pinned and what changes

The HF API inventory is saved in `hf_fp8_lock.json`. Before setup artifacts or
server initialization, every cached file is size- and checksum-verified: Git
blob SHA1 for small files, LFS SHA256 for large files, including the weights,
tokenizer, config and chat template. Missing/corrupt cache is an error.
There are 24,699,207,680 F8_E4M3 and 3,082,220,272 BF16 parameters in the public
inventory. FP8 means the official block128x128 E4M3 weights with dynamic
activation scaling and declared BF16 exclusions; it does not mean every
tensor is FP8. Q8_0 integer weights are never substituted.

The owned loopback backend is vLLM 0.24.0, image
`vllm/vllm-openai:0.24.0`, with Transformers 5.8.0. It requests one L40S,
one sequence, 65,536 context, eager execution and 90% GPU allocation.
`--quantization fp8` is explicit. KV uses `auto` with explicitly selected
BF16 compute dtype, hence the intended cache is BF16; it is not FP8 KV.
This follows vLLM's documented auto-dtype behavior and avoids its restricted
explicit BF16-cache override. Runtime kernel/cache dtype has not been observed.
Recurrent-state cache behavior remains the pinned backend's default.

Thinking effort is explicitly medium with historical reasoning retained.
The pinned HF template, vLLM `qwen3` reasoning parser and `qwen3_coder` tool
parser replace Ollama's renderer/parser. Tool results are matched in order to
stable generated call IDs; unmatchable history fails. Final-only generation
disables thinking through the same template kwargs used in preflight.
Generation uses temperature1.0, top_p0.95, top_k20, seed0, explicit native
token ceilings and no implicit model generation-config overrides.

Each request first calls the owned server's `/tokenize` with the same
messages, tools and rendering kwargs; native count and returned context limit
are checked. Generation must return exactly the preflight prompt count and a
bounded complete native generation count. Mismatch fails and preserves history;
there is no heuristic tokenizer fallback or silent truncation.

The nonstream OpenAI-compatible response is translated to the existing runner
message format. Every incomplete generation kills the owned process group,
then the existing Linux cleanup verifies no live member remains. This sacrifices
server retention on cancellation; restart must fit the existing readiness cap.
An interrupted nonstream request has no recoverable partial response text or
native completion count. The event explicitly records that limitation; it is
never scored as a successful zero-token response.

## Verification boundary

Mock checks cover message/thinking/tool mapping, native context and usage
checks, cancellation dispatch, cached-file validation, legacy rejection before
output, and paired schedule/checkpoint/resume integration. They do not establish
L40S memory fit, actual kernels, startup latency, parser behavior on real output,
throughput, accuracy or cancellation latency. The image has not been built and
no weights, server or experimental agent have been launched for this migration.

This changes backend, template/parser and KV precision along with weights.
Treat future runs as a new configuration, not an isolated Q4-versus-FP8 ablation.
The official HF tokenizer has not been proven token-identical to the historic
GGUF tokenizer; it cannot retroactively supply exact counts for old prompts.

## Primary source evidence inspected September 12, 2026

- [Official FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Pinned metadata inventory](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/revision/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a?blobs=true)
- [Pinned HF configuration](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json)
- [Pinned HF chat template](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/chat_template.jinja)
- [vLLM model recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [v0.24.0 tokenize protocol](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/serve/tokenize/protocol.py)
- [v0.24.0 chat protocol](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/openai/chat_completion/protocol.py)
- [v0.24.0 history conversion](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/chat_utils.py)
- [v0.24.0 cache configuration](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/cache.py)
