"""Run a provenance batch on a Modal GPU container.

    modal run orchestrator/provenance/modal_run.py --output-name prov-01

A deliberately thin wrapper: it builds the argv `batch.py`'s own `main()` would
parse locally and calls it unchanged inside the container, so the remote path
and the local one cannot drift apart. Two volumes, not one, so the
Hugging Face cache is never affected by deleting or renaming a run.

Unlike the MARL runner this trains nothing and writes no weights, but it is
still a billed GPU job and still falls under the repository rule: do not run
it without being asked to.
"""
import pathlib

import modal

APP_NAME = "agent-swarming-provenance"
# MuSiQue-Ans comes from the Hub by default -- `bdsaglam/musique`, config
# `answerable`, which preserves the original AllenAI schema this package reads.
# It lands in the same Hugging Face cache volume as the weights, so it is
# fetched once across runs. A local export is still accepted and is mounted
# when present, for an air-gapped or pinned copy.
HUB_DATASET = "hf:bdsaglam/musique:answerable:validation"
DATASET = pathlib.Path("musique_ans_v1.0_dev.jsonl")
REMOTE_DATASET = "/root/musique_ans_v1.0_dev.jsonl"
# 48GB. The default policy is Qwen3.8-27B in FP8: roughly 27GB of weights, and
# the rest is KV cache for a batch of agent turns at a 64k context, which is
# cheap on this architecture because only 16 of its 64 layers are full
# attention. An A10G (24GB) does not fit the weights at all, and the BF16
# checkpoint (~54GB) does not fit this card -- for that one pass
# `--gpu "H100:1"` or `--gpu "L40S:2"` with a matching --tensor-parallel-size.
GPU = "L40S"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM pulls its own matching torch build, so torch is deliberately not
    # pinned separately here the way the MARL image pins cu126 -- two pins
    # would fight, and vLLM's own resolution is the one that has to win.
    # No autoawq: that package *produces* AWQ checkpoints and pins its own
    # torch, which would fight the build vLLM brings. Loading an existing AWQ
    # checkpoint for inference is native to vLLM and needs nothing extra.
    # vLLM and Transformers floors are both set by the policy: the `qwen3_5`
    # architecture Qwen3.8-27B uses exists in neither of the versions the
    # previous Qwen2.5 default ran on.
    .pip_install("vllm>=0.29", "transformers>=5.0", "wandb>=0.17", "datasets>=3.0")
    .env({"PYTHONPATH": "/root", "HF_HOME": "/root/.cache/huggingface",
          # vLLM's default multiprocessing method deadlocks in some container
          # runtimes; spawn is the supported setting for containerized serving.
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          # flashinfer's sampler JIT-compiles a CUDA kernel and needs a full
          # toolkit, which these pip wheels do not provide. `generator.py`
          # already avoids that path for greedy runs by not asking for top-k or
          # top-p; this keeps a nonzero-temperature run working too, rather
          # than adding a multi-gigabyte toolkit to the image for one kernel.
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .workdir("/root")
    # Last, so editing the experiment code does not reinstall vLLM.
    .add_local_dir("orchestrator", remote_path="/root/orchestrator",
                   ignore=["**/__pycache__", "**/*.pyc", "**/.pytest_cache"])
)
if DATASET.exists():
    image = image.add_local_file(str(DATASET), remote_path=REMOTE_DATASET)

app = modal.App(APP_NAME, image=image)
runs_volume = modal.Volume.from_name(f"{APP_NAME}-runs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)
# The key lives in Modal, never in this file, the image or the repo. Create it
# once with:  modal secret create wandb-secret-2 WANDB_API_KEY=...
# `required_keys` is kept although the usual snippet omits it: without it a
# secret that exists under a different variable name still mounts, wandb.init
# falls back to anonymous, and the run throws its metrics away while looking
# like it worked.
WANDB_SECRET = "wandb-secret-2"
wandb_secret = modal.Secret.from_name(WANDB_SECRET, required_keys=["WANDB_API_KEY"])


@app.function(gpu=GPU, volumes={"/runs": runs_volume,
                                "/root/.cache/huggingface": hf_cache_volume},
              secrets=[wandb_secret], timeout=6 * 60 * 60)
def run_batch(output_name: str, extra_args: list[str]):
    """`extra_args` is every batch.py flag verbatim except --output, which
    always points at the runs volume. batch.py refuses a directory that
    already exists, so a reused name fails rather than merging two runs."""
    import sys

    from orchestrator.provenance.batch import main as batch_main

    sys.argv = ["batch.py", "--output", f"/runs/{output_name}", *extra_args]
    try:
        return batch_main()
    finally:
        # Committed even on failure: the episodes and replays completed before
        # a crash are the expensive part and are still worth keeping.
        runs_volume.commit()


@app.function(volumes={"/runs": runs_volume}, secrets=[wandb_secret], timeout=30 * 60)
def run_audit(output_name: str, wandb: bool = True, audit_name: str = ""):
    """CPU-only. Also runnable locally after `modal volume get`, which is the
    point of keeping scoring out of the GPU function: re-weighting the
    provenance score must never mean paying for another batch.

    `audit_name` directs the output somewhere other than `<run>/audit`, which
    `audit.py` refuses to overwrite. Re-auditing a run after adding an arm to it
    needs that: the previous audit is the record of what the previous scoring
    said, and a re-score that silently replaced it would leave no way to see
    which number came from which battery.
    """
    import sys

    from orchestrator.provenance.audit import main as audit_main

    sys.argv = ["audit.py", "--run-root", f"/runs/{output_name}"]
    if audit_name:
        sys.argv += ["--output-dir", f"/runs/{output_name}/{audit_name}"]
    if wandb:
        sys.argv += ["--wandb", "--wandb-name", output_name]
    try:
        return audit_main()
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def main(output_name: str = "prov-01", dataset: str = "", hops: str = "2,3,4",
         per_hop: int = 30,
         conditions: str = "closed_book,hop_probe,isolated,solo,flat,planner",
         planner_style: str = "persuasive", worker_style: str = "standard",
         model: str = "", quantization: str = "", seed: int = 0,
         gpu: str = GPU, tensor_parallel_size: int = 1, workers: int = 2,
         max_num_seqs: int = 16, max_new_tokens: int = 1024,
         resume: bool = False, paraphrase_mode: str = "", redo: str = "",
         paraphrases: str = "", paraphrases_sentence: str = "",
         localize: str = "seeded", audit: bool = True, audit_name: str = "",
         wandb: bool = True,
         smoke: bool = False, session: bool = False, rounds: int = 10,
         block: int = 5, cohorts: int = 1, budget_shape: str = "declining",
         calibrate_from: str = ""):
    """Defaults to MuSiQue-Ans from the Hub, cached in the same volume as the
    weights. Pass `--dataset` for a different source, or put
    `musique_ans_v1.0_dev.jsonl` at the repository root to mount a local export
    and name it explicitly.

    `--smoke` runs the three-question fixture. It is the only way to get it:
    there is no silent fallback, because a GPU run that quietly answered three
    invented questions would produce a full set of plausible-looking numbers.

    `--wandb` tracks the run. Nothing here is trained, so that is experiment
    tracking rather than optimization; `sweep.yaml` is the searching version,
    with the caveat it needs."""
    extra = ["--hops", hops, "--per-hop", str(per_hop), "--conditions", conditions,
             "--planner-style", planner_style, "--worker-style", worker_style,
             "--tensor-parallel-size", str(tensor_parallel_size), "--seed", str(seed),
             "--workers", str(workers), "--localize", localize,
             "--max-num-seqs", str(max_num_seqs),
             "--max-new-tokens", str(max_new_tokens)]
    if resume:
        extra += ["--resume"]
    # Adding an arm to a finished run: `--resume` skips per arm, so the identity,
    # ablation and swap replays already on the volume are reused and only the
    # new perturbations cost a GPU. The frozen paraphrase sets are named inside
    # the volume, not on this machine.
    if paraphrase_mode:
        extra += ["--paraphrase-mode", paraphrase_mode]
    if paraphrases:
        extra += ["--paraphrases", f"/runs/{paraphrases}"]
    if paraphrases_sentence:
        extra += ["--paraphrases-sentence", f"/runs/{paraphrases_sentence}"]
    if redo:
        extra += ["--redo", redo]
    # Passed only when asked for: a pre-quantized checkpoint declares its own
    # scheme, and naming a different one fails at load.
    if quantization:
        extra += ["--quantization", quantization]
    if session:
        # `--calibrate-from` names a finished run *inside the volume*, so the
        # schedule is built from what that run really spent rather than from
        # fractions chosen here.
        extra += ["--session", "--rounds", str(rounds), "--block", str(block),
                  "--cohorts", str(cohorts), "--budget-shape", budget_shape]
        if calibrate_from:
            extra += ["--calibrate-from", f"/runs/{calibrate_from}"]
    if smoke:
        extra += ["--smoke"]
    elif dataset:
        extra += ["--dataset", dataset]
    else:
        extra += ["--dataset", HUB_DATASET]
    if model:
        extra += ["--model", model]
    if wandb:
        extra += ["--wandb", "--wandb-name", output_name]
    # `with_options` rather than a second decorated function: the GPU is a
    # deployment detail of the same code, and a bigger checkpoint should not
    # need a duplicate entrypoint to run on a bigger card.
    run_batch.with_options(gpu=gpu).remote(output_name=output_name, extra_args=extra)
    if audit:
        run_audit.remote(output_name=output_name, wandb=wandb, audit_name=audit_name)
    print(f"Done. Fetch the run with:\n"
          f"  modal volume get {APP_NAME}-runs {output_name} runs/{output_name}")
