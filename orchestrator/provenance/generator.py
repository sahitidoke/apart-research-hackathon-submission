"""Text generation behind one small interface.

Two implementations, and the seam between them is the point: `episode.py` and
`replay.py` only ever call `render` and `generate`, so every test in this
package runs the real control flow against scripted completions with no model,
no GPU and no download.

vLLM rather than `transformers.generate`: the latter exists to compute gradients
through an adapter and needs the model in the training process. Here nothing is
trained, the model is quantized, and what matters is throughput over a batch of
agent turns. vLLM also loads a quantized checkpoint directly, which a
Transformers 4.x path cannot do for the Qwen3.5 (`qwen3_5`) architecture
Qwen3.8-27B is built on — hence the Transformers 5 floor in `modal_run.py`.

Sampling defaults to greedy. That is deliberate and load-bearing for the
audit: if temperature were nonzero, a final answer that changed after a
perturbation could always have changed anyway, and every flip rate would carry
a resampling floor that has to be estimated and subtracted. At temperature 0
the episode is a function of its inputs, an unperturbed replay reproduces the
original byte for byte, and a flip means the edited message mattered.
"""
import json

# Qwen3.8-27B, in Qwen's own FP8 build: roughly 27GB of weights, which fits an
# L40S (48GB) with room for the cache. The BF16 checkpoint is 18 shards and
# ~54GB and does not, so it needs two cards or an 80GB one.
#
# The architecture is `qwen3_5`, which needs Transformers 5 and a vLLM new
# enough to have it in the model registry -- see the `provenance` extra. It is
# natively vision-language; nothing here sends an image, and `VLLMGenerator`
# turns the multimodal limits off so the profiler does not reserve memory for
# inputs this experiment never produces.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
# None means "read it off the checkpoint". A pre-quantized checkpoint declares
# its own scheme in `quantization_config`, and naming a different one here
# fails at load; the flag stays for checkpoints that do not declare one.
DEFAULT_QUANTIZATION = None
# Asked for 64k, and clamped down to whatever the checkpoint actually supports
# -- see `supported_context`. A four-worker episode carries every agent's
# paragraphs plus a bus that grows every turn, so more is genuinely useful.
# Qwen3.8-27B declares 262144 positions, so this is a memory budget rather than
# an architectural ceiling: only 16 of its 64 layers are full attention, the
# other 48 are linear and keep a constant-size state, so a 64k sequence costs
# about 4GB of KV cache rather than the ~16GB a dense 64-layer model would.
# The window is per sequence; the cache it draws from is sized by
# gpu_memory_utilization, not by this.
DEFAULT_MAX_MODEL_LEN = 65536
# Concurrent sequences. vLLM defaults to 256, which this architecture cannot
# give you: each of the 48 Gated DeltaNet layers keeps a constant-size
# recurrent state *per sequence in flight*, so concurrency is capped by how
# many of those states fit beside the weights. On an L40S holding 28GB of FP8
# weights that is 167, and asking for 256 fails at startup rather than
# degrading -- "max_num_seqs (256) exceeds available Mamba cache blocks (167)".
#
# 16 is well above what this experiment uses. An episode generates one prompt
# at a time (`episode._turn` hands `generate` a single-element list); the only
# batched call is the paraphrase freeze, which this comfortably covers. The
# blocks not spent on idle concurrency are left for the KV cache instead.
DEFAULT_MAX_NUM_SEQS = 16


class ScriptedGenerator:
    """Canned completions for tests. `script(agent, step, messages) -> text`."""

    def __init__(self, script, max_model_len=DEFAULT_MAX_MODEL_LEN):
        self.script = script
        self.max_model_len = max_model_len
        self.calls = []
        self.model = "scripted"
        self.settings = {"model": "scripted", "temperature": 0.0}

    def render(self, messages):
        return json.dumps(messages)

    def count_tokens(self, prompt):
        return len(prompt) // 4 + 1

    def generate(self, prompts, max_new_tokens=512, tags=None):
        tags = tags or [(None, None)] * len(prompts)
        outputs = []
        for prompt, (agent, step) in zip(prompts, tags):
            text = self.script(agent, step, json.loads(prompt))
            self.calls.append((agent, step))
            # A deterministic stand-in for a tokenizer, so budget arithmetic is
            # exercised offline. Four bytes per token is the same rough ratio
            # `simulated_web` uses for its own preflight estimates.
            outputs.append({"text": text, "finish_reason": "stop",
                            "prompt_tokens": len(prompt) // 4 + 1,
                            "completion_tokens": len(text) // 4 + 1})
        return outputs


def supported_context(config, requested):
    """The largest window this checkpoint really has, never more than asked for.

    vLLM refuses a `max_model_len` above the model's declared positions unless
    an override env var is set, and that refusal is correct: with rotary
    embeddings, positions past the trained maximum do not degrade gracefully,
    they produce NaN. So the request is treated as a ceiling to aim for rather
    than a figure to insist on, and the clamp is recorded in settings so a run
    cannot quietly claim a context it never had.
    """
    declared = getattr(config, "max_position_embeddings", None)
    text_config = getattr(config, "text_config", None)
    if declared is None and text_config is not None:
        declared = getattr(text_config, "max_position_embeddings", None)
    if declared is None:
        return requested
    return min(requested, int(declared))


class VLLMGenerator:
    """A quantized instruct model served in-process by vLLM."""

    def __init__(self, model=DEFAULT_MODEL, quantization=DEFAULT_QUANTIZATION,
                 max_model_len=DEFAULT_MAX_MODEL_LEN, tensor_parallel_size=1,
                 gpu_memory_utilization=0.90, temperature=0.0, seed=0, dtype="auto",
                 max_num_seqs=DEFAULT_MAX_NUM_SEQS):
        # Imported here rather than at module scope on purpose, and it is the
        # one exception to this repository's imports-at-the-top rule: every
        # other module in this package is stdlib-only so the test suite runs
        # without vLLM installed, and a top-level import would make importing
        # `generator` at all require a CUDA build.
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        requested = max_model_len
        max_model_len = supported_context(AutoConfig.from_pretrained(model), max_model_len)
        self.clamped_from = requested if max_model_len != requested else None
        if self.clamped_from:
            print(f"Context window clamped from {requested} to {max_model_len}: "
                  f"{model} declares no more. Raising it past what the checkpoint "
                  f"supports makes RoPE return NaN rather than extending anything.")
        # No image or video is ever sent here. Saying so keeps vLLM's memory
        # profiler from reserving encoder and cache space for dummy multimodal
        # inputs on a vision-language checkpoint -- memory that would otherwise
        # come straight out of the KV cache this experiment does use.
        self.llm = LLM(model=model, quantization=quantization, dtype=dtype,
                       max_model_len=max_model_len, seed=seed,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=max_num_seqs,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.sampling = SamplingParams
        self.model = model
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.settings = {"model": model, "quantization": quantization,
                         "max_model_len": max_model_len,
                         "max_model_len_requested": requested,
                         "temperature": temperature,
                         "tensor_parallel_size": tensor_parallel_size,
                         "gpu_memory_utilization": gpu_memory_utilization,
                         "max_num_seqs": max_num_seqs,
                         "dtype": dtype, "seed": seed}

    def render(self, messages):
        """The exact prompt string the model is given, and the replay cache key.

        `enable_thinking=False` is passed unconditionally, and it is
        load-bearing on Qwen3.8: its template treats thinking as *on* whenever
        the variable is undefined, at `reasoning_effort` xhigh, and only a
        literal false makes it close the `<think>` block itself before the
        model writes anything. Leaving it on would not break parsing -- text
        outside a `<msg>` block is private by design -- but it would spend the
        per-turn token budget on reasoning and end turns at `length_limit`
        before any message was emitted. Older Qwen2.5 templates ignore the
        unknown variable, so the same call is right for both.
        """
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)

    def count_tokens(self, prompt):
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def sampling_params(self, max_new_tokens):
        """Greedy asks for nothing but a temperature of zero.

        Passing `top_p`/`top_k` alongside it is not merely redundant: it sends
        the request down vLLM's flashinfer sampling path, which JIT-compiles a
        CUDA kernel and fails outright on an image without `nvcc`. Greedy
        argmax needs no such kernel, so the truncation parameters are supplied
        only when there is actually a distribution to truncate.
        """
        if self.temperature == 0:
            return self.sampling(temperature=0.0, max_tokens=max_new_tokens)
        return self.sampling(temperature=self.temperature, top_p=1.0, top_k=-1,
                             max_tokens=max_new_tokens)

    def generate(self, prompts, max_new_tokens=512, tags=None):
        parameters = self.sampling_params(max_new_tokens)
        results = self.llm.generate(prompts, parameters, use_tqdm=False)
        outputs = []
        for result in results:
            completion = result.outputs[0]
            # Native counts from the engine, not an estimate. Budgets are
            # charged in these, and they are logged so a replay charges exactly
            # what the original run did.
            outputs.append({"text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "prompt_tokens": len(result.prompt_token_ids),
                            "completion_tokens": len(completion.token_ids)})
        return outputs


class CachedGenerator:
    """Replays logged completions when the prompt is unchanged, generates when not.

    This is how a perturbed replay keeps its untouched prefix honest. Rather
    than tracking "the first edited message" and pinning everything before it,
    the cache is keyed on the rendered prompt itself: an agent whose inputs a
    perturbation did not reach builds a byte-identical prompt and gets its
    original completion back, and an agent downstream of the edit builds a
    different prompt, misses, and is actually asked. An unperturbed replay is
    therefore a pure cache hit and reproduces the original exactly -- which is
    also the cheapest possible regression test that replay is faithful.
    """

    def __init__(self, generator, cache):
        self.inner = generator
        self.cache = dict(cache)
        self.max_model_len = getattr(generator, "max_model_len", DEFAULT_MAX_MODEL_LEN)
        self.settings = getattr(generator, "settings", {})
        self.hits = 0
        self.misses = 0

    def render(self, messages):
        return self.inner.render(messages)

    def count_tokens(self, prompt):
        return self.inner.count_tokens(prompt)

    def generate(self, prompts, max_new_tokens=512, tags=None):
        tags = tags or [(None, None)] * len(prompts)
        outputs = [None] * len(prompts)
        pending = []
        for index, prompt in enumerate(prompts):
            if prompt in self.cache:
                outputs[index] = dict(self.cache[prompt])
                outputs[index]["cached"] = True
                self.hits += 1
            else:
                pending.append(index)
        if pending:
            fresh = self.inner.generate([prompts[i] for i in pending], max_new_tokens,
                                        [tags[i] for i in pending])
            for index, output in zip(pending, fresh):
                outputs[index] = {**output, "cached": False}
                self.misses += 1
        return outputs
