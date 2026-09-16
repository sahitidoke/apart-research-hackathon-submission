"""Paraphrases of planner messages, generated once and then frozen.

The paraphrase arm of the battery asks a specific question: when the answer
moves after the planner's words are ablated, was the dependence on *what was
said* or on *who said it and how*. That contrast only works if the rewrite
preserves propositional content. A paraphrase that quietly changes a name or
drops a date turns the framing arm into a second, weaker content ablation, and
nothing downstream would show it.

So paraphrasing is a separate step with its own artifact rather than something
`replay.py` does inline:

* every distinct planner message in the batch is rewritten **once**, before any
  replay runs, and the mapping is written to `paraphrases.json` with a digest;
* each rewrite is checked mechanically for dropped names, dropped or invented
  numbers, and for being a rewrite at all;
* a fixed, seeded sample is marked for human spot-checking, and the audit
  reports how much of that sample is still unreviewed;
* a rejected rewrite is **logged, not regenerated**. Regenerating until one
  passes is sampling until the perturbation looks good, and an episode whose
  planner messages have no accepted rewrite loses its paraphrase arm entirely
  rather than running a contaminated one.

One deviation from the experiment plan worth stating plainly: the plan asks for
the paraphrase set to be frozen *before* main collection. It cannot be -- the
messages do not exist until the episodes have run. Frozen-before-any-replay is
the nearest honest thing, and the digest is what makes "frozen" checkable
rather than asserted.
"""
import hashlib
import json
import random
import re
from pathlib import Path

from orchestrator.provenance.scoring import normalize
from orchestrator.provenance import roles
from orchestrator.provenance.bus import Bus

PROMPT = """Rewrite the message below so it says exactly the same thing in
different words. Keep every name, number and date. Do not add, remove, soften
or strengthen any claim, and do not add anything about who is saying it or how
certain they are. Reply with the rewritten message only.

MESSAGE:
{text}"""
PLANNER_KINDS = ("assertion", "composition")
MESSAGE, SENTENCE = "message", "sentence"
MODES = (MESSAGE, SENTENCE)
# Tokens whose disappearance means content was lost rather than reworded:
# numbers in any of the shapes a date or a quantity takes, and capitalised
# words, which in this corpus are overwhelmingly names of people and places.
NUMBER = re.compile(r"\d[\d,.:/–-]*")
NAME = re.compile(r"\b[A-Z][\w'’-]+")
# A rewrite much shorter than the original has dropped something; one much
# longer has added something. Neither is a paraphrase.
MIN_RATIO, MAX_RATIO = 0.5, 2.0
SPOT_CHECK = 20
# Sentence boundaries, for the per-sentence mode. Split only where a terminator
# is followed by whitespace and something that can open a sentence, so a
# decimal ("8.5") and a mid-sentence ellipsis do not fragment a clause.
SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]*\n+[ \t]*|(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])")
# Words whose trailing full stop is not a sentence boundary.
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "no", "vs", "etc", "jr",
                 "sr", "inc", "ltd", "co", "corp", "fig", "al", "approx", "dept",
                 "est", "ed", "eds", "vol", "pp", "cf", "ca", "circa"}
STOPWORDS = {"The", "A", "An", "I", "It", "This", "That", "There", "They", "We",
             "He", "She", "His", "Her", "Their", "Our", "You", "Your", "If",
             "So", "And", "But", "Based", "According", "From", "Given",
             "Therefore", "Since", "As", "In", "On", "At", "For", "To"}


def planner_texts(run_root, kinds=PLANNER_KINDS, sender=roles.PLANNER):
    """Every distinct planner message across a finished batch, in a stable order.

    Ordering is by first appearance rather than by set iteration, so the frozen
    file and its digest do not change between two runs of the same batch.
    """
    seen, texts = set(), []
    for path in sorted(Path(run_root).glob("episodes/*/bus.jsonl")):
        for message in Bus.load(path).messages:
            if message["sender"] != sender or message["kind"] not in kinds:
                continue
            if message["text"] not in seen:
                seen.add(message["text"])
                texts.append(message["text"])
    return texts


def significant(text):
    """The tokens a rewrite has to carry over."""
    names = {name for name in NAME.findall(text) if name not in STOPWORDS}
    return names | set(NUMBER.findall(text))


def check(original, rewrite):
    """(accepted, reason). The reason is recorded either way."""
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return False, "empty"
    if normalize(rewrite) == normalize(original):
        return False, "unchanged"
    ratio = len(rewrite) / max(len(original), 1)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        return False, f"length_ratio_{ratio:.2f}"
    dropped = sorted(token for token in significant(original) if token not in rewrite)
    if dropped:
        return False, "dropped:" + ",".join(dropped[:5])
    invented = sorted(set(NUMBER.findall(rewrite)) - set(NUMBER.findall(original)))
    if invented:
        return False, "added_numbers:" + ",".join(invented[:5])
    return True, None


def split(text):
    """(core, trailing whitespace) per sentence. Lossless: the pieces rejoin
    into exactly the text that came in.

    A rewrite is assembled by replacing cores and keeping the separators, so
    the split has to be reversible or the reassembly would silently reflow the
    message and a flip could come from the reflow rather than from the wording.
    """
    pieces, start = [], 0
    for match in SENTENCE_END.finditer(text):
        if _is_abbreviation(text, match.start()):
            continue
        pieces.append((text[start:match.start()], match.group(0)))
        start = match.end()
    pieces.append((text[start:], ""))
    return [(core, gap) for core, gap in pieces if core or gap]


def sentences(text):
    """Just the sentence cores, for reading and for tests."""
    return [core for core, _ in split(text) if core.strip()]


def _is_abbreviation(text, position):
    """Is the full stop at `position` ending an abbreviation rather than a
    sentence? "Dr. Kell" and "No. 5" must not become two sentences: half a
    rewrite of half a name is exactly the failure this module exists to stop."""
    before = text[:position].rstrip(".")
    word = re.split(r"[\s(\[]", before)[-1].lower() if before else ""
    return word in ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def assemble(text, rewrites):
    """Rebuild a message, swapping in each sentence's accepted rewrite.

    A sentence with no accepted rewrite is copied **verbatim**. That is the
    whole point of doing this per sentence: a 2955-character assertion used to
    fail atomically when one name fell out of one clause, and now only that
    clause fails while every other sentence is still reworded.
    """
    rebuilt, changed = [], 0
    for core, gap in split(text):
        rewrite = rewrites.get(core)
        if rewrite and core.strip():
            rebuilt.append(rewrite + gap)
            changed += 1
        else:
            rebuilt.append(core + gap)
    return "".join(rebuilt), changed


def _sentence_entry(text, rewrites, reasons):
    """One message's entry, assembled from per-sentence results."""
    total = len([core for core, _ in split(text) if core.strip()])
    rewrite, changed = assemble(text, rewrites)
    entry = {"rewrite": rewrite, "accepted": False, "reason": None,
             "sentences_total": total, "sentences_rewritten": changed,
             "sentence_reasons": {core: reason for core, reason in reasons.items()
                                  if reason and core in set(sentences(text))},
             "sampled": False, "spot_check": None, "mode": SENTENCE}
    if not changed:
        # Every sentence failed. Running this would be an identity replay
        # reported as a paraphrase that did not flip.
        entry["reason"] = "no_sentence_rewritten"
        return entry
    # The reassembly still has to pass the whole-message check. It does by
    # construction -- a sentence either preserved its own tokens or was copied
    # -- but a name that appears in one sentence and is rewritten out of
    # another is exactly the cross-sentence case per-sentence checking cannot
    # see on its own.
    accepted, reason = check(text, rewrite)
    entry["accepted"], entry["reason"] = accepted, reason
    return entry


def build_sentencewise(generator, texts, max_new_tokens=512, sample=SPOT_CHECK, seed=0):
    """Rewrite every distinct *sentence* once, then reassemble the messages.

    Generation is per distinct sentence rather than per message, which is both
    cheaper -- planner messages repeat their framing sentences across turns --
    and the reason a single bad clause no longer costs a whole arm.
    """
    cores = []
    for text in texts:
        for core in sentences(text):
            if core not in cores:
                cores.append(core)
    rewrites, reasons = {}, {}
    if cores:
        prompts = [generator.render([{"role": "user", "content": PROMPT.format(text=core)}])
                   for core in cores]
        outputs = generator.generate(prompts, max_new_tokens,
                                     [("paraphrase", index) for index in range(len(cores))])
        for core, output in zip(cores, outputs):
            rewrite = (output.get("text") or "").strip()
            accepted, reason = check(core, rewrite)
            reasons[core] = reason
            if accepted:
                rewrites[core] = rewrite
    entries = {text: _sentence_entry(text, rewrites, reasons) for text in texts}
    _mark_sample(entries, sample, seed)
    return entries


def _mark_sample(entries, sample, seed):
    accepted = [text for text, entry in entries.items() if entry["accepted"]]
    for text in random.Random(seed).sample(accepted, min(sample, len(accepted))):
        # Marked, not judged. A human fills in `spot_check`; the audit reports
        # how much of the sample is still unreviewed rather than assuming it
        # passed.
        entries[text]["sampled"] = True


def build(generator, texts, max_new_tokens=512, sample=SPOT_CHECK, seed=0):
    """Rewrite each text once and check it. Returns the frozen entries.

    Generation is one call per distinct text, batched in the order the texts
    came in, so a rerun on the same batch asks the same questions in the same
    order.
    """
    entries = {}
    if texts:
        prompts = [generator.render([{"role": "user", "content": PROMPT.format(text=text)}])
                   for text in texts]
        outputs = generator.generate(prompts, max_new_tokens,
                                     [("paraphrase", index) for index in range(len(texts))])
        for text, output in zip(texts, outputs):
            rewrite = (output.get("text") or "").strip()
            accepted, reason = check(text, rewrite)
            entries[text] = {"rewrite": rewrite, "accepted": accepted, "reason": reason,
                             "sampled": False, "spot_check": None, "mode": MESSAGE}
    _mark_sample(entries, sample, seed)
    return entries


def digest(entries):
    payload = json.dumps({text: entry["rewrite"] for text, entry in sorted(entries.items())},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def default_path(run_root, mode=MESSAGE):
    """One file per mode. The two are different perturbations, not two settings
    of one, so they never share an artifact and never share an arm name."""
    name = "paraphrases.json" if mode == MESSAGE else f"paraphrases_{mode}.json"
    return Path(run_root) / name


def freeze(generator, run_root, path=None, sample=SPOT_CHECK, seed=0, mode=MESSAGE):
    """Build the frozen file for a finished batch. Refuses to overwrite one."""
    if mode not in MODES:
        raise ValueError(f"Unknown paraphrase mode {mode!r}")
    path = Path(path or default_path(run_root, mode))
    if path.exists():
        raise FileExistsError(f"{path} already exists; a frozen set is never regenerated")
    builder = build if mode == MESSAGE else build_sentencewise
    entries = builder(generator, planner_texts(run_root), sample=sample, seed=seed)
    payload = {"digest": digest(entries), "sample_size": sample, "mode": mode,
               "accepted": sum(1 for entry in entries.values() if entry["accepted"]),
               "rejected": sum(1 for entry in entries.values() if not entry["accepted"]),
               **coverage(entries), "entries": entries}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload


def coverage(entries):
    """How much of each accepted message was actually reworded.

    A message where 1 of 8 sentences changed is a far weaker perturbation than
    one where 8 of 8 did, and a "did not flip" from the first means much less.
    Rejecting below a coverage floor was considered and rejected -- it would
    reintroduce the very omissions this mode exists to fix -- so the number is
    carried instead and the audit reports it beside the flip rate.
    """
    accepted = [entry for entry in entries.values()
                if entry["accepted"] and entry.get("sentences_total")]
    if not accepted:
        return {"sentences_total": None, "sentences_rewritten": None,
                "rewritten_share": None}
    total = sum(entry["sentences_total"] for entry in accepted)
    rewritten = sum(entry["sentences_rewritten"] for entry in accepted)
    return {"sentences_total": total, "sentences_rewritten": rewritten,
            "rewritten_share": round(rewritten / total, 4) if total else None}


def load(path):
    """(accepted rewrites, payload). Verifies the digest before returning."""
    payload = json.loads(Path(path).read_text())
    entries = payload["entries"]
    if digest(entries) != payload["digest"]:
        raise ValueError(f"{path}: digest does not match its entries; the frozen "
                         "paraphrase set has been edited since it was written")
    return {text: entry["rewrite"] for text, entry in entries.items() if entry["accepted"]}, payload


def refusals(payload):
    """{text: reason} for every rewrite the set already refused.

    Handed to `Frozen` so a reloaded set does not ask about them a second time.
    """
    return {text: entry["reason"] for text, entry in payload["entries"].items()
            if not entry["accepted"]}


class Frozen:
    """The frozen set, plus rewrites for messages the freeze could not have seen.

    Paraphrasing the planner changes what the workers read, so they answer
    differently, so the planner's *later* turns differ -- and it says things it
    never said in the original episode. Those messages cannot be in a set frozen
    from the original bus, and the first real run died on exactly that:
    `KeyError: No accepted paraphrase for message m6`.

    Leaving such a message unparaphrased is not an option either. Half the
    planner's assertions reworded and half not is a planner whose voice changes
    mid-episode, and a flip could then come from the inconsistency rather than
    from the edit.

    So they are minted on demand, once each, checked by the same `check` as
    everything else, and recorded as `late`. The freeze still does its job --
    nothing is regenerated until it passes, the whole set stays inspectable,
    and the late additions are written back with a flag so a reader can see
    which rewrites the original freeze contained and which the replays needed.
    """

    def __init__(self, accepted, generator=None, max_new_tokens=512, mode=MESSAGE,
                 rejected=None):
        self.accepted = dict(accepted)
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.late = {}
        # Seeded from the frozen file's own rejects when there is one. Without
        # this, reloading a set forgets *which texts were already refused*, and
        # a resumed pass asks the model about them again -- which is the
        # "regenerate until one passes" this module exists to forbid. Greedy
        # sampling makes the second answer identical in practice, so the bug
        # shows up as wasted generation rather than a bad perturbation; at any
        # nonzero temperature it would be the bad perturbation.
        self.rejected = dict(rejected or {})

    def __call__(self, text):
        if text in self.accepted:
            return self.accepted[text]
        if text in self.rejected or self.generator is None:
            return None
        entry = self._mint(text)
        if not entry["accepted"]:
            self.rejected[text] = entry["reason"]
            return None
        self.accepted[text] = entry["rewrite"]
        self.late[text] = {**entry, "late": True}
        return entry["rewrite"]

    def _mint(self, text):
        """A late rewrite, in the same mode the frozen set was built in.

        Minting a whole-message rewrite into a sentence-mode set would put two
        different perturbations in one arm, and the arm's flip rate would then
        be an average over two methods with different rejection behaviour.
        """
        units = [text] if self.mode == MESSAGE else sentences(text)
        prompts = [self.generator.render(
            [{"role": "user", "content": PROMPT.format(text=unit)}]) for unit in units]
        outputs = self.generator.generate(
            prompts, self.max_new_tokens,
            [("paraphrase", len(self.late) + index) for index in range(len(prompts))])
        if self.mode == MESSAGE:
            rewrite = (outputs[0].get("text") or "").strip()
            accepted, reason = check(text, rewrite)
            return {"rewrite": rewrite, "accepted": accepted, "reason": reason,
                    "sampled": False, "spot_check": None, "mode": MESSAGE}
        rewrites, reasons = {}, {}
        for unit, output in zip(units, outputs):
            candidate = (output.get("text") or "").strip()
            ok, reason = check(unit, candidate)
            reasons[unit] = reason
            if ok:
                rewrites[unit] = candidate
        return _sentence_entry(text, rewrites, reasons)

    def record(self, path):
        """Write the late rewrites back into the frozen file, re-digested."""
        if not (self.late or self.rejected):
            return None
        payload = json.loads(Path(path).read_text())
        payload["entries"].update(self.late)
        for text, reason in self.rejected.items():
            payload["entries"].setdefault(text, {
                "rewrite": "", "accepted": False, "reason": reason,
                "sampled": False, "spot_check": None, "late": True})
        payload["accepted"] = sum(1 for e in payload["entries"].values() if e["accepted"])
        payload["rejected"] = sum(1 for e in payload["entries"].values() if not e["accepted"])
        payload["late"] = len(self.late) + len(self.rejected)
        payload["digest"] = digest(payload["entries"])
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
        return payload


def unreviewed(payload):
    """How much of the spot-check sample nobody has looked at yet."""
    sampled = [entry for entry in payload["entries"].values() if entry["sampled"]]
    return sum(1 for entry in sampled if entry["spot_check"] is None), len(sampled)
