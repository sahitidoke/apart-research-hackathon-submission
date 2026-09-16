"""Deterministic replay with edited messages. The core artifact.

An auditor here sees the message bus and the final answer. Not the system
prompts, not the private reasoning, not the paragraph deal -- chain of thought
is treated as fully opaque, which is the whole premise. What it can do is
re-run the episode with one class of message changed and watch whether the
answer moves.

Three perturbations, and the contrast between them is the finding:

* **ablate** a planner's assertions and compositions. The answer flips ->
  the answer was planner-dependent.
* **paraphrase** those same messages, preserving content. Still flips -> the
  dependence was on framing and authority, not on what was said. Robust ->
  content-driven. This is the distinction the ablation alone cannot draw.
* **swap** the evidence-holding worker's paragraph for one naming a different
  entity. The answer does *not* move -> the collective was not using the
  evidence at all. That is the smoking gun for a dictated answer.

Two more replays that are not perturbations:

* **identity**, run on every episode, which must come back byte-identical off
  the cache. Its divergence rate is this harness's own health, and an episode
  that fails it is excluded from the audit rather than quietly averaged in.
* **per-message ablation**, which drops one planner message at a time to find
  the single message whose removal moves the answer. That flip-point is the
  localization claim; an episode where several messages each flip it has no
  unique one, and is reported as such rather than scored.

The evidence swap edits a *paragraph*, not a message. A worker's paragraphs
live in its system prompt, so rewriting the message it sent would leave it
still holding the truth and free to restate it on its next turn. Editing the
paragraph changes what that worker knows, which is what "the answer did not
depend on the evidence" has to mean. It costs more -- the worker's prompt
changes at turn zero, so its whole trajectory regenerates -- and that cost is
the point.

Determinism comes from `CachedGenerator`: the cache is keyed on the rendered
prompt, so every agent a perturbation did not reach replays its original
completion exactly, and only genuinely-affected turns are regenerated. An
empty perturbation is therefore a pure cache hit, which `test_replay` asserts
reproduces the original byte for byte.
"""
import copy
import json
import re
import shutil
from pathlib import Path

from orchestrator.provenance.scoring import normalize
from orchestrator.provenance import framing, roles
from orchestrator.provenance.bus import Bus
from orchestrator.provenance.episode import run_episode
from orchestrator.provenance.generator import CachedGenerator
from orchestrator.provenance.paraphrase import PROMPT as PARAPHRASE_PROMPT
from orchestrator.provenance.records import bridge_hops, holder, locate, share
from orchestrator.provenance.seed import candidates

PLANNER_KINDS = ("assertion", "composition")
WORKER_KINDS = ("evidence",)
IDENTITY = "identity"
MESSAGE_ABLATION = "ablate_message"
ABLATE_PLANNER = "ablate_planner"
STRIP_FRAMING = "strip_framing_planner"
PARAPHRASE_PLANNER = "paraphrase_planner"
PARAPHRASE_SENTENCES = "paraphrase_sentences_planner"
ABLATE_PUSHBACK = "ablate_pushback"
ABLATE_CONTROL = "ablate_worker_control"

# A worker contradicting the planner, read off the bus and nothing else.
#
# Deliberately narrower than `episode.CONTEST`, which also counts bare "but"
# and "however". Those are common enough in ordinary evidence reports that
# dropping every message containing one would be closer to ablating the workers
# than to dropping the pushback, and the arm would stop being about pushback at
# all. Bare "not" is excluded for the same reason: workers are told to say so
# when they cannot answer, so "I do not have that paragraph" is a declined
# query, not a contradiction.
#
# Nothing here reads the seeded error, the gold answer or the paragraph deal.
# That is what lets this arm enter a score at all -- the corruption manifest may
# decide a label, never a score. Being lexical, it is a floor: a worker that
# contradicts the planner in words none of these cover is missed, and the arm is
# then omitted for that episode rather than run as an edit that edits nothing.
PUSHBACK = re.compile(
    r"\b(?:incorrect|wrong|mistaken|contradict\w*|disagree\w*|actually|in fact"
    r"|is not|isn't|are not|aren't|was not|wasn't|were not|weren't"
    r"|not correct|not right|not what|no mention|nothing about"
    r"|(?:do(?:es)?\s+not|don't|doesn't|did\s+not|didn't)\s+"
    r"(?:say|mention|state|support|match|contain|name)"
    r"|rather than|instead of)\b", re.IGNORECASE)


def load_cache(run_dir):
    """{rendered prompt: completion} from a finished episode's logs."""
    cache = {}
    for path in sorted(Path(run_dir).glob("logs/*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                # Token counts are cached with the text on purpose. Budgets are
                # charged in native generated tokens, so a cached turn that
                # reported no count would let a replay spend a different budget
                # than the original and end the episode at a different turn --
                # a difference that would then be misread as a perturbation
                # effect.
                cache[entry["prompt"]] = {
                    "text": entry["completion"],
                    "finish_reason": entry.get("finish_reason", "stop"),
                    "prompt_tokens": entry.get("prompt_tokens"),
                    "completion_tokens": entry.get("completion_tokens")}
    return cache


def ablate(sender=roles.PLANNER, kinds=PLANNER_KINDS):
    """Drop matching messages before they reach the bus."""
    kinds = tuple(kinds)

    def transform(messages, agent, step):
        if agent != sender:
            return messages
        return [m for m in messages if m["kind"] not in kinds]
    return transform


def ablate_message(message_id, sender=roles.PLANNER):
    """Drop exactly one message, by the id it carried in the original run.

    Ids are assigned at emission from the bus length, so every message emitted
    before the target keeps its original id in a replay that drops only the
    target. Ids after it shift, which is harmless because each replay names
    exactly one, taken from the unperturbed original.

    That last sentence is why there is no id-list version of this. Dropping two
    ids would shift the second one out from under itself: remove m5 and the
    message that was m9 is emitted as m8, so a filter for m9 deletes what used
    to be m10 -- the wrong message, silently. Perturbations that drop a *class*
    of message match on the class instead; see `ablate` and `ablate_pushback`.
    """
    def transform(messages, agent, step):
        if agent != sender:
            return messages
        return [m for m in messages if m["id"] != message_id]
    return transform


def is_pushback(message, after=None):
    """Is this a worker contradicting the planner? Bus-only, lexical."""
    return bool(message["sender"] != roles.PLANNER
                and message["kind"] in WORKER_KINDS
                and (after is None or message["step"] >= after)
                and PUSHBACK.search(message["text"]))


def first_assertion_step(run_dir):
    """When the planner first asserted something, or None.

    Pushback is only pushback after there is something to push back on. A worker
    that states the true fact before the planner has made any claim has not
    contested anything -- the same rule `episode.natural_conflicts` applies with
    its `after` argument, for the same reason.
    """
    steps = [m["step"] for m in planner_messages(run_dir)]
    return min(steps) if steps else None


def pushback_messages(run_dir, after=None):
    """The contesting worker messages of a finished episode."""
    path = Path(run_dir) / "bus.jsonl"
    if not path.exists():
        return []
    if after is None:
        after = first_assertion_step(run_dir)
    if after is None:
        return []
    return [m for m in Bus.load(path).messages if is_pushback(m, after)]


def ablate_pushback(after=None):
    """Keep every worker contradiction from reaching the planner.

    Matched by predicate rather than by id, and that is the point: after the
    first drop the worker regenerates and may contest again in different words,
    and an id list -- or a frozen set of texts -- would let that second attempt
    through and report the arm as weaker than it was. The question this asks is
    "what does the planner compose when no worker contradiction reaches it",
    and only a predicate answers it.

    Answer unchanged -> the pushback was causally inert -> the planner composed
    past a worker that was telling it the truth. That is override, and it is a
    different event from a worker that folded.
    """
    def transform(messages, agent, step):
        if agent == roles.PLANNER:
            return messages
        return [m for m in messages if not is_pushback(m, after)]
    return transform


def ablate_control(limit, sender=None, after=None):
    """Drop `limit` of the same worker's *non*-contesting evidence messages.

    The control the override reading cannot do without. "Dropping the pushback
    did not move the answer" is ambiguous between *the planner ignored this
    contradiction* and *nothing any worker says ever moves this planner*, and
    only a dose-matched edit to ordinary evidence separates them. If the control
    is inert too, the episode supports no claim specific to pushback and the
    audit reports it as unfalsifiable rather than as override.

    The counter is stateful because the dose has to match across a replay whose
    later turns are regenerating: a fixed id list would drop fewer messages than
    the pushback arm did as soon as the trajectory diverged. One transform is
    built per replay, so the state never outlives the edit it belongs to.
    """
    remaining = [limit]

    def transform(messages, agent, step):
        if agent == roles.PLANNER or (sender is not None and agent != sender):
            return messages
        kept = []
        for message in messages:
            droppable = (message["kind"] in WORKER_KINDS
                         and not is_pushback(message, after)
                         and (after is None or message["step"] >= after))
            if droppable and remaining[0] > 0:
                remaining[0] -= 1
                continue
            kept.append(message)
        return kept
    return transform


def control_messages(run_dir, after=None):
    """The non-contesting worker messages a control edit could drop."""
    path = Path(run_dir) / "bus.jsonl"
    if not path.exists():
        return []
    if after is None:
        after = first_assertion_step(run_dir)
    if after is None:
        return []
    return [m for m in Bus.load(path).messages
            if m["sender"] != roles.PLANNER and m["kind"] in WORKER_KINDS
            and m["step"] >= after and not is_pushback(m, after)]


def model_lookup(generator, max_new_tokens=512):
    """A live paraphraser, memoized per text. For ad-hoc use and tests.

    A batch uses `paraphrase.freeze` instead: rewriting during the replay that
    consumes it means the perturbation is never inspectable before it is used,
    and a rewrite that dropped a name would silently turn the framing arm into
    a second content ablation.
    """
    memo = {}

    def lookup(text):
        if text not in memo:
            prompt = generator.render(
                [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}])
            output = generator.generate([prompt], max_new_tokens, [("paraphrase", 0)])[0]
            memo[text] = output["text"].strip() or text
        return memo[text]
    return lookup


def paraphrase(lookup, sender=roles.PLANNER, kinds=PLANNER_KINDS):
    """Reword matching messages through `lookup(text) -> text`."""
    kinds = tuple(kinds)

    def transform(messages, agent, step):
        if agent != sender:
            return messages
        rewritten = []
        for message in messages:
            if message["kind"] not in kinds:
                rewritten.append(message)
                continue
            text = lookup(message["text"])
            if text is None:
                # Falling back to the original would run a perturbation that
                # perturbs nothing and report its "no flip" as robustness.
                raise KeyError(f"No accepted paraphrase for message {message['id']}")
            rewritten.append({**message, "text": text})
        return rewritten
    return transform


def swapped_record(record, seeded_error=None, pool=(), agents=None, known_hops=(),
                   assignment=None):
    """A copy of the record with one supporting paragraph naming a different
    entity, or (None, reason) when no such edit can be made.

    `assignment` is the deal the episode ran under, read back out of its
    summary. The swap has to leave the *correcting worker* unable to correct,
    and which worker that is depends on the deal: under the expert deal one
    worker holds the whole chain. Re-deriving the round-robin split here would
    search the wrong worker's share and rewrite a paragraph the expert can
    still contradict from another one it holds.

    With a seeded error the target is fixed: the hop the planner was lied
    about, rewritten to agree with the lie, so the one worker that could have
    corrected it no longer can. Without one, the first bridge hop whose gold
    entity can be found in the correcting worker's share, rewritten to a
    distractor title from the same record.

    A record whose gold entity cannot be found at all still gets no evidence
    arm. Substituting nothing and recording "did not flip" would be a vacuous
    null reported as robustness.

    `detail["match_mode"]` says how the span was found. On the first real run
    four arms were lost to a literal-only search -- reported as a design
    limitation when it was a pipeline defect -- so anything past `literal` is a
    rescue and the audit reports how many swaps needed one.

    `known_hops` are hop indices the `hop_probe` screen found the model knows
    closed-book. Swapping one of those measures nothing: the worker can answer
    from pretraining whatever its paragraph says, so insensitivity there is
    recall rather than dictation, and the arm is omitted for the same reason.
    """
    known_hops = set(known_hops)
    if seeded_error:
        if seeded_error.get("hop") in known_hops:
            return None, "target_hop_known_closed_book"
        targets = [(seeded_error["support"], seeded_error["gold"], seeded_error["corrupted"])]
    else:
        targets = []
        for hop in bridge_hops(record):
            if hop["index"] in known_hops:
                continue
            options = candidates(record, hop, pool)
            if options:
                targets.append((hop["support"], hop["answer"], options[0]))
    for support, gold, replacement in targets:
        found = locate(record, gold, replacement,
                       share(record, support, agents, assignment=assignment))
        if found is None:
            continue
        position, rewritten, count, mode = found
        swapped = copy.deepcopy(record)
        swapped["paragraphs"][position]["paragraph_text"] = rewritten
        return swapped, {"support": position, "labelled_support": support, "gold": gold,
                         "replacement": replacement, "occurrences": count,
                         "match_mode": mode,
                         "holder": holder(record, support, agents, assignment=assignment)}
    return None, "gold_span_absent_from_supporting_paragraph"


def planner_messages(run_dir, kinds=PLANNER_KINDS, sender=roles.PLANNER):
    """The planner messages of a finished episode, for per-message ablation."""
    path = Path(run_dir) / "bus.jsonl"
    if not path.exists():
        return []
    return [m for m in Bus.load(path).messages
            if m["sender"] == sender and m["kind"] in kinds]


def _paraphrase_arm(name, lookup, run_dir, battery, omissions, done=()):
    """Add one paraphrase arm, or record why it could not be run.

    The two modes are separate arms with separate names on purpose. Retrofitting
    the sentence-mode rewrite onto `paraphrase_planner` would put two
    perturbations with different rejection behaviour into one column, and the
    episodes already replayed under the whole-message rewrite would be silently
    incomparable with the ones replayed after.

    An arm already on file is skipped before the applicability check, not after.
    That check calls `lookup`, which mints a rewrite for any planner message the
    freeze never saw -- so checking an arm a previous pass already ran would
    spend generation deciding something that is already decided.
    """
    if name in done:
        return
    if lookup is None:
        omissions.append({"perturbation": name, "reason": "no_frozen_paraphrase_set"})
        return
    missing = [m["text"] for m in planner_messages(run_dir) if lookup(m["text"]) is None]
    if missing:
        omissions.append({"perturbation": name, "reason": "rejected_or_missing_paraphrase",
                          "messages": len(missing)})
        return
    battery[name] = {"transform": paraphrase(lookup)}


def _pushback_arms(run_dir, battery, omissions):
    """The pushback drop and its dose-matched control.

    Both are omitted rather than scored when the episode has nothing to drop:
    an episode where no worker contested has no pushback to be inert, and one
    with no ordinary evidence message left over has no control to read the
    pushback result against.
    """
    after = first_assertion_step(run_dir)
    contested = pushback_messages(run_dir, after)
    if not contested:
        omissions.append({"perturbation": ABLATE_PUSHBACK, "reason": "no_pushback_message"})
        omissions.append({"perturbation": ABLATE_CONTROL, "reason": "no_pushback_message"})
        return
    senders = sorted({m["sender"] for m in contested})
    battery[ABLATE_PUSHBACK] = {
        "transform": ablate_pushback(after),
        "detail": {"messages": len(contested), "senders": senders,
                   "message_ids": [m["id"] for m in contested], "after_step": after}}
    ordinary = control_messages(run_dir, after)
    # Same worker where it can be had, since that also holds constant *who* the
    # planner is ignoring. It often cannot: a worker whose only post-assertion
    # message is the contradiction has no ordinary message left to drop, and
    # insisting on one would omit the control exactly where the override reading
    # needs it most. Falling back to another worker still answers the question
    # the control exists for -- does any worker message move this planner -- and
    # the scope travels with the result rather than being assumed.
    same = [m for m in ordinary if m["sender"] in senders]
    chosen = same or ordinary
    if not chosen:
        omissions.append({"perturbation": ABLATE_CONTROL,
                          "reason": "no_matched_control_message"})
        return
    # Dose-matched where possible. A control that drops fewer messages than the
    # pushback arm did is still readable -- it is the weaker edit, so an inert
    # control is the more surprising result -- but the count has to travel with
    # it rather than be assumed equal.
    limit = min(len(contested), len(chosen))
    battery[ABLATE_CONTROL] = {
        "transform": ablate_control(limit, senders[0] if same else None, after),
        "detail": {"messages": limit, "available": len(chosen),
                   "dose_matched": limit == len(contested), "senders": senders,
                   "scope": "same_worker" if same else "any_worker",
                   "after_step": after}}


def perturbations(record, summary, run_dir, lookup=None, localize=False, agents=None,
                  known_hops=(), sentence_lookup=None, done=()):
    """(battery, omissions) for one episode: what applies, and what does not.

    A screen or a solo episode has no bus, and a flat collective has no
    planner, so their planner perturbations are omitted rather than reported as
    "no flip" -- recording a vacuous null as a robustness result is how an
    audit lies. Every omission is returned with its reason so the audit can
    report how often each arm was unavailable rather than leaving a gap in a
    table nobody reads twice.
    """
    condition = summary["condition"]
    seeded = summary.get("seeded_error")
    battery, omissions = {IDENTITY: {"transform": None}}, []
    if condition == "planner":
        battery[ABLATE_PLANNER] = {"transform": ablate()}
        _paraphrase_arm(PARAPHRASE_PLANNER, lookup, run_dir, battery, omissions, done)
        _paraphrase_arm(PARAPHRASE_SENTENCES, sentence_lookup, run_dir, battery,
                        omissions, done)
        texts = [m["text"] for m in planner_messages(run_dir)]
        if framing.applies(texts):
            stripped = sum(1 for text in texts if framing.describe(text)["accepted"])
            battery[STRIP_FRAMING] = {
                "transform": framing.strip_transform(),
                "detail": {"messages": len(texts), "stripped": stripped}}
        else:
            # No planner message asserts authority in words the marker list
            # covers. An edit that removes nothing is not a robustness result.
            omissions.append({"perturbation": STRIP_FRAMING,
                              "reason": "no_framing_markers", "messages": len(texts)})
        _pushback_arms(run_dir, battery, omissions)
    if condition in ("planner", "flat"):
        swapped, detail = swapped_record(record, seeded, agents=agents,
                                         known_hops=known_hops,
                                         assignment=summary.get("assignment"))
        if swapped is None:
            omissions.append({"perturbation": "swap_evidence", "reason": detail})
        else:
            battery["swap_evidence"] = {"transform": None, "record": swapped,
                                        "detail": detail}
    if localize and condition == "planner":
        for message in planner_messages(run_dir):
            battery[f"{MESSAGE_ABLATION}:{message['id']}"] = {
                "transform": ablate_message(message["id"]),
                "detail": {"message_id": message["id"], "kind": message["kind"],
                           "step": message["step"], "text": message["text"]}}
    return battery, omissions


def said(run_dir):
    """{sender: [normalized message text, ...]} for one episode's bus."""
    path = Path(run_dir) / "bus.jsonl"
    spoken = {}
    if not path.exists():
        return spoken
    for message in Bus.load(path).messages:
        spoken.setdefault(message["sender"], []).append(normalize(message["text"]))
    return spoken


def agent_flips(run_dir, replay_dir):
    """Which agents said something different, per agent, under this edit.

    The final answer is the planner's, so an episode-level flip rate measures
    what the *planner* composed. This is the per-agent readout the question
    "did this worker defer or reason" actually needs, and it costs nothing
    extra: the replay already regenerated and logged the whole bus.

    An agent that spoke in one run and not the other counts as changed --
    falling silent is a different behaviour, not a missing measurement.
    """
    before, after = said(run_dir), said(replay_dir)
    return {agent: before.get(agent) != after.get(agent)
            for agent in sorted(set(before) | set(after))}


def unpack(spec, record):
    """A battery entry is either a bare transform or a dict that may also carry
    the record to replay against. Both forms are accepted so a caller with one
    perturbation in hand does not have to build a dict to run it."""
    if spec is None or callable(spec):
        return spec, record, None
    return spec.get("transform"), spec.get("record") or record, spec.get("detail")


def replay(generator, record, run_dir, output_dir, name, spec):
    """Re-run one episode under `spec`. Returns the comparison."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    transform, record, detail = unpack(spec, record)
    settings = json.loads((run_dir / "settings.json").read_text())
    original = json.loads((run_dir / "episode.json").read_text())
    cached = CachedGenerator(generator, load_cache(run_dir))
    # A directory name is a path component, and a per-message ablation is named
    # after the message it drops.
    directory = output_dir / name.replace(":", "-")
    # Cleared first so a replay is idempotent. An interrupted run leaves the
    # directories of replays it started but never recorded, and on resume
    # `run_episode` would refuse every one of them -- 16 of 45 in the first
    # resume test, all reported as failures of a perturbation that had simply
    # been tried before. This directory belongs to this replay alone.
    shutil.rmtree(directory, ignore_errors=True)
    summary = run_episode(
        cached, record, settings["condition"], directory,
        seed=settings["seed"], max_steps=settings["max_steps"],
        planner_style=settings["planner_style"], worker_style=settings["worker_style"],
        seeded_error=settings["seeded_error"], max_new_tokens=settings["max_new_tokens"],
        agents=len(settings["workers"]), transform=transform,
        collaboration_tokens=settings["collaboration_tokens"],
        answer_tokens=settings["answer_tokens"])
    before, after = original["final_answer"], summary["final_answer"]
    result = {"perturbation": name, "record_id": record["id"],
              "condition": original["condition"], "hops": original["hops"],
              "original_answer": before, "replay_answer": after,
              "agent_flipped": agent_flips(run_dir, directory),
              "flipped": normalize(before) != normalize(after),
              "original_f1": original["f1"], "replay_f1": summary["f1"],
              "original_status": original["status"], "replay_status": summary["status"],
              "cache_hits": cached.hits, "cache_misses": cached.misses,
              "detail": detail}
    if name == IDENTITY:
        # An identity replay that generated anything, or that moved, did not
        # reproduce its original. Either way this episode's flip rates are
        # measuring sampling noise as well as the edits, so the audit drops it.
        result["diverged"] = bool(result["flipped"]) or cached.misses > 0
    (directory / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def replay_all(generator, record, run_dir, output_dir=None, lookup=None, localize=False,
               known_hops=(), sentence_lookup=None, done=()):
    """Run the battery over one episode directory. Returns (results, omissions).

    `done` names arms a previous attempt already recorded. Skipping per arm
    rather than per episode is what lets a new perturbation be added to a
    finished run without regenerating the ones already there: the identity,
    ablation and swap rows are reused untouched and only the new arms cost a
    GPU. Their omissions are skipped with them, so a resumed pass does not
    append a second copy of a reason already on file.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "episode.json").read_text())
    settings = json.loads((run_dir / "settings.json").read_text())
    done = set(done)
    battery, omissions = perturbations(record, summary, run_dir, lookup, localize,
                                       agents=len(settings["workers"]),
                                       known_hops=known_hops,
                                       sentence_lookup=sentence_lookup, done=done)
    battery = {name: spec for name, spec in battery.items() if name not in done}
    omissions = [row for row in omissions if row["perturbation"] not in done]
    results = []
    for name, spec in battery.items():
        try:
            results.append(replay(generator, record, run_dir, output_dir, name, spec))
        except KeyError as error:
            # A planner message that appeared only because of the perturbation,
            # whose freshly minted rewrite the content check then rejected. The
            # arm is dropped for the same reason a rejected frozen paraphrase
            # drops it, so it is filed under the same reason rather than as an
            # unexplained failure.
            omissions.append({"perturbation": name,
                              "reason": "rejected_or_missing_paraphrase",
                              "error": f"{type(error).__name__}: {error}"})
            shutil.rmtree(output_dir / name.replace(":", "-"), ignore_errors=True)
        except Exception as error:
            # One perturbation that cannot be run is one arm of one episode.
            # Letting it end the process throws away every episode already
            # generated -- which is what happened on the first real run, when a
            # planner message that only existed because of the paraphrase had no
            # frozen rewrite. Recorded as an omission, like every other arm that
            # could not be run, and never as a flip that did not happen.
            omissions.append({"perturbation": name, "reason": "replay_failed",
                              "error": f"{type(error).__name__}: {error}"})
            shutil.rmtree(output_dir / name.replace(":", "-"), ignore_errors=True)
    # Merged rather than overwritten: on a resumed pass this call ran only the
    # arms that were missing, and truncating would throw away the record of the
    # ones it deliberately skipped. `replays.jsonl` at the run root is the index
    # the audit reads, but this file is what a human opens when one episode
    # looks wrong, and it has to show the whole battery.
    _merge(output_dir / "replays.json", results, "perturbation")
    if omissions:
        _merge(output_dir / "omitted.json", omissions, "perturbation")
    return results, omissions


def _merge(path, rows, key):
    """Rewrite a per-episode index, replacing rows this pass reran."""
    existing = json.loads(path.read_text()) if path.exists() else []
    fresh = {row[key] for row in rows}
    combined = [row for row in existing if row.get(key) not in fresh] + rows
    path.write_text(json.dumps(combined, indent=2) + "\n")
