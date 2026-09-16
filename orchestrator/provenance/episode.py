"""One question, one collective, one recorded episode.

The loop is lockstep, the same choice `marl/rollout.py` makes and for the same
reason: a worker needs whatever the planner has already asked, not wall-clock
simultaneity, and aligned step numbers are what let two runs of the same
episode be compared turn by turn.

Everything an agent sees is (its system prompt) + (messages the bus delivered
to it). There is no browser, no wiki and no shared file. That is a real
reduction from `marl/env.py`, and it buys the one property the audit needs:
editing the bus edits the agent's entire world, so a final answer that does
not move when the evidence moves was genuinely not using the evidence.
"""
import json
import re
from collections import Counter
from pathlib import Path

from orchestrator.provenance.scoring import answer_score, normalize
from orchestrator.provenance import roles
from orchestrator.provenance.bus import BROADCAST, Bus, parse, render
from orchestrator.provenance.records import (bridge_hops, deal, hop_count, holder,
                                             resolve_question,
                                             workers as worker_names)
from orchestrator.provenance.seed import candidates

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Three collective conditions and two screens. The screens are not comparisons
# -- nothing is ever read against `closed_book` accuracy -- they decide which
# questions may enter the deference analysis at all:
#
# * `closed_book` answers with no paragraphs. A question it gets right is one
#   the model already knew, and an answer that survives an evidence swap on
#   such a question is parametric recall, not dictation.
# * `hop_probe` asks each *bridge sub-question* with no paragraphs. The
#   question-level screen above is not enough on its own: a model can fail the
#   full multi-hop question and still know one intermediate fact outright, and
#   a worker that knows the corrupted hop from pretraining is insensitive to
#   its own paragraph for a reason that is neither deference nor reasoning
#   from evidence. Per-hop results gate the perturbation that targets that hop.
# * `isolated` gives each worker its own share and nobody to talk to. A
#   question one worker solves alone never needed the other, so a collective
#   answering it correctly proves nothing about routing evidence.
CONDITIONS = ("closed_book", "hop_probe", "isolated", "solo", "flat", "planner")
SCREENS = ("closed_book", "hop_probe", "isolated")
# Lexical markers for a worker contesting rather than restating. See `_followed`.
CONTEST = re.compile(r"\b(not|isn't|is not|no[t]?|actually|contradict\w*|disagree\w*|"
                     r"incorrect|wrong|mistaken|however|but|instead|rather)\b", re.IGNORECASE)
MAX_NEW_TOKENS = 512
MAX_STEPS = 8
# Per-episode native generated-token budgets, in two phases. Collaboration is
# shared by every agent across every turn; the answer phase is the planner's
# (or the collective's) final composition and has its own separate allowance,
# so exhausting the discussion can never leave an episode unable to answer.
# That is why there is no "reserve" carved out of a single pool the way
# `simulated_web/session.py` does it -- two independent budgets get the same
# guarantee without the bookkeeping.
COLLABORATION_TOKENS = 16000
ANSWER_TOKENS = 12000
COLLABORATION, ANSWERING = "collaboration", "answer"


class Budget:
    """Native generated tokens spent per phase, and what is left.

    Only generated tokens count. Prompt tokens are recorded for cost reporting
    but never charged: an agent does not choose how much history it is handed,
    and charging for it would make a four-worker episode pay for the same bus
    once per reader.
    """

    def __init__(self, collaboration=COLLABORATION_TOKENS, answer=ANSWER_TOKENS):
        self.caps = {COLLABORATION: collaboration, ANSWERING: answer}
        self.generated = {COLLABORATION: 0, ANSWERING: 0}
        self.prompt = {COLLABORATION: 0, ANSWERING: 0}

    def remaining(self, phase):
        return max(0, self.caps[phase] - self.generated[phase])

    def exhausted(self, phase):
        return self.remaining(phase) <= 0

    def allowance(self, phase, max_new_tokens):
        """What a single turn may generate: the per-response cap, or whatever
        is left of the phase if that is smaller."""
        return min(max_new_tokens, self.remaining(phase))

    def charge(self, phase, output):
        self.generated[phase] += int(output.get("completion_tokens") or 0)
        self.prompt[phase] += int(output.get("prompt_tokens") or 0)

    def usage(self):
        return {"generated": dict(self.generated), "prompt": dict(self.prompt),
                "caps": dict(self.caps),
                "generated_total": sum(self.generated.values()),
                "prompt_total": sum(self.prompt.values())}


def extract_answer(text):
    """The tagged span, or None. Deliberately not `scoring.extract_answer`:
    that one falls back to the whole completion, which is right when every
    episode has ended and something must be scored, and wrong here, where the
    absence of a tag is how an agent says it is still working."""
    matches = ANSWER.findall(text)
    return matches[-1].strip() if matches else None


class Conversation:
    """One agent's private history, and what it has been shown."""

    def __init__(self, agent, system):
        self.agent = agent
        self.messages = [{"role": "system", "content": system}]
        self.seen = set()
        self.turns = 0
        self.done = False
        self.status = "step_limit"
        self.answer = None
        self.votes = []

    def prompt(self, generator, user):
        self.messages.append({"role": "user", "content": user})
        return generator.render(self.messages)

    def record(self, text):
        self.messages.append({"role": "assistant", "content": text})
        self.turns += 1

    def finish(self, status, answer=None):
        self.done, self.status = True, status
        if answer is not None:
            self.answer = answer


def probe_hops(record):
    """Bridge hops, each with a sub-question that stands on its own.

    A "#1" back-reference is resolved to the answer it names rather than asked
    verbatim -- see `records.resolve_question`. A model failing "Who founded
    #1?" would look like ignorance of the fact when it was really ignorance of
    the reference, and telling those apart is the probe's whole job. A hop
    whose reference cannot be resolved is dropped, and the audit then treats
    that hop as unscreened rather than as screened and clean.
    """
    hops = []
    for hop in bridge_hops(record):
        question = resolve_question(record, hop.get("question", ""))
        if question:
            hops.append({**hop, "question": question})
    return hops


def _opening(record, condition):
    if condition == "planner":
        return (f"Question: {record['question']}\n\n"
                "You hold no paragraphs. Begin by asking the researchers for what you need.")
    return f"Question: {record['question']}"


def _turn(generator, bus, conversation, step, user, recipients, max_new_tokens, log,
          transform=None, budget=None, phase=COLLABORATION):
    """One agent's turn: render, generate, parse, publish. Returns its answer.

    `transform` is where a replay perturbation acts. It edits messages between
    the sender and the bus, not inside the sender: the sender keeps its own
    words in its own history and only the recipients see the edit. That is the
    right seam for this audit, which asks what the *channel* carried, and it
    is also what makes the untouched part of a replay free -- the sender's
    prompt is unchanged, so it comes back from the cache.
    """
    allowance = budget.allowance(phase, max_new_tokens) if budget else max_new_tokens
    if allowance <= 0:
        # Refuse rather than ask for zero tokens: a zero-length completion is
        # indistinguishable from a model that chose to say nothing, and the
        # two mean opposite things when reading a transcript.
        conversation.finish("budget_limit")
        return None
    prompt = conversation.prompt(generator, user)
    window = getattr(generator, "max_model_len", None)
    if window is not None and generator.count_tokens(prompt) + allowance > window:
        # End here rather than truncate, the same call `marl/rollout.py` makes:
        # every recorded prompt stays a prompt the model was really given, and
        # a silently shortened history would change what an agent knew without
        # any of it appearing in the transcript.
        conversation.finish("context_limit")
        return None
    output = generator.generate([prompt], allowance, [(conversation.agent, step)])[0]
    if budget:
        budget.charge(phase, output)
    text = output["text"]
    conversation.record(text)
    entry = {"step": step, "agent": conversation.agent, "phase": phase, "prompt": prompt,
             "completion": text, "finish_reason": output.get("finish_reason"),
             "cached": output.get("cached", False), "allowance": allowance,
             "prompt_tokens": output.get("prompt_tokens"),
             "completion_tokens": output.get("completion_tokens"), "emitted": []}
    log.append(entry)
    if output.get("finish_reason") not in (None, "stop"):
        # A completion cut off mid-block never had its messages published, so
        # publishing a prefix of them would put words in the agent's mouth.
        conversation.finish("length_limit")
        return None
    try:
        emitted = parse(text, conversation.agent, step, recipients, start_id=bus.next_id())
    except ValueError as error:
        entry["error"] = str(error)
        conversation.finish("malformed_message")
        return None
    if transform is not None:
        original = [dict(message) for message in emitted]
        emitted = transform(emitted, conversation.agent, step)
        if original != emitted:
            entry["perturbed"] = {"before": original, "after": emitted}
    bus.extend(emitted)
    entry["emitted"] = [message["id"] for message in emitted]
    conversation.votes.extend(m["text"] for m in emitted if m["kind"] == "vote")
    return extract_answer(text)


def _deliver(bus, conversation):
    messages = bus.delivered(conversation.agent, conversation.seen)
    conversation.seen.update(message["id"] for message in messages)
    return messages


def run_episode(generator, record, condition, run_dir, seed=0, max_steps=MAX_STEPS,
                planner_style="persuasive", worker_style="standard", seeded_error=None,
                max_new_tokens=MAX_NEW_TOKENS, agents=None, transform=None,
                collaboration_tokens=COLLABORATION_TOKENS, answer_tokens=ANSWER_TOKENS,
                assignment=None, expert=None, session=None):
    """Run one episode and write its artifacts. Returns the summary dict.

    `run_dir` must not already exist. Reusing one would mix a previous
    episode's bus into this one's logs, and the perturbation replays read
    those logs as ground truth about what was said.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()

    names = worker_names(record, agents)
    # An explicit assignment is how the expert deal reaches here. It is passed
    # in rather than chosen here so that a replay reads the same one out of
    # settings.json and cannot re-derive a different deal.
    assignment = assignment or deal(record, agents)
    budget = Budget(collaboration_tokens, answer_tokens)
    bus, log = Bus(), []
    conversations = _build(record, condition, names, assignment, planner_style,
                           worker_style, seeded_error)
    status = _run(generator, record, condition, bus, log, conversations, names,
                  max_steps, max_new_tokens, transform, budget)

    summary = _summarize(record, condition, names, assignment, conversations, bus,
                         status, seeded_error, planner_style, worker_style, seed, budget,
                         expert, session)
    settings = {"condition": condition, "record_id": record["id"], "seed": seed,
                "expert": expert, "session": session,
                "max_steps": max_steps, "max_new_tokens": max_new_tokens,
                "planner_style": planner_style, "worker_style": worker_style,
                "collaboration_tokens": collaboration_tokens,
                "answer_tokens": answer_tokens,
                "seeded_error": seeded_error, "workers": names,
                "assignment": assignment, "generator": getattr(generator, "settings", {}),
                "prompts": {name: c.messages[0]["content"] for name, c in conversations.items()}}
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    bus.dump(run_dir / "bus.jsonl")
    for name in conversations:
        with (run_dir / "logs" / f"{name}.jsonl").open("w") as handle:
            for entry in log:
                if entry["agent"] == name:
                    handle.write(json.dumps(entry) + "\n")
    (run_dir / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _build(record, condition, names, assignment, planner_style, worker_style, seeded_error):
    conversations = {}
    if condition == "closed_book":
        conversations["solo"] = Conversation("solo", roles.closed_book_prompt())
        return conversations
    if condition == "solo":
        positions = sorted(p for ps in assignment.values() for p in ps)
        conversations["solo"] = Conversation("solo", roles.solo_prompt(record, positions))
        return conversations
    if condition == "hop_probe":
        # One fresh conversation per bridge hop, never one conversation asked
        # several questions: a probe that let the model see its own previous
        # sub-answer would measure chaining, not recall of this hop.
        #
        # This is the one place a decomposition sub-question is shown to a
        # model, and it is safe only because the episode is sealed: it has no
        # bus, no peers, and its output is never read back into any collective
        # episode. The repository rule that decompositions must not reach an
        # agent still holds everywhere that an agent could act on one.
        for hop in probe_hops(record):
            name = f"hop-{hop['index']}"
            conversations[name] = Conversation(name, roles.hop_probe_prompt())
        return conversations
    if condition == "isolated":
        # Each worker gets the share it would hold in a collective and nobody
        # to send a message to. The prompt is `solo_prompt`, not `worker_prompt`:
        # a worker told a planner is coming would reasonably wait for one, and
        # the screen asks what it can do with its own paragraphs, not whether
        # it follows a protocol nobody is running.
        for name in names:
            conversations[name] = Conversation(name, roles.solo_prompt(record, assignment[name]))
        return conversations
    for name in names:
        peers = [other for other in names if other != name]
        if condition == "flat":
            system = roles.flat_prompt(record, name, assignment[name], peers)
        else:
            system = roles.worker_prompt(record, name, assignment[name], peers, worker_style)
        conversations[name] = Conversation(name, system)
    if condition == "planner":
        belief = None
        if seeded_error:
            belief = {"question": seeded_error["question"], "corrupted": seeded_error["corrupted"]}
        conversations[roles.PLANNER] = Conversation(
            roles.PLANNER, roles.planner_prompt(names, planner_style, belief))
    return conversations


def _run(generator, record, condition, bus, log, conversations, names, max_steps,
         max_new_tokens, transform=None, budget=None):
    opening = _opening(record, condition)
    if condition in ("solo", "closed_book"):
        return _run_solo(generator, opening, bus, log, conversations["solo"],
                         max_steps, max_new_tokens, transform, budget)
    if condition == "hop_probe":
        return _run_hop_probe(generator, record, bus, log, conversations,
                              max_steps, max_new_tokens, transform, budget)
    if condition == "isolated":
        return _run_isolated(generator, opening, bus, log, conversations, names,
                             max_steps, max_new_tokens, transform, budget)
    if condition == "flat":
        return _run_flat(generator, opening, bus, log, conversations, names,
                         max_steps, max_new_tokens, transform, budget)
    return _run_planner(generator, opening, bus, log, conversations, names,
                        max_steps, max_new_tokens, transform, budget)


def _run_solo(generator, opening, bus, log, conversation, max_steps, max_new_tokens,
              transform=None, budget=None):
    """Work for up to `max_steps` turns, then one forced answer.

    Deliberately the same shape as `_run_planner`, and that is the whole point.
    This used to press for the answer on every turn after the first, and the
    model obliged on turn one every time -- 1.0 turns against 5.5 for the flat
    pair and 11.0 for the planner-led collective. With thinking disabled the
    turns are the reasoning, so the "reference ceiling" was a one-shot baseline
    being compared against multi-turn ones, and then used to disqualify every
    question it failed.

    Charged to the answer phase throughout: there is nobody to collaborate
    with, so charging these to the collaboration budget would make the
    reference condition compete for a pool the thing it references does not use.
    """
    for step in range(max_steps):
        user = opening if step == 0 else roles.CONTINUE
        answer = _turn(generator, bus, conversation, step, user, set(), max_new_tokens,
                       log, transform, budget, ANSWERING)
        if conversation.done:
            return conversation.status
        if answer is not None:
            conversation.finish("complete", answer)
            return "complete"
    return _final_turn(generator, bus, log, conversation, max_steps, set(),
                       max_new_tokens, transform, budget,
                       "Give your final answer now.")


def _run_hop_probe(generator, record, bus, log, conversations, max_steps,
                   max_new_tokens, transform=None, budget=None):
    """Ask each bridge sub-question on its own, with no paragraphs.

    Charged to the answer phase for the same reason `_run_solo` is: this is a
    screen, and a screen that could outspend the condition it screens for
    would not be screening the same thing.
    """
    for hop in probe_hops(record):
        conversation = conversations[f"hop-{hop['index']}"]
        for step in range(max_steps):
            user = (f"Query: {hop['question']}" if step == 0
                    else "Give your final answer now.")
            answer = _turn(generator, bus, conversation, step, user, set(), max_new_tokens,
                           log, transform, budget, ANSWERING)
            if conversation.done:
                break
            if answer is not None:
                conversation.finish("complete", answer)
                break
    answered = [c for c in conversations.values() if c.answer is not None]
    if not conversations:
        # Every bridge hop was a back-reference, so there was nothing askable.
        # Not a failure, and not a pass either: `audit` reads the empty hop
        # list and gates nothing on it.
        return "no_probeable_hop"
    if len(answered) == len(conversations):
        return "complete"
    return "partial" if answered else next(iter(conversations.values())).status


def _run_isolated(generator, opening, bus, log, conversations, names, max_steps,
                  max_new_tokens, transform=None, budget=None):
    """Every worker answers alone, one after another, on the answer budget.

    They share the episode's budget rather than getting one each, for the same
    reason `_run_solo` charges to the answer phase: this is a screen, and a
    screen that could spend more than the condition it screens for would not be
    screening the same thing. Any worker answering correctly is the finding --
    it means the question never needed the pair.
    """
    for name in names:
        conversation = conversations[name]
        for step in range(max_steps):
            user = opening if step == 0 else roles.CONTINUE
            answer = _turn(generator, bus, conversation, step, user, set(), max_new_tokens,
                           log, transform, budget, ANSWERING)
            if conversation.done:
                break
            if answer is not None:
                conversation.finish("complete", answer)
                break
        else:
            # Same forced close as `_run_solo`: a worker that used every turn
            # without answering is asked once, so "could not do it alone" is a
            # measurement rather than a missing answer.
            _final_turn(generator, bus, log, conversation, max_steps, set(),
                        max_new_tokens, transform, budget,
                        "Give your final answer now.")
    answered = [conversations[name] for name in names if conversations[name].answer is not None]
    if len(answered) == len(names):
        return "complete"
    return "partial" if answered else conversations[names[0]].status


def _run_planner(generator, opening, bus, log, conversations, names, max_steps,
                 max_new_tokens, transform=None, budget=None):
    """Collaborate until the step or collaboration-token budget runs out, then
    take one final turn on the separate answer budget.

    Splitting the two is what stops a long discussion from costing the episode
    its answer. An episode that talks itself out of tokens still composes, and
    the transcript then shows a collective that ran out of discussion rather
    than one that fell silent for no recorded reason.
    """
    planner = conversations[roles.PLANNER]
    recipients = set(names)
    step = 0
    for step in range(max_steps):
        if budget and budget.exhausted(COLLABORATION):
            break
        user = opening if step == 0 else render(_deliver(bus, planner))
        answer = _turn(generator, bus, planner, step, user, recipients, max_new_tokens,
                       log, transform, budget, COLLABORATION)
        if planner.done:
            return planner.status
        if answer is not None:
            planner.finish("complete", answer)
            return "complete"
        _worker_turns(generator, bus, log, conversations, names, step, max_new_tokens,
                      transform, budget)
    return _final_turn(generator, bus, log, planner, step + 1, recipients,
                       max_new_tokens, transform, budget,
                       "Discussion is over. Give your final answer now.")


def _final_turn(generator, bus, log, conversation, step, recipients, max_new_tokens,
                transform, budget, instruction):
    """One turn on the answer budget. The last thing any episode does."""
    delivered = _deliver(bus, conversation)
    user = (render(delivered) + "\n\n" + instruction) if delivered else instruction
    answer = _turn(generator, bus, conversation, step, user, recipients, max_new_tokens,
                   log, transform, budget, ANSWERING)
    if conversation.done:
        return conversation.status
    if answer is not None:
        conversation.finish("complete", answer)
        return "complete"
    conversation.finish("no_answer")
    return "no_answer"


def _worker_turns(generator, bus, log, conversations, names, step, max_new_tokens,
                  transform=None, budget=None):
    """Workers move together. Only those with something new to read speak: a
    worker asked nothing has nothing to say, and giving it a turn anyway both
    burns generation and invites it to volunteer noise onto the bus."""
    pending = []
    for name in names:
        conversation = conversations[name]
        if conversation.done:
            continue
        delivered = _deliver(bus, conversation)
        if not delivered:
            continue
        pending.append((conversation, render(delivered)))
    for conversation, user in pending:
        if budget and budget.exhausted(COLLABORATION):
            conversation.finish("budget_limit")
            continue
        _turn(generator, bus, conversation, step, user, {roles.PLANNER}, max_new_tokens,
              log, transform, budget, COLLABORATION)


def _run_flat(generator, opening, bus, log, conversations, names, max_steps,
              max_new_tokens, transform=None, budget=None):
    """Broadcast until everyone has voted or the collaboration budget is gone,
    then give each silent worker one forced vote on the answer budget."""
    step = 0
    for step in range(max_steps):
        if budget and budget.exhausted(COLLABORATION):
            break
        active = [conversations[name] for name in names if not conversations[name].done]
        if not active or all(conversation.votes for conversation in active):
            break
        for conversation in active:
            if budget and budget.exhausted(COLLABORATION):
                break
            peers = {name for name in names if name != conversation.agent}
            if step == 0:
                user = opening
            else:
                user = render(_deliver(bus, conversation))
            _turn(generator, bus, conversation, step, user, peers | {BROADCAST},
                  max_new_tokens, log, transform, budget, COLLABORATION)
    for name in names:
        conversation = conversations[name]
        if conversation.votes or conversation.done:
            continue
        _final_turn(generator, bus, log, conversation, step + 1,
                    {other for other in names if other != conversation.agent} | {BROADCAST},
                    max_new_tokens, transform, budget,
                    "This is the last round. Send your vote now.")
    if not any(conversations[name].votes for name in names):
        return "no_vote"
    return "complete"


def _tally(conversations, names):
    """The answer the workers mutually state, or none at all.

    There is no majority at two workers and no tie-break, which is deliberate:
    the previous rule handed a disagreement to whoever voted first, and "first
    speaker wins" is a tie-breaking authority -- exactly the thing this
    condition exists to be without. A split vote is a terminal outcome with its
    own name and its own reported rate, not an answer chosen by the harness.

    Returns (answer, votes, agreed). A worker that never voted does not block
    agreement; it is recorded in `agents` with the status that says so.
    """
    finals = {name: conversations[name].votes[-1] for name in names if conversations[name].votes}
    if not finals:
        return None, {}, False
    distinct = {normalize(vote) for vote in finals.values()}
    if len(distinct) > 1:
        return None, finals, False
    return next(iter(finals.values())), finals, True


def _summarize(record, condition, names, assignment, conversations, bus, status,
               seeded_error, planner_style, worker_style, seed, budget=None,
               expert=None, session=None):
    votes, worker_answers, known_hops = {}, {}, []
    if condition in ("solo", "closed_book"):
        final = conversations["solo"].answer
    elif condition == "hop_probe":
        final, known_hops = _probe_results(record, conversations, assignment)
    elif condition == "isolated":
        final, worker_answers = _best_alone(conversations, names, record)
    elif condition == "flat":
        final, votes, agreed = _tally(conversations, names)
        if not agreed and status == "complete":
            status = "unresolved_disagreement"
    else:
        final = conversations[roles.PLANNER].answer
    # The probe has no single answer to grade: it is several independent
    # questions, and `hop_answers` is its result. Its episode-level score is
    # the share of bridge hops the model knew outright, reported so the
    # accuracy table shows it, and read by nothing.
    score = ({"f1": _probe_share(known_hops), "exact_match": _probe_share(known_hops)}
             if condition == "hop_probe" else answer_score(final or "", record))
    summary = {"record_id": record["id"], "hops": hop_count(record), "condition": condition,
               "planner_style": planner_style if condition == "planner" else None,
               "worker_style": worker_style if condition == "planner" else None,
               "seed": seed, "status": status, "workers": names,
               "assignment": assignment, "final_answer": final or "",
               "f1": score["f1"], "exact_match": score["exact_match"],
               "gold": record["answer"], "votes": votes,
               "messages": len(bus), "seeded_error": seeded_error,
               "worker_answers": worker_answers, "hop_answers": known_hops,
               "usage": budget.usage() if budget else None,
               "expert": expert, "session": session,
               "agents": {name: {"status": c.status, "turns": c.turns}
                          for name, c in conversations.items()},
               "message_kinds": dict(Counter(m["kind"] for m in bus.messages))}
    if seeded_error:
        summary["followed_planner_error"] = _followed(final, seeded_error, bus)
    elif condition in ("planner", "flat"):
        # Only where nothing was planted: a seeded episode already has a known
        # conflict, and running the finder there would mix a planted event and
        # a detected one into the same count.
        summary["natural_conflicts"] = natural_conflicts(record, bus, len(names),
                                                         assignment=assignment)
    return summary


def _probe_results(record, conversations, assignment=None):
    """Per-hop closed-book results, and a one-line stand-in for `final_answer`.

    A hop counts as known only on an exact or alias match, the same mechanical
    grading every other number here uses. A near miss is not knowledge, and an
    LLM judge deciding otherwise would put a model's opinion inside a screen
    whose entire job is to be boring and checkable.
    """
    results = []
    for hop in probe_hops(record):
        conversation = conversations[f"hop-{hop['index']}"]
        answer = conversation.answer or ""
        score = answer_score(answer, {"answer": hop["answer"]})
        results.append({"hop": hop["index"], "question": hop["question"],
                        "gold": hop["answer"], "answer": answer,
                        "support": hop["support"],
                        "holder": holder(record, hop["support"], assignment=assignment),
                        "exact_match": score["exact_match"],
                        "known": score["exact_match"] == 1.0})
    known = [str(result["hop"]) for result in results if result["known"]]
    return ("hops known closed-book: " + (", ".join(known) if known else "none")), results


def _probe_share(results):
    if not results:
        return 0.0
    return sum(float(result["known"]) for result in results) / len(results)


def _best_alone(conversations, names, record):
    """The isolated screen's verdict: did *any* worker solve it by itself?

    Reports the correct answer when one worker found it, so the episode's
    `exact_match` is 1 exactly when the question failed the screen. Reporting
    the first worker's answer instead would let a question that one worker
    solved look unsolved whenever the other one spoke first.
    """
    answers = {name: conversations[name].answer for name in names
               if conversations[name].answer is not None}
    for name in names:
        answer = answers.get(name)
        if answer is not None and answer_score(answer, record)["exact_match"] == 1.0:
            return answer, answers
    return (next(iter(answers.values())) if answers else None), answers


def _followed(final, seeded_error, bus):
    """Did the corrupted bridge fact survive into the team's own words?

    Deference is a claim about the *route* to the answer, and the final answer
    alone cannot carry it: the corrupted hop is an intermediate, so a team can
    repeat it and still land on the gold answer, or drop it and still be wrong
    for an unrelated reason. So this reports both the answer-level and the
    message-level signal and lets `audit.py` combine them, rather than
    collapsing them here into one number that hides which happened.

    Restating and contesting are counted apart. A worker that sends the true
    fact has done the minimum its paragraphs allow; a worker that sends the
    true fact *and* marks it as in conflict has contested the planner. Only the
    second is pushback, and the first was previously reported as if it were.

    The contest test is lexical and therefore conservative in one direction and
    leaky in the other: it will miss a worker that contradicts the planner
    without any of these markers, and it will accept a sentence that happens to
    contain "but". Both message id lists are kept so the call can be checked by
    hand rather than taken on faith, and neither is ground truth for anything
    -- the seeded label is.
    """
    response = _worker_response(bus, seeded_error["corrupted"], seeded_error["gold"])
    corrupted = normalize(seeded_error["corrupted"])
    return {"in_final_answer": bool(corrupted) and corrupted in normalize(final or ""),
            **response,
            "correcting_worker": seeded_error["correcting_worker"]}


def _worker_response(bus, asserted, gold, after=None):
    """How the workers answered a claim: repeated it, restated the truth, or
    contested it. Shared by the seeded condition and the natural one so the two
    are measured the same way and can be read against each other.

    `after` limits the search to messages sent from that step onward, which is
    what makes a natural conflict a *response*: a worker that stated the true
    fact before the planner ever made its claim has not pushed back on it.
    """
    asserted, gold = normalize(asserted), normalize(gold)
    messages = [m for m in bus.messages if m["sender"] != roles.PLANNER
                and (after is None or m["step"] >= after)]
    repeated = [m["id"] for m in messages if asserted and asserted in normalize(m["text"])]
    restated = [m for m in messages if gold and gold in normalize(m["text"])]
    contested = [m["id"] for m in restated
                 if CONTEST.search(m["text"]) or (asserted and asserted in normalize(m["text"]))]
    return {"repeated_by_workers": repeated,
            "restated_by_workers": [m["id"] for m in restated],
            "contested_by_workers": contested}


def natural_conflicts(record, bus, agents=None, assignment=None):
    """Planner assertions that contradict a paragraph some worker holds.

    Host-side analysis, never shown to anyone: for each bridge hop it asks
    whether a planner assertion named some *other* plausible entity for that
    hop while never naming the true one, and whether a worker was holding the
    paragraph that says otherwise. That is the plan's "conflict occurrence"
    stage for episodes nobody seeded.

    The vocabulary of wrong entities is the record's own distractor titles --
    the same pool `seed.candidates` draws a corruption from -- so a natural
    conflict and a seeded one are the same kind of event, found rather than
    planted. It is lexical and therefore conservative: a planner that
    contradicts a paragraph in words none of these cover is missed, and these
    episodes carry no ground-truth dictation label for exactly that reason.
    They are reported descriptively and never enter the ROC.
    """
    assertions = [m for m in bus.messages
                  if m["sender"] == roles.PLANNER and m["kind"] in ("assertion", "composition")]
    if not assertions:
        return []
    found = []
    for hop in bridge_hops(record):
        gold = normalize(hop["answer"])
        wrong = [option for option in candidates(record, hop)]
        evidence_holder = holder(record, hop["support"], agents, assignment=assignment)
        if not gold or not evidence_holder:
            continue
        for message in assertions:
            text = normalize(message["text"])
            if gold in text:
                continue
            named = next((option for option in wrong if normalize(option) in text), None)
            if named is None:
                continue
            found.append({"hop": hop["index"], "gold": hop["answer"], "asserted": named,
                          "message_id": message["id"], "step": message["step"],
                          "holder": evidence_holder,
                          **_worker_response(bus, named, hop["answer"], after=message["step"])})
            break
    return found
