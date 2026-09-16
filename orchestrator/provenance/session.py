"""Sessions: one cohort, a sequence of questions, a budget that shrinks.

A batch without this is a set of independent questions. A session is a cohort
answering them in order, in blocks, under a per-round cap on generated tokens
and turns that declines as the session goes on. The prediction it exists to
test is the plan's: verification and pushback are the first casualties of a
shrinking budget, so deference rises as the budget falls.

Three things this deliberately does *not* do.

**No memory across questions.** Each question is still its own episode, with
its own bus and its own agents. Carrying a conversation across a block would
make the whole session the unit of replay -- and replay fidelity is the
load-bearing invariant of this package, not something to spend on a
convenience. It would also leak: MuSiQue questions in a block share evidence
(315 of 1077 sub-chain ids in a sample of 800 records are reused across
questions, and 437 question pairs share a supporting-paragraph title), so a
worker could answer question four from what it read in question two, and the
closed-book screen would stop describing the agent that actually answered.
Session-wide memory is a real experiment; it is a different one, and it needs
its own replay design before it is worth having.

**No wall-clock pressure.** Inference time is not a quantity anyone here
controls, so "time pressure" is the per-round cap on turns, and "token
pressure" the cap on generated tokens. Both decline together. This is the
plan's own operationalization and the substitution is stated rather than
smuggled.

**No earmarks.** The per-round budget is one pool per phase, as it already
was. Nothing is reserved for pushback or for evidence relay -- subsidising the
measured behaviour would buy it into existence, which is the same objection
that forbids prompting for it.
"""
import random

# Fractions of the uncapped per-episode budget. The plan asks for levels set as
# percentiles of an uncapped calibration run (p90 down to p25) rather than as
# round numbers, and `levels_from_usage` does that from a finished run. These
# are what a first calibration run uses before any distribution exists, and a
# run records whichever it used in its manifest.
DEFAULT_START, DEFAULT_END = 1.0, 0.35
DEFAULT_ROUNDS = 10
DEFAULT_BLOCK = 5
SHAPES = ("declining", "flat")


def percentile(values, fraction):
    """Nearest-rank percentile. No numpy for one line of arithmetic."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def levels_from_usage(usages, rounds, high=0.90, low=0.25):
    """Per-round token caps as percentiles of what an uncapped run really spent.

    `usages` is the per-episode generated-token total from a calibration batch.
    Taking the schedule from the distribution rather than from round numbers is
    what makes "the budget is tight" mean something about this model on this
    task, instead of about a number somebody picked.
    """
    top, bottom = percentile(usages, high), percentile(usages, low)
    if top is None:
        return None
    return [round(level) for level in steps(top, bottom, rounds)]


def steps(start, end, rounds, shape="declining"):
    """The per-round levels: equal steps from `start` down to `end`.

    `flat` is the control arm and sits at the schedule's midpoint, so the two
    arms spend the same total across a session and differ only in whether the
    budget moves. Round index and budget level are perfectly confounded inside
    the declining arm; the flat arm is the only thing that separates them, and
    a deference-versus-round curve from one arm alone supports no causal claim
    about scarcity.
    """
    if rounds < 1:
        raise ValueError("A session needs at least one round")
    if shape not in SHAPES:
        raise ValueError(f"Unknown schedule shape {shape!r}")
    # Not rounded: these are levels in whatever unit the caller passed, and the
    # callers pass both token counts and fractions of a budget. Rounding here
    # silently floors a schedule of fractions to 1, 1, ... 0, which is not a
    # declining schedule but an on/off switch.
    if shape == "flat":
        return [(start + end) / 2] * rounds
    if rounds == 1:
        return [start]
    span = (start - end) / (rounds - 1)
    return [start - span * index for index in range(rounds)]


def budgets(collaboration, answer, max_steps, rounds, start=DEFAULT_START,
            end=DEFAULT_END, shape="declining", levels=None):
    """Per-round (collaboration tokens, answer tokens, turn cap).

    The answer allowance declines with everything else but never to zero: an
    episode that cannot emit an answer produces no measurement at all, and a
    schedule that manufactures no-answer episodes at its tight end would look
    exactly like one that manufactures deference.
    """
    # A calibrated schedule arrives as absolute token counts; an uncalibrated
    # one as fractions of the uncapped budget. Both become fractions here so
    # the turn cap and the answer allowance scale with the same number.
    scale = ([level / max(collaboration, 1) for level in levels] if levels
             else steps(start, end, rounds, shape))
    return [{"round": index,
             "collaboration_tokens": max(1, round(collaboration * fraction)),
             "answer_tokens": max(1, round(answer * max(fraction, 0.5))),
             "max_steps": max(2, round(max_steps * fraction))}
            for index, fraction in enumerate(scale)]


def sequence(records, rounds, block, base_seed, cohort=0):
    """One cohort's question order and which worker is the expert in each block.

    Question-to-round assignment is randomised per cohort and the block order
    is flipped on alternate cohorts, because otherwise the first worker is
    always the expert early and every "expert deferred less in round one"
    reading is also "agent-1 went first". Difficulty and position are
    decorrelated the same way.
    """
    rng = random.Random(f"{base_seed}:cohort:{cohort}")
    chosen = list(records)
    rng.shuffle(chosen)
    chosen = chosen[:rounds]
    names = ["agent-1", "agent-2"]
    if cohort % 2:
        names.reverse()
    plan = []
    for index, record in enumerate(chosen):
        plan.append({"round": index, "block": index // block,
                     "expert": names[(index // block) % len(names)],
                     "record_id": record["id"]})
    return plan


def overlap(records):
    """Supporting-paragraph titles shared between consecutive questions.

    Reported, not prevented. A sequence whose neighbours share evidence is one
    where a memory-carrying variant would leak, and where even without memory
    the same entity turns up twice in a session; a reader comparing two
    sessions deserves to know which one was built out of overlapping chains.
    """
    titles = []
    for record in records:
        titles.append({p["title"] for p in record["paragraphs"] if p.get("is_supporting")})
    shared = [len(a & b) for a, b in zip(titles, titles[1:])]
    return {"pairs": len(shared), "sharing": sum(1 for n in shared if n),
            "mean_shared_titles": round(sum(shared) / len(shared), 3) if shared else None}
