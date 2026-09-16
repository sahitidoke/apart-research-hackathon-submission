"""Seeding the planner with one wrong intermediate belief.

Natural errors give ground truth about correctness but not about *provenance*:
when a 4-hop collective is wrong, nobody knows whether it followed bad
evidence or followed the planner. Flipping one hop by hand fixes that. The
planner is told a false bridge fact as settled; exactly one worker holds the
paragraph that contradicts it; and whether the collective's answer tracks the
planner or the worker is then a label, not an inference.

Two invariants this module exists to keep:

* the final hop is never touched (`records.bridge_hops` already drops it), so
  the planner is never simply handed the wrong answer;
* the corruption never reaches a worker. It goes into the planner's system
  prompt only, and `episode.py` is the one place that reads it.
"""
from orchestrator.provenance.scoring import normalize
from orchestrator.provenance.records import bridge_hops, holder, locatable, share


def candidates(record, hop, pool=()):
    """Plausible replacements for one hop's gold answer, best first.

    A distractor title from the record's own paragraph set is the strongest
    option: MuSiQue distractors are chosen to be topically adjacent, so the
    substitution reads as an ordinary mistake rather than as noise. Bridge
    answers from other records in the batch are the fallback for records whose
    distractor titles all collide with the gold text.
    """
    blocked = {normalize(hop["answer"]), normalize(record["answer"])}
    blocked.update(normalize(alias) for alias in record.get("answer_aliases", []))
    options = []
    for position, paragraph in enumerate(record["paragraphs"]):
        if paragraph.get("is_supporting"):
            continue
        options.append(paragraph["title"])
    options.extend(pool)
    # A replacement that shares a substantial word with the gold answer would
    # score as partially correct on token F1, which blurs the very label this
    # corruption exists to make crisp.
    overlapping = significant_words(blocked)
    seen, chosen = set(), []
    for option in options:
        key = normalize(option)
        if not key or key in blocked or key in seen or key in overlapping:
            continue
        seen.add(key)
        chosen.append(option.strip())
    return chosen


def significant_words(texts):
    return {word for text in texts for word in text.split() if len(word) > 3}


def corrupt(record, rng, pool=(), agents=None, assignment=None):
    """Flip one bridge hop. Returns the label, or None when no hop is usable.

    The label carries everything the audit needs later: which hop, what the
    planner was told, what the truth is, and which worker holds the paragraph
    that says so. `agents` has to match the cohort the episode will actually
    run with, or `correcting_worker` names a worker that was dealt something
    else -- and that name is the ground truth every deference rate is
    conditioned on.

    `assignment` is the other half of that requirement and matters for the same
    reason. The count alone is enough only while the deal is the round-robin
    split; a session run deals the expert every supporting paragraph, and
    without the actual assignment this names the bystander -- a worker holding
    nothing but distractors -- as the one that could have corrected the planner.
    The episode then runs, completes, and reports a label no agent in it could
    have earned.

    A hop whose gold entity cannot be found in the correcting worker's share is
    skipped rather than planted on. Corrupting one produces an episode that
    looks complete and then silently has no evidence arm, because the swap that
    arm depends on has nothing to substitute -- four arms were lost that way on
    the first real run, and it read as a limitation of the design rather than
    as the pipeline defect it was. `records.locatable` is the same search
    `swapped_record` will later perform, so the two can no longer disagree.

    `rng` is expected to be seeded per record. Threading one stream through a
    whole batch means re-planting one question's error shifts every later
    question's, which makes repairing a single episode impossible without
    invalidating the rest -- see `batch.plan`.
    """
    hops = bridge_hops(record)
    rng.shuffle(hops)
    for hop in hops:
        correcting = holder(record, hop["support"], agents, assignment=assignment)
        if correcting is None:
            continue
        if not locatable(record, hop["answer"],
                         share(record, hop["support"], agents, assignment=assignment)):
            continue
        options = candidates(record, hop, pool)
        if not options:
            continue
        return {"hop": hop["index"], "question": hop["question"],
                "gold": hop["answer"], "corrupted": rng.choice(options),
                "support": hop["support"], "correcting_worker": correcting}
    return None


def bridge_pool(records):
    """Bridge answers from a batch, for records whose own distractors are unusable."""
    return [hop["answer"] for record in records for hop in bridge_hops(record)]
