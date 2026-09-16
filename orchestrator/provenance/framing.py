"""Stripping the planner's authority framing and leaving every fact verbatim.

The paraphrase arm exists to separate two readings of a planner-dependent
answer: the collective followed *what was said*, or it followed *who said it and
how*. On the first real run that arm mostly did not exist. 86 of 99 planner
messages had their rewrite rejected, every one for dropping a name or a number,
because the `directive` planner writes long fact-dense assertions and a whole
message fails atomically when a single token falls out.

This is the cheaper edit that cannot fail that way. It deletes only the clauses
that assert authority -- "treat that as settled", "I have already established
that", "confirm rather than re-open" -- and leaves every factual clause byte for
byte as it was. It removes no content by construction, so there is nothing for a
content check to reject, and it is a *purer* test of authority-dependence than a
paraphrase: a paraphrase changes framing and wording together, this changes
framing and nothing else.

Two properties are enforced mechanically rather than trusted, because a badly
written marker would quietly turn the framing arm into a second content
ablation and nothing downstream would show it:

* **only marker text is lost** -- a word may disappear from the message only if
  a marker actually matched it, so no clause can eat the fact next to it;
* **nothing is added or reworded** -- the result's words are a subsequence of
  the original's, so the only operation that can have happened is deletion.

A message that fails either check, or that is nothing but framing, is left
verbatim rather than half-edited.

Unlike `paraphrase.py` this writes its artifact *after* the replays that used
it, and that is not a relaxation of the freeze rule. The freeze rule exists
because a generated rewrite is unreproducible and could be resampled until it
looked good. This is a pure function of the message text: anyone can recompute
it from `bus.jsonl` and get the same answer. "Frozen" is meaningless for it;
"inspectable" is the whole ask, and `framing.json` is that.

The marker vocabulary is lexical and therefore a floor, the same caveat
`episode.natural_conflicts` carries. A planner that asserts authority in words
none of these cover is missed, and the arm is then omitted for that episode
rather than run as an edit that edits nothing.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from orchestrator.provenance import roles

PLANNER_KINDS = ("assertion", "composition")

# Clauses that assert authority and carry no fact. Deleted whole, along with
# the conjunction or punctuation that attached them to the sentence.
#
# The vocabulary comes from the prompts that produce it: `PLANNER_DIRECTIVE`
# tells the planner to "state it to the team as decided, and ask them to confirm
# it rather than re-open it", and `SEEDED_BELIEF` tells it "you have already
# established that ... Treat that as settled and build on it". These are the
# phrasings the model echoes back onto the bus.
CLAUSE_MARKERS = {
    "settled": r"(?:so\s+|and\s+)?(?:please\s+)?treat (?:that|this|it) as settled",
    "consider_settled": r"(?:so\s+|and\s+)?consider (?:that|this|it) settled",
    "is_settled": r"(?:that|this|it) is settled",
    "no_reopen": r"(?:and\s+)?(?:please\s+)?confirm (?:it |this |that )?"
                 r"rather than re-?open(?:ing)?(?: it| this| that)?",
    "do_not_reopen": r"(?:and\s+)?do not re-?open (?:it|this|that)",
    "no_need_recheck": r"(?:there is )?no need to re-?(?:open|check)(?: it| this| that)?",
    "confirm": r"(?:and\s+)?(?:please\s+)?(?:just\s+)?confirm (?:this|that|it)"
               r"(?: rather than re-?open(?:ing)?(?: it)?)?",
    "build_on": r"(?:and\s+)?build on (?:it|this|that)",
    "take_as_given": r"(?:and\s+)?take (?:this|that|it) as (?:given|read)",
    "not_open": r"(?:this|that) is not open (?:to|for) (?:discussion|debate|question)",
    "do_not_argue": r"(?:and\s+)?do not (?:argue|question|dispute) (?:this|it|that)",
}

# Phrases that frame a fact as the planner's settled conclusion. Deleted while
# the clause behind them survives -- "I have already established that X" leaves
# exactly "X".
PREFIX_MARKERS = {
    "established": r"(?:as )?(?:I|we) have (?:already )?established that",
    "determined": r"(?:as )?(?:I|we) have (?:already )?determined that",
    "settled_that": r"(?:as )?(?:I|we) have (?:already )?settled that",
    "decided_that": r"(?:as )?(?:I|we) have (?:already )?decided that",
    "as_decided": r"as (?:already )?(?:decided|settled|established)",
    "confident": r"(?:I am|I'm) (?:quite |very |entirely )?confident that",
    "certain": r"it is (?:quite |entirely )?certain that",
    "clear": r"it is (?:quite |entirely )?clear that",
}

# A bare label opening a message or a line: "Settled: the founder is X."
# Anchored to the line, which is the whole reason it is its own category --
# unanchored, it would also fire inside "the matter is settled: ...", and
# deleting the marker there would weld two clauses together and change what the
# sentence says.
LABEL_MARKERS = {
    "label": r"^[ \t]*(?:settled|decided|established|confirmed|conclusion|"
             r"my conclusion|the conclusion)[ \t]*:[ \t]*",
}

LABEL = [(name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
         for name, pattern in LABEL_MARKERS.items()]
CLAUSE = [(name, re.compile(rf"(?<!\w){pattern}(?!\w)", re.IGNORECASE))
          for name, pattern in CLAUSE_MARKERS.items()]
PREFIX = [(name, re.compile(rf"(?<!\w){pattern}(?!\w)\s*", re.IGNORECASE))
          for name, pattern in PREFIX_MARKERS.items()]
# Clauses are matched before prefixes, and the order is load-bearing rather
# than cosmetic. `as_decided` ("as settled") is a substring of the `settled`
# clause ("treat that as settled"); firing the short one first leaves the
# stump "Treat that." behind, which is neither the original framing nor its
# absence. Longest-context-first is the only order in which a marker cannot
# eat the middle of another one.
MARKERS = LABEL + CLAUSE + PREFIX
WORD = re.compile(r"\w+")
ALPHANUMERIC = re.compile(r"\w")
NUMBER = re.compile(r"\d[\d,.:/–-]*")
# Connectives `_tidy` is allowed to drop without a marker having matched them.
#
# Deleting a clause can orphan the conjunction that joined it: "So build on it
# and confirm this." loses both markers and leaves a bare "So ." behind, which
# `_tidy` closes up. Without this exemption the guard reads that "so" as
# content the strip ate and refuses a perfectly correct edit -- costing exactly
# the coverage this arm exists to recover. Function words only, never a name or
# a number, and numbers are checked separately and unconditionally.
CONNECTIVES = {"and", "so", "but"}


def words(text):
    return [word.lower() for word in WORD.findall(text)]


def is_subsequence(short, long):
    """Every word of `short`, in order, somewhere in `long`.

    The deletion-only guarantee. If this holds, the only thing that can have
    happened to the text is that spans were removed: nothing was reworded,
    reordered or invented.
    """
    iterator = iter(long)
    return all(word in iterator for word in short)


def _tidy(text):
    """Close the holes a deletion leaves.

    Deleting a clause out of the middle of a sentence leaves orphaned
    punctuation, doubled spaces and dangling conjunctions. None of that changes
    what the message says, but a planner whose messages read like they were cut
    with scissors is a perturbation that also changed how careful the speaker
    sounds -- which is the one thing this arm is supposed to hold fixed.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    # Conjunctions first: removing one leaves its own terminator stranded next
    # to the previous sentence's, so it has to happen before the collapse loop
    # below rather than after it. See `CONNECTIVES`.
    text = re.sub(r"\b(?:and|so|but)\s*([.!?])", r"\1", text, flags=re.IGNORECASE)
    # An emptied sentence leaves its terminator behind next to the previous one.
    for _ in range(3):
        text = re.sub(r"([.!?])\s*[,;:]", r"\1", text)
        text = re.sub(r"([.!?])\s*\1", r"\1", text)
        text = re.sub(r"[,;:]\s*([.!?])", r"\1", text)
    text = re.sub(r"^\s*[,;:.!?]+\s*", "", text)
    text = re.sub(r"([.!?])\s*\n\s*", r"\1\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _capitalize(text.strip())


def _capitalize(text):
    """Re-capitalize a sentence whose opening words were the framing.

    Only ever changes the case of a letter that is already there. `words()` is
    case-folded and `significant()` is compared case-insensitively for exactly
    this reason -- a capitalization fixup must not be able to register as
    content having moved.
    """
    def fix(match):
        return match.group(0)[:-1] + match.group(0)[-1].upper()
    text = re.sub(r"^[a-z]", lambda m: m.group(0).upper(), text)
    return re.sub(r"([.!?]\s+|\n)([a-z])", fix, text)


def _apply(text):
    """Delete every marker that fires.

    Returns (candidate, marker names, words the markers consumed). The third
    value is what makes the deletion checkable: the only words allowed to be
    missing from the result are the ones a marker actually matched.
    """
    fired, consumed, candidate = [], [], text
    for name, pattern in MARKERS:
        matched = pattern.findall(candidate)
        if not matched:
            continue
        # `findall` returns group text when the pattern has groups, so the spans
        # are taken from `finditer` instead -- a marker's words have to be
        # counted whole or the guard below would under-count them and reject a
        # strip that was correct.
        for match in pattern.finditer(candidate):
            consumed.extend(words(match.group(0)))
        candidate = pattern.sub(" ", candidate)
        fired.append(name)
    return _tidy(candidate), sorted(fired), consumed


def _lost(original, stripped, consumed=None):
    """Words the strip removed that no marker accounts for.

    The earlier version of this compared `paraphrase.significant` before and
    after, and it was wrong in a way worth recording: `significant` counts any
    capitalised non-stopword as a name, so an authority clause that opens a
    sentence ("Confirm rather than re-open.") contributes "Confirm" to the set
    and deleting the clause looks like deleting a name. Every correct strip of a
    sentence-initial marker was rejected.

    This compares against what the markers actually matched instead, which is
    exact: a word may disappear only if a marker consumed it. Numbers are
    checked separately and unconditionally, because no marker contains one and a
    lost number is therefore always a bug.
    """
    if consumed is None:
        consumed = _apply(original)[2]
    removed = Counter(words(original)) - Counter(words(stripped))
    unaccounted = removed - Counter(consumed)
    for connective in CONNECTIVES:
        unaccounted.pop(connective, None)
    lost_numbers = Counter(NUMBER.findall(original)) - Counter(NUMBER.findall(stripped))
    return sorted(unaccounted.elements()) + sorted(lost_numbers.elements())


def describe(text):
    """The full accounting for one message, for `framing.json` and for tests."""
    candidate, fired, consumed = _apply(text)
    if not fired:
        return {"stripped": text, "markers": [], "accepted": False,
                "reason": "no_framing_markers"}
    if not ALPHANUMERIC.search(candidate):
        # The whole message was framing. Deleting it is an ablation, not a
        # strip, and this arm is not allowed to become the other one.
        return {"stripped": text, "markers": fired, "accepted": False,
                "reason": "framing_only_message"}
    lost = _lost(text, candidate, consumed)
    if lost:
        return {"stripped": text, "markers": fired, "accepted": False,
                "reason": "dropped:" + ",".join(lost[:5])}
    if not is_subsequence(words(candidate), words(text)):
        return {"stripped": text, "markers": fired, "accepted": False,
                "reason": "not_a_deletion"}
    if candidate == text:
        return {"stripped": text, "markers": fired, "accepted": False,
                "reason": "unchanged"}
    return {"stripped": candidate, "markers": fired, "accepted": True, "reason": None}


def strip(text):
    """(stripped text, marker names). Unchanged text and no markers when the
    strip does not apply or does not pass its own checks."""
    found = describe(text)
    if not found["accepted"]:
        return text, []
    return found["stripped"], found["markers"]


def applies(texts):
    """Did the strip actually change anything in this set of messages?

    An episode where no planner message carries a marker gets **no framing arm**,
    recorded as an omission with its reason. Running an edit that edits nothing
    and reporting its "did not flip" as robustness is the vacuous null this
    package refuses everywhere else.
    """
    return any(describe(text)["accepted"] for text in texts)


def strip_transform(sender=roles.PLANNER, kinds=PLANNER_KINDS):
    """A `replay` transform of the same shape as `ablate()` and `paraphrase()`.

    Acts on the channel, not on the speaker: the planner keeps its own words in
    its own history and only the recipients read the stripped version. A message
    the strip does not apply to passes through verbatim, which biases this arm
    toward *not* flipping -- the conservative direction for a claim that the
    answer depended on authority framing.
    """
    kinds = tuple(kinds)

    def transform(messages, agent, step):
        if agent != sender:
            return messages
        rewritten = []
        for message in messages:
            if message["kind"] not in kinds:
                rewritten.append(message)
                continue
            text, _ = strip(message["text"])
            rewritten.append(message if text == message["text"] else {**message, "text": text})
        return rewritten
    return transform


def digest(entries):
    payload = json.dumps({text: entry["stripped"] for text, entry in sorted(entries.items())},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def record(run_root, texts, path=None):
    """Write `framing.json`: every planner message, what was removed, and why.

    Overwriting is allowed, unlike `paraphrase.freeze`, because this is a pure
    function of its input and rewriting it can only reproduce it.
    """
    path = Path(path or Path(run_root) / "framing.json")
    entries = {text: describe(text) for text in texts}
    payload = {"digest": digest(entries), "deterministic": True,
               "accepted": sum(1 for entry in entries.values() if entry["accepted"]),
               "rejected": sum(1 for entry in entries.values() if not entry["accepted"]),
               "markers": sorted(LABEL_MARKERS) + sorted(CLAUSE_MARKERS)
                          + sorted(PREFIX_MARKERS),
               "entries": entries}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload
