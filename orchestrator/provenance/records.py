"""Selecting MuSiQue records and reading their hop structure.

Hop count is the difficulty axis. MuSiQue encodes it in the record id
(`2hop__...`, `3hop1__...`, `4hop2__...`), which is the only place it appears
in the public record, so it is parsed rather than inferred.

`question_decomposition` is host-side ground truth: it names each intermediate
hop, its gold sub-answer, and the paragraph that supports it. `seed.corrupt`
needs all three to flip exactly one hop. None of it reaches an agent.
"""
import hashlib
import random
import re

from orchestrator.provenance.split import agent_names, split_positions

HOP_ID = re.compile(r"^(\d+)hop")
# MuSiQue writes a decomposition answer as "#2" when the hop's answer is
# another hop's answer rather than a literal span.
PLACEHOLDER = re.compile(r"^#\d+$")
# The same back-reference *inside* a sub-question ("Who founded #1?"), where it
# is not the whole string. Anchored and unanchored are kept apart on purpose:
# searching with the anchored pattern would silently never match here.
REFERENCE = re.compile(r"#\d+")
# Word-ish tokens, for finding a gold entity that the supporting paragraph
# writes in a shorter form than the decomposition does.
TOKEN = re.compile(r"[\w'’-]+")
# Tokens that could match half the corpus and so cannot stand for an entity.
TOKEN_STOPWORDS = {"the", "and", "for", "from", "with", "that", "this", "which",
                   "were", "was", "his", "her", "their", "its", "county",
                   "city", "town", "state", "river", "school", "university",
                   "college", "church", "national", "international", "company",
                   "museum", "island", "district", "province", "region"}
# One planner and exactly two workers, at every hop count.
#
# Sizing the cohort to the hop count instead -- which this module used to do --
# means a 4-hop question runs four workers and a 2-hop question runs two, so
# hop count, the difficulty axis every rate is stratified by, moves the cohort
# size with it. A deference rate that rose with hops would then have two
# explanations and no way to separate them.
#
# Two also fixes what deference *means* here: the corrupted hop has exactly one
# evidence holder, so the measurement is one agent yielding to authority, not
# conformity to a majority. Three or more workers is the deferred worker-count
# study, not a knob to turn during collection.
WORKERS = 2


def hop_count(record):
    """Hops in this question, from the record id. Raises on an unparseable id."""
    match = HOP_ID.match(str(record.get("id", "")))
    if not match:
        raise ValueError(f"Record id {record.get('id')!r} does not name a hop count")
    hops = int(match.group(1))
    if hops < 2:
        raise ValueError(f"Record id {record.get('id')!r} claims {hops} hops")
    return hops


def bridge_hops(record):
    """Intermediate hops: every decomposition step except the last.

    The final step produces the answer itself. Corrupting it would not test
    deference to a planner's *reasoning*, it would just hand the planner the
    wrong answer outright, so it is excluded here rather than filtered later.

    Each returned hop carries a literal `answer` and the position of the
    paragraph that supports it. Steps whose answer is a "#n" back-reference
    are dropped: there is no span to replace.
    """
    steps = record.get("question_decomposition")
    if not isinstance(steps, list) or len(steps) < 2:
        return []
    hops = []
    for position, step in enumerate(steps[:-1]):
        if not isinstance(step, dict):
            continue
        answer = step.get("answer")
        support = step.get("paragraph_support_idx")
        if not isinstance(answer, str) or not answer.strip() or PLACEHOLDER.match(answer.strip()):
            continue
        if type(support) is not int or support < 0 or support >= len(record["paragraphs"]):
            continue
        hops.append({"index": position, "question": step.get("question", ""),
                     "answer": answer.strip(), "support": support})
    return hops


def resolve_question(record, question):
    """A sub-question that stands on its own, or None.

    MuSiQue writes later hops as "Which university did #1 attend?", where "#1"
    is the first step's answer. Asked verbatim that is unanswerable by anyone,
    so the reference is substituted with the answer it names: "Which university
    did Josiah Fenn attend?". Without this, only the first hop of a chain is
    ever probeable, which on this dataset is most of the chain unscreened.

    This does put an earlier hop's gold answer into a prompt, and that is the
    whole point of the probe -- it asks whether the model knows the *next* fact
    given the bridge entity, which is exactly the parametric knowledge that
    would make an evidence swap meaningless. It is confined to `hop_probe`
    episodes, which have no bus and whose output is never read back.
    """
    steps = record.get("question_decomposition") or []

    def substitute(match):
        index = int(match.group(0)[1:]) - 1
        if not 0 <= index < len(steps):
            raise LookupError(match.group(0))
        answer = steps[index].get("answer")
        if not isinstance(answer, str) or not answer.strip() or PLACEHOLDER.match(answer.strip()):
            raise LookupError(match.group(0))
        return answer.strip()

    try:
        return REFERENCE.sub(substitute, question or "") or None
    except LookupError:
        return None


def supporting_positions(record):
    return [i for i, p in enumerate(record["paragraphs"]) if p.get("is_supporting")]


def worker_count(record):
    """Two, for every record. See `WORKERS`.

    Takes the record anyway because `usable` has already guaranteed it has
    enough supporting paragraphs to deal two non-empty shares, and a caller
    reading this signature should not have to know that.
    """
    return WORKERS


def deal(record, agents=None):
    """{worker: [paragraph position, ...]} using the reviewed MARL dealer.

    Supporting paragraphs go round-robin, so with two workers a 2-hop question
    gives each worker one hop's evidence and a 3- or 4-hop question gives each
    of them several. Neither ever holds all of them, which is the property the
    whole design rests on.
    """
    return split_positions(record, agents or worker_count(record))


def expert_deal(record, expert, agents=None):
    """{worker: [position, ...]} with one worker holding *every* support.

    The other side of `deal`. There the evidence is split so neither worker can
    answer alone and the messages have to carry it; here one worker holds the
    whole chain and the other holds only distractors, so nothing needs routing
    and what is left to measure is deference itself: an agent that can see the
    planner is wrong about material it holds in full, and an agent with nothing
    to go on at all.

    The two are not variants of one setting, they are different experiments,
    and the screens invert between them -- see `audit.retention`. Under this
    deal a question the expert *cannot* answer alone is the useless one, since
    an expert that yields was then not deferring, it was guessing.
    """
    names = workers(record, agents)
    if expert not in names:
        raise ValueError(f"{expert!r} is not one of {names}")
    supporting = supporting_positions(record)
    distractors = [i for i in range(len(record["paragraphs"])) if i not in set(supporting)]
    if not distractors:
        raise ValueError(f"{record['id']} has no distractors to give the other worker")
    assignment = {name: [] for name in names}
    assignment[expert] = list(supporting)
    others = [name for name in names if name != expert]
    for rank, position in enumerate(distractors):
        assignment[others[rank % len(others)]].append(position)
    # The expert gets a share of the distractors too. Without them its prompt
    # would be all and only supporting paragraphs, which is a much easier task
    # than anyone else in this experiment is given and would flatter its
    # accuracy for a reason that has nothing to do with deference.
    for rank, position in enumerate(distractors[::len(others) + 1]):
        if position not in assignment[expert]:
            assignment[expert].append(position)
            assignment[others[rank % len(others)]].remove(position)
    for name in names:
        assignment[name].sort()
    return assignment


def holder(record, position, agents=None, assignment=None):
    """Which worker was dealt the paragraph at `position`, or None.

    `assignment` is the deal the episode will actually run with, and has to be
    passed whenever that is not the round-robin split -- which is exactly the
    expert deal. Re-deriving it from `deal` there names whichever worker
    round-robin *would* have given the paragraph to, and under the expert deal
    one worker holds every support, so that answer is the bystander about half
    the time. The bystander holds no supporting paragraph at all and cannot
    contradict anything, so the label ends up on an agent that could not have
    produced the behaviour it is labelling.
    """
    for agent, positions in (assignment or deal(record, agents)).items():
        if position in positions:
            return agent
    return None


def substitute(text, original, replacement):
    """Replace whole occurrences of `original`, case-insensitively. Returns
    (text, count) so a caller can tell a swap that happened from one that
    silently matched nothing."""
    pattern = re.compile(rf"(?<!\w){re.escape(original)}(?!\w)", re.IGNORECASE)
    return pattern.subn(replacement, text)


def distinctive(gold):
    """The longest token of `gold` that could stand for it in running text.

    MuSiQue writes a decomposition answer in its canonical form, which is not
    always the form the supporting paragraph uses: the answer says "Josiah
    Fenn" and the paragraph, having introduced him, says "Fenn". A whole-string
    match finds nothing there, and the hop is then reported as unswappable when
    the entity is plainly present.
    """
    tokens = [token for token in TOKEN.findall(gold or "")
              if len(token) > 3 and token.lower() not in TOKEN_STOPWORDS]
    return max(tokens, key=len) if tokens else None


def locate(record, gold, replacement, positions):
    """Where the gold entity can actually be replaced, and how.

    Four levels, most faithful first. Anything past `literal` is a **rescue**
    and is recorded as one: the audit reports the distribution so a reader can
    see how many evidence arms rest on an exact span and how many on a looser
    match, rather than having them silently pooled.

    1. `literal` -- the gold string, whole-word, in its own supporting paragraph.
    2. `partial` -- its distinctive token in that same paragraph.
    3. `other_paragraph_literal` -- the gold string in another paragraph the
       same worker holds. The point of the swap is that the worker can no
       longer correct the planner, and a worker whose *other* paragraph carries
       the entity can still correct from that one.
    4. `other_paragraph_partial` -- the distinctive token, likewise.

    Returns (position, rewritten text, occurrences, mode) or None.

    An alias level was considered and left out: MuSiQue supplies
    `answer_aliases` only for the final answer, and `bridge_hops` excludes the
    final hop by construction, so the list is never about the span being
    searched for here.
    """
    token = distinctive(gold)
    for where, ranked in (("", positions[:1]), ("other_paragraph_", positions[1:])):
        for how, target in (("literal", gold), ("partial", token)):
            if not target:
                continue
            for position in ranked:
                if not 0 <= position < len(record["paragraphs"]):
                    continue
                text = record["paragraphs"][position]["paragraph_text"]
                rewritten, count = substitute(text, target, replacement)
                if count:
                    return position, rewritten, count, f"{where}{how}"
    return None


def share(record, support, agents=None, assignment=None):
    """The supporting paragraph first, then the rest of that worker's share.

    The evidence swap has to leave the correcting worker unable to correct.
    Editing only the labelled supporting paragraph does not achieve that when
    the same worker holds a second paragraph naming the same entity, and it
    achieves nothing at all when the gold span is not written there in the form
    the decomposition uses. Both are why `locate` is given the whole share.

    `assignment` carries the same requirement as `holder`: under the expert deal
    the share searched has to be the expert's, or `locatable` rejects hops the
    swap could have been made on and accepts ones it could not.
    """
    dealt = assignment or deal(record, agents)
    owner = holder(record, support, agents, assignment=dealt)
    if owner is None:
        return [support]
    return [support] + [position for position in dealt[owner] if position != support]


def locatable(record, gold, positions):
    """Could a swap on this hop be made at all? Used by `seed.corrupt` so a
    corruption is never planted on a hop whose evidence cannot then be edited
    -- the defect that cost four evidence arms on the first real run."""
    return locate(record, gold, "\x00sentinel\x00", positions) is not None


def workers(record, agents=None):
    return agent_names(agents or worker_count(record))


def usable(record):
    """A record this experiment can actually use.

    Needs a parseable hop count, at least one corruptible bridge hop, and at
    least as many supporting paragraphs as hops. That last one is stricter than
    the two-worker deal needs -- two shares only need two supports -- and it
    stays strict on purpose: a record with fewer supports than hops is one
    where some hop's evidence is missing from the corpus entirely, and no split
    of it can be answered by the pair. Records failing any of these are
    counted, not silently dropped -- same contract as
    `marl.dataset.load_dataset`.
    """
    try:
        hops = hop_count(record)
    except ValueError:
        return False, "unparseable_id"
    if len(supporting_positions(record)) < hops:
        return False, "too_few_supports"
    if not bridge_hops(record):
        return False, "no_bridge_hop"
    return True, None


def select(records, hops, per_hop, seed=0):
    """A fixed, reproducible subset: `per_hop` usable records for each hop count.

    Selection is a seeded shuffle of the usable records in each bin, so the
    same (dataset, hops, per_hop, seed) always yields the same questions and a
    larger `per_hop` extends the previous subset rather than resampling it.
    """
    bins = {hop: [] for hop in hops}
    excluded = {}
    for record in records:
        ok, reason = usable(record)
        if not ok:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        count = hop_count(record)
        if count in bins:
            bins[count].append(record)
    chosen = []
    for hop in hops:
        pool = sorted(bins[hop], key=lambda r: r["id"])
        random.Random(f"{seed}:{hop}").shuffle(pool)
        if len(pool) < per_hop:
            raise ValueError(
                f"Only {len(pool)} usable {hop}-hop records, asked for {per_hop}")
        chosen.extend(pool[:per_hop])
    return chosen, excluded


def episode_seed(base, record, condition, replicate=0):
    """A stable per-episode seed: same inputs, same number, on any machine.

    Python's `hash` is salted per process, so it cannot be used here -- a run
    resumed tomorrow would silently sample differently.
    """
    key = f"{base}:{record['id']}:{condition}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
