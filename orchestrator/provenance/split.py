"""Deal one MuSiQue record's paragraphs across a cohort.

Each agent receives its own share of the paragraphs, with the *supporting* ones
dealt round-robin so no agent holds the evidence for every hop. That is what
makes the bus load-bearing: an answer that needs two hops needs a message, and
an answer that appears without one did not come from the evidence.

Splitting is on paragraph *position*, computed from the public record only.
`is_supporting` is a label: it decides the deal and then disappears. The split
map stays host-side, an agent's system prompt names only its own paragraphs,
and no support label, decomposition or gold answer ever reaches an agent.

Only the dealing logic lives here. The training package's browser-page
construction (`paragraph_pages`, `split_corpus`, the per-agent index URLs) is
deliberately absent: this package gives every agent its paragraphs inline and
talks only over the bus, so there is no page to build and no `Browser` to
enforce visibility. See "What is reused" in README.md for why that reduction is
accepted.
"""


def agent_names(agents):
    return [f"agent-{i + 1}" for i in range(agents)]


def split_positions(record, agents):
    """Deal paragraph positions to agents; supporting paragraphs round-robin.

    Returns {agent: [position, ...]}. Raises when the record cannot produce a
    genuine split -- fewer supporting paragraphs than agents means somebody
    holds no evidence, and that is a task about one agent working while the
    other guesses, not a cooperative task.
    """
    if agents < 2:
        raise ValueError("A split corpus needs at least two agents")
    names = agent_names(agents)
    supporting = [i for i, p in enumerate(record["paragraphs"]) if p.get("is_supporting")]
    distractors = [i for i, p in enumerate(record["paragraphs"]) if not p.get("is_supporting")]
    if len(supporting) < agents:
        raise ValueError(
            f"Record has {len(supporting)} supporting paragraphs for {agents} agents; "
            "cannot split so that every agent holds evidence")
    assignment = {name: [] for name in names}
    for rank, position in enumerate(supporting):
        assignment[names[rank % agents]].append(position)
    # Offset the distractors so they do not shadow the supporting rotation.
    for rank, position in enumerate(distractors):
        assignment[names[(rank + 1) % agents]].append(position)
    for name in names:
        assignment[name].sort()
    return assignment
