"""The message bus: the only thing that crosses between agents.

Every model input in an episode is rendered from this log and nothing else.
That is the property the whole audit rests on -- if an agent could learn
something through a side path (a shared file, a tool result, a wiki), then
editing a message in replay would not be editing the agent's world, and a
final answer that failed to move would tell us nothing.

Messages are append-only and addressed. `to` is a single agent name or "all";
there is no broadcast-by-default, because a perturbation has to be able to
name exactly which recipients it affects.
"""
import json
import re

KINDS = ("query", "evidence", "assertion", "composition", "vote", "final")
BLOCK = re.compile(r"<msg\b([^>]*)>(.*?)</msg>", re.DOTALL)
ATTRIBUTE = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")
BROADCAST = "all"
MAX_TEXT = 4000


def parse(text, sender, step, recipients, start_id=0):
    """Pull addressed messages out of one completion.

    Strict, like `marl.toolcall.parse_tool_calls` after its review: a
    malformed block raises rather than being dropped. A dropped message is an
    agent whose peers never heard it, which looks exactly like an agent that
    chose to stay silent -- and that distinction is the experiment.
    """
    remainder = BLOCK.sub("", text)
    if "<msg" in remainder or "</msg" in remainder:
        raise ValueError("Unbalanced <msg> delimiters")
    messages = []
    for raw_attributes, body in BLOCK.findall(text):
        attributes = dict(ATTRIBUTE.findall(raw_attributes))
        kind = attributes.get("kind", "").strip()
        target = attributes.get("to", "").strip()
        if kind not in KINDS:
            raise ValueError(f"Message kind {kind!r} is not one of {KINDS}")
        if target != BROADCAST and target not in recipients:
            raise ValueError(f"Message addressed to unknown recipient {target!r}")
        body = body.strip()
        if not body:
            raise ValueError("Message body is empty")
        messages.append({"id": f"m{start_id + len(messages)}",
                         "step": step, "sender": sender, "to": target,
                         "kind": kind, "text": body[:MAX_TEXT]})
    return messages


class Bus:
    """Ordered message log with per-recipient delivery."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def __len__(self):
        return len(self.messages)

    def next_id(self):
        return len(self.messages)

    def extend(self, messages):
        self.messages.extend(messages)
        return messages

    def visible(self, agent):
        """Everything this agent may read: addressed to it, or broadcast.

        An agent does not see its own messages replayed back to it -- they are
        already in its own conversation history.
        """
        return [m for m in self.messages
                if m["sender"] != agent and m["to"] in (agent, BROADCAST)]

    def delivered(self, agent, seen):
        """New messages for `agent` since it last looked. `seen` is a set of ids."""
        return [m for m in self.visible(agent) if m["id"] not in seen]

    def by(self, sender=None, kind=None):
        return [m for m in self.messages
                if (sender is None or m["sender"] == sender)
                and (kind is None or m["kind"] == kind)]

    def copy(self):
        return Bus([dict(message) for message in self.messages])

    def dump(self, path):
        with open(path, "w") as handle:
            for message in self.messages:
                handle.write(json.dumps(message) + "\n")

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            return cls([json.loads(line) for line in handle if line.strip()])


def render(messages):
    """How delivered messages appear in an agent's next user turn."""
    if not messages:
        return "No new messages."
    lines = []
    for message in messages:
        address = "to everyone" if message["to"] == BROADCAST else f"to {message['to']}"
        lines.append(f"[{message['id']}] {message['sender']} {address} ({message['kind']}):\n"
                     f"{message['text']}")
    return "\n\n".join(lines)
