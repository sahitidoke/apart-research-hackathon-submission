"""Getting MuSiQue-Ans in, from the Hub or from a file, with the schema checked.

    --dataset hf:bdsaglam/musique:answerable:validation   # the default source
    --dataset /root/musique_ans_v1.0_dev.jsonl            # a local export

The Hub copy at `bdsaglam/musique` (config `answerable`) preserves the original
AllenAI release field for field -- `id` carrying the hop count, `paragraphs`
with `is_supporting`, and `question_decomposition` with `paragraph_support_idx`
-- which is the whole reason it is usable here. Several other mirrors flatten
the paragraphs or drop the decomposition, and this package would then reject
every record for reasons that look like a bug in `records.usable`.

So the schema is checked on load and a mismatch raises, naming the missing
fields. The alternative is worse than a crash: `select` would count every
record as unusable, report a tidy exclusion tally, and hand back an empty run.
"""
import json
from pathlib import Path

HUB_PREFIX = "hf:"
DEFAULT_SOURCE = "hf:bdsaglam/musique:answerable:validation"
# Exactly the fields `records.py`, `seed.py` and `roles.py` read. Kept here as
# data rather than as a docstring so the check cannot drift from the readers.
REQUIRED = ("id", "question", "answer", "paragraphs", "question_decomposition")
PARAGRAPH_FIELDS = ("title", "paragraph_text", "is_supporting")
STEP_FIELDS = ("question", "answer", "paragraph_support_idx")


def parse_source(source):
    """('hub', (name, config, split)) or ('file', path)."""
    if not source.startswith(HUB_PREFIX):
        return "file", source
    parts = source[len(HUB_PREFIX):].split(":")
    if not parts[0]:
        raise ValueError(f"{source!r} names no dataset")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else "answerable"
    split = parts[2] if len(parts) > 2 and parts[2] else "validation"
    return "hub", (name, config, split)


def problems(record):
    """Everything about this record the rest of the package cannot work with."""
    found = [field for field in REQUIRED if field not in record]
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        found.append("paragraphs[]")
    else:
        found += [f"paragraphs[].{field}" for field in PARAGRAPH_FIELDS
                  if not isinstance(paragraphs[0], dict) or field not in paragraphs[0]]
    steps = record.get("question_decomposition")
    if not isinstance(steps, list) or not steps:
        found.append("question_decomposition[]")
    else:
        found += [f"question_decomposition[].{field}" for field in STEP_FIELDS
                  if not isinstance(steps[0], dict) or field not in steps[0]]
    return found


def check(records, source):
    """Raise unless the first record has every field the readers need.

    Checked on one record rather than all of them on purpose: a single record
    missing a field is a bad record, and `records.usable` already counts those.
    A *dataset* with the wrong shape fails on its first row, and failing there
    is the difference between a clear error and an empty run.
    """
    if not records:
        raise ValueError(f"{source} holds no records")
    found = problems(records[0])
    if found:
        raise ValueError(
            f"{source} is not MuSiQue in its original schema. Missing: "
            f"{', '.join(found)}. The first record has "
            f"{sorted(records[0])}. Use a mirror that preserves the AllenAI "
            f"fields -- {DEFAULT_SOURCE} does.")
    return records


def from_hub(name, config, split):
    # Imported here, not at module scope: the rest of this package is
    # stdlib-only so its tests need nothing installed, and a top-level import
    # would make `batch` unimportable without `datasets` even for a local file.
    from datasets import load_dataset

    split_data = load_dataset(name, config, split=split)
    return [dict(row) for row in split_data]


def from_file(path):
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as error:
            raise ValueError(f"{path}: line {number}: {error}") from error
    return records


def load(source):
    """Records from a Hub spec or a JSONL path, schema-checked."""
    kind, target = parse_source(source)
    records = from_hub(*target) if kind == "hub" else from_file(target)
    return check(records, source)
