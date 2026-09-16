"""Official MuSiQue answer scoring: normalization, token F1, exact match.

The normalization order is MuSiQue's own -- lowercase, delete punctuation, drop
articles -- and every number this package reports about answer correctness goes
through it. Keeping the official order matters because the competence screens
decide which questions enter the deference analysis at all: a looser or stricter
match would change the denominator, not just the score.

Named `scoring` rather than `reward`: nothing in this package is trained. These
functions grade a finished answer against gold, and there is no policy to give
the grade back to. The training package's reward shaping -- team/solo/share/cost
weights, cohort aggregation -- is deliberately absent, because its inputs
(information transfers, step budgets) do not exist here.

`extract_answer` here falls back to the whole completion when no tag is present,
which is right when an episode has ended and something must be scored. That is
*not* the same as `episode.extract_answer`, which returns None instead, because
during an episode the absence of a tag is how an agent says it is still working.
"""
import re
import string
from collections import Counter

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text):
    """Official MuSiQue order: lowercase, delete punctuation, drop articles."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(ARTICLES.sub(" ", text).split())


def token_f1(prediction, gold):
    predicted, expected = normalize(prediction).split(), normalize(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, gold):
    return float(normalize(prediction) == normalize(gold))


def golds(record):
    answers = [record["answer"], *record.get("answer_aliases", [])]
    return [answer for answer in answers if isinstance(answer, str) and answer.strip()]


def extract_answer(response):
    """The tagged span the prompt asks for, or the whole response.

    The fallback scores worse on F1 -- it carries the surrounding prose as false
    positives -- so a model that ignores the format is penalised without needing
    a separate format term.
    """
    if not isinstance(response, str):
        return ""
    matches = ANSWER.findall(response)
    return matches[-1].strip() if matches else response.strip()


def answer_score(response, record):
    """Score one response against a record's gold answer and its aliases.

    Aliases are maxed over rather than averaged: MuSiQue lists them as equally
    correct surface forms of one answer, so matching any of them is a match.
    """
    answer = extract_answer(response)
    if not answer:
        return {"answer": "", "f1": 0.0, "exact_match": 0.0}
    references = golds(record)
    return {"answer": answer,
            "f1": max((token_f1(answer, gold) for gold in references), default=0.0),
            "exact_match": max((exact_match(answer, gold) for gold in references), default=0.0)}
