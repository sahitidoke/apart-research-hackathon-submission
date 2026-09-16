"""Run a whole condition grid and then replay every episode.

One process, three stages, each resumable only by starting a fresh output
root: collect episodes, freeze the paraphrase set, replay the perturbation
battery over each episode, and leave `batch.jsonl` and `replays.jsonl` for
`audit.py`. Nothing here scores, plots or decides anything -- that separation
is what lets the audit be re-run and re-weighted without touching a GPU again.

Episodes and replays are two passes rather than one interleaved loop, and the
paraphrase freeze is why: a paraphrase set that is generated as it is consumed
can never be inspected before it is used. Collecting every episode first means
the whole set is written, digested and open to a human's eye before a single
flip rate depends on it.
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from orchestrator.provenance import audit as audit_module
from orchestrator.provenance import dataset as dataset_module
from orchestrator.provenance import framing
from orchestrator.provenance import paraphrase as paraphrase_module
from orchestrator.provenance import replay as replay_module
from orchestrator.provenance import roles
from orchestrator.provenance import seed as seed_module
from orchestrator.provenance import session
from orchestrator.provenance import tracking
from orchestrator.provenance.audit import main as audit_main
from orchestrator.provenance.episode import (ANSWER_TOKENS, COLLABORATION_TOKENS, CONDITIONS,
                                             MAX_NEW_TOKENS, MAX_STEPS, run_episode)
from orchestrator.provenance.generator import (DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_NUM_SEQS,
                                               DEFAULT_MODEL, DEFAULT_QUANTIZATION,
                                               VLLMGenerator)
from orchestrator.provenance.records import (WORKERS, episode_seed, expert_deal,
                                             select)
from orchestrator.provenance.seed import bridge_pool

LOCALIZE = ("none", "seeded", "all")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smoke.jsonl"


def resolve_dataset(arguments):
    """The dataset a run will actually use, or a refusal.

    There is deliberately no default. The fixture used to be one, and a default
    that is three invented questions is how a smoke test ends up reported as a
    run: every command still works, nothing warns loudly enough, and the
    numbers look like numbers. Asking for it by name (`--smoke`) is the only
    way to get it, and the manifest records that a run took it.
    """
    if arguments.smoke and arguments.dataset:
        raise SystemExit("Pass --dataset or --smoke, not both")
    if arguments.smoke:
        print("SMOKE TEST: three invented questions. These numbers are not results.")
        return str(FIXTURE)
    if not arguments.dataset:
        raise SystemExit(
            "No --dataset. Use the Hub copy that keeps MuSiQue's original schema:\n"
            f"  --dataset {dataset_module.DEFAULT_SOURCE}\n"
            "or a local JSONL export, or --smoke for the three-question fixture "
            "whose numbers are not results.")
    return arguments.dataset


def session_plan(records, conditions, seeded, base_seed, planner_style, worker_style,
                 agents=WORKERS, rounds=session.DEFAULT_ROUNDS, block=session.DEFAULT_BLOCK,
                 cohorts=1, shape="declining", collaboration=COLLABORATION_TOKENS,
                 answer=ANSWER_TOKENS, max_steps=MAX_STEPS, levels=None):
    """Every episode of every cohort's session, decided before anything runs.

    A session carries three things a flat batch does not: the round a question
    is answered in, the budget that round allows, and which worker holds the
    evidence for the block it belongs to. Everything else is unchanged, and in
    particular each question is still its own episode -- see `session.py` for
    why memory does not cross them.
    """
    schedule = session.budgets(collaboration, answer, max_steps, rounds,
                               shape=shape, levels=levels)
    by_id = {record["id"]: record for record in records}
    pool = bridge_pool(records)
    planned = []
    for cohort in range(cohorts):
        for slot in session.sequence(records, rounds, block, base_seed, cohort):
            record = by_id[slot["record_id"]]
            allowance = schedule[slot["round"]]
            shared = {"planner_style": planner_style, "worker_style": worker_style,
                      "expert": slot["expert"],
                      "session": {"cohort": cohort, "round": slot["round"],
                                  "block": slot["block"], "shape": shape,
                                  **{k: v for k, v in allowance.items() if k != "round"}},
                      **{k: v for k, v in allowance.items() if k != "round"}}
            for condition in conditions:
                planned.append({"record_id": record["id"], "condition": condition,
                                "seeded_error": None, **shared,
                                "seed": episode_seed(base_seed, record, condition, cohort)})
            if seeded and "planner" in conditions:
                # The deal this episode will actually run with, not the
                # round-robin split. Under the expert deal one worker holds
                # every support, so without this the corruption names the
                # bystander as the correcting worker and the episode carries a
                # label no agent in it could have earned. See `seed.corrupt`.
                error = seed_module.corrupt(record, _corruption_rng(base_seed, record, cohort),
                                            pool, agents,
                                            assignment=expert_deal(record, slot["expert"], agents))
                if error is not None:
                    planned.append({"record_id": record["id"], "condition": "planner",
                                    "seeded_error": error, **shared,
                                    "seed": episode_seed(base_seed, record,
                                                         "planner-seeded", cohort)})
    return planned


def _corruption_rng(base_seed, record, replicate=0):
    """A corruption stream belonging to one record and nothing else.

    This used to be a single `random.Random(base_seed)` threaded through the
    whole plan, which made every planted error depend on every earlier one:
    re-planting one question's corruption -- to repair a hop whose evidence
    turned out to be unswappable, say -- silently moved the corruption of every
    question after it, so a four-episode repair would have invalidated the other
    twenty-six. Per record, a repair touches exactly what it repairs.

    It does mean the planted facts differ from those a pre-repair run drew, so
    two runs across this change are not the same corpus of corruptions and
    their deference counts are not pooled.
    """
    return random.Random(episode_seed(base_seed, record, "planner-seeded", replicate))


def plan(records, conditions, seeded, base_seed, planner_style, worker_style, agents=WORKERS):
    """Every episode this batch will run, decided before anything is generated.

    Materializing the plan first means the seeded-error labels are drawn from
    one seeded RNG over a known record list, so a rerun with the same
    arguments produces the same corruptions -- and an interrupted batch can be
    diffed against what it was supposed to do.
    """
    pool = bridge_pool(records)
    planned = []
    for record in records:
        for condition in conditions:
            planned.append({"record_id": record["id"], "condition": condition,
                            "seeded_error": None, "planner_style": planner_style,
                            "worker_style": worker_style,
                            "seed": episode_seed(base_seed, record, condition)})
        if seeded and "planner" in conditions:
            error = seed_module.corrupt(record, _corruption_rng(base_seed, record), pool, agents)
            if error is not None:
                planned.append({"record_id": record["id"], "condition": "planner",
                                "seeded_error": error, "planner_style": planner_style,
                                "worker_style": worker_style,
                                "seed": episode_seed(base_seed, record, "planner-seeded")})
    return planned


def directory_name(item):
    suffix = "-seeded" if item["seeded_error"] else ""
    found = item.get("session")
    prefix = f"c{found['cohort']}r{found['round']}-" if found else ""
    return f"{prefix}{item['record_id']}-{item['condition']}{suffix}"


def run_batch(generator, records, output, conditions=CONDITIONS,
              seeded=True, base_seed=0, planner_style="persuasive",
              worker_style="standard", max_steps=MAX_STEPS,
              max_new_tokens=MAX_NEW_TOKENS, do_replays=True, tracker=None,
              collaboration_tokens=COLLABORATION_TOKENS, answer_tokens=ANSWER_TOKENS,
              agents=WORKERS, localize="seeded", paraphrases=None, planned=None,
              resume=False, paraphrase_modes=(paraphrase_module.MESSAGE,),
              sentence_paraphrases=None, redo=()):
    """Collect, freeze, replay and index. Returns (summaries, replay results).

    `planned` lets a caller hand in a session plan; without one this is a flat
    batch of independent questions, which is what every condition other than
    the budget schedule is read against.
    """
    tracker = tracker or tracking.NullTracker()
    output = Path(output)
    # Resuming reuses the directory on purpose; without `resume` the refusal
    # stands, because two runs merged into one directory is a silently mixed
    # dataset. A three-hour run died on exactly this: the container took a
    # SIGINT with 210 episodes already committed, Modal retried the function,
    # and the retry refused the directory rather than continuing from it.
    output.mkdir(parents=True, exist_ok=bool(resume))
    if redo:
        if not resume:
            raise SystemExit("--redo only makes sense with --resume")
        _drop(output, redo)
    done_episodes, done_replays = _already_done(output) if resume else (set(), set())
    if resume and (done_episodes or done_replays):
        print(f"Resuming: {len(done_episodes)} episodes and {len(done_replays)} replays "
              "already recorded are skipped.")
    by_id = {record["id"]: record for record in records}
    if planned is None:
        planned = plan(records, conditions, seeded, base_seed, planner_style,
                       worker_style, agents)
    # `plan.json` describes what the original collection set out to do, and a
    # resumed pass must not claim to rewrite that history: an episode already on
    # disk keeps the corruption it actually ran with, which a freshly drawn plan
    # may no longer name. The resumed plan is written beside it instead.
    name = "plan.json"
    if resume and (output / name).exists():
        name = f"plan-resume-{time.strftime('%Y%m%d-%H%M%S')}.json"
    (output / name).write_text(json.dumps(planned, indent=2) + "\n")

    summaries = []
    for item in planned:
        record = by_id[item["record_id"]]
        directory = output / "episodes" / directory_name(item)
        relative = str(directory.relative_to(output))
        if relative in done_episodes:
            summaries.append(_recorded(output, relative))
            continue
        shutil.rmtree(directory, ignore_errors=True)
        expert = item.get("expert")
        summary = run_episode(generator, record, item["condition"], directory,
                              seed=item["seed"],
                              max_steps=item.get("max_steps", max_steps),
                              planner_style=item["planner_style"],
                              worker_style=item["worker_style"],
                              seeded_error=item["seeded_error"],
                              max_new_tokens=max_new_tokens, agents=agents,
                              collaboration_tokens=item.get("collaboration_tokens",
                                                            collaboration_tokens),
                              answer_tokens=item.get("answer_tokens", answer_tokens),
                              assignment=expert_deal(record, expert, agents) if expert else None,
                              expert=expert, session=item.get("session"))
        summary["directory"] = str(directory.relative_to(output))
        summaries.append(summary)
        _append(output / "batch.jsonl", summary)
        tracker.log(tracking.episode_metrics(summary), step=len(summaries))
    tracker.table("episodes", [{k: v for k, v in s.items()
                                if not isinstance(v, (dict, list))} for s in summaries])
    if not do_replays:
        return summaries, []
    return summaries, _replay_pass(generator, by_id, planned, summaries, output, tracker,
                                   localize, paraphrases, agents, done_replays,
                                   paraphrase_modes, sentence_paraphrases)


def _recorded(output, relative):
    """The summary a previous attempt already wrote for this episode."""
    for row in audit_module.read_jsonl(output / "batch.jsonl"):
        if row["directory"] == relative:
            return row
    raise KeyError(relative)


def _paraphrase_lookup(generator, output, summaries, mode, reuse=None):
    """(lookup, path, payload) for one paraphrase mode, or three Nones.

    Three ways in, and the middle one is what makes a resumed run possible at
    all: `freeze` refuses to overwrite, so a second pass over a directory that
    already holds a frozen set has to *load* it rather than rebuild it. Before
    this, resuming a run that had got as far as its paraphrases died on
    `FileExistsError` with every episode already paid for.
    """
    if reuse:
        # A reviewed set from an earlier batch. Its digest is checked on load,
        # so a hand-edited file fails here rather than at the point where its
        # numbers are already in a table.
        accepted, payload = paraphrase_module.load(reuse)
        return (paraphrase_module.Frozen(accepted, generator,
                                         mode=payload.get("mode", mode),
                                         rejected=paraphrase_module.refusals(payload)),
                Path(reuse), payload)
    if not any(summary["condition"] == "planner" for summary in summaries):
        return None, None, None
    path = paraphrase_module.default_path(output, mode)
    if path.exists():
        accepted, payload = paraphrase_module.load(path)
        return paraphrase_module.Frozen(
            accepted, generator, mode=payload.get("mode", mode),
            rejected=paraphrase_module.refusals(payload)), path, payload
    path, payload = paraphrase_module.freeze(generator, output, mode=mode)
    accepted = {text: entry["rewrite"] for text, entry in payload["entries"].items()
                if entry["accepted"]}
    return paraphrase_module.Frozen(accepted, generator, mode=mode), path, payload


def _announce(mode, payload):
    pending, sampled = paraphrase_module.unreviewed(payload)
    share = payload.get("rewritten_share")
    coverage = f", {share} of sentences reworded" if share is not None else ""
    print(f"Paraphrases ({mode}): {payload['accepted']} accepted, "
          f"{payload['rejected']} rejected, {pending}/{sampled} of the spot-check "
          f"sample unreviewed{coverage}.")


def _replay_pass(generator, by_id, planned, summaries, output, tracker, localize,
                 paraphrases, agents, done_replays=frozenset(),
                 paraphrase_modes=(paraphrase_module.MESSAGE,), sentence_paraphrases=None):
    """Freeze the paraphrase sets, then replay every episode against them."""
    lookups, frozen = {}, {}
    reuse = {paraphrase_module.MESSAGE: paraphrases,
             paraphrase_module.SENTENCE: sentence_paraphrases}
    for mode in paraphrase_modes:
        lookup, path, payload = _paraphrase_lookup(generator, output, summaries, mode,
                                                   reuse.get(mode))
        if lookup is None:
            continue
        lookups[mode], frozen[mode] = lookup, (path, payload)
        _announce(mode, payload)

    # Deterministic, so it is written from the bus rather than generated, and
    # written whether or not any episode ends up using it: what was removed
    # from each planner message has to be readable by hand. See `framing.py`
    # for why this is not subject to the freeze rule the rewrites are.
    if any(summary["condition"] == "planner" for summary in summaries):
        _, stripped = framing.record(output, paraphrase_module.planner_texts(output))
        print(f"Framing strip: {stripped['accepted']} of "
              f"{stripped['accepted'] + stripped['rejected']} planner messages carry an "
              "authority marker.")

    # The per-hop closed-book screen runs as its own condition in pass one, so
    # by now it is known which bridge facts the model already had. A swap that
    # targets one of those measures recall, not provenance, and the arm is
    # omitted rather than run.
    known = audit_module.known_hops(summaries)
    done_by_directory = defaultdict(set)
    for directory, name in done_replays:
        done_by_directory[directory].add(name)

    replays = []
    for step, (item, summary) in enumerate(zip(planned, summaries), start=1):
        record = by_id[item["record_id"]]
        localizing = localize == "all" or (localize == "seeded" and bool(item["seeded_error"]))
        # Per arm, not per episode. Skipping the whole episode as soon as it had
        # *any* replay on file is what made adding a perturbation to a finished
        # run impossible: the new arm was skipped along with the old ones, and
        # the only way to get it was to regenerate all 429 replays.
        results, omissions = replay_module.replay_all(
            generator, record, output / summary["directory"],
            lookup=lookups.get(paraphrase_module.MESSAGE), localize=localizing,
            known_hops=known.get(item["record_id"], ()),
            sentence_lookup=lookups.get(paraphrase_module.SENTENCE),
            done=done_by_directory.get(summary["directory"], ()))
        for result in results:
            result["directory"] = summary["directory"]
            result["seeded"] = bool(item["seeded_error"])
            replays.append(result)
            _append(output / "replays.jsonl", result)
            tracker.log(tracking.replay_metrics(result), step=step)
        for omission in omissions:
            _append(output / "omissions.jsonl",
                    {**omission, "directory": summary["directory"],
                     "condition": summary["condition"]})
    _reconcile_omissions(output)
    for mode, (path, _) in frozen.items():
        updated = lookups[mode].record(path)
        if updated:
            print(f"Paraphrases ({mode}): {updated['late']} minted during replay for "
                  f"planner messages the freeze could not have seen; "
                  f"{updated['accepted']} accepted, {updated['rejected']} rejected.")
    return replays


def _reconcile_omissions(output):
    """Make the omission index say what is true *now*, not what was true once.

    A resumed pass deliberately retries arms that were omitted before -- that is
    the whole point, since a new paraphrase mode or a widened span search can
    make one runnable. Appending blindly leaves two kinds of lie behind:

    * an arm omitted twice gets two rows, and `arm_coverage` counts it twice;
    * an arm that was omitted and has since *run* keeps its stale "could not
      run" row, so the same arm is counted as both.

    Either one corrupts the one table this whole change exists to make
    trustworthy. So the index is rebuilt from the two facts on disk: an arm with
    a replay row is not an omission, and otherwise the most recent row wins.
    Derivable from the files alone, and idempotent however many passes ran.
    """
    path = output / "omissions.jsonl"
    if not path.exists():
        return
    ran = {(row["directory"], row["perturbation"])
           for row in audit_module.read_jsonl(output / "replays.jsonl")}
    latest = {}
    for row in audit_module.read_jsonl(path):
        key = (row["directory"], row["perturbation"])
        if key not in ran:
            latest[key] = row
    path.write_text("".join(json.dumps(row) + "\n" for row in latest.values()))


INDEXES = ("batch.jsonl", "replays.jsonl", "omissions.jsonl")


def _drop(output, directories):
    """Forget named episodes so a resumed pass re-runs them from scratch.

    The repair path for an episode that completed but should not have -- a
    seeded error planted on a hop whose evidence then turned out to be
    unswappable, which cost four evidence arms on the first real run. Those
    episodes need a fresh corruption, and a fresh corruption means a fresh
    episode, not a fresh replay.

    Every index is backed up before it is filtered. This deletes work that cost
    GPU time, and a run directory is the only copy of it.
    """
    directories = set(directories)
    removed = {}
    for name in INDEXES:
        path = output / name
        if not path.exists():
            continue
        rows = audit_module.read_jsonl(path)
        keep = [row for row in rows if row.get("directory") not in directories]
        if len(keep) == len(rows):
            continue
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("".join(json.dumps(row) + "\n" for row in keep))
        removed[name] = len(rows) - len(keep)
    for directory in sorted(directories):
        shutil.rmtree(output / directory, ignore_errors=True)
    print(f"Redo: dropped {len(directories)} episode(s) and their rows "
          f"({removed or 'nothing recorded'}); previous indexes kept as .bak.")
    return removed


def _already_done(output):
    """What a previous attempt at this directory finished.

    Read back from the two indexes rather than from the directory tree: those
    are appended only after an episode or replay is complete, so a half-written
    episode directory is not mistaken for a finished one. It is re-run and its
    directory removed first.
    """
    episodes = {row["directory"] for row in audit_module.read_jsonl(output / "batch.jsonl")}
    replays = {(row["directory"], row["perturbation"])
               for row in audit_module.read_jsonl(output / "replays.jsonl")}
    return episodes, replays


def _append(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",
                        help="A Hub spec such as "
                             f"{dataset_module.DEFAULT_SOURCE}, or a path to a "
                             "MuSiQue-Ans JSONL export. No default: see --smoke.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the bundled three-question fixture instead of a "
                             "dataset. It exercises the pipeline; its numbers are "
                             "not results, and the manifest says so.")
    parser.add_argument("--output", help="Fresh run directory. Must not exist.")
    parser.add_argument("--output-root",
                        help="Parent directory to create a uniquely named run under. "
                             "Use this instead of --output in a sweep, where every "
                             "trial needs its own directory and none can be named "
                             "in advance. The name comes from WANDB_RUN_ID when the "
                             "sweep agent sets one, so a directory on disk can be "
                             "matched back to its wandb run.")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="`closed_book` and `isolated` are the competence screens: "
                             "drop them and the deference analysis has nothing to "
                             "exclude parametrically-answerable or singly-solvable "
                             "questions with.")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Workers per collective. The design is {WORKERS}; anything "
                             "else is the deferred worker-count study and is not "
                             "comparable with a run at the default.")
    parser.add_argument("--session", action="store_true",
                        help="Run cohorts through a question sequence in rounds, with "
                             "one worker holding all the evidence per block and a "
                             "per-round budget that declines. Without it, questions "
                             "are independent and the split is the usual one.")
    parser.add_argument("--rounds", type=int, default=session.DEFAULT_ROUNDS,
                        help="Questions per cohort session.")
    parser.add_argument("--block", type=int, default=session.DEFAULT_BLOCK,
                        help="Consecutive rounds one worker stays the expert for. "
                             "The other worker holds only distractors meanwhile.")
    parser.add_argument("--cohorts", type=int, default=1,
                        help="Independent sessions. Block order and question order are "
                             "counterbalanced across them.")
    parser.add_argument("--budget-shape", default="declining", choices=session.SHAPES,
                        help="`declining` is the pressure arm; `flat` is its control, "
                             "at the schedule midpoint and the same session total. "
                             "Round index and budget are confounded within either arm "
                             "alone, so neither supports a scarcity claim by itself.")
    parser.add_argument("--budget-start", type=float, default=session.DEFAULT_START,
                        help="First round's share of the uncapped per-episode budget.")
    parser.add_argument("--budget-end", type=float, default=session.DEFAULT_END,
                        help="Last round's share. See --calibrate-from for the "
                             "percentile-based schedule the plan actually asks for.")
    parser.add_argument("--calibrate-from",
                        help="A finished uncapped run. The schedule is then p90 down to "
                             "p25 of what that run really spent per episode, rather "
                             "than fractions somebody picked.")
    parser.add_argument("--localize", default="seeded", choices=LOCALIZE,
                        help="Per-message planner ablation, to find the single message "
                             "whose removal flips the answer. Costs one extra replay "
                             "per planner message, so it defaults to the labelled "
                             "(seeded) episodes, which are the ones it is scored on.")
    parser.add_argument("--paraphrases",
                        help="Reuse a frozen, human-reviewed paraphrases.json from an "
                             "earlier batch instead of generating a new set. Its digest "
                             "is verified on load.")
    parser.add_argument("--paraphrase-mode", default=paraphrase_module.MESSAGE,
                        help="Comma-separated: `message` rewrites each planner message "
                             "whole, `sentence` rewrites it a sentence at a time and "
                             "keeps verbatim any sentence whose rewrite fails. They are "
                             "separate arms with separate names, never pooled. The "
                             "default is unchanged so an existing run's numbers stay "
                             "reproducible; `sentence` is the one that recovers the "
                             "arms a long fact-dense assertion loses.")
    parser.add_argument("--paraphrases-sentence",
                        help="The same reuse, for the sentence-mode set.")
    parser.add_argument("--redo",
                        help="Comma-separated episode directories to forget and re-run, "
                             "with --resume. For an episode that completed but should "
                             "not have -- a corruption planted on a hop whose evidence "
                             "cannot be swapped. Indexes are backed up to .bak first.")
    parser.add_argument("--hops", default="2,3,4")
    parser.add_argument("--per-hop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-seeded-errors", action="store_true")
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a run whose directory already exists, skipping "
                             "episodes and replays it already recorded. Without this a "
                             "existing directory is refused, because two runs merged "
                             "into one is a silently mixed dataset.")
    parser.add_argument("--planner-style", default="persuasive",
                        choices=tuple(roles.PLANNERS),
                        help="`persuasive` is the natural arm. `directive` prices "
                             "speed and tells the planner to have conclusions "
                             "confirmed rather than re-opened; its deference rate "
                             "is induced and is not natural propensity.")
    parser.add_argument("--worker-style", default="standard",
                        choices=tuple(roles.WORKERS),
                        help="`standard` is the natural arm. `cooperative` scores "
                             "the team together and rewards converging; "
                             "`deferential` instructs yielding outright. Both are "
                             "induced conditions.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Cap on a single response. Applies within every phase "
                             "budget, so no one turn can consume the episode.")
    parser.add_argument("--collaboration-tokens", type=int, default=COLLABORATION_TOKENS,
                        help="Native generated tokens for the whole episode's "
                             "discussion, shared by every agent.")
    parser.add_argument("--answer-tokens", type=int, default=ANSWER_TOKENS,
                        help="Separate budget for the final composition, so a long "
                             "discussion can never leave an episode unable to answer.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION,
                        help="Leave unset for a checkpoint that declares its own "
                             "scheme, which every pre-quantized Qwen3.8 build does. "
                             "Naming a different one than the checkpoint carries "
                             "fails at load.")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS,
                        help="Concurrent sequences. Capped on this architecture by how "
                             "many per-sequence Gated DeltaNet states fit beside the "
                             "weights; vLLM's own default of 256 does not fit and fails "
                             "at startup.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 keeps replay deterministic; above 0 every flip rate "
                             "carries a resampling floor you must estimate separately.")
    parser.add_argument("--wandb", action="store_true",
                        help="Track this batch. Nothing is trained here, so this is "
                             "experiment tracking, not optimization; see sweep.yaml.")
    parser.add_argument("--wandb-project", default="agent-swarming-provenance")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--audit", action="store_true",
                        help="Run the audit in the same process, so a sweep gets its "
                             "objective metric without a second invocation.")
    return parser


def resolve_output(arguments):
    if bool(arguments.output) == bool(arguments.output_root):
        raise SystemExit("Pass exactly one of --output or --output-root")
    if arguments.output:
        return Path(arguments.output)
    unique = os.environ.get("WANDB_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    return Path(arguments.output_root) / unique


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    conditions = [c.strip() for c in arguments.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"Unknown conditions: {sorted(unknown)}")
    hops = [int(h) for h in arguments.hops.split(",") if h.strip()]
    modes = [m.strip() for m in arguments.paraphrase_mode.split(",") if m.strip()]
    unknown_modes = set(modes) - set(paraphrase_module.MODES)
    if unknown_modes:
        raise SystemExit(f"Unknown paraphrase modes: {sorted(unknown_modes)}")
    source = resolve_dataset(arguments)
    records, excluded = select(dataset_module.load(source), hops,
                               arguments.per_hop, arguments.seed)
    generator = VLLMGenerator(
        model=arguments.model, quantization=arguments.quantization,
        max_model_len=arguments.max_model_len, temperature=arguments.temperature,
        tensor_parallel_size=arguments.tensor_parallel_size,
        gpu_memory_utilization=arguments.gpu_memory_utilization, seed=arguments.seed,
        max_num_seqs=arguments.max_num_seqs)
    output = resolve_output(arguments)
    run = tracking.tracker(arguments.wandb, arguments.wandb_project,
                           arguments.wandb_name or output.name, vars(arguments),
                           arguments.wandb_mode, job_type="batch")
    planned = None
    if arguments.session:
        levels = None
        if arguments.calibrate_from:
            spent = [row["usage"]["generated_total"]
                     for row in audit_module.read_jsonl(
                         Path(arguments.calibrate_from) / "batch.jsonl")
                     if (row.get("usage") or {}).get("generated_total")]
            levels = session.levels_from_usage(spent, arguments.rounds)
            if levels is None:
                raise SystemExit(f"{arguments.calibrate_from} records no token usage "
                                 "to calibrate a schedule from")
            print(f"Schedule calibrated from {len(spent)} episodes: {levels}")
        planned = session_plan(
            records, conditions, not arguments.no_seeded_errors, arguments.seed,
            arguments.planner_style, arguments.worker_style, arguments.workers,
            rounds=arguments.rounds, block=arguments.block, cohorts=arguments.cohorts,
            shape=arguments.budget_shape, collaboration=arguments.collaboration_tokens,
            answer=arguments.answer_tokens, max_steps=arguments.max_steps, levels=levels)
    try:
        summaries, replays = run_batch(
            generator, records, output, conditions=conditions,
            seeded=not arguments.no_seeded_errors, base_seed=arguments.seed,
            planner_style=arguments.planner_style, worker_style=arguments.worker_style,
            max_steps=arguments.max_steps, max_new_tokens=arguments.max_new_tokens,
            do_replays=not arguments.no_replays, tracker=run,
            collaboration_tokens=arguments.collaboration_tokens,
            answer_tokens=arguments.answer_tokens, agents=arguments.workers,
            localize=arguments.localize, paraphrases=arguments.paraphrases,
            planned=planned, resume=arguments.resume,
            paraphrase_modes=modes,
            sentence_paraphrases=arguments.paraphrases_sentence,
            redo=[d.strip() for d in (arguments.redo or "").split(",") if d.strip()])
        if arguments.audit:
            audit_main(["--run-root", str(output)], tracker=run)
    except Exception:
        run.finish(exit_code=1)
        raise
    (output / "manifest.json").write_text(json.dumps(
        {"dataset": source, "smoke": bool(arguments.smoke), "excluded": excluded,
         "paraphrase_modes": modes, "redo": arguments.redo,
         "records": [record["id"] for record in records], "hops": hops,
         "per_hop": arguments.per_hop, "conditions": conditions,
         "episodes": len(summaries), "replays": len(replays),
         "session": ({"rounds": arguments.rounds, "block": arguments.block,
                      "cohorts": arguments.cohorts, "shape": arguments.budget_shape,
                      "calibrated_from": arguments.calibrate_from,
                      "sequence_overlap": session.overlap(records)}
                     if arguments.session else None),
         "arguments": vars(arguments)}, indent=2) + "\n")
    print(f"{len(summaries)} episodes, {len(replays)} replays -> {output}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
