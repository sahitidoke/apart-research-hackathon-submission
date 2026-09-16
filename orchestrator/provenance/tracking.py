"""Weights & Biases tracking, and a no-op that stands in for it.

Call sites are unconditional: `tracker(...)` always returns something with the
same methods, so `batch.py` and `audit.py` have no `if wandb:` branches and
the offline test suite exercises the same code path a tracked run takes.

Worth being precise about what wandb is doing here, because this package
trains nothing and has no loss. There are no weights to optimize. Two real
uses remain:

* **Tracking.** One run per batch, carrying the config, a live per-episode
  stream, and the audit's tables. This is the honest default.
* **Sweeps.** The knobs worth searching are the *experiment's* -- prompt
  ladder arm, step budget, hop mix -- against the audit's own AUC, i.e. "under
  which conditions is this audit actually discriminative?". `sweep.yaml`
  does that, and carries the warning it needs to carry.
"""
NO_WANDB = ("wandb is not installed. Install it with `pip install wandb` "
            "or drop --wandb.")


class NullTracker:
    """Does nothing, successfully. The default when tracking is off."""

    enabled = False

    def log(self, values, step=None):
        pass

    def table(self, name, rows):
        pass

    def curve(self, name, points, x, y):
        pass

    def summary(self, values):
        pass

    def finish(self, exit_code=0):
        pass


class WandbTracker:
    """One wandb run. Tables are logged whole rather than streamed, because
    the audit produces them once and re-logging a growing table every episode
    is how a run's storage bill outgrows its own results."""

    enabled = True

    def __init__(self, project, name, config, mode, job_type):
        # Imported here, not at module scope, for the same reason as
        # `generator.VLLMGenerator`: the rest of this package is stdlib-only so
        # its tests need nothing installed, and a top-level import would make
        # `batch` unimportable without wandb even for an untracked run.
        try:
            import wandb
        except ImportError as error:
            raise SystemExit(NO_WANDB) from error
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config,
                              mode=mode, job_type=job_type)

    def log(self, values, step=None):
        self.run.log(values, step=step)

    def table(self, name, rows):
        if not rows:
            return
        columns = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
        table = self.wandb.Table(columns=columns,
                                 data=[[row.get(key) for key in columns] for row in rows])
        self.run.log({name: table})

    def curve(self, name, points, x, y):
        if not points:
            return
        table = self.wandb.Table(columns=[x, y], data=[[p[x], p[y]] for p in points])
        self.run.log({name: self.wandb.plot.line(table, x, y, title=name)})

    def summary(self, values):
        for key, value in values.items():
            self.run.summary[key] = value

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)


def tracker(enabled=False, project="agent-swarming-provenance", name=None,
            config=None, mode="online", job_type="batch"):
    if not enabled:
        return NullTracker()
    return WandbTracker(project, name, config or {}, mode, job_type)


def episode_metrics(summary):
    """The per-episode scalars worth a live chart while a batch is running."""
    followed = summary.get("followed_planner_error") or {}
    values = {f"{summary['condition']}/exact_match": summary["exact_match"],
              f"{summary['condition']}/f1": summary["f1"],
              f"{summary['condition']}/messages": summary["messages"],
              "hops": summary["hops"],
              "complete": float(summary["status"] == "complete")}
    if summary.get("seeded_error"):
        # A live chart of "wrong under a seeded error", which is not the same
        # thing as the audit's deference label -- that one also requires the
        # corruption to be traceable, and is computed in `audit.dictated` once
        # the episode is over. Named for what it is so the two are not read as
        # the same number.
        values["seeded/wrong"] = float(summary["exact_match"] != 1.0)
        values["seeded/worker_pushed_back"] = float(bool(followed.get("contested_by_workers")))
        values["seeded/worker_restated"] = float(bool(followed.get("restated_by_workers")))
    return values


def replay_metrics(result):
    return {f"replay/{result['perturbation']}/flipped": float(result["flipped"]),
            f"replay/{result['perturbation']}/cache_misses": result["cache_misses"]}


def audit_summary(result):
    """The scalars that make one audited run comparable to another.

    `audit/auc` is the sweep objective: how well the provenance score separates
    seeded episodes the collective deferred on from ones it derived past.
    """
    values = {"audit/auc": result["roc"]["auc"],
              "audit/labelled_deferred": result["roc"]["positives"],
              "audit/labelled_derived": result["roc"]["negatives"],
              # Health and retention travel with the AUC on purpose: an AUC over
              # a handful of retained questions, or one computed on a run whose
              # replays diverged, is not the same result as one that is not.
              "health/identity_diverged_rate": result["health"]["diverged_rate"],
              "health/identity_unchecked": result["health"]["unchecked"],
              "retention/questions": result["competence"]["questions"],
              "retention/retained": result["competence"].get("retained", 0),
              "retention/answerable_closed_book":
                  result["competence"].get("answerable_closed_book", 0),
              "retention/solved_by_one_worker":
                  result["competence"].get("solved_by_one_worker", 0),
              "retention/solo_baseline_failed":
                  result["competence"].get("solo_baseline_failed", 0)}
    # An AUC is only as good as the share of labelled episodes that carried a
    # score at all. The first run's 0.333 was computed over half its positive
    # class, and nothing in the summary said so -- so coverage travels with the
    # objective now, and a sweep trial that scored fewer episodes is visible as
    # such rather than just as a different number.
    scored = [row for row in result.get("episodes") or []
              if row.get("eligible") and row.get("label") is not None]
    values["audit/labelled_eligible"] = len(scored)
    values["audit/scored_episodes"] = sum(1 for row in scored
                                          if row.get("score") is not None)
    values["audit/scored_share"] = (round(values["audit/scored_episodes"] / len(scored), 4)
                                    if scored else None)
    for stratum in result.get("roc_strata") or []:
        if stratum["group"] in ("variant", "override"):
            slug = stratum["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            values[f"audit/auc/{slug}"] = stratum["roc"]["auc"]
    for row in result.get("arm_coverage") or []:
        values[f"arms/{row['perturbation']}/coverage"] = row["coverage"]
        values[f"arms/{row['perturbation']}/omitted"] = row["omitted"]
    for row in result.get("override") or []:
        values[f"override/{row['hops']}hop/override"] = row["override"]
        values[f"override/{row['hops']}hop/capitulation"] = row["capitulation"]
        values[f"override/{row['hops']}hop/pushback_inert"] = row["pushback_inert_rate"]
        values[f"override/{row['hops']}hop/control_inert"] = row["control_inert_rate"]
    for row in result["localization"]:
        values[f"localization/{row['hops']}hop/unique_rate"] = row["unique_flip_point_rate"]
        values[f"localization/{row['hops']}hop/accuracy"] = row["flip_point_accuracy"]
    for row in result.get("workers") or []:
        # The agent that held the contradicting paragraph is the one whose
        # deference is the finding; the other worker is reported separately
        # rather than averaged into it.
        role = "holder" if row["holds_contradicting_evidence"] else "bystander"
        values[f"workers/{role}/planner_sensitivity"] = row["mean_planner_sensitivity"]
        values[f"workers/{role}/evidence_sensitivity"] = row["mean_evidence_sensitivity"]
        values[f"workers/{role}/inert_rate"] = row["inert_rate"]
    for row in result.get("natural_conflicts") or []:
        values[f"natural/{row['hops']}hop/conflict_rate"] = row["episodes_with_conflict_rate"]
        values[f"natural/{row['hops']}hop/pushed_back"] = row["worker_pushed_back_rate"]
    for row in result["accuracy"]:
        values[f"accuracy/{row['condition']}/{row['hops']}hop/exact_match"] = row["exact_match"]
        values[f"accuracy/{row['condition']}/{row['hops']}hop/f1"] = row["f1"]
    for row in result["deference"]:
        values[f"deference/{row['hops']}hop/rate"] = row["deference_rate"]
        values[f"deference/{row['hops']}hop/pushed_back"] = row["worker_pushed_back_rate"]
        values[f"deference/{row['hops']}hop/restated"] = row["worker_restated_evidence_rate"]
        values[f"deference/{row['hops']}hop/unattributed"] = row["unattributed_errors"]
    for row in result["ladder"]:
        arm = f"{row['planner_style']}-{row['worker_style']}"
        values[f"ladder/{arm}/deference_rate"] = row["deference_rate"]
        values[f"ladder/{arm}/flagged_rate"] = row["flagged_rate"]
        values[f"ladder/{arm}/inert_rate"] = row["inert_rate"]
    scores = [row["score"] for row in result["episodes"] if row.get("score") is not None]
    if scores:
        values["audit/mean_provenance_score"] = round(sum(scores) / len(scores), 4)
        values["audit/scored_episodes"] = len(scores)
    return values
