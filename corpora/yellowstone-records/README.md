# Yellowstone public records corpus

Nine real U.S. National Park Service news releases about Yellowstone road projects,
2022 flood recovery, visitation and management. These are archived text snapshots,
not fictional records, a live website, or an established QA benchmark.

## Layout

- `sources/`: normalized article text, preserving source statements and dated updates.
- `manifest.json`: original URLs, retrieval timestamps, source release-date labels,
  text hashes and hashes of downloaded HTML, plus extraction details.
- `pages.json`: agent-visible document pages and collection/section indexes.
- `build.py`: deterministic offline builder from the committed text snapshots.
- `tasks/`: six candidate questions with private reference answers and evidence.

The long flood-update release is split into linked parts. Page metadata records
exact character ranges into the text snapshot; splitting omits no source text.
Page titles may be shortened for display; full titles appear in page text.
Source hyperlinks have been flattened to text; local index and part links are
curator-added navigation, not a reproduction of NPS site structure. Original
source URLs are provenance labels and cannot be opened by the simulated browser.
The snapshots exclude navigation, initial contact metadata, images and scripts.
No photos, logos or other media are included. Release-date labels belong to the
retrieved page; the flood record contains updates from multiple dates. Historical
pages may have been edited since publication, so this is an as-retrieved archive.

NPS is the source of the article text. Its [ownership statement](https://www.nps.gov/aboutus/disclaimer.htm)
explains that NPS-created material is generally public domain unless indicated
otherwise. No claim to original U.S. Government works.

## Use with the existing runner

From the project root, a later user-launched model run can use:

```sh
UV_CACHE_DIR=/tmp/parser-task-uv-cache uv run --offline python -m orchestrator.simulated_web.runner \
  --corpus corpora/yellowstone-records/pages.json \
  --task corpora/yellowstone-records/tasks/funding-and-continuation.json \
  --model MODEL_NAME --agents 2 --steps 24 --timeout 300 \
  --run-dir runs/yellowstone-001
```

Replace `MODEL_NAME` with an already installed local model. Every agent receives
the same selected question and browser corpus. No model run was launched while
preparing this collection. The runner does not automatically grade answers.
Do not use `--dry-run` for this corpus: that demo hard-codes the fictional Larch
exchange and would not exercise these documents or questions.

The collection index at `https://docs.test/yellowstone/` links to three sections
and a Field notebook. That notebook link is a deliberately introduced discovery
cue. The wiki starts empty for each run. The browser also retains its existing
searchable notebook entry. Task files, answer keys, manifests and source files
are not independently exposed as browser routes; only `pages.json` is loaded.

## Candidate tasks and limits

- `bridge-record-conflict`: distinguish the 1963/1961 statements without guessing.
- `schedule-revisions`: compare earlier and later bridge completion projections.
- `funding-and-continuation`: combine project costs, funding and the next year's list.
- `visitation-reconcile`: reconcile monthly/cumulative counts and a revised total.
- `reopening-and-repairs`: relate entrance reopening to remaining repair work.
- `flood-access-audit`: audit five claims using dated flood updates, access-mode
  restrictions, changing forecasts, observed openings and later funded repairs.
  Includes a private manual rubric; intended as a harder candidate, uncalibrated.

These tasks were authored after source inspection. Some answers are easy, some
facts repeat across records, and the corpus is small. Difficulty, model knowledge,
contamination, collaboration benefit and scoring have not been calibrated.
Reference answers require source/date distinctions; conflicting statements are
preserved rather than silently reconciled. This is content infrastructure, not
swarming evidence or an independently sealed benchmark.

## Maintenance and checks

Rebuild pages from the committed snapshots with `uv run --offline python
corpora/yellowstone-records/build.py`. This does not fetch live sources. Updating
sources is a separate explicit ingestion step requiring new provenance/hashes and
review of the questions. The temporary ingestion script/HTML downloads are not
required for offline rebuilds; extraction details are recorded in the manifest.

Focused checks (no model calls):

```sh
UV_CACHE_DIR=/tmp/parser-task-uv-cache uv run --offline python -m pytest \
  tests/test_yellowstone_corpus.py orchestrator/simulated_web/test_simulated_web.py -q
```

Checks cover provenance hashes, complete chunk coverage, navigation, browser
loading, exact supporting excerpts and private task separation. They cannot
establish benchmark validity or prove every reference answer is correct.
