#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path.cwd()
SITE = ROOT / "site"
PAGES = SITE / "pages"

MANIFEST = SITE / "site-manifest.json"

SOURCE_DOC = ROOT / "docs" / "v2_integrity_repair_record.md"
SOURCE_COPY = (
    SITE / "assets" / "sources" /
    "v2_integrity_repair_record.md"
)

DATASET_IDENTITY = (
    ROOT / "docs" /
    "v2_confirm_dataset_identity.json"
)

DATASET_IDENTITY_COPY = (
    SITE / "assets" / "sources" /
    "v2_confirm_dataset_identity.json"
)

MARKER = "<!-- V2-INTEGRITY-REPAIR -->"

D_CONFIRM_SHA = (
    "e781dbebe0956322476cd8841675ad9a"
    "22e9d680c6dec7a47f5d23c35829c367"
)

NAV_ITEMS = [
    ("home", "01", "Overview", "index.html"),
    ("v0", "02", "Understanding V0", "pages/v0.html"),
    ("v1", "03", "Understanding V1", "pages/v1.html"),
    ("v2", "04", "V2 confirmation", "pages/v2.html"),
    (
        "evidence",
        "05",
        "Results & evidence",
        "pages/evidence.html",
    ),
    ("models", "06", "Models", "pages/models.html"),
    (
        "data",
        "07",
        "Data & pipeline",
        "pages/data.html",
    ),
    ("roadmap", "08", "Roadmap", "pages/roadmap.html"),
    (
        "decisions",
        "09",
        "Design decisions",
        "pages/decisions.html",
    ),
    (
        "research",
        "10",
        "Research horizons",
        "pages/research.html",
    ),
    (
        "build-log",
        "11",
        "Build log",
        "pages/build-log.html",
    ),
    (
        "journey",
        "12",
        "The journey",
        "pages/journey.html",
    ),
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read(path: Path) -> str:
    return path.read_text(
        encoding="utf-8"
    )


def write(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def rel_href(
    html_path: Path,
    target: str,
) -> str:

    depth = (
        len(
            html_path
            .relative_to(SITE)
            .parents
        )
        - 1
    )

    return "../" * depth + target


def nav_html(
    html_path: Path,
    active: str,
) -> str:

    links = []

    for (
        key,
        number,
        label,
        target,
    ) in NAV_ITEMS:

        attrs = ""

        if key == active:
            attrs = (
                ' class="active" '
                'aria-current="page"'
            )

        links.append(
            f'<a data-page="{key}" '
            f'href="{rel_href(html_path, target)}"'
            f'{attrs}>'
            f'<span>{number}</span>'
            f'{label}</a>'
        )

    return (
        '<nav class="nav">'
        + "".join(links)
        + "</nav>"
    )


def all_html_files() -> list[Path]:

    return sorted([
        SITE / "index.html",
        *PAGES.glob("*.html"),
    ])


def insert_before_footer(
    path: Path,
    block: str,
) -> None:

    text = read(path)

    if MARKER in text:
        return

    idx = text.find(
        '<footer class="footer">'
    )

    if idx < 0:
        raise RuntimeError(
            f"Footer not found in {path}"
        )

    write(
        path,
        (
            text[:idx]
            + MARKER
            + "\n"
            + block.strip()
            + "\n"
            + text[idx:]
        ),
    )


def write_source_record() -> None:

    body = f"""# V2 Confirmation Data-Integrity Repair Record

Status: **data/pipeline bug-fix phase complete; D_CONFIRM build + QA passed; corrected development baseline rebuild pending; no D_CONFIRM model predictions or metrics have been calculated.**

## Experiment boundary

- D_CONFIRM selection was frozen before scoring.
- First 30 eligible matches were selected chronologically.
- No reserve substitution was allowed after seeing data behavior.
- Freeze commit: `4614f91`.
- Map: Mirage.
- Raw timing contract: 64 ticks/sec.
- awpy: 2.0.2.
- demoparser2: 0.42.0.
- All repairs below happened before confirmation prediction.

## Debug path

### 1. Historical pickup / inventory gap

Nemesis vs SINNERS exposed a state where inventory did not show C4 even though the bomb event stream identified a pickup carrier.

Repair:

- use pickup only to identify WHO carries the bomb;
- use that player's CURRENT snapshot XYZ;
- never reuse stale pickup-event coordinates.

The timing-corrected legacy regression later passed all 20 matches and all 1686 observations.

### 2. Missing player snapshots

NiP vs FOKUS round 8 and ShindeN vs Fluxo round 19 exposed truncated player-state streams.

New rule:

- target snapshot must exist at the target tick or within +1 raw tick;
- otherwise exclude that observation;
- no interpolation;
- no carry-forward;
- no fabricated snapshot.

Reason code:

`MISSING_PLAYER_SNAPSHOT`

Commit:

`367bf2a` — `fix: exclude unavailable V2 player snapshots`

### 3. Wrong hypothesis: bomb round-number alignment

Leo vs UNiTY initially looked like a bomb-event round association problem.

We tested:

- event `round_num`
- independently derived round from the event tick interval

Result:

**0 round-number mismatches.**

So that hypothesis was rejected.

### 4. Wrong assumption: demo.rounds.bomb_site as truth

We then temporarily treated `demo.rounds.bomb_site` as the expected A/B label.

That assumption was wrong.

Across frozen D_CONFIRM:

- 298 valid plant events
- 188 event A ↔ planter place A
- 110 event B ↔ planter place B
- 0 event/place mismatch
- total agreement = **298 / 298**

XYZ positions also formed clean A-site and B-site clusters.

But `demo.rounds.bomb_site` reported B for:

- all 188 valid A-site plants
- all 110 valid B-site plants

Therefore:

`demo.rounds.bomb_site`

is NOT canonical A/B truth for this parser output.

### 5. Ghost plant events

We found plant events occurring after their associated round had already ended.

D_CONFIRM:

- 18 ghost plant events

Legacy 20-match corpus:

- 11 ghost plant events

Canonical rule:

A plant event is label-eligible only if:

`round.start <= plant_tick <= round.end`

Anything after round end is:

`GHOST_PLANT_EVENT`

and cannot create a label.

### 6. Legacy impact

The 11 ghost rounds contaminated the old timing-corrected development supervision.

Total observations:

1686

Affected labels:

44 / 1686 = 2.6097%

Transitions:

- 24 `A_PLANT → NO_PLANT`
- 20 `B_PLANT → NO_PLANT`

Corrected label totals:

- `NO_PLANT = 844`
- `A_PLANT = 551`
- `B_PLANT = 291`

Observation count remained exactly:

1686

So this defect was localized to y labels, not the frozen 37-feature X matrix.

### 7. Missing plant-event cases

Two D_CONFIRM rounds had metadata indicating a plant but no valid plant event:

- Dendele vs Turma do Pagode round 7
- NiP vs FOKUS round 8

Their player telemetry ended too early to reconstruct plant site reliably.

We explicitly refused to guess A/B from `demo.rounds.bomb_site`.

Reason:

`PLANT_LABEL_UNRESOLVED`

Action:

exclude the round.

### 8. Canonical plant-label contract

1. Plant event must occur inside the round interval.
2. Valid `BombsiteA → A_PLANT`.
3. Valid `BombsiteB → B_PLANT`.
4. Ghost plant cannot create a label.
5. Metadata plant with no valid event does not justify guessing A/B.
6. Unresolved site → `PLANT_LABEL_UNRESOLVED`.
7. `demo.rounds.bomb_site` is integrity evidence only.

Commit:

`13e35b7` — `fix: reject ghost plants in canonical labels`

### 9. Bomb-state availability

After Leo round 4's label was correctly changed to `NO_PLANT`, a separate feature problem remained.

Observation ticks:

- 10s = 30888
- 20s = 31528
- 30s = 32168
- 40s = 32808

First trustworthy pickup:

32231

Therefore the first three observations had no trustworthy bomb state available at or before their target time.

We explicitly forbid:

future pickup → earlier observation reconstruction

Instead:

`BOMB_STATE_UNRESOLVED`

excludes the observation.

Only missing historical evidence is excluded.

Real invariant failures such as:

- multiple C4 carriers
- missing pickup identity
- ambiguous pickup carrier
- invalid event state

still hard-fail.

Commit:

`383b678` — `fix: exclude unresolved V2 bomb states`

## Final D_CONFIRM build

Matches:

**30**

Feature-valid observations:

**2269**

Labels:

- A_PLANT = 693
- B_PLANT = 416
- NO_PLANT = 1160

Horizons:

- 10s = 600
- 20s = 596
- 30s = 577
- 40s = 496

Dataset SHA256:

`{D_CONFIRM_SHA}`

## Exclusion layers

Round-level:

- 5 × `MISSING_FREEZE_END`
- 2 × `PLANT_LABEL_UNRESOLVED`

Observation-level:

- 1 × `MISSING_PLAYER_SNAPSHOT`
- 3 × `BOMB_STATE_UNRESOLVED`

## Post-build QA

PASS:

- exactly 30 frozen matches
- exactly 2269 rows
- all 37 frozen features present
- no null
- no NaN
- no Inf
- exact label counts
- exact horizon counts
- zero duplicate observation keys
- exact label vocabulary

No model predictions or confirmation metrics were calculated during these repairs.

## Scientific consequence

The old V0 / H0 / H1 / H2 / T0 metrics remain useful historical evidence, but they are now classified as:

**pre-label-repair development results**

They are not the final canonical baseline because 44 / 1686 labels were wrong.

The next experiment is constrained:

1. rebuild corrected 20-match development data;
2. keep the same 1686 observations;
3. keep the same 37 features;
4. keep the same five match-level folds;
5. keep the same XGB-A5 hyperparameters;
6. keep the same H2 procedure;
7. DO NOT RETUNE;
8. freeze corrected V0/H2 artifacts;
9. then run one formal untouched D_CONFIRM evaluation.

## Permanent engineering lesson

Every data bug should leave a guardrail.

Debugging flow:

detect problem
→ determine root cause
→ fix current case
→ formulate invariant
→ add reason code / checker
→ regression test
→ prevent the same class of bug from silently returning.

Never repair an observation with future information.
"""

    write(
        SOURCE_DOC,
        body,
    )

    SOURCE_COPY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        SOURCE_DOC,
        SOURCE_COPY,
    )

    if DATASET_IDENTITY.exists():

        DATASET_IDENTITY_COPY.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            DATASET_IDENTITY,
            DATASET_IDENTITY_COPY,
        )


def make_v2_page() -> None:

    template = read(
        PAGES / "v1.html"
    )

    body = f'''
<header class="hero">
  <div class="kicker">
    V2 confirmation · data integrity repair
  </div>

  <h1>V2 Confirmation Integrity</h1>

  <p class="lead">
    The untouched confirmation corpus did exactly what
    it was supposed to do: it exposed hidden parser and
    adapter assumptions before any model scoring.
    The bugs are now repaired, D_CONFIRM passes QA,
    and the next gate is a corrected no-retuning
    rebuild of the development baseline.
  </p>

  <div class="badges">
    <span class="badge ok">
      Bug-fix phase complete
    </span>
    <span class="badge">
      30 frozen matches
    </span>
    <span class="badge warn">
      No confirmation scoring yet
    </span>
  </div>
</header>

<section class="section">
  <div class="section-head">
    <h2>Current checkpoint</h2>
    <p>
      D_CONFIRM is built and QA-passed.
      The historical development baseline must now
      be rebuilt on corrected labels.
    </p>
  </div>

  <div class="grid">

    <article class="card">
      <div class="stat-label">
        D_CONFIRM
      </div>

      <div class="stat">
        2269
      </div>

      <p>
        Feature-valid observations from
        30 frozen matches.
      </p>
    </article>

    <article class="card">
      <div class="stat-label">
        Legacy label repair
      </div>

      <div class="stat">
        44 / 1686
      </div>

      <p>
        2.61% of development observation labels
        changed from A/B Plant to No Plant.
      </p>
    </article>

    <article class="card">
      <div class="stat-label">
        Confirmation scoring
      </div>

      <div class="stat">
        0
      </div>

      <p>
        No prediction, metric, tuning, or reserve
        swapping guided these repairs.
      </p>
    </article>

  </div>
</section>

<section class="section">

  <div class="section-head">
    <h2>
      How debugging changed our understanding
    </h2>

    <p>
      The important record is not only the final fix,
      but which hypotheses were rejected.
    </p>
  </div>

  <div class="timeline">

    <div class="entry">
      <time>
        01 · Carrier coverage
      </time>

      <h3>
        Inventory was not always sufficient
        to locate the bomb.
      </h3>

      <p>
        Nemesis vs SINNERS showed a historical pickup
        event with no C4 in player inventory.
        Pickup identifies the carrier.
        Current snapshot coordinates define position.
      </p>
    </div>

    <div class="entry">
      <time>
        02 · Snapshot coverage
      </time>

      <h3>
        Some demos stop emitting player state
        before a nominal horizon.
      </h3>

      <p>
        NiP and ShindeN exposed missing snapshots.
        V2 excludes only the affected observation,
        with no interpolation or carry-forward.
      </p>
    </div>

    <div class="entry">
      <time>
        03 · Rejected hypothesis
      </time>

      <h3>
        We suspected bomb-event round numbers
        were wrong.
      </h3>

      <p>
        Independent tick-derived round association
        found zero round-number mismatches.
        The hypothesis was rejected instead of patched.
      </p>
    </div>

    <div class="entry">
      <time>
        04 · Rejected assumption
      </time>

      <h3>
        We briefly treated
        <code>demo.rounds.bomb_site</code>
        as A/B truth.
      </h3>

      <p>
        That assumption failed.
        Across 298 valid D_CONFIRM plants,
        event site matched planter place 298/298,
        while round metadata reported B for
        all 188 A-site plants.
      </p>
    </div>

    <div class="entry">
      <time>
        05 · Root cause
      </time>

      <h3>
        Ghost plant events were contaminating labels.
      </h3>

      <p>
        18 confirmation events and 11 legacy events
        occurred after round end.
        The 11 legacy ghost rounds changed
        44 / 1686 development labels.
      </p>
    </div>

    <div class="entry">
      <time>
        06 · Separate feature issue
      </time>

      <h3>
        Leo round 4 still had no bomb state
        at early horizons.
      </h3>

      <p>
        The label was fixed to No Plant,
        but 10s / 20s / 30s had no trustworthy
        bomb evidence before the target tick.
        Those observations are excluded rather than
        backfilled from a future pickup.
      </p>
    </div>

  </div>
</section>

<section class="section">

  <div class="section-head">
    <h2>Permanent data contracts</h2>

    <p>
      Each debugged failure now leaves a guardrail.
    </p>
  </div>

  <div
    class="table-wrap"
    role="region"
    aria-label="V2 data integrity contracts"
    tabindex="0"
  >

    <table>

      <thead>
        <tr>
          <th>Contract</th>
          <th>Invariant</th>
          <th>Fail-closed action</th>
        </tr>
      </thead>

      <tbody>

        <tr>
          <td>PLANT-001</td>
          <td>
            Plant event must fall inside
            [round.start, round.end].
          </td>
          <td>
            Ignore ghost event.
            It cannot create a label.
          </td>
        </tr>

        <tr>
          <td>PLANT-002</td>
          <td>
            Valid event bombsite defines A/B.
            Planter place is the integrity cross-check.
          </td>
          <td>
            Block on a true site conflict.
          </td>
        </tr>

        <tr>
          <td>PLANT-003</td>
          <td>
            Metadata plant without valid event
            does not justify guessing site.
          </td>
          <td>
            PLANT_LABEL_UNRESOLVED
            → exclude round.
          </td>
        </tr>

        <tr>
          <td>SNAPSHOT-001</td>
          <td>
            Observation needs a real player snapshot
            at target or within +1 raw tick.
          </td>
          <td>
            MISSING_PLAYER_SNAPSHOT
            → exclude observation.
          </td>
        </tr>

        <tr>
          <td>BOMB-001</td>
          <td>
            Bomb state may use only evidence
            at or before observation time.
          </td>
          <td>
            BOMB_STATE_UNRESOLVED
            → exclude observation.
            Never use future events.
          </td>
        </tr>

        <tr>
          <td>DATASET-001</td>
          <td>
            37 frozen features,
            finite values,
            unique observation keys,
            frozen labels/horizons.
          </td>
          <td>
            Block dataset before scoring.
          </td>
        </tr>

      </tbody>

    </table>
  </div>
</section>

<section class="section">

  <div class="section-head">
    <h2>Final D_CONFIRM dataset</h2>

    <p>
      Built after deterministic pre-inference repairs.
    </p>
  </div>

  <div class="grid">

    <article class="card half">
      <div class="stat-label">
        Labels
      </div>

      <h3>
        A 693 · B 416 · No Plant 1160
      </h3>

      <p>
        Canonical post-repair class counts.
      </p>
    </article>

    <article class="card half">
      <div class="stat-label">
        Horizons
      </div>

      <h3>
        600 · 596 · 577 · 496
      </h3>

      <p>
        10s · 20s · 30s · 40s observations.
      </p>
    </article>

  </div>

  <div class="callout">
    <strong>Dataset SHA256:</strong>
    <code>{D_CONFIRM_SHA}</code>
  </div>

  <div class="callout">
    <strong>QA:</strong>
    30/30 frozen matches,
    all 37 model features present,
    no null/NaN/Inf,
    zero duplicate observation keys,
    exact label and horizon counts.
  </div>

</section>

<section class="section">

  <div class="section-head">
    <h2>What happens next</h2>

    <p>
      Repair the development baseline without
      adding new researcher degrees of freedom.
    </p>
  </div>

  <div class="flow">
    <div class="node">
      <b>Correct y</b><br>
      44 labels
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Same X</b><br>
      37 features
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Same CV</b><br>
      5 folds
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Same models</b><br>
      V0 + H2
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>No retuning</b><br>
      freeze
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>D_CONFIRM</b><br>
      one formal eval
    </div>
  </div>

  <div class="callout warn">
    <strong>
      Historical V0/V1 metrics are now
      pre-label-repair evidence.
    </strong>

    Keep them for research history,
    but do not treat them as the final
    canonical baseline until the corrected-label
    rerun is complete.
  </div>

  <p>
    <a
      href="../assets/sources/v2_integrity_repair_record.md"
      download
    >
      Download the complete V2 integrity repair record ↓
    </a>
  </p>

</section>
'''

    template = re.sub(
        r"<title>.*?</title>",
        (
            "<title>"
            "V2 Confirmation Integrity — "
            "CS2 Tactical Intelligence Lab"
            "</title>"
        ),
        template,
        count=1,
        flags=re.S,
    )

    template = re.sub(
        r'<body data-page="[^"]+">',
        '<body data-page="v2">',
        template,
        count=1,
    )

    template = re.sub(
        r'<span class="breadcrumb">.*?</span>',
        (
            '<span class="breadcrumb">'
            'The notebook '
            '<span>/</span> '
            'V2 confirmation integrity'
            '</span>'
        ),
        template,
        count=1,
        flags=re.S,
    )

    template = re.sub(
        r'<span class="top-meta">.*?</span>',
        (
            '<span class="top-meta">'
            'MIRAGE '
            '<span>·</span> '
            'V2 DATA INTEGRITY PASS'
            '</span>'
        ),
        template,
        count=1,
        flags=re.S,
    )

    start = (
        template.index(
            '<div class="content">'
        )
        + len(
            '<div class="content">'
        )
    )

    footer_start = template.index(
        '<footer class="footer">',
        start,
    )

    footer_end = (
        template.index(
            '</footer>',
            footer_start,
        )
        + len('</footer>')
    )

    footer = (
        '<footer class="footer">'
        '<a href="../index.html">'
        'CS2 Tactical Intelligence Lab'
        '</a>'
        '<span>'
        'V2 data integrity passed · '
        'baseline rebuild pending · '
        'September 2026'
        '</span>'
        '<a href="../pages/evidence.html#v2-integrity">'
        'Inspect the evidence ↗'
        '</a>'
        '</footer>'
    )

    out = (
        template[:start]
        + "\n"
        + body.strip()
        + "\n"
        + footer
        + template[footer_end:]
    )

    write(
        PAGES / "v2.html",
        out,
    )


def update_existing_pages() -> None:

    insert_before_footer(
        SITE / "index.html",
        f'''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>V2 confirmation data gate passed</h2>

    <p>
      Untouched confirmation data exposed
      parser/adapter assumptions before
      any model scoring.
    </p>
  </div>

  <div class="grid">

    <article class="card half">
      <div class="stat-label">
        D_CONFIRM
      </div>

      <h3>
        30 frozen matches · 2269 observations
      </h3>

      <p>
        Final dataset passes feature, label,
        horizon, finiteness, and uniqueness QA.
      </p>

      <p>
        SHA256:
        <code>{D_CONFIRM_SHA}</code>
      </p>
    </article>

    <article class="card half">

      <div class="stat-label">
        Scientific consequence
      </div>

      <h3>
        44 / 1686 legacy labels repaired
      </h3>

      <p>
        Old V0/H2 metrics remain
        pre-repair research history.
        Rebuild the same frozen methods with
        corrected labels and no retuning before
        the formal D_CONFIRM evaluation.
      </p>

    </article>

  </div>

  <p>
    <a class="button" href="pages/v2.html">
      Read the V2 integrity record
      <span>↗</span>
    </a>
  </p>

</section>
''',
    )

    insert_before_footer(
        PAGES / "v1.html",
        '''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      Post-freeze label-integrity note
    </h2>

    <p>
      V2 confirmation data exposed a
      supervision defect in the original
      development corpus.
    </p>
  </div>

  <div class="callout warn">

    <strong>
      44 / 1686 timing-corrected V0
      observation labels were contaminated
      by ghost plant events.
    </strong>

    The V0/H0/H1/H2/T0 results above remain
    important historical evidence,
    but they are now classified as
    pre-label-repair development results.

    A corrected no-retuning rerun is required
    before the final D_CONFIRM comparison.

  </div>

  <p>
    <a href="v2.html">
      Read the V2 integrity investigation ↗
    </a>
  </p>

</section>
''',
    )

    insert_before_footer(
        PAGES / "evidence.html",
        f'''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>V2 data-integrity evidence</h2>

    <p>
      Untouched confirmation data converted
      hidden assumptions into measurable contracts.
    </p>
  </div>

  <div class="evidence">

    <div class="e-head">
      <span class="eid">
        EVID-V2-01
      </span>

      <span class="badge ok">
        Confirmed
      </span>
    </div>

    <div class="e-body">

      <h3>
        Valid plant-event site semantics are
        independently supported.
      </h3>

      <div class="kv">

        <b>D_CONFIRM</b>
        <span>
          298 valid plant events.
        </span>

        <b>Independent check</b>
        <span>
          298/298 event site ↔ planter place agreement:
          188 A↔A,
          110 B↔B,
          0 mismatch.
        </span>

        <b>Rejected source</b>
        <span>
          <code>demo.rounds.bomb_site</code>
          reported B for all 188 A plants
          and the 110 B plants.
        </span>

      </div>
    </div>
  </div>

  <div class="evidence">

    <div class="e-head">
      <span class="eid">
        EVID-V2-02
      </span>

      <span class="badge warn">
        Legacy defect measured
      </span>
    </div>

    <div class="e-body">

      <h3>
        Ghost plants contaminated
        44 development labels.
      </h3>

      <div class="kv">

        <b>Legacy audit</b>
        <span>
          11 ghost plant rounds across
          the 20-match development corpus.
        </span>

        <b>Impact</b>
        <span>
          44 / 1686 observations =
          2.6097% label contamination.
        </span>

        <b>Transitions</b>
        <span>
          24 A_PLANT → NO_PLANT;
          20 B_PLANT → NO_PLANT.
        </span>

        <b>Scope</b>
        <span>
          Observation count remained 1686.
          Defect was localized to y labels.
        </span>

      </div>
    </div>
  </div>

  <div class="evidence">

    <div class="e-head">
      <span class="eid">
        EVID-V2-03
      </span>

      <span class="badge ok">
        QA passed
      </span>
    </div>

    <div class="e-body">

      <h3>
        D_CONFIRM builds without model scoring.
      </h3>

      <div class="kv">

        <b>Matches</b>
        <span>30 frozen matches.</span>

        <b>Observations</b>
        <span>2269.</span>

        <b>Labels</b>
        <span>
          A 693 · B 416 · No Plant 1160.
        </span>

        <b>Horizons</b>
        <span>
          10s 600 ·
          20s 596 ·
          30s 577 ·
          40s 496.
        </span>

        <b>SHA256</b>
        <span>
          <code>{D_CONFIRM_SHA}</code>
        </span>

      </div>
    </div>
  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "data.html",
        f'''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      V2 data-integrity gate
    </h2>

    <p>
      Dataset construction now fails closed
      when telemetry cannot support a
      trustworthy observation or label.
    </p>
  </div>

  <div class="table-wrap">

    <table>

      <thead>
        <tr>
          <th>Reason</th>
          <th>Level</th>
          <th>Action</th>
        </tr>
      </thead>

      <tbody>

        <tr>
          <td>MISSING_FREEZE_END</td>
          <td>Round</td>
          <td>Exclude round</td>
        </tr>

        <tr>
          <td>PLANT_LABEL_UNRESOLVED</td>
          <td>Round</td>
          <td>
            Exclude round.
            Never guess site from bad metadata.
          </td>
        </tr>

        <tr>
          <td>MISSING_PLAYER_SNAPSHOT</td>
          <td>Observation</td>
          <td>
            Exclude observation.
            No interpolation.
          </td>
        </tr>

        <tr>
          <td>BOMB_STATE_UNRESOLVED</td>
          <td>Observation</td>
          <td>
            Exclude observation.
            No future-event backfill.
          </td>
        </tr>

      </tbody>

    </table>
  </div>

  <div class="callout">
    <strong>
      Current frozen D_CONFIRM:
    </strong>
    30 matches ·
    2269 rows ·
    37 model features ·
    SHA256
    <code>{D_CONFIRM_SHA}</code>.
  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "models.html",
        '''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      Baseline status after V2 integrity audit
    </h2>

    <p>
      Model design stays frozen.
      Development supervision changed.
    </p>
  </div>

  <div class="callout warn">

    <strong>
      44 / 1686 development labels were repaired.
    </strong>

    Therefore old V0/H2 metrics are historical
    pre-repair metrics.

    The next baseline must reuse the same:

    20 matches,
    1686 observations,
    37 features,
    five folds,
    XGB-A5 hyperparameters,
    and H2 procedure.

    There will be no retuning.

  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "roadmap.html",
        '''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      V2 confirmation gate
    </h2>

    <p>
      Data integrity passed.
      Formal confirmation scoring still has not begun.
    </p>
  </div>

  <div class="flow">

    <div class="node">
      <b>D_CONFIRM</b><br>
      frozen
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Integrity</b><br>
      passed
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Legacy y</b><br>
      correct 44
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Rebuild</b><br>
      no retuning
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Freeze</b><br>
      corrected V0/H2
    </div>

    <span class="arrow">→</span>

    <div class="node">
      <b>Evaluate</b><br>
      once
    </div>

  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "decisions.html",
        '''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      V2 integrity decisions
    </h2>

    <p>
      Decisions made before any confirmation-model scoring.
    </p>
  </div>

  <div class="decision-list">

    <article class="decision">
      <span>ADR-008</span>
      <div>
        <h3>
          Valid in-round plant events are
          canonical A/B site evidence.
        </h3>
        <p>
          Event bombsite achieved 298/298 agreement
          with planter place in D_CONFIRM.
        </p>
      </div>
    </article>

    <article class="decision">
      <span>ADR-009</span>
      <div>
        <h3>
          Round-level bomb_site is not
          canonical A/B truth.
        </h3>
        <p>
          It reported B for all 188 independently
          verified A-site plants in D_CONFIRM.
        </p>
      </div>
    </article>

    <article class="decision">
      <span>ADR-010</span>
      <div>
        <h3>
          Ghost plant events cannot create labels.
        </h3>
        <p>
          Plant tick must fall inside the
          authoritative round interval.
        </p>
      </div>
    </article>

    <article class="decision">
      <span>ADR-011</span>
      <div>
        <h3>
          Unresolved plant site excludes the round.
        </h3>
        <p>
          Never guess A/B from conflicting metadata.
        </p>
      </div>
    </article>

    <article class="decision">
      <span>ADR-012</span>
      <div>
        <h3>
          Missing snapshot/bomb state excludes only
          the affected observation when explicitly
          allowed by the V2 confirmation adapter.
        </h3>
      </div>
    </article>

    <article class="decision">
      <span>ADR-013</span>
      <div>
        <h3>
          No future information may repair
          an earlier observation.
        </h3>
        <p>
          Future pickup/drop/plant evidence is
          forbidden for reconstruction at time t.
        </p>
      </div>
    </article>

  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "build-log.html",
        f'''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      V2 confirmation integrity repair
    </h2>

    <p>
      Chronology of debugging and evidence gates.
    </p>
  </div>

  <div class="timeline">

    <div class="entry">
      <time>
        2026-09-12 · Experiment boundary
      </time>

      <h3>
        D_CONFIRM selection frozen before scoring
      </h3>

      <p>
        First 30 eligible matches frozen.
        No reserve substitution after seeing
        data behavior.
        Freeze commit:
        <code>4614f91</code>.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Carrier coverage
      </time>

      <h3>
        Historical pickup fallback
      </h3>

      <p>
        Nemesis round 16 exposed missing C4 inventory
        despite a pickup event.
        Pickup identity now selects the current carrier
        snapshot.
        Stale event XYZ is not reused.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Regression correction
      </time>

      <h3>
        Old parity script pointed at the wrong
        dataset contract
      </h3>

      <p>
        The first parity check compared against
        the pre-timing-correction 1268-row dataset.

        The correct timing-corrected regression
        passed all 20 matches and all
        1686 observations.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Snapshot coverage
      </time>

      <h3>
        Missing snapshot exclusion added
      </h3>

      <p>
        NiP round 8 and ShindeN round 19 exposed
        truncated player-state streams.

        Commit:
        <code>367bf2a</code>.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Label semantics
      </time>

      <h3>
        Leo round 4 exposed ghost plants and bad
        round-level site metadata
      </h3>

      <p>
        Round-number-misalignment hypothesis
        was rejected.

        Valid event bombsite was established
        as canonical A/B evidence.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Legacy audit
      </time>

      <h3>
        44 / 1686 development labels corrected
      </h3>

      <p>
        11 ghost rounds changed
        24 A labels and
        20 B labels to No Plant.

        Plant-contract commit:
        <code>13e35b7</code>.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Bomb availability
      </time>

      <h3>
        Future pickup backfill explicitly forbidden
      </h3>

      <p>
        Leo 10s/20s/30s had no trustworthy
        bomb state before the target tick.

        Commit:
        <code>383b678</code>.
      </p>
    </div>

    <div class="entry">
      <time>
        2026-09-13 · Data gate complete
      </time>

      <h3>
        D_CONFIRM built and QA passed
      </h3>

      <p>
        30 matches,
        2269 observations,
        37 finite frozen features,
        zero duplicate keys,
        exact label/horizon counts.

        SHA256:
        <code>{D_CONFIRM_SHA}</code>.

        No model predictions or metrics
        were calculated.
      </p>
    </div>

  </div>

</section>
''',
    )

    insert_before_footer(
        PAGES / "journey.html",
        '''
<section class="section" id="v2-integrity">

  <div class="section-head">
    <h2>
      V2 debugging journey
    </h2>

    <p>
      The important change was not only
      fixing bugs.
      It was changing how the project
      thinks about data.
    </p>
  </div>

  <div class="timeline">

    <div class="entry">

      <time>Entry 005</time>

      <h3>
        Untouched data challenged assumptions
        development data never challenged.
      </h3>

      <p>
        The confirmation corpus did not immediately
        tell us whether the model generalized.

        First, it showed that our adapter assumptions
        were incomplete.

        That is exactly why untouched data mattered.
      </p>

    </div>

    <div class="entry">

      <time>Entry 006</time>

      <h3>
        Evidence had to be allowed to overturn
        our own debugging story.
      </h3>

      <p>
        We first suspected bomb-event round-number
        misalignment.

        The audit returned zero mismatches.

        Then we treated round-level bomb-site metadata
        as expected truth.

        298 independent plant/place checks showed
        that assumption was wrong too.
      </p>

    </div>

    <div class="entry">

      <time>Entry 007</time>

      <h3>
        The legacy bug was real,
        but measurable rather than catastrophic.
      </h3>

      <p>
        Ghost plants affected 44 / 1686 labels,
        not the entire feature system.

        Measuring exact scope let us repair
        supervision without redesigning the model.
      </p>

    </div>

    <div class="entry">

      <time>Entry 008</time>

      <h3>
        Debugging became part of the architecture.
      </h3>

      <p>
        Pickup gaps,
        snapshot gaps,
        ghost plants,
        unresolved labels,
        and unavailable bomb states
        are now explicit integrity contracts.

        The goal is for the next strange demo
        to be classified automatically
        before it becomes another manual investigation.
      </p>

    </div>

  </div>

</section>
''',
    )


def normalize_nav_and_status(
    path: Path,
) -> None:

    text = read(path)

    match = re.search(
        r'<body data-page="([^"]+)"',
        text,
    )

    if not match:
        raise RuntimeError(
            f"Missing body data-page in {path}"
        )

    active = match.group(1)

    text, count = re.subn(
        r'<nav class="nav">.*?</nav>',
        nav_html(
            path,
            active,
        ),
        text,
        count=1,
        flags=re.S,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not replace nav in {path}"
        )

    text = re.sub(
        (
            r'<div class="status">'
            r'<i class="dot"></i>'
            r'.*?</div>'
        ),
        (
            '<div class="status">'
            '<i class="dot"></i>'
            'V2 · Data integrity passed'
            '</div>'
        ),
        text,
        count=1,
        flags=re.S,
    )

    write(
        path,
        text,
    )


def update_cache_bust() -> None:

    for path in all_html_files():

        text = read(path)

        text = re.sub(
            (
                r'assets/style\.css'
                r'(?:\?v=[^"]+)?'
            ),
            (
                'assets/style.css'
                '?v=v2-integrity'
            ),
            text,
        )

        text = re.sub(
            (
                r'\.\./assets/style\.css'
                r'(?:\?v=[^"]+)?'
            ),
            (
                '../assets/style.css'
                '?v=v2-integrity'
            ),
            text,
        )

        write(
            path,
            text,
        )


def update_manifest() -> None:

    data = json.loads(
        read(MANIFEST)
    )

    pages = data.get(
        "pages",
        [],
    )

    if "pages/v2.html" not in pages:

        try:
            idx = (
                pages.index(
                    "pages/v1.html"
                )
                + 1
            )

        except ValueError:
            idx = 2

        pages.insert(
            idx,
            "pages/v2.html",
        )

    data.update(
        stage=(
            "V2 confirmation data integrity passed"
        ),
        v2_status=(
            "D_CONFIRM BUILT + QA PASSED · "
            "BASELINE REBUILD PENDING · "
            "NO CONFIRMATION SCORING YET"
        ),
        d_confirm_matches=30,
        d_confirm_observations=2269,
        d_confirm_sha256=D_CONFIRM_SHA,
        legacy_label_repairs=44,
        legacy_label_repair_fraction=(
            44 / 1686
        ),
        baseline_status=(
            "Pre-repair V0/V1 metrics are historical; "
            "corrected no-retuning rebuild required"
        ),
        next_gate=(
            "Rebuild corrected V0/H2 with frozen methods, "
            "freeze artifacts, then one formal "
            "D_CONFIRM evaluation"
        ),
        pages=pages,
    )

    write(
        MANIFEST,
        json.dumps(
            data,
            indent=2,
        )
        + "\n",
    )


def validate_nav() -> None:

    for path in all_html_files():

        text = read(path)

        active = len(
            re.findall(
                (
                    r'class="active" '
                    r'aria-current="page"'
                ),
                text,
            )
        )

        if active != 1:
            raise RuntimeError(
                f"{path}: expected 1 active nav "
                f"entry, found {active}"
            )

        if 'data-page="v2"' not in text:
            raise RuntimeError(
                f"{path}: V2 nav entry missing"
            )


def validate_local_links() -> None:

    broken = []

    for path in all_html_files():

        text = read(path)

        for href in re.findall(
            r'href="([^"]+)"',
            text,
        ):

            if href.startswith((
                "#",
                "http://",
                "https://",
                "mailto:",
                "javascript:",
            )):
                continue

            clean = urlsplit(
                href
            ).path

            if not clean:
                continue

            target = (
                path.parent /
                clean
            ).resolve()

            try:
                target.relative_to(
                    SITE.resolve()
                )

            except ValueError:
                continue

            if not target.exists():
                broken.append(
                    (
                        path.relative_to(ROOT),
                        href,
                    )
                )

    if broken:
        raise RuntimeError(
            "Broken local links:\n"
            + "\n".join(
                f"  {path}: {href}"
                for path, href in broken
            )
        )


def validate_v2_content() -> None:

    checks = {

        PAGES / "v2.html": [
            "2269",
            "44 / 1686",
            "298/298",
            D_CONFIRM_SHA,
            "No retuning",
        ],

        PAGES / "evidence.html": [
            "EVID-V2-01",
            "298/298",
            "44 / 1686",
            "2269",
            D_CONFIRM_SHA,
        ],

        PAGES / "data.html": [
            "PLANT_LABEL_UNRESOLVED",
            "BOMB_STATE_UNRESOLVED",
            "2269",
        ],

        PAGES / "decisions.html": [
            "ADR-008",
            "ADR-013",
            "No future information",
        ],

        PAGES / "build-log.html": [
            "4614f91",
            "367bf2a",
            "13e35b7",
            "383b678",
            "2269",
        ],

        PAGES / "journey.html": [
            "Entry 005",
            "Entry 008",
            "zero mismatches",
        ],

        PAGES / "models.html": [
            "44 / 1686",
            "no retuning",
        ],

        PAGES / "v1.html": [
            "Post-freeze label-integrity note",
            "44 / 1686",
        ],
    }

    for path, values in checks.items():

        text = read(path)

        for value in values:

            if (
                value.lower()
                not in text.lower()
            ):
                raise RuntimeError(
                    f"{path}: missing required "
                    f"V2 value/text: {value}"
                )


def main() -> None:

    required = [
        SITE,
        PAGES,
        MANIFEST,
        SITE / "index.html",
        PAGES / "v0.html",
        PAGES / "v1.html",
        PAGES / "evidence.html",
        PAGES / "models.html",
        PAGES / "data.html",
        PAGES / "roadmap.html",
        PAGES / "decisions.html",
        PAGES / "research.html",
        PAGES / "build-log.html",
        PAGES / "journey.html",
    ]

    for path in required:
        require(path)

    write_source_record()
    make_v2_page()
    update_existing_pages()
    update_manifest()

    for path in all_html_files():
        normalize_nav_and_status(
            path
        )

    update_cache_bust()

    validate_nav()
    validate_v2_content()
    validate_local_links()

    print(
        "\nV2 integrity website update complete."
    )

    print(
        "\nCreated/updated:"
    )

    for path in [
        PAGES / "v2.html",
        SITE / "index.html",
        PAGES / "v1.html",
        PAGES / "evidence.html",
        PAGES / "data.html",
        PAGES / "models.html",
        PAGES / "roadmap.html",
        PAGES / "decisions.html",
        PAGES / "build-log.html",
        PAGES / "journey.html",
        SOURCE_DOC,
        SOURCE_COPY,
        MANIFEST,
    ]:
        print(
            " ",
            path.relative_to(ROOT),
        )

    if DATASET_IDENTITY.exists():

        print(
            " ",
            DATASET_IDENTITY_COPY
            .relative_to(ROOT),
        )

    print(
        "\nValidation: "
        "V2 scientific values/text PASS"
    )

    print(
        "Validation: "
        "one active nav item per page PASS"
    )

    print(
        "Validation: "
        "local links PASS"
    )

    print(
        "\nNext:"
    )

    print(
        "  git diff --check"
    )

    print(
        "  git status --short"
    )

    print(
        "  git diff --stat"
    )


if __name__ == "__main__":
    main()
