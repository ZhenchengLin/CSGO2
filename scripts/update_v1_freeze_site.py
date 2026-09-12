#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path.cwd()

SITE = (
    ROOT
    / "site"
)

PAGES = (
    SITE
    / "pages"
)

SOURCE_DOC = (
    ROOT
    / "docs"
    / "v1_scientific_record.md"
)

SOURCE_COPY = (
    SITE
    / "assets"
    / "sources"
    / "v1_scientific_record.md"
)

MANIFEST = (
    SITE
    / "site-manifest.json"
)


# ============================================================
# Verified scientific values
# ============================================================

EXPECTED = {

    "V0": (
        "0.838182",
        "0.503190",
        "0.594899",
        "0.566090",
    ),

    "H0": (
        "0.852546",
        "0.504516",
        "0.592527",
        "0.566455",
    ),

    "H1": (
        "0.840649",
        "0.499879",
        "0.590154",
        "0.564673",
    ),

    "H2": (
        "0.837397",
        "0.499781",
        "0.587189",
        "0.555824",
    ),

    "T0": (
        "0.844859",
        "0.507984",
        "0.593120",
        "0.563414",
    ),
}


# ============================================================
# Navigation
# ============================================================

NAV_ITEMS = [

    (
        "home",
        "01",
        "Overview",
        "index.html",
    ),

    (
        "v0",
        "02",
        "Understanding V0",
        "pages/v0.html",
    ),

    (
        "v1",
        "03",
        "Understanding V1",
        "pages/v1.html",
    ),

    (
        "evidence",
        "04",
        "Results & evidence",
        "pages/evidence.html",
    ),

    (
        "models",
        "05",
        "Models",
        "pages/models.html",
    ),

    (
        "data",
        "06",
        "Data & pipeline",
        "pages/data.html",
    ),

    (
        "roadmap",
        "07",
        "Roadmap",
        "pages/roadmap.html",
    ),

    (
        "decisions",
        "08",
        "Design decisions",
        "pages/decisions.html",
    ),

    (
        "research",
        "09",
        "Research horizons",
        "pages/research.html",
    ),

    (
        "build-log",
        "10",
        "Build log",
        "pages/build-log.html",
    ),

    (
        "journey",
        "11",
        "The journey",
        "pages/journey.html",
    ),
]


MARKER = (
    "<!-- V1-SCIENTIFIC-FREEZE -->"
)


# ============================================================
# Utilities
# ============================================================

def require(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


def read(
    path,
):

    return path.read_text(
        encoding="utf-8"
    )


def write(
    path,
    text,
):

    path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Validate scientific source
# ============================================================

def assert_source_numbers():

    text = read(
        SOURCE_DOC
    )

    missing = []


    for name, values in (
        EXPECTED.items()
    ):

        for value in values:

            if value not in text:

                missing.append(
                    f"{name}: {value}"
                )


    extra_required = [

        "9 / 20",
        "12 / 20",

        "-0.001852",

        "[-0.016685, +0.012170]",

        "0.5918",

        "-0.004880",

        "[-0.013676, +0.003981]",

        "0.8621",

        "0.471829",

        "0.154054",

        "5.015625",
    ]


    for value in extra_required:

        if value not in text:

            missing.append(
                f"required: {value}"
            )


    if missing:

        raise RuntimeError(
            "Scientific source is missing "
            "expected verified values:\n  "
            +
            "\n  ".join(
                missing
            )
        )


# ============================================================
# Navigation helpers
# ============================================================

def rel_href(
    html_path,
    target,
):

    relative = (
        html_path
        .relative_to(
            SITE
        )
    )

    depth = (
        len(
            relative.parents
        )
        - 1
    )

    prefix = (
        "../"
        * depth
    )

    return (
        prefix
        + target
    )


def nav_html(
    html_path,
    active,
):

    items = []


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


        items.append(

            f'<a data-page="{key}" '
            f'href="{rel_href(html_path, target)}"'
            f'{attrs}>'

            f"<span>{number}</span>"
            f"{label}"

            "</a>"
        )


    return (
        '<nav class="nav">'
        +
        "".join(
            items
        )
        +
        "</nav>"
    )


# ============================================================
# Normalize nav + site status
# ============================================================

def normalize_nav_and_global_status(
    path,
):

    text = read(
        path
    )


    match = re.search(
        r'<body data-page="([^"]+)"',
        text,
    )


    if not match:

        raise RuntimeError(
            f"Missing body data-page in {path}"
        )


    active = (
        match.group(
            1
        )
    )


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

        r'<div class="status">'
        r'<i class="dot"></i>'
        r'.*?</div>',

        '<div class="status">'
        '<i class="dot"></i>'
        'V1 · Science frozen'
        '</div>',

        text,

        count=1,

        flags=re.S,
    )


    text = text.replace(

        "Corrected V0 freeze · September 2026",

        (
            "V1 scientific freeze · "
            "V0 remains selected · "
            "September 2026"
        ),
    )


    write(
        path,
        text,
    )


# ============================================================
# Create V1 page
# ============================================================

def make_v1_page():

    template_path = (
        PAGES
        / "models.html"
    )

    template = read(
        template_path
    )


    lead = (

        "A controlled sequence of hierarchical, "
        "calibration, robustness, and temporal "
        "experiments. H2 was promising, but V0 "
        "remains the selected development baseline."
    )


    body = r'''
<header class="hero">

  <div class="kicker">
    V1 scientific freeze
  </div>

  <h1>
    Understanding V1
  </h1>

  <p class="lead">
    A controlled sequence of experiments asked whether
    task structure or longer history could improve the
    timing-corrected V0 forecast. The answer was
    scientifically useful but mixed: H2 slightly improved
    pooled probability metrics, yet not robustly enough
    to replace V0.
  </p>

  <div class="badges">

    <span class="badge ok">
      Scientific side frozen
    </span>

    <span class="badge">
      20 development matches
    </span>

    <span class="badge warn">
      V0 remains selected
    </span>

  </div>

</header>


<div class="research-board">

  <div class="board-story">

    <div class="kicker">
      Final V1 decision
    </div>

    <h2>
      Better pooled probability quality
      was not enough.
    </h2>

    <p>
      H2 reached the lowest pooled OOF log loss,
      but the gain over V0 was tiny, only 9 of 20
      matches improved on log loss, and the equal-match
      bootstrap interval crossed zero.
    </p>

    <div class="outcomes">

      <span class="outcome">
        V0 SELECTED
      </span>

      <span class="outcome">
        H2 PRESERVED
      </span>

      <span class="outcome">
        T0 REJECTED
      </span>

    </div>

  </div>


  <div class="board-metrics">

    <div class="metric">

      <span>
        Selected V0 LL ↓
      </span>

      <strong>
        0.838182
      </strong>

      <small>
        XGB-A5
      </small>

    </div>


    <div class="metric">

      <span>
        Best pooled V1 LL ↓
      </span>

      <strong>
        0.837397
      </strong>

      <small>
        H2, not promoted
      </small>

    </div>


    <div class="metric">

      <span>
        H2 LL match wins
      </span>

      <strong>
        9/20
      </strong>

      <small>
        Weak robustness
      </small>

    </div>


    <div class="metric">

      <span>
        T0 LL ↓
      </span>

      <strong>
        0.844859
      </strong>

      <small>
        Temporal baseline lost
      </small>

    </div>

  </div>

</div>


<section class="section">

  <div class="section-head">

    <h2>
      The V1 experiment chain
    </h2>

    <p>
      Each step changed one motivated part of the system,
      then stopped when the evidence gate failed.
    </p>

  </div>


  <div class="flow">

    <div class="node">
      <b>V0</b>
      <br>
      corrected baseline
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>H0</b>
      <br>
      hierarchy only
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>H1</b>
      <br>
      specialized heads
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>H2</b>
      <br>
      nested calibration
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Robustness</b>
      <br>
      match bootstrap
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>T0</b>
      <br>
      3s / 5s history
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Freeze</b>
      <br>
      V0 retained
    </div>

  </div>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      The hierarchy hypothesis
    </h2>

    <p>
      Split plant feasibility from site choice,
      then compose a valid three-class forecast.
    </p>

  </div>


  <div class="grid">

    <article class="card half">

      <div class="stat-label">
        Head 1
      </div>

      <h3>
        Plant feasibility
      </h3>

      <p>
        <code>q = P(Plant | X)</code>
        <br>
        H1 uses geometry + motion + combat + defense,
        36 features.
      </p>

    </article>


    <article class="card half">

      <div class="stat-label">
        Head 2
      </div>

      <h3>
        Conditional site choice
      </h3>

      <p>
        <code>r = P(A | Plant, X)</code>
        <br>
        H1 uses geometry + motion + defense,
        27 features. Combat was deliberately removed.
      </p>

    </article>

  </div>


  <pre><code>P(A)  = q * r
P(B)  = q * (1 - r)
P(NO) = 1 - q</code></pre>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      What changed from H0 to H2
    </h2>

    <p>
      Hierarchy alone lost. Feature specialization
      recovered most of the gap. Calibration then
      improved the Site Head probabilities.
    </p>

  </div>


  <div
    class="table-wrap"
    role="region"
    aria-label="V1 model comparison"
    tabindex="0"
  >

    <table>

      <thead>

        <tr>
          <th>Model</th>
          <th>Log loss ↓</th>
          <th>Brier ↓</th>
          <th>Accuracy ↑</th>
          <th>Macro F1 ↑</th>
          <th>Decision</th>
        </tr>

      </thead>


      <tbody>

        <tr class="best">

          <td>
            V0 XGB-A5
          </td>

          <td>
            0.838182
          </td>

          <td>
            0.503190
          </td>

          <td>
            0.594899
          </td>

          <td>
            0.566090
          </td>

          <td>
            Selected baseline
          </td>

        </tr>


        <tr>

          <td>
            V1-H0
          </td>

          <td>
            0.852546
          </td>

          <td>
            0.504516
          </td>

          <td>
            0.592527
          </td>

          <td>
            0.566455
          </td>

          <td>
            Reject
          </td>

        </tr>


        <tr>

          <td>
            V1-H1
          </td>

          <td>
            0.840649
          </td>

          <td>
            0.499879
          </td>

          <td>
            0.590154
          </td>

          <td>
            0.564673
          </td>

          <td>
            Promising
          </td>

        </tr>


        <tr>

          <td>
            V1-H2
          </td>

          <td>
            <b>0.837397</b>
          </td>

          <td>
            <b>0.499781</b>
          </td>

          <td>
            0.587189
          </td>

          <td>
            0.555824
          </td>

          <td>
            Promising, not promoted
          </td>

        </tr>


        <tr>

          <td>
            V1-T0
          </td>

          <td>
            0.844859
          </td>

          <td>
            0.507984
          </td>

          <td>
            0.593120
          </td>

          <td>
            0.563414
          </td>

          <td>
            Reject
          </td>

        </tr>

      </tbody>

    </table>

  </div>


  <div class="callout">

    <strong>
      Important:
    </strong>

    H2 has the best pooled OOF log loss and Brier
    in this table, but this is development evidence
    on the same 20 matches. It is not a robust or
    independent win over V0.

  </div>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      H2 improved Site probability quality
    </h2>

    <p>
      Nested Platt calibration changed only the Site
      Head probability mapping.
    </p>

  </div>


  <div class="grid">

    <article class="card half">

      <div class="stat-label">
        Site log loss
      </div>

      <div class="stat">
        0.478017 → 0.471829
      </div>

      <p>
        Probability quality improved.
      </p>

    </article>


    <article class="card half">

      <div class="stat-label">
        Site Brier
      </div>

      <div class="stat">
        0.156244 → 0.154054
      </div>

      <p>
        Calibration improved without changing
        the underlying Site classifier accuracy.
      </p>

    </article>

  </div>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      Why H2 was not promoted
    </h2>

    <p>
      Per-match behavior weakened the pooled result.
    </p>

  </div>


  <div
    class="table-wrap"
    role="region"
    aria-label="H2 robustness"
    tabindex="0"
  >

    <table>

      <thead>

        <tr>
          <th>Evidence</th>
          <th>Log loss</th>
          <th>Brier</th>
        </tr>

      </thead>


      <tbody>

        <tr>

          <td>
            Matches improved
          </td>

          <td>
            9 / 20
          </td>

          <td>
            12 / 20
          </td>

        </tr>


        <tr>

          <td>
            Equal-match mean Δ
          </td>

          <td>
            -0.001852
          </td>

          <td>
            -0.004880
          </td>

        </tr>


        <tr>

          <td>
            95% bootstrap CI
          </td>

          <td>
            [-0.016685, +0.012170]
          </td>

          <td>
            [-0.013676, +0.003981]
          </td>

        </tr>


        <tr>

          <td>
            P(mean Δ &lt; 0)
          </td>

          <td>
            0.5918
          </td>

          <td>
            0.8621
          </td>

        </tr>

      </tbody>

    </table>

  </div>


  <p>
    The log-loss median delta was
    <b>+0.007943</b>,
    meaning the typical match slightly favored V0
    even though the equal-match mean favored H2.
    Both bootstrap intervals cross zero.
  </p>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      T0 tested longer temporal information
    </h2>

    <p>
      Before considering a sequence neural network,
      a simple controlled baseline tested whether
      longer history itself added useful signal.
    </p>

  </div>


  <div class="grid">

    <article class="card half">

      <div class="stat-label">
        Feature contract
      </div>

      <div class="stat">
        37 + 34 = 71
      </div>

      <p>
        Frozen V0 features plus 34 hand-designed
        3s / 5s state deltas.
      </p>

    </article>


    <article class="card half">

      <div class="stat-label">
        Timing QA
      </div>

      <div class="stat">
        1,686 rows
      </div>

      <p>
        3s exact: 1686/1686.
        5s exact: 1685/1686.
        One parser gap used 5.015625 seconds.
      </p>

    </article>

  </div>


  <div class="callout warn">

    <strong>
      T0 failed the gate.
    </strong>

    Log loss worsened from 0.838182 to 0.844859
    and Brier worsened from 0.503190 to 0.507984.

    A GRU, LSTM, Transformer, or temporal convolution
    was therefore not justified.

  </div>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      What V1 actually taught us
    </h2>

    <p>
      A negative or mixed result is useful when the
      experiment was controlled.
    </p>

  </div>


  <div class="finding-list">

    <article class="finding-row">

      <span class="finding-index">
        01
      </span>

      <div>

        <h3>
          Task decomposition is useful diagnostically.
        </h3>

        <p>
          Plant feasibility and site choice behave like
          different predictive subproblems.
        </p>

      </div>

      <div class="finding-value">
        H0 → H1
        <small>
          Specialization mattered
        </small>
      </div>

    </article>


    <article class="finding-row">

      <span class="finding-index">
        02
      </span>

      <div>

        <h3>
          Probability quality and classification
          accuracy are different.
        </h3>

        <p>
          H2 improved Site Head log loss and Brier
          even though classification accuracy did not
          meaningfully change.
        </p>

      </div>

      <div class="finding-value">
        0.471829
        <small>
          Calibrated Site LL
        </small>
      </div>

    </article>


    <article class="finding-row">

      <span class="finding-index">
        03
      </span>

      <div>

        <h3>
          Pooled OOF is not enough for a tiny gain.
        </h3>

        <p>
          Equal-match robustness changed the
          interpretation of the nominal H2 improvement.
        </p>

      </div>

      <div class="finding-value">
        9 / 20
        <small>
          H2 LL match wins
        </small>
      </div>

    </article>


    <article class="finding-row">

      <span class="finding-index">
        04
      </span>

      <div>

        <h3>
          More history is not automatically more signal.
        </h3>

        <p>
          The 71-feature T0 model was worse than the
          37-feature V0 model.
        </p>

      </div>

      <div class="finding-value">
        0.844859
        <small>
          T0 log loss
        </small>
      </div>

    </article>

  </div>

</section>


<section class="section">

  <div class="section-head">

    <h2>
      Scientific freeze
    </h2>

    <p>
      The next meaningful gain should preferably come
      from new information or new data, not repeated
      tuning against the same 20 matches.
    </p>

  </div>


  <div class="callout">

    <strong>
      Selected development baseline:
      V0 XGB-A5.
    </strong>

    H2 is preserved as a promising research result.
    T0 is rejected.
    Sequence modeling is not justified by the
    current evidence.

  </div>


  <p>
    <a
      href="../assets/sources/v1_scientific_record.md"
      download
    >
      Download the complete V1 scientific record ↓
    </a>
  </p>

</section>
'''


    template = re.sub(

        r"<title>.*?</title>",

        (
            "<title>"
            "Understanding V1 — "
            "CS2 Tactical Intelligence Lab"
            "</title>"
        ),

        template,

        count=1,

        flags=re.S,
    )


    template = re.sub(

        r'<meta name="description" '
        r'content="[^"]*">',

        (
            '<meta name="description" '
            f'content="{lead}">'
        ),

        template,

        count=1,
    )


    template = re.sub(

        r'<body data-page="[^"]+">',

        '<body data-page="v1">',

        template,

        count=1,
    )


    template = re.sub(

        r'<span class="breadcrumb">'
        r'.*?</span>',

        (
            '<span class="breadcrumb">'
            'The notebook '
            '<span>/</span> '
            'Understanding V1'
            '</span>'
        ),

        template,

        count=1,

        flags=re.S,
    )


    template = re.sub(

        r'<span class="top-meta">'
        r'.*?</span>',

        (
            '<span class="top-meta">'
            'MIRAGE '
            '<span>·</span> '
            'V1 SCIENCE FREEZE'
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
        +
        len(
            '<div class="content">'
        )
    )


    footer_start = (
        template.index(
            '<footer class="footer">',
            start,
        )
    )


    footer_end = (

        template.index(
            "</footer>",
            footer_start,
        )

        +

        len(
            "</footer>"
        )
    )


    footer = (

        '<footer class="footer">'

        '<a href="../index.html">'
        'CS2 Tactical Intelligence Lab'
        '</a>'

        '<span>'
        'V1 scientific freeze · '
        'V0 remains selected · '
        'September 2026'
        '</span>'

        '<a href="../pages/evidence.html#v1-freeze">'
        'Inspect the V1 evidence ↗'
        '</a>'

        '</footer>'
    )


    output = (

        template[
            :start
        ]

        +

        "\n"

        +

        body.strip()

        +

        "\n"

        +

        footer

        +

        template[
            footer_end:
        ]
    )


    write(
        PAGES
        / "v1.html",
        output,
    )


# ============================================================
# Insert blocks into existing pages
# ============================================================

def insert_before_footer(
    path,
    block,
):

    text = read(
        path
    )


    if MARKER in text:

        return


    index = text.find(
        '<footer class="footer">'
    )


    if index < 0:

        raise RuntimeError(
            f"Footer not found in {path}"
        )


    write(

        path,

        (
            text[:index]
            +
            MARKER
            +
            "\n"
            +
            block.strip()
            +
            "\n"
            +
            text[index:]
        ),
    )


def update_existing_pages():

    # --------------------------------------------------------
    # Homepage
    # --------------------------------------------------------

    insert_before_footer(

        SITE
        / "index.html",

        r'''
<section
  class="section"
  id="v1-freeze"
>

  <div class="section-head">

    <h2>
      V1 is scientifically frozen
    </h2>

    <p>
      The hierarchy nearly matched V0 on pooled
      probability metrics, but robustness did not
      support promotion.
    </p>

  </div>


  <div class="grid">

    <article class="card half">

      <div class="stat-label">
        Best V1 pooled result
      </div>

      <h3>
        H2 calibrated hierarchy
      </h3>

      <p>
        LL 0.837397 and Brier 0.499781,
        but only 9/20 matches improved on log loss
        and the bootstrap interval crossed zero.
      </p>

    </article>


    <article class="card half">

      <div class="stat-label">
        Final decision
      </div>

      <h3>
        V0 remains selected
      </h3>

      <p>
        T0 temporal history also failed its gate.
        The next step should seek new information
        or new data rather than more tuning on the
        same 20 matches.
      </p>

    </article>

  </div>


  <p>

    <a
      class="button"
      href="pages/v1.html"
    >
      Understand V1
      <span>↗</span>
    </a>

  </p>

</section>
''',
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    insert_before_footer(

        PAGES
        / "models.html",

        r'''
<section
  class="section"
  id="v1-model-selection"
>

  <div class="section-head">

    <h2>
      V1 model-selection result
    </h2>

    <p>
      H2 won pooled probability metrics by a tiny margin,
      but not robustly enough to replace V0.
    </p>

  </div>


  <div
    class="table-wrap"
    role="region"
    aria-label="V1 model selection"
    tabindex="0"
  >

    <table>

      <thead>

        <tr>
          <th>Model</th>
          <th>Log loss ↓</th>
          <th>Brier ↓</th>
          <th>Accuracy ↑</th>
          <th>Macro F1 ↑</th>
          <th>Status</th>
        </tr>

      </thead>


      <tbody>

        <tr class="best">
          <td>V0 XGB-A5</td>
          <td>0.838182</td>
          <td>0.503190</td>
          <td>0.594899</td>
          <td>0.566090</td>
          <td>Selected</td>
        </tr>

        <tr>
          <td>H0</td>
          <td>0.852546</td>
          <td>0.504516</td>
          <td>0.592527</td>
          <td>0.566455</td>
          <td>Rejected</td>
        </tr>

        <tr>
          <td>H1</td>
          <td>0.840649</td>
          <td>0.499879</td>
          <td>0.590154</td>
          <td>0.564673</td>
          <td>Promising</td>
        </tr>

        <tr>
          <td>H2</td>
          <td>0.837397</td>
          <td>0.499781</td>
          <td>0.587189</td>
          <td>0.555824</td>
          <td>Preserved, not promoted</td>
        </tr>

        <tr>
          <td>T0</td>
          <td>0.844859</td>
          <td>0.507984</td>
          <td>0.593120</td>
          <td>0.563414</td>
          <td>Rejected</td>
        </tr>

      </tbody>

    </table>

  </div>


  <p>
    <a href="v1.html">
      Read the complete V1 experiment chain ↗
    </a>
  </p>

</section>
''',
    )


    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    insert_before_footer(

        PAGES
        / "evidence.html",

        r'''
<section
  class="section"
  id="v1-freeze"
>

  <div class="section-head">

    <h2>
      V1 scientific freeze
    </h2>

    <p>
      Development evidence from the same
      20 frozen match groups.
    </p>

  </div>


  <div
    class="table-wrap"
    role="region"
    aria-label="V1 robustness evidence"
    tabindex="0"
  >

    <table>

      <thead>

        <tr>
          <th>Evidence</th>
          <th>Log loss</th>
          <th>Brier</th>
        </tr>

      </thead>


      <tbody>

        <tr>
          <td>V0 pooled</td>
          <td>0.838182</td>
          <td>0.503190</td>
        </tr>

        <tr>
          <td>H2 pooled</td>
          <td>0.837397</td>
          <td>0.499781</td>
        </tr>

        <tr>
          <td>H2 match wins</td>
          <td>9 / 20</td>
          <td>12 / 20</td>
        </tr>

        <tr>
          <td>Equal-match mean Δ</td>
          <td>-0.001852</td>
          <td>-0.004880</td>
        </tr>

        <tr>
          <td>95% bootstrap CI</td>
          <td>[-0.016685, +0.012170]</td>
          <td>[-0.013676, +0.003981]</td>
        </tr>

        <tr>
          <td>P(mean Δ &lt; 0)</td>
          <td>0.5918</td>
          <td>0.8621</td>
        </tr>

      </tbody>

    </table>

  </div>


  <div class="callout">

    <strong>
      Interpretation:
    </strong>

    H2's pooled advantage is too small and too unstable
    across matches to justify replacing V0.

    T0 also lost on both primary probability metrics.

  </div>


  <p>
    <a href="v1.html">
      Read the V1 scientific record ↗
    </a>
  </p>

</section>
''',
    )


    # --------------------------------------------------------
    # Roadmap
    # --------------------------------------------------------

    insert_before_footer(

        PAGES
        / "roadmap.html",

        r'''
<section
  class="section"
  id="v1-freeze"
>

  <div class="section-head">

    <h2>
      V1 scientific branch is closed
    </h2>

    <p>
      The project now has an evidence boundary
      for what not to tune further.
    </p>

  </div>


  <div class="flow">

    <div class="node">
      <b>Hierarchy</b>
      <br>
      promising
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Calibration</b>
      <br>
      helpful
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Robustness</b>
      <br>
      insufficient
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Temporal deltas</b>
      <br>
      rejected
    </div>

    <span class="arrow">
      →
    </span>

    <div class="node">
      <b>Next</b>
      <br>
      new information / data
    </div>

  </div>


  <div class="callout">

    <strong>
      Next research direction:
    </strong>

    preserve V0 as the baseline and seek genuinely new
    signal, such as more untouched matches, map-aware
    semantic positions, richer tactical state, player
    roles, or a clearly motivated event sequence
    representation.

  </div>

</section>
''',
    )


    # --------------------------------------------------------
    # Decisions
    # --------------------------------------------------------

    insert_before_footer(

        PAGES
        / "decisions.html",

        r'''
<section
  class="section"
  id="v1-freeze"
>

  <div class="section-head">

    <h2>
      Decision: do not promote V1
    </h2>

    <p>
      A better pooled score is not sufficient when
      the improvement is tiny and match-level robustness
      is weak.
    </p>

  </div>


  <div class="grid">

    <article class="card half">

      <h3>
        Preserve H2
      </h3>

      <p>
        Nested Site calibration improved probability
        quality and produced the best pooled V1 result.

        Keep it as a research finding.
      </p>

    </article>


    <article class="card half">

      <h3>
        Keep V0 selected
      </h3>

      <p>
        H2 won log loss on only 9/20 matches and its
        equal-match bootstrap interval crossed zero.

        T0 also failed.

        No sequence model is justified.
      </p>

    </article>

  </div>

</section>
''',
    )


# ============================================================
# Manifest
# ============================================================

def update_manifest():

    data = json.loads(
        read(
            MANIFEST
        )
    )


    pages = data.get(
        "pages",
        [],
    )


    if (
        "pages/v1.html"
        not in pages
    ):

        try:

            index = (
                pages.index(
                    "pages/v0.html"
                )
                + 1
            )

        except ValueError:

            index = 1


        pages.insert(
            index,
            "pages/v1.html",
        )


    data.update({

        "stage":
            "V1 scientific freeze",

        "v1_status":
            (
                "SCIENTIFICALLY FROZEN · "
                "H2 PROMISING NOT PROMOTED · "
                "T0 REJECTED"
            ),

        "selected_model":
            (
                "V0 XGB-A5 · "
                "corrected timing"
            ),

        "v1_best_pooled_model":
            (
                "H2 calibrated hierarchy"
            ),

        "v1_best_pooled_log_loss":
            0.837397,

        "v1_decision":
            (
                "V0 remains selected; "
                "seek new information/data "
                "before more tuning"
            ),

        "pages":
            pages,
    })


    write(

        MANIFEST,

        (
            json.dumps(
                data,
                indent=2,
            )
            +
            "\n"
        ),
    )


# ============================================================
# Copy source document
# ============================================================

def copy_source():

    SOURCE_COPY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    shutil.copyfile(
        SOURCE_DOC,
        SOURCE_COPY,
    )


# ============================================================
# All HTML files
# ============================================================

def all_html_files():

    return sorted([

        SITE
        / "index.html",

        *PAGES.glob(
            "*.html"
        ),
    ])


# ============================================================
# Cache bust CSS
# ============================================================

def update_cache_bust():

    for path in all_html_files():

        text = read(
            path
        )


        text = re.sub(

            r'assets/style\.css\?v=[^"]+',

            (
                "assets/style.css"
                "?v=v1-science-freeze"
            ),

            text,
        )


        text = re.sub(

            r'\.\./assets/style\.css'
            r'\?v=[^"]+',

            (
                "../assets/style.css"
                "?v=v1-science-freeze"
            ),

            text,
        )


        write(
            path,
            text,
        )


# ============================================================
# Validation
# ============================================================

def validate_nav():

    for path in all_html_files():

        text = read(
            path
        )


        active_count = len(
            re.findall(
                (
                    r'class="active" '
                    r'aria-current="page"'
                ),
                text,
            )
        )


        if active_count != 1:

            raise RuntimeError(
                f"{path}: expected 1 active "
                f"nav entry, found "
                f"{active_count}"
            )


        if (
            'data-page="v1"'
            not in text
        ):

            raise RuntimeError(
                f"{path}: V1 nav entry missing"
            )


def validate_local_links():

    broken = []


    for path in all_html_files():

        text = read(
            path
        )


        hrefs = re.findall(
            r'href="([^"]+)"',
            text,
        )


        for href in hrefs:

            if href.startswith(
                (
                    "#",
                    "http://",
                    "https://",
                    "mailto:",
                    "javascript:",
                )
            ):

                continue


            clean = (
                urlsplit(
                    href
                )
                .path
            )


            if not clean:

                continue


            target = (
                path.parent
                / clean
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
                        path.relative_to(
                            ROOT
                        ),
                        href,
                    )
                )


    if broken:

        raise RuntimeError(

            "Broken local links:\n"

            +

            "\n".join(

                (
                    f"  {path}: "
                    f"{href}"
                )

                for path, href
                in broken
            )
        )


def validate_science():

    v1 = read(
        PAGES
        / "v1.html"
    )


    evidence = read(
        PAGES
        / "evidence.html"
    )


    for name, values in (
        EXPECTED.items()
    ):

        for value in values:

            if value not in v1:

                raise RuntimeError(
                    f"V1 page missing "
                    f"{name} value "
                    f"{value}"
                )


    required = [

        "9 / 20",
        "12 / 20",

        "[-0.016685, +0.012170]",

        "[-0.013676, +0.003981]",

        "0.5918",
        "0.8621",

        "5.015625",

        "V0 XGB-A5",
    ]


    for value in required:

        if value not in v1:

            raise RuntimeError(
                f"V1 page missing "
                f"required evidence "
                f"{value}"
            )


    for value in [

        "0.837397",
        "0.838182",
        "9 / 20",
        "12 / 20",

    ]:

        if value not in evidence:

            raise RuntimeError(
                "Evidence page missing "
                f"V1 value {value}"
            )


# ============================================================
# Main
# ============================================================

def main():

    required_paths = [

        SITE,

        PAGES,

        SOURCE_DOC,

        MANIFEST,

        PAGES
        / "models.html",

        PAGES
        / "evidence.html",

        PAGES
        / "roadmap.html",

        PAGES
        / "decisions.html",

        SITE
        / "index.html",
    ]


    for path in required_paths:

        require(
            path
        )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "V1 WEBSITE SCIENTIFIC FREEZE"
    )

    print(
        "=" * 100
    )


    print(
        "\n[1/8] Validate scientific source"
    )

    assert_source_numbers()

    print(
        "  ✅ verified V1 scientific numbers"
    )


    print(
        "\n[2/8] Create Understanding V1 page"
    )

    make_v1_page()

    print(
        "  ✅ site/pages/v1.html"
    )


    print(
        "\n[3/8] Update existing research pages"
    )

    update_existing_pages()

    print(
        "  ✅ overview"
    )

    print(
        "  ✅ models"
    )

    print(
        "  ✅ evidence"
    )

    print(
        "  ✅ roadmap"
    )

    print(
        "  ✅ decisions"
    )


    print(
        "\n[4/8] Copy frozen scientific source"
    )

    copy_source()

    print(
        "  ✅ site/assets/sources/"
        "v1_scientific_record.md"
    )


    print(
        "\n[5/8] Update site manifest"
    )

    update_manifest()

    print(
        "  ✅ site/site-manifest.json"
    )


    print(
        "\n[6/8] Normalize navigation"
    )

    for path in all_html_files():

        normalize_nav_and_global_status(
            path
        )

    print(
        "  ✅ Understanding V1 added site-wide"
    )


    print(
        "\n[7/8] Update CSS cache key"
    )

    update_cache_bust()

    print(
        "  ✅ v1-science-freeze"
    )


    print(
        "\n[8/8] Validate site"
    )

    validate_nav()

    print(
        "  ✅ exactly one active nav per page"
    )


    validate_science()

    print(
        "  ✅ scientific numbers match freeze record"
    )


    validate_local_links()

    print(
        "  ✅ local links exist"
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "✅ V1 WEBSITE FREEZE UPDATE COMPLETE"
    )

    print(
        "=" * 100
    )


    print(
        "\nCreated / updated:"
    )


    targets = [

        PAGES
        / "v1.html",

        SITE
        / "index.html",

        PAGES
        / "models.html",

        PAGES
        / "evidence.html",

        PAGES
        / "roadmap.html",

        PAGES
        / "decisions.html",

        SOURCE_COPY,

        MANIFEST,
    ]


    for path in targets:

        print(
            " ",
            path.relative_to(
                ROOT
            ),
        )


    print(
        "\nNavigation/status also updated "
        "across all site HTML pages."
    )


    print(
        "\nNow run:"
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
