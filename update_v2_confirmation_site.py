#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
SITE = ROOT / "site"
PAGES = SITE / "pages"

VALIDATION_SOURCE = (
    ROOT / "docs" / "v2_confirm_evaluation_validation.md"
)

VALIDATION_COPY = (
    SITE / "assets" / "sources" /
    "v2_confirm_evaluation_validation.md"
)

MARKER = "<!-- V2-CONFIRMATION-FINAL -->"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(
        text,
        encoding="utf-8",
    )


def insert_before_footer(
    path: Path,
    block: str,
) -> None:
    text = read(path)

    if MARKER in text:
        return

    pos = text.find(
        '<footer class="footer">'
    )

    if pos < 0:
        raise RuntimeError(
            f"Footer not found: {path}"
        )

    text = (
        text[:pos]
        + MARKER
        + "\n"
        + block.strip()
        + "\n"
        + text[pos:]
    )

    write(
        path,
        text,
    )


# ============================================================
# 1. Global project status
# ============================================================

for path in [
    SITE / "index.html",
    *PAGES.glob("*.html"),
]:
    text = read(path)

    text = text.replace(
        "V2 · Data integrity passed",
        "V2 · Confirmation complete",
    )

    text = text.replace(
        "V2 DATA INTEGRITY PASS",
        "V2 CONFIRMATION COMPLETE",
    )

    text = text.replace(
        "style.css?v=v2-integrity",
        "style.css?v=v2-confirmation",
    )

    write(
        path,
        text,
    )


# ============================================================
# 2. Rewrite V2 hero
# ============================================================

v2_path = PAGES / "v2.html"
text = read(v2_path)

new_hero = """
<header class="hero">
  <div class="kicker">
    V2 confirmation · independent evaluation complete
  </div>

  <h1>V2 Independent Confirmation</h1>

  <p class="lead">
    The untouched 30-match confirmation corpus first exposed
    hidden data-integrity failures before model scoring.
    Those failures were repaired, the development labels were
    rebuilt without retuning, corrected V0 and H2 models were
    frozen and verified, and D_CONFIRM was then evaluated once.
    V0 remains the primary model.
  </p>

  <div class="badges">
    <span class="badge ok">
      Confirmation complete
    </span>

    <span class="badge">
      30 frozen matches
    </span>

    <span class="badge">
      2269 observations
    </span>

    <span class="badge ok">
      One formal evaluation
    </span>
  </div>
</header>
"""

text, count = re.subn(
    r'<header class="hero">.*?</header>',
    new_hero.strip(),
    text,
    count=1,
    flags=re.S,
)

if count != 1:
    raise RuntimeError(
        "Could not replace V2 hero."
    )


# ============================================================
# 3. Rewrite old Current checkpoint section
# ============================================================

checkpoint = """
<section class="section">
  <div class="section-head">
    <h2>Final checkpoint</h2>

    <p>
      V2 is complete. The corrected development experiment
      and untouched confirmation experiment point in the
      same model-selection direction.
    </p>
  </div>

  <div class="grid">

    <article class="card">
      <div class="stat-label">
        D_CONFIRM
      </div>

      <div class="stat">
        30
      </div>

      <p>
        Frozen confirmation matches containing
        2269 feature-valid observations.
      </p>
    </article>

    <article class="card">
      <div class="stat-label">
        Corrected V0
      </div>

      <div class="stat">
        0.835090
      </div>

      <p>
        Confirmation multiclass log loss.
      </p>
    </article>

    <article class="card">
      <div class="stat-label">
        Corrected H2
      </div>

      <div class="stat">
        0.836643
      </div>

      <p>
        Confirmation multiclass log loss.
      </p>
    </article>

  </div>
</section>
"""

text, count = re.subn(
    r'<section class="section">\s*'
    r'<div class="section-head">\s*'
    r'<h2>Current checkpoint</h2>'
    r'.*?</section>',
    checkpoint.strip(),
    text,
    count=1,
    flags=re.S,
)

if count != 1:
    raise RuntimeError(
        "Could not replace V2 current checkpoint."
    )

write(
    v2_path,
    text,
)


# ============================================================
# 4. Final scientific result block for V2
# ============================================================

v2_result_block = """
<section class="section">
  <div class="section-head">
    <h2>Independent confirmation result</h2>

    <p>
      Frozen corrected models were evaluated once on
      untouched D_CONFIRM. No tuning followed the result.
    </p>
  </div>

  <div class="table-wrap"
       role="region"
       aria-label="V2 confirmation results"
       tabindex="0">

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Log loss</th>
          <th>Brier</th>
          <th>Accuracy</th>
          <th>Macro F1</th>
        </tr>
      </thead>

      <tbody>
        <tr>
          <td>V0</td>
          <td>0.835090</td>
          <td>0.508136</td>
          <td>0.589246</td>
          <td>0.541890</td>
        </tr>

        <tr>
          <td>H2</td>
          <td>0.836643</td>
          <td>0.508533</td>
          <td>0.589246</td>
          <td>0.535487</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="callout">
    <strong>Decision:</strong>
    V0 remains the primary model.
    H2 minus V0 log loss was +0.001553.
    H2 was better on 12 of 30 matches.
    The equal-match bootstrap 95% interval was
    [-0.007585, +0.014124], crossing zero.
  </div>

  <div class="table-wrap"
       role="region"
       aria-label="V2 confirmation results by horizon"
       tabindex="0">

    <table>
      <thead>
        <tr>
          <th>Horizon</th>
          <th>V0 log loss</th>
          <th>H2 log loss</th>
          <th>H2 − V0</th>
        </tr>
      </thead>

      <tbody>
        <tr>
          <td>10 s</td>
          <td>0.902952</td>
          <td>0.903492</td>
          <td>+0.000540</td>
        </tr>

        <tr>
          <td>20 s</td>
          <td>0.858702</td>
          <td>0.857766</td>
          <td>−0.000936</td>
        </tr>

        <tr>
          <td>30 s</td>
          <td>0.793310</td>
          <td>0.800057</td>
          <td>+0.006747</td>
        </tr>

        <tr>
          <td>40 s</td>
          <td>0.773230</td>
          <td>0.772956</td>
          <td>−0.000274</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p>
    The H2 effect was not consistently favorable across
    horizons. Its Site-head calibration improvement on the
    development corpus did not translate into an improvement
    of the primary three-class confirmation metric.
  </p>

  <p>
    <a href="../assets/sources/v2_confirm_evaluation_validation.md">
      Read the confirmation validation record ↗
    </a>
  </p>
</section>
"""

insert_before_footer(
    v2_path,
    v2_result_block,
)


# ============================================================
# 5. Results & evidence
# ============================================================

evidence_block = """
<section class="section">
  <div class="section-head">
    <h2>V2 · Independent confirmation</h2>

    <p>
      The untouched confirmation experiment is now complete.
    </p>
  </div>

  <article class="evidence">
    <div class="e-head">
      <span class="eid">V2-CONFIRM</span>
      <span>Independent confirmation</span>
    </div>

    <div class="e-body">
      <h3>
        H2 did not confirm an improvement over V0.
      </h3>

      <p>
        Across 2269 observations from 30 frozen Mirage
        matches, V0 achieved log loss 0.835090 and H2
        achieved 0.836643.
      </p>

      <p>
        H2 was better on 12 / 30 matches.
        Equal-match mean log-loss delta H2−V0 was
        +0.003106 with bootstrap 95% CI
        [-0.007585, +0.014124].
      </p>

      <p>
        Therefore V0 remains the primary model and
        H2 is not promoted.
      </p>
    </div>
  </article>
</section>
"""

insert_before_footer(
    PAGES / "evidence.html",
    evidence_block,
)


# ============================================================
# 6. Models
# ============================================================

models_block = """
<section class="section">
  <div class="section-head">
    <h2>V2 model status</h2>

    <p>
      Independent confirmation finalized the model-selection
      decision.
    </p>
  </div>

  <div class="grid">
    <article class="card half">
      <h3>V0 · Primary</h3>

      <p>
        Frozen 37-feature XGB-A5.
        D_CONFIRM log loss: <b>0.835090</b>.
      </p>
    </article>

    <article class="card half">
      <h3>H2 · Secondary challenger</h3>

      <p>
        Hierarchical Plant/Site model with nested
        Site-head calibration.
        D_CONFIRM log loss: <b>0.836643</b>.
        Not promoted.
      </p>
    </article>
  </div>
</section>
"""

insert_before_footer(
    PAGES / "models.html",
    models_block,
)


# ============================================================
# 7. Decisions
# ============================================================

decisions_block = """
<section class="section">
  <div class="section-head">
    <h2>Decision from V2 confirmation</h2>

    <p>
      Independent evidence closes the V2 model-selection
      question.
    </p>
  </div>

  <article class="evidence">
    <div class="e-head">
      <span class="eid">ADR-V2-001</span>
      <span>Accepted after confirmation</span>
    </div>

    <div class="e-body">
      <h3>Keep V0 as the primary model</h3>

      <p>
        Corrected development evaluation already favored V0
        on the primary three-class log-loss metric.
        Untouched D_CONFIRM preserved the same direction:
        V0 0.835090 versus H2 0.836643.
      </p>

      <p>
        H2 is retained as a documented research challenger,
        but there is no promotion.
      </p>
    </div>
  </article>
</section>
"""

insert_before_footer(
    PAGES / "decisions.html",
    decisions_block,
)


# ============================================================
# 8. Roadmap
# ============================================================

roadmap_block = """
<section class="section">
  <div class="section-head">
    <h2>V2 closed</h2>

    <p>
      The confirmation gate is complete.
    </p>
  </div>

  <div class="callout">
    <strong>Completed:</strong>
    data-integrity repair → corrected development rebuild →
    corrected model freeze → freeze verification →
    one untouched D_CONFIRM evaluation.
  </div>

  <p>
    V0 remains the primary early-round outcome forecaster.
    The next research version should begin from a new question
    or capability, not from additional tuning against D_CONFIRM.
  </p>
</section>
"""

insert_before_footer(
    PAGES / "roadmap.html",
    roadmap_block,
)


# ============================================================
# 9. Build log
# ============================================================

build_log_block = """
<section class="section">
  <div class="section-head">
    <h2>V2 confirmation closeout</h2>

    <p>
      The experiment ended with the original confirmation
      boundary intact.
    </p>
  </div>

  <div class="timeline">
    <div class="entry">
      <time>01 · Integrity</time>
      <h3>Hidden data defects were repaired before scoring.</h3>
      <p>44 / 1686 development labels changed.</p>
    </div>

    <div class="entry">
      <time>02 · Rebuild</time>
      <h3>V0, H1, and H2 were rebuilt without retuning.</h3>
      <p>V0 remained primary on development log loss.</p>
    </div>

    <div class="entry">
      <time>03 · Freeze</time>
      <h3>Corrected V0 and H2 deployment models were frozen.</h3>
      <p>Hashes, feature ordering, calibration identity, and reload behavior passed verification.</p>
    </div>

    <div class="entry">
      <time>04 · Confirmation</time>
      <h3>D_CONFIRM was evaluated once.</h3>
      <p>V0 0.835090 vs H2 0.836643 log loss.</p>
    </div>
  </div>
</section>
"""

insert_before_footer(
    PAGES / "build-log.html",
    build_log_block,
)


# ============================================================
# 10. Journey
# ============================================================

journey_block = """
<section class="section">
  <div class="section-head">
    <h2>What V2 taught us</h2>

    <p>
      The most important V2 result was not a larger model.
      It was learning to protect the experiment.
    </p>
  </div>

  <div class="callout">
    A confirmation dataset first exposed problems in the data
    pipeline, not superiority in a model. We repaired the
    pipeline before looking at predictions, rebuilt the
    development evidence without retuning, froze the corrected
    models, and only then opened D_CONFIRM once.
  </div>

  <p>
    The result was deliberately uneventful:
    V0 remained slightly better on the primary metric.
    That is useful evidence. The project did not promote H2
    simply because H2 was more complicated or because one
    component of it calibrated better during development.
  </p>
</section>
"""

insert_before_footer(
    PAGES / "journey.html",
    journey_block,
)


# ============================================================
# 11. Overview
# ============================================================

overview_block = """
<section class="section">
  <div class="section-head">
    <h2>V2 independent confirmation complete</h2>

    <p>
      Thirty untouched Mirage matches tested the frozen
      corrected V0 and H2 models.
    </p>
  </div>

  <div class="grid">
    <article class="card">
      <div class="stat-label">V0 log loss</div>
      <div class="stat">0.835090</div>
      <p>Primary model retained.</p>
    </article>

    <article class="card">
      <div class="stat-label">H2 log loss</div>
      <div class="stat">0.836643</div>
      <p>Secondary challenger not promoted.</p>
    </article>

    <article class="card">
      <div class="stat-label">Confirmation matches</div>
      <div class="stat">30</div>
      <p>2269 untouched observations.</p>
    </article>
  </div>

  <p>
    <a href="pages/v2.html">
      Read the complete V2 confirmation record ↗
    </a>
  </p>
</section>
"""

insert_before_footer(
    SITE / "index.html",
    overview_block,
)


# ============================================================
# 12. Repair explicitly stale confirmation wording
# ============================================================

replacements = {
    "D_CONFIRM builds without model scoring.":
        "D_CONFIRM was evaluated once after the corrected models were frozen and verified.",

    "No confirmation scoring yet":
        "Confirmation complete",

    "corrected development baseline rebuild pending":
        "corrected development baseline rebuilt",

    "no D_CONFIRM model predictions or metrics have been calculated":
        "D_CONFIRM was evaluated once after the corrected model freeze",
}

for path in [
    SITE / "index.html",
    *PAGES.glob("*.html"),
]:
    text = read(path)

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    write(
        path,
        text,
    )


# ============================================================
# 13. Copy final validation source
# ============================================================

if not VALIDATION_SOURCE.exists():
    raise FileNotFoundError(
        VALIDATION_SOURCE
    )

VALIDATION_COPY.parent.mkdir(
    parents=True,
    exist_ok=True,
)

shutil.copy2(
    VALIDATION_SOURCE,
    VALIDATION_COPY,
)


# ============================================================
# 14. Manifest
# ============================================================

manifest_path = (
    SITE / "site-manifest.json"
)

if manifest_path.exists():
    manifest = json.loads(
        read(manifest_path)
    )

    manifest[
        "current_stage"
    ] = "V2 independent confirmation complete"

    manifest[
        "v2_confirmation"
    ] = {
        "status":
            "complete",

        "matches":
            30,

        "observations":
            2269,

        "v0_log_loss":
            0.835090,

        "h2_log_loss":
            0.836643,

        "h2_minus_v0_log_loss":
            0.001553,

        "h2_better_matches":
            12,

        "paired_mean_delta":
            0.003106,

        "bootstrap_95_ci": [
            -0.007585,
            0.014124,
        ],

        "decision":
            "V0 remains primary; H2 not promoted",
    }

    write(
        manifest_path,
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
    )


print("✅ V2 confirmation website update complete")
print("✅ Existing debugging history preserved")
print("✅ Final confirmation evidence added")
print("✅ Validation record copied into site sources")
