"""Apply the timing-corrected V0 freeze checkpoint to the static research site."""

from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def replace_content(path: Path, body: str) -> None:
    text = path.read_text()
    start = text.index('<div class="content">') + len('<div class="content">')
    end = text.index('<footer class="footer">', start)
    footer = (
        '<footer class="footer"><a href="../index.html">CS2 Tactical Intelligence Lab</a>'
        '<span>Corrected V0 freeze · September 2026</span>'
        '<a href="../pages/v0.html#limitations">Read the limitations ↗</a></footer>'
    )
    path.write_text(text[:start] + body + footer + text[text.index('</footer>', end) + 9 :])


def table(headers, rows, best=None):
    head = "".join(f'<th scope="col">{value}</th>' for value in headers)
    body = "".join(
        '<tr' + (' class="best"' if best is not None and row[0] == best else '') + '>'
        + "".join(f"<td>{value}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap" role="region" aria-label="Research evidence table" tabindex="0">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def update_home():
    path = SITE / "index.html"
    text = path.read_text()
    text = re.sub(
        r'(?:<section class="section"><div class="callout warn"><strong>Timing correction / methodology correction\.</strong>.*?</div></section>)+',
        '',
        text,
        flags=re.S,
    )
    text = text.replace('<strong>1,268</strong><small>Four time horizons</small>', '<strong>1,686</strong><small>Four time horizons</small>')
    text = text.replace("0.8024", "0.838182")
    text = text.replace("62.85", "59.49")
    text = text.replace("0.8157 → 0.838182", "0.857756 → 0.838182")
    text = text.replace("404 of 471", "575 of 683")
    text = text.replace("85.8%", "84.2%")
    text = text.replace(
        "Frozen development cross-validation · Full observer information · No untouched final test or live inference claim.",
        "Timing-corrected grouped OOF evaluation · Full observer information · Synthetic serving validated; real-CS2 capture deferred.",
    )
    marker = '<section class="section"><div class="section-head"><h2>What the evidence taught us</h2>'
    correction = (
        '<section class="section"><div class="callout warn"><strong>Timing correction / methodology correction.</strong> '
        'Awpy supplied a default 128 ticks/s, while the raw demo clock measured 64 ticks/s. The earlier nominal '
        '10 / 20 / 30 / 40-second evidence therefore represented roughly 20 / 40 / 60 / 80 seconds. The current '
        'site uses the rebuilt 1,686-observation dataset and preserves the old 1,268-observation result only as '
        '<a href="pages/evidence.html#historical">historical superseded evidence</a>.</div></section>'
    )
    text = text.replace(marker, correction + marker)
    path.write_text(text)


def update_evidence():
    metrics = table(
        ["Model", "Inputs", "Log loss ↓", "Brier ↓", "Accuracy ↑", "Macro F1 ↑"],
        [
            ["Logistic A3", "26", "0.887596", "0.529669", "0.575326", "0.537563"],
            ["XGB-A3", "26", "0.857756", "0.516930", "0.584816", "0.556304"],
            ["XGB-A4", "29", "0.873736", "0.524943", "0.584223", "0.552078"],
            ["XGB-A5", "37", "0.838182", "0.503190", "0.594899", "0.566090"],
        ],
        "XGB-A5",
    )
    errors = table(
        ["True → predicted", "A_PLANT", "B_PLANT", "NO_PLANT"],
        [["A_PLANT", "305", "45", "225"], ["B_PLANT", "63", "135", "113"], ["NO_PLANT", "181", "56", "563"]],
    )
    robustness = table(
        ["Evidence", "Corrected result"],
        [
            ["Match wins · Log loss", "14 / 20"],
            ["Match wins · Brier", "16 / 20"],
            ["Log-loss bootstrap 95% CI", "[−0.1073, −0.0191]"],
            ["Brier bootstrap 95% CI", "[−0.0422, −0.0134]"],
            ["Top-label ECE", "0.0509"],
        ],
    )
    historical = table(
        ["Evidence version", "Observations", "Effective horizons", "XGB-A5 LL", "Status"],
        [
            ["Historical timing v1", "1,268", "≈ 20 / 40 / 60 / 80s", "0.8024", "Superseded"],
            ["Corrected V0 freeze", "1,686", "10 / 20 / 30 / 40s", "0.838182", "Current"],
        ],
        "Corrected V0 freeze",
    )
    body = f'''
<header class="hero"><div class="kicker">Corrected grouped-OOF scientific record</div><h1>Results &amp; evidence</h1><p class="lead">The official V0 record after repairing timing semantics, rebuilding the dataset, rerunning evaluation, and restoring runtime parity.</p><div class="badges"><span class="badge ok">1,686 corrected observations</span><span class="badge ok">20 frozen match groups</span><span class="badge">Real capture deferred</span></div></header>
<section class="section"><div class="callout warn"><strong>Methodology correction.</strong> The earlier run used Awpy's 128 ticks/s default against demos whose raw clock measured 64 ticks/s. Its nominal horizons were approximately twice as late as stated. The corrected record below answers the intended 10 / 20 / 30 / 40-second question.</div></section>
<section class="section"><div class="section-head"><h2>Selected corrected baseline</h2><p>Same frozen match folds; corrected observations and timing semantics.</p></div>{metrics}</section>
<section class="section"><div class="section-head"><h2>Error composition</h2><p>The main classification boundary remains plant feasibility.</p></div>{errors}<div class="metric-strip"><div><strong>683</strong><span>Total errors</span></div><div><strong>575</strong><span>Plant ↔ No Plant</span></div><div><strong>84.2%</strong><span>Share crossing feasibility</span></div><div><strong>108</strong><span>Direct A ↔ B</span></div></div></section>
<section class="section"><div class="section-head"><h2>Strongest scientific result</h2><p>The feature families separate into two tactical subproblems.</p></div><div class="grid"><article class="card half"><div class="stat-label">Plant vs No Plant</div><h3>Combat dominates</h3><p>Permuting combat increases binary log loss by 0.1906, versus 0.0298 for offensive geometry.</p></article><article class="card half"><div class="stat-label">A vs B · given a plant</div><h3>Offensive geometry dominates</h3><p>Permuting offensive geometry increases conditional site log loss by 0.4593; combat is approximately neutral at −0.0056.</p></article></div></section>
<section class="section"><div class="section-head"><h2>Calibration and match robustness</h2><p>Probability behavior and consistency across held-out matches.</p></div>{robustness}<p>The bootstrap resamples paired match deltas. Negative intervals favor XGB-A5 over Logistic A3, but the 20-match development population still limits generalization.</p></section>
<section class="section" id="historical"><div class="section-head"><h2>Historical evidence retained</h2><p>Preserved for auditability; excluded from current claims.</p></div>{historical}<div class="callout">The lower historical loss does not describe a better corrected model. It describes later effective horizons and therefore a different, easier forecasting question.</div></section>
<section class="section"><div class="section-head"><h2>Downloadable record</h2><p>Corrected and historical tables remain separate.</p></div><div class="grid"><article class="card half"><h3>Corrected V0 artifacts</h3><p><a href="../assets/data/corrected/v0_tc_model_summary.csv">Model summary ↓</a><br><a href="../assets/data/corrected/v0_tc_xgb_a5_confusion_matrix.csv">Confusion matrix ↓</a><br><a href="../assets/data/corrected/v0_tc_match_uncertainty.csv">Match uncertainty ↓</a></p></article><article class="card half"><h3>Historical superseded artifacts</h3><p><a href="../assets/data/historical/v0_xgboost_summary.csv">Old model summary ↓</a><br><a href="../assets/data/historical/v0_xgb_a5_confusion_matrix.csv">Old confusion matrix ↓</a><br><a href="../assets/data/manifest.json">Artifact manifest ↓</a></p></article></div></section>'''
    replace_content(SITE / "pages/evidence.html", body)


def update_models():
    path = SITE / "pages/models.html"
    old = path.read_text()
    graph = re.search(r'<section class="section" id="experiment-structure">.*?</section>', old, re.S)
    graph_html = graph.group(0) if graph else ""
    comparison = table(
        ["Stage", "Information", "Inputs", "Corrected log loss ↓", "Decision"],
        [
            ["A0", "Class prior", "0", "1.036019", "Baseline"],
            ["A1", "+ offensive geometry", "13", "0.947481", "Keep"],
            ["A2", "+ motion", "17", "0.942335", "Keep"],
            ["A3", "+ combat", "26", "0.887596", "Keep"],
            ["A4", "+ economy", "29", "0.895020", "Exclude"],
            ["A5", "+ defense from A3", "37", "0.896782", "Linear test"],
            ["XGB-A5", "A5 with nonlinear interactions", "37", "0.838182", "Selected"],
        ],
        "XGB-A5",
    )
    body = f'''
<header class="hero"><div class="kicker">Corrected model selection</div><h1>Models</h1><p class="lead">The same feature ladder, reevaluated on true 10 / 20 / 30 / 40-second observations.</p></header>
<div class="research-board"><div class="board-story"><div class="kicker">Selected corrected V0 baseline</div><h2>XGB-A5</h2><p>Horizon + offensive geometry + motion + combat + defense. A nonlinear tabular forecast over full observer state.</p><div class="outcomes"><span class="outcome">37 inputs</span><span class="outcome">3 probabilities</span><span class="outcome">No economy</span></div></div><div class="board-metrics"><div class="metric"><span>Log loss ↓</span><strong>0.838182</strong></div><div class="metric"><span>Brier ↓</span><strong>0.503190</strong></div><div class="metric"><span>Accuracy ↑</span><strong>59.49%</strong></div><div class="metric"><span>Macro F1 ↑</span><strong>0.566090</strong></div></div></div>
{graph_html}
<section class="section"><div class="section-head"><h2>Corrected feature ladder</h2><p>A4 and A5 branch independently from A3.</p></div>{comparison}</section>
<section class="section"><div class="section-head"><h2>What defense taught us</h2><p>Model family changes whether the extra information can be used.</p></div><div class="callout"><strong>Defense adds little linear value but meaningful nonlinear value in XGBoost.</strong> Logistic A5 has log loss 0.896782, slightly worse than Logistic A3 at 0.887596. XGB-A5 improves over XGB-A3 from 0.857756 to 0.838182.</div></section>
<section class="section"><div class="section-head"><h2>Evidence-derived V1 hypothesis</h2><p>A focused experiment, not a casual future idea.</p></div><div class="grid"><article class="card half"><div class="stat-label">Stage 1</div><h3>Plant feasibility</h3><p>Estimate P(plant). Combat is the dominant feature family in this diagnostic task.</p></article><article class="card half"><div class="stat-label">Stage 2</div><h3>Conditional site choice</h3><p>Estimate P(A | plant). Offensive geometry is the dominant feature family.</p></article></div><p>Compose the three-class output as P(A)=q·r, P(B)=q·(1−r), and P(NO)=1−q. V0 evidence motivates this structure; only a matched V1 evaluation can establish whether it improves forecasting.</p></section>'''
    replace_content(path, body)


def update_data():
    horizons = table(
        ["Horizon", "Observations", "Resolver contract"],
        [["10s", "442", "first snapshot at/after target"], ["20s", "442", "first snapshot at/after target"], ["30s", "429", "first snapshot at/after target"], ["40s", "373", "first snapshot at/after target"]],
    )
    sources = table(
        ["Path", "V0 freeze evidence", "Status"],
        [
            ["Historical demo → offline builder", "1,686 corrected rows", "✅"],
            ["Historical demo → replay runtime", "1,686 / 1,686 exact parity", "✅"],
            ["Synthetic state → LiveEngine", "four horizons; true 1.000s motion", "✅"],
            ["Synthetic GSI → HTTP → predictor", "end-to-end inference + artifact identity", "✅"],
            ["Real CS2 machine → GSI capture", "source contract documented", "Deferred"],
        ],
    )
    body = f'''
<header class="hero"><div class="kicker">Timing-corrected source contract</div><h1>Data &amp; pipeline</h1><p class="lead">What the corrected dataset contains, how snapshots resolve, and exactly which runtime paths have been validated.</p></header>
<div class="metric-strip"><div><strong>20</strong><span>Mirage demos</span></div><div><strong>1,686</strong><span>Corrected observations</span></div><div><strong>64</strong><span>Measured ticks / second</span></div><div><strong>1.000s</strong><span>Motion interval</span></div></div>
<section class="section"><div class="section-head"><h2>Corrected observation distribution</h2><p>A575 · B311 · NO800.</p></div>{horizons}</section>
<section class="section"><div class="section-head"><h2>The timing repair</h2><p>Semantic correctness beyond parity.</p></div><div class="flow"><div class="node"><b>Raw clock</b><br>measure 64 tick/s</div><span class="arrow">→</span><div class="node"><b>Request</b><br>freeze_end + h×64</div><span class="arrow">→</span><div class="node"><b>Resolve</b><br>first at/after</div><span class="arrow">→</span><div class="node"><b>Guard</b><br>≤ 1 tick late</div><span class="arrow">→</span><div class="node"><b>Feature</b><br>true 1.000s motion</div></div><p>The old implementation used Awpy's 128-tick default. Because both offline and replay paths shared it, parity passed while the nominal horizons were still wrong. The corrected resolver accepts the first available snapshot at or after the target; observed lateness is at most one tick and the snapshot must remain before plant or round end.</p></section>
<section class="section"><div class="section-head"><h2>Runtime validation boundary</h2><p>Completed synthetic and replay evidence; real source remains explicit.</p></div>{sources}<div class="callout warn">“Synthetic GSI HTTP E2E passed” means the adapter, state engine, feature builder, model identity, and response path work together under controlled payloads. It does not mean a real CS2 machine feed has been captured and validated.</div></section>
<section class="section"><div class="section-head"><h2>Label and feature contracts</h2><p>Rules that prevent convenient fields from becoming hidden assumptions.</p></div><div class="grid"><article class="card half"><h3>Event-derived outcome</h3><p>BombsiteA → A_PLANT, BombsiteB → B_PLANT, and no recorded plant → NO_PLANT. Plant events remain the source of truth.</p></article><article class="card half"><h3>Current bomb state</h3><p>Current C4 inventory takes precedence over stale drop history. The model uses only state available at the resolved observation.</p></article></div></section>'''
    replace_content(SITE / "pages/data.html", body)


def update_roadmap():
    status = table(
        ["Checkpoint", "Evidence", "Status"],
        [
            ["Replay parity", "20 matches · 1,686 / 1,686 rows", "✅"],
            ["Synthetic LiveEngine", "10 / 20 / 30 / 40s + true 1.000s motion", "✅"],
            ["Synthetic GSI HTTP E2E", "request → state → features → predictor", "✅"],
            ["Runtime artifact identity", "corrected model and dataset metadata bound", "✅"],
            ["GSI source contract", "fields, cadence, and audit boundary documented", "✅"],
            ["Real CS2 machine capture", "real-source payload validation", "Deferred"],
        ],
    )
    body = f'''
<header class="hero"><div class="kicker">V0 freeze checkpoint</div><h1>Roadmap</h1><p class="lead">The timing-corrected research record and controlled runtime path are frozen. V1 begins from this evidence boundary.</p></header>
<section class="section"><div class="section-head"><h2>The repair became part of the research result</h2><p>Passing parity was necessary, but an independent audit found a shared semantic error.</p></div><div class="flow"><div class="node"><b>Parity passed</b></div><span class="arrow">↓</span><div class="node"><b>Timing audit</b><br>shared semantic bug</div><span class="arrow">↓</span><div class="node"><b>Old evidence</b><br>invalidated</div><span class="arrow">↓</span><div class="node"><b>Dataset</b><br>rebuilt</div></div><div class="flow"><div class="node"><b>Old folds</b><br>frozen</div><span class="arrow">↓</span><div class="node"><b>Evaluation</b><br>rerun</div><span class="arrow">↓</span><div class="node"><b>Runtime</b><br>rebuilt</div><span class="arrow">↓</span><div class="node"><b>Parity + correctness</b><br>restored</div></div></section>
<section class="section"><div class="section-head"><h2>Live engineering status</h2><p>Claims are limited to the path that was actually tested.</p></div>{status}</section>
<section class="section"><div class="section-head"><h2>V1 starts with one evidence-derived hypothesis</h2><p>Separate plant feasibility from conditional site choice.</p></div><div class="grid"><article class="card half"><h3>Why test it</h3><p>84.2% of corrected classification errors cross the Plant / No Plant boundary. Combat dominates this subproblem, while geometry dominates site choice given a plant.</p></article><article class="card half"><h3>What would count</h3><p>Freeze the V1 plan, use untouched matches or a suitable nested design, compose full three-class probabilities, and compare log loss, Brier, calibration, class metrics, and match robustness against corrected XGB-A5.</p></article></div></section>
<div class="callout">Corrected V0 is now the formal research record. Historical timing-v1 artifacts remain available only for auditability.</div>'''
    replace_content(SITE / "pages/roadmap.html", body)


def update_build_log():
    rows = table(
        ["Step", "What changed", "Evidence"],
        [
            ["1 · Parity passed", "Offline and replay shared one timing contract", "Agreement established"],
            ["2 · Independent timing audit", "Raw tick clock measured 64, not Awpy default 128", "Shared semantic bug found"],
            ["3 · Historical evidence invalidated", "Nominal horizons were actually about twice as late", "Old record marked superseded"],
            ["4 · Dataset rebuilt", "True 10 / 20 / 30 / 40s cutoffs; true 1.000s motion", "1,686 corrected rows"],
            ["5 · Match folds preserved", "Old grouped fold assignment reused", "Comparison discipline retained"],
            ["6 · Evaluation rerun", "Models, calibration, errors, importance, robustness", "Corrected OOF artifacts"],
            ["7 · Runtime rebuilt", "Model package bound to corrected dataset identity", "Identity test passed"],
            ["8 · Correctness restored", "All-match replay and synthetic serving tests", "1,686 / 1,686 parity"],
        ],
    )
    body = f'''
<header class="hero"><div class="kicker">A research correction, end to end</div><h1>Build log</h1><p class="lead">The most valuable V0 engineering story is how a passing system test was challenged, corrected, and revalidated.</p></header>
<section class="section"><div class="section-head"><h2>Timing repair sequence</h2><p>The exact chain from false confidence to a defensible freeze.</p></div>{rows}</section>
<section class="section"><div class="section-head"><h2>Why this matters</h2><p>Parity and correctness answer different questions.</p></div><div class="grid"><article class="card half"><h3>Parity asks whether paths agree</h3><p>The original offline and replay implementations agreed because they inherited the same 128-tick assumption.</p></article><article class="card half"><h3>Semantic audit asks whether the contract is true</h3><p>Measuring the raw demo clock showed the mismatch. Rebuilding every dependent artifact repaired the claim, not just the code.</p></article></div></section>
<section class="section"><div class="section-head"><h2>Freeze output</h2><p>What can be carried safely into V1.</p></div><div class="metric-strip"><div><strong>20</strong><span>Frozen match groups</span></div><div><strong>1,686</strong><span>Corrected rows</span></div><div><strong>37</strong><span>Selected inputs</span></div><div><strong>0.838182</strong><span>XGB-A5 log loss</span></div></div></section>'''
    replace_content(SITE / "pages/build-log.html", body)


def copy_artifacts():
    corrected = SITE / "assets/data/corrected"
    historical = SITE / "assets/data/historical"
    corrected.mkdir(parents=True, exist_ok=True)
    historical.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "artifacts").glob("v0_tc_*.csv")):
        shutil.copyfile(source, corrected / source.name)
    for source in sorted((ROOT / "artifacts").glob("v0_*.csv")):
        if not source.name.startswith("v0_tc_"):
            shutil.copyfile(source, historical / source.name)
    manifest = {
        "current_record": "V0-TIMING-CORRECTED",
        "current_observations": 1686,
        "historical_record": "V0 timing v1 (superseded)",
        "historical_observations": 1268,
        "correction": "Awpy default 128 ticks/s replaced by measured 64 ticks/s; first snapshot at/after target, at most one tick late.",
        "corrected_directory": "corrected/",
        "historical_directory": "historical/",
    }
    (SITE / "assets/data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def update_global_status():
    for path in [SITE / "index.html", *sorted((SITE / "pages").glob("*.html"))]:
        text = path.read_text()
        text = text.replace('assets/style.css"', 'assets/style.css?v=v0-tc-freeze"')
        text = text.replace('assets/style.css?v=v0-tc-freeze?v=v0-tc-freeze"', 'assets/style.css?v=v0-tc-freeze"')
        text = text.replace("V0 · Evaluation frozen", "V0 · Corrected freeze")
        text = text.replace("V0 research record · September 2026", "Corrected V0 freeze · September 2026")
        text = text.replace("85.8% of V0 classification errors", "84.2% of corrected V0 classification errors")
        text = text.replace(
            "V0 offline evaluation is complete. The next model hypothesis and the path to live inference remain separate pieces of work.",
            "The timing-corrected V0 evidence and controlled runtime path are frozen; real-CS2 machine capture remains deferred.",
        )
        path.write_text(text)

    manifest_path = SITE / "site-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        stage="Corrected V0 freeze checkpoint",
        v0_status="TIMING CORRECTED · EVIDENCE + RUNTIME FROZEN · REAL CAPTURE DEFERRED",
        selected_model="XGB-A5 · corrected timing",
        observation_scope="Mirage, full observer, true 10/20/30/40 seconds after freeze_end",
        observations=1686,
        measured_tick_rate=64,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    update_home()
    update_evidence()
    update_models()
    update_data()
    update_roadmap()
    update_build_log()
    copy_artifacts()
    update_global_status()
    print("Updated the static site to the timing-corrected V0 freeze checkpoint.")
