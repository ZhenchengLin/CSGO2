"""Keep current site summaries aligned with the frozen V1 decision.

Run through update_v1_freeze_site.py after the V0 renderer. Historical research
chapters retain their original hypothesis, with an explicit current-status note.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "Best development result": "Selected V0 development baseline",
    "This motivates testing a hierarchical V1, whose benefit remains unproven.": "This motivated V1 hierarchy experiments. H2 slightly improved pooled loss, but weak match robustness did not support replacing V0.",
    "Separate completed offline research from future replay, live inference, and the next modeling hypothesis.": "Review completed replay and synthetic inference, the V1 freeze, and the deferred real-CS2 capture.",
    "V1 begins from this evidence boundary.": "V1 has now completed its hierarchy and temporal experiments; V0 remains selected.",
    "V1 starts with one evidence-derived hypothesis": "The hypothesis that led to V1",
    "<h3>What would count</h3>": "<h3>What was tested</h3>",
    "Freeze the V1 plan, use untouched matches or a suitable nested design, compose full three-class probabilities, and compare log loss, Brier, calibration, class metrics, and match robustness against corrected XGB-A5.": "H0, H1 and H2 composed full three-class forecasts on the same frozen development folds. H2 used nested calibration, but this was not an untouched final test. Its small pooled gain and weak match robustness did not justify promotion.",
    "The immediate V1 hypothesis": "The completed V1 investigation",
    "A testable direction suggested by V0 failures.": "A V0-derived hypothesis tested in the frozen V1 experiments.",
    "A separately trained hierarchy may help, but it has not yet beaten the V0 baseline.": "H2 slightly improved pooled probability scores, but did not robustly outperform V0 across matches. V0 remains selected; T0 did not justify sequence-model escalation.",
    'href="v0.html#v1">Read the proposed experiment': 'href="v1.html">Read the completed experiments',
    "A focused experiment, not a casual future idea.": "The original motivation, now tested in H0, H1 and H2.",
    "V0 evidence motivates this structure; only a matched V1 evaluation can establish whether it improves forecasting.": "V0 evidence motivated this structure. The completed V1 evaluation below found a tiny H2 pooled gain without sufficient match robustness for promotion.",
    "Live inference, richer semantics, and improved V1 models remain future claims to earn with new evidence.": "Replay and synthetic inference are validated; real-CS2 capture remains deferred. V1 subsequently tested hierarchy and temporal features without replacing V0.",
    "What can be carried safely into V1.": "The corrected baseline carried into the completed V1 experiments.",
    "Promising ideas are preserved without expanding the frozen V0 scope.": "Future ideas are preserved beyond the frozen V0 and V1 investigations.",
    "Visible, but outside frozen V0.": "Visible, but outside the completed V0 and V1 experiments.",
    "P(mean Δ &lt; 0)": "Bootstrap fraction with mean Δ &lt; 0",
    "a different, easier forecasting question.": "a different forecasting question with later horizons and a different eligible population.",
    "The official V0 record after repairing timing semantics, rebuilding the dataset, rerunning evaluation, and restoring runtime parity.": "The corrected V0 record and completed V1 comparisons: timing semantics, frozen evaluation, model selection and runtime boundaries.",
}


def reconcile():
    for path in [ROOT / "site/index.html", *sorted((ROOT / "site/pages").glob("*.html"))]:
        text = path.read_text()
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        if path.stem != "v0":
            text = text.replace('MIRAGE <span>·</span> VERSION 0', 'MIRAGE <span>·</span> V1 SCIENCE FREEZE')
        if path.stem == "v0" and 'id="v1-current-status"' not in text:
            text = text.replace('<section id="v1">', '<section id="v1"><div class="callout" id="v1-current-status"><strong>Historical V0 proposal — current status:</strong> This chapter preserves the hypothesis at the V0 freeze. V1 has since tested H0, H1, H2 and T0. H2 slightly improved pooled scores but was not promoted; V0 remains selected. <a href="v1.html">Read the completed V1 record ↗</a></div>')
        if path.stem in {"journey", "build-log"} and 'id="v1-completion"' not in text:
            text = text.replace('<footer class="footer">', '<section class="section" id="v1-completion"><h2>V1 completed: evidence before promotion</h2><p>H0 tested hierarchy; H1 specialized the heads; H2 calibrated site probabilities. H2 reached log loss 0.837397, but improved only 9 of 20 matches and its bootstrap interval crossed zero. T0 temporal features also failed their gate. V0 XGB-A5 remains selected; real-CS2 capture remains deferred.</p><a href="v1.html">Read the V1 experiments and retrospective ↗</a></section><footer class="footer">')
        path.write_text(text)


if __name__ == "__main__":
    reconcile()
