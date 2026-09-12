# CS2 Tactical Intelligence Lab — research website

A static HTML, CSS, and JavaScript research notebook. Open `index.html` directly or serve this directory locally. The existing GitHub Pages workflow publishes `site/` on a push to `main`.

The ten pages share navigation and responsive styling. `pages/v0.html` contains the full explanation, with in-page chapter links and a reading indicator. Corrected CSV snapshots live under `assets/data/corrected/`; the 128-tick historical record is preserved under `assets/data/historical/` and marked superseded. Explanations and original research documents are under `assets/sources/`.

## Editing the V0 explanation

Edit `docs/v0_explained.md` in the repository root, then run `python3 scripts/build_research_site.py` followed by `python3 scripts/update_v0_freeze_site.py`. The first command renders the full explanation and copies source evidence; the second updates the companion pages and separates corrected from historical artifacts. Review evidence changes against the corrected freeze before publishing.

The small renderer supports the Markdown syntax used in the guide. Named math fences are typeset as native MathML using `scripts/research_equations.py`, with no external font or script requests. It has no third-party dependencies. Other website pages remain directly editable HTML. The model code, raw demos, and model training do not run in the website.

## Publishing

The GitHub Pages workflow is preserved. A separate private Sites preview uses the project identity in `.openai/hosting.json` and a static copy of the website in `dist/`. That generated directory is ignored by Git. There are no app credentials in the site.

V0 is a corrected offline development baseline with replay and controlled synthetic serving validation. Publishing these pages does not deploy a live inference service or claim real-CS2 source validation. Preserve this freeze record when adding V1.
