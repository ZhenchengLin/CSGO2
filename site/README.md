# CS2 Tactical Intelligence Lab — research website

A static HTML, CSS, and JavaScript research notebook. Open `index.html` directly or serve this directory locally. The existing GitHub Pages workflow publishes `site/` on a push to `main`.

The ten pages share navigation and responsive styling. `pages/v0.html` contains the full explanation, with in-page chapter links and a reading indicator. Research results are published as frozen CSV snapshots under `assets/data/`; the explanations and original research documents are under `assets/sources/`.

## Editing the V0 explanation

Edit `docs/v0_explained.md` in the repository root, then run `python3 scripts/build_research_site.py` from that root. This renders the V0 chapter into the current companion-page shell and copies source documents and the local frozen CSV artifacts into the website. Review any evidence changes against the frozen evaluation before publishing.

The small renderer supports the Markdown syntax used in the guide. It has no third-party dependencies. Other website pages remain directly editable HTML. The model code, raw demos, and model training do not run in the website.

## Publishing

The GitHub Pages workflow is preserved. A separate private Sites preview uses the project identity in `.openai/hosting.json` and a static copy of the website in `dist/`. That generated directory is ignored by Git. There are no app credentials in the site.

V0 is an offline development baseline. Publishing these pages does not deploy a live inference service. Preserve this version's research story when adding later versions.
