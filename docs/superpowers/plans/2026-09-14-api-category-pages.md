# Nested Reference Sidebar: One Page per Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The left sidebar under SuperGLM and SuperLSS opens to the task groups (Build, Fit, Inference, Predict, …), and each group opens to its members, instead of a flat list of every generated page or of nothing.

**Architecture:** Each group on `api/model.md` and `api/distributional.md` becomes its own page under `api/model/` and `api/distributional/`, carrying the group's paragraph and its autosummary table; the generated member pages stay and are children of the group page. The two class pages become overviews: intro, the lifecycle cards linking to the group pages, and a visible toctree of the groups. Sidebar depth rises to three so the group and member levels render as collapsible entries; generated pages take short titles so the member level reads as `fit_reml`, not `superglm.SuperGLM.fit_reml`.

**Tech Stack:** MyST, autosummary (`:toctree: ../generated` from the subpages), sphinx-design cards with `:link-type: doc`, pydata-sphinx-theme `navigation_depth`.

**Spec:** `docs/superpowers/specs/2026-09-13-docs-rebuild-design.md` §8; Max, 2026-09-14: "instead of dumping all names on the side in that dropdown thing you have like the categories build fit predict inference", then "Model the dropdown, superglm fit dropdown, then another dropdown for the inference methods etc". Decision: group pages keep the per-member pages (option chosen over inlining members).

## Global Constraints

- Branch `docs/api-category-pages`, stacked on `docs/api-model-narrative` (#394). One simple shell command per Bash call.
- Group prose moves verbatim; no new behavioural claims. Relative links gain one `../` and autosummary toctrees become `../generated` so stubs keep their paths under `api/generated/`.
- Strict build `SUPERGLM_DOCS_EXECUTE=off sphinx-build -n -W --keep-going` at zero warnings; `pytest tests/docs -m "not docs"` and `tests/test_release_packaging.py` pass.
- Docnames `api/model` and `api/distributional` keep their paths (the redirects target them).

---

### Task 1: Split the two class pages

- [ ] Run `split_reference.py` (scratchpad): writes `api/model/{build,fit,inference,predict,plot,diagnose,constrain-shapes,screen-interactions,deploy,results-and-records}.md` and `api/distributional/{families,declarations,fit,inference,predict,check-the-fit,price,plot,certification,save-and-load,configuration}.md`, and rewrites the two overviews. The Build page holds the SuperGLM class entry plus the configuration group; the SuperLSS Fit page holds the class entry, the fit members and `FitDiagnosticReport` in one block. "Read the fit" is renamed Inference on both pages and on card 3.
- [ ] Group order follows the cards: SuperGLM = Build, Fit, Inference, Predict, Plot, Diagnose, Constrain shapes, Screen interactions, Deploy, Results and records; SuperLSS = Families, Declarations, Fit, Inference, Predict, Check the fit, Price and portfolio views, Plot, Smoothing certification and telemetry, Save and load, Configuration (Max, after the first build: families in their own dropdown).

### Task 2: Short titles on generated pages

- [ ] `_templates/autosummary/base.rst` (new), `class.rst` and `module.rst` (new, Sphinx's builtin with the title line changed) use `{{ name | escape | underline }}`. The two `class-no-members` templates keep `fullname`, so the class stub never shares a label with the overview entry one level above it; their seealso text names the SuperGLM / SuperLSS overview pages.

### Task 3: Sidebar depth and tests

- [ ] `docs/conf.py`: `navigation_depth: 3`, comment rewritten (depth 3 = section page, group, member; other reference pages show their classes as one collapsed level).
- [ ] `tests/docs/test_api_reference.py`: `documented_names()` walks `docs/api` recursively; `MEMBER_PAGES` maps each class to its group directory and `listed_members` reads every page in it.
- [ ] `tests/test_release_packaging.py`: the `design_summary` assertion reads `docs/api/model/inference.md`.

### Task 4: Verify

- [ ] Strict build zero warnings; both test files pass; in the built HTML, `api/model/fit.html`'s sidebar nests SuperGLM › Fit › `fit`, `fit_reml`, … and no other group is expanded; every card href resolves; screenshot of `api/model.html` and `api/model/fit.html` at 1440 light read correctly. Refresh the preview overlay.
