# SuperGLM documentation rebuild: design

Date: 2026-09-13. Status: approved in conversation, pending review of this
document. Branch: `worktree-docs-rebuild-sphinx`, based on `origin/master`
at `7c4e70ff` (0.33.0).

## 1. Decision record

The published site (52 pages, MkDocs + Material) is rebuilt on Sphinx with a
Diátaxis information architecture, executed tutorials, and a complete API
reference generated from docstrings. Delivery is hybrid: one structural pull
request switches the generator, installs the new architecture, generates the
full API reference, and moves every existing page into its new home with
mechanical syntax conversion only. Content is then rewritten one area per
pull request, each executed, tested and published on merge.

Why now: Material for MkDocs has a published end-of-life date of
2026-11-05 (critical fixes only until then), MkDocs core has had no release
since 2024-08-30 and no commit since 2025-10-20, and the successor Zensical
cannot render notebooks and gives no date for doing so. Sphinx with the
pydata theme, MyST and MyST-NB is the stack used by 14 of 17 peer libraries
probed on 2026-09-13, and the only candidate with no blocked feature.

Why hybrid: Diátaxis's own guidance is to avoid tearing a large doc set
down and starting again, and to publish each improvement as it lands
(NumPy did the same under NEP 44). A single 70,000-word rewrite would be
unreviewable and would diverge for weeks from a master that ships every few
days.

## 2. Goals and non-goals

Goals:

1. A new reader lands, installs, and has a fitted pricing model on public
   data inside one tutorial of at most seven steps.
2. Every public name in `superglm.__all__` and every public member of
   `SuperGLM` and `SuperLSS` has a reference page, generated, not hand
   written.
3. Every code example that shows output is executed at build time and by a
   test; a library change reds a test rather than serving a stale render.
4. Each page has one Diátaxis job, one reader, and a title that predicts its
   content.
5. The site is portable: plain Markdown prose, plain notebook sources, and
   one-line docstring directives, so a later move to another generator is a
   mechanical conversion.
6. Nothing external breaks: every URL live today redirects to its successor.

Non-goals for this rebuild:

- Versioned documentation (the pydata theme's switcher can be added later).
- Running code in the browser. superglm hard-depends on numba and tabmat,
  neither of which has a WebAssembly build, so JupyterLite is not possible.
  Binder is possible but too slow and flaky to put on a page.
- Prose linting (Vale, markdownlint). Considered, deferred; see §14.
- Changing the package API or fixing numerical claims beyond the staleness
  ledger in the audit. Docstring wording is in scope (§8).
- Re-measuring the benchmark studies that move out of the user docs.

## 3. Audience and content principles

Primary reader: a pricing actuary or analyst who knows GLMs from Emblem,
Radar or R, is new to this package, and may be new to Python GAMs.
Secondary readers, served by the same material plus their own section:
model-risk and validation reviewers; Python data scientists and mgcv users
evaluating the package.

Consequences:

- Tutorials teach the pricing job end to end on the public French MTPL2
  data: frequency, severity, pure premium, constraints, rating table.
- REML, EDF, penalty, basis, smoothing parameter and similar terms are
  defined on first use and linked to a glossary page. mgcv and Emblem
  equivalents are named where they exist.
- Pages are typed by the Diátaxis compass (action or cognition; acquisition
  or application) into tutorial, how-to, reference or explanation, plus the
  two entry types Cloudflare's style guide adds: an overview (the landing
  page) and a get-started path.
- How-to headings are bare-infinitive tasks ("Fit a monotone spline"), per
  the Google developer documentation style guide. Tutorials state "By the
  end of this tutorial you will be able to…", run 15 to 60 minutes, use at
  most seven primary steps, and end with "Main takeaways" and "Next steps"
  (Good Docs Project templates, MIT-0).
- A tutorial is not the place for explanation; it links out.
- One canonical location per code pattern. Today the discrete-REML fit
  appears on ten pages; after the rebuild it appears on one and the rest
  link to it.

## 4. Evidence base

Audit of the live site against master at 0.33.0 (full report in the
session scratchpad; the numbers that shaped the design):

- 128,298 words built; 43% (the roadmap and ten research reports) is not in
  the nav; the API reference is 1,110 words (0.9%).
- 42 of 131 public names are never mentioned on a user page; 69 of 131 have
  no reference entry; the SuperGLM page lists 15 of 47 members.
- Ten pages carry the same fit; four pages compete to be "start here".
- Guide pages are accretions of per-PR notes: `results.md` is 78% rating
  table export, `fitting.md` 44% a dated benchmark, `screening.md` 37%
  provenance. 194 docs commits with subjects such as "Say what a per-unit
  offset block is".
- Five notebooks, none executed in CI; two have no committed outputs; one
  untouched since 2026-03-28; the main walkthrough uses deprecated spellings
  and three private attributes.
- One genuine tutorial exists, `getting-started/distributional.md`, and it
  is the template.

Tooling facts verified directly on 2026-09-13 (sources in §16):

- Material for MkDocs issue #8523: "final maintenance period", "critical bug
  fixes and security updates through November 5, 2026", "new feature
  development is now focused on Zensical".
- Zensical issue #52 and backlog #9: notebook support "we just haven't
  considered it" (2026-08-23); no estimate given (2026-08-24); backlog item
  untouched since 2025-11-13.
- MkDocs 2.0 dev releases on PyPI carry no licence, no repository URL and no
  plugin system; they must never arrive through a lockfile refresh.
- mkdocs-jupyter, MyST-NB and Quarto all execute notebooks with caching;
  only MyST-NB does so on a stack with a maintained API-reference generator
  and no end-of-life date.

## 5. Toolchain

Generator: Sphinx 9.1 (BSD-2-Clause). Theme: pydata-sphinx-theme 0.21
(BSD-3-Clause) by default; shibuya (BSD-3-Clause) is built in the spike for
comparison and removed if not chosen. Markdown: MyST-Parser (MIT).
Notebooks: MyST-NB 1.4 (BSD-3-Clause) with jupyter-cache. API reference:
`sphinx.ext.autodoc` + `sphinx.ext.autosummary` + numpydoc 1.10 (BSD).
Also: sphinx-design (MIT) for the landing page grid and tab sets,
sphinx-copybutton (MIT), jupytext 1.19 (MIT) for notebook pairing,
nbstripout (MIT) to keep outputs out of git, and a redirect extension
(sphinxext-rediraffe or sphinx-reredirects; the licence is read from the
package before install, and rediraffe is preferred because it can check
that every renamed page has a redirect).

Removed from the `docs` dependency group: mkdocs-material, mkdocstrings,
mkdocs-jupyter, nbconvert. `mkdocs.yml` is deleted.

Rejected, with the deciding fact: Zensical (no notebook rendering, alpha);
Quarto (its Python API tool quartodoc has had no release in 15 months and no
commit in nine; revisit if it revives, it would also allow executed R and
mgcv comparisons); Jupyter Book 2 (no docstring API reference; open request
since 2024); MkDocs forks ProperDocs and MkDocs NG (fix the core, leave the
theme's end of life untouched); a custom generator (search, nav, theme,
cross-references, link checking, math and a docstring renderer are each a
project, and every one is solved in Sphinx).

Portability rules that keep the next move mechanical:

1. Prose is CommonMark plus MyST fenced directives; no theme-specific HTML
   outside `index.md`.
2. Notebooks are MyST Markdown notebooks, convertible with jupytext.
3. API pages contain only `{autosummary}` blocks and short orientation prose.
4. Math is dollar-sign LaTeX.
5. Cross-references use MyST roles; no hand-built relative HTML links.

## 6. Information architecture

Eight top-level entries, rendered as the theme's top navigation bar.
Filenames are kebab-case; every page is `.md`.

```
docs/
  index.md                          Home (landing)
  get-started/
    index.md
    installation.md                 keep
    first-pricing-model.md          T0, notebook: MTPL2 frequency, ~20 min
    coming-from-mgcv.md             concept map: s(), bs=, k, REML, gam.check
    coming-from-emblem.md           concept map: factors, bands, relativities,
                                    rating tables, what "penalised" buys
  tutorials/
    index.md                        gallery with one line and one figure each
    severity-and-pure-premium.md    T1: freMTPL2sev join, Gamma, Tweedie,
                                    product vs single Tweedie, holdout
    monotone-rating-curves.md       T2: constraints, QP vs SCOP, what is
                                    withheld after a repair
    interactions-and-credibility.md T3: screening, RandomEffect, FactorSmooth
    from-model-to-rating-table.md   T4: export_rating_tables, ppform, round
                                    trip check
    distributional-model.md         T5: existing page, converted to a
                                    notebook
    edit-a-model-in-the-browser.md  T6: superglm.editor.edit, screenshots
  how-to/
    index.md
    choose-a-fitting-path.md        from workflows + fitting (decision table)
    set-weights-and-offsets.md      from families + README
    specify-features.md             from features (splines, categorical,
                                    numeric, polynomial, piecewise)
    manage-categorical-levels.md    from features (level universe, collapse,
                                    specials, bind_levels)
    specify-interactions.md         from interactions (dispatch map,
                                    separated-cells warning kept verbatim)
    add-a-credibility-term.md       from credibility (usage half)
    constrain-a-smooth.md           from monotone (usage half)
    screen-interactions.md          from screening (guide half)
    compare-models-on-holdout.md    from validation
    read-a-summary-and-plot-effects.md  from results (inspection half) +
                                    plotting notebook
    run-adequacy-tests.md           new: dispersion_test, score_test_zi,
                                    zero_inflation_index, vuong_test
    profile-tweedie-p.md            from families + tweedie notebook
    fit-a-distributional-model.md   from models/distributional (usage half)
    check-a-distributional-fit.md   from models/distributional-inference
    deploy-a-fitted-model.md        from deployment (pickle, scoring)
    export-a-rating-table.md        from results + deployment (one copy)
    use-scikit-learn-pipelines.md   new: SuperGLMRegressor, SuperGLMClassifier
    handle-large-data.md            new: discrete REML, n_bins, threads
  explanation/
    index.md
    penalised-glms-and-reml.md      from optimization (ladder, selection)
    families-and-weights.md         from families (weight semantics essay)
    bases-and-penalties.md          from features (k, null spaces, m=, select)
    shape-constraints.md            from monotone (stricter than the
                                    literature; withheld inference)
    credibility-as-smoothing.md     from credibility (essay half)
    what-screening-does.md          from screening (limits) + evaluation
                                    summary, evidence linked to notes/
    distributional-regression.md    from models/distributional (EFS,
                                    derivative orders, limits)
    solvers-and-internals.md        from optimization (solver sections)
    glossary.md                     Sphinx glossary; {term} targets
  api/
    index.md                        orientation + grouped tables
    model.md  features.md  families-and-links.md  penalties.md
    inference.md  validation-and-diagnostics.md  plotting.md  export.md
    sklearn.md  stats.md  editor.md  warnings-and-exceptions.md
    generated/                      autosummary output, gitignored
  governance/
    index.md
    model-risk-pack.md              keep
    reproducibility.md              from cost-and-timing (policy half)
    python-support.md               keep
  development/
    index.md
    contributing.md                 new, short: environment, tests, docs
    releases.md                     policy; changelog = GitHub Releases
    migrations/                     keep, each stamped with its version
    internals/
      data-and-solver-boundaries.md
      distributional-family-development.md
      editor-frontend.md
      tabmat-integration-notes.md   with a "checked on" banner
      reading-order.md              from optimization §16
      benchmarking-policy.md        from cost-and-timing (machine half)
  _static/   logo-light.png logo-dark.png custom.css superglm.mplstyle
             hero.png (generated by T0, committed)
  _templates/autosummary/  class.rst module.rst
  conf.py
notes/                              not built, not ignored
  ROADMAP.md  research/  audit/     moved from docs/
```

`docs/superpowers/` stays where the planning tooling expects it (tracked and
ignored) and is excluded from the build. `notes/ROADMAP.md` needs a
`.gitignore` exception because `ROADMAP.md` is ignored globally. The 15 path
references to the moved directories (AGENTS.md, two test docstrings, one
source docstring, one source comment, five benchmark files) are updated in
the structural pull request.

Changelog: there is no `CHANGELOG.md`; the consolidated changelog is the
release commit and the GitHub release. The releases page links to GitHub
Releases rather than duplicating them.

## 7. Executable documentation

Format. Tutorials, and any how-to that shows output, are MyST Markdown
notebooks: YAML front matter with a `jupytext` `md:myst` text
representation and a `python3` kernelspec, code in `{code-cell} ipython3`
fences, prose as ordinary Markdown. Jupyter opens them directly with the
jupytext extension. Percent-format Python was considered and rejected
because prose-heavy documents read badly as commented code.

Pairing. `jupytext.toml` at the repo root pairs `docs/tutorials/*.md` with
`.ipynb`. A pre-commit hook runs `jupytext --sync` on that directory and
nbstripout keeps outputs out of the `.ipynb`. The build excludes `*.ipynb`
so nothing renders twice. Each tutorial opens with an "Open in Colab"
badge pointing at the `.ipynb` on GitHub and a download link; its first
cell is `%pip install -q superglm` tagged `skip-execution` so the build and
the tests never install over the checked-out source.

Execution. `nb_execution_mode` is read from the environment variable
`SUPERGLM_DOCS_EXECUTE` (default `cache`; the pull-request build sets
`off`). `nb_execution_raise_on_error = True`, `nb_execution_timeout = 900`,
cache at `docs/_build/.jupyter_cache`, restored in CI with `actions/cache`
keyed on the hash of `docs/**/*.md` and `uv.lock`.

Data. MTPL2 from OpenML through `sklearn.datasets.fetch_openml`
(`data_id` 41214 frequency, 41215 severity; licence CC0; cite Dutang and
Charpentier, CASdatasets). Each notebook has its own self-contained loader
cell so it runs alone and in Colab. CI caches `~/scikit_learn_data` keyed
`openml-mtpl2-v1`. Full data by default; if a notebook exceeds 120 seconds
in CI it subsamples in the text and says so. The scikit-learn MTPL2
examples run the full 678k rows in under 20 seconds, so this is a
safeguard, not an expectation. Private book data never appears in anything
committed.

Figures. Matplotlib only in executed pages. A hidden first cell (tag
`remove-cell`) sets rcParams to the site palette and sizes; Colab users get
matplotlib defaults, which is acceptable. `superglm.mplstyle` in `_static`
is the same settings for anyone reproducing figures elsewhere. Plotly pages
show static screenshots.

Tests, three tiers:

1. `tests/docs/test_notebooks.py` executes every MyST notebook under
   `docs/` with nbclient (`timeout` 900, `skip-execution` honoured),
   parametrised per file, marker `docs`. The four-way CI split gains
   `and not docs`; a new job runs `-m docs` on one cell.
2. `tests/docs/test_api_reference.py` asserts every name in
   `superglm.__all__` appears in an `{autosummary}` block under `docs/api`.
   A new export without a reference entry fails the suite.
3. `tests/docs/test_conventions.py` greps executed pages for private
   attribute access (`._`-prefixed names on package objects); the audit
   found four such uses in notebooks and none in prose.

Plain fenced code (not executed) is allowed only for fragments, such as a
constructor signature, and by convention stays under ten lines.

## 8. API reference

Autosummary with `:toctree: generated` and `:nosignatures:` generates one
page per listed name. A class template lists methods and attributes in
their own toctree, the pandas pattern, so the 47 public members of
`SuperGLM` and the members of `SuperLSS` each get a page.
`numpydoc_show_class_members = False` and
`numpydoc_class_members_toctree = False` avoid duplicate member tables.
`nitpicky = True` with `-W` rejects any unresolved cross-reference;
`nitpick_ignore_regex` covers third-party types absent from intersphinx.
Intersphinx maps Python, NumPy, SciPy, pandas, scikit-learn and matplotlib,
so return types link to their own docs.

Index pages: `api/index.md` orients the reader in one paragraph per
subsystem, in the style of the existing `api/distributional.md`, which was
the only reference page with useful prose. Per-group pages carry the
`{autosummary}` tables and nothing else that would rot.

Docstrings. 102 of 103 public names on the audited branch have docstrings;
`Constraint` has none; 24 members of `SuperGLM` have one-line docstrings.
numpydoc validation runs as a pre-commit hook on `src/superglm`, starting
with the missing-docstring check on public names and widening in the
remediation pull request, with `# numpydoc ignore=` for deliberate
exceptions.

## 9. Look and feel

Landing page: logo, one-paragraph pitch, the install line, a twelve-line
worked example, four route cards (get started, tutorials, how-to,
explanation) plus API and governance links, and one real fitted curve from
T0 on MTPL2 with its confidence band. Built with sphinx-design grids; the
only page allowed theme-specific markup.

Visual system: one matplotlib style for every figure on the site; light and
dark logo variants; a small `custom.css` for palette and spacing; the
theme's own dark mode.

Theme choice: the spike builds the same skeleton under pydata-sphinx-theme
and shibuya and captures Playwright screenshots of the landing page, a
tutorial page and an API page in light and dark. Max picks from the
pictures. If no preference, pydata stays.

## 10. Build, CI and deployment

Commands. `sphinx-build -b html -n -W --keep-going docs docs/_build/html`.
`docs/_build/` is gitignored.

Pull request (`dev-ci.yml`): the `docs` job installs the `docs` group with
the `plotting` extra and builds with `SUPERGLM_DOCS_EXECUTE=off`, so it is
a strict structural check in about the time of today's MkDocs build. The
`docs-notebooks` job runs `pytest tests/docs -m docs` with the OpenML
cache. Neither job executes inside Sphinx on a pull request.

Push to master (`docs.yml`): restore the jupyter cache and the OpenML
cache, build with execution, upload with `actions/upload-pages-artifact`,
deploy with `actions/deploy-pages`. This requires the repository's Pages
source to be switched once from "deploy from a branch" to "GitHub
Actions"; that is Max's click. The `gh-pages` branch is left in place until
the first successful Actions deploy, then deleted. Actions are pinned by
SHA as elsewhere in the repo.

Weekly (`docs-linkcheck.yml`): `sphinx-build -b linkcheck` on a schedule,
non-blocking, opens nothing, just reports.

Redirects. The 52 URLs live on the deployed site today (listed from the
`gh-pages` sitemap and committed as `tests/docs/fixtures/legacy_urls.txt`)
each get a redirect stub to their successor page. A test asserts every
legacy URL is in the redirect map and every target source exists.

Hazards to check in the structural pull request: how `.test_durations` and
the coverage threshold treat a new test directory (this has redded
unrelated pull requests before); whether `numpydoc` parses every existing
docstring without warnings under `-W`; MyST-NB's rendering of any HTML
outputs from pandas Styler objects.

## 11. Migration mapping

Action codes: keep (moved and converted, no rewrite), split (moved whole in
PR-1, carved in the named later PR), rewrite, retire (content absorbed
elsewhere, URL redirected), notes (moved to `notes/`, not built).

| Current page | Destination | Action | Carved in |
|---|---|---|---|
| README.md | README.md | rewrite to ≤400 words, absolute links | PR-1 |
| index.md | index.md | rewrite | PR-1 |
| getting-started/installation.md | get-started/installation.md | keep | — |
| getting-started/quickstart.md | get-started/first-pricing-model.md | retire; redirect | PR-2 |
| getting-started/distributional.md | tutorials/distributional-model.md | keep, converted to notebook | PR-1 |
| guide/workflows.md | how-to/choose-a-fitting-path.md | retire into | PR-9 |
| guide/fitting.md | how-to/choose-a-fitting-path.md | split; benchmark tables → notes/ | PR-9 |
| guide/features.md | how-to/specify-features.md | split → manage-categorical-levels, migrations (PR-8); bases-and-penalties (PR-11) | PR-8, PR-11 |
| guide/credibility.md | explanation/credibility-as-smoothing.md | split → how-to/add-a-credibility-term; tail sections → releases | PR-11 |
| guide/monotone.md | how-to/constrain-a-smooth.md | split → explanation/shape-constraints (PR-11) | PR-8, PR-11 |
| guide/validation.md | how-to/compare-models-on-holdout.md | keep, light edit | PR-9 |
| guide/results.md | how-to/read-a-summary-and-plot-effects.md | split → how-to/export-a-rating-table | PR-9 |
| guide/deployment.md | how-to/deploy-a-fitted-model.md | split; export rules merged into export how-to | PR-9 |
| guide/families.md | explanation/families-and-weights.md | split → set-weights-and-offsets (PR-9), profile-tweedie-p (PR-10); version notes → releases (PR-12) | PR-9, PR-10, PR-12 |
| guide/interactions.md | how-to/specify-interactions.md | keep, light edit | PR-8 |
| guide/screening.md | how-to/screen-interactions.md | split → explanation/what-screening-does (PR-11) | PR-10, PR-11 |
| guide/screening-evaluation.md | notes/research/ | notes; summarised in what-screening-does | PR-10 |
| guide/optimization.md | explanation/solvers-and-internals.md | split → penalised-glms-and-reml, internals/reading-order | PR-11 |
| guide/editor.md | tutorials/edit-a-model-in-the-browser.md | rewrite as tutorial | PR-7 |
| guide/scop-performance-prototype.md | notes/audit/ | notes; redirect to development/index | PR-1 |
| models/distributional.md | how-to/fit-a-distributional-model.md | split → explanation/distributional-regression (PR-11) | PR-10, PR-11 |
| models/distributional-inference.md | how-to/check-a-distributional-fit.md | keep, light edit | PR-10 |
| development/releases.md | development/releases.md | keep; agent runbook lines → `.codex/agents` reference | PR-12 |
| development/migrations/*.md (3) | development/migrations/*.md | keep; stamp versions | PR-12 |
| development/cost-and-timing.md | governance/reproducibility.md | split → internals/benchmarking-policy | PR-12 |
| development/python-support.md | governance/python-support.md | keep | — |
| development/data-and-solver-boundaries.md | development/internals/ | keep | — |
| distributional-family-development.md | development/internals/ | keep | — |
| editor_frontend.md | development/internals/editor-frontend.md | keep | — |
| tabmat-integration-notes.md | development/internals/ | keep, dated banner | — |
| governance/model_risk_pack.md | governance/model-risk-pack.md | keep | — |
| notebooks/editor_demo.ipynb | — | retire; superseded by T6 | PR-7 |
| notebooks/mtpl2_frequency_walkthrough.ipynb | — | retire; superseded by T0 | PR-2 |
| notebooks/plotting_diagnostics_demo.ipynb | — | retire; mined for read-a-summary | PR-9 |
| notebooks/ordered_categorical_smoothing.ipynb | how-to/specify-features.md | retire; section made executable | PR-8 |
| notebooks/tweedie_profile_estimation.ipynb | how-to/profile-tweedie-p.md | retire; made executable | PR-10 |
| api/*.md (10) | api/*.md (12) | rewrite as autosummary pages | PR-1 |
| ROADMAP.md, research/ (10), audit/ | notes/ | notes | PR-1 |

Retired pages keep their URL as a redirect to the page that absorbed them.
Until a split PR lands, the moved page is complete and reachable; nothing
is dropped in PR-1 except the branch-plan page, whose content is a design
record.

Mechanical conversion in PR-1, measured on master: 22 admonitions in four
files (`!!! note` → `{note}` directive), 69 mkdocstrings directives
(replaced by autosummary), 14 `attr_list` uses, 133 display-math blocks
(already dollar-delimited; unchanged), eight images, three footnotes, raw
HTML only on `index.md`. No tabs, snippets, mermaid or HTML tables.

## 12. Delivery sequence

Each pull request is reviewed by both review bots, with their summary
comments read as well as their threads. Max gives a go/no-go on the spike
preview, on PR-1, and on each tutorial.

| PR | Scope | Acceptance |
|---|---|---|
| spike | Skeleton under both themes; `api/model.md` autosummaried; T5 executed by MyST-NB; screenshots | Max picks a theme; numpydoc and MyST-NB behave on real pages; T5 execution time known |
| PR-1 structural | Toolchain, `conf.py`, architecture, landing page, complete API reference, all pages moved and converted, redirects, `notes/` move with reference updates, README, CI jobs, deploy workflow | `-n -W` green; every `__all__` name has a page; every legacy URL redirects; T5 executes; live site deployed by Actions |
| PR-2 | T0 first pricing model; coming-from-mgcv; coming-from-emblem; glossary; quickstart retired | T0 ≤7 steps, executes in CI, ends with takeaways; hero figure regenerated |
| PR-3 | T1 severity and pure premium | executes; compares product model with single Tweedie on holdout |
| PR-4 | T2 monotone rating curves | executes; shows what is withheld after a repair |
| PR-5 | T4 from model to rating table | executes; round-trip check against `predict` |
| PR-6 | T3 interactions and credibility | executes; separated-cells warning demonstrated |
| PR-7 | T6 editor tutorial; editor demo notebook retired | screenshots current; no private API |
| PR-8 | How-to split A: features, levels, interactions, constraints | old pages retired with redirects |
| PR-9 | How-to split B: fitting path, weights and offsets, large data, summary and plots, deployment, export | duplicate fit blocks reduced to one canonical location each |
| PR-10 | How-to split C: screening, adequacy tests, Tweedie p, distributional usage and checking | new adequacy-tests page covers all four tests |
| PR-11 | Explanation pages | each page passes the compass test; solver essay trimmed |
| PR-12 | Governance and development; migrations stamped | releases page links GitHub Releases |
| PR-13 | Docstring remediation; numpydoc validation widened | validation hook green on `src/superglm` |

Concurrent feature pull requests keep landing their doc notes in the moved
pages. PR-1 moves with `git mv` so rename detection keeps conflicts small;
the docs PR author resolves any that arise.

The first implementation plan covers the spike and PR-1 in full. Each
content pull request gets its own short plan when it starts, written
against the site as it then stands, since the pages it carves will have
absorbed feature-PR notes in the meantime.

## 13. Conventions for authors

- Page templates: tutorial (overview, "by the end", before you begin,
  steps, main takeaways, next steps); how-to (overview, before you begin,
  steps, see also); explanation (free form, ends with further reading).
- Every executed page is self-contained: its own imports, its own data
  loader, no shared helper module.
- No private API in any page. No dated, machine-specific benchmark in a
  user page; those live in `notes/` or `development/internals`.
- Version claims name a published tag, checked with `git tag`. "Removed in
  0.24.0" is not acceptable when no 0.24.0 was published.
- One canonical location per code pattern; other pages link with a MyST
  cross-reference, never a copy.
- New public exports come with a reference entry in the same pull request;
  the test in §7 enforces it.
- Adding a tutorial: copy the template, pair it with jupytext, add it to
  `tutorials/index.md`, run `pytest tests/docs -m docs -k <name>`.

## 14. Risks and open questions

- numpydoc may warn on existing docstrings under `-W`. The spike measures
  this; the fallback is a temporary `nitpick_ignore` list burned down in
  PR-13.
- MyST-NB execution time for a full MTPL2 REML fit is unmeasured in this
  stack. The spike times T5 and one 678k-row fit.
- OpenML outages would red the `docs-notebooks` job. Mitigation: the cache,
  and a retry in the loader cell. A committed CC0 subsample is the fallback
  if outages recur.
- Theme choice is deferred to the spike screenshots.
- Two tool behaviours are assumed and checked in the spike: nbclient skips
  cells tagged `skip-execution` by default, and MyST-NB honours the same
  tag at build time.
- Deferred tooling, each a one-line addition later: Vale with its AI-prose
  style, markdownlint, an `llms.txt` generator, the pydata version switcher.
- The `.test_durations` and coverage interplay with a new test directory is
  a known trap; checked in PR-1 before the first CI run.

## 15. Success criteria

1. `sphinx-build -n -W` green on every pull request; the live site is
   deployed by Actions from master.
2. Every name in `superglm.__all__` and every public member of `SuperGLM`
   and `SuperLSS` has a generated reference page.
3. Every tutorial and executable how-to runs in CI on every pull request.
4. All 52 legacy URLs redirect to a live page.
5. The discrete-REML fit block appears on exactly one page.
6. No user page exceeds 3,000 words except explanation essays.
7. A new reader reaches a fitted, validated model on public data within
   T0's seven steps.

## 16. Sources

- Diátaxis: https://diataxis.fr/ ; compass https://diataxis.fr/compass/ ;
  iterative migration https://diataxis.fr/how-to-use-diataxis/
- NumPy NEP 44: https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html
- Cloudflare content types:
  https://developers.cloudflare.com/style-guide/documentation-content-strategy/information-architecture/
- Google developer documentation style guide, headings and procedures:
  https://developers.google.com/style/headings ;
  https://developers.google.com/style/procedures
- Good Docs Project templates (MIT-0): https://www.thegooddocsproject.dev/template
- Material for MkDocs end of life: https://github.com/squidfunk/mkdocs-material/issues/8523
- Zensical notebook support: https://github.com/zensical/zensical/issues/52 ;
  https://github.com/zensical/backlog/issues/9
- Material on MkDocs 2.0:
  https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/
- MyST-NB execution: https://myst-nb.readthedocs.io/en/latest/computation/execute.html
- numpydoc validation: https://numpydoc.readthedocs.io/en/latest/validation.html
- jupytext: https://github.com/mwouts/jupytext
- pydata-sphinx-theme: https://pydata-sphinx-theme.readthedocs.io/
- Polars docs-as-tests pattern:
  https://raw.githubusercontent.com/pola-rs/polars/main/py-polars/tests/docs/test_user_guide.py
- scikit-learn MTPL2 examples:
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html ;
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html
- glum French motor tutorial:
  https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html
- OpenML freMTPL2freq / freMTPL2sev (CC0):
  https://www.openml.org/api/v1/json/data/41214 ;
  https://www.openml.org/api/v1/json/data/41215
