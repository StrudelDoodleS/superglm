# Docs Rebuild: Spike and Structural PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MkDocs + Material site with a Sphinx site that has the new eight-section architecture, a complete generated API reference, one executed tutorial, redirects for every live URL, and a CI pipeline that builds strictly, executes notebooks as tests, and deploys to GitHub Pages.

**Architecture:** Sphinx 9 reads MyST Markdown from `docs/`; MyST-NB executes notebook pages; autosummary generates one reference page per public name from numpy docstrings; the pydata theme carries the pulp-comic identity through `custom.css`. Existing pages are moved with `git mv` and converted mechanically, not rewritten. Three small pytest files under `tests/docs/` guard API coverage, redirects, notebook execution and conventions.

**Tech Stack:** Sphinx 9.1 (BSD-2), pydata-sphinx-theme 0.21 (BSD-3), MyST-Parser 5.1 (MIT), MyST-NB 1.4 (BSD-3), numpydoc 1.10 (BSD), sphinx-design 0.7 (MIT), sphinx-copybutton 0.5 (MIT), sphinxext-rediraffe 0.3 (MIT), jupytext 1.19 (MIT), nbstripout 0.9 (MIT), nbclient (BSD-3), uv, GitHub Pages actions, Playwright for spike screenshots.

**Spec:** `docs/superpowers/specs/2026-09-13-docs-rebuild-design.md`

## Global Constraints

- Work only in the worktree `/home/max/projects/superglm/.claude/worktrees/docs-rebuild-sphinx` on branch `worktree-docs-rebuild-sphinx`, based on `origin/master` at `7c4e70ff` (0.33.0). Never `cd` to the main checkout. Never use bare `git stash`.
- Every command runs through `uv run` in this worktree; its `.venv` imports `superglm` from this worktree's `src/`. Never touch the main checkout's `.venv`.
- Edit existing files surgically (Edit tool, `sed`, or short Python); whole-file writes are for new files only.
- Never run the full test suite. Run `tests/docs` and the named focused tests only; `uv run python run_test.py` is the smoke test.
- Existing pages are moved and syntax-converted in this PR, never rewritten. Only `index.md`, `README.md` and the six new section index pages are new prose.
- Strict build command, used everywhere: `uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`. Zero warnings is the pass condition.
- No private attribute access (`obj._name`) in any executed page.
- Version claims name a published tag. No dated, machine-specific benchmark is added to a user page.
- Every new dependency's licence is in the permissive bucket (verified from PyPI on 2026-09-13, listed in Tech Stack). Never add a dependency without reading its licence.
- Commit after every task. End every commit message with the two attribution trailer lines from the session's system reminder (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the `Claude-Session:` URL). `docs/superpowers/` is gitignored but tracked: plan and spec edits need `git add -f`.
- The identity (spec §9): ink `#15171C`, red `#D6402B`, yellow `#F4B942`, ground `#FFFFFF`, off-white `#F7F5F0`; Bangers for the hero headline, landing section titles and tutorial gallery headings only; Source Sans 3 for reading text; IBM Plex Mono for code; 2px ink borders with 4px hard offset shadows on cards, buttons and admonitions.

---

## File structure

Created:

- `docs/conf.py` — the whole Sphinx configuration; theme selectable by `SUPERGLM_DOCS_THEME`, execution mode by `SUPERGLM_DOCS_EXECUTE`.
- `docs/_static/custom.css` — the identity: fonts, palette, comic panels, hero, one entrance animation.
- `docs/_templates/autosummary/class.rst` — class template that gives every public method and attribute its own page.
- `docs/redirects.txt` — legacy URL → new page map read by rediraffe.
- `docs/index.md` (rewritten), `docs/get-started/index.md`, `docs/tutorials/index.md`, `docs/how-to/index.md`, `docs/explanation/index.md`, `docs/api/index.md`, `docs/governance/index.md`, `docs/development/index.md`, `docs/development/internals/index.md`, `docs/examples/index.md` — section pages carrying the toctrees.
- `docs/api/*.md` (twelve pages) — autosummary lists with one orientation paragraph each.
- `docs/tutorials/distributional-model.md` (+ paired `.ipynb`) — the first executed notebook, converted from the existing page.
- `tests/docs/__init__.py`, `tests/docs/test_api_reference.py`, `tests/docs/test_redirects.py`, `tests/docs/test_notebooks.py`, `tests/docs/test_conventions.py`, `tests/docs/fixtures/legacy_urls.txt`.
- `jupytext.toml` — pairing config for `docs/tutorials/`.
- `notes/` — receives `docs/research/`, `docs/audit/`, `docs/ROADMAP.md`.

Modified: `pyproject.toml` (docs group, pytest marker, numpydoc validation), `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/dev-ci.yml`, `.github/workflows/ci.yml`, `.github/workflows/docs.yml`, `README.md`, `AGENTS.md`, four source and test files whose docstrings cite `docs/audit` or `docs/research`, six benchmark files with the same citations.

Deleted: `mkdocs.yml`, `docs/javascripts/mathjax.js`, `docs/stylesheets/extra.css`, `docs/guide/scop-performance-prototype.md` (moved to `notes/audit/`).

Moved: every page in the spec §11 table, exact commands in Task 10.

---

## Phase A: the spike

The spike's files are the seed of the structural PR, not throwaway. Only the Shibuya build is throwaway.

### Task 1: Docs dependency group

**Files:**
- Modify: `pyproject.toml` (the `[dependency-groups] docs` list, currently lines 153-160)

**Interfaces:**
- Produces: a synced environment where `import sphinx, myst_nb, numpydoc, sphinx_design, sphinx_copybutton, sphinxext.rediraffe, jupytext, nbclient` all succeed.

- [ ] **Step 1: Replace the docs group**

Replace the six-line list under `docs = [` with:

```toml
docs = [
    "sphinx>=9.1,<10",
    "pydata-sphinx-theme>=0.21",
    "myst-nb>=1.4",
    "numpydoc>=1.10",
    "sphinx-design>=0.7",
    "sphinx-copybutton>=0.5",
    "sphinxext-rediraffe>=0.3",
    "jupytext>=1.19",
    "nbstripout>=0.9",
    "ipykernel>=6.0",
]
```

- [ ] **Step 2: Lock and sync**

Run: `uv lock && uv sync --extra dev --extra plotting --group docs`
Expected: lock succeeds; `mkdocs`, `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter` are removed from the environment.

- [ ] **Step 3: Verify imports**

Run: `uv run python -c 'import sphinx, myst_nb, numpydoc, sphinx_design, sphinx_copybutton, sphinxext.rediraffe, jupytext, nbclient, pydata_sphinx_theme; print(sphinx.__version__, myst_nb.__version__)'`
Expected: prints `9.1.x 1.4.x` with no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Docs: replace the MkDocs dependency group with the Sphinx stack"
```

### Task 2: Sphinx skeleton that builds strictly

**Files:**
- Create: `docs/conf.py`
- Create: `docs/_static/custom.css`
- Create: `docs/_templates/autosummary/class.rst`
- Modify: `.gitignore` (append build directories)
- Temporarily modify: `docs/index.md` (replaced fully in Task 12)

**Interfaces:**
- Produces: `sphinx-build` runs to completion with `-n -W` on a one-page site; the CSS classes `sg-hero`, `sg-hero__logo`, `sg-hero__copy`, `sg-hero__title`, `sg-hero__meta`, `sg-btn`, `sg-btn--ghost`, `sg-display` used by Task 12.

- [ ] **Step 1: Write `docs/conf.py`**

```python
"""Sphinx configuration for the superglm documentation.

Environment switches used by CI and the spike:

- ``SUPERGLM_DOCS_EXECUTE``: MyST-NB execution mode, default ``cache``.
  Pull-request builds set ``off``.
- ``SUPERGLM_DOCS_THEME``: theme name, default ``pydata_sphinx_theme``.
"""

from __future__ import annotations

import os
from importlib.metadata import version as _dist_version
from pathlib import Path

HERE = Path(__file__).resolve().parent

project = "superglm"
author = "Max Hicks"
copyright = "2026, Max Hicks"  # noqa: A001
release = _dist_version("superglm")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.rediraffe",
]

exclude_patterns = [
    "_build",
    "superpowers/**",
    "tutorials/*.ipynb",
    "**/.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

# MyST Markdown
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

# MyST-NB execution
nb_execution_mode = os.environ.get("SUPERGLM_DOCS_EXECUTE", "cache")
nb_execution_cache_path = str(HERE / "_build" / ".jupyter_cache")
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_excludepatterns = ["examples/*.ipynb"]
nb_merge_streams = True

# API reference
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {"show-inheritance": True}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_timeout = 30

nitpicky = True
# Docstring type shorthands that are prose, not importable objects. Extend
# only with entries the build reports; never with a genuine dotted path.
nitpick_ignore_regex = [
    ("py:class", r"array[-_]like"),
    ("py:class", r"ArrayLike"),
    ("py:class", r"callable"),
    ("py:class", r"optional"),
    ("py:class", r"default .*"),
    ("py:class", r"(list|dict|tuple|sequence|iterable|mapping) of .*"),
    ("py:class", r"DataFrame|Series|ndarray"),
]

# Redirects from the MkDocs site's URLs
rediraffe_redirects = "redirects.txt"

# HTML
html_theme = os.environ.get("SUPERGLM_DOCS_THEME", "pydata_sphinx_theme")
html_title = "superglm"
html_static_path = ["_static", "images"]
html_favicon = "images/logo.png"
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Bangers&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_sidebars = {"index": []}
html_context = {
    "github_user": "StrudelDoodleS",
    "github_repo": "superglm",
    "github_version": "master",
    "doc_path": "docs",
}

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "logo": {
            "image_light": "_static/logo.png",
            "image_dark": "_static/logo.png",
            "text": "superglm",
        },
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/StrudelDoodleS/superglm",
                "icon": "fa-brands fa-github",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/superglm/",
                "icon": "fa-brands fa-python",
            },
        ],
        "navbar_align": "left",
        "header_links_before_dropdown": 8,
        "show_toc_level": 2,
        "use_edit_page_button": False,
        "footer_start": ["copyright"],
        "footer_end": [],
    }
elif html_theme == "shibuya":
    html_theme_options = {
        "github_url": "https://github.com/StrudelDoodleS/superglm",
        "accent_color": "red",
        "nav_links": [
            {"title": "Get started", "url": "get-started/index"},
            {"title": "Tutorials", "url": "tutorials/index"},
            {"title": "How-to", "url": "how-to/index"},
            {"title": "Explanation", "url": "explanation/index"},
            {"title": "API", "url": "api/index"},
        ],
    }

copybutton_exclude = ".linenos, .gp, .go"
```

If the pydata theme warns that an option name is unknown, read
<https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html>
and rename it; do not delete the option silently.

- [ ] **Step 2: Write `docs/_static/custom.css`**

```css
/* superglm documentation identity: pulp comic energy from the logo,
   disciplined typography for the maths. See spec §9. */

html {
  --pst-font-family-base: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
  --pst-font-family-heading: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
  --pst-font-family-monospace: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --sg-display: "Bangers", Impact, "Arial Black", sans-serif;
  --sg-red: #d6402b;
  --sg-yellow: #f4b942;
}

html[data-theme="light"] {
  --pst-color-primary: #d6402b;
  --pst-color-secondary: #15171c;
  --pst-color-accent: #f4b942;
  --pst-color-text-base: #15171c;
  --pst-color-background: #ffffff;
  --pst-color-on-background: #f7f5f0;
  --pst-color-surface: #f7f5f0;
  --pst-color-link: #b8321f;
  --pst-color-link-hover: #15171c;
  --sg-ink: #15171c;
  --sg-shadow: #15171c;
}

html[data-theme="dark"] {
  --pst-color-primary: #f4b942;
  --pst-color-secondary: #f26b52;
  --pst-color-accent: #f4b942;
  --pst-color-text-base: #f2efe6;
  --pst-color-background: #15171c;
  --pst-color-on-background: #1c1f26;
  --pst-color-surface: #1c1f26;
  --pst-color-link: #f4b942;
  --pst-color-link-hover: #ffffff;
  --sg-ink: #f2efe6;
  --sg-shadow: #f4b942;
}

/* Comic panels: the recurring component. */
.sd-card,
div.admonition {
  border: 2px solid var(--sg-ink);
  border-radius: 4px;
  box-shadow: 4px 4px 0 var(--sg-shadow);
}

div.admonition.warning,
div.admonition.danger {
  border-color: var(--sg-red);
}

div.admonition > .admonition-title {
  font-weight: 600;
}

.sd-card {
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}

.sd-card:hover {
  transform: translate(-1px, -1px);
  box-shadow: 6px 6px 0 var(--sg-shadow);
}

.sd-card .sd-card-title,
.sg-display {
  font-family: var(--sg-display);
  font-weight: 400;
  letter-spacing: 0.04em;
}

.sd-card .sd-card-title {
  font-size: 1.5rem;
}

.sg-display {
  font-size: 2rem;
}

/* Hero band on the landing page only. */
.sg-hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(220px, 320px) 1fr;
  gap: 2rem;
  align-items: center;
  padding: 2.5rem 2rem;
  margin: 0 0 2rem;
  border: 2px solid var(--sg-ink);
  border-radius: 6px;
  box-shadow: 6px 6px 0 var(--sg-shadow);
  background: linear-gradient(100deg, #15171c 0%, #15171c 48%, #8c2418 78%, #d6402b 100%);
  color: #ffffff;
  overflow: hidden;
}

.sg-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(circle at 1.5px 1.5px, rgba(244, 185, 66, 0.75) 1.2px, transparent 1.9px) 0 0 / 9px 9px;
  -webkit-mask-image: linear-gradient(90deg, transparent 30%, #000 100%);
  mask-image: linear-gradient(90deg, transparent 30%, #000 100%);
}

.sg-hero > * {
  position: relative;
}

.sg-hero__logo img {
  display: block;
  width: 100%;
  max-width: 320px;
  margin: 0 auto;
  animation: sg-pop 0.45s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}

.sg-hero__title {
  font-family: var(--sg-display);
  font-weight: 400;
  font-size: clamp(2.6rem, 6vw, 4.2rem);
  line-height: 0.9;
  letter-spacing: 0.03em;
  color: var(--sg-yellow);
  text-shadow: 2px 2px 0 var(--sg-red), 4px 4px 0 #15171c;
  margin: 0 0 0.75rem;
}

.sg-hero__copy p {
  font-size: 1.15rem;
  line-height: 1.45;
  color: #f2efe6;
  max-width: 40rem;
  margin: 0 0 1rem;
}

.sg-hero__meta {
  display: block;
  margin-top: 0.75rem;
  font-size: 0.9rem;
  letter-spacing: 0.02em;
  color: var(--sg-yellow);
}

.sg-btn,
.sg-btn:visited {
  display: inline-block;
  padding: 0.5rem 1rem;
  margin: 0 0.6rem 0.6rem 0;
  border: 2px solid #15171c;
  border-radius: 3px;
  background: var(--sg-yellow);
  color: #15171c !important;
  font-weight: 600;
  text-decoration: none !important;
  box-shadow: 3px 3px 0 #15171c;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}

.sg-btn--ghost,
.sg-btn--ghost:visited {
  background: transparent;
  color: #ffffff !important;
  border-color: #ffffff;
  box-shadow: 3px 3px 0 var(--sg-yellow);
}

.sg-btn:hover,
.sg-btn:focus-visible {
  transform: translate(-1px, -1px);
  box-shadow: 5px 5px 0 #15171c;
}

.sg-btn--ghost:hover,
.sg-btn--ghost:focus-visible {
  box-shadow: 5px 5px 0 var(--sg-yellow);
}

@keyframes sg-pop {
  from {
    transform: scale(0.92) rotate(-6deg);
    opacity: 0;
  }
  to {
    transform: none;
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .sg-hero__logo img {
    animation: none;
  }
  .sd-card,
  .sg-btn {
    transition: none;
  }
  .sd-card:hover,
  .sg-btn:hover {
    transform: none;
  }
}

@media (max-width: 768px) {
  .sg-hero {
    grid-template-columns: 1fr;
    text-align: center;
    padding: 2rem 1.25rem;
  }
}
```

- [ ] **Step 3: Write `docs/_templates/autosummary/class.rst`**

```rst
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
```

- [ ] **Step 4: Replace `docs/index.md` with a temporary one-page landing**

The full landing page is Task 12. For the spike, overwrite `docs/index.md` with:

````markdown
---
html_theme.sidebar_secondary.remove: true
---

# superglm

```{raw} html
<div class="sg-hero">
  <div class="sg-hero__logo"><img src="_static/logo.png" alt="superglm"></div>
  <div class="sg-hero__copy">
    <div class="sg-hero__title">Super GLM</div>
    <p>Penalised GLMs and GAM pricing models for insurance, with the smoothness chosen by REML and the constraints you would otherwise enforce by hand.</p>
    <a class="sg-btn" href="tutorials/distributional-model.html">Fit your first model</a>
    <a class="sg-btn sg-btn--ghost" href="api/index.html">API reference</a>
    <span class="sg-hero__meta">MIT licensed · free for everyone · built on NumPy, SciPy and pandas</span>
  </div>
</div>
```

```{toctree}
:hidden:

tutorials/index
api/index
```
````

Create `docs/tutorials/index.md` and `docs/api/index.md` as one-line placeholders that Tasks 3 and 4 fill:

```markdown
# Tutorials

```{toctree}
:maxdepth: 1
```
```

```markdown
# API reference

```{toctree}
:maxdepth: 1
```
```

- [ ] **Step 5: Append build directories to `.gitignore`**

Append after the line `.cache/` (line 101):

```
# Sphinx build output and generated API pages
docs/_build/
docs/api/generated/
docs/jupyter_execute/
```

- [ ] **Step 6: Build strictly**

Run: `uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`
Expected: `build succeeded.` with no warnings. Common first failures and their fixes: an empty toctree warning (add the placeholder pages to it, or remove the directive until Task 3); a font URL rejected by `html_css_files` (Sphinx accepts absolute URLs; check the spelling); `redirects.txt` missing (create an empty file with a comment line `# legacy path  new page`).

- [ ] **Step 7: Commit**

```bash
git add docs/conf.py docs/_static/custom.css docs/_templates/autosummary/class.rst docs/index.md docs/tutorials/index.md docs/api/index.md docs/redirects.txt .gitignore
git commit -m "Docs: Sphinx skeleton with the pydata theme and the comic identity"
```

### Task 3: First executed notebook, from the distributional walkthrough

**Files:**
- Move: `docs/getting-started/distributional.md` → `docs/tutorials/distributional-model.md`
- Create: `docs/tutorials/distributional-model.ipynb` (generated by jupytext, outputs stripped)
- Create: `jupytext.toml`
- Modify: `docs/tutorials/index.md`

**Interfaces:**
- Produces: the MyST notebook front matter shape every later tutorial copies; proof that `skip-execution` is honoured by both MyST-NB and nbclient.

- [ ] **Step 1: Move the page**

Run: `git mv docs/getting-started/distributional.md docs/tutorials/distributional-model.md`

- [ ] **Step 2: Add notebook front matter and the Colab row**

Insert at the very top of `docs/tutorials/distributional-model.md`:

````markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Your first distributional model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StrudelDoodleS/superglm/blob/master/docs/tutorials/distributional-model.ipynb)
[Download the notebook](https://raw.githubusercontent.com/StrudelDoodleS/superglm/master/docs/tutorials/distributional-model.ipynb)

```{code-cell} ipython3
:tags: [skip-execution]

%pip install -q superglm
```
````

and delete the original `# Your first distributional model` heading line that follows.

- [ ] **Step 3: Convert every ` ```python ` fence to a code cell**

Run from the worktree root:

```bash
uv run python - <<'EOF'
from pathlib import Path
p = Path("docs/tutorials/distributional-model.md")
text = p.read_text(encoding="utf-8")
text = text.replace("```python\n", "```{code-cell} ipython3\n")
p.write_text(text, encoding="utf-8")
print(text.count("{code-cell}"), "code cells")
EOF
```

Expected: `7 code cells` (six executed fits and prints, plus the skip-execution cell).

The final "Choose another response family" block constructs a Tweedie model without fitting it; it runs in under a second and stays a code cell.

- [ ] **Step 4: Fix the three relative links at the bottom of the page**

The page links to `../models/distributional.md#family-predictor-names`, `../api/distributional.md` and `../models/distributional-inference.md`. Those pages move in Task 10. For now change them to the destinations they will have after Task 10:

```
../how-to/fit-a-distributional-model.md#family-predictor-names
../api/distributional.md
../how-to/check-a-distributional-fit.md
```

Until Task 10 lands, the strict build will report these as missing targets. To keep the spike building, wrap the closing paragraph in a MyST comment (`% ` prefix on each line) and remove the comment prefix in Task 10 step 9.

- [ ] **Step 5: Write `jupytext.toml`**

```toml
# Pair tutorial notebooks: the Markdown file is the source of truth, the
# .ipynb twin exists for Colab and download and never carries outputs.
[formats]
"docs/tutorials/" = "md:myst,ipynb"
```

- [ ] **Step 6: Generate the paired notebook**

Run: `uv run jupytext --sync docs/tutorials/distributional-model.md`
Expected: creates `docs/tutorials/distributional-model.ipynb`. Verify no outputs: `uv run python -c "import json; nb=json.load(open('docs/tutorials/distributional-model.ipynb')); print(sum(len(c.get('outputs', [])) for c in nb['cells']), 'outputs')"` prints `0 outputs`.

- [ ] **Step 7: Add the page to the tutorials toctree**

`docs/tutorials/index.md` becomes:

````markdown
# Tutorials

Executed on every build. Each one opens in Colab from its badge.

```{toctree}
:maxdepth: 1

distributional-model
```
````

- [ ] **Step 8: Build with execution and time it**

Run: `time SUPERGLM_DOCS_EXECUTE=force uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`
Expected: build succeeds; the rendered page `docs/_build/html/tutorials/distributional-model.html` contains the printed summary and the `held_out_loss` table; the `%pip` cell shows no output. Record the wall time in the PR description later (it is the T5 execution figure the spec asks for).

- [ ] **Step 9: Prove nbclient skips the `skip-execution` cell**

Run:

```bash
uv run python - <<'EOF'
import jupytext
from nbclient import NotebookClient
nb = jupytext.read("docs/tutorials/distributional-model.md")
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "docs/tutorials"}})
client.execute()
first = nb.cells[1] if nb.cells[0].cell_type == "markdown" else nb.cells[0]
pip_cell = next(c for c in nb.cells if c.cell_type == "code" and "%pip" in c.source)
print("pip cell executed:", pip_cell.get("execution_count") is not None)
print("fit cells with output:", sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs")))
EOF
```

Expected: `pip cell executed: False` and `fit cells with output: 5` or more. If the pip cell executed, set `client = NotebookClient(..., skip_cells_with_tag="skip-execution")` explicitly and record that in Task 14's test.

- [ ] **Step 10: Commit**

```bash
git add docs/tutorials/distributional-model.md docs/tutorials/distributional-model.ipynb docs/tutorials/index.md jupytext.toml
git commit -m "Docs: convert the distributional walkthrough into an executed MyST notebook"
```

### Task 4: API skeleton with autosummary and numpydoc

**Files:**
- Create: `docs/api/model.md`
- Modify: `docs/api/index.md`

**Interfaces:**
- Produces: the `{eval-rst}` + `.. autosummary::` block shape used by every API page in Task 9 (a literal ```` ```{autosummary} ```` MyST fence renders but generates no stubs: autosummary's stub scanner matches the reST directive `.. autosummary::` only); a count of numpydoc warnings on real docstrings.

- [ ] **Step 1: Write `docs/api/model.md`**

````markdown
# Model

`SuperGLM` is the estimator for penalised GLMs and GAM-style pricing models.
Construct it with a family and a feature specification, fit it with
`fit_reml` for REML smoothness selection or `fit` for fixed penalties, then
read the fit through `summary`, `term_inference` and the plotting methods.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLM
   superglm.PathResult
   superglm.REMLResult
   superglm.LambdaPolicy
   superglm.warmup
   superglm.ModelSummary
   superglm.ModelMetrics
   superglm.FitDiagnosticReport
   superglm.DiscretizationResult
   superglm.discretization_impact
```
````

- [ ] **Step 2: Point the API index at it**

`docs/api/index.md` becomes:

````markdown
# API reference

Generated from the docstrings of the public names in `superglm.__all__`.
Every method and attribute of `SuperGLM` and `SuperLSS` has its own page.

```{toctree}
:maxdepth: 1

model
```
````

- [ ] **Step 3: Build and collect warnings**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n --keep-going docs docs/_build/html 2>&1 | tee /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/api-warnings.txt | grep -c WARNING`
Expected: a warning count (any number). Read the file. Three kinds appear:

1. `py:class reference target not found: <shorthand>` for prose type names in docstrings: add a regex to `nitpick_ignore_regex` in `conf.py`.
2. `py:class reference target not found: superglm.<module>.<Name>` for real objects that have no page yet. Do **not** assume these self-resolve: Task 9 is driven by `superglm.__all__`, so a dotted path whose leaf is not exported never gets a page. Check the leaf against `superglm.__all__` first. Measured on the spike: of the 18 residual warnings, 14 occurrences (11 distinct names) name objects outside `__all__` and were dispositioned in Task 4 with named `nitpick_ignore_regex` entries — `superglm.distributions.Distribution`, `superglm.links.Link`, `superglm.penalties.base.Penalty`, `superglm.solvers.pirls.PIRLSResult`, `superglm.types.FeatureSpec`, `superglm.diagnostics.fit_report.DiagnosticFinding` / `FitWorkProfile` / `JsonValue`, plus the bare `_CoefRow`, `_BasisDetailRow` and `SummaryLevelDisplay`. Only the four bare names that *are* in `__all__` (`FactorSmoothResult`, `RandomEffectResult`, `TermInference`, `InteractionInference`) can resolve from Task 9's pages; they carry a TEMPORARY ignore in `conf.py` that Task 9 step 5 must delete and re-verify, qualifying the docstring instead if an unqualified name still will not bind.
3. numpydoc parse warnings (`Unknown section`, `Unexpected section title`): fix the docstring in `src/superglm/` surgically. These are docs work and in scope.

- [ ] **Step 4: Verify the generated member pages exist**

Run: `ls docs/api/generated | grep -c 'superglm.SuperGLM\.'`
Expected: at least 40 (one page per public method and attribute of `SuperGLM`).

- [ ] **Step 5: Commit**

```bash
git add docs/api/model.md docs/api/index.md docs/conf.py src/superglm
git commit -m "Docs: autosummary reference for the model surface, numpydoc warnings settled"
```

### Task 5: Theme comparison screenshots and the decision gate

**Files:**
- Create (scratch, not committed): `/tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/spike/shoot.py`

**Interfaces:**
- Consumes: the site from Tasks 2-4.
- Produces: twelve screenshots (two themes × three pages × light and dark) and Max's theme decision.

- [ ] **Step 1: Build under pydata**

Run: `SUPERGLM_DOCS_EXECUTE=cache uv run sphinx-build -b html -n --keep-going docs docs/_build/pydata`

- [ ] **Step 2: Build under Shibuya without adding it to the project**

Run: `SUPERGLM_DOCS_THEME=shibuya SUPERGLM_DOCS_EXECUTE=cache uv run --with shibuya sphinx-build -b html -n --keep-going docs docs/_build/shibuya`
Expected: builds; the custom CSS variables prefixed `--pst-` have no effect under Shibuya, which is fine for a comparison of the themes' own looks.

- [ ] **Step 3: Write the screenshot script**

```python
"""Screenshot three pages under two themes in light and dark."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path("/home/max/projects/superglm/.claude/worktrees/docs-rebuild-sphinx/docs/_build")
OUT = Path(__file__).parent
PAGES = {
    "landing": "index.html",
    "tutorial": "tutorials/distributional-model.html",
    "api": "api/generated/superglm.SuperGLM.html",
}
with sync_playwright() as p:
    browser = p.chromium.launch()
    for theme in ("pydata", "shibuya"):
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme=scheme)
            for name, rel in PAGES.items():
                page = ctx.new_page()
                page.goto((ROOT / theme / rel).as_uri(), wait_until="load")
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / f"{theme}-{name}-{scheme}.jpg"), type="jpeg", quality=80)
                page.close()
            ctx.close()
    browser.close()
print("done")
```

Run: `uv run python /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/spike/shoot.py`
Expected: twelve `.jpg` files. Note: pydata's dark mode is chosen by its own toggle stored in `localStorage`, not only by `prefers-color-scheme`; if the dark screenshots come out light, add `page.evaluate("document.documentElement.setAttribute('data-theme','dark')")` before the screenshot.

- [ ] **Step 4: Put the screenshots in front of Max**

Assemble them into one HTML page (same pattern as the mood board: one frame per screenshot, theme and page labelled) and publish it as an artifact, or send the twelve files with SendUserFile. Ask one question: A (pydata) or B (Shibuya). Default on silence: A.

- [ ] **Step 5: Record the decision**

If A: remove the `elif html_theme == "shibuya":` block from `conf.py`. If B: add `shibuya>=2026.7` to the docs group, set the default theme name to `shibuya`, and port the `--pst-*` variables in `custom.css` to Shibuya's `--sy-*` variables per <https://shibuya.lepture.com/customisation/colors/>. Commit either way:

```bash
git add docs/conf.py docs/_static/custom.css pyproject.toml uv.lock
git commit -m "Docs: settle the theme after the spike comparison"
```

---

## Phase B: the structural pull request

### Task 6: Test scaffolding and the API coverage test

**Files:**
- Create: `tests/docs/__init__.py` (empty)
- Create: `tests/docs/test_api_reference.py`
- Modify: `pyproject.toml` (`markers` list under `[tool.pytest.ini_options]`, lines 135-138)

**Interfaces:**
- Produces: the `docs` pytest marker; `documented_names()` helper used only inside this file.

- [ ] **Step 1: Register the marker**

Add to the `markers` list in `pyproject.toml`:

```toml
    "docs: executes documentation notebooks and checks docs conventions (deselect with '-m \"not docs\"')",
```

- [ ] **Step 2: Write the failing test**

`tests/docs/test_api_reference.py`:

```python
"""Every public name in ``superglm.__all__`` has an entry in the API reference.

The reference is generated by autosummary from the ``.. autosummary::``
directives inside the ``{eval-rst}`` fences under ``docs/api``. A new
export without an entry fails here, not on a reader's screen.
"""

from __future__ import annotations

import re
from pathlib import Path

import superglm

DOCS_API = Path(__file__).resolve().parents[2] / "docs" / "api"
AUTOSUMMARY_BLOCK = re.compile(
    r"```\{eval-rst\}\s*\n\s*\.\.\s+autosummary::(.*?)```", re.S
)


def documented_names() -> set[str]:
    names: set[str] = set()
    for page in DOCS_API.glob("*.md"):
        for block in AUTOSUMMARY_BLOCK.findall(page.read_text(encoding="utf-8")):
            for raw in block.splitlines():
                line = raw.strip()
                if line and not line.startswith(":"):
                    names.add(line.removeprefix("superglm."))
    return names


def test_every_public_name_has_a_reference_entry() -> None:
    missing = sorted(set(superglm.__all__) - documented_names())
    assert missing == [], f"public names without an API reference entry: {missing}"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/docs/test_api_reference.py -q`
Expected: FAIL listing about 120 missing names (everything except the ten on `api/model.md`).

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/docs/__init__.py tests/docs/test_api_reference.py pyproject.toml
git commit -m "Docs tests: every public name must have an API reference entry"
```

### Task 7: Move the design records out of `docs/`

**Files:**
- Move: `docs/research/` → `notes/research/`, `docs/audit/` → `notes/audit/`, `docs/ROADMAP.md` → `notes/ROADMAP.md`, `docs/guide/scop-performance-prototype.md` → `notes/audit/scop-performance-prototype.md`
- Modify: `.gitignore` (line 70 `!docs/ROADMAP.md`), `AGENTS.md:82`, `tests/test_reml_tol_determination.py:297`, `tests/test_nb_theta_estimation_correctness.py:6`, `tests/test_tweedie_reml_exact_scale.py:8`, `src/superglm/model/screening_ops.py:1150`, `src/superglm/_group_matrix/_group_matrix_support.py:48`, `benchmarks/c3_c1_complete_fit.md:91`, `benchmarks/benchmark_support_stress.py:11`, `benchmarks/benchmark_tensor_cost.py:11`, `benchmarks/pr381_fable_followup.md:135-136`, `benchmarks/lss_convergence_repair.md:127-135`, `benchmarks/c3_practical/README.md:46`

- [ ] **Step 1: Move**

```bash
mkdir -p notes
git mv docs/research notes/research
git mv docs/audit notes/audit
git mv docs/ROADMAP.md notes/ROADMAP.md
git mv docs/guide/scop-performance-prototype.md notes/audit/scop-performance-prototype.md
```

- [ ] **Step 2: Fix the ignore exception**

In `.gitignore` change line 70 `!docs/ROADMAP.md` to `!notes/ROADMAP.md`. Verify: `git check-ignore -v notes/ROADMAP.md` prints nothing (not ignored).

- [ ] **Step 3: Rewrite the path references**

```bash
sed -i 's#docs/research/#notes/research/#g; s#docs/audit/#notes/audit/#g; s#docs/ROADMAP\.md#notes/ROADMAP.md#g' \
  AGENTS.md tests/test_reml_tol_determination.py tests/test_nb_theta_estimation_correctness.py \
  tests/test_tweedie_reml_exact_scale.py src/superglm/model/screening_ops.py \
  src/superglm/_group_matrix/_group_matrix_support.py benchmarks/c3_c1_complete_fit.md \
  benchmarks/benchmark_support_stress.py benchmarks/benchmark_tensor_cost.py \
  benchmarks/pr381_fable_followup.md benchmarks/lss_convergence_repair.md benchmarks/c3_practical/README.md
```

Relative links inside `benchmarks/*.md` were `../docs/research/...`; after the sed they read `../notes/research/...`, which is correct from `benchmarks/`.

- [ ] **Step 4: Verify nothing still points at the old paths**

Run: `grep -rn -E 'docs/(research|audit|ROADMAP)' --include='*.md' --include='*.py' --include='*.toml' --include='*.yml' . | grep -v -E '^\./(\.venv|notes|docs/superpowers)/'`
Expected: no output.

- [ ] **Step 5: Run the three touched tests' modules to confirm they still import**

Run: `uv run pytest tests/test_reml_tol_determination.py tests/test_nb_theta_estimation_correctness.py tests/test_tweedie_reml_exact_scale.py -q -x --co | tail -1`
Expected: `N tests collected` with no error.

- [ ] **Step 6: Commit**

```bash
git add -A notes docs .gitignore AGENTS.md tests src benchmarks
git commit -m "Move research reports, audits and the roadmap out of the published docs tree"
```

### Task 8: Complete API reference

**Files:**
- Create: `docs/api/distributional.md`, `docs/api/features.md`, `docs/api/families-and-links.md`, `docs/api/penalties.md`, `docs/api/inference.md`, `docs/api/validation-and-diagnostics.md`, `docs/api/plotting.md`, `docs/api/export.md`, `docs/api/sklearn.md`, `docs/api/stats.md`, `docs/api/editor.md`, `docs/api/warnings-and-exceptions.md`
- Delete: `docs/api/diagnostics.md`, `docs/api/model_selection.md`, `docs/api/validation.md` (the old mkdocstrings stubs; the other old stubs are overwritten by name)
- Modify: `docs/api/index.md`

**Interfaces:**
- Consumes: the block shape from Task 4.
- Produces: an entry for all 131 names in `superglm.__all__`; page names used by the redirect map in Task 11.

- [ ] **Step 1: Delete the old stubs that keep no name**

```bash
git rm docs/api/diagnostics.md docs/api/model_selection.md docs/api/validation.md
```

The remaining old stubs (`distributional.md`, `families.md`, `features.md`, `inference.md`, `penalties.md`, `plotting.md`) are overwritten below; `families.md` becomes `families-and-links.md`, so `git rm docs/api/families.md` as well.

- [ ] **Step 2: Write the twelve pages**

Each page is one orientation paragraph followed by an `{eval-rst}` fence holding a `.. autosummary::` directive (the shape Task 4 settled; a literal `{autosummary}` MyST fence generates no stubs). The name lists below are complete; together with `model.md` they cover all 131 public names. Write each file exactly.

`docs/api/distributional.md`:

````markdown
# Distributional models

`SuperLSS` fits several parameters of a response distribution together, one
predictor per parameter. Pass a family first, then one predictor declaration
per parameter using the family's helper methods; build the terms inside each
declaration with `s`, `cat`, `re`, `ti`, `term` and `interaction`. Start
with the [tutorial](../tutorials/distributional-model.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperLSS
   superglm.Predictor
   superglm.BoundPredictor
   superglm.bind_predictor
   superglm.BoundTerm
   superglm.BoundInteraction
   superglm.term
   superglm.s
   superglm.cat
   superglm.re
   superglm.ti
   superglm.interaction
   superglm.GaussianLS
   superglm.GammaLS
   superglm.LogNormalLS
   superglm.NegativeBinomialLS
   superglm.GeneralizedGammaLSS
   superglm.GeneralizedParetoLSS
   superglm.TweedieLSS
   superglm.TwoPieceLogNormalLSS
   superglm.TwoPieceNormalLSS
```
````

`docs/api/features.md`:

````markdown
# Features

Feature specifications turn raw columns into model terms. `Spline` is the
public factory for smooth terms; the concrete spline classes are what it
returns. Categorical, ordered-categorical, numeric, polynomial and piecewise
terms cover the rest of a rating structure; `FactorSmooth` and
`RandomEffect` add credibility-style shrinkage; the interaction classes
combine terms; `Constraint` requests monotone or curvature constraints.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.Spline
   superglm.PSpline
   superglm.BSplineSmooth
   superglm.NaturalSpline
   superglm.CubicRegressionSpline
   superglm.n_knots_from_k
   superglm.Categorical
   superglm.OrderedCategorical
   superglm.Numeric
   superglm.Piecewise
   superglm.Polynomial
   superglm.FactorSmooth
   superglm.RandomEffect
   superglm.LevelGrouping
   superglm.collapse_levels
   superglm.Constraint
   superglm.ConstraintSpec
   superglm.LinearConstraintSet
   superglm.SplineCategorical
   superglm.PolynomialCategorical
   superglm.NumericCategorical
   superglm.CategoricalInteraction
   superglm.NumericInteraction
   superglm.PolynomialInteraction
   superglm.TensorInteraction
```
````

`docs/api/families-and-links.md`:

````markdown
# Families and links

Response families define the variance function and the weight semantics;
links map the linear predictor to the mean. The negative-binomial and Tweedie
profilers estimate the extra parameters those families carry.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.families
   superglm.Poisson
   superglm.Gaussian
   superglm.Gamma
   superglm.Binomial
   superglm.NegativeBinomial
   superglm.Tweedie
   superglm.LogLink
   superglm.LogitLink
   superglm.IdentityLink
   superglm.ProbitLink
   superglm.CloglogLink
   superglm.CauchitLink
   superglm.InverseLink
   superglm.InverseSquaredLink
   superglm.SqrtLink
   superglm.PowerLink
   superglm.NegativeBinomialLink
   superglm.estimate_nb_theta
   superglm.NBProfileResult
   superglm.estimate_tweedie_p
   superglm.estimate_phi
   superglm.TweedieProfileResult
   superglm.TweedieProfileCIDetails
   superglm.TweedieProfileCIDensityProvenance
   superglm.TweedieProfileCIEndpoint
   superglm.TweedieProfileCIEvaluation
   superglm.tweedie_logpdf
   superglm.generate_tweedie_cpg
```
````

`docs/api/penalties.md`:

````markdown
# Penalties

Penalty objects are the low-level interface behind `selection_penalty=` and
`spline_penalty=`. Prefer the model-level arguments; reach for these classes
when you need a specific group structure or an adaptive weighting.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.GroupElasticNet
   superglm.GroupLasso
   superglm.SparseGroupLasso
   superglm.Ridge
   superglm.Adaptive
```
````

`docs/api/inference.md`:

````markdown
# Inference results

The objects returned by `term_inference`, `random_effects`, `factor_smooth`,
the shape-repair methods and the spline diagnostics. They are plain data
containers; the model methods that produce them are documented on the
[model page](model.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.TermInference
   superglm.SmoothCurve
   superglm.InteractionInference
   superglm.SplineMetadata
   superglm.RandomEffectResult
   superglm.FactorSmoothResult
   superglm.MonotoneRepairer
   superglm.MonotoneRepairResult
   superglm.SplineRedundancyReport
```
````

`docs/api/validation-and-diagnostics.md`:

````markdown
# Validation and diagnostics

Cross-validation, the actuarial ranking and lift charts, and the
model-adequacy tests for dispersion, zero inflation and non-nested
comparison.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.cross_validate
   superglm.CrossValidationResult
   superglm.lorenz_curve
   superglm.LorenzCurveResult
   superglm.lift_chart
   superglm.LiftChartResult
   superglm.double_lift_chart
   superglm.DoubleLiftChartResult
   superglm.loss_ratio_chart
   superglm.LossRatioChartResult
   superglm.dispersion_test
   superglm.DispersionTestResult
   superglm.score_test_zi
   superglm.ScoreTestZIResult
   superglm.zero_inflation_index
   superglm.ZeroInflationResult
   superglm.vuong_test
   superglm.VuongTestResult
```
````

`docs/api/plotting.md`:

````markdown
# Plotting

Term comparison across models. The per-model plotting methods (`plot`,
`plot_data`, `plot_diagnostics`) live on `SuperGLM` and are documented on
the [model page](model.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.plot_term_comparison
```
````

`docs/api/export.md`:

````markdown
# Export

Rating-table export for a fitted model, and the error raised when a base
level cannot be represented in the requested block shape.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.export_rating_tables
   superglm.RatingTableBaseNotRepresentableError
```
````

`docs/api/sklearn.md`:

````markdown
# scikit-learn wrappers

Estimators with the scikit-learn interface, for pipelines, grid search and
cross-validation tooling that expects `fit`, `predict` and `get_params`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SuperGLMRegressor
   superglm.SuperGLMClassifier
```
````

`docs/api/stats.md`:

````markdown
# Statistical helpers

Distribution functions used by the smooth-term tests, exposed for readers
who want to reproduce a p-value by hand.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.psum_chisq
   superglm.satterthwaite
   superglm.wood_test_smooth
```
````

`docs/api/editor.md`:

````markdown
# Model editor

The browser editor opens a fitted model for interactive editing of levels,
groupings and offsets, then returns the edited model. These names live in
`superglm.editor`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.editor.edit
   superglm.editor.EditorSession
   superglm.editor.EditableTerm
   superglm.editor.EditRecord
```
````

`docs/api/warnings-and-exceptions.md`:

````markdown
# Warnings and exceptions

Everything superglm raises or warns with that a caller may want to catch or
silence. Each docstring says when it fires and what to do about it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.PublicationModeError
   superglm.RatingTableBaseNotRepresentableError
   superglm.SeparationError
   superglm.SeparationWarning
   superglm.NBThetaBoundWarning
   superglm.FractionalFrequencyWeightWarning
   superglm.PriorWeightLatticeWarning
```
````

- [ ] **Step 3: Rewrite `docs/api/index.md`**

````markdown
# API reference

Generated from the docstrings of every public name in `superglm.__all__`.
Every method and attribute of `SuperGLM` and `SuperLSS` has its own page.
Start from the model pages; the rest are the objects they take and return.

```{toctree}
:maxdepth: 1

model
distributional
features
families-and-links
penalties
inference
validation-and-diagnostics
plotting
export
sklearn
stats
editor
warnings-and-exceptions
```
````

- [ ] **Step 4: Run the coverage test**

Run: `uv run pytest tests/docs/test_api_reference.py -q`
Expected: PASS. If a name is reported missing, it was added to `__all__` after this plan was written; place it on the page whose paragraph describes it.

- [ ] **Step 5: Build strictly and burn down the reference warnings**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html 2>&1 | grep -E 'WARNING|ERROR' | sort | uniq -c | sort -rn | head -40`

Work through the output. Rules: a shorthand type name gets a `nitpick_ignore_regex` entry; a genuine dotted path that does not resolve means the docstring names a private or moved object, fix the docstring; a numpydoc section warning is a docstring format fix. Repeat until the command prints nothing, then run the full strict build once and confirm `build succeeded`.

- [ ] **Step 6: Commit**

```bash
git add docs/api docs/conf.py src/superglm
git commit -m "Docs: complete API reference, one autosummary page per public name"
```

### Task 9: Admonition conversion script

**Files:**
- Create (scratch, not committed): `/tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/convert_admonitions.py`

**Interfaces:**
- Produces: a converter used in Task 10 step 6 on the four files that carry the 22 `!!!` admonitions (`guide/families.md`, `guide/interactions.md`, `guide/optimization.md`, `models/distributional-inference.md`).

- [ ] **Step 1: Write the converter**

```python
"""Convert Material-style admonitions to MyST admonition directives.

    !!! note "Title"
        body line
        body line

becomes

    ```{admonition} Title
    :class: note
    body line
    body line
    ```

An untitled ``!!! note`` becomes a bare ```{note}``` directive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER = re.compile(r'^!!!\s+(\w+)(?:\s+"([^"]*)")?\s*$')


def convert(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    count = 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        kind, title = m.group(1), m.group(2)
        i += 1
        body: list[str] = []
        while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
            body.append(lines[i][4:] if lines[i].startswith("    ") else "")
            i += 1
        while body and body[-1] == "":
            body.pop()
        if title:
            out.append(f"```{{admonition}} {title}")
            out.append(f":class: {kind}")
        else:
            out.append(f"```{{{kind}}}")
        out.extend(body)
        out.append("```")
        out.append("")
        count += 1
    return "\n".join(out) + "\n", count


if __name__ == "__main__":
    total = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        new, n = convert(path.read_text(encoding="utf-8"))
        path.write_text(new, encoding="utf-8")
        print(f"{path}: {n}")
        total += n
    print("total", total)
```

- [ ] **Step 2: Dry-run on a copy**

Run: `cp docs/guide/interactions.md /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/interactions-copy.md && uv run python /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/convert_admonitions.py /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/interactions-copy.md`
Expected: `...: 1` and the copy contains a `{admonition} Rank and aggregate metrics cannot detect this failure` block with `:class: warning` and its original body dedented. No commit; scratch only.

### Task 10: Move and convert every existing page

**Files:**
- Move: 30 pages, exact commands below.
- Modify: the moved pages (admonitions, links, the T5 comment from Task 3).
- Delete: `mkdocs.yml`, `docs/javascripts/mathjax.js`, `docs/stylesheets/extra.css`.

**Interfaces:**
- Produces: the final page paths that Task 11's section indexes, Task 12's landing page and Task 13's redirect map refer to.

- [ ] **Step 1: Create the destination directories**

```bash
mkdir -p docs/get-started docs/how-to docs/explanation docs/governance docs/development/internals docs/examples
```

- [ ] **Step 2: Move the pages**

```bash
git mv docs/getting-started/installation.md docs/get-started/installation.md
git mv docs/getting-started/quickstart.md docs/get-started/quickstart.md
git mv docs/guide/workflows.md docs/how-to/recommended-workflows.md
git mv docs/guide/fitting.md docs/how-to/choose-a-fitting-path.md
git mv docs/guide/features.md docs/how-to/specify-features.md
git mv docs/guide/interactions.md docs/how-to/specify-interactions.md
git mv docs/guide/monotone.md docs/how-to/constrain-a-smooth.md
git mv docs/guide/validation.md docs/how-to/compare-models-on-holdout.md
git mv docs/guide/results.md docs/how-to/read-a-summary-and-plot-effects.md
git mv docs/guide/deployment.md docs/how-to/deploy-a-fitted-model.md
git mv docs/guide/screening.md docs/how-to/screen-interactions.md
git mv docs/models/distributional.md docs/how-to/fit-a-distributional-model.md
git mv docs/models/distributional-inference.md docs/how-to/check-a-distributional-fit.md
git mv docs/guide/editor.md docs/tutorials/edit-a-model-in-the-browser.md
git mv docs/guide/credibility.md docs/explanation/credibility-as-smoothing.md
git mv docs/guide/families.md docs/explanation/families-and-weights.md
git mv docs/guide/optimization.md docs/explanation/solvers-and-internals.md
git mv docs/guide/screening-evaluation.md docs/explanation/what-screening-does.md
git mv docs/governance/model_risk_pack.md docs/governance/model-risk-pack.md
git mv docs/development/cost-and-timing.md docs/governance/reproducibility.md
git mv docs/development/python-support.md docs/governance/python-support.md
git mv docs/development/data-and-solver-boundaries.md docs/development/internals/data-and-solver-boundaries.md
git mv docs/distributional-family-development.md docs/development/internals/distributional-family-development.md
git mv docs/editor_frontend.md docs/development/internals/editor-frontend.md
git mv docs/tabmat-integration-notes.md docs/development/internals/tabmat-integration-notes.md
git mv docs/notebooks/editor_demo.ipynb docs/examples/editor_demo.ipynb
git mv docs/notebooks/mtpl2_frequency_walkthrough.ipynb docs/examples/mtpl2_frequency_walkthrough.ipynb
git mv docs/notebooks/ordered_categorical_smoothing.ipynb docs/examples/ordered_categorical_smoothing.ipynb
git mv docs/notebooks/plotting_diagnostics_demo.ipynb docs/examples/plotting_diagnostics_demo.ipynb
git mv docs/notebooks/tweedie_profile_estimation.ipynb docs/examples/tweedie_profile_estimation.ipynb
git rm mkdocs.yml docs/javascripts/mathjax.js docs/stylesheets/extra.css
rmdir docs/getting-started docs/guide docs/models docs/notebooks docs/javascripts docs/stylesheets 2>/dev/null; true
```

`docs/development/releases.md` and `docs/development/migrations/*.md` stay where they are.

- [ ] **Step 3: Rewrite cross-page links for the moves**

Every old path substring is unique, so one `sed` pass over all Markdown under `docs/` (excluding `superpowers/`) handles links written as `../guide/x.md`, `guide/x.md` or `x.md` from a sibling:

```bash
find docs -name '*.md' -not -path 'docs/superpowers/*' -print0 | xargs -0 sed -i \
  -e 's#getting-started/installation\.md#get-started/installation.md#g' \
  -e 's#getting-started/quickstart\.md#get-started/quickstart.md#g' \
  -e 's#getting-started/distributional\.md#tutorials/distributional-model.md#g' \
  -e 's#guide/workflows\.md#how-to/recommended-workflows.md#g' \
  -e 's#guide/fitting\.md#how-to/choose-a-fitting-path.md#g' \
  -e 's#guide/features\.md#how-to/specify-features.md#g' \
  -e 's#guide/interactions\.md#how-to/specify-interactions.md#g' \
  -e 's#guide/monotone\.md#how-to/constrain-a-smooth.md#g' \
  -e 's#guide/validation\.md#how-to/compare-models-on-holdout.md#g' \
  -e 's#guide/results\.md#how-to/read-a-summary-and-plot-effects.md#g' \
  -e 's#guide/deployment\.md#how-to/deploy-a-fitted-model.md#g' \
  -e 's#guide/screening\.md#how-to/screen-interactions.md#g' \
  -e 's#guide/screening-evaluation\.md#explanation/what-screening-does.md#g' \
  -e 's#models/distributional\.md#how-to/fit-a-distributional-model.md#g' \
  -e 's#models/distributional-inference\.md#how-to/check-a-distributional-fit.md#g' \
  -e 's#guide/editor\.md#tutorials/edit-a-model-in-the-browser.md#g' \
  -e 's#guide/credibility\.md#explanation/credibility-as-smoothing.md#g' \
  -e 's#guide/families\.md#explanation/families-and-weights.md#g' \
  -e 's#guide/optimization\.md#explanation/solvers-and-internals.md#g' \
  -e 's#governance/model_risk_pack\.md#governance/model-risk-pack.md#g' \
  -e 's#development/cost-and-timing\.md#governance/reproducibility.md#g' \
  -e 's#development/python-support\.md#governance/python-support.md#g' \
  -e 's#development/data-and-solver-boundaries\.md#development/internals/data-and-solver-boundaries.md#g' \
  -e 's#distributional-family-development\.md#development/internals/distributional-family-development.md#g' \
  -e 's#editor_frontend\.md#development/internals/editor-frontend.md#g' \
  -e 's#tabmat-integration-notes\.md#development/internals/tabmat-integration-notes.md#g' \
  -e 's#notebooks/\([a-z_]*\)\.ipynb#examples/\1.ipynb#g' \
  -e 's#api/families\.md#api/families-and-links.md#g' \
  -e 's#api/validation\.md#api/validation-and-diagnostics.md#g' \
  -e 's#api/diagnostics\.md#api/validation-and-diagnostics.md#g' \
  -e 's#api/model_selection\.md#api/validation-and-diagnostics.md#g'
```

Links whose source page changed depth (the three pages now under `development/internals/`, and `governance/reproducibility.md` which was under `development/`) may now have one `../` too few or too many. The strict build in step 8 reports each as `myst.xref_missing`; fix those by hand.

- [ ] **Step 4: Rewrite the research links that now point outside the site**

Four pages link into `../research/...` (`how-to/screen-interactions.md`, `explanation/what-screening-does.md`, `how-to/fit-a-distributional-model.md` twice). Replace each with the GitHub URL of the moved file, for example:

```
https://github.com/StrudelDoodleS/superglm/blob/master/notes/research/2026-09-psst-reference-variance.md
```

Find them with `grep -rn 'research/' docs --include='*.md' | grep -v superpowers`.

- [ ] **Step 5: Convert the admonitions**

Run: `uv run python /tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/convert_admonitions.py docs/explanation/families-and-weights.md docs/how-to/specify-interactions.md docs/explanation/solvers-and-internals.md docs/how-to/check-a-distributional-fit.md`
Expected: `total 22`. Then `grep -rn '^!!!' docs --include='*.md' | grep -v superpowers` prints nothing.

- [ ] **Step 6: Remove the hand-written table of contents in the solvers page**

`docs/explanation/solvers-and-internals.md` opens with a `## Contents` section that duplicates the theme's own page contents. Delete that section (the heading and its bullet list) only.

- [ ] **Step 7: Restore the T5 closing paragraph**

Remove the `% ` comment prefixes added in Task 3 step 4 from the closing paragraph of `docs/tutorials/distributional-model.md`; its three links now resolve.

- [ ] **Step 8: Write the examples index**

`docs/examples/index.md`:

````markdown
# Examples

Earlier demonstration notebooks, rendered from their committed outputs. Each
is retired as the tutorial that replaces it lands.

```{toctree}
:maxdepth: 1

mtpl2_frequency_walkthrough
plotting_diagnostics_demo
ordered_categorical_smoothing
tweedie_profile_estimation
editor_demo
```
````

- [ ] **Step 9: Build strictly and fix what it reports**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html 2>&1 | grep -E 'WARNING|ERROR'`

Expected categories and fixes:

- `document isn't included in any toctree`: expected for every moved page until Task 11 adds the section indexes; ignore in this task only.
- `myst.xref_missing` for a relative link: fix the path by hand.
- `myst.header` "Non-consecutive header level": a page skips from `#` to `###`; change the heading level.
- A fence language Sphinx's highlighter does not know (`text`, `console`, `pycon` are fine; `python title="..."` is not): remove the attribute.
- `Duplicate explicit target name` from repeated link text: leave; it is informational and not a warning under `-W` unless reported as one.

Stop when the only remaining warnings are the toctree ones.

- [ ] **Step 10: Commit**

```bash
git add -A docs mkdocs.yml
git commit -m "Docs: move every page into the Diátaxis tree and convert MkDocs syntax"
```

### Task 11: Section index pages and the top-level navigation

**Files:**
- Create: `docs/get-started/index.md`, `docs/how-to/index.md`, `docs/explanation/index.md`, `docs/governance/index.md`, `docs/development/index.md`, `docs/development/internals/index.md`
- Modify: `docs/tutorials/index.md`, `docs/index.md` (toctree only; the landing content is Task 12)

**Interfaces:**
- Produces: every page in exactly one toctree; the seven navbar entries.

- [ ] **Step 1: Write the section indexes**

`docs/get-started/index.md`:

````markdown
# Get started

Install the package, then fit your first model. The quick start collects the
fit paths in one place until the first pricing tutorial replaces it.

```{toctree}
:maxdepth: 1

installation
quickstart
```
````

`docs/tutorials/index.md`:

````markdown
# Tutorials

Lessons that take you from data to a decision. Each executed page opens in
Colab from its badge. The examples section holds earlier demonstration
notebooks that are being replaced one by one.

```{toctree}
:maxdepth: 1

distributional-model
edit-a-model-in-the-browser
../examples/index
```
````

`docs/how-to/index.md`:

````markdown
# How-to guides

Directions for one goal at a time. These pages assume you have a frame, a
response and a question; the tutorials are where to learn the workflow.

```{toctree}
:maxdepth: 1

recommended-workflows
choose-a-fitting-path
specify-features
specify-interactions
constrain-a-smooth
compare-models-on-holdout
read-a-summary-and-plot-effects
deploy-a-fitted-model
screen-interactions
fit-a-distributional-model
check-a-distributional-fit
```
````

`docs/explanation/index.md`:

````markdown
# Explanation

Why the package works the way it does: the estimators behind `fit_reml`,
what weights mean, how credibility becomes smoothing, and what interaction
screening can and cannot detect.

```{toctree}
:maxdepth: 1

families-and-weights
credibility-as-smoothing
what-screening-does
solvers-and-internals
```
````

`docs/governance/index.md`:

````markdown
# Governance

What a model-risk reviewer needs: the evidence pack a fitted model carries,
how cost and timing claims are made, and which Python versions are
supported and why.

```{toctree}
:maxdepth: 1

model-risk-pack
reproducibility
python-support
```
````

`docs/development/index.md`:

````markdown
# Development

Release policy, migration notes for behaviour changes, and the internals a
contributor needs. Research reports, audits and the roadmap live in the
repository under [`notes/`](https://github.com/StrudelDoodleS/superglm/tree/master/notes).

```{toctree}
:maxdepth: 1

releases
migrations/group-pricing-rank
migrations/weight-semantics-prior
migrations/family-bound-predictors
internals/index
```
````

`docs/development/internals/index.md`:

````markdown
# Internals

```{toctree}
:maxdepth: 1

data-and-solver-boundaries
distributional-family-development
editor-frontend
tabmat-integration-notes
```
````

- [ ] **Step 2: Set the root toctree**

Replace the `{toctree}` block at the bottom of `docs/index.md` with:

````markdown
```{toctree}
:hidden:

get-started/index
tutorials/index
how-to/index
explanation/index
api/index
governance/index
development/index
```
````

- [ ] **Step 3: Build strictly**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`
Expected: `build succeeded` with zero warnings. A page still reported as not in any toctree is missing from one of the indexes above; add it.

- [ ] **Step 4: Check the navbar**

Run: `grep -o 'class="nav-link[^>]*>[^<]*' docs/_build/html/index.html | head -12`
Expected: the seven section titles in order.

- [ ] **Step 5: Commit**

```bash
git add docs
git commit -m "Docs: section index pages and the eight-entry navigation"
```

### Task 12: Landing page

**Files:**
- Modify: `docs/index.md` (replace everything above the toctree)

**Interfaces:**
- Consumes: CSS classes from Task 2; images already in `docs/images/`.

- [ ] **Step 1: Write the landing content**

Replace everything in `docs/index.md` above the `{toctree}` block with:

````markdown
---
html_theme.sidebar_secondary.remove: true
---

# superglm

```{raw} html
<div class="sg-hero">
  <div class="sg-hero__logo"><img src="_static/logo.png" alt="superglm"></div>
  <div class="sg-hero__copy">
    <div class="sg-hero__title">Super GLM</div>
    <p>Penalised GLMs and GAM pricing models for insurance, with the smoothness chosen by REML and the constraints you would otherwise enforce by hand.</p>
    <a class="sg-btn" href="get-started/index.html">Get started</a>
    <a class="sg-btn sg-btn--ghost" href="tutorials/index.html">Tutorials</a>
    <span class="sg-hero__meta">MIT licensed · free for everyone · built on NumPy, SciPy and pandas</span>
  </div>
</div>
```

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Get started
:link: get-started/index
:link-type: doc
:img-bottom: images/readme_vehage.png

Install, then fit and validate a frequency model on French motor data.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
:img-bottom: images/readme_bonusmalus.png

Frequency, severity, pure premium, constraints and rating tables, executed on every build.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc
:img-bottom: images/readme_drivage_bands.png

One goal per page: weights and offsets, credibility, constraints, screening, deployment.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc
:img-bottom: images/readme_mtpl2_relativities.png

Why REML, why penalties, why the constraints are stricter than the literature.
:::

::::

## Twelve lines to a fitted model

```python
from superglm import Categorical, Numeric, Spline, SuperGLM

features = {
    "DrivAge": Spline(kind="ps", k=14, knot_strategy="quantile_rows"),
    "VehAge": Spline(kind="cr", k=10, knot_strategy="quantile_rows"),
    "BonusMalus": Spline(kind="cr", k=12, knot_strategy="quantile_tempered"),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}
model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
model.fit_reml(df, y, sample_weight=exposure)
print(model.summary())
```

The [API reference](api/index.md) documents every public name. The
[governance section](governance/index.md) is for model-risk reviewers.
````

The four card images are real fitted curves from the README; the first pricing tutorial regenerates them in the next pull request.

- [ ] **Step 2: Build strictly and look once**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`
Then screenshot `docs/_build/html/index.html` at 1440×900 with the Task 5 script pattern and read it. Fix only what is visibly broken (a card image missing, the hero overflowing). Do not iterate on taste; that is Max's review.

- [ ] **Step 3: Commit**

```bash
git add docs/index.md
git commit -m "Docs: landing page with the hero band and four route cards"
```

### Task 13: Redirects for every live URL

**Files:**
- Create: `tests/docs/fixtures/legacy_urls.txt`
- Create: `tests/docs/test_redirects.py`
- Modify: `docs/redirects.txt`

**Interfaces:**
- Consumes: page paths from Tasks 8, 10 and 11.
- Produces: `redirect_map()` used only inside the test.

- [ ] **Step 1: Write the legacy URL fixture**

`tests/docs/fixtures/legacy_urls.txt` (the 51 non-root pages live on the deployed site on 2026-09-13, from the `gh-pages` tree):

```
api/diagnostics/
api/families/
api/features/
api/inference/
api/model/
api/model_selection/
api/penalties/
api/plotting/
api/validation/
development/cost-and-timing/
development/data-and-solver-boundaries/
development/migrations/group-pricing-rank/
development/migrations/weight-semantics-prior/
development/python-support/
development/releases/
distributional-family-development/
editor_frontend/
getting-started/installation/
getting-started/quickstart/
governance/model_risk_pack/
guide/credibility/
guide/deployment/
guide/editor/
guide/families/
guide/features/
guide/fitting/
guide/interactions/
guide/monotone/
guide/optimization/
guide/results/
guide/scop-performance-prototype/
guide/screening/
guide/screening-evaluation/
guide/validation/
guide/workflows/
models/distributional/
models/distributional-inference/
notebooks/editor_demo/
notebooks/mtpl2_frequency_walkthrough/
notebooks/ordered_categorical_smoothing/
notebooks/plotting_diagnostics_demo/
notebooks/tweedie_profile_estimation/
research/2026-09-c3-c1-completion-evidence/
research/2026-09-c3-stress-evidence/
research/2026-09-discrete-performance-plan/
research/2026-09-discrete-performance-report/
research/2026-09-pragmatic-convergence/
research/2026-09-superglm-feature-roadmap-additions/
research/2026-09-superglm-feature-roadmap-dossier/
ROADMAP/
tabmat-integration-notes/
```

- [ ] **Step 2: Write the failing test**

`tests/docs/test_redirects.py`:

```python
"""Every URL that was live on the MkDocs site redirects to an existing page.

The MkDocs site served directory URLs (``guide/monotone/``). The redirect
map lists the source document rediraffe writes for each one
(``guide/monotone/index.md``) and the page it forwards to.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "docs" / "fixtures" / "legacy_urls.txt"
REDIRECTS = ROOT / "docs" / "redirects.txt"
DOCS = ROOT / "docs"


def redirect_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in REDIRECTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        old, new = line.split()
        mapping[old] = new
    return mapping


def legacy_urls() -> list[str]:
    return [line.strip() for line in LEGACY.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_legacy_url_has_a_redirect() -> None:
    mapping = redirect_map()
    missing = [url for url in legacy_urls() if f"{url}index.md" not in mapping]
    assert missing == [], f"live URLs with no redirect: {missing}"


def test_every_redirect_target_exists() -> None:
    dangling = [new for new in redirect_map().values() if not (DOCS / new).exists()]
    assert dangling == [], f"redirect targets that are not pages: {dangling}"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/docs/test_redirects.py -q`
Expected: FAIL, 51 live URLs with no redirect.

- [ ] **Step 4: Write the redirect map**

`docs/redirects.txt`:

```
# legacy source document                              page it forwards to
api/diagnostics/index.md                              api/validation-and-diagnostics.md
api/families/index.md                                 api/families-and-links.md
api/features/index.md                                 api/features.md
api/inference/index.md                                api/inference.md
api/model/index.md                                    api/model.md
api/model_selection/index.md                          api/validation-and-diagnostics.md
api/penalties/index.md                                api/penalties.md
api/plotting/index.md                                 api/plotting.md
api/validation/index.md                               api/validation-and-diagnostics.md
development/cost-and-timing/index.md                  governance/reproducibility.md
development/data-and-solver-boundaries/index.md       development/internals/data-and-solver-boundaries.md
development/migrations/group-pricing-rank/index.md    development/migrations/group-pricing-rank.md
development/migrations/weight-semantics-prior/index.md development/migrations/weight-semantics-prior.md
development/python-support/index.md                   governance/python-support.md
development/releases/index.md                         development/releases.md
distributional-family-development/index.md            development/internals/distributional-family-development.md
editor_frontend/index.md                              development/internals/editor-frontend.md
getting-started/installation/index.md                 get-started/installation.md
getting-started/quickstart/index.md                   get-started/quickstart.md
governance/model_risk_pack/index.md                   governance/model-risk-pack.md
guide/credibility/index.md                            explanation/credibility-as-smoothing.md
guide/deployment/index.md                             how-to/deploy-a-fitted-model.md
guide/editor/index.md                                 tutorials/edit-a-model-in-the-browser.md
guide/families/index.md                               explanation/families-and-weights.md
guide/features/index.md                               how-to/specify-features.md
guide/fitting/index.md                                how-to/choose-a-fitting-path.md
guide/interactions/index.md                           how-to/specify-interactions.md
guide/monotone/index.md                               how-to/constrain-a-smooth.md
guide/optimization/index.md                           explanation/solvers-and-internals.md
guide/results/index.md                                how-to/read-a-summary-and-plot-effects.md
guide/scop-performance-prototype/index.md             development/index.md
guide/screening/index.md                              how-to/screen-interactions.md
guide/screening-evaluation/index.md                   explanation/what-screening-does.md
guide/validation/index.md                             how-to/compare-models-on-holdout.md
guide/workflows/index.md                              how-to/recommended-workflows.md
models/distributional/index.md                        how-to/fit-a-distributional-model.md
models/distributional-inference/index.md              how-to/check-a-distributional-fit.md
notebooks/editor_demo/index.md                        examples/editor_demo.ipynb
notebooks/mtpl2_frequency_walkthrough/index.md        examples/mtpl2_frequency_walkthrough.ipynb
notebooks/ordered_categorical_smoothing/index.md      examples/ordered_categorical_smoothing.ipynb
notebooks/plotting_diagnostics_demo/index.md          examples/plotting_diagnostics_demo.ipynb
notebooks/tweedie_profile_estimation/index.md         examples/tweedie_profile_estimation.ipynb
research/2026-09-c3-c1-completion-evidence/index.md   development/index.md
research/2026-09-c3-stress-evidence/index.md          development/index.md
research/2026-09-discrete-performance-plan/index.md   development/index.md
research/2026-09-discrete-performance-report/index.md development/index.md
research/2026-09-pragmatic-convergence/index.md       development/index.md
research/2026-09-superglm-feature-roadmap-additions/index.md development/index.md
research/2026-09-superglm-feature-roadmap-dossier/index.md development/index.md
ROADMAP/index.md                                      development/index.md
tabmat-integration-notes/index.md                     development/internals/tabmat-integration-notes.md
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/docs/test_redirects.py -q`
Expected: PASS, two tests.

- [ ] **Step 6: Confirm rediraffe writes the stubs**

Run: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html && ls docs/_build/html/guide/monotone/index.html && grep -o 'url=[^"]*' docs/_build/html/guide/monotone/index.html`
Expected: the file exists and its refresh target is `../../how-to/constrain-a-smooth.html`. If rediraffe rejects the source spelling, read its README at <https://github.com/wpilibsuite/sphinxext-rediraffe> for the accepted form (it takes source paths relative to the docs directory, with or without extension) and adjust both the file and the test's `f"{url}index.md"` key in one place.

- [ ] **Step 7: Commit**

```bash
git add docs/redirects.txt tests/docs/fixtures/legacy_urls.txt tests/docs/test_redirects.py
git commit -m "Docs: redirect every URL of the MkDocs site to its successor"
```

### Task 14: Notebook execution test

**Files:**
- Create: `tests/docs/test_notebooks.py`

**Interfaces:**
- Consumes: the MyST notebook front matter shape from Task 3.

- [ ] **Step 1: Write the test**

```python
"""Execute every MyST Markdown notebook under ``docs/`` with a fresh kernel.

This is the authoritative check on executable documentation. The Sphinx
build caches renders by file content, so a library change that breaks a
tutorial without touching its text would otherwise go unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parents[2] / "docs"
SKIP_DIRS = {"superpowers", "_build", "api"}


def myst_notebooks() -> list[Path]:
    found: list[Path] = []
    for path in sorted(DOCS.rglob("*.md")):
        if SKIP_DIRS & set(path.relative_to(DOCS).parts):
            continue
        head = path.read_text(encoding="utf-8")[:600]
        if head.startswith("---") and "format_name: myst" in head:
            found.append(path)
    return found


@pytest.mark.docs
@pytest.mark.parametrize("path", myst_notebooks(), ids=lambda p: str(p.relative_to(DOCS)))
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = jupytext.read(path)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        skip_cells_with_tag="skip-execution",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    executed = [c for c in notebook.cells if c.cell_type == "code" and c.get("execution_count")]
    assert executed, f"{path.name} has no executed code cells"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/docs/test_notebooks.py -m docs -q`
Expected: PASS for `tutorials/distributional-model.md`, in well under a minute.

- [ ] **Step 3: Confirm it is excluded by the marker**

Run: `uv run pytest tests/docs -m "not docs" -q`
Expected: the redirect and API tests run; the notebook test is deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/docs/test_notebooks.py
git commit -m "Docs tests: execute every MyST notebook with a fresh kernel"
```

### Task 15: Conventions test

**Files:**
- Create: `tests/docs/test_conventions.py`

- [ ] **Step 1: Write the test**

```python
"""Conventions for executed documentation pages.

Executed pages must use the public API only. The audit of the old site
found four private-attribute reads in notebooks; this keeps them out.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
CODE_CELL = re.compile(r"```\{code-cell\}[^\n]*\n(.*?)```", re.S)
PRIVATE_ACCESS = re.compile(r"[\w\)\]]\._(?!_)[A-Za-z]\w*")


def executed_pages() -> list[Path]:
    return [p for p in sorted(DOCS.rglob("*.md")) if "superpowers" not in p.parts and "_build" not in p.parts]


def test_no_private_attribute_access_in_executed_pages() -> None:
    offenders: list[str] = []
    for path in executed_pages():
        for cell in CODE_CELL.findall(path.read_text(encoding="utf-8")):
            for match in PRIVATE_ACCESS.finditer(cell):
                offenders.append(f"{path.relative_to(DOCS)}: {match.group(0)}")
    assert offenders == [], "private attribute access in executed pages:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/docs/test_conventions.py -q`
Expected: PASS (the distributional tutorial uses only public calls).

- [ ] **Step 3: Commit**

```bash
git add tests/docs/test_conventions.py
git commit -m "Docs tests: executed pages use the public API only"
```

### Task 16: README as a front door

**Files:**
- Modify: `README.md` (replace everything after the badges)

- [ ] **Step 1: Replace the body**

Keep the logo image, the three badges and the first paragraph. Replace everything from `## Installation` to the end with:

````markdown
## Install

```bash
pip install superglm
```

Interactive Plotly charts are optional: `pip install "superglm[plotting]"`.
The browser model editor is included.

## Fit a pricing model

```python
from superglm import Categorical, Numeric, Spline, SuperGLM

features = {
    "DrivAge": Spline(kind="ps", k=14, knot_strategy="quantile_rows"),
    "VehAge": Spline(kind="cr", k=10, knot_strategy="quantile_rows"),
    "BonusMalus": Spline(kind="cr", k=12, knot_strategy="quantile_tempered"),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}
model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
model.fit_reml(train_df, y_train, sample_weight=exposure_train)
print(model.summary())
```

REML chooses the smoothness of every spline. Monotone and curvature
constraints are enforced inside the fit. `SuperLSS` fits location, scale and
shape parameters together for severity and distributional work.

## Documentation

- [Get started](https://strudeldoodles.github.io/superglm/get-started/index.html)
- [Tutorials](https://strudeldoodles.github.io/superglm/tutorials/index.html)
- [How-to guides](https://strudeldoodles.github.io/superglm/how-to/index.html)
- [Explanation](https://strudeldoodles.github.io/superglm/explanation/index.html)
- [API reference](https://strudeldoodles.github.io/superglm/api/index.html)
- [Governance](https://strudeldoodles.github.io/superglm/governance/index.html)

## Licence

MIT. Free for everyone, commercial use included.
````

- [ ] **Step 2: Check the length and the links**

Run: `wc -w README.md && grep -c '](docs/' README.md`
Expected: under 400 words; `0` relative `docs/` links.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: a front door with absolute links to the site"
```

### Task 17: Pull-request CI

**Files:**
- Modify: `.github/workflows/dev-ci.yml` (the `docs` job, lines 117-133; the `pytest-314` matrix command, line 168; the `pytest-312` command, line 194)
- Modify: `.github/workflows/ci.yml` (lines 66 and 75)
- Modify: `.pre-commit-config.yaml` (the pre-push pytest entry)

- [ ] **Step 1: Replace the docs job's last two steps**

In `dev-ci.yml`, the `docs` job's `Install documentation dependencies` and `Docs build check` steps become:

```yaml
      - name: Install documentation dependencies
        run: uv sync --python 3.14 --group docs --extra plotting

      - name: Docs build check (strict, no execution)
        env:
          SUPERGLM_DOCS_EXECUTE: "off"
        run: uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

- [ ] **Step 2: Add the notebooks job after the docs job**

```yaml
  docs-notebooks:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1

      - name: Install uv
        uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9

      - name: Set up Python
        run: uv python install 3.14

      - name: Install dependencies
        run: uv sync --python 3.14 --extra dev --extra plotting --group docs

      - name: Cache OpenML downloads
        uses: actions/cache@ACTIONS_CACHE_SHA
        with:
          path: ~/scikit_learn_data
          key: openml-mtpl2-v1

      - name: Execute documentation notebooks
        run: uv run pytest tests/docs -m docs -q
```

Resolve `ACTIONS_CACHE_SHA` with `gh api repos/actions/cache/commits/v4 --jq .sha` and paste the 40-character value.

- [ ] **Step 3: Exclude docs tests from the regular matrices**

In `dev-ci.yml` change both `-m "not browser"` occurrences to `-m "not browser and not docs"`. In `ci.yml` change lines 66 and 75 the same way. In `.pre-commit-config.yaml` change `-m "not slow"` to `-m "not slow and not docs"`.

- [ ] **Step 4: Validate the YAML**

Run: `uv run python -c "import yaml, pathlib; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ('.github/workflows/dev-ci.yml', '.github/workflows/ci.yml', '.pre-commit-config.yaml')]; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/dev-ci.yml .github/workflows/ci.yml .pre-commit-config.yaml
git commit -m "CI: strict Sphinx build and a notebook execution job on pull requests"
```

### Task 18: Deploy workflow on GitHub Pages actions

**Files:**
- Modify: `.github/workflows/docs.yml` (replace the whole file)

- [ ] **Step 1: Resolve the action SHAs**

```bash
for a in actions/cache actions/configure-pages actions/upload-pages-artifact actions/deploy-pages; do
  printf '%s ' "$a"; gh api "repos/$a/releases/latest" --jq .tag_name; done
```

Then for each `tag_name`, `gh api repos/<action>/commits/<tag> --jq .sha`. Paste the SHAs below in place of the placeholders.

- [ ] **Step 2: Write the workflow**

```yaml
name: Docs

on:
  push:
    branches: [master]
    paths:
      - "docs/**"
      - "src/**"
      - "README.md"
      - "pyproject.toml"
      - "uv.lock"
      - ".github/workflows/docs.yml"
  workflow_dispatch:

concurrency:
  group: docs-deploy
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1

      - name: Install uv
        uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9

      - run: uv python install 3.12

      - name: Install docs dependencies
        run: uv sync --python 3.12 --group docs --extra plotting

      - name: Cache OpenML downloads
        uses: actions/cache@ACTIONS_CACHE_SHA
        with:
          path: ~/scikit_learn_data
          key: openml-mtpl2-v1

      - name: Cache executed notebooks
        uses: actions/cache@ACTIONS_CACHE_SHA
        with:
          path: docs/_build/.jupyter_cache
          key: docs-notebooks-${{ hashFiles('docs/tutorials/**', 'docs/how-to/**', 'uv.lock') }}
          restore-keys: |
            docs-notebooks-

      - name: Build with execution
        env:
          SUPERGLM_DOCS_EXECUTE: cache
        run: uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html

      - uses: actions/configure-pages@CONFIGURE_PAGES_SHA

      - uses: actions/upload-pages-artifact@UPLOAD_PAGES_SHA
        with:
          path: docs/_build/html

  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@DEPLOY_PAGES_SHA
```

- [ ] **Step 3: Validate and commit**

Run: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/docs.yml').read_text()); print('ok')"`

```bash
git add .github/workflows/docs.yml
git commit -m "CI: deploy the docs with the GitHub Pages actions"
```

Record in the PR description: before the first deploy, the repository setting Pages → Source must be switched to "GitHub Actions" (Max). The `gh-pages` branch is deleted after the first successful Actions deploy.

### Task 19: Pre-commit hooks for docstrings and notebook pairing

**Files:**
- Modify: `.pre-commit-config.yaml` (append two repos)
- Modify: `pyproject.toml` (append a `[tool.numpydoc_validation]` table)
- Modify: `src/superglm/features/constraint.py` (one missing class docstring on `Constraint`)

- [ ] **Step 1: Configure numpydoc validation**

Append to `pyproject.toml`:

```toml
# ── numpydoc docstring validation (pre-commit) ──────────────────
[tool.numpydoc_validation]
checks = ["GL08"]
exclude = [
    '\._',          # private modules and members
    '\.__',         # dunders
    '^superglm\.editor\.',
]
```

- [ ] **Step 2: Append the hooks**

Resolve tags first: `gh api repos/numpy/numpydoc/releases/latest --jq .tag_name`, `gh api repos/mwouts/jupytext/releases/latest --jq .tag_name`, `gh api repos/kynan/nbstripout/releases/latest --jq .tag_name`. Then append to `.pre-commit-config.yaml`:

```yaml
  # ── Docstrings and documentation notebooks ──────────────────
  - repo: https://github.com/numpy/numpydoc
    rev: NUMPYDOC_TAG
    hooks:
      - id: numpydoc-validation
        files: ^src/superglm/

  - repo: https://github.com/mwouts/jupytext
    rev: JUPYTEXT_TAG
    hooks:
      - id: jupytext
        args: [--sync]
        files: ^docs/tutorials/.*\.md$

  - repo: https://github.com/kynan/nbstripout
    rev: NBSTRIPOUT_TAG
    hooks:
      - id: nbstripout
        files: ^docs/tutorials/.*\.ipynb$
```

- [ ] **Step 3: Run the docstring hook and fix what it reports**

Run: `uv run pre-commit run numpydoc-validation --all-files`
Expected: reports `Constraint` (no docstring). Add a numpy-style class docstring to `Constraint` in `src/superglm/features/constraint.py` describing the `fit` and `postfit` namespaces and their `increasing`, `decreasing`, `convex`, `concave` members. Re-run until clean. Any other report is either a public object that needs a one-paragraph docstring (write it) or a name that should be excluded (add a regex to `exclude`, with a comment saying why).

- [ ] **Step 4: Run the notebook hooks**

Run: `uv run pre-commit run jupytext --all-files && uv run pre-commit run nbstripout --all-files`
Expected: both pass with no file changes (the pair is already in sync and output-free).

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml src/superglm/features/constraint.py
git commit -m "Pre-commit: numpydoc validation and tutorial notebook pairing"
```

### Task 20: Final verification and the pull request

**Files:** none new.

- [ ] **Step 1: Clean build with execution**

Run: `rm -rf docs/_build && SUPERGLM_DOCS_EXECUTE=force uv run sphinx-build -b html -n -W --keep-going docs docs/_build/html`
Expected: `build succeeded`, zero warnings, in under five minutes.

- [ ] **Step 2: All docs tests**

Run: `uv run pytest tests/docs -q`
Expected: PASS for API coverage, both redirect tests, the conventions test and the notebook execution.

- [ ] **Step 3: Smoke test and the three touched test modules**

Run: `uv run python run_test.py | tail -1 && uv run pytest tests/test_reml_tol_determination.py tests/test_nb_theta_estimation_correctness.py tests/test_tweedie_reml_exact_scale.py -q -m "not slow" | tail -1`
Expected: `END-TO-END COMPLETE`; the three modules pass or skip.

- [ ] **Step 4: Lint**

Run: `uv run ruff check tests/docs src/superglm/features/constraint.py && uv run ruff format --check tests/docs`
Expected: clean.

- [ ] **Step 5: Screenshots for review**

Screenshot the landing page, `tutorials/distributional-model.html`, `api/generated/superglm.SuperGLM.html` and `how-to/constrain-a-smooth.html` in light and dark with the Task 5 script pattern, and send them to Max with SendUserFile or as an artifact.

- [ ] **Step 6: Push and open the pull request**

```bash
git push -u origin worktree-docs-rebuild-sphinx
```

Open the PR with `gh pr create` against `master`. Title: `Rebuild the documentation on Sphinx: structure, API reference, execution, deploy`. Body sections: what changed (the spec's summary), the spike measurements (T5 execution time, numpydoc warning count settled, theme chosen), the one manual step (Pages source → GitHub Actions), and the follow-up PRs from spec §12. End the body with the attribution lines from the session's system reminder. Both review bots run on the PR; read their summary comments as well as their review threads, and resolve threads only after fixing.

---

## Self-review against the spec

- §5 toolchain: Task 1. §6 architecture: Tasks 10, 11, 12. §7 executable docs: Tasks 3, 14, 15, 19. §8 API reference: Tasks 4, 8, 19. §9 look and feel: Tasks 2, 5, 12. §10 build, CI, deploy, redirects: Tasks 13, 17, 18. §11 migration mapping: Tasks 7, 10, 13. §12 sequence (spike and PR-1): all. §14 hazards: `.test_durations` is unaffected because docs tests are deselected from the split matrices; numpydoc warnings are burned down in Tasks 4 and 8; `skip-execution` is proven in Task 3; the theme is decided in Task 5.
- Not in this plan by design: tutorials T0–T4 and T6, the how-to splits, explanation carving, migrations stamping, the docstring remediation beyond `Constraint`. Each is a later PR per spec §12.
- Names used across tasks: `SUPERGLM_DOCS_EXECUTE` and `SUPERGLM_DOCS_THEME` (Tasks 2, 5, 17, 18); CSS classes `sg-hero`, `sg-hero__logo`, `sg-hero__copy`, `sg-hero__title`, `sg-hero__meta`, `sg-btn`, `sg-btn--ghost` (Tasks 2, 12); the `docs` marker (Tasks 6, 14, 17); `redirects.txt` and the `<url>index.md` key convention (Tasks 2, 13); page paths (Tasks 8, 10, 11, 13, 16) checked against each other line by line.
