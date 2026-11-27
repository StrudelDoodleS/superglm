"""What the editor's Assess tab shows, read from the page, for each kind of book.

Run ``uv run python -m tests.editor.assess_table`` to print one Markdown row per
book: the top bar, the Assess list, the overview cards, each report's heading,
and where Open in new window is offered. It reads whatever page the head
serves, so on a page without Assess the Assess columns read "—".

The session builders and the ``open_report`` / ``wait_for_report`` /
``assert_report_open`` helpers are shared with the browser suites.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from sklearn.model_selection import KFold

from superglm import Categorical, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession

MISSING = "—"
REPORTS = ("validation", "cv", "final", "diagnose")
COLUMNS = ("top bar", "list", "cards", "reports", "new window")
_STATUS_CHARS = 60
# The evidence states a report settles in; "idle", "loading" and "updating" are on the way.
_SETTLED = ", ".join(
    f'#reportFrame[aria-busy="false"][data-freshness="{status}"]'
    for status in ("current", "stale", "error")
)
_N_TRAIN = 800


@functools.cache
def _poisson_rows() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """1,000 rows of ``age`` and a four-level ``region``, with a count and a 0/1 response."""
    rng = np.random.default_rng(20261127)
    n = 1000
    X = pd.DataFrame(
        {"age": rng.uniform(18.0, 80.0, n), "region": rng.choice(["A", "B", "C", "D"], n)}
    )
    eta = (
        -0.5
        + 0.3 * np.sin(X["age"].to_numpy() / 12.0)
        + 0.25 * (X["region"] == "B").to_numpy()
        - 0.2 * (X["region"] == "D").to_numpy()
    )
    counts = rng.poisson(np.exp(eta)).astype(np.float64)
    claims = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    return X, counts, claims


def _glm(family: str) -> SuperGLM:
    return SuperGLM(
        family=family,
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "region": Categorical(base="first")},
    )


def _glm_cv_session(family: str) -> EditorSession:
    X, counts, claims = _poisson_rows()
    y = counts if family == "poisson" else claims
    X_train, y_train = X.iloc[:_N_TRAIN], y[:_N_TRAIN]
    cv = cross_validate(
        _glm(family),
        X_train,
        y_train,
        cv=KFold(3, shuffle=True, random_state=0),
        fit_mode="fit_reml",
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    return EditorSession.from_model(
        _glm(family).fit_reml(X_train, y_train),
        train_data=(X_train, y_train),
        validation_data=(X.iloc[_N_TRAIN:], y[_N_TRAIN:]),
        cv=cv,
    )


def poisson_cv_session() -> EditorSession:
    """A Poisson GLM on 800 rows, its last 200 as validation, and a 3-fold ``cv=``."""
    return _glm_cv_session("poisson")


def binomial_cv_session() -> EditorSession:
    """The same rows and split as ``poisson_cv_session``, fitted as a binomial."""
    return _glm_cv_session("binomial")


def pareto_lss_session() -> EditorSession:
    """A generalized Pareto SuperLSS on its 400 rows, its last 100 as validation, a 3-fold ``cv=``.

    The editor needs the rows the shared fixture's model was fitted on, which are all
    400, so the validation rows are among them: they show the page, not a holdout.
    """
    from tests._lss_editor_fixtures import scalar_family_case

    case = scalar_family_case("GeneralizedParetoLSS")
    X, y = case.X, np.asarray(case.y)
    cv = cross_validate(
        case.model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=0),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    return EditorSession.from_model(
        case.model, train_data=(X, y), validation_data=(X.iloc[-100:], y[-100:]), cv=cv
    )


def bare_glm_session() -> EditorSession:
    """The Poisson model opened with no data and no ``cv=``."""
    X, counts, _claims = _poisson_rows()
    return EditorSession.from_model(_glm("poisson").fit_reml(X.iloc[:_N_TRAIN], counts[:_N_TRAIN]))


def bundle_session(path: str | Path) -> EditorSession:
    """A saved demo bundle ``{model, train, validation, cv}``, opened as the demo server does."""
    import joblib

    saved = joblib.load(path)
    return EditorSession.from_model(
        saved["model"],
        train_data=saved.get("train"),
        validation_data=saved.get("validation"),
        cv=saved.get("cv"),
    )


def _present(root, selector: str) -> bool:
    return root.locator(selector).count() > 0


def open_report(page, name: str) -> None:
    """Open report ``name`` as an analyst would: from the Assess list, else from its tab."""
    if _present(page, "#assessTab"):
        if page.locator("#assessView").is_hidden():
            page.locator("#assessTab").click()
        page.locator(f'#assessList [data-assess-entry="{name}"]').click()
    else:
        page.locator(f'.app-tab[data-view="{name}"]').click()


def wait_for_report(page, name: str) -> None:
    """Wait until report ``name`` is shown and its evidence has settled."""
    page.locator(f'#reportPanel[data-report="{name}"]').wait_for(state="visible")
    page.locator(_SETTLED).wait_for(state="attached")


def assert_report_open(page, name: str) -> None:
    """Assert the Assess tab is selected with ``name`` as its current entry."""
    assert page.locator('#assessTab[aria-selected="true"]').count() == 1
    assert page.locator(f'[data-assess-entry="{name}"][aria-current="true"]').count() == 1


# The text a reader sees: text nodes whose element and ancestors up to ``root`` render a
# box larger than a pixel, unclipped and not hidden. ``innerText`` cannot say this: it keeps
# text that is clipped away (a screen-reader label), and under a hidden ancestor it returns
# text CSS would suppress.
_SHOWN_TEXT = """(root) => {
  const shown = (element) => {
    for (let node = element; node; node = node === root ? null : node.parentElement) {
      const style = getComputedStyle(node);
      const box = node.getBoundingClientRect();
      if (style.display === "none" || style.visibility === "hidden") return false;
      if (box.width <= 1 || box.height <= 1) return false;
      if ((style.clip && style.clip !== "auto") || style.clipPath !== "none") return false;
    }
    return true;
  };
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const parts = [];
  for (let text = walker.nextNode(); text; text = walker.nextNode()) {
    if (text.data.trim() && shown(text.parentElement)) parts.push(text.data);
  }
  return parts.join(" ");
}"""


def _text(locator) -> str:
    """The text ``locator``'s first match shows, or "—" when there is no such element."""
    if locator.count() == 0:
        return MISSING
    return " ".join(locator.first.evaluate(_SHOWN_TEXT).split())


def _label(locator) -> str:
    """The visible name, or the accessible one marked "(icon)" where only an icon shows."""
    shown = _text(locator)
    if shown not in ("", MISSING):
        return shown
    return f"{locator.get_attribute('aria-label') or '?'} (icon)"


def _read_list(root) -> str:
    if not _present(root, "#assessList"):
        return MISSING
    # "Overview; **Reports**: Validation, Cross-validation": a shown heading starts a group.
    # Entries and headings the page does not show (display, visibility, ``hidden``, a hidden
    # ancestor: no box) are skipped, and so is a heading that shows no text.
    groups: list[tuple[str, list[str]]] = [("", [])]
    for item in root.locator(
        "#assessList .assess-group-title, #assessList [data-assess-entry]"
    ).all():
        if not item.is_visible():
            continue
        if item.get_attribute("data-assess-entry") is not None:
            groups[-1][1].append(_label(item))
        elif heading := _text(item):
            groups.append((f"**{heading}**: ", []))
    return "; ".join(
        (heading + ", ".join(entries)).strip() for heading, entries in groups if heading or entries
    )


_CARDS_TIMEOUT_MS = 5000


def _show_overview(root) -> None:
    if root.locator("#assessView").is_hidden():
        root.locator("#assessTab").click()
    if _present(root, '[data-assess-entry="assess"]'):
        root.locator('[data-assess-entry="assess"]').click()
    # Shown, not "visible": an overview with nothing in it yet has no box.
    root.locator("#assessView:not([hidden]) #assessOverview:not([hidden])").wait_for(
        state="attached"
    )


def _read_cards(root) -> str:
    if not _present(root, "#assessCards"):
        return MISSING
    cards = root.locator("#assessCards [data-card]")
    try:
        cards.first.wait_for(state="visible", timeout=_CARDS_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return MISSING
    # Cards carry their evidence state once they have one; wait out the loading ones.
    root.locator(
        ", ".join(
            f'#assessCards [data-card][data-freshness="{status}"]'
            for status in ("idle", "loading", "updating")
        )
    ).first.wait_for(state="detached")
    read = []
    for card in cards.all():
        title = _text(card.locator(".assess-card-title"))
        value = _text(card.locator(".assess-card-value"))
        note = _text(card.locator(".assess-card-note"))
        read.append(f"**{title}**: {value} / {note}")
    return "; ".join(read)


def _window_button_shown(root) -> bool:
    return root.locator("[data-assess-window]:visible").count() > 0


def read_page(page_or_frame) -> dict[str, str]:
    """What the page shows, column by column, read from its DOM alone.

    Works on a Playwright page or a frame locator (the notebook cell). It opens
    each report through ``open_report``; a missing element reads "—".
    """
    root = page_or_frame
    tabs = root.locator(".app-tabs .app-tab:visible").all()
    top_bar = " · ".join(_label(tab) for tab in tabs) or MISSING
    if _present(root, "#assessTab"):
        # The list and cards are read as shown, so open the overview first.
        _show_overview(root)
    listed = _read_list(root)
    cards = _read_cards(root)
    has_window_button = _present(root, "[data-assess-window]")
    window_shown = []
    if has_window_button and _present(root, "#assessOverview"):
        window_shown.append(("overview", _window_button_shown(root)))
    reports = []
    for name in REPORTS:
        open_report(root, name)
        wait_for_report(root, name)
        title = _text(root.locator("#reportTitle"))
        status = _text(root.locator("#reportStatus"))[:_STATUS_CHARS]
        reports.append(f"{title}: {status}" if status else title)
        if has_window_button:
            window_shown.append((name, _window_button_shown(root)))
    return {
        "top bar": top_bar,
        "list": listed,
        "cards": cards,
        "reports": " | ".join(reports),
        "new window": _window_column(has_window_button, window_shown),
    }


def _window_column(has_button: bool, shown: list[tuple[str, bool]]) -> str:
    if not has_button:
        return MISSING
    where = [name for name, visible in shown if visible]
    if not where:
        return "Not offered"
    if len(where) == len(shown):
        return "Shown on " + ", ".join(where)
    return (
        "Shown on "
        + ", ".join(where)
        + "; not on "
        + ", ".join(name for name, visible in shown if not visible)
    )


def _read_served(browser, session: EditorSession, viewport: dict[str, int]) -> dict[str, str]:
    with ExitStack() as resources:
        widget = session.widget()
        resources.callback(widget.close)
        page = browser.new_page(viewport=viewport)
        resources.callback(page.close)
        page.goto(f"{widget.app_url}&test=1", wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        return read_page(page)


def _read_notebook(browser) -> dict[str, str] | str:
    if importlib.util.find_spec("anywidget") is None:
        return "anywidget is not installed, so notebook mode was not read"
    import pytest
    from tests.editor.test_editor_notebook_browser import _notebook_editor

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        _notebook_editor(browser, poisson_cv_session(), monkeypatch) as (_page, frame, _w, _v),
    ):
        return read_page(frame)


_WIDE = {"width": 1278, "height": 860}


def _rows(bundles: list[str]):
    """(book, how to read it) for every row of the table, in order."""
    for builder, book in (
        (poisson_cv_session, "Poisson GLM, validation, 3-fold `cv=` (fit_reml)"),
        (binomial_cv_session, "Binomial GLM, validation, 3-fold `cv=` (fit_reml)"),
        (pareto_lss_session, "Generalized Pareto SuperLSS, validation, 3-fold `cv=`"),
        (bare_glm_session, "Poisson GLM opened with no data and no `cv=`"),
    ):
        yield book, functools.partial(_read_served, session=builder(), viewport=_WIDE)
    yield "Notebook mode, the Poisson book", _read_notebook
    for width in (1180, 1000):
        yield (
            f"Poisson book, window {width} × 720",
            functools.partial(
                _read_served,
                session=poisson_cv_session(),
                viewport={"width": width, "height": 720},
            ),
        )
    for path in bundles:
        yield (
            f"Bundle `{Path(path).name}`",
            functools.partial(_read_served, session=bundle_session(path), viewport=_WIDE),
        )


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _table(rows: list[tuple[str, dict[str, str] | str]]) -> str:
    header = ["Book", "Top bar", "Assess list", "Overview cards", "Each report", "New window"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for book, read in rows:
        if isinstance(read, str):
            cells = [read] + [MISSING] * (len(COLUMNS) - 1)
        else:
            cells = [read[column] for column in COLUMNS]
        lines.append("| " + " | ".join(_cell(cell) for cell in [book, *cells]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        metavar="PATH",
        help="a saved {model, train, validation, cv} joblib bundle; adds a row (repeatable)",
    )
    parser.add_argument("--out", metavar="PATH", help="also write the table here as Markdown")
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    rows = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for book, read in _rows(args.bundle):
                rows.append((book, read(browser)))
        finally:
            browser.close()
    table = _table(rows)
    print(table, end="")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
