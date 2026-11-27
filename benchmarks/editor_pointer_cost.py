"""Main-thread cost per pointer move in the browser editor, old source tree against new.

    uv run python benchmarks/editor_pointer_cost.py \\
        --tree old=.claude/worktrees/stage1-base --tree new=. \\
        [--check-bundle saved.joblib] --out FILE

Each tree is served from its own subprocesses, with ``PYTHONPATH=<tree>/src:<head>``
and this interpreter, so only the editor's source differs between the two. Every
tree serves the same two sessions from this head: ``poisson_cv_session()`` from
``tests.editor.assess_table``, and for the check plots a SuperLSS session, the
saved bundle ``--check-bundle`` (``bundle_session``) or else
``pareto_lss_session()``. One headless Chromium drives them.

A run opens a fresh page on one tree, sets the CPU slowdown with
``Emulation.setCPUThrottlingRate`` and measures five scenarios. The cost of a
scenario is the change in ``Performance.getMetrics`` ``TaskDuration``, the
renderer main thread's busy time, over its gestures:

- a point drag on the Editor chart with the Move tool, 120 pointer moves;
- hovering across the fold legend of the Cross-validation report, 120 moves;
- hovering across the Assess overview cards, 120 moves (a page with no overview
  reads "—");
- hovering across a check's large view on the SuperLSS session, 60 moves over the
  Q-Q plot, then 60 over Actual over expected (a page with no checks reads "—");
- switching Validation and Cross-validation 20 times after each has been visited
  once, as ms per switch, with the ``/report`` requests those switches send.

For each CPU rate, the runs go old, new, old again, ``--pairs`` times. The table
gives the median old and new figures and the old-against-old drift, the median
gap between the two old runs of a pair, against which an old-to-new difference
can be judged. The headless browser uses a software renderer and the metric
counts main-thread tasks only, so raster and GPU work are not measured.

This is a benchmark, not a test: nothing here asserts on time.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

HEAD = Path(__file__).resolve().parents[1]
if str(HEAD) not in sys.path:
    sys.path.insert(0, str(HEAD))

from tests.editor._chart_layout import wait_for_chart_to_fit  # noqa: E402
from tests.editor.assess_table import open_report, wait_for_report  # noqa: E402

MOVES = 120
SWITCHES = 20
BUDGET_MS = 16.7
VIEWPORT = {"width": 1278, "height": 860}
DRAG_TERM = "region"
DRAG_LEVEL = 2
MISSING = "—"
SCENARIOS = (
    ("drag", "Point drag, Editor chart (Move tool)", "ms/move"),
    ("legend", "Hover across the Cross-validation fold legend", "ms/move"),
    ("cards", "Hover across the Assess overview cards", "ms/move"),
    ("check", "Hover across a check plot (Q-Q, then A/E)", "ms/move"),
    ("switch", "Switch Validation ↔ Cross-validation", "ms/switch"),
)

CHECK_PLOTS = ("check_qq", "check_ae")
# Build the session named on the command line, print the URL and the superglm it imported,
# serve until stdin closes.
_SERVE = """
import sys
import superglm
from tests.editor import assess_table

builder, *args = sys.argv[1:]
widget = getattr(assess_table, builder)(*args).widget(mode="server")
print("URL", widget.app_url, superglm.__file__, flush=True)
sys.stdin.read()
widget.close()
"""


@contextmanager
def serve(tree: Path, session: list[str]) -> Iterator[str]:
    """Serve ``tree``'s editor on ``session``, a builder in ``assess_table`` and its
    arguments; yield the page URL."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tree / "src"), str(HEAD)]))
    process = subprocess.Popen(
        [sys.executable, "-c", _SERVE, *session],
        cwd=HEAD,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        line = process.stdout.readline().split()
        if len(line) != 3 or line[0] != "URL":
            raise RuntimeError(f"the server for {tree} did not start: {line!r}")
        _, url, imported = line
        expected = (tree / "src" / "superglm").resolve()
        if Path(imported).resolve().parent != expected:
            raise RuntimeError(f"the server for {tree} imported superglm from {imported}")
        yield url
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def head_sha(tree: Path) -> str:
    run = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "--short=9", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return run.stdout.strip() + (" (with uncommitted changes)" if dirty else "")


def task_seconds(cdp) -> float:
    metrics = cdp.send("Performance.getMetrics")["metrics"]
    return next(metric["value"] for metric in metrics if metric["name"] == "TaskDuration")


def settle(page) -> None:
    """Let the frames the gestures queued draw before the busy time is read."""
    page.evaluate(
        "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))"
    )


def sweep(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """``n`` positions along the path through ``points``, there and back twice."""
    segments = list(zip(points, points[1:], strict=False)) or [(points[0], points[0])]
    lengths = [math.dist(a, b) for a, b in segments]
    total = sum(lengths)
    out = []
    for step in range(n):
        phase = (2.0 * step / max(n - 1, 1)) % 1.0
        along = total * (1.0 - abs(2.0 * phase - 1.0))
        for index, ((a, b), length) in enumerate(zip(segments, lengths, strict=True)):
            if along <= length or index == len(segments) - 1:
                t = 0.0 if length == 0 else min(along / length, 1.0)
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                break
            along -= length
    return out


def centre(box: dict[str, float]) -> tuple[float, float]:
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def hover_cost(page, cdp, points: list[tuple[float, float]], moves: int = MOVES) -> float:
    """Main-thread ms per move over ``moves`` pointer moves along ``points``."""
    path = sweep(points, moves)
    page.mouse.move(*path[0])
    settle(page)
    before = task_seconds(cdp)
    for x, y in path:
        page.mouse.move(x, y)
    settle(page)
    return (task_seconds(cdp) - before) * 1000.0 / moves


def measure_legend(page, cdp) -> float:
    open_report(page, "cv")
    wait_for_report(page, "cv")
    keys = page.locator("#reportPanel [data-cv-chart]").first.locator("[data-cv-fold-key]")
    keys.first.wait_for(state="visible")
    keys.first.scroll_into_view_if_needed()
    boxes = [key.bounding_box() for key in keys.all()]
    first, last = boxes[0], boxes[-1]
    y = first["y"] + first["height"] / 2
    return hover_cost(page, cdp, [(first["x"] + 2, y), (last["x"] + last["width"] - 2, y)])


def measure_cards(page, cdp) -> float | None:
    if page.locator("#assessTab").count() == 0:
        return None
    if page.locator("#assessView").is_hidden():
        page.locator("#assessTab").click()
    page.locator('[data-assess-entry="assess"]').click()
    cards = page.locator("#assessCards [data-card]")
    cards.first.wait_for(state="visible")
    page.locator(
        ", ".join(
            f'#assessCards [data-card][data-freshness="{status}"]'
            for status in ("idle", "loading", "updating")
        )
    ).first.wait_for(state="detached")
    cards.first.scroll_into_view_if_needed()
    return hover_cost(page, cdp, [centre(card.bounding_box()) for card in cards.all()])


def open_check(page, name: str) -> None:
    """Open check ``name`` from the Assess list and wait until its plot is drawn."""
    page.locator(f'#assessList [data-assess-entry="{name}"]').click()
    page.locator(f'#checkPanel[data-report="{name}"]').wait_for(state="visible")
    page.locator('#checkFrame[aria-busy="false"][data-freshness="current"] svg').first.wait_for(
        state="visible"
    )
    settle(page)


def plot_diagonal(page) -> list[tuple[float, float]]:
    """Bottom left to top right of the part of the check's plot the frame shows, inset 5%."""
    frame = page.locator("#checkFrame").bounding_box()
    plot = page.locator("#checkFrame svg").first.bounding_box()
    left, top = max(frame["x"], plot["x"]), max(frame["y"], plot["y"])
    right = min(frame["x"] + frame["width"], plot["x"] + plot["width"])
    bottom = min(frame["y"] + frame["height"], plot["y"] + plot["height"])
    dx, dy = 0.05 * (right - left), 0.05 * (bottom - top)
    return [(left + dx, bottom - dy), (right - dx, top + dy)]


def measure_check(page, cdp) -> float | None:
    """ms per move over ``MOVES`` moves: half over the Q-Q plot, half over A/E."""
    if page.locator(f'[data-assess-entry="{CHECK_PLOTS[0]}"]').count() == 0:
        return None
    if page.locator("#assessView").is_hidden():
        page.locator("#assessTab").click()
    half = MOVES // len(CHECK_PLOTS)
    costs = []
    for name in CHECK_PLOTS:
        open_check(page, name)
        costs.append(hover_cost(page, cdp, plot_diagonal(page), moves=half))
        page.mouse.move(0, 0)
    return statistics.mean(costs)


def measure_switch(page, cdp) -> tuple[float, int]:
    """ms per switch, and the ``/report`` requests the switches sent."""
    for name in ("validation", "cv"):
        open_report(page, name)
        wait_for_report(page, name)
    sent = []

    def count(request) -> None:
        if request.method == "POST" and request.url.split("?", 1)[0].endswith("/report"):
            sent.append(request.url)

    page.on("request", count)
    settle(page)
    before = task_seconds(cdp)
    for step in range(SWITCHES):
        name = ("validation", "cv")[step % 2]
        open_report(page, name)
        wait_for_report(page, name)
    settle(page)
    cost = (task_seconds(cdp) - before) * 1000.0 / SWITCHES
    page.remove_listener("request", count)
    return cost, len(sent)


def measure_drag(page, cdp) -> tuple[float, int]:
    """ms per move of a held Move-tool drag, and the ``/drag`` requests it sent.

    The drag is cancelled, not released, so the session keeps its curve for the next run.
    """
    page.locator("#editorTab").click()
    page.locator("#chart path.edited").first.wait_for()
    page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name="Move", exact=True
    ).click()
    wait_for_chart_to_fit(page)
    point = page.locator(f'#chart circle.point[data-index="{DRAG_LEVEL}"]')
    x, y = centre(point.bounding_box())
    sent = []

    def count(request) -> None:
        if request.method == "POST" and request.url.split("?", 1)[0].endswith("/drag"):
            sent.append(request.url)

    page.on("request", count)
    page.mouse.move(x, y)
    page.mouse.down()
    cost = hover_cost(page, cdp, [(x, y - 40), (x, y + 40)])
    page.dispatch_event("#chart", "pointercancel")
    page.mouse.up()
    settle(page)
    page.remove_listener("request", count)
    return cost, len(sent)


def open_page(browser, url: str):
    page = browser.new_page(viewport=VIEWPORT)
    page.goto(f"{url}&test=1", wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    if page.locator("#status").get_attribute("data-term") != DRAG_TERM:
        page.locator(f'#featureList [data-term="{DRAG_TERM}"]').click()
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term", arg=DRAG_TERM
        )
        page.locator("#chart path.edited").first.wait_for()
    wait_for_chart_to_fit(page)
    return page


@contextmanager
def throttled(page, rate: float):
    """A CDP session on ``page`` with metrics on and the CPU slowed by ``rate``."""
    with ExitStack() as resources:
        resources.callback(page.close)
        cdp = page.context.new_cdp_session(page)
        resources.callback(cdp.detach)
        cdp.send("Performance.enable")
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": rate})
        yield cdp
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})


def open_check_page(browser, url: str):
    page = browser.new_page(viewport=VIEWPORT)
    page.goto(f"{url}&test=1", wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    return page


def run_once(browser, urls: dict[str, str], rate: float) -> dict[str, float | int | None]:
    """Fresh pages on one tree's ``urls`` at CPU slowdown ``rate``: every scenario's cost."""
    page = open_page(browser, urls["glm"])
    with throttled(page, rate) as cdp:
        legend = measure_legend(page, cdp)
        cards = measure_cards(page, cdp)
        switch, reports = measure_switch(page, cdp)
        drag, drags = measure_drag(page, cdp)
    page = open_check_page(browser, urls["check"])
    with throttled(page, rate) as cdp:
        check = measure_check(page, cdp)
    return {
        "drag": drag,
        "legend": legend,
        "cards": cards,
        "check": check,
        "switch": switch,
        "reports": reports,
        "drags": drags,
    }


def _median(runs: list[dict], key: str) -> float | None:
    values = [run[key] for run in runs if run[key] is not None]
    return statistics.median(values) if values else None


def _drift(first: list[dict], second: list[dict], key: str) -> float | None:
    gaps = [
        abs(a[key] - b[key])
        for a, b in zip(first, second, strict=True)
        if a[key] is not None and b[key] is not None
    ]
    return statistics.median(gaps) if gaps else None


def _fmt(value: float | None) -> str:
    return MISSING if value is None else f"{value:.2f}"


def table(
    results: dict[float, dict[str, list[dict]]], shas: dict[str, str], pairs: int, check: str
) -> str:
    lines = [
        "| Scenario | CPU | old ms/move | new ms/move | old-vs-old drift | budget 16.7 |",
        "|---|---|---|---|---|---|",
    ]
    for key, label, unit in SCENARIOS:
        for rate, runs in results.items():
            old, new, again = runs["old"], runs["new"], runs["old again"]
            new_median = _median(new, key)
            if unit != "ms/move":
                within = "n/a, one click"
            elif new_median is None:
                within = MISSING
            else:
                within = "yes" if new_median <= BUDGET_MS else "no"
            name = label if unit == "ms/move" else f"{label} ({unit})"
            lines.append(
                f"| {name} | {rate:g}x | {_fmt(_median(old, key))} | {_fmt(new_median)} "
                f"| {_fmt(_drift(old, again, key))} | {within} |"
            )
    notes = []
    for rate, runs in results.items():
        counts = {
            side: [run["reports"] / SWITCHES for run in runs[side]] for side in ("old", "new")
        }
        notes.append(
            f"{rate:g}x: old {_fmt(statistics.median(counts['old']))}, "
            f"new {_fmt(statistics.median(counts['new']))}"
        )
    drags = sum(run["drags"] for runs in results.values() for side in runs.values() for run in side)
    lines += [
        "",
        "`/report` requests per switch, after each report's first visit (median over "
        f"{pairs} run{'s' if pairs != 1 else ''} a side): {'; '.join(notes)}.",
        "",
        f"Old tree {shas['old']}, new tree {shas['new']}, driver head {shas['head']}; "
        f"{pairs} old/new/old runs per CPU rate, {MOVES} moves per hover or drag, "
        f"{SWITCHES} switches; the check plots on `{check}`. Main-thread `TaskDuration` only, in headless Chromium's "
        "software renderer: raster and GPU work are not measured."
        + ("" if drags == 0 else f" The cancelled drags still sent {drags} `/drag` requests."),
    ]
    return "\n".join(lines) + "\n"


def _tree(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or name not in ("old", "new"):
        raise argparse.ArgumentTypeError("expected old=PATH or new=PATH")
    tree = Path(path).expanduser().resolve()
    if not (tree / "src" / "superglm").is_dir():
        raise argparse.ArgumentTypeError(f"{tree} has no src/superglm")
    return name, tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tree",
        type=_tree,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="the old and the new source tree, as old=PATH and new=PATH",
    )
    parser.add_argument("--pairs", type=int, default=5, help="old/new/old runs per CPU rate")
    parser.add_argument(
        "--rates", type=float, nargs="+", default=[1.0, 4.0], help="CPU slowdowns to run at"
    )
    parser.add_argument(
        "--check-bundle",
        metavar="PATH",
        help="a saved SuperLSS {model, train, validation, cv} bundle for the check plots",
    )
    parser.add_argument("--out", metavar="PATH", help="also write the table here as Markdown")
    args = parser.parse_args(argv)
    trees = dict(args.tree)
    sessions = {
        "glm": ["poisson_cv_session"],
        "check": (
            ["bundle_session", str(Path(args.check_bundle).expanduser().resolve())]
            if args.check_bundle
            else ["pareto_lss_session"]
        ),
    }
    if set(trees) != {"old", "new"}:
        parser.error("give both --tree old=PATH and --tree new=PATH")

    from playwright.sync_api import sync_playwright

    shas = {name: head_sha(tree) for name, tree in trees.items()} | {"head": head_sha(HEAD)}
    results: dict[float, dict[str, list[dict]]] = {}
    with ExitStack() as resources:
        urls = {
            name: {
                kind: resources.enter_context(serve(tree, session))
                for kind, session in sessions.items()
            }
            for name, tree in trees.items()
        }
        playwright = resources.enter_context(sync_playwright())
        browser = playwright.chromium.launch(headless=True)
        resources.callback(browser.close)
        for rate in args.rates:
            runs: dict[str, list[dict]] = {"old": [], "new": [], "old again": []}
            for pair in range(args.pairs):
                for side, tree in (("old", "old"), ("new", "new"), ("old again", "old")):
                    runs[side].append(run_once(browser, urls[tree], rate))
                print(f"{rate:g}x: pair {pair + 1} of {args.pairs} done", file=sys.stderr)
            results[rate] = runs
    text = table(results, shas, args.pairs, " ".join(sessions["check"]))
    print(text, end="")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
