"""Untimed editor x revision sweep against exact references (#447, #448, #449).

Run it once per tree, with that tree's ``src`` first on the path, then compare:

    PYTHONPATH=<tree>/src python benchmarks/editor_revision_sweep.py run <out>
    python benchmarks/editor_revision_sweep.py compare <out_a> <out_b>

``run`` writes ``<out>.json`` (one record per scenario) and ``<out>.npz`` (each
scenario's linear predictor).  ``compare`` prints, per group, how many
scenarios are within their bound on each tree, how many predict bit for bit
alike, and every scenario where A's error exceeds B's by more than one ulp of
the row.

Two families of scenarios:

``sweep`` (576): family (Gaussian, Poisson) x offset of the numeric column
(0, 40, 1e12, 1e16) x ``retain_fit_state`` x fresh or pickle-reloaded x edited
term (spline ``s``, categorical ``g``, numeric ``x``) x change (``halve``: half
the effect, no intercept change; ``halve_plus``: plus 0.1, an intercept change
for ``s`` and ``g`` and a slope change for ``x``) x then (nothing, a post-fit
shape repair, a refit).  References are exact (``Fraction`` of the float
inputs):

- nothing: ``eta_before + d + X_pub dbeta``, with ``d`` the editor's own
  intercept change;
- repair: ``a_r + X_pub beta_r``, with ``beta_r`` the published projection
  and ``a_r`` the exact profiled intercept (Gaussian: the weighted mean
  residual; Poisson: ``log sum w y - log sum w exp(X_pub beta_r)``);
- refit: a clean fit, bit for bit.

Bounds are ``gamma_(n+p+8)`` of the fit's own centred magnitudes, plus the
repair's stopping rule.

``slopes``: numeric slope edits.  Sol's fixtures from the review of #448 and
#449, with their mirrors: ``x = +-1e16 {0, 1, 2}`` edited to slopes ``+-1, +-0.5,
1.1, 0.3, 3, 1e-3``, alone and beside a categorical in either feature order;
``x = 1e-308`` and ``1e-300 {-1, 0, 1}`` and ``1e-300 {1, 2, 3}`` through -1e308
and 1e308 both ways; two columns at ``+-1e12`` a few ulps apart edited to
``+-5e7``; a year column edited to slope 0; and a column at 1e16 halved, nudged
and zeroed.  Each is compared through ``predict``, ``metrics`` and a pickle
with the exact ``eta_before + sum x dbeta``, in ulps of each row.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd

U = float(np.finfo(np.float64).eps) / 2.0


def gamma(k: float) -> float:
    return k * U / (1.0 - k * U)


# ── the 576-scenario sweep ────────────────────────────────────────────────────


def _frame(offset: float, family: str):
    rng = np.random.default_rng(447)
    n = 120
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(0.0, 1.0, n)
    g = np.resize(np.array(["a", "b", "c", "d"], dtype=object), n)
    level = np.resize(np.array([0.0, 0.2, 0.3, -0.4]), n)
    eta = 0.2 * z + 0.8 * s + 0.1 * np.sin(9.0 * s) + level
    if family == "gaussian":
        y = 3.0 + eta + 0.05 * rng.normal(size=n)
    else:
        y = rng.poisson(np.exp(0.3 + 0.5 * eta)).astype(np.float64)
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), n)
    return pd.DataFrame({"x": offset + z, "s": s, "g": g}), y, w


def _build(family: str, retain: bool):
    from superglm import Categorical, Constraint, Numeric, PSpline, SuperGLM

    return SuperGLM(
        family=family,
        selection_penalty=0.0,
        spline_penalty=0.8,
        features={
            "x": Numeric(),
            "s": PSpline(
                n_knots=6, knot_strategy="uniform", constraint=Constraint.postfit.increasing
            ),
            "g": Categorical(base="first"),
        },
        weight_semantics="frequency",
        retain_fit_state=retain,
    )


def _public_design(model, frame) -> np.ndarray:
    design = np.zeros((len(frame), sum(group.size for group in model._groups)))
    for group in model._groups:
        block = model._specs[group.feature_name].transform(frame[group.feature_name].to_numpy())
        block = block.toarray() if hasattr(block, "toarray") else block
        design[:, group.sl] = np.asarray(block, dtype=np.float64).reshape(len(frame), -1)
    return design


def _magnitude(model, design, beta_old, beta_new, d) -> np.ndarray:
    result = model.result
    if result.centred_intercept is None or result.state_center is None:
        alpha, lo, centre = float(result.intercept), 0.0, np.zeros(result.beta.size)
    else:
        alpha = float(result.centred_intercept)
        lo = float(result.centred_intercept_lo or 0.0)
        centre = np.asarray(result.state_center, dtype=np.float64)
    spread = np.abs(design - centre[None, :]) @ (np.abs(beta_old) + np.abs(beta_new))
    return (
        abs(alpha) + abs(lo) + abs(d) + spread + float(np.abs(centre) @ np.abs(beta_new - beta_old))
    )


def _exact_rows(design, beta) -> list[Fraction]:
    coefficients = [Fraction(float(b)) for b in beta]
    return [
        sum((Fraction(float(v)) * b for v, b in zip(row, coefficients) if v != 0.0), Fraction(0))
        for row in design
    ]


def _editor_change(model, term, name: str) -> float:
    """The intercept change the editor applies for this term, by its own rule."""
    from superglm.editor import apply as editor_apply
    from superglm.editor.terms import native_log_effect_values

    if name == "x":
        return 0.0
    spec = model._specs[name]
    if name == "g":
        value = float(editor_apply._level_target_map(term, spec)[str(spec._base_level)])
    else:
        value, _ = editor_apply._solve_with_intercept(
            editor_apply._as_dense(spec.transform(term.x)),
            native_log_effect_values(term),
            editor_apply._term_weights(term),
        )
    return 0.0 if abs(value) < 1e-15 else float(value)


def _repair_reference(family, design, beta_r, y, w) -> np.ndarray:
    eta0 = _exact_rows(design, beta_r)
    if family == "gaussian":
        weights = [Fraction(float(v)) for v in w]
        level = sum(
            (wi * (Fraction(float(yi)) - e) for wi, yi, e in zip(weights, y, eta0)), Fraction(0)
        ) / sum(weights)
        return np.array([float(level + e) for e in eta0])
    relative = np.array([float(e - eta0[0]) for e in eta0])
    level = math.log(math.fsum(w * y)) - math.log(math.fsum(w * np.exp(relative)))
    return level + relative


def _sweep_scenario(model, design, eta_before, X, y, w, family, retain, name, change, post):
    from superglm.editor import EditorSession

    beta_old = np.asarray(model.result.beta, dtype=np.float64)
    session = EditorSession.from_model(model, terms=[name], train_data=(X, y, w))
    term = session.terms[name]
    effect = np.asarray(term.edited_log_effect, dtype=np.float64)
    term.edited_log_effect = 0.5 * effect + (0.1 if change == "halve_plus" else 0.0)
    d = _editor_change(model, term, name)
    edited = session.to_model()
    beta_new = np.asarray(edited.result.beta, dtype=np.float64)
    if post == "none":
        eta = edited._predict_eta_raw_exact(X)
        reference = [
            Fraction(float(e)) + Fraction(d) + r
            for e, r in zip(eta_before, _exact_rows(design, beta_new - beta_old))
        ]
        error = np.array([abs(float(Fraction(float(v)) - r)) for v, r in zip(eta, reference)])
        magnitude = _magnitude(model, design, beta_old, beta_new, d)
        magnitude = magnitude + _magnitude(model, design, beta_old, beta_old, 0.0)
        bound = 4.0 * gamma(len(beta_new) + 8) * magnitude
    elif post == "repair":
        edited.apply_shape_postfit(X, sample_weight=w, n_grid=80)
        eta = edited._predict_eta_raw_exact(X)
        beta_r = np.asarray(edited.result.beta, dtype=np.float64)
        reference = _repair_reference(family, design, beta_r, y, w)
        error = np.abs(eta - reference)
        magnitude = _magnitude(model, design, beta_old, beta_new, d)
        magnitude = magnitude + _magnitude(model, design, beta_new, beta_r, 0.0)
        mu = np.exp(reference) if family == "poisson" else reference
        if family == "gaussian":
            residual = max(1.0, math.fsum(np.abs(w * (y - reference))))
            tolerance = 128.0 * U * residual / math.fsum(w)
        else:
            tolerance = 1e-11 * (1.0 + math.fsum(np.abs(w * (y - mu)))) / math.fsum(w * mu)
        bound = 2.0 * tolerance + 4.0 * gamma(len(y) + len(beta_new) + 8) * (
            float(np.max(magnitude)) + float(np.max(np.abs(y)))
        )
    else:
        edited.fit(X, y, sample_weight=w)
        eta = edited._predict_eta_raw_exact(X)
        error = np.abs(
            eta - _build(family, retain).fit(X, y, sample_weight=w)._predict_eta_raw_exact(X)
        )
        bound = np.zeros_like(error)
    return np.asarray(eta, dtype=np.float64), error, np.broadcast_to(bound, error.shape)


def run_sweep(rows: list[dict], predictors: dict[str, np.ndarray]) -> None:
    for family in ("gaussian", "poisson"):
        for offset in (0.0, 40.0, 1e12, 1e16):
            X, y, w = _frame(offset, family)
            for retain in (True, False):
                fitted = _build(family, retain).fit(X, y, sample_weight=w)
                for reload in (False, True):
                    model = pickle.loads(pickle.dumps(fitted)) if reload else fitted
                    design = _public_design(model, X)
                    eta_before = model._predict_eta_raw_exact(X)
                    for name in ("s", "g", "x"):
                        for change in ("halve", "halve_plus"):
                            for post in ("none", "repair", "refit"):
                                key = (
                                    f"sweep|{family}|{offset:g}|retain={retain}|reload={reload}"
                                    f"|{name}|{change}|{post}"
                                )
                                record = {"key": key}
                                try:
                                    eta, error, bound = _sweep_scenario(
                                        model,
                                        design,
                                        eta_before,
                                        X,
                                        y,
                                        w,
                                        family,
                                        retain,
                                        name,
                                        change,
                                        post,
                                    )
                                    predictors[key] = eta
                                    ulps = error / np.spacing(np.maximum(np.abs(eta), 5e-324))
                                    record.update(
                                        status="ok",
                                        error=float(np.max(error)),
                                        bound=float(np.max(bound)),
                                        within=bool(np.all(error <= bound)),
                                        ulps=float(np.max(ulps)),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    record.update(status=f"{type(exc).__name__}: {str(exc)[:80]}")
                                rows.append(record)


# ── numeric slope edits: Sol's fixtures and their mirrors ────────────────────


def _slope_cases():
    cases = []
    y3 = np.resize([2.0, 3.0, 2.0], 60)
    for scale in (1e16, -1e16):
        x = scale * np.resize([0.0, 1.0, 2.0], 60)
        for slope in (1.0, -1.0, 0.5, -0.5, 1.1, 0.3, 3.0, 1e-3):
            cases.append((f"P2a scale {scale:g} slope {slope:g}", {"x": x}, y3, [("x", slope)]))
    x = 1e16 * np.resize([0.0, 1.0, 2.0], 60)
    g = np.resize(np.array(["a", "b", "c", "d", "e"], dtype=object), 60)
    yg = y3 + np.resize([0.0, 0.3, -0.2, 0.1, 0.25], 60)
    for slope in (1.0, 1.1):
        cases.append(
            (f"P2a + categorical after, slope {slope:g}", {"x": x, "g": g}, yg, [("x", slope)])
        )
        cases.append(
            (f"P2a + categorical before, slope {slope:g}", {"g": g, "x": x}, yg, [("x", slope)])
        )
    for scale, pattern, label in (
        (1e-308, [-1.0, 0.0, 1.0], "zero centre"),
        (1e-300, [-1.0, 0.0, 1.0], "zero centre"),
        (1e-300, [1.0, 2.0, 3.0], "centred"),
    ):
        x = scale * np.resize(pattern, 60)
        for first, second in ((-1e308, 1e308), (1e308, -1e308)):
            cases.append(
                (
                    f"P2b {label} {scale:g} {first:g}->{second:g}",
                    {"x": x},
                    y3,
                    [("x", first), ("x", second)],
                )
            )
    spacing = float(np.spacing(1e12))
    a = np.resize([-3.0, -1.0, 1.0, 3.0], 64)
    b = np.repeat(np.resize([-3.0, -1.0, 1.0, 3.0], 16), 4)[:64]
    for sign in (1.0, -1.0):
        xx, tt = sign * (1e12 + spacing * a), sign * (1e12 + spacing * b)
        yy = 2.5 + 1e8 * (xx - tt)
        for edits in ((("x", 5e7), ("t", -5e7)), (("x", -5e7), ("t", 5e7))):
            cases.append(
                (f"cancelling 1e12 sign {sign:g} {edits}", {"x": xx, "t": tt}, yy, list(edits))
            )
    year = 2000.0 + np.resize(np.arange(20.0), 60)
    cases.append(
        (
            "year column to slope 0",
            {"x": year},
            1.0 + 0.01 * (year - 2010.0) + np.resize([0.1, -0.1, 0.05], 60),
            [("x", 0.0)],
        )
    )
    far = 1e16 + 2.0 * np.resize([-4.0, -2.0, 0.0, 2.0, 4.0], 60)
    for slope, label in ((0.1, "halve"), (0.2 + 2**-40, "nudge"), (0.0, "to zero")):
        cases.append(
            (f"far 1e16 slope {label}", {"x": far}, 3.0 + 0.2 * (far - 1e16), [("x", slope)])
        )
    return cases


def run_slopes(rows: list[dict]) -> None:
    from superglm import Categorical, Numeric, SuperGLM
    from superglm.editor import EditorSession

    for name, columns, y, edits in _slope_cases():
        frame = pd.DataFrame(columns)
        record = {"key": f"slopes|{name}"}
        try:
            features = {
                c: Categorical(base="first") if c == "g" else Numeric() for c in frame.columns
            }
            model = SuperGLM(family="gaussian", selection_penalty=0.0, features=features)
            model.fit(frame, y)
            edited = model
            for column, slope in edits:
                session = EditorSession.from_model(edited, terms=[column], train_data=(frame, y))
                session.terms[column].edited_log_effect = np.array([slope], dtype=np.float64)
                edited = session.to_model()
            groups = {group.feature_name: group for group in model._groups}
            final = dict(edits)
            changes = {
                c: Fraction(s) - Fraction(float(model.result.beta[groups[c].sl][0]))
                for c, s in final.items()
            }
            reference = []
            for i, before in enumerate(model.predict(frame)):
                value = Fraction(float(before))
                for column, change in changes.items():
                    value += Fraction(float(frame[column].iloc[i])) * change
                reference.append(value)
            worst = 0.0
            for values in (
                edited.predict(frame),
                edited.metrics(frame, y).eta,
                pickle.loads(pickle.dumps(edited)).predict(frame),
            ):
                for v, r in zip(values, reference):
                    error = float(abs(Fraction(float(v)) - r))
                    worst = max(worst, error / float(np.spacing(abs(float(r)) or 5e-324)))
            record.update(status="ok", ulps=worst)
        except Exception as exc:  # noqa: BLE001
            record.update(status=f"{type(exc).__name__}: {str(exc)[:80]}")
        rows.append(record)


# ── compare ──────────────────────────────────────────────────────────────────


def compare(first: str, second: str) -> None:
    from collections import Counter, defaultdict

    rows_a = {r["key"]: r for r in json.load(open(first + ".json"))}
    rows_b = {r["key"]: r for r in json.load(open(second + ".json"))}
    eta_a, eta_b = np.load(first + ".npz"), np.load(second + ".npz")
    groups: dict[tuple, Counter] = defaultdict(Counter)
    worse = []
    for key, a in rows_a.items():
        b = rows_b.get(key, {"status": "missing"})
        parts = key.split("|")
        group = (parts[0], parts[-1]) if parts[0] == "sweep" else ("slopes",)
        counts = groups[group]
        counts["A ok" if a["status"] == "ok" else "A refused"] += 1
        counts["B ok" if b["status"] == "ok" else "B refused"] += 1
        if a["status"] == "ok" and "within" in a:
            counts["A within"] += a["within"]
        if b["status"] == "ok" and "within" in b:
            counts["B within"] += b["within"]
        if key in eta_a.files and key in eta_b.files:
            counts["bit-identical"] += bool(np.array_equal(eta_a[key], eta_b[key]))
        if a["status"] != "ok" and b["status"] == "ok":
            worse.append((key, "A refuses, B does not"))
        elif a["status"] == "ok" and b["status"] == "ok" and a["ulps"] > b["ulps"] + 1.0:
            worse.append((key, f"A {a['ulps']:.3g} ulps, B {b['ulps']:.3g}"))
    for group in sorted(groups):
        print(group, dict(groups[group]))
    print(
        f"\nA worse than B by more than one ulp of the row, or refusing where B does not: {len(worse)}"
    )
    for key, note in worse:
        print(f"  {note}  {key}")


def main(argv: list[str]) -> None:
    if argv[0] == "compare":
        compare(argv[1], argv[2])
        return
    import superglm

    print(superglm.__file__, file=sys.stderr)
    warnings.simplefilter("ignore")
    rows: list[dict] = []
    predictors: dict[str, np.ndarray] = {}
    run_sweep(rows, predictors)
    run_slopes(rows)
    np.savez(argv[1] + ".npz", **predictors)
    with open(argv[1] + ".json", "w") as handle:
        json.dump(rows, handle, indent=0)
    print(len(rows), "scenarios", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
