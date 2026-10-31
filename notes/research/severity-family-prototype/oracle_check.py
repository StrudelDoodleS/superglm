"""Generated oracle checks for every declared family: the test a new declaration gets for free.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python oracle_check.py OUT_DIR

Three independent references per family and point:

1. ``scipy.stats`` log density, log survival and log CDF: checks the declared
   law is scipy's law (the declaration is the only hand-written input).
2. The declaration's own SymPy derivatives evaluated by mpmath at 60 digits:
   checks the derivation, the printed code and the numba primitives.
3. Central differences of scipy's log density in the natural parameters
   (Richardson, order 1 and 2): checks the derivatives without SymPy.

Errors are in units of ``u = 2**-53``, relative to the row's channel scale (the
largest reference magnitude of that derivative order in the row), as
tests/test_family_derivatives_symbolic.py measures the hand-written families,
and separately as the worst entrywise relative error.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

import declarations as D
import sevgen

U = 2.0**-53
mp.mp.dps = 60

POINTS = {
    "weibull": [(1000.0, s) for s in (0.25, 0.7, 1.5, 6.0)],
    "weibull_mean": [(1000.0, s) for s in (0.25, 0.7, 1.5, 6.0)],
    "burr12": [(1000.0, c, k) for c, k in ((0.5, 0.8), (2.0, 1.5), (6.0, 0.3), (1.2, 8.0))],
    "log_logistic": [(1000.0, s) for s in (0.6, 1.5, 4.0, 12.0)],
    "lomax": [(1000.0, s) for s in (0.3, 1.1, 3.0, 25.0)],
    "gamma": [(1000.0, cv) for cv in (0.02, 0.3, 1.2, 5.0)],
    "naive_gamma": [(1000.0, cv) for cv in (0.02, 0.3, 1.2, 5.0)],
    "naive_burr12": [(1000.0, c, k) for c, k in ((0.5, 0.8), (2.0, 1.5), (6.0, 0.3), (1.2, 8.0))],
    "naive_lomax": [(1000.0, s) for s in (0.3, 1.1, 3.0, 25.0)],
    "lomax_log1p": [(1000.0, s) for s in (0.3, 1.1, 3.0, 25.0)],
}
BODY = (1e-3, 0.1, 0.5, 0.9, 0.999)
LEFT = (1e-12, 1e-30, 1e-100)
RIGHT = (1e-12, 1e-30, 1e-100)


def responses(law) -> list[tuple[str, float]]:
    out = [(f"p={p:g}", float(law.ppf(p))) for p in BODY]
    out += [(f"F={p:g}", float(law.ppf(p))) for p in LEFT]
    out += [(f"S={p:g}", float(law.isf(p))) for p in RIGHT]
    return [(label, y) for label, y in out if np.isfinite(y) and y > 0.0]


def errors(got: np.ndarray, reference: list) -> list[float]:
    """``[largest absolute error, largest reference magnitude]`` over the channel's entries."""
    if not np.all(np.isfinite(got)):
        return [float("inf"), max(abs(float(r)) for r in reference)]
    absolute = max(abs(mp.mpf(float(g)) - r) for g, r in zip(got, reference, strict=True))
    return [float(absolute), max(abs(float(r)) for r in reference)]


def link_scale(theta, score, hessian):
    """Score and packed Hessian in ``eta = log(theta)``: what a log-link solve reads."""
    k = len(theta)
    g = [theta[i] * score[i] for i in range(k)]
    h = []
    for m, (i, j) in enumerate(itertools.combinations_with_replacement(range(k), 2)):
        h.append(theta[i] * theta[j] * hessian[m] + (theta[i] * score[i] if i == j else 0))
    return g, h


def link_error(theta, got_score, got_hessian, ref_score, ref_hessian) -> list[float]:
    """:func:`errors` of the Hessian in ``eta = log(theta)``, as a log-link solve reads it."""
    theta_mp = [mp.mpf(t) for t in theta]
    _, ref = link_scale(theta_mp, ref_score, ref_hessian)
    if not (np.all(np.isfinite(got_score)) and np.all(np.isfinite(got_hessian))):
        return [float("inf"), float(max(abs(v) for v in ref))]
    got = [mp.mpf(float(v)) for v in got_hessian]
    _, got = link_scale(theta_mp, [mp.mpf(float(v)) for v in got_score], got)
    return [float(max(abs(x - r) for x, r in zip(got, ref, strict=True))), float(max(abs(v) for v in ref))]


def finite_difference(law_of, theta, y, step=1e-3):
    """Richardson central differences of scipy's logpdf in each natural parameter."""
    k = len(theta)

    def f(t):
        return float(law_of(*t).logpdf(y))

    def shifted(i, h, j=None, g=0.0):
        t = list(theta)
        t[i] *= 1.0 + h
        if j is not None:
            t[j] *= 1.0 + g
        return f(t)

    grad = []
    for i in range(k):
        h = step
        d1 = (shifted(i, h) - shifted(i, -h)) / (2 * h * theta[i])
        d2 = (shifted(i, h / 2) - shifted(i, -h / 2)) / (h * theta[i])
        grad.append((4 * d2 - d1) / 3)
    hess = []
    for i, j in itertools.combinations_with_replacement(range(k), 2):
        h = 10 * step

        def mixed(hh):
            return (
                shifted(i, hh, j, hh)
                - shifted(i, hh, j, -hh)
                - shifted(i, -hh, j, hh)
                + shifted(i, -hh, j, -hh)
            ) / (4 * hh * hh * theta[i] * theta[j])

        if i == j:

            def mixed(hh):  # noqa: F811
                return (shifted(i, hh) - 2 * f(theta) + shifted(i, -hh)) / (hh * theta[i]) ** 2

        hess.append((4 * mixed(h / 2) - mixed(h)) / 3)
    return np.array(grad), np.array(hess)


def check_family(declaration, out_dir: Path) -> dict:
    # Textbook forms cancel in their own reference too (1 - (1 + 1e-100)^-k), so
    # their mpmath oracle carries enough digits to resolve the deepest tail point.
    mp.mp.dps = 400 if declaration in D.NAIVE else 60
    kernel, report = sevgen.build(declaration, out_dir)
    derived = kernel.derived
    fns = {
        order: [sevgen.mpmath_function(derived, e) for e in exprs]
        for order, exprs in derived.density.items()
    }
    carrier = sevgen.mpmath_function(derived, derived.carrier)
    tails = {
        kind: {o: [sevgen.mpmath_function(derived, e) for e in ex] for o, ex in ch.items()}
        for kind, ch in derived.tails.items()
    }
    rows = []
    for theta in POINTS[declaration.name]:
        law = declaration.scipy_law(*theta)
        for label, y in responses(law):
            args = [mp.mpf(y), *(mp.mpf(t) for t in theta)]
            th = np.array([theta])
            value, score, hess = kernel.evaluate(np.array([y]), th, np.ones(1), 2)
            d3, d4 = kernel.third_fourth(np.array([y]), th, np.ones(1))
            gen_logpdf = value[0] + kernel.carrier(np.array([y]), np.ones(1))[0]
            ref_logpdf = fns[0][0](*args) + carrier(*args)
            row = {
                "theta": theta,
                "point": label,
                "y": y,
                "logpdf_vs_mpmath": errors(np.array([gen_logpdf]), [ref_logpdf]),
                "scipy_logpdf_vs_mpmath": errors(np.array([law.logpdf(y)]), [ref_logpdf]),
                "score": errors(score[0], [f(*args) for f in fns[1]]),
                "hessian": errors(hess[0], [f(*args) for f in fns[2]]),
                "link_hessian": link_error(
                    theta, score[0], hess[0], [f(*args) for f in fns[1]], [f(*args) for f in fns[2]]
                ),
                "third": errors(d3[0], [f(*args) for f in fns[3]]),
                "fourth": errors(d4[0], [f(*args) for f in fns[4]]),
            }
            for kind, channels in tails.items():
                v, s, h = kernel.tail(kind, np.array([y]), th)
                reference = [channels[0][0](*args)]
                scipy_value = law.logsf(y) if kind == "logsf" else law.logcdf(y)
                row[kind] = errors(v, reference)
                row[f"scipy_{kind}"] = errors(np.array([scipy_value]), reference)
                row[f"{kind}_score"] = errors(s[0], [f(*args) for f in channels[1]])
                row[f"{kind}_hessian"] = errors(h[0], [f(*args) for f in channels[2]])
                row[f"{kind}_link_hessian"] = link_error(
                    theta, s[0], h[0], [f(*args) for f in channels[1]], [f(*args) for f in channels[2]]
                )
            if label.startswith("p="):
                grad_fd, hess_fd = finite_difference(declaration.scipy_law, list(theta), y)
                row["fd_score_rel"] = float(
                    np.max(np.abs(score[0] - grad_fd)) / max(np.max(np.abs(grad_fd)), 1e-300)
                )
                row["fd_hessian_rel"] = float(
                    np.max(np.abs(hess[0] - hess_fd)) / max(np.max(np.abs(hess_fd)), 1e-300)
                )
            rows.append(row)
    return {"report": report, "rows": rows}


VALUES = ("logpdf_vs_mpmath", "scipy_logpdf_vs_mpmath", "logsf", "scipy_logsf", "logcdf", "scipy_logcdf")


def summarise(result: dict) -> dict:
    """Worst error in ``u``, body and tail apart.

    A log-likelihood value's error is absolute below one and relative above
    (``|error| / max(|value|, 1)``): a fit sums these values.  A derivative
    channel's error is relative to the larger of the row's own largest entry of
    that channel and the channel's largest magnitude over the body points of the
    same parameter cell, the curvature scale the fit sums it into: a tail row
    whose true curvature is negligible is not credited with a relative error the
    fit never sees, and one whose curvature is large is measured against itself.
    """
    rows = result["rows"]
    keys = [k for k in rows[0] if k not in ("theta", "point", "y")]
    scale = {}
    for row in rows:
        if row["point"].startswith("p="):
            cell = scale.setdefault(tuple(row["theta"]), {})
            for key in keys:
                if key in row and isinstance(row[key], list | tuple):
                    cell[key] = max(cell.get(key, 0.0), row[key][1])
    summary = {}
    for key in keys:
        body, tail = [], []
        for row in rows:
            if key not in row or row[key] is None:
                continue
            if not isinstance(row[key], list | tuple):
                value = row[key]
            elif key in VALUES:
                value = row[key][0] / max(row[key][1], 1.0) / U
            else:
                floor = max(scale[tuple(row["theta"])][key], row[key][1], 1e-300)
                value = row[key][0] / floor / U
            (body if row["point"].startswith("p=") else tail).append(value)
        summary[key] = {"body_max": max(body, default=None), "tail_max": max(tail, default=None)}
    return summary


def main() -> None:
    out_dir = Path(sys.argv[1])
    results = {}
    for declaration in D.FAMILIES + D.NAIVE:
        result = check_family(declaration, out_dir / "kernels")
        results[declaration.name] = result
        result["summary"] = summarise(result)
        print(declaration.name, json.dumps(result["summary"], default=str))
    (out_dir / "oracle_results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
