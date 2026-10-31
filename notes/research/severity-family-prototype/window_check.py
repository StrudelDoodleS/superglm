"""Generated log tails through PR 545's window combiner, against mpmath log window probabilities.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python window_check.py OUT_DIR

``combine`` is the combination documented in PR 545's ``_response_tails.py``
(written here from its stated formulas): with ``Delta = log T(far) - log T(near)``,
``kappa = 1/expm1(-Delta)`` and ``d = g_near - g_far``, the window's log
probability is ``log T(near) + log(-expm1(Delta))``, its score
``g_near + kappa d`` and its Hessian
``H_near + kappa (H_near - H_far) - kappa (1 + kappa) d d^T``.  ``T`` is the
survival function for a window in the upper half of the law and the CDF
otherwise.  The reference differentiates ``log(S(a) - S(b))`` with SymPy and
evaluates it with mpmath at 100 digits.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

import declarations as D
import sevgen

U = 2.0**-53
mp.mp.dps = 100
WINDOWS = ((0.5, 0.9), (0.9, 0.999), (0.999, 1 - 1e-9), (1e-9, 1e-3), (0.3, 0.3 + 1e-4))
CELLS = {
    "weibull": [(1000.0, 0.7), (1000.0, 3.0)],
    "burr12": [(1000.0, 2.0, 1.5), (1000.0, 1.2, 8.0)],
    "log_logistic": [(1000.0, 1.5), (1000.0, 6.0)],
    "lomax": [(1000.0, 1.1), (1000.0, 6.0)],
}


def combine(near, far, k):
    (vn, gn, hn), (vf, gf, hf) = near, far
    delta = vf - vn
    kappa = 1.0 / math.expm1(-delta)
    d = gn - gf
    outer = np.array([d[i] * d[j] for i, j in itertools.combinations_with_replacement(range(k), 2)])
    return vn + math.log(-math.expm1(delta)), gn + kappa * d, hn + kappa * (hn - hf) - kappa * (1 + kappa) * outer


def main() -> None:
    out_dir = Path(sys.argv[1])
    report = {}
    for declaration in (D.WEIBULL, D.BURR12, D.LOG_LOGISTIC, D.LOMAX):
        kernel, _ = sevgen.build(declaration, out_dir / "kernels")
        y, theta = sevgen.symbols(declaration)
        a, b = sp.symbols("a b", positive=True)
        window = sp.log(
            sp.exp(declaration.log_survival(a, *theta)) - sp.exp(declaration.log_survival(b, *theta))
        )
        orders = sevgen.derivatives(window, theta, 2)
        namespace = dict(sevgen.sevprims.MPMATH)
        reference = {
            order: [sp.lambdify((a, b, *theta), e, modules=[namespace, "mpmath"]) for e in exprs]
            for order, exprs in orders.items()
        }
        k = len(theta)
        rows = []
        for cell in CELLS[declaration.name]:
            law = declaration.scipy_law(*cell)
            th = np.array([cell])
            for pa, pb in WINDOWS:
                lo, hi = float(law.ppf(pa)), float(law.ppf(pb))
                upper = law.sf(lo) < 0.5
                kind, near, far = ("logsf", lo, hi) if upper else ("logcdf", hi, lo)
                ends = [kernel.tail(kind, np.array([x]), th) for x in (near, far)]
                ends = [(v[0], g[0], h[0]) for v, g, h in ends]
                value, score, hess = combine(ends[0], ends[1], k)
                args = [mp.mpf(lo), mp.mpf(hi), *(mp.mpf(t) for t in cell)]
                ref = [[f(*args) for f in reference[o]] for o in (0, 1, 2)]

                def error(got, want):
                    scale = max(abs(w) for w in want)
                    return float(max(abs(mp.mpf(float(g)) - w) for g, w in zip(got, want)) / scale) / U

                rows.append(
                    {
                        "cell": cell,
                        "window": [pa, pb],
                        "tail": kind,
                        "value": error([value], ref[0]),
                        "score": error(score, ref[1]),
                        "hessian": error(hess, ref[2]),
                    }
                )
        report[declaration.name] = rows
        worst = {key: max(r[key] for r in rows) for key in ("value", "score", "hessian")}
        print(declaration.name, json.dumps(worst))
        for r in rows:
            if r["window"] == [0.3, 0.3 + 1e-4]:
                print("   narrow", r["cell"], {key: round(r[key], 1) for key in ("value", "score", "hessian")})
    (out_dir / "window_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
