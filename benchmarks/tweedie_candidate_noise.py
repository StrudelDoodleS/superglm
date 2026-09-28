"""Measure how far a Tweedie power search's recorded profile values sit from the exact profile.

The lower-point floor in ``superglm.profiling.tweedie`` bounds each recorded
value's error by its fit's certificate times a measured ratio
(``_CANDIDATE_ERROR_RATIO`` for REML, shape-constrained and selection-penalty
fits, ``_SCORING_ERROR_RATIO`` for Fisher scoring). This script is that
measurement. Each row is a JSON line.

``reml SHAPE N P_TRUE SEED [REFERENCE_TOL]`` fits candidates at powers
``P_TRUE + offset`` over ``OFFSETS``, once at the search grade
``_SEARCH_REML_TOL = 1e-6`` and once at a tight reference, each from a cold
start, and reports ``n |e| / (tol (1 + |V|))``: ``e`` is the mean-NLL
difference and ``V`` the candidate's REML objective, whose change and projected
gradient are what ``tol (1 + |V|)`` certifies. A REML fit minimises ``V``, not
the NLL, so ``e`` is two-sided and first order in the smoothing parameters'
residual error.

``ml SHAPE N SEED TOL [TOL ...]`` runs ``estimate_p(fit_mode="fit")`` with its 95%
interval at model ``tol``, replays its evaluation sequence (checked bitwise),
and continues every fit to tol 1e-13 from its own coefficients. It reports the
ratio at ``tol`` and, for each fit, ``|e| / (c (1 + n |nll|) / n)`` with ``c``
the relative change the fit last achieved. Every tol in ``(c, previous
change]`` returns the same iterate, so the latter is the worst ratio any
placement of tol can show.

Shapes: ``book`` (a smooth ``Spline(n_knots=10)`` and a five-level
``Categorical``, compound Poisson-gamma at phi 6); ``trend_book`` (the same
with a linear trend); ``flat`` (``tests/_tweedie_profile_fixtures``
``flat_lambda_fixture``: saturated cr terms, nearly flat in log lambda);
``char:<case>`` (``benchmarks/tweedie_nb_characterisation.py`` cases,
e.g. ``re_flat_p18``, ``re_flat_p15_lowzero``, ``re_ident_p15``, 3,000 rows
with a random effect); ``lasso:<shape>`` adds ``selection_penalty="auto"`` and
``qp:<shape>`` an increasing cr constraint on ``x``.

Recorded 2026-09-28 (threads pinned, ``0d42d1b8`` plus the floor):

* REML, 198 fits over ``char:re_flat_p18``, ``char:re_flat_p15_lowzero``,
  ``char:re_ident_p15`` (3k), ``flat`` (12k and 100k) and ``book`` (100k) at
  ``P_TRUE`` 1.3, 1.5 and 1.7, reference 1e-11 (1e-9 on ``flat`` at 100k):
  ratio 6e-5 to 5.4, worst ``char:re_flat_p18`` at p = 1.701.
  ``_CANDIDATE_ERROR_RATIO = 8``.
* ML Fisher scoring (``book``/``trend_book`` 20k and 100k, ``flat`` 12k and
  100k, the characterisation books): at most 1.1e-4 at the tol run and 9.0e-4
  at the worst placement (``flat`` 12k). ``_SCORING_ERROR_RATIO = 2e-3``.
* ML with a selection penalty: 0.12 to 0.21 at tol 1e-4 to 1e-10; with the
  increasing cr constraint (``qp:trend_book`` 20k): 0.14 at tol 1e-6, 6.7 at
  1e-9. Both keep ``_CANDIDATE_ERROR_RATIO``.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 uv run python benchmarks/tweedie_candidate_noise.py \\
        reml char:re_flat_p18 3000 1.7 1

It reads private names of ``superglm.profiling.tweedie``, so it documents the
code it was run against.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import superglm.profiling.tweedie as tw
from superglm import (
    Categorical,
    Constraint,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
    generate_tweedie_cpg,
)

ROOT = Path(__file__).resolve().parents[1]
LEVEL_EFFECTS = np.array([0.0, 0.2, -0.3, 0.1, 0.4])
OFFSETS = (0.0, 1e-6, -1e-6, 1e-5, 1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.03, -0.03, 0.05)
SEARCH_TOL = 1e-6
TIGHT_ML_TOL = 1e-13


def _book(n: int, p: float, seed: int, trend: float):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    level = rng.integers(0, len(LEVEL_EFFECTS), n)
    mu = np.exp(-2.4 + 0.5 * np.sin(2.0 * np.pi * x) + trend * x + LEVEL_EFFECTS[level])
    y = generate_tweedie_cpg(n, mu, 6.0, p, rng=rng)
    frame = pd.DataFrame({"x": x, "level": np.char.add("L", level.astype(str))})
    return frame, y, None, None, {"x": Spline(n_knots=10), "level": Categorical()}, {}


def shape(name: str, n: int, p: float, seed: int):
    """(frame, y, weights, offset, features, model options) for a named shape."""
    if name.startswith("lasso:"):
        frame, y, w, off, features, _ = shape(name[len("lasso:") :], n, p, seed)
        return frame, y, w, off, features, {"selection_penalty": "auto"}
    if name.startswith("qp:"):
        frame, y, w, off, features, _ = shape(name[len("qp:") :], n, p, seed)
        constrained = Spline(n_knots=10, kind="cr", constraint=Constraint.fit.increasing)
        return frame, y, w, off, {**features, "x": constrained}, {}
    if name in ("book", "trend_book"):
        return _book(n, p, seed, 0.8 if name == "trend_book" else 0.0)
    if name == "flat":
        sys.path.insert(0, str(ROOT / "tests"))
        from _tweedie_profile_fixtures import flat_lambda_fixture

        frame, y, w, off, features = flat_lambda_fixture(n, seed)
        return frame, y, w, off, features, {}
    if name.startswith("char:"):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from tweedie_nb_characterisation import build_case

        case = name[len("char:") :]
        _, frame, y = build_case(case)
        if case.startswith("re_"):
            features = {"u": Categorical(), "r": RandomEffect()}
        elif case == "unpen":
            features = {"band": Categorical(), "level": Categorical()}
        else:
            features = {"x": Spline(n_knots=10), "level": Categorical()}
        return frame, y, None, None, features, {}
    raise SystemExit(f"unknown shape {name!r}")


def _quietly(call, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args)


def reml(name: str, n: int, p_true: float, seed: int, reference_tol: float = 1e-11) -> None:
    frame, y, w, off, features, options = shape(name, n, p_true, seed)
    rows = len(y)
    w = np.ones(rows) if w is None else w
    model = SuperGLM(family=Tweedie(p=p_true), features=features, **options)
    profile = tw._PowerProfile(model, frame, y, w, off, "fit_reml")
    for offset in OFFSETS:
        p = p_true + offset
        fits = {}
        for tol in (SEARCH_TOL, reference_tol):
            tw._SEARCH_REML_TOL = tol
            profile.warm_beta = profile.warm_intercept = None
            fits[tol] = (_quietly(profile, p), profile.clone._reml_result.objective)
        (nll, objective), (reference, _) = fits[SEARCH_TOL], fits[reference_tol]
        ratio = rows * abs(nll - reference) / (SEARCH_TOL * (1.0 + abs(objective)))
        record = {"shape": name, "n": rows, "p_true": p_true, "p": p, "ratio": ratio}
        record["statistic_error"] = 2.0 * rows * (nll - reference)
        print(json.dumps(record), flush=True)


def ml(name: str, n: int, seed: int, tols: list[float]) -> None:
    frame, y, w, off, features, options = shape(name, n, 1.5, seed)
    rows = len(y)
    w = np.ones(rows) if w is None else w
    final_change: list[float] = []
    solve = tw._solve_coefficients

    def recording(*args, **kwargs):
        result = solve(*args, **{**kwargs, "record_diagnostics": True})
        final_change.append(result.iteration_log[-1].convergence_value)
        return result

    tw._solve_coefficients = recording
    for tol in tols:
        model = SuperGLM(family=Tweedie(p=1.5), features=features, tol=tol, **options)
        result = _quietly(lambda: model.estimate_p(frame, y, sample_weight=w, offset=off))
        _quietly(result.interval, 0.05)
        sequence = [(p, v) for p, v in result._objective.values.items() if np.isfinite(v)]
        loose, tight = (
            tw._PowerProfile(
                SuperGLM(family=Tweedie(p=1.5), features=features, tol=grade, **options),
                frame,
                y,
                w,
                off,
                "fit",
            )
            for grade in (tol, TIGHT_ML_TOL)
        )
        at_tol = placed = 0.0
        bitwise = True
        for p, recorded in sequence:
            final_change.clear()
            value = _quietly(loose, p)
            change = final_change[-1]
            bitwise &= value == recorded
            tight.warm_beta, tight.warm_intercept = loose.warm_beta, loose.warm_intercept
            error = abs(value - _quietly(tight, p))
            scale = (1.0 + rows * abs(value)) / rows
            at_tol = max(at_tol, error / (tol * scale))
            placed = max(placed, error / (max(change, np.finfo(float).tiny) * scale))
        record = {"shape": name, "n": rows, "tol": tol, "fits": len(sequence)}
        record |= {"replay_bitwise": bool(bitwise), "ratio": at_tol, "worst_placement": placed}
        print(json.dumps(record), flush=True)
    tw._solve_coefficients = solve


if __name__ == "__main__":
    mode, name, *rest = sys.argv[1:]
    if mode == "reml":
        n, p_true, seed = int(rest[0]), float(rest[1]), int(rest[2])
        reml(name, n, p_true, seed, *(float(t) for t in rest[3:4]))
    else:
        ml(name, int(rest[0]), int(rest[1]), [float(t) for t in rest[2:]])
