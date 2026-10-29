"""Bitwise characterisation of ``fit_irls_direct`` against another tree on the same machine.

Fits a fixed matrix of cases and records every call the fits make to
``fit_irls_direct``, so a refactor of the solver can be checked bit for bit
against master.  It is not a CI test: it checks same-machine, same-thread
identity, not identity across machines or BLAS builds.

    list                          the case names
    run --out DIR [--jobs N] [--only REGEX] [--shard I/K] [--coverage]
                                  fit the cases, one fresh process each
    compare A B                   exit 1 unless every case is bitwise identical
    census DIR                    which solver modes and branches the run reached

Run it on each tree with that tree's interpreter and the same harness file::

    master/.venv/bin/python tree/benchmarks/irls_direct_characterisation.py run --out A
    tree/.venv/bin/python tree/benchmarks/irls_direct_characterisation.py run --out B
    python tree/benchmarks/irls_direct_characterisation.py compare A B

The matrix (``m/``): every family and link of ``release_differential``'s GLM
corpus, on its 400-row data, through the dense route (exact and binned by
``fit_reml``, exact by ``fit``), the shape-constrained P-spline on the QP
(``scop``, its term's name; the QP from #537) and on SCOP (``scopeng``, the
same term with ``shape_engine="scop"``), both exact and binned, the
constrained QP, QR, and the structured random-effect, ``fs`` and ``sz``
routes, with binned, ``fit`` and further term kinds on a few pairs.  The
fixtures (``x/``): #437's mean-space Newton on the dense, QP and structured
routes and fits that do not engage it, the Gamma/log Fisher-to-Newton switch
and its latch, Tweedie/log observed Newton, nested random-effect chains,
shape-constrained P-splines with repeated rows, two groups and a translated
column, on the QP (``x/scop/``) and on SCOP (``x/scop-engine/``), a term the
QP holds flat beside a fitted one (its active face), identity-link fits at
the zero edge, where the working QP goes unbounded, and the QP step's
infeasible-start restarts (``x/qp/``, some with the QP solve or the line
search's trial replaced as master's tests replace them), the true-score
certificate's constrained, latent and truncated-direction branches, score
stagnation, the float64 fixed-point stop, an ndarray design and the input
refusals.  ``census`` says which routes and branches a run reached, the
active faces among them; ``--coverage`` adds line coverage.

Each case runs in its own process with every thread pool pinned to one thread
and ``PYTHONHASHSEED=0``.  The process rebinds ``fit_irls_direct`` and
``solve_constrained_qp`` in every module that imported them, by identity.
Each call to ``fit_irls_direct`` is recorded:

- its inputs (warm start, lambdas, weights, offset, private switches);
- every field of the returned ``PIRLSResult`` and the rest of the returned
  tuple, with ``record_diagnostics`` forced on so the per-iteration
  diagnostics (deviance, halvings, trials, step length, condition, the stop
  rule's numbers and the termination reason) are pinned on every call, and
  put back to ``None`` for a caller that did not ask for them;
- the profile counters the call wrote, with a dict supplied when the caller
  passed none (the profile only counts; it decides nothing);
- ``cache_out``, the Newton latch and the deferred rank-info count after it;
- the published predictor ``eta``, ``mu`` and Fisher weights at the result;
- every constrained-QP solve inside it, with its active set.

Then the model-level result, held-out predictions, REML diagnostics, per-term
EDF and the warnings raised.  The ``t/`` cases also run with
``SUPERGLM_DEBUG=2`` and pin the trace and debug rows.  Floats are kept as
``float.hex``, arrays as a hash of dtype, shape and bytes.  Wall-clock fields
(``*_s``, timestamps, run ids), objects' caches and memos (some are keyed by
``id()``) and logging text are left out; the design, its group matrices and the
structured layouts derived from it are named, not walked.

Output per case: ``cases/<id>.hash.json`` (the hashes ``compare`` reads),
``cases/<id>.leaves.json.gz`` (every recorded value, read only for a case that
differs) and ``cases/<id>.npz`` (each call's ``beta`` and the last call's
``eta``, ``mu`` and weights).  Run it through benchboard: ``benchboard heavy
<label> --cores N -- timeout T <python> <this file> run --out DIR --jobs N``.

Only the standard library is imported at module level: the thread pins must
reach the case processes before numpy is loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
import types
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "SUPERGLM_BLAS_THREADS": "1",
    "SUPERGLM_N_JOBS": "1",
    "PYTHONHASHSEED": "0",
}
SMALL_N = 400
CASE_TIMEOUT_S = 600

# Wall-clock and per-run identifiers: never compared.
_SKIPPED_KEY = re.compile(
    r"(?:_s|_seconds)$|(?:^|_)(?:time|timestamp|elapsed|wall|duration|run_id|base_dir)(?:_|$)"
)

# term (``release_differential.glm_terms``) and ``direct_solve`` per solver mode
MODES = {
    "dense": ("mains", "gram"),
    "qp": ("qp", "gram"),
    "qr": ("re", "qr"),
    "qrmains": ("mains", "qr"),
    "scop": ("scop", "auto"),
    # the ``scop`` term with its P-spline on SCOP (``shape_engine="scop"``)
    "scopeng": ("scop", "auto"),
    "re": ("re", "structured"),
    "fs": ("fs", "structured"),
    "sz": ("sz", "structured"),
    "tensor": ("tensor", "gram"),
    "splcat": ("splcat", "gram"),
    "select": ("select", "gram"),
}

# Cases that also run traced (``SUPERGLM_DEBUG=2``), as ``t/<case>``.
TRACED = [
    "m/poisson-log/dense/exact/fit_reml",
    "m/gamma-log/dense/binned/fit_reml",
    "m/binomial-log/dense/exact/fit_reml",
    "m/poisson-log/scop/exact/fit_reml",
    "m/gaussian-identity/scop/binned/fit_reml",
    "m/poisson-log/scopeng/exact/fit_reml",
    "m/gaussian-identity/scopeng/binned/fit_reml",
    "m/gaussian-identity/qp/exact/fit_reml",
    "m/poisson-log/qr/exact/fit_reml",
    "m/binomial-logit/re/exact/fit_reml",
    "m/tweedie15-log/fs/exact/fit_reml",
    "m/poisson-log/sz/exact/fit_reml",
    "x/binlog/thin-events/gram",
    "x/binlog/thin-events/auto",
]


# --------------------------------------------------------------------------- case matrix


def _famlink_tag(family: str, link: str) -> str:
    return f"{family}-{link}"


def _registry() -> dict[str, tuple[Callable, dict]]:
    import release_differential as rd

    cases: dict[str, tuple[Callable, dict]] = {}

    def matrix(family, link, mode, route, fit):
        name = f"m/{_famlink_tag(family, link)}/{mode}/{route}/{fit}"
        cases[name] = (_matrix_case, {"family": family, "link": link, "mode": mode,
                                      "route": route, "fit": fit})  # fmt: skip

    for family, link in rd.GLM_FAMLINKS:
        for route in ("exact", "binned"):
            matrix(family, link, "dense", route, "fit_reml")
            matrix(family, link, "scop", route, "fit_reml")
            matrix(family, link, "scopeng", route, "fit_reml")
        matrix(family, link, "dense", "exact", "fit")
        for mode in ("qp", "qr", "re", "fs", "sz"):
            matrix(family, link, mode, "exact", "fit_reml")
    for family, link in [("poisson", "log"), ("gamma", "log"), ("binomial", "logit"),
                         ("gaussian", "identity"), ("tweedie15", "log")]:  # fmt: skip
        matrix(family, link, "qp", "binned", "fit_reml")
    for family, link in [("poisson", "log"), ("gaussian", "identity"), ("binomial", "logit")]:
        matrix(family, link, "qrmains", "exact", "fit")
    matrix("poisson", "log", "qr", "binned", "fit_reml")
    for family, link in [("poisson", "log"), ("tweedie15", "log")]:
        for mode in ("re", "fs", "sz"):
            matrix(family, link, mode, "binned", "fit_reml")
    for family, link in [("poisson", "log"), ("gamma", "log"), ("binomial", "logit"),
                         ("gaussian", "identity")]:  # fmt: skip
        for mode in ("tensor", "splcat", "select"):
            matrix(family, link, mode, "exact", "fit_reml")
    for family, link in [("poisson", "log"), ("gaussian", "identity"), ("gamma", "log"),
                         ("binomial", "logit"), ("tweedie15", "log")]:  # fmt: skip
        matrix(family, link, "scop", "exact", "fit")
        matrix(family, link, "scopeng", "exact", "fit")
    matrix("poisson", "log", "scop", "binned", "fit")
    matrix("poisson", "log", "scopeng", "binned", "fit")
    matrix("poisson", "log", "fs", "exact", "fit")

    specials = {
        "x/binlog/msn-dense": (_msn_event_rows, {"constrained": False}),
        "x/binlog/msn-qp": (_msn_event_rows, {"constrained": True}),
        "x/binlog/msn-lowered/deviance": (_msn_lowered, {"convergence": "deviance"}),
        "x/binlog/msn-lowered/coefficients": (_msn_lowered, {"convergence": "coefficients"}),
        "x/binlog/thin-events/auto": (_thin_events, {"direct_solve": "auto"}),
        "x/binlog/thin-events/gram": (_thin_events, {"direct_solve": "gram"}),
        "x/binlog/offset-plus2/fit": (_binlog_offset, {"offset": 2.0, "plain": True}),
        "x/binlog/offset-plus2/fit_reml": (_binlog_offset, {"offset": 2.0, "plain": False}),
        "x/binlog/offset-zero/fit": (_binlog_offset, {"offset": 0.0, "plain": True}),
        "x/gamma/fisher-to-newton": (_gamma_switch, {}),
        "x/gamma/fisher-kept": (_gamma_kept, {}),
        "x/tweedie/observed-direct": (_tweedie_observed, {}),
        "x/nested/poisson-log": (_nested, {"family": "poisson"}),
        "x/nested/gamma-log": (_nested, {"family": "gamma"}),
        "x/scop/lone-repeated-rows": (_scop_variant, {"variant": "repeated", "route": "exact"}),
        "x/scop/two-groups/exact": (_scop_variant, {"variant": "two", "route": "exact"}),
        "x/scop/two-groups/binned": (_scop_variant, {"variant": "two", "route": "binned"}),
        "x/scop/translated-column": (_scop_variant, {"variant": "translated", "route": "exact"}),
        "x/scop/binomial-log-two-groups": (_scop_variant, {"variant": "binlog", "route": "exact"}),
        "x/scop-engine/lone-repeated-rows": (_scop_variant, {"variant": "repeated", "route": "exact",
                                                             "engine": "scop"}),
        "x/scop-engine/two-groups/exact": (_scop_variant, {"variant": "two", "route": "exact",
                                                           "engine": "scop"}),
        "x/scop-engine/two-groups/binned": (_scop_variant, {"variant": "two", "route": "binned",
                                                            "engine": "scop"}),
        "x/scop-engine/translated-column": (_scop_variant, {"variant": "translated",
                                                            "route": "exact", "engine": "scop"}),
        "x/scop-engine/binomial-log-two-groups": (_scop_variant, {"variant": "binlog",
                                                                  "route": "exact",
                                                                  "engine": "scop"}),
        "x/binlog/scop-engine-lowered": (_binlog_scop_lowered, {"engine": "scop"}),
        "x/qp/flat-held/fit_reml": (_flat_held, {"fit": "fit_reml"}),
        "x/qp/flat-held/fit": (_flat_held, {"fit": "fit"}),
        "x/qp/zero-edge/n100-k6": (_zero_edge, {"n": 100, "n_knots": 6}),
        "x/qp/zero-edge/n200-k6": (_zero_edge, {"n": 200, "n_knots": 6}),
        "x/qp/zero-edge/n200-k8": (_zero_edge, {"n": 200, "n_knots": 8}),
        "x/qp/unbounded/first-solve": (_qp_unbounded, {"raise_on": 1, "link": "identity"}),
        "x/qp/unbounded/second-solve": (_qp_unbounded, {"raise_on": 2, "link": "identity"}),
        "x/qp/unbounded/log-link": (_qp_unbounded, {"raise_on": 2, "link": "log"}),
        "x/qp/unbounded/infeasible-start": (_qp_unbounded_infeasible_start, {}),
        "x/qp/infeasible-start/no-feasible-proposal": (_qp_no_feasible_proposal, {}),
        "x/qp/infeasible-start/outside-link-domain": (_qp_outside_link_domain, {}),
        "x/qp/rejected-step/active-set": (_qp_rejected_second_step, {}),
        "x/separation/logit-numeric": (_separation, {"separation": "warn"}),
        "x/separation/logit-numeric-error": (_separation, {"separation": "error"}),
        "x/binlog/scop-lowered": (_binlog_scop_lowered, {}),
        "x/binlog/qp-lowered/gram": (_binlog_qp_lowered, {"direct_solve": "gram"}),
        "x/binlog/qp-lowered/qr": (_binlog_qp_lowered, {"direct_solve": "qr"}),
        "x/binlog/collinear-truncated/gram": (_binlog_collinear, {"direct_solve": "gram"}),
        "x/binlog/collinear-truncated/qr": (_binlog_collinear, {"direct_solve": "qr"}),
        "x/binlog/light-cut/refused-gram": (_light_cut, {"offset": -20.0, "responses": (0.0, 1.0),
                                                          "direct_solve": "gram"}),
        "x/binlog/light-cut/refused-qr": (_light_cut, {"offset": -20.0, "responses": (0.0, 1.0),
                                                        "direct_solve": "qr"}),
        "x/binlog/light-cut/certified": (_light_cut, {"offset": 1.3, "responses": (0.0, 1.0),
                                                       "direct_solve": "gram"}),
        "x/binlog/light-cut/all-events-gram": (_light_cut, {"offset": -20.0,
                                                             "responses": (1.0, 1.0),
                                                             "direct_solve": "gram"}),
        "x/binlog/light-cut/all-events-qr": (_light_cut, {"offset": -20.0, "responses": (1.0, 1.0),
                                                           "direct_solve": "qr"}),
        "x/binlog/light-cut/no-events": (_light_cut, {"offset": -20.0, "responses": (0.0, 0.0),
                                                       "direct_solve": "gram"}),
        "x/stagnation/separated-level": (_stalling, {}),
        "x/fixed-point/gaussian-exact": (_fixed_point, {}),
        "x/input/ndarray-design": (_ndarray_design, {}),
        "x/input/refusals": (_refusals, {}),
    }  # fmt: skip
    cases.update(specials)
    for name in TRACED:
        builder, kwargs = cases[name]
        cases[f"t/{name}"] = (builder, kwargs)
    return cases


def _case_order(name: str) -> tuple[int, str]:
    """Slow kinds first, so the pool's tail is short."""
    slow = ("scop", "/fs/", "/sz/", "/re/", "nested", "t/")
    return (0 if any(token in name for token in slow) else 1, name)


def _case_id(name: str) -> str:
    return name.replace("/", "__")


# --------------------------------------------------------------------------- case builders


def _model_outputs(model, test, test_offset, fit: str) -> dict:
    import numpy as np

    from superglm.model.state_ops import group_edf

    mu = model.predict(test) if test_offset is None else model.predict(test, offset=test_offset)
    out = {
        "result": model.result,
        "predict": np.asarray(mu, dtype=np.float64),
        "group_edf": group_edf(model),
    }
    if fit == "fit_reml":
        out["reml_diagnostics"] = model.reml_diagnostics()
    return out


def _matrix_case(family: str, link: str, mode: str, route: str, fit: str) -> dict:
    """``release_differential``'s small fit for one family, link and term kind."""
    import numpy as np
    import release_differential as rd

    from superglm import SuperGLM

    term, direct_solve = MODES[mode]
    n = SMALL_N
    seed = 1000 + 17 * rd.GLM_FAMLINKS.index((family, link)) + 1
    frame, eta, rng = rd.make_data(n, seed)
    y = rd.glm_response(family, link, eta, frame, rng, False)
    features, interactions = rd.glm_terms(term)
    if mode == "scopeng":
        from superglm import Constraint, PSpline

        increasing = Constraint.fit.increasing
        features = {
            **features,
            "x1": PSpline(n_knots=8, constraint=increasing, shape_engine="scop"),
        }
    model = SuperGLM(
        family=rd.glm_family(family),
        link=link,
        features=features,
        interactions=interactions,
        discrete=route == "binned",
        direct_solve=direct_solve,
    )
    kwargs, offset = {}, None
    if family in ("poisson", "nb2fixed", "nb2auto") and link == "log":
        offset = np.log(frame["expo"].to_numpy())
        kwargs["offset"] = offset[:n]
    if family.startswith("tweedie"):
        kwargs["sample_weight"] = frame["expo"].to_numpy()[:n]
    getattr(model, fit)(frame.iloc[:n], y[:n], **kwargs)
    return _model_outputs(model, frame.iloc[n:], None if offset is None else offset[n:], fit)


def _solver():
    """The hooked ``fit_irls_direct``, read at call time."""
    from superglm.solvers import irls_direct

    return irls_direct.fit_irls_direct


def _msn_event_rows(constrained: bool) -> dict:
    """An event row at offset -25 beside 200 rows at +2: Newton steps on dense or QP."""
    import numpy as np

    from superglm.distributions import Binomial
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice, LinearConstraintSet

    rng = np.random.default_rng(0)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.3)
    y = np.append((rng.uniform(size=200) < 0.2).astype(np.float64), 1.0)
    offset = np.append(np.full(200, 2.0), -25.0)
    constraints = LinearConstraintSet(A=np.eye(1), b=np.zeros(1)) if constrained else None
    groups = [GroupSlice("x", 0, 1, constraints=constraints,
                         monotone_engine="qp" if constrained else None)]  # fmt: skip
    result, inverse = _solver()(
        DesignMatrix([DenseGroupMatrix(x[:, None])], n=201, p=1),
        y,
        np.ones(201),
        Binomial(),
        LogLink(),
        groups,
        lambda2=0.0,
        offset=offset,
        weight_semantics="frequency",
        record_diagnostics=True,
    )
    return {"result": result, "inverse": inverse}


def _msn_lowered(convergence: str) -> dict:
    """Seed 1285 of #437's random sweep: a lowered start whose stop Newton steps certify."""
    import numpy as np
    import pandas as pd

    from superglm import Numeric, SuperGLM

    rng = np.random.default_rng(1285)
    n = int(rng.choice([60, 150, 400, 1500]))
    k = int(rng.choice([1, 2, 3]))
    x = rng.uniform(-1.0, 1.0, (n, k))
    slopes = rng.normal(0.0, 0.6, k)
    top = rng.choice([-0.3, -0.05, -0.01, -1e-3])
    linear = x @ slopes
    eta = linear - linear.max() + top
    y = (rng.uniform(size=n) < np.exp(eta)).astype(np.float64)
    rng.uniform()
    offset = rng.uniform(-0.5, 1.0, n)
    columns = {f"x{j}": x[:, j] for j in range(k)}
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        convergence=convergence,
        features={name: Numeric() for name in columns},
    )
    frame = pd.DataFrame(columns)
    model.fit(frame, y, offset=offset)
    return _model_outputs(model, frame, offset, "fit")


def _thin_events(direct_solve: str) -> dict:
    """A random effect beside a slope, with four one-row event levels (REML, structured or gram)."""
    import numpy as np
    import pandas as pd

    from superglm import Numeric, RandomEffect, SuperGLM

    rng = np.random.default_rng(7)
    n, levels = 1500, 60
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    eta = -2.3 + 0.4 * x + rng.normal(0.0, 0.3, levels)[g]
    y = (rng.uniform(size=n) < np.exp(eta)).astype(float)
    extra = pd.DataFrame({"x": rng.uniform(size=4), "g": [f"s{j}" for j in range(4)]})
    frame = pd.concat(
        [pd.DataFrame({"x": x, "g": [f"g{c:03d}" for c in g]}), extra], ignore_index=True
    )
    y = np.concatenate([y, np.ones(4)])
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        features={"x": Numeric(), "g": RandomEffect()},
    )
    model.fit_reml(frame, y)
    return _model_outputs(model, frame, None, "fit_reml")


def _binlog_offset(offset: float, plain: bool) -> dict:
    """An interior binomial/log optimum under a constant offset (a lowered start at +2)."""
    import numpy as np
    import pandas as pd

    from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, SuperGLM

    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-1.0, 1.0, n)
    g = np.tile(np.arange(30), 10)
    y = (rng.random(n) < 0.4).astype(np.float64)
    frame = pd.DataFrame({"x": x, "g": [f"g{c:02d}" for c in g]})
    level = (
        Categorical(base="first") if plain else RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve="gram",
        selection_penalty=0.0,
        features={"x": Numeric(), "g": level},
    )
    offsets = np.full(n, offset)
    fit = "fit" if plain else "fit_reml"
    getattr(model, fit)(frame, y, offset=offsets)
    return _model_outputs(model, frame, offsets, fit)


def _gamma_switch() -> dict:
    """Gamma/log whose Fisher scoring cannot reach the bar: it switches to Newton, then latches."""
    import numpy as np

    from superglm.distributions import Gamma
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    n, far = 200, 6
    x = np.zeros(n)
    x[:far] = 8.0
    x[far:] = np.linspace(-1.0, 1.0, n - far)
    design = np.column_stack([np.ones(n), x])
    ratio = np.ones(n)
    ratio[:far] = 0.01
    correction = np.linalg.lstsq(design[far:].T, -design[:far].T @ (ratio[:far] - 1.0), rcond=None)
    ratio[far:] = 1.0 + correction[0]
    y = ratio * np.exp(1.0 + 0.2 * x)

    def solve(latch):
        return _solver()(
            X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=0.0,
            convergence="mode_score",
            weight_semantics="frequency",
            _newton_latch=latch,
        )[0]

    latch = [False]
    return {"unlatched": solve(None), "switching": solve(latch), "latched": solve(latch)}


def _gamma_kept() -> dict:
    """A Gamma/log ``mode_score`` solve whose Fisher score reaches the bar and never switches."""
    import numpy as np

    from superglm.distributions import Gamma
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    rng = np.random.default_rng(458)
    n = 120
    X_raw = rng.normal(size=(n, 3))
    y = rng.gamma(shape=2.0, scale=np.exp(0.2 + X_raw @ np.array([0.1, -0.2, 0.3])) / 2.0)
    result, inverse = _solver()(
        X=DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3),
        y=y,
        weights=np.ones(n),
        family=Gamma(),
        link=LogLink(),
        groups=[GroupSlice(name="x", start=0, end=3)],
        lambda2=0.1,
        convergence="mode_score",
        weight_semantics="frequency",
        _newton_latch=[False],
    )
    return {"result": result, "inverse": inverse}


def _tweedie_observed() -> dict:
    """Tweedie(1.5)/log on a dense design: observed-Newton steps from the first iteration."""
    import numpy as np

    from superglm.distributions import Tweedie
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    rng = np.random.default_rng(15)
    n = 500
    X_raw = rng.normal(size=(n, 4))
    mu = np.exp(0.3 + X_raw @ np.array([0.2, -0.1, 0.15, 0.05]))
    counts = rng.poisson(mu / 2.0)
    y = np.array([rng.gamma(2.0, 1.0, k).sum() for k in counts])
    weights = rng.uniform(0.2, 1.0, n)
    out = {}
    for convergence in ("deviance", "mode_score"):
        out[convergence] = _solver()(
            X=DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=4),
            y=y,
            weights=weights,
            family=Tweedie(p=1.5),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=4)],
            lambda2=0.5,
            convergence=convergence,
            weight_semantics="prior",
        )[0]
    return out


def _nested(family: str) -> dict:
    """A region random effect nested in a country random effect: the structured nested chain."""
    import numpy as np
    import pandas as pd

    from superglm import Numeric, RandomEffect, SuperGLM

    rng = np.random.default_rng(31)
    n, regions, per_parent = 900, 24, 4
    leaf = rng.integers(0, regions, n)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "region": [f"r{code:02d}" for code in leaf],
            "country": [f"c{code // per_parent}" for code in leaf],
        }
    )
    effect = (
        rng.normal(0.0, 0.3, regions)[leaf]
        + rng.normal(0.0, 0.2, regions // per_parent)[leaf // per_parent]
    )
    mean = np.exp(0.2 * frame["x"].to_numpy() + effect)
    y = rng.poisson(mean).astype(float) if family == "poisson" else rng.gamma(3.0, mean / 3.0)
    model = SuperGLM(
        family=family,
        link="log",
        features={
            "x": Numeric(),
            "country": RandomEffect(),
            "region": RandomEffect(nested_in="country"),
        },
        selection_penalty=0,
        direct_solve="structured",
    )
    model.fit_reml(frame, y)
    return _model_outputs(model, frame, None, "fit_reml")


def _scop_variant(variant: str, route: str, engine: str | None = None) -> dict:
    """Shape-constrained P-splines beyond the matrix: compressed lone group, two groups, a translated column.

    On the QP by default (#537), on SCOP with ``engine="scop"``.
    """
    import numpy as np
    import release_differential as rd

    from superglm import Categorical, Constraint, Numeric, Spline, SuperGLM
    from superglm import PSpline as _PSpline

    def PSpline(**kwargs):  # noqa: N802
        return _PSpline(**kwargs, **({} if engine is None else {"shape_engine": engine}))

    frame, eta, rng = rd.make_data(SMALL_N, 77)
    frame = frame.copy()
    increasing, decreasing = Constraint.fit.increasing, Constraint.fit.decreasing
    family, link = "poisson", "log"
    if variant == "repeated":
        # one SCOP group whose rows repeat: its exact support is compressed
        frame["x1"] = np.round(frame["x1"].to_numpy() * 2.0) / 2.0
        features = {"x1": PSpline(n_knots=8, constraint=increasing), "c1": Categorical()}
    elif variant in ("two", "binlog"):
        features = {
            "x1": PSpline(n_knots=8, constraint=increasing),
            "noise": PSpline(n_knots=6, constraint=decreasing),
            "x2": Spline(kind="cr", k=6),
        }
        if variant == "binlog":
            family = "binomial"
    else:
        # an ordinary column at a large translation: the centred trial predictor
        frame["shifted"] = frame["lin"].to_numpy() + 1.0e4
        features = {
            "x1": PSpline(n_knots=8, constraint=increasing),
            "shifted": Numeric(),
            "c1": Categorical(),
        }
    y = rd.glm_response(family, link, eta, frame, rng, False)
    model = SuperGLM(
        family=rd.glm_family(family), link=link, features=features, discrete=route == "binned"
    )
    n = SMALL_N
    kwargs, offset = {}, None
    if family == "poisson":
        offset = np.log(frame["expo"].to_numpy())
        kwargs["offset"] = offset[:n]
    model.fit_reml(frame.iloc[:n], y[:n], **kwargs)
    return _model_outputs(model, frame.iloc[n:], None if offset is None else offset[n:], "fit_reml")


def _separation(separation: str) -> dict:
    """A numeric column that separates a binomial/logit response: the runtime backstop."""
    import numpy as np
    import pandas as pd

    from superglm import Numeric, SuperGLM

    rng = np.random.default_rng(41)
    n = 300
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.normal(size=n)
    y = (x > 0.1).astype(np.float64)
    frame = pd.DataFrame({"x": x, "z": z})
    model = SuperGLM(
        family="binomial",
        link="logit",
        selection_penalty=0.0,
        separation=separation,
        features={"x": Numeric(), "z": Numeric()},
    )
    model.fit(frame, y)
    return _model_outputs(model, frame, None, "fit")


def _binlog_scop_lowered(engine: str | None = None) -> dict:
    """An increasing PSpline under offset +2: on the QP, or on SCOP (``engine="scop"``) in latent coordinates."""
    import numpy as np
    import pandas as pd

    from superglm import Constraint, PSpline, SuperGLM

    rng = np.random.default_rng(8)
    x = rng.uniform(0.0, 1.0, 60)
    y = (rng.uniform(size=60) < 0.1 + 0.3 * x).astype(np.float64)
    shape = {} if engine is None else {"shape_engine": engine}
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=5, constraint=Constraint.fit.increasing, **shape)},
    )
    frame, offset = pd.DataFrame({"x": x}), np.full(60, 2.0)
    model.fit(frame, y, offset=offset)
    return _model_outputs(model, frame, offset, "fit")


def _flat_held(fit: str) -> dict:
    """An increasing P-spline on a decreasing signal beside a fitted ``cr`` term: the QP holds it flat.

    Its binding rows (``RankInfo.face_rows``) and active face carry the fit's
    EDF and covariance (#537, #538, #540).
    """
    import numpy as np
    import pandas as pd

    from superglm import Constraint, PSpline, Spline, SuperGLM

    rng = np.random.default_rng(12)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.uniform(0.0, 1.0, n)})
    mean = np.exp(0.4 - 0.8 * frame["x"].to_numpy() + np.sin(3.0 * frame["z"].to_numpy()))
    y = rng.poisson(mean).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        link="log",
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing),
            "z": Spline(kind="cr", k=8),
        },
    )
    getattr(model, fit)(frame, y)
    return _model_outputs(model, frame, None, fit)


def _zero_edge(n: int, n_knots: int) -> dict:
    """Poisson/identity, two events in the last tenth: an increasing mean at zero on most rows.

    There the working QP can go unbounded (``H is rank-deficient``), and the
    fit returns its last accepted state, unconverged (#537).
    """
    import numpy as np
    import pandas as pd

    from superglm import Constraint, PSpline, SuperGLM

    x = np.linspace(0.0, 1.0, n)
    y = np.zeros(n)
    y[[90 * n // 100, 96 * n // 100]] = 1.0
    model = SuperGLM(
        family="poisson",
        link="identity",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=n_knots, constraint=Constraint.fit.increasing)},
    )
    frame = pd.DataFrame({"x": x})
    model.fit(frame, y)
    return _model_outputs(model, frame, None, "fit")


def _replace_everywhere(original, replacement) -> None:
    """``replacement`` for ``original`` in every module that binds it (``_rebind``).

    The module layout and the package layout bind the QP solve in different
    modules; replaced by identity, the fixture is the same on both.
    """
    if not _rebind(original, replacement):
        raise RuntimeError(f"no module binds {original!r}")


def _qp_solve():
    from superglm.solvers.irls_direct import _solve_constrained_qp_with_cold_retry

    return _solve_constrained_qp_with_cold_retry


def _qp_unbounded(raise_on: int, link: str) -> dict:
    """The working QP raises on solve ``raise_on``: under the identity link the fit ends there.

    Disclosed with a ``ConvergenceWarning``; under the log link, whose domain
    has no edge, the error is raised (#537).
    """
    import numpy as np
    import pandas as pd

    from superglm import Constraint, PSpline, SuperGLM

    real_qp = _qp_solve()
    calls: list[int] = []

    def unbounded(*args, **kwargs):
        calls.append(1)
        if len(calls) == raise_on:
            raise ValueError("solve_constrained_qp: H is rank-deficient")
        return real_qp(*args, **kwargs)

    _replace_everywhere(real_qp, unbounded)
    x = np.linspace(0.0, 1.0, 80)
    y = np.random.default_rng(0).poisson(1.0 + 2.0 * x).astype(float)
    model = SuperGLM(
        family="poisson",
        link=link,
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    )
    frame = pd.DataFrame({"x": x})
    model.fit(frame, y)
    return {**_model_outputs(model, frame, None, "fit"), "qp_calls": len(calls)}


def _bspline_increasing_design(family: str, link: str, y):
    """An increasing ``bs`` term's design, response, weights, offset and groups."""
    import numpy as np
    import pandas as pd

    from superglm import BSplineSmooth, Constraint, SuperGLM
    from superglm.model.base import model_build_design_matrix

    x = np.linspace(0.0, 1.0, 80)
    model = SuperGLM(
        family=family,
        link=link,
        selection_penalty=0.0,
        features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    response = y(x)
    y_fit, weights, offset = model_build_design_matrix(
        model, pd.DataFrame({"x": x}), response, np.ones_like(response), None
    )
    return model, y_fit, weights, offset


def _qp_unbounded_infeasible_start() -> dict:
    """Every working QP raises, from an infeasible start: phase one, then the zero-edge stop."""
    import numpy as np

    def unbounded(*args, **kwargs):
        raise ValueError("solve_constrained_qp: H is rank-deficient")

    _replace_everywhere(_qp_solve(), unbounded)
    model, y_fit, weights, offset = _bspline_increasing_design(
        "poisson",
        "identity",
        lambda x: np.random.default_rng(0).poisson(1.0 + 2.0 * x).astype(float),
    )
    group = model._groups[0]
    infeasible_beta = np.zeros(model._dm.p)
    infeasible_beta[group.sl] = -0.1 * group.constraints.A[0]
    result, inverse = _solver()(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x": 1.0},
        offset=offset,
        beta_init=infeasible_beta,
        intercept_init=float(np.mean(y_fit)),
        max_iter=50,
        tol=1e-8,
        weight_semantics="frequency",
    )
    return {"result": result, "inverse": inverse}


def _qp_no_feasible_proposal() -> dict:
    """The first working QP at an infeasible start returns an infeasible proposal: phase one."""
    import numpy as np

    from superglm.solvers.constrained_qp import QPResult

    model, y_fit, weights, offset = _bspline_increasing_design(
        "gaussian", "identity", lambda x: 0.5 + 2.0 * x + 0.02 * np.sin(8.0 * x)
    )
    group = model._groups[0]
    infeasible_beta = np.zeros(model._dm.p)
    infeasible_beta[group.sl] = -10.0 * group.constraints.A[0]
    real_qp = _qp_solve()
    calls: list[int] = []

    def no_feasible_first_proposal(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return QPResult(beta=infeasible_beta.copy(), converged=False)
        return real_qp(*args, **kwargs)

    _replace_everywhere(real_qp, no_feasible_first_proposal)
    out = {}
    for max_iter in (1, 50):
        calls.clear()
        out[f"max_iter={max_iter}"] = _solver()(
            model._dm,
            y_fit,
            weights,
            model._distribution,
            model._link,
            model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            beta_init=infeasible_beta,
            intercept_init=0.0,
            max_iter=max_iter,
            tol=1e-8,
            weight_semantics="frequency",
        )[0]
    return out


def _qp_outside_link_domain() -> dict:
    """From an infeasible start, every trial rejected, a feasible QP proposal at ``mu <= 0``.

    Accepted for its feasibility, its intercept raised into the link's domain (#537).
    """
    import numpy as np

    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult
    from superglm.solvers.irls_state import _IRLSStepDecision, _select_irls_trial
    from superglm.types import GroupSlice, LinearConstraintSet

    def feasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([10.0]), converged=True)

    def reject_every_trial(**_kwargs):
        return _IRLSStepDecision(
            alpha=0.0, step_halvings=20, step_rejected=True, trials_attempted=20
        )

    _replace_everywhere(_qp_solve(), feasible_qp_proposal)
    _replace_everywhere(_select_irls_trial, reject_every_trial)
    x = np.linspace(-1.0, 1.0, 41)
    y = np.random.default_rng(0).poisson(1.5, x.size).astype(float)
    constraint = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))
    result, inverse = _solver()(
        DesignMatrix([DenseGroupMatrix(x[:, None])], n=x.size, p=1),
        y,
        np.ones(x.size),
        Poisson(),
        IdentityLink(),
        [GroupSlice("x", 0, 1, constraints=constraint, monotone_engine="qp")],
        lambda2=0.0,
        beta_init=np.array([-1.0]),
        intercept_init=3.0,
        max_iter=1,
        weight_semantics="frequency",
    )
    return {"result": result, "inverse": inverse}


def _qp_rejected_second_step() -> dict:
    """An increasing P-spline whose second working QP drops two rows active at the first.

    Its second step is rejected, so the fit returns the first iterate, whose
    statistics are on the face of the rows that bind there: the active set
    that iteration started from, not the rejected proposal's.
    """
    import numpy as np
    import pandas as pd

    from superglm import Constraint, PSpline, Spline, SuperGLM
    from superglm.solvers.irls_state import _IRLSStepDecision, _select_irls_trial

    calls: list[int] = []

    def reject_the_second(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            return _IRLSStepDecision(
                alpha=0.0, step_halvings=20, step_rejected=True, trials_attempted=20
            )
        return _select_irls_trial(**kwargs)

    _replace_everywhere(_select_irls_trial, reject_the_second)
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)})
    signal = 0.6 * np.minimum(frame["x"] * 3, 1.0) + 0.5 * np.sin(4 * frame["z"])
    y = rng.poisson(np.exp(signal)).astype(float)
    model = SuperGLM(
        family="poisson",
        link="log",
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing),
            "z": Spline(kind="cr", k=8),
        },
    )
    model.fit(frame, y)
    return {**_model_outputs(model, frame, None, "fit"), "line_searches": len(calls)}


def _binlog_qp_lowered(direct_solve: str) -> dict:
    """Non-negative step increments with two bounds active, at offsets 0 and +1.5 (NNLS KKT)."""
    import numpy as np

    from superglm.distributions import Binomial
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice, LinearConstraintSet

    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < 0.1 + 0.3 * np.sin(np.pi * x)).astype(np.float64)
    steps = (x[:, None] > np.array([0.2, 0.4, 0.6, 0.8])[None, :]).astype(np.float64)
    k = steps.shape[1]
    out = {}
    for shift in (0.0, 1.5):
        groups = [GroupSlice("steps", 0, k, monotone_engine="qp",
                             constraints=LinearConstraintSet(A=np.eye(k), b=np.zeros(k)))]  # fmt: skip
        out[f"shift={shift}"] = _solver()(
            DesignMatrix([DenseGroupMatrix(steps)], n=n, p=k),
            y,
            np.ones(n),
            Binomial(),
            LogLink(),
            groups,
            lambda2=0.0,
            offset=np.full(n, shift),
            direct_solve=direct_solve,
            weight_semantics="frequency",
        )[0]
    return out


def _binlog_collinear(direct_solve: str) -> dict:
    """Two columns 1e-9 apart under offset +3: a truncated direction the certificate judges."""
    import numpy as np
    import pandas as pd

    from superglm import Numeric, SuperGLM

    rng = np.random.default_rng(11)
    n = 3000
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = x1 + 1e-9 * rng.normal(size=n)
    y = (rng.uniform(size=n) < np.exp(-3.0 + 0.8 * x1)).astype(np.float64)
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"x1": Numeric(), "x2": Numeric()},
    )
    frame, offset = pd.DataFrame({"x1": x1, "x2": x2}), np.full(n, 3.0)
    model.fit(frame, y, offset=offset)
    return _model_outputs(model, frame, offset, "fit")


def _light_cut(offset: float, responses: tuple, direct_solve: str) -> dict:
    """Two crossed 2 x 2 blocks of weight 1e8 joined by two cells of weight 1e-8 at ``offset``."""
    import pandas as pd

    from superglm import Categorical, SuperGLM

    heavy = [(a, b) for a in ("a0", "a1") for b in ("b0", "b1")]
    heavy += [(a, b) for a in ("a2", "a3") for b in ("b2", "b3")]
    rows = [(a, b, response, 1e8, 1.3) for a, b in heavy for response in (0.0, 1.0)]
    rows += [(a, b, r, 1e-8, offset) for a, b in [("a1", "b2"), ("a1", "b3")] for r in responses]
    frame = pd.DataFrame(rows, columns=["A", "B", "y", "w", "off"])
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"A": Categorical(base="a0"), "B": Categorical(base="b0")},
    )
    offsets = frame["off"].to_numpy()
    model.fit(frame[["A", "B"]], frame["y"].to_numpy(), sample_weight=frame["w"].to_numpy(),
              offset=offsets)  # fmt: skip
    out = _model_outputs(model, frame[["A", "B"]], offsets, "fit")
    out["diagnostics"] = model.diagnostics()["_model"]
    return out


def _stalling() -> dict:
    """A binomial level with a separated row at lambda 1e-7 and weights 1e4: the score stalls."""
    import numpy as np
    import pandas as pd

    from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM

    rng = np.random.default_rng(5)
    n, big = 1500, 35
    popularity = rng.gamma(1.5, size=big)
    g = np.concatenate(
        [rng.choice(big, size=n - 5, p=popularity / popularity.sum()), np.arange(big, 40)]
    )
    g = g[rng.permutation(n)]
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "u": rng.uniform(size=n),
            "cat": np.array([f"c{c}" for c in rng.integers(0, 8, n)], dtype=object),
            "g": np.array([f"g{c:02d}" for c in g], dtype=object),
        }
    )
    eta = 0.1 + 0.3 * frame["x1"].to_numpy() + 0.4 * np.sin(4 * frame["u"].to_numpy())
    eta = eta + rng.normal(0, 0.3, 40)[g]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    levels = [f"g{c:02d}" for c in range(40)]
    model = SuperGLM(
        family="binomial",
        features={
            "x1": Numeric(),
            "u": Spline(kind="ps", k=8),
            "cat": Categorical(),
            "g": RandomEffect(levels=levels, lambda_policy=LambdaPolicy.fixed(1e-7)),
        },
        selection_penalty=0,
    )
    model.fit_reml(frame, y, sample_weight=np.full(n, 1e4))
    return _model_outputs(model, frame, None, "fit_reml")


def _fixed_point() -> dict:
    """A Gaussian/identity response the model fits exactly: the float64 fixed-point stop."""
    import numpy as np

    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.types import GroupSlice

    rng = np.random.default_rng(3)
    n = 200
    X_raw = rng.uniform(-1.0, 1.0, (n, 2))
    y = 1.0 + X_raw @ np.array([2.0, -0.5])
    out = {}
    for name, lambda2 in (("unpenalized", 0.0), ("penalized", 1e-3)):
        out[name] = _solver()(
            X=DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2),
            y=y,
            weights=np.ones(n),
            family=Gaussian(),
            link=IdentityLink(),
            groups=[GroupSlice(name="x", start=0, end=2)],
            lambda2=lambda2,
            convergence="mode_score",
            weight_semantics="frequency",
        )[0]
    return out


def _ndarray_design() -> dict:
    """A plain ndarray design, wrapped by the solver itself."""
    import numpy as np

    from superglm.distributions import Poisson
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    rng = np.random.default_rng(9)
    n = 300
    X_raw = rng.normal(size=(n, 3))
    y = rng.poisson(np.exp(0.2 + X_raw @ np.array([0.3, -0.2, 0.1]))).astype(float)
    groups = [GroupSlice(name="a", start=0, end=2), GroupSlice(name="b", start=2, end=3)]
    result, inverse = _solver()(
        X_raw, y, np.ones(n), Poisson(), LogLink(), groups, lambda2={"a": 0.5, "b": 2.0},
        weight_semantics="frequency",
    )  # fmt: skip
    return {"result": result, "inverse": inverse}


def _refusals() -> dict:
    """Inputs the solver refuses, each attempt recorded with its message."""
    import numpy as np

    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.types import GroupSlice

    rng = np.random.default_rng(10)
    n = 50
    X_raw = rng.normal(size=(n, 2))
    y = rng.poisson(1.0, n).astype(float)
    base = {
        "X": DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2),
        "y": y,
        "weights": np.ones(n),
        "family": Poisson(),
        "link": LogLink(),
        "groups": [GroupSlice(name="x", start=0, end=2)],
        "lambda2": 1.0,
        "weight_semantics": "frequency",
    }
    attempts = {
        "max_iter_zero": {"max_iter": 0},
        "weights_shape": {"weights": np.ones(n - 1)},
        "weights_negative": {"weights": np.r_[-1.0, np.ones(n - 1)]},
        "weights_zero": {"weights": np.zeros(n)},
        "deferred_with_rank_info": {"_defer_rank_info": []},
        "working_system_with_rank_info": {"_return_working_system": True},
        "no_statistics_with_rank_info": {"_compute_fit_statistics": False},
        "no_geometry_with_statistics": {"_compute_reml_geometry": False,
                                        "compute_rank_info": False},
        "structured_without_components": {"direct_solve": "structured"},
    }  # fmt: skip
    out = {}
    for name, change in attempts.items():
        try:
            _solver()(**{**base, **change})
            out[name] = "accepted"
        except Exception as exc:
            out[name] = f"{type(exc).__qualname__}: {exc}"
    return out


# --------------------------------------------------------------------------- digests


_SKIPPED: set[str] = set()
_CACHE_ATTRIBUTE = re.compile(r"cache|memo", re.IGNORECASE)


def _skip(name: str) -> bool:
    if _SKIPPED_KEY.search(name):
        _SKIPPED.add(name)
        return True
    return False


def _hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=10).hexdigest()


def _array_digest(values) -> object:
    import numpy as np

    if values.dtype == object:
        return [_digest(item) for item in values.ravel().tolist()]
    contiguous = np.ascontiguousarray(values)
    return f"a:{values.dtype.str}:{list(values.shape)}:{_hash(contiguous.tobytes())}"


def _stop_summary(obj) -> str | None:
    """Inputs, layouts derived from the design and runtime machinery are named, not walked."""
    module = type(obj).__module__ or ""
    name = type(obj).__qualname__
    if module.startswith(("superglm.group_matrix", "superglm._group_matrix")):
        return f"<{name} {list(getattr(obj, 'shape', ()) or ())}>"
    if module.startswith("superglm.distributions"):
        params = {key: getattr(obj, key) for key in ("theta", "p", "power") if hasattr(obj, key)}
        parts = ",".join(
            f"{key}={value.hex() if isinstance(value, float) else value}"
            for key, value in sorted(params.items())
        )
        return f"<{name} {parts}>"
    if module.startswith("superglm.links") or name.endswith("Layout"):
        return f"<{name}>"
    if module.startswith(
        ("superglm._fit_trace", "superglm.model.reml_debug", "superglm.model.api")
    ):
        return f"<{name}>"
    if module.startswith(("scipy", "numba", "threading", "weakref", "concurrent", "pandas")):
        return f"<{name}>"
    return None


def _digest(obj, depth: int = 0, active: set | None = None) -> object:
    """A JSON tree of ``obj`` whose leaves pin it bitwise (floats as ``float.hex``)."""
    import numpy as np

    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return "f:" + obj.hex()
    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return f"f{obj.dtype.str}:{float(obj).hex()}"
        return f"g:{obj.dtype.str}:{_hash(np.asarray(obj).tobytes())}"
    if isinstance(obj, np.ndarray):
        return _array_digest(obj)
    if isinstance(obj, bytes | bytearray):
        return "b:" + _hash(bytes(obj))
    if depth > 12:
        return f"<deep {type(obj).__qualname__}>"
    active = set() if active is None else active
    key = id(obj)
    if key in active:
        return f"<cycle {type(obj).__qualname__}>"
    active.add(key)
    try:
        if isinstance(obj, dict):
            items = sorted(((str(k), v) for k, v in obj.items()), key=lambda item: item[0])
            return {k: _digest(v, depth + 1, active) for k, v in items if not _skip(k)} or "{}"
        if isinstance(obj, list | tuple):
            return [_digest(item, depth + 1, active) for item in obj] or "[]"
        if isinstance(obj, set | frozenset):
            return sorted((_digest(item, depth + 1, active) for item in obj), key=repr) or "set()"
        if isinstance(obj, types.FunctionType | types.MethodType | types.BuiltinFunctionType):
            return f"<callable {getattr(obj, '__qualname__', type(obj).__qualname__)}>"
        if isinstance(obj, functools.partial):
            return f"<partial {getattr(obj.func, '__qualname__', '?')}>"
        if isinstance(obj, type | types.ModuleType):
            return f"<{getattr(obj, '__qualname__', getattr(obj, '__name__', '?'))}>"
        stop = _stop_summary(obj)
        if stop is not None:
            return stop
        return _digest_object(obj, depth, active)
    finally:
        active.discard(key)


def _digest_object(obj, depth: int, active: set) -> object:
    cls = type(obj)
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = list(getattr(obj, "__dict__", {}))
        for klass in cls.__mro__:
            slots = klass.__dict__.get("__slots__", ())
            names.extend([slots] if isinstance(slots, str) else list(slots))
    out: dict = {"__type__": cls.__qualname__}
    for name in sorted(set(names)):
        if name in ("__dict__", "__weakref__") or _skip(name):
            continue
        # an object's caches and memos (some keyed by ``id()``) are not its value
        if _CACHE_ATTRIBUTE.search(name):
            _SKIPPED.add(f"{cls.__qualname__}.{name}")
            continue
        # a functools.cached_property value is a cache, filled whenever someone read it
        if isinstance(inspect.getattr_static(cls, name, None), functools.cached_property):
            continue
        if not hasattr(obj, name):
            continue
        out[name] = _digest(getattr(obj, name), depth + 1, active)
    if len(out) == 1 and callable(obj):
        return f"<callable {cls.__qualname__}>"
    return out


def _flatten(tree, prefix: str, out: dict) -> None:
    if isinstance(tree, dict):
        for key, value in tree.items():
            _flatten(value, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            _flatten(value, f"{prefix}[{index}]", out)
    else:
        out[prefix] = tree if isinstance(tree, str) else json.dumps(tree)


def _leaves_hash(leaves: dict) -> str:
    text = "\n".join(f"{path}={value}" for path, value in sorted(leaves.items()))
    return _hash(text.encode())


# --------------------------------------------------------------------------- recording


class Recorder:
    """Every ``fit_irls_direct`` call and constrained-QP solve of one case, in call order."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.outside_qp: list[dict] = []
        self.arrays: dict = {}
        self.census: Counter = Counter()
        self._open: list[dict] = []

    # The private arguments that are state shared between calls are named, not walked.
    _SHARED = (
        "profile", "cache_out", "debug_recorder", "trace_run", "_initial_data_reuse",
        "_raw_moment_policy", "_fisher_data_reuse", "_scop_run_centring", "_defer_rank_info",
    )  # fmt: skip

    def _inputs(self, arguments: dict) -> dict:
        import numpy as np

        out = {}
        for name, value in arguments.items():
            if name == "X":
                if isinstance(value, np.ndarray):
                    out[name] = _digest(value)
                else:
                    gms = getattr(value, "group_matrices", ())
                    out[name] = f"<{type(value).__qualname__} n={value.n} p={value.p} " + (
                        ",".join(type(gm).__qualname__ for gm in gms) + ">"
                    )
            elif name in self._SHARED:
                out[name] = None if value is None else f"<{type(value).__qualname__}>"
            elif name == "groups":
                out[name] = [
                    [g.name, g.start, g.end, g.monotone_engine, _digest(g.constraints), g.penalized]
                    for g in value
                ]
            else:
                out[name] = _digest(value)
        return out

    def _count(self, arguments: dict, result) -> None:
        census = self.census
        census["calls"] += 1
        census[f"backend:{result.direct_backend}"] += 1
        census[f"reason:{result.termination_reason}"] += 1
        census["iterations"] += int(result.n_iter)
        groups = arguments["groups"]
        if any(g.monotone_engine == "scop" for g in groups):
            census["scop_calls"] += 1
        if any(g.constraints is not None for g in groups):
            census["constrained_calls"] += 1
        rank_info = result.rank_info
        # a constrained fit's binding rows and active face (#537, #540)
        if rank_info is not None and getattr(rank_info, "face_rows", None) is not None:
            census["face_rows_recorded"] += 1
            census["face_rows"] += int(rank_info.face_rows.shape[0])
        if rank_info is not None and getattr(rank_info, "face", None) is not None:
            census["face_bound_calls"] += 1
        for switch in ("_return_working_system", "_working_newton"):
            if arguments[switch]:
                census[f"switch:{switch}"] += 1
        for name in ("_defer_rank_info", "_initial_data_reuse", "_fisher_data_reuse",
                     "_scop_run_centring", "_centred_init", "scop_state_init", "_newton_latch",
                     "trace_run", "debug_recorder", "cache_out", "_deviance_init"):  # fmt: skip
            if arguments[name] is not None:
                census[f"given:{name}"] += 1
        if not arguments["_compute_fit_statistics"]:
            census["switch:no_fit_statistics"] += 1
        census[f"convergence:{arguments['convergence']}"] += 1
        if result.mean_space_true_mode:
            census["mean_space_true_mode"] += 1
        if result.mean_space_boundary_rows:
            census["mean_space_boundary"] += 1
        for row in result.iteration_log or ():
            census["halvings"] += int(row.step_halvings)
            census["trials"] += int(row.trials_attempted)
            census["rejected_steps"] += int(bool(row.step_rejected))
            census[f"criterion:{row.convergence_criterion}"] += 1
            if row.used_svd_fallback:
                census["svd_fallback_iterations"] += 1

    def wrap_fit(self, original: Callable) -> Callable:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def fit_irls_direct(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            caller_diagnostics = arguments["record_diagnostics"]
            arguments["record_diagnostics"] = True
            profile = arguments["profile"] if arguments["profile"] is not None else {}
            arguments["profile"] = profile
            before = dict(profile)
            deferred = arguments["_defer_rank_info"]
            deferred_before = None if deferred is None else len(deferred)
            record: dict = {"index": len(self.calls), "inputs": self._inputs(arguments), "qp": []}
            self.calls.append(record)
            self._open.append(record)
            try:
                out = original(*bound.args, **bound.kwargs)
            except BaseException as exc:
                record["error"] = {"type": type(exc).__qualname__, "message": str(exc)}
                self.census[f"call_error:{type(exc).__qualname__}"] += 1
                raise
            finally:
                self._open.pop()
                record["profile"] = _digest(
                    {k: v for k, v in profile.items() if k not in before or before[k] != v}
                )
                for key, value in profile.items():
                    if key.startswith("irls_") and isinstance(value, int) and not _skip(key):
                        increase = value - before.get(key, 0)
                        if increase:
                            self.census[f"profile:{key}"] += increase
                latch = arguments["_newton_latch"]
                record["latch_after"] = None if latch is None else list(latch)
                record["deferred_added"] = (
                    None if deferred is None else len(deferred) - deferred_before
                )
            result = out[0]
            record["returned"] = _digest(list(out))
            if arguments["cache_out"] is not None:
                record["cache_out"] = _digest(arguments["cache_out"])
            record["published"] = self._published(arguments, result)
            self._count(arguments, result)
            if not caller_diagnostics:
                result.iteration_log = None
            return out

        return fit_irls_direct

    def _published(self, arguments: dict, result) -> object:
        """``eta``, ``mu`` and the Fisher weights every consumer reads at the result."""
        import numpy as np

        from superglm.group_matrix import DesignMatrix
        from superglm.solvers.irls_state import _evaluate_irls_state
        from superglm.solvers.mode_score import linear_predictor
        from superglm.solvers.working_rows import coefficient_working_rows

        try:
            X, y = arguments["X"], arguments["y"]
            if isinstance(X, DesignMatrix):
                dm = X
            else:
                from superglm.solvers.pirls import _wrap_dense_X

                dm = _wrap_dense_X(X, arguments["groups"])
            weights = np.asarray(arguments["weights"], dtype=np.float64)
            offset = arguments["offset"]
            offset = np.zeros(dm.n) if offset is None else np.asarray(offset, dtype=np.float64)
            eta_unclipped = linear_predictor(dm, result, offset)
            state = _evaluate_irls_state(
                dm,
                y,
                weights,
                arguments["family"],
                arguments["link"],
                offset,
                result.beta,
                result.intercept,
                eta_unclipped=eta_unclipped,
            )
            rows = coefficient_working_rows(
                distribution=arguments["family"],
                link=arguments["link"],
                y=y,
                mu=state.mu,
                eta=state.eta,
                sample_weight=weights,
                prefer_observed=False,
            )
        except Exception as exc:  # recorded, not raised: the fit itself succeeded
            return {"error": f"{type(exc).__qualname__}: {exc}"}
        index = len(self.calls) - 1
        self.arrays[f"call{index:04d}_beta"] = np.array(result.beta, dtype=np.float64)
        self.arrays["last_eta"] = np.array(state.eta_unclipped, dtype=np.float64)
        self.arrays["last_mu"] = np.array(state.mu, dtype=np.float64)
        self.arrays["last_weights"] = np.array(rows.weights, dtype=np.float64)
        return {
            "eta_unclipped": _digest(state.eta_unclipped),
            "eta": _digest(state.eta),
            "mu": _digest(state.mu),
            "deviance": _digest(float(state.deviance)),
            "fisher_weights": _digest(rows.weights),
        }

    def wrap_qp(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def solve_constrained_qp(*args, **kwargs):
            result = original(*args, **kwargs)
            entry = {"inputs": _digest([list(args), kwargs]), "result": _digest(result)}
            (self._open[-1]["qp"] if self._open else self.outside_qp).append(entry)
            self.census["qp_solves"] += 1
            if not getattr(result, "converged", True):
                self.census["qp_unconverged"] += 1
            if getattr(result, "active_set", None):
                self.census["qp_solves_with_active_rows"] += 1
            return result

        return solve_constrained_qp


def _rebind(original, replacement) -> int:
    """Point every module-level binding of ``original`` at ``replacement``."""
    count = 0
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for key, value in list(namespace.items()):
            if value is original:
                namespace[key] = replacement
                count += 1
    return count


def _install_hooks(recorder: Recorder) -> dict:
    import importlib

    for name in (
        "superglm",
        "superglm.solvers.irls_direct",
        "superglm.solvers.constrained_qp",
        "superglm.reml.direct",
        "superglm.reml.discrete",
        "superglm.reml.scop_efs",
        "superglm.model.fit_ops",
        "superglm.model.reml_execute",
        "superglm.model.reml_finalize",
    ):
        importlib.import_module(name)
    from superglm.solvers import constrained_qp, irls_direct

    fit, qp = irls_direct.fit_irls_direct, constrained_qp.solve_constrained_qp
    return {
        "fit_irls_direct": _rebind(fit, recorder.wrap_fit(fit)),
        "solve_constrained_qp": _rebind(qp, recorder.wrap_qp(qp)),
    }


def _trace_rows(debug_dir: Path) -> dict:
    """The debug run's trace and rows, without run ids, timestamps or wall time."""
    runs: dict[str, dict] = {}
    for path in sorted(debug_dir.glob("*_run.json")):
        run_id = path.name.removesuffix("_run.json")
        runs[run_id] = {"run": json.loads(path.read_text(encoding="utf-8"))}
        for rows in sorted(debug_dir.glob(f"{run_id}_*.jsonl")):
            suffix = rows.name.removeprefix(f"{run_id}_")
            lines = rows.read_text(encoding="utf-8").splitlines()
            runs[run_id][suffix] = [json.loads(line) for line in lines if line.strip()]
    # run ids are ``fit_reml_<ms>_<uuid>``: order by their clock, then name by position
    ordered = sorted(runs, key=lambda run_id: (int(run_id.split("_")[2]), run_id))
    return {f"run{k}": runs[run_id] for k, run_id in enumerate(ordered)}


# --------------------------------------------------------------------------- case process


def _child(name: str, out: Path) -> int:
    for key, value in PINNED_ENV.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"case process needs {key}={value}; run it through `run`")
    import tempfile
    import warnings

    import numpy as np

    recorder = Recorder()
    rebound = _install_hooks(recorder)
    builder, kwargs = _registry()[name]
    traced = name.startswith("t/")
    debug_dir = None
    if traced:
        import superglm._debug as debug

        debug_dir = Path(tempfile.mkdtemp(prefix="trace-", dir=out))
        os.environ["SUPERGLM_DEBUG_DIR"] = str(debug_dir)
        debug.set_debug_level(2)
    status, error, outputs = "ok", None, {}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outputs = builder(**kwargs)
        except Exception as exc:
            status, error = "error", {"type": type(exc).__qualname__, "message": str(exc)}
    elapsed = time.perf_counter() - started
    warned = Counter(f"{w.category.__qualname__}: {w.message}" for w in caught)
    tree = {
        "status": status,
        "error": error,
        "calls": recorder.calls,
        "qp_outside_calls": recorder.outside_qp,
        "model": _digest(outputs),
        "warnings": sorted(f"{count}x {text}" for text, count in warned.items()),
    }
    if debug_dir is not None:
        tree["trace"] = _digest(_trace_rows(debug_dir))
    if isinstance(outputs, dict) and isinstance(outputs.get("predict"), np.ndarray):
        recorder.arrays["model_predict"] = outputs["predict"]
    leaves: dict = {}
    _flatten(tree, "", leaves)
    call_hashes = []
    for k in range(len(recorder.calls)):
        prefix = f"calls[{k}]"
        call_hashes.append(
            _leaves_hash({p: v for p, v in leaves.items() if p.startswith(prefix + ".")})
        )
    model_leaves = {p: v for p, v in leaves.items() if not p.startswith("calls[")}
    census = dict(recorder.census)
    census["traced"] = int(traced)
    header = {
        "case": name,
        "status": status,
        "error": error,
        "case_hash": _leaves_hash(leaves),
        "model_hash": _leaves_hash(model_leaves),
        "call_hashes": call_hashes,
        "leaf_count": len(leaves),
        "census": census,
        "rebound": rebound,
        "skipped_keys": sorted(_SKIPPED),
        "elapsed_s": round(elapsed, 3),
    }
    stem = out / "cases" / _case_id(name)
    with gzip.open(f"{stem}.leaves.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(leaves, handle)
    np.savez_compressed(f"{stem}.npz", **recorder.arrays)
    Path(f"{stem}.hash.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- run, compare


def _tree_identity() -> dict:
    import importlib.util

    spec = importlib.util.find_spec("superglm")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    info: dict = {"superglm": str(origin), "python": sys.version.split()[0]}
    if origin is not None:
        root = origin.parents[2]

        def git(*args):
            done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
            return done.stdout.strip()

        info["head"] = git("rev-parse", "HEAD")
        info["dirty_src"] = bool(git("status", "--porcelain", "--", "src"))
    return info


def _run(args) -> int:
    out = Path(args.out).resolve()
    (out / "cases").mkdir(parents=True, exist_ok=True)
    names = sorted(_registry(), key=_case_order)
    if args.only:
        pattern = re.compile(args.only)
        names = [name for name in names if pattern.search(name)]
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        names = [name for k, name in enumerate(names) if k % count == index - 1]
    env = {**os.environ, **PINNED_ENV}
    command = [sys.executable]
    if args.coverage:
        (out / "coverage").mkdir(exist_ok=True)
        command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={_coverage_rc(out)}",
                    f"--data-file={out / 'coverage' / '.coverage'}"]  # fmt: skip
    command += [str(Path(__file__).resolve()), "_case"]
    started = time.perf_counter()

    def one(name: str) -> tuple[str, str, float]:
        stem = out / "cases" / _case_id(name)
        Path(f"{stem}.hash.json").unlink(missing_ok=True)
        t0 = time.perf_counter()
        try:
            done = subprocess.run(
                [*command, name, "--out", str(out)],
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            status = None if done.returncode == 0 else f"crash (exit {done.returncode})"
            detail = done.stderr[-2000:]
        except subprocess.TimeoutExpired:
            status, detail = f"timeout ({args.timeout} s)", ""
        if status is not None:
            header = {"case": name, "status": status, "error": {"type": status, "message": detail},
                      "case_hash": status, "model_hash": status, "call_hashes": [], "census": {}}  # fmt: skip
            Path(f"{stem}.hash.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
        header = json.loads(Path(f"{stem}.hash.json").read_text(encoding="utf-8"))
        return name, header["status"], time.perf_counter() - t0

    statuses: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, name) for name in names]
        for future in as_completed(futures):
            name, status, seconds = future.result()
            statuses[status] += 1
            if status != "ok" or args.verbose:
                print(f"{status:>8}  {seconds:7.1f} s  {name}", flush=True)
    wall = time.perf_counter() - started
    manifest = {
        "tree": _tree_identity(),
        "harness": _hash(Path(__file__).read_bytes()),
        "env": {key: env.get(key) for key in (*PINNED_ENV, "NUMBA_CACHE_DIR")},
        "selection": {"only": args.only, "shard": args.shard},
        "cases": {},
    }
    for name in names:
        header = json.loads((out / "cases" / f"{_case_id(name)}.hash.json").read_text())
        manifest["cases"][name] = header["case_hash"]
    tag = "all" if not args.shard else args.shard.replace("/", "of")
    (out / f"manifest-{tag}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"{len(names)} cases in {wall:.0f} s: " + ", ".join(f"{v} {k}" for k, v in statuses.items())
    )
    if args.coverage:
        _coverage_report(out)
    _print_census(out, names)
    return 0


# What ``--coverage`` reports, in every file of the solver: its entry point, and the
# fit it runs, which is a wrapper, a method-object class, its fields, recorder and
# step classes (every top-level class named ``_IRLSDirect*``), the centred
# finaliser, and the module functions it binds with ``partial``.  Its total (``the
# fit``) is everything but the entry point.
_COVERAGE_FUNCTIONS = (
    "fit_irls_direct",
    "_fit_irls_direct_once",
    "_certify_centered_factor",
    "_finalise_centred",
    "_final_centred_system",
    "_retained_rank_info",
)
_COVERAGE_CLASS_PREFIX = "_IRLSDirect"


def _line_ranges(lines: list[int]) -> str:
    ranges, start, previous = [], None, None
    for line in lines:
        if previous is None or line != previous + 1:
            if start is not None:
                ranges.append((start, previous))
            start = line
        previous = line
    if start is not None:
        ranges.append((start, previous))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def _coverage_rc(out: Path) -> Path:
    """The run's own coverage settings, measuring and reporting the solver only.

    The solver is one module on master and a package on the branch; a ``*`` in
    coverage's patterns does not cross a ``/``, so each layout has its pattern.
    Without these settings coverage reads the working directory's configuration,
    and a ``source`` there (this repository's ``pyproject.toml``) replaces the
    include.
    """
    rc = out / "coverage" / "coveragerc"
    include = "include =\n    */superglm/solvers/irls_direct.py\n    */superglm/solvers/irls_direct/*.py\n"
    rc.write_text(f"[run]\n{include}[report]\n{include}", encoding="utf-8")
    return rc


def _coverage_report(out: Path) -> None:
    import ast

    data, rc = out / "coverage" / ".coverage", f"--rcfile={_coverage_rc(out)}"
    subprocess.run([sys.executable, "-m", "coverage", "combine", rc, f"--data-file={data}"],
                   cwd=out / "coverage", capture_output=True, check=False)  # fmt: skip
    report = out / "coverage" / "coverage.json"
    subprocess.run([sys.executable, "-m", "coverage", "json", rc, f"--data-file={data}", "-o",
                    str(report)], capture_output=True, check=False)  # fmt: skip
    if not report.exists():
        print("coverage: no data")
        return
    files = json.loads(report.read_text())["files"]
    fit_hit = fit_statements = 0
    for path, entry in sorted(files.items()):
        executed = set(entry["executed_lines"])
        missing = set(entry["missing_lines"])
        tree = ast.parse(Path(path).read_text())
        for node in tree.body:
            wanted = (isinstance(node, ast.FunctionDef) and node.name in _COVERAGE_FUNCTIONS) or (
                isinstance(node, ast.ClassDef) and node.name.startswith(_COVERAGE_CLASS_PREFIX)
            )
            if not wanted:
                continue
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            span = set(range(first, node.end_lineno + 1))
            hit, miss = len(executed & span), sorted(missing & span)
            if node.name != "fit_irls_direct":
                fit_hit, fit_statements = fit_hit + hit, fit_statements + hit + len(miss)
            share = hit / max(hit + len(miss), 1)
            print(f"coverage {node.name}: {hit}/{hit + len(miss)} statements ({share:.1%})")
            if miss:
                print("  missed: " + _line_ranges(miss))
        print(f"coverage {Path(path).name}: {entry['summary']['percent_covered']:.1f}%")
    share = fit_hit / max(fit_statements, 1)
    print(f"coverage the fit: {fit_hit}/{fit_statements} statements ({share:.1%})")


def _headers(directory: Path) -> dict[str, dict]:
    headers = {}
    for path in sorted((directory / "cases").glob("*.hash.json")):
        header = json.loads(path.read_text(encoding="utf-8"))
        headers[header["case"]] = header
    return headers


def _print_census(directory: Path, names: list[str] | None = None) -> None:
    headers = _headers(directory)
    if names is not None:
        headers = {name: headers[name] for name in names if name in headers}
    total: Counter = Counter()
    reached: Counter = Counter()
    for header in headers.values():
        census = header.get("census", {})
        total.update({key: value for key, value in census.items() if isinstance(value, int)})
        reached.update(key for key, value in census.items() if value)
    print(f"census over {len(headers)} cases (key: total, cases reaching it)")
    for key in sorted(total):
        print(f"  {key:48s} {total[key]:9d} {reached[key]:5d}")
    errors = [name for name, header in headers.items() if header["status"] != "ok"]
    if errors:
        print(f"{len(errors)} cases did not finish ok (their outcome is pinned too):")
        for name in errors:
            error = headers[name].get("error") or {}
            print(f"  {headers[name]['status']}: {name}: {error.get('type')}: "
                  f"{str(error.get('message'))[:160]}")  # fmt: skip


def _leaves(directory: Path, name: str) -> dict:
    path = directory / "cases" / f"{_case_id(name)}.leaves.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _show(value: str | None) -> str:
    if value is None:
        return "<absent>"
    if value.startswith("f:"):
        return repr(float.fromhex(value[2:]))
    return value if len(value) <= 80 else value[:77] + "..."


def _compare(args) -> int:
    a_dir, b_dir = Path(args.a), Path(args.b)
    a_headers, b_headers = _headers(a_dir), _headers(b_dir)
    harnesses = set()
    for label, directory in (("A", a_dir), ("B", b_dir)):
        manifests = sorted(directory.glob("manifest-*.json"))
        if manifests:
            meta = json.loads(manifests[0].read_text())
            harnesses.add(meta.get("harness"))
            tree, env = meta.get("tree", {}), meta.get("env", {})
            print(f"{label}: {directory}  superglm {tree.get('superglm')} @ {tree.get('head')}"
                  f"{' (dirty src)' if tree.get('dirty_src') else ''}  "
                  f"PYTHONHASHSEED={env.get('PYTHONHASHSEED')}")  # fmt: skip
    if len(harnesses) > 1:
        print("warning: the two runs used different versions of this harness")
    only_a = sorted(set(a_headers) - set(b_headers))
    only_b = sorted(set(b_headers) - set(a_headers))
    common = sorted(set(a_headers) & set(b_headers))
    print(f"cases: {len(a_headers)} in A, {len(b_headers)} in B, {len(common)} in both")
    for name in only_a:
        print(f"  only in A: {name}")
    for name in only_b:
        print(f"  only in B: {name}")
    different = [n for n in common if a_headers[n]["case_hash"] != b_headers[n]["case_hash"]]
    for name in different:
        a, b = a_headers[name], b_headers[name]
        print(f"DIFFERENT {name}  (status A {a['status']}, B {b['status']})")
        a_calls, b_calls = a["call_hashes"], b["call_hashes"]
        first = next(
            (k for k, (x, y) in enumerate(zip(a_calls, b_calls, strict=False)) if x != y),
            None if len(a_calls) == len(b_calls) else min(len(a_calls), len(b_calls)),
        )
        print(f"  calls: A {len(a_calls)}, B {len(b_calls)}; first differing call: {first}")
        a_leaves, b_leaves = _leaves(a_dir, name), _leaves(b_dir, name)
        prefix = "" if first is None else f"calls[{first}]."
        paths = sorted(
            path
            for path in set(a_leaves) | set(b_leaves)
            if path.startswith(prefix) and a_leaves.get(path) != b_leaves.get(path)
        )
        print(f"  {len(paths)} values differ{' in that call' if prefix else ''}; first ones:")
        for path in paths[: args.show]:
            print(f"    {path}: A={_show(a_leaves.get(path))}  B={_show(b_leaves.get(path))}")
    identical = len(common) - len(different)
    print(f"identical: {identical} of {len(common)} common cases")
    return 0 if not different and not only_a and not only_b else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--jobs", type=int, default=8)
    run.add_argument("--only")
    run.add_argument("--shard", help="I/K: the I-th of K interleaved shards, from 1")
    run.add_argument("--timeout", type=int, default=CASE_TIMEOUT_S)
    run.add_argument("--coverage", action="store_true")
    run.add_argument("--verbose", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--show", type=int, default=12)
    census = sub.add_parser("census")
    census.add_argument("directory")
    case = sub.add_parser("_case")
    case.add_argument("name")
    case.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name in sorted(_registry(), key=_case_order):
            print(name)
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "census":
        _print_census(Path(args.directory))
        return 0
    return _child(args.name, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
