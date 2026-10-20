"""Release differential: the same seeded fits on the last release and on the candidate.

A fit the last release converged to a sensible answer must not be refused, left
unconverged or made worse on the candidate.  This runs one seeded corpus in two
environments and flags every such case.  It needs only superglm's public API,
so it runs unchanged on any release from v0.40.0.

    run --out FILE [--only REGEX] [--subset] [--shard I/K]
                     fit the corpus in this process, one JSON line per case
    release REF --out DIR [--subset]
                     run the corpus at REF (a detached worktree with its own
                     environment, shared with ``books.py``) and here, then compare
    compare BASE CANDIDATE RELEASE
                     flag candidate outcomes worse than the base; exit 1 on any flag

The corpus: SuperGLM over every built-in family and link, nine term kinds
(mains, random effect, tensor pair, spline-by-category, ``fs`` and ``sz`` factor
smooths, SCOP and QP monotone splines, ``select=True``), exact and binned, by
``fit_reml`` and ``fit``, at three sizes (400 rows, 4,000 rows, 4,000 sparse
rows); and every SuperLSS family, two predictor specs, exact and binned, both
outer loops, at two sizes.  Each record holds the status (converged, not
converged, error), the objective, the EDF and held-out deviance, NLL, Gini and
A/E on 2,000 rows from the same process.  The ``hist/`` cases reproduce every
fit the release-history sweep (v0.12.0 to v0.41) found converging on one
release and broken on a later one (``HISTORY``).  Run it through benchboard
(``benchboard heavy <label> -- ...``); pass ``--shard`` to spread it over cores.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_N = 2000

GLM_FAMLINKS = [
    ("poisson", "log"), ("poisson", "sqrt"), ("poisson", "identity"),
    ("gamma", "log"), ("gamma", "inverse"), ("gamma", "identity"),
    ("gaussian", "identity"), ("gaussian", "log"),
    ("binomial", "logit"), ("binomial", "probit"), ("binomial", "cloglog"),
    ("binomial", "log"), ("binomial", "cauchit"),
    ("tweedie15", "log"), ("tweedie12", "log"), ("nb2fixed", "log"), ("nb2auto", "log"),
]  # fmt: skip
COUNT_FAMILIES = {"poisson", "binomial", "tweedie15", "tweedie12", "nb2fixed", "nb2auto"}
TERMS = ["mains", "re", "tensor", "splcat", "fs", "sz", "scop", "qp", "select"]
SIZES = {"small": (400, False), "mid": (4000, False), "sparse": (4000, True)}
LSS_FAMILIES = [
    "GaussianLS", "GammaLS", "LogNormalLS", "NegativeBinomialLS", "TweedieLSS",
    "GeneralizedGammaLSS", "GeneralizedParetoLSS", "TwoPieceNormalLSS", "TwoPieceLogNormalLSS",
]  # fmt: skip

# Pairs whose published criterion changed in a release, keyed by that release:
# their objectives are not comparable across the change (held-out metrics still
# are), and are compared again against any base from that release on. v0.41's
# binned route takes observed curvature for these pairs (#500), v0.40.0 Fisher.
CRITERION_CHANGES = {
    (0, 41, 0): {
        ("glm", family, link, "binned"): "#500 binned observed curvature"
        for family, link in [
            ("gamma", "log"), ("tweedie15", "log"), ("tweedie12", "log"), ("nb2fixed", "log"),
            ("nb2auto", "log"), ("binomial", "probit"), ("binomial", "cloglog"),
        ]
    }
}  # fmt: skip


def _release_tuple(release: str) -> tuple[int, ...]:
    """``v0.40.0`` as (0, 40, 0); a ref that is not a release tag, such as ``master`` or a
    sha, as a version after every criterion change, so none of them is exempted."""
    if not re.fullmatch(r"v?\d+(\.\d+)*", release):
        return (sys.maxsize,)
    return tuple(int(part) for part in release.lstrip("v").split("."))


def glm_cases() -> list[str]:
    cases = []
    for family, link in GLM_FAMLINKS:
        for size, (_, sparse) in SIZES.items():
            if sparse and family not in COUNT_FAMILIES:
                continue
            for term in TERMS:
                for route in ("exact", "binned"):
                    cases.append(f"glm/{family}/{link}/{size}/{term}/{route}/fit_reml")
                    if term in ("mains", "tensor"):
                        cases.append(f"glm/{family}/{link}/{size}/{term}/{route}/fit")
    return cases


def lss_cases() -> list[str]:
    return [
        f"lss/{family}/{spec}/{size}/{route}/{outer}"
        for family in LSS_FAMILIES
        for size in ("small", "mid")
        for spec in ("simple", "rich")
        for route in ("exact", "binned")
        for outer in ("efs", "efs+newton")
    ]


# Fits that converged on an earlier release and broke on a later one (the
# release-history sweep, v0.12.0 to v0.41). ``hist/<row>/<case>``: <row> is the
# sweep's breaks-table row. Each is fitted on the sweep's own synthetic book
# (``history_book``), so the case reproduces the break exactly. A case the
# break left refused by design (rows 5, 7, 9-12) errors on both sides and
# flags nothing; it is here so the assessment sees the decision every release.
# Rows 8, 14 and 15 are calls rather than fits, and rows 19-21 are saved
# models (``tests/test_userspace_040.py``).
HISTORY = {
    "hist/r01/convex-beside-cr/exact": ("poisson", None, "convex_cr", 4000, "exact", "fit_reml"),
    "hist/r01/convex-beside-cr/binned": ("poisson", None, "convex_cr", 4000, "binned", "fit_reml"),
    "hist/r02/kindless-mono-beside-ps/exact": ("poisson", None, "mono_kindless_ps", 4000, "exact", "fit_reml"),
    "hist/r02/kindless-mono-beside-ordcat-ps/exact": ("poisson", None, "mono_kindless_ordcat", 4000, "exact", "fit"),
    "hist/r03/small-n-150/exact": ("poisson", None, "edge", 150, "exact", "fit_reml"),
    "hist/r04/select-gamma/exact": ("gamma", None, "select", 4000, "exact", "fit_reml"),
    "hist/r05/constant-column/binned": ("poisson", None, "constant", 4000, "binned", "fit_reml"),
    "hist/r07/no-features/exact": ("poisson", None, "none", 4000, "exact", "fit_reml"),
    "hist/r09/convex-bs-degree4/exact": ("poisson", None, "convex4", 4000, "exact", "fit_reml"),
    "hist/r10/mono-ps-beside-bs/exact": ("poisson", None, "mono_ps_bs", 4000, "exact", "fit_reml"),
    "hist/r11/ordcat-step/exact": ("poisson", None, "ordcat_step", 4000, "exact", "fit_reml"),
    "hist/r12/polynomial-3-distinct/exact": ("poisson", None, "poly3", 4000, "exact", "fit_reml"),
    "hist/r13/kindless-degree2/exact": ("poisson", None, "degree2", 4000, "exact", "fit_reml"),
    "hist/r13/kindless-degree1/exact": ("poisson", None, "degree1", 4000, "exact", "fit_reml"),
    "hist/r13/kindless-m24/exact": ("poisson", None, "m24", 4000, "exact", "fit_reml"),
    "hist/r16/tweedie-default-selection/exact": ("tweedie15", None, "base", 4000, "exact", "fit"),
    "hist/r17/tweedie-p19/exact": ("tweedie19", None, "base", 4000, "exact", "fit_reml"),
    "hist/r18/kindless-mains/exact": ("poisson", None, "kindless", 4000, "exact", "fit"),
}  # fmt: skip

CASES = glm_cases() + lss_cases() + list(HISTORY)

# Fits an earlier release converged that this one must too, held in every full
# suite by ``tests/test_userspace_040.py`` (row 1 is held by its own test in
# ``tests/test_scop_efs.py``).
HISTORY_SUBSET = [
    "hist/r02/kindless-mono-beside-ps/exact",
    "hist/r03/small-n-150/exact",
    "hist/r13/kindless-degree2/exact",
]

# A compact subset that converges on v0.40.0 and runs in about half a minute:
# every family and link on main effects, every term kind on Poisson/log, and
# every SuperLSS family.  ``tests/test_userspace_040.py`` holds it to v0.40.0.
SUBSET = (
    [
        f"glm/{family}/{link}/small/mains/{route}/fit_reml"
        for family, link in GLM_FAMLINKS
        for route in ("exact", "binned")
        if (family, link) not in {("poisson", "identity"), ("binomial", "cauchit")}
        and (family, link, route)
        not in {("gaussian", "log", "exact"), ("binomial", "log", "exact")}
    ]
    + [
        f"glm/poisson/log/small/{term}/{route}/fit_reml"
        for term in TERMS[1:]
        for route in ("exact", "binned")
    ]
    + ["glm/poisson/log/small/mains/exact/fit", "glm/poisson/log/small/tensor/binned/fit"]
    + [f"lss/{family}/simple/small/exact/efs" for family in LSS_FAMILIES]
)


# --------------------------------------------------------------------------- data


def make_data(n: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    total = n + TEST_N
    x1, x2 = rng.uniform(0, 10, total), rng.uniform(-2, 2, total)
    levels = [f"a{i}" for i in range(6)]
    c1 = rng.choice(levels, total, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    c2 = rng.choice([f"g{i:02d}" for i in range(25)], total)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "c1": c1,
            "c2": c2,
            "lin": rng.normal(0, 1, total),
            "noise": rng.uniform(0, 1, total),
            "expo": rng.uniform(0.2, 1.0, total),
        }
    )
    c1_effect = dict(zip(levels, [0, 0.2, -0.15, 0.3, -0.1, 0.25], strict=True))
    c2_effect = dict(zip(sorted(set(c2)), rng.normal(0, 0.15, 25), strict=False))
    eta = (
        0.06 * x1
        + 0.3 * np.sin(1.3 * x2)
        + frame["c1"].map(c1_effect).to_numpy()
        + frame["c2"].map(c2_effect).to_numpy()
        + 0.1 * frame["lin"].to_numpy()
        + 0.04 * x1 * np.tanh(x2)
    )
    return frame, eta - eta.mean(), rng


def glm_response(family, link, eta, frame, rng, sparse):
    import numpy as np
    from scipy.special import expit

    base = -3.0 if sparse else 0.0
    exposure = frame["expo"].to_numpy()
    if family == "poisson":
        return rng.poisson(np.exp(eta + base) * (exposure if link == "log" else 1.0)).astype(float)
    if family.startswith("nb2"):
        mu = np.exp(eta + base) * exposure
        return rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    if family.startswith("tweedie"):
        counts = rng.poisson(np.exp(eta + base + 0.5) * exposure * 0.5)
        return np.array([rng.gamma(2.0, 1.0, k).sum() for k in counts]) / exposure
    if family == "gamma":
        mu = np.exp(0.5 * eta + 1.0)
        return rng.gamma(3.0, mu / 3.0)
    if family == "gaussian":
        if link == "log":
            return np.exp(0.5 * eta + 1.0) + rng.normal(0, 0.3, len(eta))
        return 1.0 + eta + rng.normal(0, 0.5, len(eta))
    if link == "log":
        p = np.clip(0.15 * np.exp(0.5 * eta + base / 2), 0, 0.9)
    else:
        p = expit(eta - 1.0 + base)
    return (rng.uniform(size=len(eta)) < p).astype(float)


def glm_family(name):
    import superglm as sg

    return {
        "poisson": sg.families.poisson,
        "gamma": sg.families.gamma,
        "gaussian": sg.families.gaussian,
        "binomial": sg.families.binomial,
        "tweedie15": lambda: sg.families.tweedie(p=1.5),
        "tweedie12": lambda: sg.families.tweedie(p=1.2),
        "nb2fixed": lambda: sg.families.nb2(theta=2.0),
        "nb2auto": lambda: sg.families.nb2(theta="auto"),
    }[name]()


def glm_terms(term):
    from superglm import (
        Categorical,
        Constraint,
        FactorSmooth,
        Numeric,
        PSpline,
        RandomEffect,
        Spline,
    )

    mains = {
        "x1": Spline(kind="cr", k=8),
        "x2": Spline(kind="cr", k=8),
        "c1": Categorical(),
        "lin": Numeric(),
    }
    increasing = Constraint.fit.increasing
    return {
        "mains": (mains, []),
        "re": ({**mains, "c2": RandomEffect()}, []),
        "tensor": (mains, [("x1", "x2")]),
        "splcat": (mains, [("x2", "c1")]),
        "fs": (
            {"x1": Spline(kind="cr", k=8), "lin": Numeric()},
            [FactorSmooth("x2", group="c1", basis="fs", k=5)],
        ),
        "sz": (
            {"x1": Spline(kind="cr", k=8), "x2": Spline(kind="cr", k=6), "lin": Numeric()},
            [FactorSmooth("x2", group="c1", basis="sz", k=5)],
        ),
        "scop": (
            {
                "x1": PSpline(n_knots=8, constraint=increasing),
                "x2": Spline(kind="cr", k=8),
                "c1": Categorical(),
            },
            [],
        ),
        "qp": (
            {
                "x1": Spline(kind="bs", k=8, constraint=increasing),
                "x2": Spline(kind="cr", k=8),
                "c1": Categorical(),
            },
            [],
        ),
        "select": ({**mains, "noise": Spline(kind="cr", k=6, select=True)}, []),
    }[term]


# --------------------------------------------------------------------------- held-out


def heldout(y, mu, weights, mass) -> dict:
    """Gini of the weighted Lorenz curve, total A/E and the worst decile's |A/E - 1|."""
    import numpy as np

    actual, expected = weights * y, weights * mu
    order = np.argsort(expected / mass, kind="stable")
    x = np.r_[0.0, np.cumsum(mass[order])] / mass.sum()
    lorenz = np.r_[0.0, np.cumsum(actual[order])] / actual.sum()
    cumulative = np.cumsum(mass[order]) / mass.sum()
    bins = np.empty(len(order), dtype=int)
    bins[order] = np.minimum((cumulative * 10).astype(int), 9)
    ae = [
        float(actual[bins == b].sum() / expected[bins == b].sum())
        for b in range(int(bins.max()) + 1)
    ]
    return {
        "gini": float(1.0 - np.sum(np.diff(x) * (lorenz[1:] + lorenz[:-1]))),
        "ae_total": float(actual.sum() / expected.sum()),
        "ae_worst_dev": float(max(abs(v - 1.0) for v in ae)),
    }


def run_glm(family, link, size, term, route, fit) -> dict:
    import numpy as np

    from superglm import SuperGLM

    n, sparse = SIZES[size]
    seed = 1000 + 17 * GLM_FAMLINKS.index((family, link)) + list(SIZES).index(size) + 1
    frame, eta, rng = make_data(n, seed)
    y = glm_response(family, link, eta, frame, rng, sparse)
    features, interactions = glm_terms(term)
    model = SuperGLM(
        family=glm_family(family),
        link=link,
        features=features,
        interactions=interactions,
        discrete=route == "binned",
    )
    weights, mass, offset, kwargs = np.ones(len(y)), np.ones(len(y)), None, {}
    if family in ("poisson", "nb2fixed", "nb2auto") and link == "log":
        offset = np.log(frame["expo"].to_numpy())
        kwargs["offset"] = offset[:n]
        mass = frame["expo"].to_numpy()
    if family.startswith("tweedie"):
        weights = mass = frame["expo"].to_numpy()
        kwargs["sample_weight"] = weights[:n]
    getattr(model, fit)(frame.iloc[:n], y[:n], **kwargs)
    if fit == "fit_reml":
        diagnostics = model.reml_diagnostics()
        out = {
            "converged": bool(diagnostics.get("converged")),
            "reason": str(diagnostics.get("termination_reason")),
            "objective": None
            if diagnostics.get("objective") is None
            else float(diagnostics["objective"]),
        }
    else:
        out = {"converged": bool(getattr(model.result, "converged", True)), "reason": "pirls"}
        out["objective"] = None
    out["edf"] = float(model.result.effective_df)
    test = frame.iloc[n:]
    mu = model.predict(test) if offset is None else model.predict(test, offset=offset[n:])
    mu = np.asarray(mu, dtype=np.float64)
    yt, wt = y[n:], weights[n:]
    distribution = model._distribution
    out["test_deviance"] = float(np.sum(wt * distribution.deviance_unit(yt, mu)))
    out["test_nll"] = _glm_nll(family, distribution, yt, mu, wt, float(model.result.phi))
    out.update(heldout(yt, mu, wt, mass[n:]))
    out["max_mu"] = float(np.max(mu))
    return out


def _glm_nll(family, distribution, y, mu, weights, phi):
    """Held-out negative log-likelihood from SciPy's densities, where the family has one."""
    import numpy as np
    from scipy import stats

    if np.any(~np.isfinite(mu)):
        return None
    if family == "poisson":
        return float(-np.sum(stats.poisson.logpmf(y, mu)))
    if family.startswith("nb2"):
        theta = float(distribution.theta)
        return float(-np.sum(stats.nbinom.logpmf(y, theta, theta / (theta + mu))))
    if family == "binomial":
        p = np.clip(mu, 1e-300, 1.0 - 1e-16)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    if family == "gamma":
        shape = 1.0 / phi
        return float(-np.sum(stats.gamma.logpdf(y, shape, scale=mu / shape)))
    if family == "gaussian":
        return float(-np.sum(stats.norm.logpdf(y, mu, math.sqrt(phi))))
    return None  # Tweedie: deviance and Gini carry the comparison


def run_lss(family_name, spec, size, route, outer) -> dict:
    import numpy as np

    import superglm as sg
    from superglm import SuperLSS, cat, re, s, ti

    n = {"small": 500, "mid": 4000}[size]
    frame, eta, rng = make_data(n, 5000 + 31 * LSS_FAMILIES.index(family_name) + (size == "mid"))
    exposure = frame["expo"].to_numpy()
    family = getattr(sg, family_name)()
    if family_name == "TweedieLSS":
        helpers = ["mu", "phi", "p"]
    else:
        helpers = [parameter.name for parameter in family.parameters]
    offsets, weights = None, None
    if family_name == "GaussianLS":
        y = 1 + eta + rng.normal(0, 0.4 + 0.2 * (frame["x2"].to_numpy() > 0), len(eta))
    elif family_name == "TwoPieceNormalLSS":
        sign = np.where(rng.uniform(size=len(eta)) < 0.6, -1, 2)
        y = 1 + eta + sign * np.abs(rng.normal(0, 0.5, len(eta)))
    elif family_name == "NegativeBinomialLS":
        mu = np.exp(eta) * exposure
        y = rng.negative_binomial(1.5, 1.5 / (1.5 + mu)).astype(float)
        offsets = {"mean": np.log(exposure)}
    elif family_name == "TweedieLSS":
        counts = rng.poisson(np.exp(eta + 0.5) * exposure * 0.5)
        y = np.array([rng.gamma(2.0, 1.0, k).sum() for k in counts]) / exposure
        weights = exposure
    elif family_name == "GeneralizedParetoLSS":
        y = rng.pareto(4.0, len(eta)) * np.exp(0.3 * eta) + 1e-3
    else:
        y = rng.gamma(3.0, np.exp(0.5 * eta + 1.0) / 3.0)
    first, second = [s("x1", kind="cr", k=8), cat("c1")], []
    if spec == "rich":
        first += [s("x2", kind="cr", k=6), re("c2"), ti("x1", "x2")]
        second = [s("x2", kind="cr", k=5)]
    declarations = [getattr(family, helpers[0])(*first), getattr(family, helpers[1])(*second)]
    declarations += [getattr(family, name)() for name in helpers[2:]]
    model = SuperLSS(family, *declarations, discrete=route == "binned")
    kwargs = {"outer": outer}
    if weights is not None:
        kwargs["sample_weight"] = weights[:n]
    train_offsets = None if offsets is None else {k: v[:n] for k, v in offsets.items()}
    test_offsets = None if offsets is None else {k: v[n:] for k, v in offsets.items()}
    model.fit_reml(frame.iloc[:n], y[:n], offsets=train_offsets, **kwargs)
    smoothing = model._model.fit_state.smoothing
    out = {
        "converged": bool(smoothing.converged),
        "reason": str(smoothing.convergence_reason),
        "objective": float(smoothing.objective),
        "edf": float(sum(float(v) for v in model._model.fit_state.result.predictor_edf.values())),
    }
    test, yt = frame.iloc[n:], y[n:]
    wt = np.ones(len(yt)) if weights is None else weights[n:]
    score_kwargs = {} if weights is None else {"sample_weight": wt}
    scores = model.scores(test, yt, which=("log",), offsets=test_offsets, **score_kwargs)
    out["test_nll"] = float(scores[scores.columns[0]].sum())
    mu = np.asarray(model.predict(test, offsets=test_offsets), dtype=np.float64)
    mass = exposure[n:] if family_name == "NegativeBinomialLS" else wt
    if np.all(mu > 0) and np.sum(yt) > 0:
        out.update(heldout(yt, mu, wt, mass))
    return out


def history_book(n: int):
    """The release-history sweep's synthetic book: its first n training rows and 2,000 test rows.

    Regenerated by the sweep's own code from its seed, including ``np.vectorize``
    typing a level-effect lookup by its first value.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261020)

    def synth(rows):
        x1, x2 = rng.uniform(0, 10, rows), rng.uniform(-2, 2, rows)
        x3, x4 = rng.normal(0, 1, rows), rng.uniform(0, 1, rows)
        glev = np.array([*"abcdefg", "thin"])
        gp = np.array([0.25, 0.2, 0.15, 0.12, 0.1, 0.09, 0.085, 0.005])
        g = glev[rng.choice(8, rows, p=gp / gp.sum())]
        o = np.array(list("ABCDE"))[rng.choice(5, rows, p=[0.1, 0.3, 0.3, 0.2, 0.1])]
        r = np.array([f"r{i:02d}" for i in range(40)])[rng.integers(0, 40, rows)]
        h = np.array(["h1", "h2", "h3", "h4"])[rng.integers(0, 4, rows)]
        expo = rng.uniform(0.2, 1.0, rows)
        geff = dict(zip(glev, [0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.05, 0.4], strict=True))
        oeff = {"A": -0.3, "B": -0.1, "C": 0, "D": 0.15, "E": 0.3}
        reff_values = rng.normal(0, 0.25, 40)
        reff = {f"r{i:02d}": reff_values[i] for i in range(40)}
        heff = {"h1": 0.0, "h2": 0.15, "h3": -0.1, "h4": 0.2}
        f = (
            0.4 * np.sin(x1 / 1.6) + 0.25 * x2**2 / 4 - 0.2 * x3 + 0.5 * x4
            + np.vectorize(geff.get)(g) + np.vectorize(oeff.get)(o)
            + np.vectorize(reff.get)(r) + np.vectorize(heff.get)(h) * x1 / 10
            + 0.1 * np.sin(x1 / 2) * x2
        )  # fmt: skip
        frame = pd.DataFrame(
            {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "g": g.astype(object), "o": o.astype(object)}
        )
        return frame, expo, f

    train, train_expo, train_f = synth(4000)
    test, test_expo, test_f = synth(2000)
    responses = {}
    for which, expo, f in (("train", train_expo, train_f), ("test", test_expo, test_f)):
        poisson = rng.poisson(expo * np.exp(-1.0 + f)).astype(float)
        rng.uniform(size=f.size)  # the sweep's binomial draw
        responses[which] = (poisson, rng.gamma(2.0, np.exp(6.0 + 0.6 * f) / 2.0))
        rng.normal(0, 1.5, f.size)  # its Gaussian draw
        counts = rng.poisson(expo * np.exp(-1.5 + f))
        severity = np.array([rng.gamma(2.0, 500.0, k).sum() if k else 0.0 for k in counts])
        responses[which] += (severity / expo,)
        rng.poisson(rng.gamma(2.0, expo * np.exp(-0.5 + f) / 2.0))  # its NB draw
    train = train.assign(x4c=1.0, n3=(np.floor(train["x4"] * 3) + 1).clip(1, 3))
    test = test.assign(x4c=1.0, n3=(np.floor(test["x4"] * 3) + 1).clip(1, 3))
    rows = slice(0, n)
    return (
        train.iloc[rows].reset_index(drop=True), train_expo[rows],
        {k: v[rows] for k, v in zip(("poisson", "gamma", "tweedie"), responses["train"], strict=True)},
        test, test_expo, dict(zip(("poisson", "gamma", "tweedie"), responses["test"], strict=True)),
    )  # fmt: skip


def history_terms(kind):
    from superglm import Categorical, Constraint, Numeric, OrderedCategorical, Polynomial, Spline

    inc = Constraint.fit.increasing
    base = {
        "x1": Spline(kind="ps", k=10),
        "x2": Spline(kind="cr", k=8),
        "x3": Numeric(),
        "g": Categorical(),
    }
    return {
        "base": lambda: base,
        "convex_cr": lambda: {"x2": Spline(kind="ps", k=10, constraint=Constraint.fit.convex), "x1": Spline(kind="cr", k=8)},
        "mono_kindless_ps": lambda: {"x4": Spline(constraint=inc), "x1": Spline(kind="ps", k=8, constraint=inc), "g": Categorical()},
        "mono_kindless_ordcat": lambda: {"x4": Spline(constraint=inc), "o": OrderedCategorical(order=list("ABCDE"), basis=Spline(kind="ps", k=5, constraint=inc))},
        "edge": lambda: {"x1": Spline(kind="ps", k=10), "x4": Spline(kind="cr", k=8), "x3": Numeric(), "g": Categorical()},
        "select": lambda: {"x1": Spline(kind="ps", k=10, select=True), "x4": Spline(kind="cr", k=8, select=True), "x3": Spline(kind="cr", k=6, select=True)},
        "constant": lambda: {"x1": Spline(n_knots=8), "x4c": Spline(n_knots=5), "g": Categorical()},
        "none": lambda: None,
        "convex4": lambda: {"x2": Spline(kind="bs", degree=4, n_knots=8, constraint=Constraint.fit.convex), "x1": Spline(kind="cr", k=8)},
        "mono_ps_bs": lambda: {"x4": Spline(kind="ps", k=8, constraint=inc), "x1": Spline(kind="bs", k=8, constraint=inc), "g": Categorical()},
        "ordcat_step": lambda: {"o": OrderedCategorical(order=list("ABCDE"), basis="step"), "x1": Spline(kind="cr", k=8)},
        "poly3": lambda: {"n3": Polynomial(degree=3), "x1": Spline(kind="cr", k=8)},
        "degree2": lambda: {"x1": Spline(degree=2, n_knots=8), "g": Categorical()},
        "degree1": lambda: {"x1": Spline(degree=1, n_knots=8), "g": Categorical()},
        "m24": lambda: {"x1": Spline(m=(2, 4), n_knots=10), "g": Categorical()},
        "kindless": lambda: {"x1": Spline(k=10), "x2": Spline(k=8), "g": Categorical()},
    }[kind]()  # fmt: skip


def run_history(name: str) -> dict:
    import numpy as np

    import superglm as sg
    from superglm import SuperGLM

    family, link, terms, n, route, fit = HISTORY[name]
    train, train_expo, train_y, test, test_expo, test_y = history_book(n)
    response = "poisson" if family == "poisson" else "gamma" if family == "gamma" else "tweedie"
    distribution = {
        "poisson": sg.families.poisson, "gamma": sg.families.gamma,
        "tweedie15": lambda: sg.families.tweedie(p=1.5), "tweedie19": lambda: sg.families.tweedie(p=1.9),
    }[family]()  # fmt: skip
    model = SuperGLM(
        family=distribution, link=link, features=history_terms(terms),
        discrete=route == "binned", selection_penalty=0.0 if fit == "fit_reml" else None,
    )  # fmt: skip
    kwargs, weights, mass, offset = {}, np.ones(len(test)), test_expo, None
    if family == "poisson":
        kwargs["offset"], offset = np.log(train_expo), np.log(test_expo)
    elif family.startswith("tweedie"):
        kwargs["sample_weight"], weights = train_expo, test_expo
    # A level the training rows never saw scores as the sweep scored it: as "a".
    test = test.assign(g=test["g"].where(test["g"].isin(set(train["g"])), "a"))
    getattr(model, fit)(train, train_y[response], **kwargs)
    out = {
        "converged": bool(getattr(model.result, "converged", True)),
        "reason": "pirls",
        "objective": None,
    }
    if fit == "fit_reml":
        diagnostics = model.reml_diagnostics()
        out.update(
            converged=bool(diagnostics.get("converged")),
            reason=str(diagnostics.get("termination_reason")),
        )
        out["objective"] = (
            None if diagnostics.get("objective") is None else float(diagnostics["objective"])
        )
    out["edf"] = float(model.result.effective_df)
    mu = np.asarray(
        model.predict(test) if offset is None else model.predict(test, offset=offset),
        dtype=np.float64,
    )
    yt = test_y[response]
    out["test_deviance"] = float(np.sum(weights * model._distribution.deviance_unit(yt, mu)))
    out["test_nll"] = _glm_nll(
        family, model._distribution, yt, mu, weights, float(model.result.phi)
    )
    out.update(heldout(yt, mu, weights, mass))
    out["max_mu"] = float(np.max(mu))
    return out


def run_case(name: str) -> dict:
    parts = name.split("/")
    if parts[0] == "hist":
        return run_history(name)
    return run_glm(*parts[1:]) if parts[0] == "glm" else run_lss(*parts[1:])


# --------------------------------------------------------------------------- compare

OBJECTIVE_TOLERANCE = 1e-5  # relative to 1 + |V|: well above the REML stop's own resolution
HELDOUT_TOLERANCE = 2e-3  # relative, plus HELDOUT_FLOOR: above criterion-level moves (#500)
HELDOUT_FLOOR = 0.5
GINI_TOLERANCE = 0.01


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def sensible(record: dict) -> bool:
    """Converged, with finite held-out results and no runaway mean."""
    if record["status"] != "ok" or not record.get("converged"):
        return False
    held = record.get("test_nll", record.get("test_deviance"))
    if not _finite(held) and not _finite(record.get("test_deviance")):
        return False
    return _sane_mean(record.get("max_mu"))


def _sane_mean(max_mu) -> bool:
    """No runaway mean; absent (an LSS record) passes, a NaN or an infinity does not."""
    return max_mu is None or (_finite(max_mu) and max_mu < 1e8)


def flags(name: str, base: dict, candidate: dict, base_release: str) -> list[str]:
    """Why the candidate's outcome is worse than a sensible base outcome (empty: it is not).

    ``base_release`` (for example ``v0.40.0``) decides which pairs' objectives are
    comparable: those whose criterion changed in a later release are not.
    """
    if not sensible(base):
        return []
    if candidate["status"] != "ok":
        return [f"{candidate['status']}: {candidate.get('error', '')[:160]}"]
    if not candidate.get("converged"):
        return [f"not converged ({candidate.get('reason')}; base {base.get('reason')})"]
    found = []
    parts = name.split("/")
    key = tuple(parts[:3]) + (parts[5] if parts[0] == "glm" else "",)
    changed = any(
        key in pairs
        for since, pairs in CRITERION_CHANGES.items()
        if _release_tuple(base_release) < since
    )
    for metric in ("objective", "test_nll", "test_deviance", "gini"):
        if _finite(base.get(metric)) and not _finite(candidate.get(metric)):
            found.append(f"{metric} {base[metric]:.6g} -> {candidate.get(metric)}")
    if not _sane_mean(candidate.get("max_mu")):
        found.append(f"max_mu {base.get('max_mu')} -> {candidate.get('max_mu')}")
    old, new = base.get("objective"), candidate.get("objective")
    if not changed and _finite(old) and _finite(new):
        if new > old + OBJECTIVE_TOLERANCE * (1.0 + abs(old)):
            found.append(f"objective {old:.8g} -> {new:.8g}")
    for metric in ("test_nll", "test_deviance"):
        if metric == "test_deviance" and "nb2auto" in name:
            continue  # each version's deviance is at its own estimated theta
        old, new = base.get(metric), candidate.get(metric)
        if (
            _finite(old)
            and _finite(new)
            and new - old > HELDOUT_TOLERANCE * abs(old) + HELDOUT_FLOOR
        ):
            found.append(f"{metric} {old:.6g} -> {new:.6g}")
    old, new = base.get("gini"), candidate.get("gini")
    if _finite(old) and _finite(new) and new < old - GINI_TOLERANCE:
        found.append(f"gini {old:.4f} -> {new:.4f}")
    return found


def compare(
    base: dict[str, dict], candidate: dict[str, dict], base_release: str
) -> dict[str, list[str]]:
    """Flags for every case in both records; a case missing from the candidate is flagged."""
    out = {}
    for name, record in base.items():
        if name not in candidate:
            if sensible(record):
                out[name] = ["missing from the candidate run"]
            continue
        found = flags(name, record, candidate[name], base_release)
        if found:
            out[name] = found
    return out


# --------------------------------------------------------------------------- CLI


class _CaseTimeoutError(Exception):
    pass


def _alarm(*_):
    raise _CaseTimeoutError


def run(names: list[str], out: Path, seconds: int = 300) -> None:
    signal.signal(signal.SIGALRM, _alarm)
    with out.open("a", encoding="utf-8") as handle:
        for name in names:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                signal.alarm(seconds)
                try:
                    record = {"status": "ok", **run_case(name)}
                except _CaseTimeoutError:
                    record = {"status": "timeout"}
                except Exception as exc:  # noqa: BLE001 - a refusal is the outcome recorded
                    record = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:400],
                        "where": traceback.format_exc().splitlines()[-3][:200],
                    }
                finally:
                    signal.alarm(0)
            categories: dict[str, int] = {}
            for item in caught:
                label = f"{item.category.__name__}: {str(item.message)[:100]}"
                categories[label] = categories.get(label, 0) + 1
            record.update(
                case=name, warnings=categories, seconds=round(time.perf_counter() - start, 2)
            )
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            print(name, record["status"], flush=True)


def load(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["case"]] = record
    return records


def _selected(args) -> list[str]:
    names = list(SUBSET) if args.subset else list(CASES)
    if args.only:
        names = [n for n in names if re.search(args.only, n)]
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        names = names[index::count]
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for command in ("run", "release"):
        p = sub.add_parser(command)
        if command == "release":
            p.add_argument("ref", help="the last release's tag, for example v0.40.0")
        p.add_argument("--out", required=True)
        p.add_argument("--only", help="a regular expression over case names")
        p.add_argument("--subset", action="store_true", help="the compact subset only")
        p.add_argument("--shard", help="I/K: every K-th case from the I-th")
    p = sub.add_parser("compare")
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("release", help="the base's release tag, for example v0.40.0")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        run(_selected(args), Path(args.out))
        return 0
    if args.cmd == "compare":
        found = compare(load(Path(args.base)), load(Path(args.candidate)), args.release)
        for name, reasons in sorted(found.items()):
            print(f"{name}: {'; '.join(reasons)}")
        print(f"{len(found)} flagged")
        return 1 if found else 0
    sys.path.insert(0, str(HERE))
    import books

    tree = books.prepare_tree(books._resolve(args.ref))
    out = Path(args.out).resolve()  # the baseline runs with its own tree as cwd
    out.mkdir(parents=True, exist_ok=True)
    forwarded = [
        *(["--only", args.only] if args.only else []),
        *(["--subset"] if args.subset else []),
        *(["--shard", args.shard] if args.shard else []),
    ]
    base, candidate = out / f"{args.ref}.jsonl", out / "candidate.jsonl"
    for path in (base, candidate):
        path.unlink(missing_ok=True)
    script = str(Path(__file__).resolve())
    subprocess.run(
        [str(books._venv_python(tree)), script, "run", "--out", str(base), *forwarded],
        cwd=tree,
        env=books._clean_env(),
        check=True,
    )
    run(_selected(args), candidate)
    return main(["compare", str(base), str(candidate), args.ref])


if __name__ == "__main__":
    sys.exit(main())
