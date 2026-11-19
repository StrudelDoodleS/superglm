"""The Gamma severity book against mgcv: every coefficient identified, each SE mgcv's ``Vp``.

The fixture is mgcv 1.9-3, run as a black box, on superglm's own design, penalty components,
REML lambdas and scale for the books case ``fam_gamma_log_exact`` (freMTPL2 severity, 21,070
training rows, 614 coefficients, 27 smoothing parameters) at master 5f28e8e80:
``gam(y ~ X, paraPen = ..., family = Gamma(log), method = "REML", scale = phi)``, so both
engines solve one penalised problem.  mgcv reports all 614 coefficients and drops none.  Master
blanked 398 of them and 10 summary rows, every ``VehBrand`` level among them, because the data
alone leave them undetermined (``VehAge``'s sparse upper tail); the penalty identifies each one
(``solvers.identifiability``).

The refit is the book's own REML fit, which reproduces the exported solver coordinates and
lambdas.  A fit at the pinned lambdas reparametrises the splines there, so its solver
coordinates are not the exported ones.  Anchored to the freMTPL2 book: it skips without both
parquet files and fails under ``SUPERGLM_REQUIRE_DATA=1``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from . import _datasets

_FREQ_SKIP_REASON = _datasets.skip_reason("freMTPL2freq.parquet")
_SEV_SKIP_REASON = _datasets.skip_reason("freMTPL2sev.parquet")
pytestmark = pytest.mark.skipif(
    _FREQ_SKIP_REASON is not None or _SEV_SKIP_REASON is not None,
    reason=_FREQ_SKIP_REASON or _SEV_SKIP_REASON or "",
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).parent / "fixtures" / "mgcv_gamma_severity_book.json"


def _books():
    """``benchmarks/books.py``, whose case built the fixture."""
    spec = importlib.util.spec_from_file_location(
        "_books_identifiability", _ROOT / "benchmarks" / "books.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def book():
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    books = _books()
    case = books.CASES[fixture["case"]]
    train, _ = books.load_book(case["dataset"])
    X, y, weights, offset = books._glm_inputs(case, train)
    model = books._glm_model(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y, sample_weight=weights, offset=offset)
    return fixture, model, X, y


def test_every_severity_book_coefficient_has_mgcvs_bayesian_standard_error(book) -> None:
    """All 614 standard errors are printed and are mgcv's ``Vp`` ones; nothing is aliased.

    ``SE^2 / phi`` against mgcv's ``Vp / phi`` at the fixture's stated tolerance, 1e-6
    (measured 2.9e-12: both engines stop PIRLS at a tolerance, so it is set 100 times looser
    than the coefficients' 2.4e-9); the coefficients to 1e-6 of the largest and the EDF to
    1e-8.  Mutation: master's data-only rule blanks 398 coefficients and the 10 ``VehBrand``
    rows here.
    """
    fixture, model, X, y = book
    names = ["Intercept"] + [f"{g.name}[{i}]" for g in model._groups for i in range(g.size)]
    assert names == fixture["coefficient_names"] and len(X) == fixture["n"]
    assert fixture["mgcv_dropped"] == [] and fixture["mgcv_rank"] == len(names)

    metrics = model.metrics(X, y)
    se = np.concatenate(
        [[metrics.intercept_se]]
        + [np.atleast_1d(metrics.coefficient_se[g.name]) for g in model._groups]
    )
    assert not np.any(np.isnan(se)), f"{int(np.isnan(se).sum())} standard errors blank"
    phi = model.result.phi
    vp = np.asarray(fixture["mgcv_se_vp"]) ** 2 / fixture["phi"]
    np.testing.assert_allclose(se**2 / phi, vp, rtol=fixture["tolerances"]["se_relative"])

    rows = {row.name: row.se for row in model.summary()._coef_rows if not row.is_spline}
    assert all(value is not None and np.isfinite(value) for value in rows.values())
    brand = next(g for g in model._groups if g.name == "VehBrand")
    levels = model._specs["VehBrand"]._non_base
    assert len(levels) == 10
    printed = np.array([rows[f"VehBrand[{level}]"] for level in levels])
    expected = np.sqrt(vp[1 + brand.start : 1 + brand.end] * phi)
    np.testing.assert_allclose(printed, expected, rtol=fixture["tolerances"]["se_relative"])

    beta = np.r_[model.result.intercept, model.result.beta]
    mgcv = np.asarray(fixture["mgcv_coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.effective_df == pytest.approx(fixture["measured"]["edf_mgcv"], rel=1e-8)
    assert model._fit_inference_info["identifiability"].aliases == ()


def _whole_frequency_book():
    """Every row of the corrected freMTPL2 frequency book, with the gate review's main effects."""
    spec = importlib.util.spec_from_file_location(
        "_fremtpl2_identifiability", _ROOT / "examples" / "fremtpl2_data.py"
    )
    assert spec is not None and spec.loader is not None
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    path = _datasets.find("freMTPL2freq.parquet")
    assert path is not None
    freq = data.load_corrected_freq(path.parent)
    freq["LogDensity"] = np.log1p(freq["Density"].to_numpy(float))
    for name in ("DrivAge", "VehAge", "BonusMalus"):
        freq[name] = freq[name].astype(float)
    for name in ("VehBrand", "Region"):
        freq[name] = freq[name].astype(str)
    X = freq[["DrivAge", "VehAge", "BonusMalus", "LogDensity", "VehBrand", "Region"]]
    exposure = freq["Exposure"].to_numpy(float)
    return X, freq["ClaimNb"].to_numpy(float) / exposure, exposure


def test_the_frequency_book_proves_existence_from_its_score(monkeypatch) -> None:
    """The whole freMTPL2 frequency book's estimate is proved finite by its score.

    The gate review k3p8's fit: all 678,007 rows, Poisson claim rate on exposure, splines in
    driver age, vehicle age and bonus-malus, log density, brand and region.  Most rows have no
    claim, and a row deep in the tail has a multiplier about its own working weight, so the
    score's unweighted test, ``min mu sqrt(lambda / max W)``, failed by 2.5 times; the weighted
    form, ``min mu_i / sqrt(W_i) sqrt(lambda)``, clears it about fifty times over.  Nothing a
    user reads differs from ``07a1401a5`` (no footnote either way): what moves is the route,
    there the rows' program and the interior rank (0.29 s of a 5.3 s fit).  Mutation check:
    with the unweighted margin the program ran.
    """
    import superglm.model.separation_ops as separation_ops
    from superglm import Categorical, Numeric, Spline, SuperGLM

    ran = []
    real = separation_ops.separating_direction
    monkeypatch.setattr(
        separation_ops,
        "separating_direction",
        lambda matrix, sides: ran.append(1) or real(matrix, sides),
    )
    X, rate, exposure = _whole_frequency_book()
    model = SuperGLM(
        family="poisson",
        weight_semantics="frequency",
        features={
            "DrivAge": Spline(n_knots=8),
            "VehAge": Spline(n_knots=12),
            "BonusMalus": Spline(n_knots=12),
            "LogDensity": Numeric(),
            "VehBrand": Categorical(),
            "Region": Categorical(),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, rate, sample_weight=exposure)
    assert "not resolved" not in str(model.summary())
    assert ran == []
    record = model._fit_inference_info["identifiability"]
    assert (record.separation.state, record.separation.decided_by) == ("absent", "certificate")
