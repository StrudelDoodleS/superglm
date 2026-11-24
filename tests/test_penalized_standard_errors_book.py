"""The Gamma severity book against mgcv: every coefficient shown, each SE mgcv's ``Vp``.

The fixture is mgcv 1.9-3, run as a black box, on superglm's own design, penalty components,
REML lambdas and scale for the books case ``fam_gamma_log_exact`` (freMTPL2 severity, 21,070
training rows, 614 coefficients, 27 smoothing parameters) at the fit's REML stop:
``gam(y ~ X, paraPen = ..., family = Gamma(log), method = "REML", scale = phi)``, so both
engines solve one penalised problem.  mgcv reports all 614 coefficients and drops none.  Master
blanked 398 of them and 10 summary rows, every ``VehBrand`` level among them, because the data
alone leave them undetermined (``VehAge``'s sparse upper tail); the penalised Hessian is
positive definite (Cholesky, scaled condition 1.1e5), which the fit's own record certifies
(``state_ops.standard_error_mask``).

The refit is the book's own REML fit, which reproduces the exported lambdas.  The fixture's
solver coordinates are whitened at those lambdas, so it belongs to one REML stop, and REML does
not place a suppressed term's lambda to the comparison's 1e-6: ``VehAge:VehBrand[B14]`` stops at
7.1e4 with its score under 1e-4 (#604), at 1.5e8 under ``reml_tol=1e-12``, and stopped at 488
while a score of ``1e-7 (1 + |V|)`` = 0.018 was held, where one of its variances was 25% larger.
A change to where the search stops rebuilds the fixture at the new stop (mgcv on the exported
design and penalties); it does not loosen the comparison.  The main splines and the
interactions now hold their exact null spaces in their own coordinates, so the
solver-to-B-spline maps of the coordinates mgcv fitted, at the fixture's lambdas, accompany it
(``mgcv_gamma_severity_spline_maps.json`` and ``mgcv_gamma_severity_interaction_maps.npz``),
and the refit is compared in those coordinates.  Anchored to the freMTPL2 book: it skips
without both parquet files and fails under ``SUPERGLM_REQUIRE_DATA=1``.
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
from .test_penalized_standard_errors import dense_raw_standard_errors, fixture_contrasts

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
        "_books_standard_errors", _ROOT / "benchmarks" / "books.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.pins_behaviour
def test_every_severity_book_coefficient_has_mgcvs_bayesian_standard_error() -> None:
    """All 614 standard errors are printed and are mgcv's ``Vp`` ones.

    ``SE^2 / phi`` against mgcv's ``Vp / phi`` at the fixture's stated tolerance, 1e-6
    (measured 1.2e-7 under the SkylakeX and Haswell OpenBLAS kernels, whose REML stop differs
    from the fixture's by 1.2e-9 in ``VehAge:VehBrand[B14]``'s lambda; the tree that built the
    fixture reproduced it to 5.0e-12; both engines stop PIRLS at a tolerance, so it is set 100
    times looser than the coefficients' 2.4e-9); the coefficients to 1e-6 of the largest and
    the EDF to 1e-8.  The raw standard errors are also the dense inverse's of
    ``[1 X]' W [1 X] + S`` built from the design and penalty (``kappa u`` about 1e-11; rtol
    1e-9).  On master 398 coefficients and the 10 ``VehBrand`` rows were blank
    before #681.  A rotation changes a spline coefficient's own SE; its original linear
    contrast and variance are invariant, so the main splines are compared after mapping to
    the fixture's coordinates.  The maps belong to the fixture's lambdas: those of master's
    earlier stop leave the main splines' ``SE^2`` 1.1e-4 from mgcv's.
    """
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    books = _books()
    case = books.CASES[fixture["case"]]
    train, _ = books.load_book(case["dataset"])
    X, y, weights, offset = books._glm_inputs(case, train)
    assert weights is None and offset is None
    model = books._glm_model(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
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
    maps = json.loads((_FIXTURE.parent / "mgcv_gamma_severity_spline_maps.json").read_text("utf-8"))
    with np.load(_FIXTURE.parent / "mgcv_gamma_severity_interaction_maps.npz") as interactions:
        maps.update(interactions)
    beta, observed_vp = fixture_contrasts(model, metrics, se, maps)
    np.testing.assert_allclose(observed_vp, vp, rtol=fixture["tolerances"]["se_relative"])
    raw = np.concatenate([np.atleast_1d(metrics.coefficient_se_raw[g.name]) for g in model._groups])
    np.testing.assert_allclose(raw, dense_raw_standard_errors(model, X), rtol=1e-9)

    rows = {row.name: row.se for row in model.summary()._coef_rows if not row.is_spline}
    assert all(value is not None and np.isfinite(value) for value in rows.values())
    brand = next(g for g in model._groups if g.name == "VehBrand")
    levels = model._specs["VehBrand"]._non_base
    assert len(levels) == 10
    printed = np.array([rows[f"VehBrand[{level}]"] for level in levels])
    expected = np.sqrt(vp[1 + brand.start : 1 + brand.end] * phi)
    np.testing.assert_allclose(printed, expected, rtol=fixture["tolerances"]["se_relative"])

    mgcv = np.asarray(fixture["mgcv_coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.effective_df == pytest.approx(fixture["measured"]["edf_mgcv"], rel=1e-8)
