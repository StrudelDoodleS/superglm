"""Leverages read back from term-pair tables (``_group_matrix/_lookup_tables.py``).

A design whose groups are lookups (categoricals, support-compressed splines,
tensors, spline-by-category levels) and dense columns reads the exact REML
gradient's leverages ``h_r = x_r' H^-1 x_r`` from per-pair tables instead of
solving a row per data row.  These tests pin the leverages against the dense
design within the bound the module derives in ``u``, the gradient against the
derivative-Gram route, bitwise determinism across worker counts, the dispatch
(no solved-row pass on a lookup design), and that a design of no lookups keeps
the solved-row route.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm._group_matrix._lookup_tables as leverage_tables
import superglm.reml.direct as direct
import superglm.reml.w_derivatives as w_derivatives
from superglm import Numeric, Spline, SuperGLM
from superglm._parallel import parallel_config
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from tests.test_reml_leverage_hessian import (
    _correction_inputs,
    _count_row_passes,
    _gamma,
    _poisson_model,
    _rounding_bounds,
)


def _centre(model: SuperGLM) -> tuple[np.ndarray, np.ndarray]:
    """The exact centre pair the weight correction reads (dense columns from the summary's pair)."""
    summary = model.result.reml_geometry
    dense = w_derivatives.dense_columns(model._dm)
    return (
        np.where(dense, summary.mean_hi, summary.mean_x),
        np.where(dense, summary.mean_lo, 0.0),
    )


def _magnitudes(model: SuperGLM, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Row magnitudes ``M_r`` with ``eta_r <= M_r' |V| M_r``, the zero-row mask, the widest basis.

    A support group's row is ``B[k] R``: its magnitude is ``(|B| |R|)[k]``,
    which also bounds the rounding of either route's product ``B R`` (``gamma_w``
    of it, ``w`` the basis width).  A raw group (categorical, spline level)
    reads its centre separately from its row, so it carries ``|c|`` twice.
    """
    dm = model._dm
    rows, column, widest = [], 0, 1
    raw = np.zeros(dm.p, dtype=bool)
    dense = np.zeros(dm.p, dtype=bool)
    for gm in dm.group_matrices:
        width = gm.shape[1]
        cols = slice(column, column + width)
        if type(gm) is DenseGroupMatrix:
            # read as its exactly centred values on both routes
            rows.append(np.abs(np.asarray(gm.M, dtype=np.float64) - centre[cols]))
            dense[cols] = True
        elif type(gm) is CategoricalGroupMatrix:
            rows.append(np.abs(gm.toarray()))
            raw[cols] = True
        elif isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
            block = np.zeros((dm.n, width))
            table = np.abs(gm.B_unique) @ np.abs(gm.R_inv)
            block[gm.row_idx] = table[gm.bin_idx_level]
            rows.append(block)
            raw[cols] = True
            widest = max(widest, gm.B_unique.shape[1])
        else:
            assert isinstance(gm, DiscretizedSSPGroupMatrix)
            rows.append((np.abs(gm.B_unique) @ np.abs(gm.R_inv))[gm.bin_idx])
            widest = max(widest, gm.B_unique.shape[1])
        column += width
    magnitude = np.hstack(rows) + np.where(dense, 0.0, np.where(raw, 2.0, 1.0)) * np.abs(centre)
    return magnitude, raw, widest


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
@pytest.mark.parametrize("offset", [1.0e3, 1.0e9])
def test_table_leverages_match_the_dense_design_within_their_bound(monkeypatch, offset, pair_form):
    """``h`` from the tables against ``x_r' V x_r`` on the materialised centred rows.

    Bound: the table route is within ``gamma_K eta_r`` (``K =
    TablePlan.count_roundings()``, module docstring); the dense reference
    within ``gamma_(2p+2) eta_r``; the two routes' raw rows ``B R`` round
    apart by ``gamma_w`` of ``|B| |R|`` each, adding ``2 gamma_w`` twice.  A
    dropped centre term was 3% off on this fixture, a million times the bound.
    """
    model = _poisson_model(offset)
    dm, inverse = model._dm, np.asarray(model._fit_inference_info["XtWX_inv"], dtype=np.float64)
    centre_hi, centre_lo = _centre(model)
    monkeypatch.setattr(leverage_tables, "_FORCE_PAIR_FORM", pair_form)
    dm._structured_layout_cache.pop("lookup_tables", None)
    plan = leverage_tables.lookup_plan(dm)
    assert plan is not None and plan.n_pairs > 0 and plan.dense_columns.size == 1
    assert plan.tab_pairs.size if pair_form != "accumulate" else not plan.tab_pairs.size
    assert plan.acc_pairs.size if pair_form != "table" else not plan.acc_pairs.size

    leverage = leverage_tables.table_leverages(
        plan, inverse, centre_hi, centre_lo, np.ones(dm.n, dtype=bool)
    )
    centred = (dm.toarray() - centre_hi) - centre_lo
    reference = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    magnitude, _, widest = _magnitudes(model, centre_hi + centre_lo)
    eta = np.einsum("ij,jk,ik->i", magnitude, np.abs(inverse), magnitude)
    k = plan.count_roundings() + 2 * dm.p + 2 + 4 * widest
    bound = _gamma(k) * eta

    assert np.all(np.abs(leverage - reference) <= bound)
    assert np.max(bound / np.abs(reference)) < 1e-9


def test_table_gradient_matches_the_derivative_grams_within_rounding(monkeypatch):
    """The table route's gradient is the exact one; it forms no cross terms."""
    model = _poisson_model(1.0e3)
    centre_hi, centre_lo = _centre(model)
    plan = leverage_tables.lookup_plan(model._dm)
    assert plan is not None
    gradient, _, gradient_bound, _ = _rounding_bounds(
        model, raw_centre_copies=2, extra_roundings=plan.count_roundings()
    )
    args, kwargs = _correction_inputs(model)

    leverage = w_derivatives.reml_w_correction(*args, leverage_hessian=True, **kwargs)

    assert leverage is not None and leverage[1] is None and leverage[2] is None
    assert np.all(np.abs(leverage[0] - gradient) <= gradient_bound)
    assert np.max(gradient_bound) < 1e-6 * np.max(np.abs(gradient))


def test_table_leverages_are_bitwise_identical_for_any_worker_count(monkeypatch):
    """Fixed row blocks, one fixed order per row: the pooled result is the serial one, bit for bit."""
    monkeypatch.setattr(leverage_tables, "_BLOCK_ROWS", 64)
    started = []
    pool = leverage_tables.ThreadPoolExecutor

    def spy(*args, **kwargs):
        started.append(kwargs["max_workers"])
        return pool(*args, **kwargs)

    monkeypatch.setattr(leverage_tables, "ThreadPoolExecutor", spy)
    model = _poisson_model(1.0e3)
    args, kwargs = _correction_inputs(model)
    centre_hi, centre_lo = _centre(model)
    plan = leverage_tables.lookup_plan(model._dm)
    inverse = args[4]
    active = np.ones(model._dm.n, dtype=bool)
    results = {}
    for workers in (1, 2, 3, 8):
        with parallel_config(n_jobs=workers):
            leverage = leverage_tables.table_leverages(plan, inverse, centre_hi, centre_lo, active)
            gradient = w_derivatives.reml_w_correction(*args, leverage_hessian=True, **kwargs)[0]
        results[workers] = leverage.tobytes() + gradient.tobytes()

    assert set(started) == {2, 3, 8}  # the pooled path ran at every count above one
    assert len(set(results.values())) == 1


def test_a_lookup_design_makes_no_solved_row_pass(monkeypatch):
    """The dispatch: every iteration reads the tables, none solves a row (v0.40.0 solved one per iteration)."""
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    counts = _count_row_passes(monkeypatch)
    tables = {"passes": 0}
    read_back = w_derivatives.table_leverages

    def counted(*args, **kwargs):
        tables["passes"] += 1
        return read_back(*args, **kwargs)

    monkeypatch.setattr(w_derivatives, "table_leverages", counted)
    model = _poisson_model(1.0e3)

    assert counts["calls"] >= 2
    assert counts["grams"] == 0
    assert counts["passes"] == 0
    assert tables["passes"] == counts["calls"]
    assert model._reml_profile["reml_w_correction_mode"] == "leverage_tables"


def _continuous_model() -> SuperGLM:
    rng = np.random.default_rng(5)
    n = 1500
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0.0, 1.0, n),
            "b": rng.uniform(-2.0, 2.0, n),
            "d": rng.normal(1.0e3, 4.0, n),
            "e": rng.normal(size=n),
        }
    )
    y = rng.poisson(np.exp(-0.5 + np.sin(3.0 * frame.a) + 0.1 * frame.e)).astype(np.float64)
    return SuperGLM(
        family="poisson",
        features={"a": Spline(n_knots=6), "b": Spline(n_knots=6), "d": Numeric(), "e": Numeric()},
        interactions=[("a", "b")],
        discrete=False,
    ).fit_reml(frame, y, runtime_validation="skip")


def test_a_design_without_lookups_keeps_the_solved_row_route(monkeypatch):
    """Continuous splines (sparse, not lookups) and dense columns: no plan, one solved-row pass an iteration."""
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    counts = _count_row_passes(monkeypatch)
    plans = []
    plan = w_derivatives.lookup_plan

    def recorded(*args, **kwargs):
        plans.append(plan(*args, **kwargs))
        return plans[-1]

    monkeypatch.setattr(w_derivatives, "lookup_plan", recorded)
    model = _continuous_model()

    assert all(type(gm) is not CategoricalGroupMatrix for gm in model._dm.group_matrices)
    assert plans and all(entry is None for entry in plans)
    assert counts["passes"] == counts["calls"] == len(plans)
    assert model._reml_profile["reml_w_correction_mode"] == "leverage_hessian"
    dense_only = type(model._dm)(
        [gm for gm in model._dm.group_matrices if type(gm) is DenseGroupMatrix],
        n=model._dm.n,
        p=sum(gm.shape[1] for gm in model._dm.group_matrices if type(gm) is DenseGroupMatrix),
    )
    assert plan(dense_only) is None
