"""Leverages read back from term-pair tables (``_group_matrix/_lookup_tables.py``).

A design whose groups are lookups (categoricals, support-compressed splines,
tensors, spline-by-category levels) and dense columns reads the exact REML
gradient's leverages ``h_r = x_r' H^-1 x_r`` from per-pair tables instead of
solving a row per data row.  These tests pin the leverages against the dense
design within the bound the module derives in ``u``, the gradient against the
derivative-Gram route, bitwise determinism across worker counts, the dispatch
(solved rows only at a candidate stop, whose second-order check reads v0.40.0's
Hessian), and that a design of no lookups keeps the solved-row route.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm._group_matrix._lookup_tables as leverage_tables
import superglm.reml.direct as direct
import superglm.reml.w_derivatives as w_derivatives
from superglm import Categorical, Numeric, Spline, SuperGLM
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
    _penalty_operands,
    _poisson_frame,
    _poisson_model,
    _rounding_bounds,
    _term_scales,
)


def _count_roundings(plan) -> int:
    """``K = 3p + S + 2``, the roundings along one leverage's longest chain (module docstring).

    ``3p + 1`` for a table entry's bilinear form, ``S`` for the row's sum,
    and one for a dense column's last product ``d_j (partial + d_j V_jj)``."""
    t, n_d = len(plan.terms), int(plan.dense_columns.size)
    return 3 * plan.p + 1 + (1 + t + plan.n_pairs + n_d * (t + n_d + 3)) + 1


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

    A support group's row is ``B[k] R``: its magnitude is ``(|B| |R|)[k]``
    plus the anchor row's ``(|B| |R|)[a]`` (bounded by the column maximum).
    The tables read ``(B[k] - B[a]) R - (c - B[a] R)``, which rounds within
    ``gamma_(w+1) (|B[k]| + |B[a]|) |R|`` for the product and ``gamma_w
    |B[a]| |R|`` for the offset (``w`` the basis width), and the other route's
    ``B[k] R`` within ``gamma_w`` of the first term.  A raw group (categorical,
    spline level, unanchored) reads its centre separately from its row, so it
    carries ``|c|`` twice.
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
            support = np.abs(gm.B_unique) @ np.abs(gm.R_inv)
            rows.append(support[gm.bin_idx] + support.max(axis=0))
            widest = max(widest, gm.B_unique.shape[1])
        column += width
    magnitude = np.hstack(rows) + np.where(dense, 0.0, np.where(raw, 2.0, 1.0)) * np.abs(centre)
    return magnitude, raw, widest


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
@pytest.mark.parametrize("offset", [1.0e3, 1.0e9])
def test_table_leverages_match_the_dense_design_within_their_bound(monkeypatch, offset, pair_form):
    """``h`` from the tables against ``x_r' V x_r`` on the materialised centred rows.

    Bound: the table route is within ``gamma_K eta_r`` (``K =
    _count_roundings(plan)``, module docstring); the dense reference
    within ``gamma_(2p+2) eta_r``; the two routes' support rows round apart
    by ``gamma_(w+1)`` of the anchored magnitudes ``_magnitudes`` carries,
    adding ``4 gamma_w`` (covered by ``4 w + 4``).  A
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
    k = _count_roundings(plan) + 2 * dm.p + 2 + 4 * widest + 4
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
        model, raw_centre_copies=2, extra_roundings=_count_roundings(plan)
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


def _force_candidate_stops(monkeypatch, *, first: int = 1) -> None:
    """Every iteration from ``first`` on is a candidate stop whose mode counts as certified.

    Whether a fit's own trajectory reaches the score test's stop depends on
    round-off (a fixture can end by another exit on another BLAS), so the
    stop tests force the event and assert what the stop does given it.
    """
    import dataclasses

    evaluate = direct.evaluate_reml_candidate

    def forced(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        return dataclasses.replace(result, converged=kwargs["iteration"] >= first)

    monkeypatch.setattr(direct, "evaluate_reml_candidate", forced)
    monkeypatch.setattr(direct, "trial_counts_as_precision_evidence", lambda *a, **k: True)


def test_a_lookup_design_solves_rows_only_at_a_candidate_stop(monkeypatch):
    """The dispatch: every iteration reads the tables; one solved-row pass per candidate stop.

    The stop's second-order check reads the W(rho) cross terms the tables do
    not form, so each candidate stop (one ``concave_live_directions`` read)
    makes exactly one solved-row pass and no other point makes any.  v0.40.0
    solved one row pass per iteration and read no table; the PR head made
    none at the stop.  The second iteration is forced to be a candidate stop.
    """
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    _force_candidate_stops(monkeypatch, first=2)
    counts = _count_row_passes(monkeypatch)
    tables = {"passes": 0}
    stops = []
    read_back = w_derivatives.table_leverages
    concave = direct.concave_live_directions

    def counted(*args, **kwargs):
        tables["passes"] += 1
        return read_back(*args, **kwargs)

    def stop(*args, **kwargs):
        stops.append(counts["passes"])
        return concave(*args, **kwargs)

    monkeypatch.setattr(w_derivatives, "table_leverages", counted)
    monkeypatch.setattr(direct, "concave_live_directions", stop)
    model = _poisson_model(1.0e3)

    assert stops and tables["passes"] >= 2
    assert counts["grams"] == 0
    # one pass made just before each stop's read, none elsewhere
    assert stops == list(range(1, len(stops) + 1))
    assert counts["passes"] == len(stops)
    assert tables["passes"] == counts["calls"] - len(stops)
    assert model._reml_profile["reml_n_stop_cross_passes"] == len(stops)
    assert model._reml_profile["reml_w_correction_mode"] == "leverage_tables"


def _record_stops(monkeypatch, *, refuse_first: bool = False) -> tuple[list[dict], list]:
    """At each candidate stop: the Hessian its checks read, and v0.40.0's at the same point.

    v0.40.0's leverage route formed the same Hessian with its own ``w_cross``
    (``reml_w_correction(leverage_hessian=True)`` on the solved rows), recomputed
    here from the iteration's own correction inputs with the table route off.
    ``refuse_first`` reports every estimated direction live at the first stop,
    so that stop is refused.
    """
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    corrections, hessians, reads, ranks = [], [], [], []
    correction, hessian = direct.reml_w_correction, direct.reml_direct_hessian
    concave, rank_of = direct.concave_live_directions, direct.direction_penalty_ranks

    def recorded_correction(*args, **kwargs):
        if kwargs.get("lookup_tables", True):
            corrections.append((args, kwargs))
        return correction(*args, **kwargs)

    def recorded_hessian(*args, **kwargs):
        hessians.append((args, kwargs))
        return hessian(*args, **kwargs)

    def recorded_ranks(*args, **kwargs):
        ranks.append(rank_of(*args, **kwargs))
        return ranks[-1]

    def read(gradient, hess, estimated, **kwargs):
        args, correction_kwargs = corrections[-1]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(w_derivatives, "lookup_plan", lambda *a, **k: None)
            cross = w_derivatives.reml_w_correction(*args, **correction_kwargs)[2]
        hessian_args, hessian_kwargs = hessians[-1]
        reference = hessian(*hessian_args, **{**hessian_kwargs, "dH2_cross": cross})
        reads.append(
            {
                "gradient": np.array(gradient, copy=True),
                "hessian": np.array(hess, copy=True),
                "reference": reference,
                "cross": cross,
                "estimated": np.array(estimated, copy=True),
                "kwargs": kwargs,
            }
        )
        live = concave(gradient, hess, estimated, **kwargs)
        return np.asarray(estimated, dtype=bool) if refuse_first and len(reads) == 1 else live

    monkeypatch.setattr(direct, "reml_w_correction", recorded_correction)
    monkeypatch.setattr(direct, "reml_direct_hessian", recorded_hessian)
    monkeypatch.setattr(direct, "concave_live_directions", read)
    monkeypatch.setattr(direct, "direction_penalty_ranks", recorded_ranks)
    return reads, ranks


def test_the_stop_reads_the_leverage_routes_hessian(monkeypatch):
    """At a candidate stop the table route's Hessian is v0.40.0's leverage route's, bit for bit.

    Both add the same solved-row ``w_cross`` to the same fixed-W Hessian:
    the same float64 operations on the same operands, so the derived bound
    is zero.  On the PR head the stop read the Hessian without ``w_cross``,
    whose diagonal ``-P_ii`` moved entries by up to 8% on this fixture.
    """
    _force_candidate_stops(monkeypatch)
    reads, _ = _record_stops(monkeypatch)
    model = _poisson_model(1.0e3)

    assert model._reml_profile["reml_w_correction_mode"] == "leverage_tables"
    assert reads
    for read in reads:
        assert np.any(read["cross"] != 0.0)
        assert np.array_equal(read["hessian"], read["reference"])


def test_a_refused_stop_steps_on_the_stops_hessian(monkeypatch):
    """A stop the cross terms refuse takes its Newton step, and that step's freeze, on their Hessian.

    Stepping on the gradient-only Hessian refroze what the stop's freeze
    revalidation had reactivated, so the next stop was refused the same way:
    on the 10-pair freMTPL2 fit two directions straddling the freeze bar
    held their gradients through ten refused stops (16 exact iterations
    against 7).  The first stop is refused here; both freeze decisions made
    at that iteration read v0.40.0's Hessian.  Every iteration from the
    first is forced to be a candidate stop.
    """
    _force_candidate_stops(monkeypatch)
    reads, _ = _record_stops(monkeypatch, refuse_first=True)
    freezes = []
    freeze = direct.freeze_flat_directions

    def recorded_freeze(gradient, hess, *args, **kwargs):
        freezes.append((len(reads), np.array(hess, copy=True)))
        return freeze(gradient, hess, *args, **kwargs)

    monkeypatch.setattr(direct, "freeze_flat_directions", recorded_freeze)
    model = _poisson_model(1.0e3)

    assert model._reml_profile["reml_n_concave_stop_refusals"] >= 1
    # the stop's revalidation, then the step's own freeze (later iterations follow)
    at_refusal = [hess for stop, hess in freezes if stop == 1][:2]
    assert len(at_refusal) == 2
    for hess in at_refusal:
        assert np.array_equal(hess, reads[0]["reference"])


def test_an_aliased_spline_by_category_stop_decides_as_the_leverage_route(monkeypatch):
    """A forced stop on spline-by-category levels aliased with their factor level.

    Levels 0 and 1 of ``c`` see a single ``e`` each, so their ``e:c``
    smooths are aliased with the level's constant (the spline-by-factor case
    of ``reml/convergence.py``) and sit on the lambda -> infinity plateau.
    At every stop the concavity and freeze decisions are the ones v0.40.0's
    leverage-route Hessian gives at that point.  That holds by construction
    once the Hessians are equal (``test_the_stop_reads_the_leverage_routes_hessian``);
    this fixture adds the aliased design.  The Hessian without the cross
    terms decides the same way here (measured on x86-64 OpenBLAS, default,
    Haswell and Zen kernels), as on every small fixture searched (Poisson
    and Gamma, eight seeds, aliased levels, plateau starts), so this test
    would pass on 1da3925d too: the case where the decisions differ is the
    10-pair freMTPL2 fit, two directions straddling the freeze bar
    (e62f0c85), too large for the suite.
    """
    from superglm.reml.convergence import concave_live_directions, freeze_flat_directions

    _force_candidate_stops(monkeypatch)
    reads, ranks = _record_stops(monkeypatch)
    frame, y, weights = _poisson_frame(1500, 1.0e3)
    frame.loc[frame.c == "0", "e"] = 1.3
    frame.loc[frame.c == "1", "e"] = 0.4
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=[("a", "b"), ("e", "c")],
        weight_semantics="frequency",
        discrete=False,
    ).fit_reml(frame, y, sample_weight=weights, runtime_validation="skip")

    assert model._reml_profile["reml_w_correction_mode"] == "leverage_tables"
    assert reads and ranks
    for read in reads:
        gradient, estimated, kwargs = read["gradient"], read["estimated"], read["kwargs"]
        decisions = []
        for hess in (read["hessian"], read["reference"]):
            freeze = freeze_flat_directions(gradient, hess, ranks[-1], estimated, **kwargs)
            decisions.append(
                (
                    concave_live_directions(gradient, hess, estimated, **kwargs).tolist(),
                    np.asarray(freeze.frozen).tolist(),
                    np.asarray(freeze.shoulder).tolist(),
                )
            )
        assert decisions[0] == decisions[1]


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


def _hand_built_design(
    n: int = 2 * 8192 + 4096,
    *,
    empty_level: bool = False,
    seed: int = 17,
    basis: int = 6,
    random_effect: bool = False,
):
    """Two dense groups (three columns, offset), two categoricals and a support spline, by hand.

    Large enough for the table Gram's row rule and three leverage row blocks.
    ``empty_level`` puts a zero-width categorical between the groups;
    ``basis=100`` makes the design wide enough for the table Gram's column rule;
    ``random_effect`` makes the last categorical a random effect over every level.
    """
    from superglm.group_matrix import (
        DesignMatrix,
        RandomEffectGroupMatrix,
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(seed)
    support = max(40, 2 * basis)
    grid = np.linspace(0.0, 1.0, support)
    knots = np.linspace(0.0, 1.0, basis)
    B = np.maximum(0.0, 1.0 - np.abs(grid[:, None] - knots[None, :]) * (basis - 1))
    spline = SupportCompressedSSPGroupMatrix(
        B, rng.normal(size=(basis, basis - 1)), rng.integers(0, support, n)
    )
    groups = [
        DenseGroupMatrix(rng.normal(1.0e3, 4.0, (n, 2))),
        CategoricalGroupMatrix(rng.integers(-1, 5, n), 5),
        spline,
        DenseGroupMatrix(rng.normal(-50.0, 2.0, (n, 1))),
        RandomEffectGroupMatrix(rng.integers(0, 3, n), 3)
        if random_effect
        else CategoricalGroupMatrix(rng.integers(-1, 3, n), 3),
    ]
    if empty_level:
        groups.insert(2, CategoricalGroupMatrix(np.full(n, -1), 0))
    return DesignMatrix(groups, n=n, p=sum(gm.shape[1] for gm in groups))


def _hand_built_magnitudes(dm, centre: np.ndarray) -> tuple[np.ndarray, int]:
    """``_magnitudes`` for the hand-built design: ``|B| |R|`` rows plus the anchor row (the
    column maximum) and ``|c|``, raw rows plus ``2 |c|``, dense rows ``|x - c|``; the widest basis."""
    rows, column, widest = [], 0, 1
    for gm in dm.group_matrices:
        cols = slice(column, column + gm.shape[1])
        if type(gm) is DenseGroupMatrix:
            rows.append(np.abs(gm.M - centre[cols]))
        elif isinstance(gm, CategoricalGroupMatrix):
            rows.append(np.abs(gm.toarray()) + 2.0 * np.abs(centre[cols]))
        else:
            support = np.abs(gm.B_unique) @ np.abs(gm.R_inv)
            rows.append(support[gm.bin_idx] + support.max(axis=0) + np.abs(centre[cols]))
            widest = max(widest, gm.B_unique.shape[1])
        column += gm.shape[1]
    return np.hstack(rows), widest


def _hand_built_operands(dm, seed: int = 4):
    """A symmetric positive definite inverse, a centre pair and active rows for the hand-built design."""
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(dm.p, dm.p))
    inverse = root @ root.T / dm.p + np.eye(dm.p)
    weights = rng.uniform(0.1, 2.0, dm.n)
    design = dm.toarray()
    mean = design.T @ weights / weights.sum()
    dense = w_derivatives.dense_columns(dm)
    centre_lo = np.where(dense, rng.normal(scale=1e-3, size=dm.p), 0.0)
    active = rng.uniform(size=dm.n) > 0.1
    return inverse, mean, centre_lo, active


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
def test_hand_built_dense_columns_match_the_materialised_design(monkeypatch, pair_form):
    """Two dense groups, three dense columns, three row blocks: ``x_r' V x_r`` and ``X_c' v``.

    Every dense-to-dense path runs here (``Vdd``'s off-diagonal ``2 V_ij d_i d_j``
    across groups, blocks after the first), which the model fixtures, one dense
    column each, never reach.  Bounds as in
    ``test_table_leverages_match_the_dense_design_within_their_bound``; ``X_c' v``
    is within ``gamma_(n + m + 3) sum_r |v_r| M_r`` on either route.
    Mutations (each fails here): ``Vdd`` without the factor 2; the dense loop over
    ``range(j + 1)``.
    """
    dm = _hand_built_design()
    monkeypatch.setattr(leverage_tables, "_FORCE_PAIR_FORM", pair_form)
    plan = leverage_tables.lookup_plan(dm)
    assert plan is not None and plan.dense_columns.size == 3 and len(plan.dense) == 2
    assert dm.n > 2 * leverage_tables._BLOCK_ROWS
    inverse, centre_hi, centre_lo, active = _hand_built_operands(dm)

    leverage = leverage_tables.table_leverages(plan, inverse, centre_hi, centre_lo, active)
    centred = (dm.toarray() - centre_hi) - centre_lo
    reference = np.where(active, np.einsum("ij,jk,ik->i", centred, inverse, centred), 0.0)
    magnitude, widest = _hand_built_magnitudes(dm, centre_hi + centre_lo)
    eta = np.einsum("ij,jk,ik->i", magnitude, np.abs(inverse), magnitude)
    bound = _gamma(_count_roundings(plan) + 2 * dm.p + 2 + 4 * widest + 4) * eta
    assert np.all(np.abs(leverage - reference) <= bound)
    assert np.max(bound[active] / np.abs(reference[active])) < 1e-9

    values = np.where(active, np.random.default_rng(9).normal(size=dm.n), 0.0)
    product = leverage_tables.table_rmatvec(plan, values, centre_hi, centre_lo)
    m_largest = max(term.raw.shape[0] for term in plan.terms)
    rmatvec_bound = _gamma(dm.n + m_largest + 4 * widest + 3) * (magnitude.T @ np.abs(values))
    assert np.all(np.abs(product - centred.T @ values) <= rmatvec_bound)


def test_a_random_effect_is_a_lookup_on_both_leverage_routes():
    """A random effect is a categorical over every level, its codes never the sink code.

    The table plan and the solved-row support route both take it, and both
    give ``x_r' V x_r`` of the materialised design within their bounds: the
    table route's of ``test_hand_built_dense_columns_match_the_materialised_design``,
    and the support route's ``gamma_(2p + G + 2)`` of the same magnitudes
    (``_leverage_rows``, module docstring: ``K = p + G + 2`` for the solved row,
    ``p`` more for its leverage).  An exact type check sent the subclass to
    neither: no plan, and the chunked rows.
    """
    import superglm.reml._leverage_rows as leverage_rows

    dm = _hand_built_design(random_effect=True)
    plan = leverage_tables.lookup_plan(dm)
    assert plan is not None
    inverse, centre_hi, centre_lo, active = _hand_built_operands(dm)
    centred = (dm.toarray() - centre_hi) - centre_lo
    reference = np.where(active, np.einsum("ij,jk,ik->i", centred, inverse, centred), 0.0)
    magnitude, widest = _hand_built_magnitudes(dm, centre_hi + centre_lo)
    eta = np.einsum("ij,jk,ik->i", magnitude, np.abs(inverse), magnitude)

    leverage = leverage_tables.table_leverages(plan, inverse, centre_hi, centre_lo, active)
    bound = _gamma(_count_roundings(plan) + 2 * dm.p + 2 + 4 * widest + 4) * eta
    assert np.all(np.abs(leverage - reference) <= bound)

    blocks = leverage_rows.solved_row_blocks(dm, inverse, centre_hi, centre_lo, active)
    assert blocks.block_bytes is not None  # the support route
    solved = np.zeros(dm.n)
    for job in blocks.jobs:
        block = job()
        if block is not None:
            rows, _, values = block
            solved[rows] = values
    support_bound = _gamma(2 * dm.p + len(dm.group_matrices) + 2) * eta
    assert np.all(np.abs(solved - reference) <= support_bound)


class _DoubledCategorical(CategoricalGroupMatrix):
    """A categorical whose design is twice its codes' one-hot rows, in every operation."""

    __slots__ = ()

    def matvec(self, v):
        return 2.0 * super().matvec(v)

    def rmatvec(self, w):
        return 2.0 * super().rmatvec(w)

    def gram(self, W):
        return 4.0 * super().gram(W)

    def toarray(self):
        return 2.0 * super().toarray()

    def row_subset(self, idx):
        return _DoubledCategorical(self.codes[idx], self.n_levels)


def test_a_categorical_subclass_keeps_the_routes_that_read_its_own_algebra():
    """A categorical subclass may change the algebra its codes imply, so no route reads its codes.

    Only the exact types of ``ONE_HOT_CODE_TYPES`` read codes: neither the
    table plan nor the solved-row support route takes the subclass, and both
    leverage routes fall back to chunked solved rows of its own ``toarray``.
    Their gradient is the exact ``sum_r w'_r (x_r' d) (h_r + 1/sum_w) / 2``, ``d
    = -H^-1 S beta``, of the materialised design within ``2 gamma_K E``
    (``_rounding_bounds``' ``E`` and ``K``).  Mutation: ``isinstance`` in place
    of the exact types sends the subclass to both tables, and the gradient
    changes sign.
    """
    import superglm.reml._leverage_rows as leverage_rows
    from superglm.distributions import Poisson
    from superglm.group_matrix import DesignMatrix
    from superglm.links import LogLink
    from superglm.solvers.pirls import PIRLSResult, REMLGeometrySummary
    from superglm.types import GroupSlice, PenaltyComponent

    n = 120
    level = _DoubledCategorical(np.tile([-1, 0, 1], n // 3), 2)
    dm = DesignMatrix([level], n=n, p=2)
    beta, intercept, link = np.array([0.2, -0.3]), -0.4, LogLink()
    design = level.toarray()
    eta = design @ beta + intercept
    mu = link.inverse(eta)  # the Poisson/log working weights
    sum_w = float(np.sum(mu))
    mean_x = mu @ design / sum_w
    centred = design - mean_x
    inverse = np.linalg.inv(centred.T @ (mu[:, None] * centred) + 2.0 * np.eye(2))
    group = GroupSlice(name="level", start=0, end=2)
    penalty = PenaltyComponent(
        name="level",
        group_name="level",
        group_index=0,
        group_sl=group.sl,
        omega_raw=None,
        penalty_kind="identity",
    )
    lambdas = {"level": 2.0}
    result = PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=1,
        deviance=0.0,
        converged=True,
        phi=1.0,
        effective_df=2.0,
        reml_geometry=REMLGeometrySummary(mean_x=mean_x, sum_w=sum_w, column_scale=np.ones(2)),
    )
    case = {
        "dm": dm,
        "link": link,
        "groups": [group],
        "pirls_result": result,
        "XtWX_S_inv": inverse,
        "lambdas": lambdas,
        "sample_weight": np.ones(n),
        "offset_arr": np.zeros(n),
        "distribution": Poisson(),
        "reml_penalties": [penalty],
    }
    weight_derivative = w_derivatives.compute_dW_deta(link, Poisson(), mu, eta, np.ones(n))
    smoothing, penalty_rhs = _penalty_operands(dm, [penalty], lambdas, beta)
    direction = -inverse @ penalty_rhs[:, 0]
    leverage = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    reference = 0.5 * np.sum(weight_derivative * (centred @ direction) * (leverage + 1.0 / sum_w))
    magnitude = np.abs(centred) + np.abs(mean_x)
    gradient_scale, _ = _term_scales(
        magnitude, inverse, weight_derivative, penalty_rhs, smoothing, sum_w
    )
    bound = 2.0 * _gamma(n + 4 * dm.p + 16) * gradient_scale

    for lookup_tables in (True, False):
        gradient, _, cross = w_derivatives.reml_w_correction(
            **case, leverage_hessian=True, lookup_tables=lookup_tables
        )
        assert np.all(np.abs(gradient - reference) <= bound)
        assert cross is not None  # solved rows, never the tables
    assert leverage_tables.lookup_plan(dm) is None
    assert leverage_rows._support_plan(dm, inverse, mean_x, None) is None


def test_an_empty_categorical_adds_nothing_to_the_tables():
    """A zero-width categorical (``n_levels=0``) reads a ``(1, 0)`` table, which is no term.

    The leverages, ``X_c' v`` and the centred Gram of a design that takes the
    table route by the row rule are, bit for bit, those of the same design
    without it.  On the PR head ``_windows`` raised in ``argmax`` on it.
    """
    from superglm.solvers.centered_system import build_centered_system

    with_empty = _hand_built_design(empty_level=True, basis=100)
    without = _hand_built_design(basis=100)
    assert with_empty.n >= leverage_tables._MIN_TABLE_GRAM_ROWS
    inverse, centre_hi, centre_lo, active = _hand_built_operands(without)
    W, z = np.random.default_rng(2).uniform(0.1, 2.0, without.n), np.zeros(without.n)
    results = []
    for dm in (with_empty, without):
        plan = leverage_tables.table_gram_plan(dm)
        assert plan is not None
        system = build_centered_system(dm=dm, W=W, z_off=z, penalty=np.eye(dm.p))
        leverage = leverage_tables.table_leverages(plan, inverse, centre_hi, centre_lo, active)
        product = leverage_tables.table_rmatvec(plan, leverage, centre_hi, centre_lo)
        results.append((system.data_gram.tobytes(), leverage.tobytes(), product.tobytes()))
    assert results[0] == results[1]


def test_the_plan_is_cached_for_a_spline_by_category_design():
    """Every level's rows and codes are owners in their own right: the cached plan is reused.

    On the PR head the level entered the owners as a tuple built per call,
    never ``is`` the cached one, so every call rebuilt the plan.
    """
    dm = _poisson_model(1.0e3)._dm
    assert any(isinstance(gm, DiscretizedSplineCategoricalGroupMatrix) for gm in dm.group_matrices)
    assert leverage_tables.lookup_plan(dm) is leverage_tables.lookup_plan(dm)


def test_replacing_a_dense_column_rebuilds_the_plan():
    """The plan holds the dense values it was built from; a replaced ``M`` invalidates it."""
    dm = _hand_built_design()
    plan = leverage_tables.lookup_plan(dm)
    assert plan is not None and leverage_tables.lookup_plan(dm) is plan
    dense = dm.group_matrices[0]
    dense.M = dense.M + 1.0
    rebuilt = leverage_tables.lookup_plan(dm)
    assert rebuilt is not plan and rebuilt.dense[0][2] is dense.M


def test_a_support_outside_the_reassociation_range_keeps_the_old_routes():
    """``B R`` is not formed when a factor lies outside ``[2**-128, 2**128]``.

    The established support routes take exact moments there
    (``_ssp_gram_needs_exact``); the table plan declines the design, so both
    the Gram and the leverages keep them.  Here ``B`` carries ``2**600`` and
    ``R`` ``2**-600``: the represented row is ordinary, yet ``B`` alone is
    out of range.  On the PR head the plan admitted it.
    """
    from superglm.group_matrix import DesignMatrix, SupportCompressedSSPGroupMatrix
    from superglm.solvers.centered_system import build_centered_system

    base = _hand_built_design()
    spline = base.group_matrices[2]
    scaled = SupportCompressedSSPGroupMatrix(
        spline.B_unique * 2.0**600, spline.R_inv * 2.0**-600, spline.bin_idx
    )
    groups = list(base.group_matrices)
    groups[2] = scaled
    dm = DesignMatrix(groups, n=base.n, p=base.p)
    assert leverage_tables.lookup_plan(dm) is None
    assert leverage_tables.table_gram_plan(dm) is None
    W = np.random.default_rng(2).uniform(0.1, 2.0, dm.n)
    system = build_centered_system(dm=dm, W=W, z_off=np.zeros(dm.n), penalty=np.eye(dm.p))
    assert np.all(np.isfinite(system.data_gram))


def test_the_plan_is_released_when_the_fit_publishes_its_state():
    """A fitted model keeps no plan (its ``(T, n)`` index and raw tables).

    The table Gram is forced on, so the fit's last centred systems come from
    the tables on the published design; on the PR head their plan stayed in
    that design's layout cache.  Predict, the diagnostics, metrics and a refit still work.
    """
    n = leverage_tables._MIN_TABLE_GRAM_ROWS
    frame, y, weights = _poisson_frame(n, 1.0e3)
    built = []
    build = leverage_tables._build_plan

    def recorded(dm):
        built.append(dm)
        return build(dm)

    with pytest.MonkeyPatch.context() as patch, leverage_tables.table_gram_config(True):
        patch.setattr(leverage_tables, "_build_plan", recorded)
        model = _poisson_model(1.0e3, n=n)
    assert any(dm is model._dm for dm in built)
    assert leverage_tables._CACHE_KEY not in model._dm._structured_layout_cache
    assert np.all(np.isfinite(model.predict(frame)))
    assert model.reml_diagnostics()["converged"]
    assert np.isfinite(model.metrics(frame, y, sample_weight=weights).deviance)
    refit = model.fit_reml(frame, y, sample_weight=weights, runtime_validation="skip")
    assert refit._reml_result.converged
    assert leverage_tables._CACHE_KEY not in refit._dm._structured_layout_cache


@pytest.mark.parametrize("forced", [False, True])
def test_a_plain_fit_keeps_no_plan(forced):
    """``fit()`` publishes through the same hook as ``fit_reml()``: no plan stays on the design.

    Unforced, the 98-column fixture is below the table Gram's crossover, which
    the group types settle (six terms and a dense block: 28 sweeps at least, so
    168 columns), and no plan is built at all; on
    e62f0c85 the plan was built to count the sweeps and kept.  Forced on, the
    fit's Grams come from the tables and the plan is released when ``fit()``
    publishes; on e62f0c85 only ``fit_reml()`` released it.
    """
    n = leverage_tables._MIN_TABLE_GRAM_ROWS
    frame, y, weights = _poisson_frame(n, 1.0e3)
    built = []
    build = leverage_tables._build_plan

    def recorded(dm):
        built.append(dm)
        return build(dm)

    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=[("a", "b"), ("e", "c")],
        weight_semantics="frequency",
        discrete=False,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leverage_tables, "_build_plan", recorded)
        with leverage_tables.table_gram_config(True if forced else None):
            model.fit(frame, y, sample_weight=weights)
    assert bool(built) == forced
    assert leverage_tables._CACHE_KEY not in model._dm._structured_layout_cache
