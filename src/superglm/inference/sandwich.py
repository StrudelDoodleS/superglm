"""Opt-in robust (sandwich) covariance estimation for ``SuperGLM``."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.special import expit

from superglm._weight_units import scale_penalties
from superglm.distributions import (
    _VARIANCE_FLOOR,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import DesignMatrix
from superglm.inference._sandwich_core import (
    BreadCurvature,
    ClusterScoreAccumulator,
    FitBinding,
    SandwichCovariance,
    SandwichTelemetry,
    ScaledGramAccumulator,
    build_available_covariance,
    build_unavailable_covariance,
    coef_digest,
    normalize_clusters,
    resolve_center_mode,
    resolve_workspace_budget,
    small_sample_multiplier,
)
from superglm.inference.covariance import _active_penalty_components, _active_penalty_matrix
from superglm.links import (
    CloglogLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    ProbitLink,
    SqrtLink,
    _declared_with,
    stabilize_eta,
)
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.model.report_ops import shape_repaired_feature_names
from superglm.model.retired_state import retained_linear_state
from superglm.model.separated_levels import separated_coefficients
from superglm.model.state_ops import (
    _grouped_active_state,
    _names_an_unpriced_level,
    _public_intercept_shift,
    _rank_active_state,
)
from superglm.penalties.base import (
    penalty_can_zero_groups,
    penalty_targets_group,
    selection_shrunk_group_names,
)
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.ridge import Ridge
from superglm.reml.observed_geometry import (
    ObservedGeometryInfeasibleError,
    _builtin_fisher_equals_observed,
    _stable_signed_mean_pair,
    classify_reml_curvature,
    compute_observed_information_weights,
    observed_row_error_scale,
    schur_curvature_is_negative,
)
from superglm.solvers._structured.state import StructuredLinearSystemState
from superglm.solvers.centered_system import (
    _NO_CENTRE,
    _design_leaf_parts,
    build_centered_system,
    grouped_augmented_factor,
    weighted_mean_pair,
)
from superglm.solvers.dispersion import FREQUENCY_WEIGHTS, model_weight_semantics
from superglm.solvers.mode_score import linear_predictor
from superglm.solvers.rank import (
    RankDecomposition,
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    selected_group_name_set,
)
from superglm.solvers.working_rows import (
    _cloglog_observed_rows,
    _probit_observed_rows,
    fisher_working_weights,
)

_MAX_CHUNK_ROWS = 8192
_TARGET_CHUNK_BYTES = 16 << 20  # 16 MiB per replay chunk


_UNBOUND_MESSAGE = (
    "This model keeps no record of the rows it was fitted on, so a sandwich covariance "
    "cannot be tied to it; refit the model."
)


def _dense_workspace_bytes(
    q: int, n_clusters: int | None, chunk_rows: int, *, bread_columns: int = 0
) -> int:
    """Bytes of the dense float64 arrays one request holds at once, each at its peak.

    Five ``q x q`` arrays while the Gram is restored to covariance units (the
    accumulator, the result and the temporaries between them); six square
    arrays of ``bread_columns`` for a bread formed here (penalty, Gram, Hessian
    and its factorization); the two ``G x q`` cluster buffers; and six
    ``chunk_rows x q`` arrays in a replay chunk (the scores, their
    positive-weight copy, the solve's right-hand side, solution and result, and
    the scaled copy the Gram takes).  The fit's per-row vectors (predictor,
    mean, score, curvature) are its own size and are not dense workspace.
    """
    cluster = 16 * n_clusters * q if n_clusters else 0
    return 40 * q * q + 48 * bread_columns * bread_columns + cluster + 48 * chunk_rows * q


def _settings(obj: Any) -> str:
    """``obj``'s class and the scalar settings it holds (a family's ``p``, a penalty's ``features``).

    Collections of scalars are listed by ``repr`` and sorted, so a set's iteration order
    (which string hashing varies between processes) cannot move the text.
    """
    cls = type(obj)
    items = []
    for key, value in sorted(getattr(obj, "__dict__", {}).items()):
        if value is None or isinstance(value, bool | int | float | str):
            items.append(f"{key}={value!r}")
        elif isinstance(value, list | tuple | set | frozenset) and all(
            isinstance(v, bool | int | float | str) for v in value
        ):
            members = [repr(v) for v in value]
            items.append(
                f"{key}={sorted(members) if isinstance(value, set | frozenset) else members}"
            )
    return f"{cls.__module__}.{cls.__qualname__}({', '.join(items)})"


def _specification_digest(model: Any) -> bytes:
    """A digest of what turns the fit's rows and coefficients into its covariance.

    The weight contract (``"prior"`` rows are one unit each, ``"frequency"`` rows ``w``
    units: different meats on the same rows), the family and link with their
    parameters, the penalty and the fitted smoothing parameters (the bread), and the
    coefficient layout.  The basis a term spans is not read: two designs give bit-equal
    coefficients on the same rows only where the response is exactly orthogonal to both.
    """
    lambda2 = fitted_lambda2(model)
    if isinstance(lambda2, dict):
        lambda2 = sorted((str(key), float(value)) for key, value in lambda2.items())
    reml = getattr(model, "_reml_lambdas", None)
    if isinstance(reml, dict):
        reml = sorted((str(key), float(value)) for key, value in reml.items())
    parts = (
        model_weight_semantics(model),
        _settings(model._distribution),
        _settings(model._link),
        _settings(fitted_penalty(model)),
        repr(lambda2),
        repr(reml),
        repr([(g.name, g.start, g.end, g.subgroup_type) for g in model._groups]),
    )
    return hashlib.blake2b("\x1f".join(parts).encode(), digest_size=16).digest()


def fit_binding_for_superglm(model: Any) -> FitBinding | None:
    """The ``FitBinding`` of ``model``'s current fit, or ``None`` where it recorded no rows.

    The rows are the fit's ``FitGeometryGuard``: the digests of the frame, response,
    weights and offset every fit writes and a saved model keeps.  The specification
    is ``_specification_digest``.
    """
    result = model._result
    if result is None:
        raise RuntimeError("Model must be fitted before building a covariance binding.")
    guard = getattr(model, "_fit_geometry_guard", None)
    if guard is None:
        return None
    rows = hashlib.blake2b(digest_size=16)
    rows.update(int(guard.n_rows).to_bytes(8, "little"))
    for part in (guard.x_digest, guard.y_digest, guard.weight_digest, guard.offset_digest):
        rows.update(part)
    state = getattr(model, "_fit_state", None)
    return FitBinding(
        model_kind="SuperGLM",
        data_digest=rows.digest(),
        specification_digest=_specification_digest(model),
        fit_revision=int(getattr(model, "_fit_revision", 0)),
        repair_revision=int(getattr(state, "repair_revision", 0)) if state is not None else 0,
        coef_digest=coef_digest(float(result.intercept), np.asarray(result.beta, dtype=np.float64)),
    )


def bound_covariance(model: Any, covariance: Any) -> SandwichCovariance:
    """``covariance`` once it is an available result of ``model``'s current fit; raises otherwise."""
    if not isinstance(covariance, SandwichCovariance):
        raise TypeError(
            f"covariance must be a SandwichCovariance or None, got {type(covariance).__name__}."
        )
    covariance.require_available()
    binding = fit_binding_for_superglm(model)
    if binding is None:
        raise ValueError(_UNBOUND_MESSAGE)
    return covariance.require_available(binding)


def _observed_score_and_curvature(
    distribution: Any,
    link: Any,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
) -> tuple[NDArray, NDArray, bool, BreadCurvature] | None:
    """Return unit-weight ``(score, observed_curvature, is_canonical, curvature_label)``.

    Returns ``None`` when the family/link combination lacks supported exact
    observed curvature derivatives.
    """
    n = len(y)
    ones = np.ones(n, dtype=np.float64)
    dist_type = type(distribution)
    link_type = type(link)

    # ── Canonical built-in pairs (observed == Fisher identically) ──
    if _builtin_fisher_equals_observed(distribution, link):
        if dist_type is Gaussian and link_type is IdentityLink:
            return y - mu, ones, True, "fisher"
        if dist_type is Poisson and link_type is LogLink:
            return y - mu, mu.copy(), True, "fisher"
        if dist_type is Binomial and link_type is LogitLink:
            # y log p + (1 - y) log(1 - p) at p = expit(eta): score y - p and curvature
            # p (1 - p), each a sum or product of non-negative terms in either tail.
            event, non_event = expit(eta), expit(-eta)
            return y * non_event - (1.0 - y) * event, event * non_event, True, "fisher"
        if dist_type is Gamma and link_type is InverseLink:
            h = fisher_working_weights(
                distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=ones
            )
            return mu - y, h, True, "fisher"
        dmu = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
        var = np.maximum(np.asarray(distribution.variance(mu), dtype=np.float64), _VARIANCE_FLOOR)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            s_unit = (y - mu) * dmu / var
        h = fisher_working_weights(
            distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=ones, dmu_deta=dmu
        )
        return s_unit, h, True, "fisher"

    # ── Tail-safe closed forms for built-in non-canonical pairs ────
    if link_type is LogLink:
        if dist_type is Gamma:
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                ratio = y / mu
                return ratio - 1.0, ratio, False, "observed"
        if dist_type is Tweedie:
            p = float(distribution.p)
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                mu_1mp = np.power(mu, 1.0 - p)
                mu_2mp = np.power(mu, 2.0 - p)
                s_unit = y * mu_1mp - mu_2mp
                h_unit = (2.0 - p) * mu_2mp + (p - 1.0) * y * mu_1mp
                return s_unit, h_unit, False, "observed"
        if dist_type is NegativeBinomial and distribution.theta != "auto":
            # theta (y - mu) / (theta + mu) and theta mu (theta + y) / (theta + mu)**2 as
            # products of ratios no larger than their operands: neither theta**2 nor
            # theta * mu is formed, and theta -> inf gives Poisson's y - mu and mu.
            theta = float(distribution.theta)
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                shrink = 1.0 / (1.0 + mu / theta)
                s_unit = (y - mu) * shrink
                h_unit = mu * shrink * ((theta + y) / (theta + mu))
                return s_unit, h_unit, False, "observed"
        if dist_type is Gaussian:
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                s_unit = (y - mu) * mu
                h_unit = mu * (2.0 * mu - y)
                return s_unit, h_unit, False, "observed"
        if dist_type is Binomial:
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                one_minus = 1.0 - mu
                s_unit = (y - mu) / one_minus
                h_unit = (1.0 - y) * mu / (one_minus * one_minus)
                return s_unit, h_unit, False, "observed"

    if dist_type is Binomial and link_type is ProbitLink:
        h_unit, s_unit = _probit_observed_rows(y=y, eta=eta, sample_weight=ones)
        return s_unit, h_unit, False, "observed"

    if dist_type is Binomial and link_type is CloglogLink:
        h_unit, s_unit = _cloglog_observed_rows(y=y, eta=eta, sample_weight=ones)
        return s_unit, h_unit, False, "observed"

    if dist_type is Poisson and link_type is SqrtLink:
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            s_unit = np.where((eta == 0.0) & (y == 0.0), 0.0, 2.0 * (y - mu) / eta)
            h_unit = np.where((mu == 0.0) & (y == 0.0), 2.0, 2.0 * (1.0 + y / mu))
            return s_unit, h_unit, False, "observed"

    # ── Remaining built-in pairs and custom family/link protocol ───
    try:
        curvature_mode = classify_reml_curvature(distribution, link)
    except (NotImplementedError, ValueError):
        return None

    if curvature_mode == "fisher":
        if not (hasattr(link, "deriv_inverse") and hasattr(distribution, "variance")):
            return None
        dmu = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
        var = np.maximum(np.asarray(distribution.variance(mu), dtype=np.float64), _VARIANCE_FLOOR)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            s_unit = (y - mu) * dmu / var
        h_unit = fisher_working_weights(
            distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=ones, dmu_deta=dmu
        )
        return s_unit, h_unit, True, "fisher"

    # Non-canonical generic observed curvature: require own-declared deriv2_inverse
    # and distribution.variance_derivative so a custom subclass never inherits stale
    # second derivatives silently.
    if _declared_with(link, "deriv2_inverse", ("inverse", "deriv_inverse")) is None:
        return None
    if not callable(getattr(distribution, "variance_derivative", None)):
        return None
    try:
        h_unit = compute_observed_information_weights(distribution, link, y, mu, eta, ones)
    except (NotImplementedError, ObservedGeometryInfeasibleError, ValueError):
        return None
    dmu = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    var = np.maximum(np.asarray(distribution.variance(mu), dtype=np.float64), _VARIANCE_FLOOR)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        s_unit = (y - mu) * dmu / var
    return s_unit, h_unit, False, "observed"


def _solve_matrix(bread_solver: Any, rhs: NDArray) -> NDArray:
    """Solve ``H_c^+ rhs`` for a 2-D ``(p, m)`` matrix in BLAS-3."""
    if rhs.shape[0] == 0 or rhs.shape[1] == 0:
        return np.zeros_like(rhs, dtype=np.float64)

    if isinstance(bread_solver, RankDecomposition):
        if bread_solver.rank == 0:
            return np.zeros_like(rhs, dtype=np.float64)
        if bread_solver.equilibrated_solution_basis is not None:
            return bread_solver._solve_deferred(rhs)
        if bread_solver.cholesky_factor is not None:
            active = bread_solver.active_columns
            scale = bread_solver.column_scale[active][:, None]
            active_rhs = rhs[active] / scale
            solved = scipy.linalg.cho_solve(
                (bread_solver.cholesky_factor, True), active_rhs, check_finite=False
            )
            out = np.zeros_like(rhs, dtype=np.float64)
            out[active] = solved / scale
            return out
        if bread_solver.solution_basis is None or bread_solver.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        proj = bread_solver.solution_basis.T @ rhs
        proj /= bread_solver.retained_values[:, None]
        return bread_solver.solution_basis @ proj

    # Structured profiled factor: solve() natively accepts 2-D (p, m).
    return np.asarray(bread_solver.solve(rhs), dtype=np.float64)


def _fill_centred_chunk(
    parts: tuple[Any, ...],
    start: int,
    stop: int,
    width: int,
    *,
    center: NDArray,
    center_lo: NDArray | None,
    row_scale: NDArray,
    out: NDArray | None = None,
) -> NDArray:
    """Fill ``row_scale[:, None] * ((X[start:stop] - center) - center_lo)`` without full X."""
    from superglm._group_matrix._group_matrix_kernels import (
        _weighted_centred_one_hot,
        _weighted_centred_rows,
    )

    m = stop - start
    buf = np.empty((m, width), dtype=np.float64) if out is None else out
    if width == 0 or m == 0:
        return buf
    c_hi = np.ascontiguousarray(center, dtype=np.float64)
    c_lo = _NO_CENTRE if center_lo is None else np.ascontiguousarray(center_lo, dtype=np.float64)
    scale = np.ascontiguousarray(row_scale, dtype=np.float64)
    for col, kind, first, second in parts:
        if kind == "one_hot":
            _weighted_centred_one_hot(buf, col, first, second, start, c_hi, c_lo, scale)
        elif kind == "gather":
            _weighted_centred_rows(buf, col, first, second, start, c_hi, c_lo, scale)
        else:
            rows = np.arange(start, stop, dtype=np.intp)
            values = np.array(first.row_subset(rows).toarray(), dtype=np.float64, order="C")
            _weighted_centred_rows(buf, col, values, rows - start, 0, c_hi, c_lo, scale)
    return buf


def _fill_centred_indices(
    parts: tuple[Any, ...],
    row_indices: NDArray[np.intp],
    width: int,
    *,
    center: NDArray,
    center_lo: NDArray | None,
    row_scale: NDArray,
) -> NDArray:
    """Fill centred design rows for arbitrary ``row_indices`` (used for cancelled clusters)."""
    m = len(row_indices)
    out = np.empty((m, width), dtype=np.float64)
    if width == 0 or m == 0:
        return out
    for col, kind, first, second in parts:
        if kind == "one_hot":
            k = int(second)
            sub = np.zeros((m, k), dtype=np.float64)
            codes = first[row_indices]
            valid = (codes >= 0) & (codes < k)
            if np.any(valid):
                sub[np.flatnonzero(valid), codes[valid]] = 1.0
            end = col + k
        elif kind == "gather":
            sub = np.array(first[second[row_indices]], dtype=np.float64, copy=True)
            end = col + sub.shape[1]
        else:
            sub = np.array(first.row_subset(row_indices).toarray(), dtype=np.float64, copy=True)
            end = col + sub.shape[1]
        sub -= center[col:end]
        if center_lo is not None:
            sub -= center_lo[col:end]
        sub *= row_scale[:, None]
        out[:, col:end] = sub
    return out


def _centred_rmatvec(
    parts: tuple[Any, ...],
    n_rows: int,
    width: int,
    vec: NDArray,
    center: NDArray,
    center_lo: NDArray | None,
    *,
    chunk_rows: int,
) -> NDArray:
    """Compute ``sum_i vec_i * ((x_i - center) - center_lo)`` with bounded chunks."""
    if width == 0:
        return np.empty(0, dtype=np.float64)
    total = np.zeros(width, dtype=np.float64)
    ones_scale = np.ones(min(chunk_rows, max(n_rows, 1)), dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        v_chunk = vec[start:stop]
        if not np.any(v_chunk):
            continue
        m = stop - start
        block = _fill_centred_chunk(
            parts,
            start,
            stop,
            width,
            center=center,
            center_lo=center_lo,
            row_scale=ones_scale[:m],
        )
        total += v_chunk @ block
    return total


def _penalized_active_groups(
    model: Any,
    active_groups: list[Any],
    lam2_solver: float | dict[str, float],
    exponent: int,
) -> tuple[frozenset[str], float]:
    """Return ``(penalized_group_names, ridge_lambda_solver)`` for active groups."""
    penalized_names: set[str] = set()
    active_by_name = {g.name: g for g in active_groups}
    for columns, lam, _omega in _active_penalty_components(
        list(model._dm.group_matrices),
        model._groups,
        active_groups,
        lam2_solver,
        reml_penalties=getattr(model, "_reml_penalties", None),
    ):
        if lam == 0.0:
            continue
        for ag in active_groups:
            if ag.start < columns.stop and ag.end > columns.start:
                penalized_names.add(ag.name)

    penalty = fitted_penalty(model)
    ridge_lam = 0.0
    if isinstance(penalty, Ridge):
        raw_lam = float(penalty.lambda1 or 0.0)
        if raw_lam > 0.0:
            ridge_lam = float(np.ldexp(raw_lam, exponent)) if exponent else raw_lam
    elif isinstance(penalty, GroupElasticNet) and float(penalty.alpha) == 0.0:
        raw_lam = float(penalty.lambda1 or 0.0)
        if raw_lam > 0.0:
            ridge_lam = float(np.ldexp(raw_lam, exponent)) if exponent else raw_lam

    if ridge_lam > 0.0:
        for orig in model._groups:
            if orig.name in active_by_name and penalty_targets_group(penalty, orig):
                penalized_names.add(orig.name)

    return frozenset(penalized_names), ridge_lam


def _build_structured_observed_bread(
    model: Any,
    linear_state: StructuredLinearSystemState,
    h_solver: NDArray,
    w_solver: NDArray,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    lam2_solver: float | dict[str, float],
) -> tuple[Any, float, NDArray, NDArray | None, int, bool] | None:
    """Build observed structured bread for a non-canonical structured fit."""
    from superglm.solvers._structured.assembly import (
        build_augmented_structured_factor,
        build_penalized_structured_operator,
    )
    from superglm.solvers._structured.balance_tree import (
        ProfiledSumToZeroTreeFactor,
        SumToZeroTreeFactor,
    )
    from superglm.solvers._structured.block_leaves import (
        FactorSmoothLeafFactor,
        ProfiledFactorSmoothLeafFactor,
    )
    from superglm.solvers._structured.layout import get_structured_layout
    from superglm.solvers._structured.moments import (
        NestedStructuredSystem,
        build_structured_system,
    )
    from superglm.solvers._structured.nested import (
        NestedSchurFactor,
        ProfiledNestedSchurFactor,
    )

    dm = model._dm
    groups = model._groups
    if isinstance(linear_state.system, NestedStructuredSystem):
        chain_indices: tuple[int, ...] = linear_state.system.chain_group_indices
        dominant_idx = chain_indices[-1]
    else:
        dominant_idx = linear_state.system.dominant_group_index
        chain_indices = ()
    layout = get_structured_layout(
        dm,
        groups,
        dominant_group_index=dominant_idx,
        chain_group_indices=chain_indices,
    )
    try:
        err_scale = observed_row_error_scale(
            model._distribution, model._link, y, mu, eta, w_solver, h_solver
        )
        system = build_structured_system(
            list(dm.group_matrices),
            groups,
            h_solver,
            np.zeros(dm.n, dtype=np.float64),
            dominant_group_index=dominant_idx,
            layout=layout,
            prior_weights=w_solver,
            signed=True,
            error=err_scale,
        )
    except (np.linalg.LinAlgError, ObservedGeometryInfeasibleError, ValueError):
        return None
    if not np.isfinite(system.sum_w) or system.sum_w <= 0.0:
        return None

    reml_penalties = getattr(model, "_reml_penalties", None)
    lambdas = lam2_solver
    try:
        penalized_op = build_penalized_structured_operator(
            system,
            list(dm.group_matrices),
            groups,
            lambdas,
            reml_penalties=reml_penalties,
        )
        augmented_factor, _ = build_augmented_structured_factor(system, penalized_op)
        schur_eigs = augmented_factor.scaled_schur_eigenvalues()
    except (np.linalg.LinAlgError, ValueError):
        return None

    if schur_curvature_is_negative(schur_eigs, augmented_factor.border_certificate):
        return None

    xtw = np.empty(dm.p, dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    mean_x = xtw / system.sum_w

    if isinstance(augmented_factor, FactorSmoothLeafFactor):
        profiled: Any = ProfiledFactorSmoothLeafFactor(
            augmented_factor=augmented_factor, sum_w=system.sum_w, xtw=xtw
        )
    elif isinstance(augmented_factor, SumToZeroTreeFactor):
        profiled = ProfiledSumToZeroTreeFactor(
            augmented_factor=augmented_factor, sum_w=system.sum_w, xtw=xtw
        )
    elif isinstance(system, NestedStructuredSystem) and isinstance(
        augmented_factor, NestedSchurFactor
    ):
        profiled = ProfiledNestedSchurFactor(
            augmented_factor=augmented_factor,
            sum_w=system.sum_w,
            xtw=xtw,
            data_operator=system.operator,
        )
    else:
        return None

    bread_rank = int(augmented_factor.rank)
    full_rank = not bool(profiled.rank_truncated)
    return profiled, float(system.sum_w), mean_x, None, bread_rank, full_rank


def compute_superglm_sandwich_covariance(
    model: Any,
    cluster: Any = None,
    *,
    correction: bool = False,
    center: Literal["auto", "mean", "none"] | bool = "auto",
    max_memory: int | str | None = None,
) -> SandwichCovariance:
    """Compute an opt-in, fit-bound sandwich covariance for a fitted ``SuperGLM``."""
    if model._result is None or model._solver_result is None:
        raise RuntimeError("Model must be fitted before calling sandwich_covariance().")

    # Validate syntax of caller-supplied control arguments upfront.
    if not isinstance(correction, bool):
        raise ValueError(f"correction must be a bool, got {correction!r}")
    resolve_center_mode(center, penalized=False)
    budget = resolve_workspace_budget(max_memory, getattr(model, "_max_memory", "auto"))

    binding = fit_binding_for_superglm(model)
    solver = model._solver_pirls_result()
    weight_semantics = model_weight_semantics(model)
    n_obs = int(model._fit_stats.n_obs) if getattr(model, "_fit_stats", None) is not None else 0
    if binding is None:
        return build_unavailable_covariance(
            unavailable_reason="missing_fit_state",
            message=_UNBOUND_MESSAGE,
            n_obs=n_obs,
            weight_semantics=weight_semantics,
        )

    # ── Explicit unavailability checks (deterministic order) ───────
    if getattr(model, "_editor_inference_stale", False):
        return build_unavailable_covariance(
            unavailable_reason="editor_stale",
            message=(
                "Manual editor coefficient edits invalidate estimating-equation sandwich "
                "covariance; refit the model before requesting sandwich_covariance()."
            ),
            binding=binding,
            n_obs=n_obs,
            weight_semantics=weight_semantics,
        )

    if shape_repaired_feature_names(model):
        return build_unavailable_covariance(
            unavailable_reason="shape_repaired",
            message=(
                "Post-fit shape repair projects coefficients onto a constraint cone; "
                "estimating-equation sandwich covariance is unavailable."
            ),
            binding=binding,
            n_obs=n_obs,
            weight_semantics=weight_semantics,
        )

    if (
        model._dm is None
        or model._fit_weights is None
        or getattr(model, "_fit_data_guard", None) is None
    ):
        return build_unavailable_covariance(
            unavailable_reason="missing_fit_state",
            message=(
                "Retained fitting design, weights, and response snapshot are required to "
                "replay row scores; refit with retain_fit_state=True."
            ),
            binding=binding,
            n_obs=n_obs,
            weight_semantics=weight_semantics,
        )

    reml_result = getattr(model, "_reml_result", None)
    if (
        not bool(getattr(solver, "converged", True))
        or (reml_result is not None and not bool(getattr(reml_result, "converged", True)))
        or int(getattr(solver, "mean_space_boundary_rows", 0)) > 0
    ):
        return build_unavailable_covariance(
            unavailable_reason="not_converged",
            message=(
                "The fitted model did not converge to an interior estimating-equation mode; "
                "sandwich covariance is unavailable."
            ),
            binding=binding,
            n_obs=model._dm.n,
            weight_semantics=weight_semantics,
        )

    penalty = fitted_penalty(model)
    if penalty_can_zero_groups(penalty) and selection_shrunk_group_names(penalty, model._groups):
        return build_unavailable_covariance(
            unavailable_reason="selection_penalty",
            message=(
                "Non-smooth selection penalties (L1 / group-lasso / elastic net with alpha > 0) "
                "violate regular estimating-equation assumptions; use refit_unpenalised() "
                "for post-selection inference in retained coordinates."
            ),
            binding=binding,
            n_obs=model._dm.n,
            weight_semantics=weight_semantics,
        )

    # A fit whose shape rows bind nowhere is the unconstrained optimum (no multiplier is
    # active), and ``RankInfo.face`` is ``None`` exactly there.  A row that binds, a fit
    # that kept no such record, and the SCOP engine's reparameterized fit are refused.
    shape_rows = any(g.constraints is not None or g.monotone_engine == "qp" for g in model._groups)
    if (
        any(g.monotone_engine not in (None, "qp") for g in model._groups)
        or getattr(solver, "scop_inference", None) is not None
        or (solver.rank_info is not None and solver.rank_info.face is not None)
        or (shape_rows and solver.rank_info is None)
    ):
        return build_unavailable_covariance(
            unavailable_reason="active_constraints",
            message=(
                "A shape constraint binds at the fitted coefficients, so they sit on the "
                "boundary of the constrained space, where the unconstrained estimating "
                "equation's sandwich covariance does not apply."
            ),
            binding=binding,
            n_obs=model._dm.n,
            weight_semantics=weight_semantics,
        )

    # ── Retained fit arrays and cluster normalization ──────────────
    # The scores are replayed on the fit's own rows.  A model rescored on other rows
    # (``EditorSession.to_model(X=..., y=...)``) keeps the fitted design beside their
    # response, weights and offset, and has no score of its own to replay.
    y = np.asarray(model._fit_data_guard.y_snapshot, dtype=np.float64)
    w_phys = np.asarray(model._fit_weights, dtype=np.float64)
    n_rows = int(model._dm.n)
    if not (
        len(y) == len(w_phys) == n_rows
        and model._fit_geometry_guard.records_vectors(y, w_phys, model._fit_offset)
    ):
        return build_unavailable_covariance(
            unavailable_reason="missing_fit_state",
            message=(
                "The response, weights or offset this model holds are not the rows it was "
                "fitted on, so the fit's scores cannot be replayed from it. Request "
                "sandwich_covariance() from the fitted model."
            ),
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )

    cluster_codes: NDArray[np.intp] | None = None
    n_clusters: int | None = None
    clustered = cluster is not None
    if clustered:
        cluster_codes, n_clusters = normalize_clusters(cluster, w_phys)

    pos = w_phys > 0.0
    n_pos = int(np.count_nonzero(pos))
    if n_pos == 0:
        return build_unavailable_covariance(
            unavailable_reason="indefinite_bread",
            message="No positive-weight rows are present in the retained fit.",
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )

    # ── Fitted predictor, mean, observed row score & curvature ─────
    eta_raw = linear_predictor(model._dm, solver, model._fit_offset)
    if (
        getattr(solver, "mean_space_true_mode", False)
        and type(model._distribution) is Binomial
        and type(model._link) is LogLink
    ):
        eta = np.asarray(eta_raw, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore"):
            mu = np.exp(eta)
        clip_held = False
    else:
        # The solver holds a binomial predictor inside the link's band (logit's +-20) and
        # its mean off 0 and 1 at clip_mu's 1e-7; past either clip its working score is
        # constant in eta while its working weight is not, so the two are derivatives of
        # no one function.  Score and bread are read from the log-likelihood at the fitted
        # predictor, so both are derivatives of the same psi (A = -dpsi/dtheta and
        # B = psi psi': Stefanski & Boos, The Calculus of M-Estimation, Am. Stat. 56(1),
        # 2002): logit, probit and cloglog in closed forms in the unclipped eta, accurate
        # in both tails, and the other links at the unclipped mean of their band.  A fit
        # holding such a row forms its own bread below.
        eta = stabilize_eta(eta_raw, model._link)
        mu = np.asarray(model._link.inverse(eta), dtype=np.float64)
        clipped = clip_mu(mu, model._distribution)
        binomial = isinstance(model._distribution, Binomial)
        clip_held = binomial and bool(np.any(((clipped != mu) | (eta != eta_raw)) & pos))
        if not binomial:
            mu = clipped
        elif type(model._distribution) is Binomial and type(model._link) in (
            LogitLink,
            ProbitLink,
            CloglogLink,
        ):
            eta = np.asarray(eta_raw, dtype=np.float64)
            with np.errstate(over="ignore", under="ignore"):
                mu = np.asarray(model._link.inverse(eta), dtype=np.float64)

    row_bundle = _observed_score_and_curvature(model._distribution, model._link, y, mu, eta)
    if row_bundle is None:
        return build_unavailable_covariance(
            unavailable_reason="unsupported_curvature",
            message=(
                f"Observed estimating-equation curvature is not supported for "
                f"{type(model._distribution).__name__} with {type(model._link).__name__}."
            ),
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )
    s_unit, h_unit, is_canonical, curvature_label = row_bundle
    if not (np.all(np.isfinite(s_unit[pos])) and np.all(np.isfinite(h_unit[pos]))):
        return build_unavailable_covariance(
            unavailable_reason="non_finite_scores",
            message="Row scores or observed curvatures are non-finite at the fitted state.",
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )
    # A zero-weight row carries no score, whatever its mean evaluates to.
    s_unit = np.where(pos, s_unit, 0.0)
    h_unit = np.where(pos, h_unit, 0.0)

    exponent = int(getattr(solver, "weight_exponent", 0))
    w_solver = np.ldexp(w_phys, exponent) if exponent else w_phys
    h_solver = np.where(pos, w_solver * h_unit, 0.0)
    if not np.all(np.isfinite(h_solver)):
        return build_unavailable_covariance(
            unavailable_reason="non_finite_scores",
            message="Weighted observed curvatures are non-finite in solver weight units.",
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )
    sum_h = float(math.fsum(h_solver[pos]))
    if not math.isfinite(sum_h) or sum_h <= 0.0:
        return build_unavailable_covariance(
            unavailable_reason="indefinite_bread",
            message="Total observed intercept curvature is non-positive or non-finite.",
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )

    # ── Active coordinates & workspace budget guard ────────────────
    linear_state = retained_linear_state(model)
    if isinstance(linear_state, StructuredLinearSystemState):
        X_active: DesignMatrix = model._dm
        active_groups = list(model._groups)
        backend = "structured"
    elif solver.rank_info is not None:
        X_active, active_groups = _rank_active_state(model, solver.rank_info)
        backend = "dense"
    else:
        selected_names = selected_group_name_set(solver, model._groups, penalty=penalty)
        X_active, active_groups, _ = _grouped_active_state(model, selected_names)
        backend = "dense"

    p_active = int(X_active.p)
    q = 1 + p_active

    # One cap covers every dense array the request holds at once
    # (``_dense_workspace_bytes``), and the replay chunk takes what the rest leaves.
    workspace_cap = budget if max_memory is not None else min(budget // 2, 1 << 30)

    def over_budget(needed: int) -> SandwichCovariance:
        return build_unavailable_covariance(
            unavailable_reason="workspace_budget_exceeded",
            message=(
                f"The dense workspace of this request ({needed / (1 << 20):.1f} MiB for q={q}"
                + (f", G={n_clusters}" if n_clusters is not None else "")
                + f") exceeds its budget ({workspace_cap / (1 << 20):.1f} MiB); "
                "pass a larger max_memory."
            ),
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )

    if _dense_workspace_bytes(q, n_clusters, 1) > workspace_cap:
        return over_budget(_dense_workspace_bytes(q, n_clusters, 1))

    lam2_solver = scale_penalties(fitted_lambda2(model), exponent)
    penalized_group_names, ridge_lam = _penalized_active_groups(
        model, active_groups, lam2_solver, exponent
    )
    penalized = bool(penalized_group_names)

    # ── Observed bread reuse or certified assembly ─────────────────
    bread_solver: Any = None
    bread_reused = False
    n_factorizations = 0
    bread_columns = 0

    if backend == "structured":
        assert isinstance(linear_state, StructuredLinearSystemState)
        if is_canonical and ridge_lam == 0.0 and not clip_held:
            profiled_factor = linear_state.profiled_factor
            bread_solver = profiled_factor
            bread_sum_h = float(profiled_factor.sum_w)
            if not math.isfinite(bread_sum_h) or bread_sum_h <= 0.0:
                return build_unavailable_covariance(
                    unavailable_reason="indefinite_bread",
                    message="Retained structured bread has non-positive intercept curvature.",
                    binding=binding,
                    n_obs=n_rows,
                    weight_semantics=weight_semantics,
                )
            centre = np.asarray(profiled_factor.xtw, dtype=np.float64) / bread_sum_h
            centre_lo: NDArray | None = None
            bread_rank = int(profiled_factor.augmented_factor.rank)
            bread_full_rank = not bool(profiled_factor.rank_truncated)
            bread_reused = True
        else:
            built = _build_structured_observed_bread(
                model, linear_state, h_solver, w_solver, y, mu, eta, lam2_solver
            )
            if built is None:
                return build_unavailable_covariance(
                    unavailable_reason="indefinite_bread",
                    message=(
                        "Structured observed coefficient Hessian is indefinite or cannot be "
                        "factored at the fitted mode."
                    ),
                    binding=binding,
                    n_obs=n_rows,
                    weight_semantics=weight_semantics,
                )
            bread_solver, bread_sum_h, centre, centre_lo, bread_rank, bread_full_rank = built
            n_factorizations = 1
    elif p_active == 0:
        bread_solver = None
        bread_sum_h = sum_h
        centre = np.empty(0, dtype=np.float64)
        centre_lo = None
        bread_rank = 1
        bread_full_rank = True
        bread_reused = True
    else:
        can_reuse_rank = (
            is_canonical
            and ridge_lam == 0.0
            and not clip_held
            and solver.rank_info is not None
            and solver.rank_info.face is None
            and len(solver.rank_info.selected_columns) == p_active
            and float(solver.rank_info.sum_w) > 0.0
            and math.isfinite(float(solver.rank_info.sum_w))
            and math.isclose(float(solver.rank_info.sum_w), sum_h, rel_tol=1e-7, abs_tol=0.0)
        )
        if can_reuse_rank:
            rank_info = solver.rank_info
            assert rank_info is not None
            bread_solver = rank_info.augmented
            bread_sum_h = float(rank_info.sum_w)
            _, centre, centre_lo = weighted_mean_pair(X_active, h_solver, bread_sum_h)
            bread_rank = 1 + int(bread_solver.rank)
            bread_full_rank = int(bread_solver.rank) == p_active
            bread_reused = True
        else:
            bread_columns = p_active
            needed = _dense_workspace_bytes(q, n_clusters, 1, bread_columns=bread_columns)
            if needed > workspace_cap:
                return over_budget(needed)
            S_active = np.array(
                _active_penalty_matrix(
                    list(model._dm.group_matrices),
                    model._groups,
                    active_groups,
                    lam2_solver,
                    reml_penalties=getattr(model, "_reml_penalties", None),
                ),
                dtype=np.float64,
                copy=True,
            )
            if ridge_lam > 0.0:
                orig_by_name = {g.name: g for g in model._groups}
                for ag in active_groups:
                    if penalty_targets_group(penalty, orig_by_name[ag.name]):
                        block = S_active[ag.sl, ag.sl]
                        block[np.diag_indices_from(block)] += ridge_lam

            nonnegative = bool(np.all(h_solver >= 0.0))
            if nonnegative:
                try:
                    centered_sys = build_centered_system(
                        dm=X_active,
                        W=h_solver,
                        z_off=np.zeros(n_rows, dtype=np.float64),
                        penalty=S_active,
                    )
                except ValueError:
                    return build_unavailable_covariance(
                        unavailable_reason="indefinite_bread",
                        message="Observed curvature weights do not define a valid centered system.",
                        binding=binding,
                        n_obs=n_rows,
                        weight_semantics=weight_semantics,
                    )
                bread_sum_h = float(centered_sys.sum_w)
                centre, centre_lo = centered_sys.centre_pair()
                hessian = centered_sys.hessian
            else:
                from superglm._group_matrix._group_matrix_centered import centered_gram_rhs

                try:
                    _, centre, centre_lo = _stable_signed_mean_pair(X_active, h_solver, sum_h)
                    data_gram, _ = centered_gram_rhs(
                        dm=X_active,
                        W=h_solver,
                        mean_x=centre,
                        z_centered=np.zeros(n_rows, dtype=np.float64),
                        mean_lo=centre_lo,
                    )
                except ValueError:
                    return build_unavailable_covariance(
                        unavailable_reason="indefinite_bread",
                        message="Signed observed Hessian could not be assembled reliably.",
                        binding=binding,
                        n_obs=n_rows,
                        weight_semantics=weight_semantics,
                    )
                bread_sum_h = sum_h
                hessian = 0.5 * (data_gram + data_gram.T) + S_active

            if nonnegative:
                try:
                    decomp = decompose_gram_if_authoritative(hessian)
                    if decomp is None:
                        decomp = decompose_factor(
                            grouped_augmented_factor(
                                X_active,
                                h_solver,
                                S_active,
                                center=centre,
                                center_lo=centre_lo,
                            )
                        )
                except ValueError:
                    return build_unavailable_covariance(
                        unavailable_reason="indefinite_bread",
                        message=(
                            "Observed coefficient Hessian is indefinite or could not be certified."
                        ),
                        binding=binding,
                        n_obs=n_rows,
                        weight_semantics=weight_semantics,
                    )
            else:
                try:
                    decomp = decompose_gram(hessian)
                except ValueError:
                    return build_unavailable_covariance(
                        unavailable_reason="indefinite_bread",
                        message="Observed coefficient Hessian is indefinite at the fitted mode.",
                        binding=binding,
                        n_obs=n_rows,
                        weight_semantics=weight_semantics,
                    )
            bread_solver = decomp
            bread_rank = 1 + int(decomp.rank)
            bread_full_rank = int(decomp.rank) == p_active
            n_factorizations = 1

    # ── Effective sample count, centering, small-sample factor ─────
    is_freq = weight_semantics == FREQUENCY_WEIGHTS
    n_units = float(math.fsum(w_phys[pos])) if is_freq else float(n_pos)
    centered = resolve_center_mode(center, penalized=penalized)
    cov_type, correction_factor = small_sample_multiplier(
        clustered=clustered,
        correction=correction,
        n_units=n_units,
        n_params=q,
        n_clusters=n_clusters,
        penalized=penalized,
        full_rank=bread_full_rank,
    )

    # ── Estimability mask in full coefficient coordinates ──────────
    p_full = len(solver.beta)
    coef_estimable = np.zeros(p_full, dtype=bool)
    orig_by_name = {g.name: g for g in model._groups}
    if p_active > 0:
        if isinstance(bread_solver, RankDecomposition):
            active_est = bread_solver.coefficient_estimable()
            for ag in active_groups:
                coef_estimable[orig_by_name[ag.name].sl] = active_est[ag.sl]
        else:
            # Structured bread: full rank when not truncated, otherwise read fit's mask.
            if bread_full_rank and (penalized or not _names_an_unpriced_level(model)):
                for ag in active_groups:
                    coef_estimable[orig_by_name[ag.name].sl] = True
            else:
                fit_mask = model._fit_inference_info.get("coefficient_estimable")
                if fit_mask is not None:
                    coef_estimable = np.asarray(fit_mask, dtype=bool).copy()
                else:
                    for ag in active_groups:
                        coef_estimable[orig_by_name[ag.name].sl] = True

    # ── Solver-to-public intercept shift & chunked score replay ────
    m_shift = _public_intercept_shift(model, active_groups, p_active)
    if p_active > 0:
        delta_shift = (m_shift - centre) if centre_lo is None else ((m_shift - centre) - centre_lo)
    else:
        delta_shift = np.empty(0, dtype=np.float64)
    # The public intercept is the centred one plus ``delta_shift' beta``: determined
    # where that contrast is orthogonal to the aliased directions (``is_estimable``).
    # A truncated structured bread carries no such test, so its intercept is not certified.
    intercept_estimable = (
        bool(bread_solver.is_estimable(delta_shift))
        if isinstance(bread_solver, RankDecomposition)
        else bread_full_rank
    )

    leaf_parts = _design_leaf_parts(X_active) if p_active > 0 else None
    parts_tuple = () if leaf_parts is None else leaf_parts.parts
    held_design_bytes = 0 if leaf_parts is None else int(leaf_parts.held)
    fixed_bytes = held_design_bytes + _dense_workspace_bytes(
        q, n_clusters, 0, bread_columns=bread_columns
    )
    chunk_rows = min(
        _MAX_CHUNK_ROWS,
        max(1, _TARGET_CHUNK_BYTES // (16 * q)),
        (workspace_cap - fixed_bytes) // (48 * q),
    )
    if chunk_rows < 1:
        return over_budget(fixed_bytes + 48 * q)
    peak_bytes = fixed_bytes + 48 * chunk_rows * q

    n_chunks = 0

    if not clustered:
        # Unclustered HC0 / HC1
        final_power_shift = exponent if is_freq else 0
        gram_acc = ScaledGramAccumulator(q, final_power_shift=final_power_shift)

        if is_freq:
            # Row i has unit score s_unit[i] * x~_i and replication mass w_i.
            # Work in solver weight units so sqrt(w_solver) never underflows;
            # ScaledGramAccumulator restores 2**exponent on finalize().
            sqrt_w = np.where(pos, np.sqrt(w_solver), 0.0)
            row_score_base = np.where(pos, s_unit, 0.0)
            if centered:
                ws_vec = np.where(pos, w_solver * s_unit, 0.0)
                sum_w_solver = float(math.fsum(w_solver[pos]))
                bar_g0 = float(math.fsum(ws_vec[pos])) / sum_w_solver
                bar_g_slopes = (
                    _centred_rmatvec(
                        parts_tuple,
                        n_rows,
                        p_active,
                        ws_vec,
                        centre,
                        centre_lo,
                        chunk_rows=chunk_rows,
                    )
                    / sum_w_solver
                )
            else:
                bar_g0 = 0.0
                bar_g_slopes = np.zeros(p_active, dtype=np.float64)
        else:
            # Prior weights: row i is one unit with weighted score w_solver[i] * s_unit[i] * x~_i.
            row_score_base = np.where(pos, w_solver * s_unit, 0.0)
            sqrt_w = None
            if centered:
                bar_g0 = float(math.fsum(row_score_base[pos])) / float(n_pos)
                bar_g_slopes = _centred_rmatvec(
                    parts_tuple,
                    n_rows,
                    p_active,
                    row_score_base,
                    centre,
                    centre_lo,
                    chunk_rows=chunk_rows,
                ) / float(n_pos)
            else:
                bar_g0 = 0.0
                bar_g_slopes = np.zeros(p_active, dtype=np.float64)

        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            n_chunks += 1
            pos_chunk = pos[start:stop]
            if not np.any(pos_chunk):
                continue
            s_chunk = row_score_base[start:stop]

            if p_active > 0:
                g_slopes_full = _fill_centred_chunk(
                    parts_tuple,
                    start,
                    stop,
                    p_active,
                    center=centre,
                    center_lo=centre_lo,
                    row_scale=s_chunk,
                )
                g_slopes = g_slopes_full[pos_chunk]
                g0 = s_chunk[pos_chunk].copy() if (centered or is_freq) else s_chunk[pos_chunk]
                if centered:
                    g_slopes -= bar_g_slopes[None, :]
                    g0 -= bar_g0
                if is_freq:
                    assert sqrt_w is not None
                    sw = sqrt_w[start:stop][pos_chunk]
                    g_slopes *= sw[:, None]
                    g0 *= sw
                z_slopes = _solve_matrix(bread_solver, g_slopes.T).T
                z0_pub = g0 / bread_sum_h + z_slopes @ delta_shift
                z_pub = np.empty((len(g0), q), dtype=np.float64)
                z_pub[:, 0] = z0_pub
                z_pub[:, 1:] = z_slopes
            else:
                g0 = s_chunk[pos_chunk].copy()
                if centered:
                    g0 -= bar_g0
                if is_freq:
                    assert sqrt_w is not None
                    g0 *= sqrt_w[start:stop][pos_chunk]
                z_pub = (g0 / bread_sum_h)[:, None]

            gram_acc.add_rows(z_pub)

        cov_pub = gram_acc.finalize(multiplier=correction_factor)
    else:
        # One-way clustered CR0 / CR1
        assert cluster_codes is not None and n_clusters is not None
        s_row = np.where(pos, w_solver * s_unit, 0.0)
        acc = ClusterScoreAccumulator(n_clusters, q, cluster_codes)

        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            n_chunks += 1
            pos_chunk = pos[start:stop]
            if not np.any(pos_chunk):
                continue
            m_chunk = stop - start
            s_chunk = s_row[start:stop]
            chunk_scores = np.empty((m_chunk, q), dtype=np.float64)
            chunk_scores[:, 0] = s_chunk
            if p_active > 0:
                _fill_centred_chunk(
                    parts_tuple,
                    start,
                    stop,
                    p_active,
                    center=centre,
                    center_lo=centre_lo,
                    row_scale=s_chunk,
                    out=chunk_scores[:, 1:],
                )
            acc.add_chunk(start, stop, chunk_scores)

        cancelled = acc.cancelled_clusters(max_chunk_rows=chunk_rows)
        if len(cancelled) > 0:

            def _exact_rows(indices: NDArray[np.intp]) -> NDArray:
                s_idx = s_row[indices]
                out = np.empty((len(indices), q), dtype=np.float64)
                out[:, 0] = s_idx
                if p_active > 0:
                    out[:, 1:] = _fill_centred_indices(
                        parts_tuple,
                        indices,
                        p_active,
                        center=centre,
                        center_lo=centre_lo,
                        row_scale=s_idx,
                    )
                return out

            acc.refine_cancelled(cancelled, _exact_rows, max_chunk_rows=chunk_rows)

        G_mat = acc.totals
        if not np.all(np.isfinite(G_mat)):
            return build_unavailable_covariance(
                unavailable_reason="non_finite_scores",
                message="Cluster score sums are non-finite at the fitted state.",
                binding=binding,
                n_obs=n_rows,
                weight_semantics=weight_semantics,
            )

        if centered:
            mean_cluster = np.array(
                [math.fsum(G_mat[:, col]) / float(n_clusters) for col in range(q)],
                dtype=np.float64,
            )
            G_mat -= mean_cluster[None, :]

        gram_acc = ScaledGramAccumulator(q)
        for c_start in range(0, n_clusters, chunk_rows):
            c_stop = min(c_start + chunk_rows, n_clusters)
            g_batch = G_mat[c_start:c_stop]
            if p_active > 0:
                z_slopes = _solve_matrix(bread_solver, g_batch[:, 1:].T).T
                z0_pub = g_batch[:, 0] / bread_sum_h + z_slopes @ delta_shift
                z_pub = np.empty_like(g_batch)
                z_pub[:, 0] = z0_pub
                z_pub[:, 1:] = z_slopes
            else:
                z_pub = g_batch / bread_sum_h
            gram_acc.add_rows(z_pub)

        cov_pub = gram_acc.finalize(multiplier=correction_factor)

    if cov_pub is None or not np.all(np.isfinite(cov_pub)):
        return build_unavailable_covariance(
            unavailable_reason="non_finite_scores",
            message="Computed sandwich covariance contains non-finite entries.",
            binding=binding,
            n_obs=n_rows,
            weight_semantics=weight_semantics,
        )

    # A standard error is blank for a coefficient the fit does not determine: an aliased
    # one, a separated level's (``separated_coefficients``, the intercept last), and the
    # intercept where either moves it.  ``covariance`` keeps the entries of the fit's own
    # identification (aliased coefficients held where the fit left them).
    separated = separated_coefficients(model._groups, model._specs, model._interaction_specs)
    se = np.sqrt(np.maximum(np.diag(cov_pub), 0.0))
    if not intercept_estimable or (separated is not None and separated[-1]):
        se[0] = np.nan
    for ag in active_groups:
        columns = orig_by_name[ag.name].sl
        blank = ~coef_estimable[columns]
        if separated is not None:
            blank |= separated[columns]
        se[1 + ag.start : 1 + ag.end][blank] = np.nan

    retained_bytes = int(cov_pub.nbytes + se.nbytes + coef_estimable.nbytes)
    telemetry = SandwichTelemetry(
        scored_rows=n_rows,
        positive_weight_rows=n_pos,
        n_chunks=n_chunks,
        n_factorizations=n_factorizations,
        backend=backend,
        bread_reused=bread_reused,
        peak_bytes=peak_bytes,
        retained_bytes=retained_bytes,
    )

    return build_available_covariance(
        binding=binding,
        cov_type=cov_type,
        clustered=clustered,
        correction=correction,
        correction_factor=correction_factor,
        centered=centered,
        penalized=penalized,
        weight_semantics=weight_semantics,
        bread_curvature=curvature_label,
        n_obs=n_rows,
        n_units=n_units,
        n_clusters=n_clusters,
        n_params=q,
        bread_rank=bread_rank,
        covariance=cov_pub,
        se=se,
        coefficient_estimable=coef_estimable,
        intercept_estimable=intercept_estimable,
        active_groups=tuple(active_groups),
        penalized_group_names=penalized_group_names,
        telemetry=telemetry,
    )


__all__ = [
    "SandwichCovariance",
    "compute_superglm_sandwich_covariance",
    "fit_binding_for_superglm",
]
