"""Post-fit shape repair operations for SuperGLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm._utils import weighted_row_sum
from superglm.features.constraint import rules_of
from superglm.group_matrix import DesignMatrix, GroupMatrix
from superglm.model.fit_state import (
    FittedStateRevision,
    canonical_intercept_shift,
    move_public_intercept,
    publish_revised_coefficients,
)
from superglm.solvers.dispersion import (
    model_weight_semantics,
    pearson_residual_degrees_of_freedom,
)
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import PenaltyComponent


def _grid_weights(spec, x_col, sample_weight, n_grid: int):
    hist_counts, bin_edges = np.histogram(x_col, bins=n_grid, weights=sample_weight)
    x_grid = np.linspace(spec._lo, spec._hi, n_grid)
    grid_weights = np.interp(
        x_grid,
        0.5 * (bin_edges[:-1] + bin_edges[1:]),
        hist_counts.astype(np.float64) + 1.0,
    )
    return np.maximum(grid_weights, 1e-6)


# The metric a post-fit repair projects in. A repair keeps the fit's smoothing
# parameters, so the standard estimate is the constrained penalized fit at them,
# and its first Newton step from the unconstrained fit is the projection of the
# fitted coefficients onto the shape cone in X'WX + S_lambda: the penalized
# Hessian (Wood 2026, Statistical Modelling 26(4), section 4.1; Mammen, Marron,
# Turlach & Wand 2001, Statistical Science 16(3), Proposition 2 and section 5.1,
# where the penalty in the norm is what keeps the projection smooth). It is the
# second-order model of the penalized objective the publication rule judges, with
# the other terms held and the intercept profiled, as the rule holds and profiles
# them, so the zero term, a feasible point, can never beat the projection there.
# The grid's L2 metric omits S_lambda and sees the wiggly fitted-basis directions
# of a sparse region only weakly, and its projection travels far along them.
def _term_indices(groups) -> NDArray:
    return np.concatenate([np.arange(g.start, g.end) for g in groups])


def _term_penalty_block(terms, indices: NDArray, width: int) -> NDArray:
    """``S_lambda`` on the term's coefficients, as ``_smooth_penalty_value`` reads it."""
    if isinstance(terms, _CompactPenaltyTerms):
        from superglm.reml.penalty_algebra import total_penalty_matvec

        # A component off the term's coefficients adds exact zeros to the entries read.
        own = [
            component
            for component in terms.penalties
            if np.any((indices >= component.group_sl.start) & (indices < component.group_sl.stop))
        ]
        block = np.zeros((indices.size, indices.size))
        for column, index in enumerate(indices):
            unit = np.zeros(width)
            unit[index] = 1.0
            product = total_penalty_matvec(unit, terms.lambdas, own, list(terms.group_matrices))
            block[:, column] = product[indices]
        return 0.5 * (block + block.T)
    from superglm.reml.penalty_algebra import ssp_penalty_matrix

    block = np.zeros((indices.size, indices.size))
    for group_slice, lam, omega, raw_projection in terms:
        where = np.flatnonzero((indices >= group_slice.start) & (indices < group_slice.stop))
        if where.size == 0:
            continue
        omega_arr = np.asarray(omega, dtype=np.float64)
        if raw_projection is not None:
            # Formed as the fit forms it (``build_penalty_matrix``): the Gram of a
            # transported root keeps the penalty's null space, where the congruence
            # ``R' Omega R`` leaves rounding that a large lambda makes curvature.
            omega_arr = ssp_penalty_matrix(raw_projection, omega_arr)
        local = indices[where] - group_slice.start
        block[np.ix_(where, where)] += lam * omega_arr[np.ix_(local, local)]
    return block


def _kept_data_gram(model, indices: NDArray, W: NDArray) -> NDArray | None:
    """The term's block of the centred data Gram the fit kept at ``W``, or ``None``.

    A fit records the Gram it decided its rank from, ``X'WX`` centred at the weighted
    column means, with a fingerprint of its weights (``RankInfo.data_gram``). It is read
    in place, so the fit's first inference still takes it.
    """
    from superglm.solvers.rank import weights_fingerprint

    record = getattr(getattr(model, "_solver_result", None), "rank_info", None)
    if record is None or record.data_gram is None:
        return None
    if record.data_gram_weights != weights_fingerprint(W):
        return None
    gram = record.data_gram
    selected = np.asarray(record.selected_columns)
    position = np.minimum(np.searchsorted(selected, indices), max(selected.size - 1, 0))
    if selected.size == 0 or not np.array_equal(selected[position], indices):
        return None
    block = np.asarray(gram, dtype=np.float64)[np.ix_(position, position)]
    return 0.5 * (block + block.T)


def _term_group_matrices(model, groups) -> list:
    by_key = {
        (g.name, g.start, g.end): gm
        for gm, g in zip(model._dm.group_matrices, model._groups, strict=True)
    }
    return [by_key[(g.name, g.start, g.end)] for g in groups]


def _term_rmatvec(model, groups, rows: NDArray) -> NDArray:
    """``X_j' rows`` over the term's fitted columns."""
    return np.concatenate(
        [
            np.asarray(gm.rmatvec(rows), dtype=np.float64)
            for gm in _term_group_matrices(model, groups)
        ]
    )


def _term_data_hessian(model, groups, W: NDArray) -> NDArray:
    """``X_j' W X_j`` over the fitted rows at the fit's working weights ``W``, intercept profiled.

    Without the fit's kept Gram, it is formed as inference forms it: from the columns
    centred at their weighted means (``build_centered_system``). The Schur complement
    ``X'WX - (X'W1)(1'WX) / 1'W`` cancels catastrophically where a term's curvature is
    small beside its raw moments, as the textbook variance formula does (Chan, Golub &
    LeVeque 1983).
    """
    kept = _kept_data_gram(model, _term_indices(groups), W)
    if kept is not None:
        return kept
    from superglm.model.state_ops import _rank_centered_data_gram

    design = DesignMatrix(
        _term_group_matrices(model, groups), n=model._dm.n, p=sum(g.size for g in groups)
    )
    gram = _rank_centered_data_gram(design, np.asarray(W, dtype=np.float64))
    return 0.5 * (gram + gram.T)


def _term_penalized_hessian(model, groups, smooth_penalty_terms, working_weights) -> NDArray:
    """``X'WX + S_lambda`` on one term's coefficients: the metric its repair projects in.

    ``working_weights`` are the unconstrained fit's, which every term of one call
    projects from.
    """
    indices = _term_indices(groups)
    penalty = _term_penalty_block(smooth_penalty_terms, indices, model.result.beta.size)
    return _term_data_hessian(model, groups, working_weights) + penalty


def _repairer(kind: str):
    from superglm.constraints import CurvatureRepairer, MonotoneRepairer

    if kind in {"increasing", "decreasing"}:
        return MonotoneRepairer(direction=kind)
    if kind in {"convex", "concave"}:
        return CurvatureRepairer(kind=kind)
    raise ValueError(f"Unsupported postfit shape kind: {kind!r}")


def _invalidate_repair_caches(model) -> None:
    for attr in ("_coef_covariance", "_fit_active_info", "_fit_inference_info", "_group_edf"):
        try:
            delattr(model, attr)
        except AttributeError:
            pass
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_stats = None


def _replace_result_beta(model, beta) -> None:
    """Update public, solver, and aliased REML results in private revision state."""
    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.beta = np.asarray(beta, dtype=np.float64).copy()
        updated.add(id(result))


def _term_publishes_a_centred_contribution(model, groups) -> bool:
    """Whether the published basis for this term asserts a zero-mean contribution.

    Runtime canonicalization records ``column_means`` for every term it registered and the
    public intercept carries ``means @ beta``, so a registered term's contribution has mean zero
    and a repair cannot move it; an unregistered one asserts nothing. Membership is
    ``hasattr(spec, "_basis_matrix")``, which an ``OrderedCategorical`` fails (it wraps the
    spline), so its ``select=True`` columns that do not sum to zero cannot refuse a repair
    (#311). The other guards (certificate feasibility, objective against the zero-term
    fallback) run on every term regardless.
    """
    runtime_state = getattr(model, "_runtime_canonical_state", None)
    if not isinstance(runtime_state, dict):
        return False
    registered = {
        tuple(group_state["solver_slice"])
        for term_state in runtime_state.get("terms", {}).values()
        if term_state.get("applied_to_public_model", False)
        for group_state in term_state.get("groups", [])
    }
    return any((group.start, group.end) in registered for group in groups)


@dataclass(frozen=True)
class _CompactPenaltyTerms:
    """Owned compact penalty geometry for one fitted REML result."""

    lambdas: float | dict[str, float]
    penalties: tuple[PenaltyComponent, ...]
    group_matrices: tuple[GroupMatrix, ...]


_LegacyPenaltyTerms = tuple[tuple[slice, float, object, object], ...]


def _build_smooth_penalty_terms(model) -> _CompactPenaltyTerms | _LegacyPenaltyTerms:
    """Prepare block/component quadratic terms without allocating a global ``p x p`` S."""
    from superglm.group_matrix import (
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )
    from superglm.model.fit_state import fitted_lambda2

    lambda2 = fitted_lambda2(model)
    group_matrices = model._dm.group_matrices
    groups = model._groups
    reml_penalties = getattr(model, "_reml_penalties", None)
    if reml_penalties is not None:
        return _CompactPenaltyTerms(
            lambdas=lambda2,
            penalties=tuple(reml_penalties),
            group_matrices=tuple(group_matrices),
        )

    terms: list[tuple[slice, float, object, object]] = []
    penalty_matrix_types = (
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )
    for gm, group in zip(group_matrices, groups, strict=True):
        if not group.penalized:
            continue
        if isinstance(gm, penalty_matrix_types):
            omega_components = getattr(gm, "omega_components", None)
            if omega_components is not None:
                from superglm.reml.penalty_algebra import resolve_component_lambda

                for suffix, omega_j in omega_components:
                    lam_j = float(resolve_component_lambda(lambda2, group.name, suffix))
                    if lam_j == 0.0:
                        continue
                    terms.append((group.sl, lam_j, omega_j, gm.R_inv))
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            lam_float = float(lam)
            if lam_float == 0.0:
                continue
            omega = gm.omega
            if omega is not None:
                terms.append((group.sl, lam_float, omega, gm.R_inv))
            continue
        lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
        lam_float = float(lam)
        if lam_float == 0.0:
            continue
        if group.scop_reparameterization is not None:
            terms.append(
                (
                    group.sl,
                    lam_float,
                    group.scop_reparameterization.penalty_matrix(),
                    None,
                )
            )
    return tuple(terms)


def _smooth_penalty_value(
    beta,
    terms: _CompactPenaltyTerms | _LegacyPenaltyTerms,
) -> float:
    """Evaluate exact ``beta' S beta`` from compact block/component terms."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if isinstance(terms, _CompactPenaltyTerms):
        from superglm.reml.penalty_algebra import total_penalty_quadratic

        return float(
            total_penalty_quadratic(
                beta_arr,
                terms.lambdas,
                list(terms.penalties),
                list(terms.group_matrices),
            )
        )
    value = 0.0
    for group_slice, lam, omega, raw_projection in terms:
        group_beta = beta_arr[group_slice]
        penalty_beta = (
            group_beta
            if raw_projection is None
            else np.asarray(raw_projection @ group_beta, dtype=np.float64)
        )
        omega_arr = np.asarray(omega, dtype=np.float64)
        value += lam * float(penalty_beta @ (omega_arr @ penalty_beta))
    return float(value)


def _residual_rounding(y, mu, eta, dmu_deta) -> np.ndarray:
    """Higham's ``gamma_n`` times the terms a residual ``y - mu`` is formed from.

    The response, the mean and the rounded predictor (through ``dmu/deta``) each
    carry rounding that the residual keeps however well they cancel.
    """
    from superglm._roundoff import gamma
    from superglm.constraints import _ROW_STEPS

    return float(gamma(_ROW_STEPS)) * (np.abs(y) + np.abs(mu) + np.abs(dmu_deta * eta))


def _deviance_rounding(model, eta, *, y, weights) -> float:
    """The rounding a weighted deviance keeps from its residuals.

    Each unit deviance is evaluated without cancellation given its mean, and its
    derivative in the mean is ``-2 (y - mu) / V(mu)``.  It keeps each residual's
    rounding times that, which near a perfect fit is proportional to the
    residual, not to its square as the deviance is.
    """
    from superglm.distributions import _VARIANCE_FLOOR, clip_mu
    from superglm.links import stabilize_eta

    carried = np.asarray(weights) != 0.0
    response, mass = np.asarray(y)[carried], np.asarray(weights)[carried]
    eta_safe = stabilize_eta(np.asarray(eta, dtype=np.float64)[carried], model._link)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
        variance = np.maximum(
            np.asarray(model._distribution.variance(mu), dtype=np.float64),
            _VARIANCE_FLOOR,
        )
        dmu_deta = np.asarray(model._link.deriv_inverse(eta_safe), dtype=np.float64)
        rows = np.abs(mass * (response - mu) / variance) * _residual_rounding(
            response, mu, eta_safe, dmu_deta
        )
    return 2.0 * float(np.sum(rows, dtype=np.float64))


class _InterceptState(NamedTuple):
    """The profiled deviance and score at one intercept shift."""

    deviance: float
    deviance_rounding: float
    score: float
    score_scale: float
    score_rounding: float
    information: float


def _profile_repaired_intercept(
    model,
    eta,
    *,
    y,
    weights,
) -> tuple[np.ndarray, float, float]:
    """Minimize deviance over the unpenalized intercept for fixed coefficients.

    Returns the profiled predictor, its shift and the deviance's rounding there
    (``_deviance_rounding``), which each evaluation already holds.
    """
    from superglm.distributions import _VARIANCE_FLOOR, Gaussian, clip_mu
    from superglm.links import IdentityLink, stabilize_eta

    eta_base = np.asarray(eta, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if eta_base.shape != y_arr.shape or weights_arr.shape != y_arr.shape:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: incoherent intercept geometry"
        )
    sum_weights = float(np.sum(weights_arr, dtype=np.float64))
    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: invalid scoring weights"
        )

    # Gaussian/identity has an exact conditional solution.  Keeping this
    # algebraic also preserves frequency-weight/physical-row replication to
    # floating-point precision when a repair leaves only a constant fit.
    if type(model._distribution) is Gaussian and type(model._link) is IdentityLink:
        weighted_residuals = weights_arr * (y_arr - eta_base)
        residual_sum = float(np.sum(weighted_residuals, dtype=np.float64))
        roundoff = (
            64.0
            * np.finfo(np.float64).eps
            * float(np.sum(np.abs(weighted_residuals), dtype=np.float64))
        )
        shift = 0.0 if abs(residual_sum) <= roundoff else residual_sum / sum_weights
        profiled_eta = eta_base + shift
        if not np.isfinite(shift) or not np.all(np.isfinite(profiled_eta)):
            raise RuntimeError(
                "Unsafe shape repair rejected before publication: invalid profiled intercept"
            )
        return (
            profiled_eta,
            shift,
            _deviance_rounding(model, profiled_eta, y=y_arr, weights=weights_arr),
        )

    carried = weights_arr != 0.0

    def evaluate(shift: float) -> _InterceptState:
        eta_safe = stabilize_eta(eta_base + shift, model._link)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
            variance = np.maximum(
                np.asarray(model._distribution.variance(mu), dtype=np.float64),
                _VARIANCE_FLOOR,
            )
            dmu_deta = np.asarray(model._link.deriv_inverse(eta_safe), dtype=np.float64)
            score_rows = weights_arr * (y_arr - mu) * dmu_deta / variance
            residual_rounding = _residual_rounding(y_arr, mu, eta_safe, dmu_deta)
            score_rounding = np.abs(weights_arr * dmu_deta / variance) * residual_rounding
            # ``_deviance_rounding`` from this pass's rows, over the carried rows.
            deviance_rows = np.abs(weights_arr * (y_arr - mu) / variance)[carried]
            deviance_rounding = 2.0 * float(
                np.sum(deviance_rows * residual_rounding[carried], dtype=np.float64)
            )
            information_rows = fisher_working_weights(
                distribution=model._distribution,
                link=model._link,
                mu=mu,
                eta=eta_safe,
                sample_weight=weights_arr,
                dmu_deta=dmu_deta,
                variance=variance,
            )
            deviance = weighted_row_sum(weights_arr, model._distribution.deviance_unit, y_arr, mu)
            score = float(np.sum(score_rows, dtype=np.float64))
            information = float(np.sum(information_rows, dtype=np.float64))
        if not (
            np.isfinite(deviance)
            and np.isfinite(score)
            and np.isfinite(information)
            and information > 0.0
        ):
            raise RuntimeError(
                "Unsafe shape repair rejected before publication: invalid profiled intercept"
            )
        return _InterceptState(
            deviance=deviance,
            deviance_rounding=deviance_rounding,
            score=score,
            score_scale=float(np.sum(np.abs(score_rows), dtype=np.float64)),
            score_rounding=float(np.sum(score_rounding, dtype=np.float64)),
            information=information,
        )

    def stationary(state: _InterceptState, relative: float) -> bool:
        # Relative to the scores' own size, and never tighter than the rounding
        # their residuals carry.
        return abs(state.score) <= relative * state.score_scale + state.score_rounding

    shift = 0.0
    state = evaluate(shift)
    for _ in range(50):
        if stationary(state, 1e-11):
            return eta_base + shift, shift, state.deviance_rounding

        proposal = state.score / state.information
        if not np.isfinite(proposal):
            break

        accepted = False
        for halving in range(21):
            trial_shift = shift + proposal * (2.0**-halving)
            trial = evaluate(trial_shift)
            # Two deviances' rounding: relative to each sum, and what each keeps
            # from its residuals.
            roundoff = (
                64.0 * np.finfo(np.float64).eps * max(abs(state.deviance), abs(trial.deviance))
                + state.deviance_rounding
                + trial.deviance_rounding
            )
            if trial.deviance <= state.deviance + roundoff:
                shift, state = trial_shift, trial
                accepted = True
                break
        if not accepted:
            break

    if not stationary(state, 1e-9):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: conditional intercept "
            "optimization did not converge"
        )
    return eta_base + shift, shift, state.deviance_rounding


def _shape_candidate_objective(
    model,
    beta,
    *,
    eta,
    y,
    weights,
    smooth_penalty_terms,
    selection_penalty,
) -> tuple[float, float]:
    """Return weighted deviance and the fitted penalized merit for a coefficient state."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.shape != model.result.beta.shape or not np.all(np.isfinite(beta_arr)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid coefficients")

    eta = np.asarray(eta, dtype=np.float64)
    if eta.shape != (model._dm.n,):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid predictor")
    if not np.all(np.isfinite(eta)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite linear predictor"
        )
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eta_safe = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
        deviance_units = np.asarray(model._distribution.deviance_unit(y, mu), dtype=np.float64)
    carried = np.asarray(weights, dtype=np.float64) != 0.0
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(deviance_units[carried])):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: invalid link-domain predictions"
        )

    deviance = weighted_row_sum(weights, lambda units: units, deviance_units)
    smooth_value = _smooth_penalty_value(beta_arr, smooth_penalty_terms)
    selection_value = 2.0 * float(selection_penalty.eval(beta_arr, model._groups))
    merit = deviance + smooth_value + selection_value
    if not np.isfinite(deviance) or not np.isfinite(merit):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite fitted objective"
        )
    return deviance, merit


def _shape_term_eta_delta(model, current_beta, candidate_beta, groups) -> np.ndarray:
    """Evaluate one public-term coefficient change through retained group kernels."""
    current = np.asarray(current_beta, dtype=np.float64)
    candidate = np.asarray(candidate_beta, dtype=np.float64)
    delta = candidate - current
    group_keys = {(group.name, group.start, group.end) for group in groups}
    contribution = np.zeros(model._dm.n, dtype=np.float64)
    for group_matrix, group in zip(
        model._dm.group_matrices,
        model._groups,
        strict=True,
    ):
        if (group.name, group.start, group.end) in group_keys:
            contribution += np.asarray(group_matrix.matvec(delta[group.sl]), dtype=np.float64)
    contribution -= canonical_intercept_shift(model, delta)
    if not np.all(np.isfinite(contribution)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite term predictor"
        )
    return contribution


def _certified_curve_scale(target: NDArray, weights: NDArray) -> float:
    """The scale a repairer certifies its projection at: the curve it repaired, as a weighted RMS.

    As ``constraints._project_shape_in_fitted_basis`` takes it: in units of a power of two
    near the curve's largest value, which is exact, and 1 for the zero curve.
    """
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    largest = float(np.max(np.abs(target), initial=0.0))
    unit = float(np.ldexp(1.0, int(np.frexp(largest)[1]))) if largest > 0.0 else 1.0
    mean_square = float(np.dot(weights, (target / unit) ** 2)) / float(np.sum(weights))
    return unit * float(np.sqrt(mean_square)) or 1.0


def _settled_on_rule(spec, kind, beta, grid, share, certified_scale) -> NDArray | None:
    """``beta`` as its rule publishes it, or ``None`` when it breaks the rule beyond rounding.

    A repairer certifies its projection at ``certified_scale``, the larger of the result's
    norm and the curve's (``_certified_curve_scale``); a later reader judges the result at
    its own norm. A result off its rule there by no more than its certified rounding is moved
    strictly inside the cone along the repairer's own inside ray by twice the violation: the
    least slack is superadditive in the coefficients, so every slack is then at least the
    violation, and the curve moves by a rounding of the curve repaired. Where the basis holds
    no ray strictly inside (a natural curvature boundary), the result is published as
    certified. ``share`` is the grid's weights, summing to one.
    """
    from superglm.constraints import (
        _inside_ray,
        shape_constraint_certificate,
        shape_slack_tolerance,
    )

    violation = -shape_constraint_certificate(spec, beta, kind).minimum_scaled_slack
    if violation <= shape_slack_tolerance(beta):
        return beta
    if violation > shape_slack_tolerance(beta, certified_scale):
        return None
    ray = _inside_ray(np.asarray(spec.transform(grid), dtype=np.float64), grid, share, kind)
    ray_slack = shape_constraint_certificate(spec, ray, kind).minimum_scaled_slack
    return beta + (2.0 * violation / ray_slack) * ray if ray_slack > 0.0 else beta


class _WorseThanFallbackError(RuntimeError):
    """The rule's refusal of a candidate that the feasible zero term beats in its merit."""


def _validate_repair_for_publication(
    model,
    *,
    spec,
    kind,
    groups,
    repair_result,
    grid_weights,
    y,
    scoring_weights,
    smooth_penalty_terms,
    selection_penalty,
    current_eta,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Certify fitted geometry and objective quality before changing private state."""
    from superglm.constraints import shape_slack_tolerance, shape_sum_tolerance

    current_beta = np.asarray(model.result.beta, dtype=np.float64)
    candidate_beta = np.asarray(repair_result.repaired_beta_reparam, dtype=np.float64)
    if candidate_beta.shape != current_beta.shape or not np.all(np.isfinite(candidate_beta)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid coefficients")

    grid = np.asarray(repair_result.grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid repair grid")
    term_beta = np.concatenate([candidate_beta[group.sl] for group in groups])
    current_term_beta = np.concatenate([current_beta[group.sl] for group in groups])
    fitted_scale = float(np.linalg.norm(current_term_beta))
    # A term within the slack's rounding of zero at the fitted term's scale is
    # the zero term, and is published as exactly that: a later call sees only
    # the published term, and would judge the rounding at its own scale.
    if float(np.linalg.norm(term_beta)) <= shape_slack_tolerance(term_beta, fitted_scale):
        candidate_beta = candidate_beta.copy()
        for group in groups:
            candidate_beta[group.sl] = 0.0
        term_beta = np.zeros_like(term_beta)
    fitted_basis = np.asarray(spec.transform(grid), dtype=np.float64)
    if fitted_basis.shape != (grid.size, term_beta.size):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: incoherent fitted-basis geometry"
        )
    repaired_curve = fitted_basis @ term_beta
    if not np.all(np.isfinite(repaired_curve)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite repair curve"
        )

    if kind in {"increasing", "decreasing", "convex", "concave"}:
        from superglm.constraints import shape_constraint_certificate

        # Judged at the scale the repairer certified at, as SuperLSS's repair judges it.
        weights = np.asarray(grid_weights, dtype=np.float64)
        certified_scale = _certified_curve_scale(repair_result.original_log_effect, weights)
        settled = _settled_on_rule(
            spec, kind, term_beta, grid, weights / float(np.sum(weights)), certified_scale
        )
        if settled is None:
            constraint_name = (
                "monotonicity" if kind in {"increasing", "decreasing"} else "curvature"
            )
            raise RuntimeError(
                f"Unsafe shape repair rejected before publication: infeasible {constraint_name}"
            )
        if settled is not term_beta:
            candidate_beta = candidate_beta.copy()
            offset = 0
            for group in groups:
                candidate_beta[group.sl] = settled[offset : offset + group.size]
                offset += group.size
            term_beta = settled
            repaired_curve = fitted_basis @ term_beta
        certificate = shape_constraint_certificate(spec, term_beta, kind)

    candidate_eta_delta = _shape_term_eta_delta(
        model,
        current_beta,
        candidate_beta,
        groups,
    )
    mean_shift = float(np.mean(candidate_eta_delta))
    centering_tolerance = shape_sum_tolerance(
        float(np.max(np.abs(candidate_eta_delta))), candidate_eta_delta.size
    )
    if not np.isfinite(mean_shift):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: fitted centering changed"
        )
    if abs(mean_shift) > centering_tolerance and _term_publishes_a_centred_contribution(
        model, groups
    ):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: fitted centering changed"
        )

    fallback_beta = current_beta.copy()
    for group in groups:
        fallback_beta[group.sl] = 0.0
    fallback_eta_delta = _shape_term_eta_delta(
        model,
        current_beta,
        fallback_beta,
        groups,
    )
    candidate_eta, candidate_intercept_shift, candidate_rounding = _profile_repaired_intercept(
        model,
        np.asarray(current_eta, dtype=np.float64) + candidate_eta_delta,
        y=y,
        weights=scoring_weights,
    )
    fallback_eta, _, fallback_rounding = _profile_repaired_intercept(
        model,
        np.asarray(current_eta, dtype=np.float64) + fallback_eta_delta,
        y=y,
        weights=scoring_weights,
    )
    candidate_deviance, candidate_merit = _shape_candidate_objective(
        model,
        candidate_beta,
        eta=candidate_eta,
        y=y,
        weights=scoring_weights,
        smooth_penalty_terms=smooth_penalty_terms,
        selection_penalty=selection_penalty,
    )
    fallback_deviance, fallback_merit = _shape_candidate_objective(
        model,
        fallback_beta,
        eta=fallback_eta,
        y=y,
        weights=scoring_weights,
        smooth_penalty_terms=smooth_penalty_terms,
        selection_penalty=selection_penalty,
    )
    deviance_rounding = candidate_rounding + fallback_rounding
    deviance_tolerance = shape_sum_tolerance(abs(fallback_deviance), len(y)) + deviance_rounding
    merit_tolerance = shape_sum_tolerance(abs(fallback_merit), len(y)) + deviance_rounding
    if (
        candidate_deviance > fallback_deviance + deviance_tolerance
        or candidate_merit > fallback_merit + merit_tolerance
    ):
        raise _WorseThanFallbackError(
            "Unsafe shape repair rejected before publication: candidate objective is worse "
            "than the feasible zero-term fallback"
        )

    repair_result.repaired_beta_reparam = candidate_beta
    repair_result.repaired_log_effect = repaired_curve.copy()
    if kind in {"increasing", "decreasing", "convex", "concave"}:
        repair_result.max_violation_after = certificate.violation
    return candidate_beta, candidate_eta, candidate_intercept_shift


# A repair's projection is the first step of the constrained penalized fit at the fit's
# smoothing parameters, and is exact only where the merit is quadratic. Under a log link,
# where the rule holds the curve far from the fit, the fit's quadratic model can rank a
# feasible curve the zero term beats in true merit. That step is then continued as
# iteratively re-weighted cone projection (Liao & Meyer, J. Stat. Softw. 89(5), 2019,
# section 1.2), the SQP of a convex merit over a polyhedral cone (Masselot et al.,
# arXiv:2509.18406, 2025, supplement B): at each feasible iterate the merit's Fisher model,
# intercept profiled, is minimised over the cone by the same repairer, and the next iterate
# lies on the segment back to the current one, which the cone holds, at Armijo's sufficient
# decrease by halving (Nocedal & Wright, Numerical Optimization, 2nd ed., 2006, section
# 3.1). The merit is convex and the zero term is feasible, so the iterates approach a curve
# no worse than it. A group-lasso selection penalty enters the line search, not the model.
#: Cone projections a continued repair takes at most; it stops long before where it converges.
_CONTINUED_STEPS = 50
#: Armijo's sufficient-decrease fraction, the customary 1e-4 (Nocedal & Wright, section 3.1).
_ARMIJO = 1e-4
#: Halvings of one continued step before the continuation stops.
_STEP_HALVINGS = 20


def _continued_repair(
    model,
    *,
    spec,
    kind,
    groups,
    repair_result,
    grid_weights,
    n_grid,
    y,
    scoring_weights,
    smooth_penalty_terms,
    selection_penalty,
    current_eta,
):
    """The refused repair continued towards the constrained penalized fit, or ``None``.

    Starts from the refused candidate and stops where the model's predicted decrease or a
    step's actual decrease is within the merit's rounding, or a step finds no decrease.
    ``None`` when no step is taken; the publication rule judges the result as before.
    """
    from dataclasses import replace

    from superglm.constraints import shape_sum_tolerance
    from superglm.distributions import _VARIANCE_FLOOR, clip_mu
    from superglm.links import stabilize_eta

    current_beta = np.asarray(model.result.beta, dtype=np.float64)
    base_eta = np.asarray(current_eta, dtype=np.float64)
    indices = _term_indices(groups)
    penalty = _term_penalty_block(smooth_penalty_terms, indices, current_beta.size)
    repairer = _repairer(kind)

    def scored(beta):
        delta = _shape_term_eta_delta(model, current_beta, beta, groups)
        eta, _, rounding = _profile_repaired_intercept(
            model, base_eta + delta, y=y, weights=scoring_weights
        )
        _, merit = _shape_candidate_objective(
            model,
            beta,
            eta=eta,
            y=y,
            weights=scoring_weights,
            smooth_penalty_terms=smooth_penalty_terms,
            selection_penalty=selection_penalty,
        )
        return merit, eta, shape_sum_tolerance(abs(merit), len(y)) + rounding

    beta = np.asarray(repair_result.repaired_beta_reparam, dtype=np.float64)
    merit, eta, tolerance = scored(beta)
    moved = False
    for _ in range(_CONTINUED_STEPS):
        eta_safe = stabilize_eta(eta, model._link)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
            variance = np.maximum(
                np.asarray(model._distribution.variance(mu), dtype=np.float64),
                _VARIANCE_FLOOR,
            )
            dmu_deta = np.asarray(model._link.deriv_inverse(eta_safe), dtype=np.float64)
            score = scoring_weights * (y - mu) * dmu_deta / variance
        W = fisher_working_weights(
            distribution=model._distribution,
            link=model._link,
            mu=mu,
            eta=eta_safe,
            sample_weight=scoring_weights,
            dmu_deta=dmu_deta,
            variance=variance,
        )
        term = beta[indices]
        H = _term_data_hessian(model, groups, W) + penalty
        # Half the merit's gradient with the intercept profiled: the Schur complement's.
        profiled_score = _term_rmatvec(model, groups, score) - _term_rmatvec(
            model, groups, W
        ) * float(np.sum(score) / np.sum(W))
        gradient = penalty @ term - profiled_score
        target = beta.copy()
        try:
            target[indices] = term - np.linalg.solve(H, gradient)
            projected = repairer.repair(
                spec, target, groups, weights=grid_weights, n_grid=n_grid, metric=H
            )
        except (np.linalg.LinAlgError, RuntimeError):
            break
        cone_point = np.asarray(projected.repaired_beta_reparam, dtype=np.float64)[indices]
        step = cone_point - term
        slope = float(gradient @ step)
        if not slope < 0.0 or -2.0 * (slope + 0.5 * float(step @ H @ step)) <= tolerance:
            break
        length = 1.0
        for _ in range(_STEP_HALVINGS):
            trial = beta.copy()
            # On the segment, exact at its ends.
            trial[indices] = (1.0 - length) * term + length * cone_point
            try:
                trial_merit, trial_eta, trial_tolerance = scored(trial)
            except RuntimeError:
                trial_merit = np.inf
            if trial_merit <= merit + _ARMIJO * length * 2.0 * slope:
                break
            length *= 0.5
        else:
            break
        decrease = merit - trial_merit
        beta, merit, eta, tolerance = trial, trial_merit, trial_eta, trial_tolerance
        moved = True
        if decrease <= tolerance:
            break
    if not moved:
        return None
    curve = np.asarray(spec.transform(repair_result.grid), dtype=np.float64) @ beta[indices]
    return replace(
        repair_result,
        repaired_beta_reparam=beta,
        repaired_log_effect=curve,
        projection_residual=float(np.max(np.abs(curve - repair_result.original_log_effect))),
    )


@dataclass
class _PendingRepair:
    """One term whose declared post-fit shape has to be projected.

    ``spec`` is the feature as the model holds it; ``curve_spec`` is the object whose
    coefficient space the repair works in.  They are the same for a plain spline and
    differ for a wrapper such as ``OrderedCategorical`` (an inner spline over mapped
    level scores): geometry comes from ``curve_spec`` while the column, groups and
    recorded result stay keyed to the declared feature.
    """

    name: str
    spec: object
    curve_spec: object
    kind: str
    groups: list[object]


def _curve_spec(name: str, spec):
    """The basis whose coefficient space a shape repair operates in.

    Selection is by declaration but resolution is by type, and the two can drift: a
    spec that grows a ``constraint=`` without being a spline (a ``Piecewise`` with a
    monotone segment) would reach consumers that read spline internals (``_lo``,
    ``_hi``, ``_knots``, ``reconstruct``).  Refuse by name instead, with
    ``NotImplementedError`` (an engine never written for the basis in hand, as in
    ``_reject_structured_fit_constraints``).
    """
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.spline import _SplineBase

    curve_spec = spec._basis_spline if isinstance(spec, OrderedCategorical) else spec
    if not isinstance(curve_spec, _SplineBase):
        raise NotImplementedError(
            f"Feature {name!r} declares a post-fit shape constraint, but post-fit "
            f"repair is only implemented for spline bases; {type(curve_spec).__name__} "
            "states no curve geometry to project. Drop the constraint, or use a "
            "Spline basis."
        )
    return curve_spec


def _pending_repairs(model) -> list[_PendingRepair]:
    """Return repairs that can change coefficients without cloning fitted state.

    Selects on the declaration (a spec that names a post-fit shape) rather than on
    the spec's type, so a constraint declared on a wrapped spline is not skipped.
    The repair is about the smooth, so it takes the feature's spline blocks through
    the shared filter: an ordered term with ``specials=`` owns an unpenalized
    free-level block that must not be read as the spline's trailing coefficients.
    """
    from superglm.inference._term_helpers import spline_groups

    beta = model.result.beta
    pending: list[_PendingRepair] = []
    for name, spec in model._specs.items():
        # A post-fit repair is always one whole-curve shape (``Constraint.postfit`` refuses
        # combinations), so no rule collection reaches ``_repairer``.
        postfit = next((rule for rule in rules_of(spec) if rule.mode == "postfit"), None)
        if postfit is None:
            continue
        kind = postfit.kind
        groups = spline_groups([group for group in model._groups if group.feature_name == name])
        if groups and any(np.any(beta[group.sl] != 0.0) for group in groups):
            pending.append(_PendingRepair(name, spec, _curve_spec(name, spec), kind, groups))
    return pending


def _repair_axis_column(pending_repair, frame) -> tuple[NDArray, NDArray]:
    """The numeric axis the term's shape lives on, and its per-row weights mask.

    A plain spline's axis IS its column. A wrapped ordinal term's column carries
    labels, and its shape lives on the inner spline's level-score axis, so the
    axis has to be resolved through the wrapper -- and rows held out of the
    smooth by ``specials=`` occupy no point on it.
    """
    from superglm.features.ordered_categorical import OrderedCategorical

    spec = pending_repair.spec
    if isinstance(spec, OrderedCategorical):
        return spec.shape_axis(frame.column_array(pending_repair.name))
    column = frame.column_array(pending_repair.name, dtype=np.float64)
    return column, np.ones(len(column), dtype=bool)


def _repair_changes_coefficients(model, pending_repair: _PendingRepair) -> bool:
    """Return whether span-wise certification requires a coefficient projection."""
    from superglm.constraints import shape_constraint_is_roundoff_feasible

    term_beta = np.concatenate([model.result.beta[group.sl] for group in pending_repair.groups])
    return not shape_constraint_is_roundoff_feasible(
        pending_repair.curve_spec, term_beta, pending_repair.kind
    )


def _refresh_repaired_geometry(model, prior_working_weights, prior_selected_names) -> None:
    """Rebuild exact fitted geometry after any published coefficient revision."""
    from types import SimpleNamespace

    from superglm.model import state_ops
    from superglm.model.fit_state import fitted_penalty
    from superglm.solvers.pirls import _selection_local_curvature_depends_on_beta
    from superglm.solvers.rank import selected_group_name_set

    current_working_weights = state_ops._solver_space_working_weights(model)
    coefficient_state = SimpleNamespace(beta=model._solver_result.beta, rank_info=None)
    current_selected_names = selected_group_name_set(
        coefficient_state,
        model._groups,
        penalty=fitted_penalty(model),
    )
    weights_unchanged = np.array_equal(current_working_weights, prior_working_weights)
    selection_unchanged = current_selected_names == prior_selected_names
    curvature_changes = _selection_local_curvature_depends_on_beta(fitted_penalty(model))
    record = model._solver_result.rank_info
    if weights_unchanged and selection_unchanged and not curvature_changes and record is not None:
        return

    # A repair moves only its own term's coefficients and the intercept, so a record
    # of the rows a constrained fit held is formed again at the new weights over the
    # active design; it is kept only where every column was selected, else dropped.
    refreshed = (
        state_ops.face_record_at(model, current_working_weights)
        if record is not None
        and record.face_rows is not None
        and selection_unchanged
        and record.selected_columns.size == record.mean_x.size
        else None
    )
    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.rank_info = refreshed
        updated.add(id(result))

    inference = state_ops.fit_inference_info(model)
    effective_df = 1.0 + float(np.sum(inference["edf"]))
    updated.clear()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.effective_df = effective_df
        updated.add(id(result))
    model.__dict__["_fit_inference_info"] = inference


def _refresh_repaired_scale_and_statistics(model) -> None:
    """Pair the repaired coefficients and final EDF with a coherent scale."""
    from superglm.distributions import _VARIANCE_FLOOR
    from superglm.model.fit_ops import _compute_fit_stats

    y = np.asarray(model._fit_y_ref, dtype=np.float64)
    weights = np.asarray(model._fit_weights, dtype=np.float64)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    if getattr(model._distribution, "scale_known", True):
        phi = 1.0
    elif getattr(model, "_last_fit_meta", {}).get("method") == "fit_reml":
        from superglm.model.reml_finalize import compute_profiled_phi

        phi = compute_profiled_phi(
            model,
            y=y,
            sample_weight=weights,
            lambdas=model._reml_lambdas,
            reml_penalties=model._reml_penalties,
            pirls_result=model._solver_result,
        )
    else:
        variance = np.maximum(model._distribution.variance(mu), _VARIANCE_FLOOR)
        # a zero-weight row holds an exact 0, as in ``weighted_row_sum`` (#369)
        carried = weights != 0.0
        terms = np.zeros_like(weights)
        terms[carried] = weights[carried] * (y[carried] - mu[carried]) ** 2 / variance[carried]
        pearson = float(np.sum(terms))
        residual_df = pearson_residual_degrees_of_freedom(
            weights,
            model._solver_result.effective_df,
            weight_semantics=model_weight_semantics(model),
        )
        phi = pearson / residual_df

    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.phi = float(phi)
        updated.add(id(result))
    model._fit_stats = _compute_fit_stats(
        y,
        mu,
        weights,
        model._fit_offset,
        model._distribution,
        model._link,
        float(phi),
        null_mu=model._fit_null_mu,
        weight_semantics=model_weight_semantics(model),
    )
    model._summary_cache = None


def apply_shape_postfit(model, X, sample_weight=None, offset=None, *, n_grid: int = 500):
    """Repair the shapes of a fitted model's smooths after the fit."""

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling apply_shape_postfit().")
    if not getattr(model, "_retain_fit_state", True):
        raise RuntimeError(
            "Post-fit shape repair requires row-scale fitted state; refit with "
            "retain_fit_state=True before calling apply_shape_postfit()."
        )

    pending_repairs = _pending_repairs(model)
    if not pending_repairs:
        return model

    y_ref = getattr(model, "_fit_y_ref", None)
    if y_ref is None:
        raise RuntimeError("Post-fit shape repair requires retained fit features and response.")
    from superglm.model.fit_data_guard import require_unchanged_fit_data

    require_unchanged_fit_data(model, X, y_ref)
    frame = as_eager_frame(X)
    fit_data_guard = model._fit_data_guard
    if not fit_data_guard.matches(
        X,
        y_ref,
        sample_weight,
        offset,
        fit_weights=getattr(model, "_fit_weights", None),
        fit_offset=getattr(model, "_fit_offset", None),
        own_frame=X is getattr(model, "_fit_X_ref", None),
    ):
        raise RuntimeError(
            "post-fit sample_weight and offset must match the fitted data; "
            "refit before repairing with different scoring geometry"
        )
    if not any(_repair_changes_coefficients(model, repair) for repair in pending_repairs):
        # Shape-feasible terms are an exact publication no-op: no metadata,
        # revision, convergence flag, accepted REML state, or result identity changes.
        return model
    if any(group.monotone_engine == "scop" for group in model._groups):
        raise RuntimeError(
            "Post-fit shape repair cannot be combined with fitted SCOP terms until "
            "the repaired mode can rebuild one coherent joint SCOP inference geometry."
        )

    from superglm.model import state_ops
    from superglm.model.fit_state import fitted_penalty
    from superglm.solvers.rank import selected_group_name_set

    prior_working_weights = state_ops._solver_space_working_weights(model)
    prior_selected_names = selected_group_name_set(
        model._solver_result,
        model._groups,
        penalty=fitted_penalty(model),
    )

    revision = FittedStateRevision.start(model)
    work_model = revision.model
    work_model._shape_repairs = dict(getattr(model, "_shape_repairs", {}))
    work_model._monotone_repairs = dict(getattr(model, "_monotone_repairs", {}))
    revision_beta = np.array(work_model.result.beta, dtype=np.float64)

    scoring_weight = (
        sample_weight if sample_weight is not None else getattr(work_model, "_fit_weights", None)
    )
    scoring_weights_arr = np.asarray(scoring_weight, dtype=np.float64)
    scoring_offset = offset if offset is not None else getattr(work_model, "_fit_offset", None)

    selection_penalty = fitted_penalty(work_model)
    smooth_penalty_terms = _build_smooth_penalty_terms(work_model)
    from superglm.solvers.mode_score import linear_predictor

    current_eta = linear_predictor(
        work_model._dm,
        work_model._solver_result,
        None if scoring_offset is None else np.asarray(scoring_offset, dtype=np.float64),
    )

    # An ordered term carrying `specials=` passes the centering check below because
    # the wrapper builds its inner spline on the ordered rows alone and zero-fills the
    # special rows afterwards: the identifiability projection is a column sum over
    # exactly the rows the block is nonzero on, so any delta confined to the spline
    # block leaves the fitted mean alone (`_validate_repair_for_publication` refuses
    # repairs that move it).  Building the block over all rows would break this
    # silently.
    for pending in pending_repairs:
        name, kind, groups = pending.name, pending.kind, pending.groups
        curve_spec = pending.curve_spec
        beta = work_model.result.beta
        beta_before = np.asarray(beta, dtype=np.float64).copy()
        x_col, axis_rows = _repair_axis_column(pending, frame)
        # Rows a `specials=` level holds out of the smooth carry no coordinate on the
        # axis, so the histogram behind the grid weights loses the same rows.
        axis_weights = scoring_weights_arr if axis_rows.all() else scoring_weights_arr[axis_rows]
        grid_weights = _grid_weights(curve_spec, x_col, axis_weights, n_grid)

        repair_result = _repairer(kind).repair(
            curve_spec,
            beta,
            groups,
            weights=grid_weights,
            n_grid=n_grid,
            metric=_term_penalized_hessian(
                work_model, groups, smooth_penalty_terms, prior_working_weights
            ),
        )
        repair_result.feature_name = name

        scoring = dict(
            spec=curve_spec,
            kind=kind,
            groups=groups,
            grid_weights=grid_weights,
            y=np.asarray(y_ref, dtype=np.float64),
            scoring_weights=scoring_weights_arr,
            smooth_penalty_terms=smooth_penalty_terms,
            selection_penalty=selection_penalty,
            current_eta=current_eta,
        )
        try:
            candidate_beta, current_eta, intercept_shift = _validate_repair_for_publication(
                work_model, repair_result=repair_result, **scoring
            )
        except _WorseThanFallbackError:
            continued = _continued_repair(
                work_model, repair_result=repair_result, n_grid=n_grid, **scoring
            )
            if continued is None:
                raise
            repair_result = continued
            candidate_beta, current_eta, intercept_shift = _validate_repair_for_publication(
                work_model, repair_result=repair_result, **scoring
            )
        # Recorded only if this term's coefficients actually moved.  The early exit
        # above concerns the whole model: once any term binds the loop runs for every
        # pending term, and an already-feasible term comes back unchanged.  Recording it
        # as repaired would withhold its published inference although no cone projection
        # happened.
        projected = not np.array_equal(candidate_beta, beta_before)
        _replace_result_beta(work_model, candidate_beta)
        move_public_intercept(work_model, intercept_shift)
        if projected:
            work_model._shape_repairs[name] = repair_result
            if kind in {"increasing", "decreasing"}:
                work_model._monotone_repairs[name] = repair_result

    publish_revised_coefficients(work_model, revision_beta)
    _invalidate_repair_caches(work_model)
    from superglm.editor.apply import _refresh_fit_statistics

    _refresh_fit_statistics(
        work_model,
        X=X,
        y=y_ref,
        sample_weight=scoring_weight,
        offset=scoring_offset,
        use_fitted_design=True,
    )
    _refresh_repaired_geometry(work_model, prior_working_weights, prior_selected_names)
    _refresh_repaired_scale_and_statistics(work_model)
    return revision.commit()
