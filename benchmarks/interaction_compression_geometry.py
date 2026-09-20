"""Saved tensor replay and estimated product-metric compression diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RankApproximation:
    coefficients: Array
    singular_values: Array
    discarded_product_energy: float
    realized_product_energy: float
    rank_budget: int
    factor_entries: int
    energy_allowance: float
    diagnostics: dict
    allowance_is_certificate: bool = False


@dataclass(frozen=True)
class MarginalMetric:
    factor: Array
    diagnostics: dict


@dataclass(frozen=True)
class MarginalModes:
    eigenvalues: Array
    modes: Array
    diagnostics: dict


def _gamma(operations: int) -> float:
    scaled = operations * (np.finfo(float).eps / 2)
    if scaled >= 1:
        raise ValueError("Arithmetic operation budget is unsupported")
    return scaled / (1 - scaled)


def _norm(value: Array) -> float:
    gamma = _gamma(value.size + 2)
    measured = float(np.linalg.norm(value))
    if gamma >= 1 or not np.isfinite(measured) or (measured == 0 and np.any(value)):
        raise ValueError("Unsupported arithmetic scale or norm budget")
    return measured / (1 - gamma)


def _mul(left: Array, right: Array) -> float:
    return _gamma(left.shape[1]) * _norm(left) * _norm(right)


def _mul3(left: Array, middle: Array, right: Array) -> float:
    return _gamma(left.shape[1] + middle.shape[1]) * _norm(left) * _norm(middle) * _norm(right)


def _sub(left: Array, right: Array) -> float:
    return _gamma(1) * (_norm(left) + _norm(right))


def _res(left: Array, right: Array) -> float:
    return _norm(left - right) + _sub(left, right)


def _svprod(left: Array, values: Array, right: Array) -> float:
    return _gamma(len(values) + 1) * _norm(left) * _norm(values) * _norm(right)


def _orthogonality(modes: Array) -> float:
    return _res(modes.T @ modes, np.eye(modes.shape[1])) + _mul(modes.T, modes)


def _readonly(value: Array) -> Array:
    result = value.copy()
    result.flags.writeable = False
    return result


def _factor_diagnostics(factor: Array, rows: int) -> dict:
    p = factor.shape[0]
    if factor.shape != (p, p) or p == 0 or np.any(np.tril(factor, -1)):
        raise ValueError("Metric factor must be nonempty square upper triangular")
    beta = p * _gamma(rows + p + 8)
    if beta >= 1 / 16:
        raise ValueError("Metric arithmetic budget is unsupported")
    tau = np.sqrt(beta)
    try:
        values = np.linalg.svd(factor, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Metric singular-value factorization failed") from exc
    minimum, maximum = float(values[-1]), float(values[0])
    if minimum <= _gamma(p + 1) * _norm(factor):
        raise ValueError("Metric rank/conditioning floor is not met")
    condition = maximum / minimum
    theta = _gamma(p) * np.sqrt(p) * condition
    if not np.isfinite(theta) or theta >= 1 or theta / (1 - theta) > tau:
        raise ValueError("Metric conditioning exceeds triangular solve budget")
    return {
        "dtype": "float64",
        "rows": rows,
        "width": p,
        "factor_condition": condition,
        "metric_condition": condition**2,
        "minimum_singular_value": minimum,
        "solve_relative_error": theta / (1 - theta),
        "tau": float(tau),
        "allowance_kind": "estimated",
        "admission_policy": "half digits above dimension-dependent arithmetic floor",
    }


def weighted_marginal_metric(basis: Array, weights: Array) -> MarginalMetric:
    """Thin QR of a weighted training margin with explicit accuracy refusals.

    Admission is an engineering error-budget policy, not a rank certificate.
    It uses only the supplied training rows, never a paired or validation norm.
    """
    basis, weights = _finite_array(basis, "training basis"), _finite_array(weights, "weights")
    if (
        basis.ndim != 2
        or basis.shape[1] == 0
        or basis.shape[0] < basis.shape[1]
        or weights.shape != (basis.shape[0],)
        or np.any(weights < 0)
        or weights.max(initial=0) <= 0
    ):
        raise ValueError("Training basis/weights do not define an admitted marginal metric")
    try:
        # Inexact subnormals invalidate the relative formation allowance, even
        # when they remain positive. Exact subnormal operations need no repair.
        with np.errstate(under="raise"):
            scaled = weights / weights.max()
            A = _finite_array(np.sqrt(scaled / scaled.sum())[:, None] * basis, "weighted basis")
    except FloatingPointError as exc:
        raise ValueError("Unsupported arithmetic scale in weight normalization/formation") from exc
    if not np.any(A):
        raise ValueError("Weighted basis has zero rank")
    try:
        Q, R = np.linalg.qr(A, mode="reduced")
    except np.linalg.LinAlgError as exc:
        raise ValueError("Training marginal QR factorization failed") from exc
    d = _factor_diagnostics(_finite_array(R, "QR factor"), len(A))
    norm_A = _norm(A)
    formation = _gamma(len(A) + 8) * norm_A
    reconstruction = _res(A, Q @ R) + _mul(Q, R) + formation
    orthogonality = _orthogonality(Q)
    h = reconstruction / d["minimum_singular_value"]
    eta = orthogonality + 2 * np.sqrt(1 + orthogonality) * h + h * h
    checks = np.array([reconstruction / norm_A, orthogonality, eta])
    if not np.isfinite(checks).all() or np.any(checks > d["tau"]):
        raise ValueError("Training QR reconstruction/metric accuracy budget exceeded")
    d.update(
        formation_allowance=formation,
        reconstruction_residual=reconstruction,
        orthogonality_residual=orthogonality,
        metric_relative_error=float(eta),
    )
    return MarginalMetric(_readonly(R), d)


def _pair_budget(left: dict, right: dict) -> tuple[float, float]:
    p, q = left["width"], right["width"]
    tau = min(left["tau"], right["tau"])
    amplification = (
        _gamma(p + q + 2) * np.sqrt(p * q) * left["factor_condition"] * right["factor_condition"]
    )
    if not np.isfinite(amplification) or amplification > tau:
        raise ValueError("Product metric conditioning exceeds two-sided operation budget")
    return tau, float(amplification)


def product_metric_factors(
    left_basis: Array, right_basis: Array, weights: Array
) -> tuple[MarginalMetric, MarginalMetric]:
    """Admit both training marginals and their joint operation budget."""
    left = weighted_marginal_metric(left_basis, weights)
    right = weighted_marginal_metric(right_basis, weights)
    _pair_budget(left.diagnostics, right.diagnostics)
    return left, right


def truncate_in_product_metric(
    coefficients: Array, left_metric_factor: Array, right_metric_factor: Array, rank: int
) -> RankApproximation:
    """Whiten, truncate, recover; report estimated spectral/recovered error agreement.

    The supplied triangular factors define the product metric. Its discrepancy
    from the intended empirical measure belongs to the separate QR diagnostics.
    No allowance here is an outward-certified enclosure.
    """
    C = _finite_array(coefficients, "coefficients")
    L = _finite_array(left_metric_factor, "left metric factor")
    R = _finite_array(right_metric_factor, "right metric factor")
    if C.ndim != 2 or L.ndim != 2 or R.ndim != 2 or C.shape != (len(L), len(R)):
        raise ValueError("Incompatible product metric dimensions")
    p, q = C.shape
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise ValueError("Rank budget must be an integer")
    if not 0 <= rank <= min(p, q):
        raise ValueError("Rank budget is outside the coefficient dimensions")
    left, right = _factor_diagnostics(L, p), _factor_diagnostics(R, q)
    tau, amplification = _pair_budget(left, right)
    H = _finite_array(L @ C @ R.T, "whitened coefficients")
    try:
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        K = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        partial = solve_triangular(L, K, lower=False)
        candidate = solve_triangular(R, partial.T, lower=False).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("Product metric SVD/triangular factorization failed") from exc
    candidate = _finite_array(candidate, "recovered coefficients")
    o_U, o_V = _orthogonality(U), _orthogonality(Vt.T)
    if not np.isfinite([o_U, o_V]).all() or max(o_U, o_V) > tau:
        raise ValueError("SVD orthogonality accuracy budget exceeded")
    d_white = _mul3(L, C, R.T)
    d_svd = _res(H, (U * s) @ Vt) + _svprod(U, s, Vt)
    d_keep = _svprod(U[:, :rank], s[:rank], Vt[:rank])
    d_rec = _res(L @ candidate @ R.T, K) + _mul3(L, candidate, R.T)
    D = C - candidate
    Z = _finite_array(L @ D @ R.T, "product residual")
    d_eval = _mul3(L, D, R.T) + _norm(L) * _norm(R) * _sub(C, candidate)
    residual_total = d_white + d_svd + d_keep + d_rec + d_eval
    norm_H = _norm(H)
    if not np.isfinite(residual_total) or residual_total > tau * norm_H:
        raise ValueError("Product reconstruction accuracy budget exceeded")
    tail = float(s[rank:] @ s[rank:])
    gamma_tail = _gamma(len(s) - rank)
    tail_upper = tail / (1 - gamma_tail)
    e_tail = gamma_tail * tail_upper
    alpha = np.sqrt((1 + o_U) * (1 + o_V))
    omega = o_U + o_V + o_U * o_V
    realized = float(np.linalg.norm(Z) ** 2)
    gamma_energy = _gamma(p * q + 2)
    e_energy = gamma_energy * realized / (1 - gamma_energy)
    allowance = (
        omega * tail_upper
        + 2 * alpha * np.sqrt(tail_upper) * residual_total
        + residual_total**2
        + e_tail
        + e_energy
    )
    discrepancy = abs(realized - tail)
    if not np.isfinite([tail, realized, allowance]).all() or discrepancy > allowance:
        raise ValueError("Spectral/recovered energy discrepancy exceeds estimated allowance")
    return RankApproximation(
        _readonly(candidate),
        _readonly(s),
        tail,
        realized,
        int(rank),
        int(rank * (p + q)),
        float(allowance),
        {
            "dtype": "float64",
            "shape": [p, q],
            "left_factor": left,
            "right_factor": right,
            "tau_pair": tau,
            "pair_amplification": amplification,
            "whitened_norm": norm_H,
            "whitening_allowance": d_white,
            "svd_residual": d_svd,
            "kept_product_allowance": d_keep,
            "recovery_residual": d_rec,
            "evaluation_allowance": d_eval,
            "left_orthogonality": o_U,
            "right_orthogonality": o_V,
            "residual_total": residual_total,
            "tail_evaluation_allowance": e_tail,
            "energy_evaluation_allowance": e_energy,
            "energy_discrepancy": discrepancy,
            "allowance_kind": "estimated",
        },
    )


def penalty_modes(penalty: Array) -> MarginalModes:
    """Freeze an admitted saved-penalty eigensystem without clipping eigenvalues."""
    S = _finite_array(penalty, "saved centered penalty")
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[0] != S.shape[1]:
        raise ValueError("Saved centered penalty must be nonempty square")
    p = len(S)
    beta = p * _gamma(2 * p + 8)
    if beta >= 1 / 16:
        raise ValueError("Penalty operation budget is unsupported")
    skew = _res(S, S.T) / 2
    if skew > beta * _norm(S):
        raise ValueError("Saved penalty asymmetry exceeds the arithmetic budget")
    T = S / 2 + S.T / 2
    target_change = _res(T, S)
    try:
        values, U = np.linalg.eigh(T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Saved penalty eigensystem factorization failed") from exc
    values, U = _finite_array(values, "eigenvalues"), _finite_array(U, "penalty modes")
    o = _orthogonality(U)
    reconstruction = _res(T, (U * values) @ U.T) + _svprod(U, values, U.T)
    if o > beta or reconstruction > beta * _norm(T):
        raise ValueError("Penalty eigensystem accuracy budget exceeded")
    nu = o / (1 + np.sqrt(1 - o))
    polar = (2 * nu + nu * nu) * _norm(values)
    allowance = target_change + reconstruction + polar
    if not np.isfinite([o, reconstruction, allowance]).all():
        raise ValueError("Penalty eigensystem arithmetic scale is unsupported")
    if values[0] < -allowance:
        raise ValueError("Saved penalty has materially negative eigenvalues")
    ambiguous = np.abs(values) <= allowance
    unresolved = np.diff(values) <= (
        2 * allowance + _gamma(1) * (np.abs(values[1:]) + np.abs(values[:-1]))
    )
    return MarginalModes(
        _readonly(values),
        _readonly(U),
        {
            "dtype": "float64",
            "width": p,
            "original_sha256": hashlib.sha256(S.tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(T.tobytes()).hexdigest(),
            "basis_sha256": hashlib.sha256(U.tobytes()).hexdigest(),
            "target_changed": bool(np.any(T != S)),
            "target_change_norm": float(np.linalg.norm(T - S)),
            "target_change_allowance": target_change,
            "skew_allowance": skew,
            "beta": beta,
            "orthogonality_residual": o,
            "reconstruction_residual": reconstruction,
            "polar_allowance": polar,
            "eigenvalue_allowance": allowance,
            "ambiguous_zero_mask": ambiguous.tolist(),
            "ambiguous_nullity": int(np.count_nonzero(ambiguous)),
            "unresolved_gap_mask": unresolved.tolist(),
            "tied_cutoffs": (np.flatnonzero(unresolved) + 1).tolist(),
            "nullspace_certified": False,
            "tied_prefix_rotation_invariant": False,
            "allowance_kind": "estimated",
        },
    )


def saved_penalty_modes(spec) -> tuple[MarginalModes, MarginalModes]:
    """Call once per admitted reference and reuse these bases for every prefix."""
    widths = _tensor_widths(spec)
    results = []
    for index, (margin, width) in enumerate(
        zip((spec._marginal1, spec._marginal2), widths, strict=True), start=1
    ):
        if margin is None or np.shape(margin.penalty) != (width, width):
            raise ValueError("Saved marginal penalty differs from centered coefficient widths")
        result = penalty_modes(margin.penalty)
        result.diagnostics["source"] = f"spec._marginal{index}.penalty"
        results.append(result)
    return results[0], results[1]


def _modal_inputs(
    coefficients, left_modes, right_modes
) -> tuple[Array, Array, Array, float, float]:
    C = _finite_array(coefficients, "coefficients")
    U = _finite_array(left_modes, "left frozen modes")
    V = _finite_array(right_modes, "right frozen modes")
    if (
        C.ndim != 2
        or min(C.shape) == 0
        or U.shape != (C.shape[0],) * 2
        or V.shape != (C.shape[1],) * 2
    ):
        raise ValueError("Incompatible modal coefficient dimensions")
    o_left, o_right = _orthogonality(U), _orthogonality(V)
    for modes, o in ((U, o_left), (V, o_right)):
        beta = len(modes) * _gamma(2 * len(modes) + 8)
        if beta >= 1 / 16 or not np.isfinite(o) or o > beta:
            raise ValueError("Frozen modes exceed orthogonality arithmetic budget")
    return C, U, V, o_left, o_right


def modal_prefix(coefficients: Array, left_modes: Array, right_modes: Array, width: int) -> Array:
    """Keep an oracle rectangular prefix in the already-frozen marginal bases."""
    C, U, V, _, _ = _modal_inputs(coefficients, left_modes, right_modes)
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)) or width < 0:
        raise ValueError("Modal width must be a nonnegative integer")
    D = _finite_array(U.T @ C @ V, "modal coefficients")
    D[width:, :] = 0
    D[:, width:] = 0
    return _finite_array(U @ D @ V.T, "modal prefix coefficients")


def modal_full_reconstruction(
    coefficients: Array,
    left_modes: Array,
    right_modes: Array,
    left_basis: Array,
    right_basis: Array,
) -> tuple[Array, dict]:
    """Full-prefix coefficients and estimated coefficient/paired-score allowances."""
    C, U, V, o_left, o_right = _modal_inputs(coefficients, left_modes, right_modes)
    left, _, right = _contraction_inputs(left_basis, C, right_basis)
    D = _finite_array(U.T @ C @ V, "modal coefficients")
    full = _finite_array(U @ D @ V.T, "full modal reconstruction")
    analysis = _mul3(U.T, C, V)
    synthesis = _mul3(U, D, V.T)
    b_C = (o_left + o_right + o_left * o_right) * _norm(C)
    b_C += np.sqrt((1 + o_left) * (1 + o_right)) * analysis + synthesis
    coefficient_allowance = (b_C + _sub(full, C)) * (1 + _gamma(C.size + 2))
    H = np.sqrt(np.sum(np.sum(left**2, axis=1) * np.sum(right**2, axis=1)))
    b_score = H * (b_C + _gamma(C.size + 2) * (_norm(C) + _norm(full)))
    scores, reference = score_pairs(left, full, right), score_pairs(left, C, right)
    score_allowance = (b_score + _sub(scores, reference)) * (1 + _gamma(len(scores) + 2))
    if not np.isfinite([coefficient_allowance, score_allowance]).all():
        raise ValueError("Full modal reconstruction arithmetic scale is unsupported")
    return full, {
        "coefficient_allowance": float(coefficient_allowance),
        "score_allowance": float(score_allowance),
        "analysis_allowance": analysis,
        "synthesis_allowance": synthesis,
        "b_C": b_C,
        "b_score": float(b_score),
        "allowance_kind": "estimated",
    }


def _finite_array(value, name: str) -> Array:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    for start in range(0, array.size, 65536):
        if not np.isfinite(array.flat[start : start + 65536]).all():
            raise ValueError(f"{name} must be finite")
    return array


def _tensor_widths(spec) -> tuple[int, int]:
    # Import lazily: the admission caller must first choose the frozen runtime.
    from superglm.features.interaction import TensorInteraction

    if type(spec) is not TensorInteraction:
        raise TypeError("Only the exact TensorInteraction class has an admitted coordinate map")
    if spec._decompose:
        raise ValueError("decomposed tensor requires a separately validated map")
    widths = (spec._p1, spec._p2)
    if any(not isinstance(width, (int, np.integer)) or width <= 0 for width in widths):
        raise ValueError("Centered marginal widths must be positive integers")
    return widths


def effective_tensor_coefficients(spec, beta: Array) -> Array:
    """Apply the saved runtime map and return centered marginal coefficients."""
    p, q = _tensor_widths(spec)
    effective = _finite_array(beta, "coefficients").ravel()
    if spec._R_inv is not None:
        mapping = _finite_array(spec._R_inv, "runtime map")
        if mapping.shape != (p * q, effective.size):
            raise ValueError("runtime map shape is incompatible with the coefficient widths")
        effective = _finite_array(mapping @ effective, "mapped coefficients")
    if effective.size != p * q:
        raise ValueError("coefficient count does not match centered marginal widths")
    return effective.reshape(p, q)


def fitted_term_beta(model, term_name: str) -> Array:
    """Read the one fitted group slice used by a nondecomposed saved term."""
    spec = model._interaction_specs[term_name]
    _tensor_widths(spec)
    groups = [group for group in model._groups if group.feature_name == term_name]
    if len(groups) != 1:
        raise ValueError("A nondecomposed tensor must have exactly one fitted group")
    beta = _finite_array(model.result.beta, "fitted coefficients")
    sl = groups[0].sl
    if (
        beta.ndim != 1
        or not isinstance(sl, slice)
        or sl.step not in (None, 1)
        or sl.start is None
        or sl.stop is None
        or not 0 <= sl.start < sl.stop <= beta.size
    ):
        raise ValueError("Invalid fitted coefficient group slice")
    selected = beta[sl].copy()
    effective_tensor_coefficients(spec, selected)
    return selected


def paired_centered_bases(spec, x1: Array, x2: Array) -> tuple[Array, Array]:
    """Evaluate saved clipping and marginal projections in runtime input units.

    The input coordinates have already had the frozen training preprocessor
    applied. No centering is learned from these query rows.
    """
    widths = _tensor_widths(spec)
    coordinates = [_finite_array(x, "marginal coordinates").ravel() for x in (x1, x2)]
    if coordinates[0].shape != coordinates[1].shape:
        raise ValueError("Paired margins must have the same number of rows")
    bases = []
    for x, info, width in zip(coordinates, (spec._marginal1, spec._marginal2), widths, strict=True):
        if info is None or not np.isfinite([info.lo, info.hi]).all() or info.lo >= info.hi:
            raise ValueError("A saved marginal must have finite ordered boundaries")
        projection = _finite_array(info.projection, "saved marginal projection")
        raw = _finite_array(info.raw_basis_eval(np.clip(x, info.lo, info.hi)), "raw basis")
        if (
            raw.ndim != 2
            or raw.shape[0] != len(x)
            or projection.ndim != 2
            or projection.shape != (raw.shape[1], width)
        ):
            raise ValueError("Saved marginal projection has incompatible widths")
        bases.append(_finite_array(raw @ projection, "centered basis"))
    return bases[0], bases[1]


def _contraction_inputs(left, coefficients, right) -> tuple[Array, Array, Array]:
    left, coefficients, right = (
        _finite_array(value, name)
        for value, name in (
            (left, "left basis"),
            (coefficients, "coefficients"),
            (right, "right basis"),
        )
    )
    if (
        left.ndim != 2
        or right.ndim != 2
        or coefficients.ndim != 2
        or left.shape[0] != right.shape[0]
        or coefficients.shape != (left.shape[1], right.shape[1])
        or min(coefficients.shape) == 0
    ):
        raise ValueError("Incompatible paired contraction dimensions")
    return left, coefficients, right


def score_pairs(left_basis: Array, coefficients: Array, right_basis: Array) -> Array:
    """Evaluate paired rows in link units with at most 4096 rows of workspace."""
    left, coefficients, right = _contraction_inputs(left_basis, coefficients, right_basis)
    scores = np.empty(left.shape[0])
    for start in range(0, len(scores), 4096):
        stop = start + 4096
        scores[start:stop] = np.einsum(
            "ij,jk,ik->i", left[start:stop], coefficients, right[start:stop], optimize=False
        )
    return _finite_array(scores, "term scores")


def contraction_roundoff_estimate(left: Array, coefficients: Array, right: Array) -> Array:
    """Estimate differences between equivalent contractions of shared operands.

    For p by q coefficients, 2*p*q+p+q+2 conservatively counts roundings
    along either a direct sum of triple products or a two-stage contraction.
    Each evaluation has error at most gamma_k times the absolute-product sum
    in the usual floating-point model, so a comparison uses 2*gamma_k.
    Dividing by 1-gamma_k allows for computing that positive sum in float64.
    Using epsilon instead of epsilon/2 is conservative. This estimate assumes
    no underflow/overflow and shared evaluated bases and mapped coefficients;
    it does not certify the basis evaluator, coordinate map or saved fit.
    """
    left, coefficients, right = _contraction_inputs(left, coefficients, right)
    p, q = coefficients.shape
    eps = np.finfo(np.float64).eps
    operations = 2 * p * q + p + q + 2
    scaled = operations * eps
    if scaled >= 0.5:
        raise ValueError("Contraction is too large for this roundoff estimate")
    gamma = scaled / (1 - scaled)
    absolute_products = np.sum((np.abs(left) @ np.abs(coefficients)) * np.abs(right), axis=1)
    return _finite_array(2 * gamma / (1 - gamma) * absolute_products, "roundoff estimate")


def replay_tensor_term(spec, beta: Array, x1: Array, x2: Array) -> dict:
    """Compare extracted scores with the saved runtime on every supplied row."""
    coefficients = effective_tensor_coefficients(spec, beta)
    left, right = paired_centered_bases(spec, x1, x2)
    extracted = score_pairs(left, coefficients, right)
    reference = _finite_array(spec.score(x1, x2, beta), "runtime term scores")
    if reference.shape != extracted.shape:
        raise ValueError("Runtime term score shape differs")
    difference = np.abs(extracted - reference)
    estimate = contraction_roundoff_estimate(left, coefficients, right)
    return {
        "rows": len(extracted),
        "coefficient_shape": list(coefficients.shape),
        "score_scale": "link",
        "max_absolute_difference": float(difference.max(initial=0)),
        "max_roundoff_estimate": float(estimate.max(initial=0)),
        "within_roundoff_estimate": bool(np.all(difference <= estimate)),
        "roundoff_is_certificate": False,
        "roundoff_model": "2*gamma_k/(1-gamma_k)*sum_abs_products, k=2*p*q+p+q+2; shared operands",
    }
