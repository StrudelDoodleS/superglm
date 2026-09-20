"""Coordinates and score replay for the saved, nondecomposed tensor pilot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite_array(value, name: str) -> Array:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
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


def paired_tensor_score(left: Array, coefficients: Array, right: Array) -> Array:
    """Evaluate b_left C b_right per row, in link units."""
    left, coefficients, right = _contraction_inputs(left, coefficients, right)
    return _finite_array(np.sum((left @ coefficients) * right, axis=1), "term scores")


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
    extracted = paired_tensor_score(left, coefficients, right)
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
