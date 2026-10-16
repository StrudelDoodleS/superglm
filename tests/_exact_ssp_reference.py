"""The exact SSP moments as master computed them before the factored form (test reference only).

``_group_matrix_kernels._exact_ssp_moments`` now forms ``R'(B'MB)R`` and
``R'(B'm)`` from vectorised exact bin sums. This is the row-at-a-time
version it replaced, kept verbatim so the tests can demand bit-for-bit
equality: both round the same exact dyadic rationals to float64 once.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def reference_exact_ssp_moments(basis, transform, weights, weighted_rhs=None, *, bin_indices=None):
    """Round the requested source-factor moments once on exceptional inputs.

    Finite binary64 entries are exact dyadic rationals. Accumulating B R,
    X' W X and optional X' W / X' Wz over those entries avoids an overflowing
    raw Gram, reciprocal basis or bin mass. No rank or regularization decision
    is made here. The target is the represented source product B R, before
    floating-point matrix multiplication rounds that product.

    A shared binary scale per source operand makes every inner operation an
    integer operation. Binary64's finite exponent span bounds integer widths,
    apart from logarithmic growth with reduction length; no inner product
    needs rational normalization. Only final outputs use Fraction rounding.
    Only one effective row and the requested output are retained. Sparse B
    stays sparse outside the current row; discrete weights are aggregated
    exactly before visiting their support rows. Ordinary operands never use
    this path.
    """
    weights = np.asarray(weights)
    if bin_indices is None:
        n_observations = basis.shape[0]
    else:
        bin_indices = np.asarray(bin_indices)
        if (
            bin_indices.ndim != 1
            or bin_indices.dtype.kind not in "iu"
            or np.any(bin_indices < 0)
            or np.any(bin_indices >= basis.shape[0])
        ):
            raise ValueError("SSP bins must be one-dimensional support-row indices.")
        n_observations = bin_indices.size
    if weights.shape != (n_observations,):
        raise ValueError("SSP moments require one weight per observation.")
    if weighted_rhs is not None:
        weighted_rhs = np.asarray(weighted_rhs)
        if weighted_rhs.shape != weights.shape:
            raise ValueError("SSP moments require one weighted RHS per observation.")

    def binary_scale(values):
        values = np.asarray(values, dtype=np.float64)
        _, exponents = np.frexp(values)
        # A binary64 value with frexp exponent e is an integer times 2**(e-53).
        # Including nonfinite values in this scan is harmless: conversion
        # below validates all weights/transforms, but only active basis rows.
        return int(np.min(exponents, where=values != 0, initial=1024)) - 53

    def exact(value, scale):
        try:
            numerator, denominator = float(value).as_integer_ratio()
        except (ValueError, OverflowError) as error:
            raise np.linalg.LinAlgError("SSP moments require finite source factors.") from error
        shift = 1 - denominator.bit_length() - scale
        # The chosen scale divides every represented source value exactly,
        # including when large integral inputs require a right shift.
        return numerator << shift if shift >= 0 else numerator >> -shift

    def rounded(value, scale):
        return float(value << scale) if scale >= 0 else float(Fraction(value, 1 << -scale))

    sparse = getattr(basis, "format", None) == "csr"
    basis_scale = binary_scale(basis.data if sparse else basis)
    transform_scale = binary_scale(transform)
    weight_scale = binary_scale(weights)
    rhs_scale = 0 if weighted_rhs is None else binary_scale(weighted_rhs)
    coefficients = [
        [exact(value, transform_scale) for value in row] for row in np.asarray(transform)
    ]
    width = transform.shape[1]
    gram = [[0 for _ in range(width)] for _ in range(width)]
    xtw = [0 for _ in range(width)] if weighted_rhs is not None else None
    xtrhs = [0 for _ in range(width)] if weighted_rhs is not None else None
    if bin_indices is None:
        masses = [exact(value, weight_scale) for value in weights]
        rhs_masses = (
            None if weighted_rhs is None else [exact(value, rhs_scale) for value in weighted_rhs]
        )
    else:
        masses = [0 for _ in range(basis.shape[0])]
        rhs_masses = None if weighted_rhs is None else [0 for _ in range(basis.shape[0])]
        for row, index in enumerate(bin_indices):
            masses[index] += exact(weights[row], weight_scale)
            if rhs_masses is not None:
                rhs_masses[index] += exact(weighted_rhs[row], rhs_scale)

    for row, mass in enumerate(masses):
        rhs_mass = 0 if rhs_masses is None else rhs_masses[row]
        if mass == 0 and rhs_mass == 0:
            continue
        if sparse:
            # Convert each stored term before summing duplicate columns.
            # Densifying or canonicalizing first can lose their exact sum.
            nonzero = [
                (basis.indices[index], exact(basis.data[index], basis_scale))
                for index in range(basis.indptr[row], basis.indptr[row + 1])
                if basis.data[index] != 0
            ]
        else:
            raw = np.asarray(basis[row]).ravel()
            nonzero = [
                (index, exact(value, basis_scale)) for index, value in enumerate(raw) if value != 0
            ]
        effective = [
            sum(value * coefficients[index][column] for index, value in nonzero)
            for column in range(width)
        ]
        for left, value in enumerate(effective):
            weighted = mass * value
            for right in range(left, width):
                gram[left][right] += weighted * effective[right]
            if xtw is not None and xtrhs is not None:
                xtw[left] += weighted
                xtrhs[left] += rhs_mass * value
    for left in range(width):
        for right in range(left):
            gram[left][right] = gram[right][left]
    effective_scale = basis_scale + transform_scale
    try:
        rounded_gram = np.array(
            [[rounded(value, weight_scale + 2 * effective_scale) for value in row] for row in gram]
        ).reshape(width, width)
        rounded_xtw = (
            None
            if xtw is None
            else np.array([rounded(value, weight_scale + effective_scale) for value in xtw])
        )
        rounded_rhs = (
            None
            if xtrhs is None
            else np.array([rounded(value, rhs_scale + effective_scale) for value in xtrhs])
        )
    except OverflowError as error:
        raise np.linalg.LinAlgError("Requested SSP moment is not representable.") from error
    return rounded_gram, rounded_xtw, rounded_rhs
