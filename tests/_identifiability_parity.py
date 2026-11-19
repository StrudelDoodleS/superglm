"""Structured and dense fits of one model read one identifiability decision.

``solvers.identifiability`` decides which coefficients a fit identifies; the structured
backends (``fs``, ``sz``, nested chains) read it from their retained factors and the dense
backend from its rank record.  On a full-rank penalised Hessian both report every standard
error, the same footnotes and the same values.  On a singular one the structured factor's
split is the minimum-norm one, so every coordinate its truncated directions reach is blank,
while the dense fit fixes the later coordinate of each alias at zero and blanks that alone;
the two agree everywhere else.
"""

from __future__ import annotations

import warnings

import numpy as np

from superglm.reml.penalty_algebra import build_penalty_matrix

EPS = np.finfo(np.float64).eps


def coefficient_ses(metrics) -> np.ndarray:
    """``metrics.coefficient_se`` over the model's groups, in fitted order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = metrics.coefficient_se
    return np.concatenate(
        [np.atleast_1d(np.asarray(se[g.name], dtype=float)) for g in metrics._model._groups]
    )


def _record(model):
    inference = model._fit_inference_info
    return inference["identifiability"], np.asarray(inference["coefficient_identified"])


def assert_same_identifiability(structured_metrics, dense_metrics, *, rtol: float) -> None:
    """The same blanks, standard errors and footnotes from a structured and a dense fit."""
    structured, dense = structured_metrics._model, dense_metrics._model
    assert structured._fit_inference_info.get("structured_covariance")
    assert not dense._fit_inference_info.get("structured_covariance")
    structured_se, dense_se = coefficient_ses(structured_metrics), coefficient_ses(dense_metrics)
    shown = ~np.isnan(dense_se)
    np.testing.assert_array_equal(~np.isnan(structured_se), shown)
    np.testing.assert_allclose(structured_se[shown], dense_se[shown], rtol=rtol)
    (structured_record, structured_flags), (dense_record, dense_flags) = (
        _record(structured),
        _record(dense),
    )
    np.testing.assert_array_equal(structured_flags, shown)
    np.testing.assert_array_equal(dense_flags, shown)
    assert structured_record.notes() == dense_record.notes()


def penalised_null_reach(model, weights: np.ndarray) -> np.ndarray:
    """Slopes a null vector of the penalised Hessian ``[1, X]' W [1, X] + S`` reaches (an oracle).

    The column-equilibrated SVD of ``[sqrt(W) [1, X]; [0, R]]``, ``R' R = S`` at the fitted
    lambdas, over the positive-weight rows: a null singular value is below ``max(m, p) eps`` of
    the largest (``dgesdd``'s backward error), and a slope it moves above ``sqrt(eps)`` is reached.
    ``R`` keeps the eigenvalues of ``S`` above ``p eps`` of its largest: the root of a rounding
    eigenvalue would put ``sqrt(eps)``-sized rows on the penalty's null space.
    """
    dm = model._dm
    X = np.asarray(dm.toarray(), dtype=np.float64)
    keep = weights > 0
    rows = np.sqrt(weights[keep])[:, None] * np.column_stack([np.ones(int(keep.sum())), X[keep]])
    S = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    values, vectors = np.linalg.eigh(0.5 * (S + S.T))
    kept = values > dm.p * EPS * values[-1]
    root = np.sqrt(values[kept])[:, None] * vectors[:, kept].T
    A = np.vstack([rows, np.column_stack([np.zeros(len(root)), root])])
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    _, singular, right = np.linalg.svd(A, full_matrices=False)
    null = right[singular <= max(A.shape) * EPS * singular[0]]
    return np.any(np.abs(null[:, 1:]) > np.sqrt(EPS), axis=0)


def truncated_reach(factor) -> np.ndarray:
    """The coordinates an augmented structured factor leaves undetermined, as the owner reads it.

    ``null_directions`` in Jacobi-scaled coordinates of the factor's penalised operator
    (``solvers.identifiability.fixed_coordinates``).
    """
    from superglm.solvers.identifiability import TruncatedDecomposition, fixed_coordinates

    width = factor.shape[0]
    hessian = factor.operator.matvec(np.eye(width))
    decomposition = TruncatedDecomposition(
        null_vectors=factor.null_directions(),
        rank=int(factor.rank),
        column_scale=np.sqrt(np.maximum(np.diag(hessian), 0.0)),
        pre_truncation_condition=1.0,
    )
    reach = np.zeros(width, dtype=bool)
    reach[fixed_coordinates(decomposition)] = True
    return reach


def assert_structured_blanks_its_truncation(
    structured_metrics, dense_metrics, reach: np.ndarray, *, rtol: float
) -> None:
    """A singular Hessian: the structured fit blanks every coordinate ``reach`` marks.

    The dense fit's blanks, its representative zeros, are among them; every other standard
    error agrees.
    """
    structured_se, dense_se = coefficient_ses(structured_metrics), coefficient_ses(dense_metrics)
    assert np.any(reach)
    np.testing.assert_array_equal(np.isnan(structured_se), reach)
    dense_blank = np.isnan(dense_se)
    assert np.any(dense_blank) and np.all(reach[dense_blank])
    np.testing.assert_allclose(structured_se[~reach], dense_se[~reach], rtol=rtol)
