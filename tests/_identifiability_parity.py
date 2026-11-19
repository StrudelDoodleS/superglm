"""Structured and dense fits of one model read one identifiability decision.

``solvers.identifiability`` decides which coefficients a fit identifies; the structured
backends (``fs``, ``sz``, nested chains) read it from their retained factors and the dense
backend from its rank record.  On a full-rank penalised Hessian both report every standard
error, the same footnotes and the same values.  On a singular one both publish one gauge
(``identifiability.Gauge``): one coordinate per null direction at zero and blank, the rest with
the same coefficients and standard errors, and no fitted value moved.  Each null direction is
held on a construction column of its term (``identifiability.ConstructionColumns``): a held
unpenalised column is blank, a held part of a smooth blanks no coordinate.
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


def without_gauge(monkeypatch, fit):
    """``fit()`` with the identifiability gauge left unpublished: the coefficients ``633289397`` published."""
    from superglm.model import state_ops

    with monkeypatch.context() as patch:
        patch.setattr(state_ops, "publish_identified_fit", lambda model: None)
        return fit()


def assert_one_gauge(structured_metrics, dense_metrics, frame, *, rtol: float, before=None) -> None:
    """A singular Hessian: both backends publish the same gauge (``identifiability.Gauge``).

    The same blanks, footnotes and standard errors (``assert_same_identifiability``), the same
    constraints, each holding its functional of the coefficients at zero, and the same
    coefficients, normwise: in this gauge they reach ``7e3`` beside fitted values of order one,
    so a solve's error is the vector's, not each entry's.  The structured fit moved its
    coefficients (its record holds the gauge),
    and each backend predicts what the same fit without the gauge (``before``, from
    ``without_gauge``) predicts, to ``1e-12`` of the predictor's own magnitude ``|b_0| + |x|'
    |beta|`` in either gauge: the size of the rounding any evaluation of it carries.  Without
    ``before`` the caller checks the predictions.
    """
    assert_same_identifiability(structured_metrics, dense_metrics, rtol=rtol)
    structured, dense = structured_metrics._model, dense_metrics._model
    blank = ~np.asarray(structured._fit_inference_info["coefficient_identified"])
    beta, reference = structured.result.beta, dense.result.beta
    assert np.all(beta[blank] == 0.0) and np.all(reference[blank] == 0.0)
    assert np.max(np.abs(beta - reference)) <= rtol * np.max(np.abs(reference))
    gauges = [model._fit_inference_info["identifiability"].gauge for model in (structured, dense)]
    assert all(gauge is not None for gauge in gauges)
    functionals = [gauge.functionals for gauge in gauges]
    np.testing.assert_allclose(functionals[0], functionals[1], rtol=rtol, atol=rtol)
    for model, gauge in zip((structured, dense), gauges, strict=True):
        held = gauge.functionals.T @ model.result.beta
        scale = np.abs(gauge.functionals).T @ np.abs(model.result.beta)
        assert np.all(np.abs(held) <= 1e-12 * np.maximum(scale, 1.0))
    X = np.abs(np.asarray(structured._dm.toarray(), dtype=np.float64))
    for model, unmoved in zip((structured, dense), before or (), strict=bool(before)):
        magnitude = np.maximum(
            abs(float(model.result.intercept)) + X @ np.abs(model.result.beta),
            abs(float(unmoved.result.intercept)) + X @ np.abs(unmoved.result.beta),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, reference_mean = model.predict(frame), unmoved.predict(frame)
        slope = np.abs(model._link.deriv_inverse(model._link.link(reference_mean)))
        assert np.all(np.abs(mean - reference_mean) <= 1e-12 * slope * magnitude)
