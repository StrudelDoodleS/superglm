"""Leverage of structured random-effect fits from the factor's pieces.

Leverage is the influence diagonal with the intercept, ``h_i = w_i a_i' H_aug^+
a_i`` with ``a_i = [1, x_i]`` (one-engine design §3.10, definition (A)).
``a' H^+ a`` splits into a random-effect part, a tree solve over
the row's reach, and a border part ``y' Q^+ y`` with ``y = a - F' b`` (Bates et
al. 2015, eqs. 63-65), so no ``K x K`` inverse block is formed.  The references
are dense Cholesky solves of ``X'WX + S`` assembled from the design, never from
the structured system, which is the same elimination in another symmetric order
(tree nodes first, the border last).  Each solve has backward error ``|dH| <=
gamma_{3p+1} |R'||R|`` with ``(|R'||R|)_ij <= sqrt(H_ii H_jj) / (1 - gamma_{p+1})``
(Higham 2002, Thm 10.4 and its proof), so under the Jacobi scaling ``H_s = D H
D`` the scaled perturbation has norm at most ``gamma_{3p+1} p / (1 -
gamma_{p+1})``, and two evaluations of ``x' H^-1 x`` agree within twice that
times ``||H_s^-1|| x' H^-1 x``.
"""

import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm import Numeric, RandomEffect, Spline, SuperGLM
from superglm.inference.covariance import _active_penalty_matrix
from superglm.model.fit_state import fitted_lambda2

EPS = np.finfo(np.float64).eps
CHAIN = ("make", "model", "variant")
# more tree nodes than the 256-coefficient inverse-block cap, which refused before
SIZES = {"single": (300,), "nested": (6, 40, 300)}


def _data(sizes: tuple[int, ...], n: int, seed: int = 20260927):
    """A strict chain (every leaf observed), a spline, a column whose mean exceeds
    its spread (the nested factor centres it), an offset and prior weights with
    two zero-weight leaves (the fit then centres on the positive-weight rows)."""
    rng = np.random.default_rng(seed)
    parents = [
        np.concatenate([np.arange(coarse), rng.integers(0, coarse, fine - coarse)])
        for coarse, fine in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    codes = [np.concatenate([np.arange(sizes[-1]), rng.integers(0, sizes[-1], n - sizes[-1])])]
    for parent in reversed(parents):
        codes.insert(0, parent[codes[0]])
    x = rng.uniform(size=n)
    year = 10.0 + rng.integers(0, 10, n)
    offset = np.log(rng.uniform(0.5, 2.0, n))
    eta = 0.3 * np.sin(6.0 * x) + 0.05 * (year - 15.0)
    eta = eta + sum(
        rng.normal(0.0, 0.25, size)[code] for size, code in zip(sizes, codes, strict=True)
    )
    y = rng.poisson(np.exp(eta + offset)).astype(float)
    labels = [codes[0].astype(str)]
    for code in codes[1:]:
        labels.append(np.char.add(np.char.add(labels[-1], ":"), code.astype(str)))
    chain = CHAIN[-len(sizes) :]
    X = pd.DataFrame({"x": x, "year": year, **dict(zip(chain, labels, strict=True))})
    weights = rng.uniform(0.5, 1.5, n)
    weights[np.isin(codes[-1], [3, 7])] = 0.0
    return X, y, weights, offset


def _fit(kind: str, *, n: int = 3000, discrete: bool = False, retain: bool = True):
    sizes = SIZES[kind] if kind in SIZES else (int(kind),)
    X, y, weights, offset = _data(sizes, n)
    chain = CHAIN[-len(sizes) :]
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        direct_solve="structured",
        discrete=discrete,
        retain_fit_state=retain,
        features={"x": Spline(n_knots=6), "year": Numeric(), **{c: RandomEffect() for c in chain}},
    ).fit_reml(X, y, sample_weight=weights, offset=offset)
    assert model._reml_profile["structured_chain"] == chain
    return model, X, y, weights, offset


def _dense_hessian(model, W):
    """``X'WX + S`` from the fit design and the fitted penalties."""
    X = model._dm.toarray()
    S = _active_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._groups,
        fitted_lambda2(model),
        reml_penalties=model._reml_penalties,
    )
    H = X.T @ (W[:, None] * X) + S
    return X, 0.5 * (H + H.T)


def _cholesky_forms(H, rows):
    """``x_i' H^-1 x_i`` by a dense Cholesky solve, and the gap bound to any other."""
    p = H.shape[0]
    forms = np.sum(rows * scipy.linalg.cho_solve(scipy.linalg.cho_factor(H), rows.T).T, axis=1)
    scale = 1.0 / np.sqrt(np.diag(H))
    smallest = np.linalg.eigvalsh(scale[:, None] * H * scale[None, :])[0]
    gamma = (3 * p + 1) * EPS / (1.0 - (3 * p + 1) * EPS)
    growth = 1.0 - (p + 1) * EPS / (1.0 - (p + 1) * EPS)
    return forms, 2.0 * gamma * p / growth / smallest * np.abs(forms)


def _augmented(X, H, W):
    """``[1, X]' W [1, X] + diag(0, S)`` from the rows ``X`` and ``H = X'WX + S``."""
    augmented = np.empty((H.shape[0] + 1, H.shape[0] + 1))
    augmented[0, 0] = np.sum(W)
    augmented[0, 1:] = augmented[1:, 0] = X.T @ W
    augmented[1:, 1:] = H
    return augmented


@pytest.mark.parametrize("kind", ["single", "nested"])
def test_row_forms_match_a_dense_cholesky_solve(kind):
    """The intercept-augmented factor's row forms (its nested form applies ``R'``
    to a centred border) on the fit rows ``[1, x]``, rows whose levels leave any
    root-to-leaf path, and a zero row, against dense Cholesky solves of the
    assembled augmented system.  No raw-coordinate coefficient factor exists
    (design §3.6); leverage reads these forms."""
    model, X, y, weights, offset = _fit(kind)
    state = model._linear_system_state
    W = model.metrics(X, y, sample_weight=weights, offset=offset)._active_info[1]
    X, H = _dense_hessian(model, W)
    rng = np.random.default_rng(5)
    scattered = rng.normal(size=(40, X.shape[1])) * (rng.uniform(size=(40, X.shape[1])) < 0.05)
    rows = np.vstack((X, scattered, np.zeros((1, X.shape[1]))))
    ones = np.vstack((np.hstack((np.ones((len(rows), 1)), rows)), np.zeros((1, X.shape[1] + 1))))
    assert not hasattr(state, "coefficient_factor")
    if kind == "nested":
        assert np.any(state.augmented_factor._center != 0.0)
    actual = state.augmented_factor.row_quadratic_forms(ones)
    expected, bound = _cholesky_forms(_augmented(X, H, W), ones)
    np.testing.assert_array_less(np.abs(actual - expected), bound + np.finfo(float).tiny)
    assert actual[-1] == 0.0


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("kind", ["single", "nested"])
def test_leverage_is_the_hat_diagonal_on_live_copied_and_pickled_models(kind, discrete):
    """More tree nodes than the inverse-block cap: leverage returns, zero-weight rows
    have zero leverage, and the live, equal-copy and pickled routes all give the
    dense influence diagonal ``w_i a_i' H_aug^-1 a_i`` of the fit design."""
    model, X, y, weights, offset = _fit(kind, discrete=discrete)
    live = model.metrics(X, y, sample_weight=weights, offset=offset)
    W = live._active_info[1]
    design, H = _dense_hessian(model, W)
    ones = np.hstack((np.ones((len(design), 1)), design))
    expected, bound = _cholesky_forms(_augmented(design, H, W), ones)
    clone = pickle.loads(pickle.dumps(model))
    routes = {
        "live": live,
        "copy": model.metrics(X.copy(), y, sample_weight=weights, offset=offset),
        "pickled": clone.metrics(X, y, sample_weight=weights, offset=offset),
    }
    for name, metrics in routes.items():
        assert metrics._uses_compact_fit_inference, name
        leverage = metrics.leverage
        assert np.all(leverage[weights == 0.0] == 0.0), name
        np.testing.assert_array_less(
            np.abs(leverage - np.clip(W * expected, 0.0, 1.0)),
            W * bound + np.finfo(float).tiny,
            err_msg=name,
        )


@pytest.mark.parametrize("kind", ["single", "nested"])
def test_a_released_model_evaluates_the_fit_rows(kind):
    """``retain_fit_state=False`` rebuilds the rows from the public spline transform
    ``B R'``, which the fit canonicalized as ``R' = R - 1 m'`` with its training
    column means ``m``.  With zero prior weights ``m`` is not the fit's centring,
    so leverage must read the fit's rows ``B (R' + 1 m')`` against the retained
    augmented ``[1, X]'W[1, X] + diag(0, S)``.  Both come from one fit: exactly they differ by ``(sum_l B_l -
    1) m``, and in floating point by the roundings of two ``k``-term products of
    a nonnegative ``B``, a sum and an addition, within ``3 gamma_{k+1} B (|R'| +
    |m|)`` (the inner-product bound ``gamma_k |x|' |y|``, Higham 2002, ch. 3) and
    ``eps`` of each entry."""
    model, X, y, weights, offset = _fit(kind, retain=False)
    metrics = model.metrics(X, y, sample_weight=weights, offset=offset)
    assert metrics._uses_compact_fit_inference
    design, W = metrics._active_info[:2]
    rows = design.toarray()
    spline = next(group.sl for group in model._groups if group.name == "x")
    shift = model._fit_inference_info["XtWX_inv_aug"].intercept_shift
    # the spline is the one canonicalized term, so only its columns move
    assert np.any(shift[spline] != 0.0) and not np.any(shift[spline.stop :])
    spec = model._specs["x"]
    basis = spec._basis_matrix(X["x"].to_numpy()).toarray()
    public, means = spec._R_inv, shift[spline]
    gamma = (basis.shape[1] + 1) * EPS / (1.0 - (basis.shape[1] + 1) * EPS)
    bound = (
        3.0 * gamma * (basis @ (np.abs(public) + np.abs(means)))
        + np.abs(basis.sum(axis=1) - 1.0)[:, None] * np.abs(means)
        + EPS * np.abs(rows[:, spline])
    )
    np.testing.assert_array_less(np.abs(rows[:, spline] - basis @ (public + means)), bound)
    operator = model._linear_system_state.penalized_operator
    H = operator.matvec(np.eye(operator.shape[0]))
    augmented = _augmented(rows, 0.5 * (H + H.T), W)
    expected, bound = _cholesky_forms(augmented, np.hstack((np.ones((len(rows), 1)), rows)))
    np.testing.assert_array_less(
        np.abs(metrics.leverage - np.clip(W * expected, 0.0, 1.0)),
        W * bound + np.finfo(float).tiny,
    )


def test_leverage_at_scale_forms_no_coefficient_block(monkeypatch):
    """The review's refusal (60k rows, 3440 levels) returns without a selected
    inverse block or a dense covariance, and its traced peak stays below one
    ``K x K`` float64 block; the small fixtures above check the values."""
    model, X, y, weights, offset = _fit("3440", n=60_000)
    factor = model._linear_system_state.augmented_factor
    metrics = model.metrics(X, y, sample_weight=weights, offset=offset)
    covariance = metrics._active_info[2]

    def refuse(*_args, **_kwargs):
        pytest.fail("leverage formed a selected inverse block or a dense covariance")

    monkeypatch.setattr(type(factor), "selected_inverse_block", refuse)
    monkeypatch.setattr(type(covariance), "__array__", refuse)
    tracemalloc.start()
    leverage = metrics.leverage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    K = len(factor.structured_indices)
    assert K == 3440
    assert peak < K * K * 8
    assert np.all(leverage[weights == 0.0] == 0.0)
    assert np.all((leverage > 0.0) == (weights > 0.0))
