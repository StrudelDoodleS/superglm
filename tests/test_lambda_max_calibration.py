"""compute_lambda_max must agree with the solver's own zeroing threshold.

The block-coordinate solver zeroes a group when ``||grad_g|| <= lambda1 * w_g``
(``pirls.py`` radial threshold and ``GroupLasso.prox_group``), where ``grad_g``
is the unnormalised score ``-X_g' W r``.  ``compute_lambda_max`` must therefore
return the smallest ``lambda1`` that zeroes every penalised group on the same
scale -- no row-count normalisation, and including the family's score factor for
non-canonical links.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import Spline
from superglm.model.base import compute_lambda_max


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )


def _model(family):
    return SuperGLM(
        family=family,
        selection_penalty=0.0,
        penalty="group_lasso",
        features={name: Spline(kind="ps", k=6) for name in ("a", "b", "c")},
    )


def _active_group_count(family, frame, response, weights, lambda1):
    fitted = SuperGLM(
        family=family,
        selection_penalty=lambda1,
        penalty="group_lasso",
        features={name: Spline(kind="ps", k=6) for name in ("a", "b", "c")},
    ).fit(frame, response, sample_weight=weights)
    beta = fitted.result.beta
    return sum(1 for g in fitted._groups if float(np.linalg.norm(beta[g.sl])) > 1e-8)


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_lambda_max_is_the_solver_zeroing_boundary(family):
    """Just above lambda_max everything zeroes; just below, something survives."""
    frame = _frame()
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.5, 1.0, len(frame))
    signal = 0.6 * frame["a"].to_numpy() + 0.4 * frame["b"].to_numpy()
    if family == "poisson":
        response = rng.poisson(np.exp(-1.0 + signal)) / weights
    else:
        response = rng.gamma(shape=2.0, scale=np.exp(-1.0 + signal) / 2.0)

    model = _model(family)
    model._build_design_matrix(frame, response, weights, None)
    lambda_max = compute_lambda_max(model, np.asarray(response, dtype=float), weights)
    assert lambda_max > 0.0

    above = _active_group_count(family, frame, response, weights, lambda_max * 1.05)
    below = _active_group_count(family, frame, response, weights, lambda_max * 0.90)

    assert above == 0, (
        f"{family}: lambda_max={lambda_max:.6g} did not zero every group; "
        f"{above} still active at 1.05x"
    )
    assert below > 0, (
        f"{family}: lambda_max={lambda_max:.6g} is too large; nothing survives at 0.90x"
    )


def test_lambda_max_does_not_scale_with_row_count():
    """A row-count factor would make lambda_max drift with n on identical data."""
    small = _frame(n=2000, seed=3)
    large = pd.concat([small] * 4, ignore_index=True)

    values = []
    for frame in (small, large):
        rng = np.random.default_rng(4)
        weights = np.full(len(frame), 0.75)
        response = rng.poisson(np.exp(-1.0 + 0.5 * frame["a"].to_numpy())) / weights
        model = _model("poisson")
        model._build_design_matrix(frame, response, weights, None)
        values.append(compute_lambda_max(model, np.asarray(response, float), weights))

    # Replicating every row 4x scales the score by 4x, so lambda_max scales by 4x.
    # A stray 1/n would instead leave it unchanged.
    assert values[1] / values[0] == pytest.approx(4.0, rel=0.25), (
        f"lambda_max did not scale with total score mass: {values}"
    )


def test_sparse_group_lasso_alpha_zero_keeps_group_threshold():
    """Review finding: alpha=0 sparse-group lasso is pure group lasso, but
    compute_lambda_max returned 0.0 — silently disabling selection_penalty
    "auto" and collapsing fit_path to an all-zero lambda sequence."""
    frame = _frame(n=2000, seed=7)
    rng = np.random.default_rng(8)
    weights = rng.uniform(0.5, 1.0, len(frame))
    response = rng.poisson(np.exp(-1.0 + 0.6 * frame["a"].to_numpy())) / weights

    from superglm.penalties.sparse_group_lasso import SparseGroupLasso

    def build(penalty, **kwargs):
        model = SuperGLM(
            family="poisson",
            penalty=penalty,
            features={name: Spline(kind="ps", k=6) for name in ("a", "b", "c")},
            **kwargs,
        )
        model._build_design_matrix(frame, response, weights, None)
        return model

    sgl = build(SparseGroupLasso(lambda1=0.0, alpha=0.0))
    gl = build("group_lasso", selection_penalty=0.0)

    sgl_lmax = compute_lambda_max(sgl, np.asarray(response, dtype=float), weights)
    gl_lmax = compute_lambda_max(gl, np.asarray(response, dtype=float), weights)

    assert sgl_lmax > 0.0
    np.testing.assert_allclose(sgl_lmax, gl_lmax, rtol=1e-12)
