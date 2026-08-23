"""max_iter must be validated, not crash with UnboundLocalError (audit S13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM


def _tiny_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.3 * df["x0"].to_numpy())).astype(float)
    return df, y


@pytest.mark.parametrize("bad_max_iter", [0, -1])
def test_fit_rejects_non_positive_max_iter(bad_max_iter: int) -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", max_iter=bad_max_iter)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit(df, y)


def test_fit_reml_rejects_zero_max_iter() -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit_reml(df, y)


def test_selection_path_rejects_zero_max_iter() -> None:
    df, y = _tiny_frame()
    model = SuperGLM(family="poisson", selection_penalty=0.1)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        model.fit(df, y, max_iter=0)


def test_fit_pirls_rejects_non_positive_inner_and_outer() -> None:
    from superglm.solvers import fit_pirls

    # Arguments are never reached: validation runs before any array handling.
    with pytest.raises(ValueError, match="max_iter_outer must be at least 1"):
        fit_pirls(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            penalty=None,
            max_iter_outer=0,
            weight_semantics="frequency",
        )
    with pytest.raises(ValueError, match="max_iter_inner must be at least 1"):
        fit_pirls(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            penalty=None,
            max_iter_inner=0,
            weight_semantics="frequency",
        )


def test_fit_irls_direct_rejects_zero_max_iter() -> None:
    from superglm.solvers import fit_irls_direct

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        fit_irls_direct(
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            weights=np.ones(2),
            family=None,
            link=None,
            groups=[],
            lambda2=0.0,
            max_iter=0,
            weight_semantics="frequency",
        )


def test_max_iter_one_remains_legal() -> None:
    """The discrete POI loop depends on max_iter=1 (reml/discrete.py:551-578)."""
    df, y = _tiny_frame()
    # Explicit features are load-bearing: with auto-detection the design is
    # intercept-only, beta has shape (0,), and asserting finiteness over it is
    # vacuously true.
    model = SuperGLM(
        family="poisson",
        max_iter=1,
        features={"x0": Numeric(), "x1": Numeric()},
    )
    model.fit(df, y)
    assert model.result.beta.size == 2
    assert np.all(np.isfinite(model.result.beta))
    # The cap genuinely binds here: one step, stopped short of convergence,
    # which is how the discrete POI loop runs PIRLS.
    assert model.result.n_iter == 1
    assert not model.result.converged
