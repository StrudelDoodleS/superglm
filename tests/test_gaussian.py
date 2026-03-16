import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from superglm import Gaussian, IdentityLink, Numeric, SuperGLM
from superglm.distributions import resolve_distribution
from superglm.links import resolve_link


def test_resolve_distribution_gaussian():
    family = resolve_distribution("gaussian")
    assert isinstance(family, Gaussian)


def test_gaussian_default_link_is_identity():
    link = resolve_link(None, Gaussian())
    assert isinstance(link, IdentityLink)


def test_gaussian_unpenalized_fit_matches_ols_and_allows_negative_predictions():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = -2.5 + 1.2 * x1 - 0.7 * x2 + rng.normal(scale=0.2, size=n)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = SuperGLM(
        family="gaussian",
        lambda1=0.0,
        lambda2=0.0,
        features={
            "x1": Numeric(standardize=False),
            "x2": Numeric(standardize=False),
        },
    )
    model.fit(X, y)

    X_ols = np.column_stack([np.ones(n), x1, x2])
    coef_ols, *_ = np.linalg.lstsq(X_ols, y, rcond=None)

    assert_allclose(model.result.intercept, coef_ols[0], atol=1e-6)
    assert_allclose(model.result.beta, coef_ols[1:], atol=1e-6)

    preds = model.predict(X)
    assert np.any(preds < 0.0)
    assert_allclose(preds, X_ols @ coef_ols, atol=1e-6)
