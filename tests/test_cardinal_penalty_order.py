"""Cardinal CR splines implement only the squared second-derivative penalty."""

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.features.interaction import interaction_spline_spec
from superglm.features.spline import CardinalCRSpline


@pytest.mark.parametrize("m", [1, (1,)])
@pytest.mark.parametrize(
    "constructor", [CardinalCRSpline, lambda **kw: Spline("cr_cardinal", **kw)]
)
def test_cardinal_rejects_first_derivative_penalty(constructor, m):
    """A first-derivative request must not silently fit the second derivative."""
    with pytest.raises(ValueError, match="only supports.*m=2"):
        constructor(n_knots=6, m=m)


@pytest.mark.parametrize("order", [1, 3])
def test_cardinal_structural_penalty_rejects_another_order(order):
    spec = CardinalCRSpline(n_knots=6)
    spec._place_knots(np.linspace(0.0, 1.0, 200))

    with pytest.raises(ValueError, match="only supports.*m=2"):
        spec._structural_penalty_for_order(order)


@pytest.mark.parametrize(
    "constructor", [CardinalCRSpline, lambda **kw: Spline("cr_cardinal", **kw)]
)
@pytest.mark.parametrize("m", [2, (2,)])
def test_cardinal_second_derivative_fit_matches_default(constructor, m):
    rng = np.random.default_rng(921)
    x = np.linspace(0.0, 1.0, 200)
    frame = pd.DataFrame({"x": x})
    y = 0.3 + 2.0 * x + 0.1 * rng.normal(size=len(x))

    fits = []
    for spec in (CardinalCRSpline(n_knots=6), constructor(n_knots=6, m=m)):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=1e4,
            features={"x": spec},
        )
        model.fit(frame, y)
        fits.append(model)

    np.testing.assert_array_equal(fits[1].predict(frame), fits[0].predict(frame))
    np.testing.assert_array_equal(fits[1].result.beta, fits[0].result.beta)
    assert fits[1].result.effective_df == fits[0].result.effective_df


@pytest.mark.parametrize("m", [1, 3])
def test_cr_interactions_keep_supported_noncardinal_orders(m):
    x = np.linspace(0.0, 1.0, 200)
    spec = Spline(kind="cr", n_knots=6, m=m)
    spec.build(x)

    assert interaction_spline_spec(spec, x) is spec
    marginal = spec.tensor_marginal_ingredients(x)
    assert marginal.basis.shape[0] == len(x)
    assert np.all(np.isfinite(marginal.penalty))
