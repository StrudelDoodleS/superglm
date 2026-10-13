"""A spline-by-factor level whose far rows all have response 0.

On the freMTPL2 screening-guide model, brand B12 has no claims past vehicle age
22.  Its VehAge curve took a smoothing parameter near 1e-3 from a sharp feature
at age 0, and past 22 nothing in the data stopped it: the curve fell until the
predictor hit the log link's floor, pricing real policies at ``exp(-80)``, and
the runtime warning named another term.  The build now finds the separated
range (``diagnostics.separation.separated_curve_support``) and holds the curve
at its value at the level's last nonzero response, in the fit and in
``predict()``, naming the term, the level and the range.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SeparationError, SeparationWarning, Spline, SuperGLM

U = np.finfo(np.float64).eps / 2


def separated_tail_design(seed=0, n=12000, n_tail=12):
    """Level C has a spike at x = 0 and, past its last x = 22, a few rows of response 0.

    Levels A and B reach the far x values with positive responses, so only
    C's curve is separated there.
    """
    rng = np.random.default_rng(seed)
    x = np.minimum(rng.geometric(0.15, size=n) - 1, 22).astype(float)
    f = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    w = rng.uniform(0.3, 1.0, size=n)
    eta = -2.0 + 0.03 * x - 0.002 * x**2 + np.where(f == "B", 0.2, 0.0)
    eta = eta + np.where((f == "C") & (x == 0), 1.5, 0.0)
    counts = rng.poisson(w * np.exp(eta)).astype(float)
    x_far = rng.uniform(23, 50, size=60).round()
    w_far = rng.uniform(0.3, 1.0, size=60)
    counts_far = np.maximum(rng.poisson(w_far * 0.4), 1).astype(float)
    x_tail = rng.uniform(30, 45, size=n_tail).round()
    w_tail = rng.uniform(0.3, 1.0, size=n_tail)
    X = pd.DataFrame(
        {
            "x": np.concatenate([x, x_far, x_tail]),
            "f": pd.Categorical(
                np.concatenate([f, rng.choice(["A", "B"], size=60), ["C"] * n_tail])
            ),
        }
    )
    w = np.concatenate([w, w_far, w_tail])
    counts = np.concatenate([counts, counts_far, np.zeros(n_tail)])
    return X, counts / w, w


def fit(X, y, w, **kwargs):
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=12), "f": Categorical()},
        interactions=[("x", "f")],
        **kwargs,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(X, y, sample_weight=w)
    curve = [
        str(c.message)
        for c in caught
        if issubclass(c.category, SeparationWarning) and "spline-by-factor" in str(c.message)
    ]
    return model, curve


def level_minus_base(model, level, x):
    """The level's log relativity to base level A at each ``x``, read through ``predict``."""
    cats = ["A", "B", "C"]
    rows = pd.DataFrame({"x": x, "f": pd.Categorical([level] * len(x), categories=cats)})
    base = pd.DataFrame({"x": x, "f": pd.Categorical(["A"] * len(x), categories=cats)})
    return np.log(model.predict(rows)) - np.log(model.predict(base))


@pytest.mark.parametrize(
    "discrete, n_bins", [(True, 256), (True, 16), (False, 256)], ids=["d256", "d16", "exact"]
)
def test_separated_tail_is_held_at_the_last_nonzero_response(discrete, n_bins):
    X, y, w = separated_tail_design()
    model, curve = fit(X, y, w, discrete=discrete, n_bins=n_bins)

    assert len(curve) == 1
    line = next(row for row in curve[0].splitlines() if "level 'C'" in row)
    assert "'x:f'" in line
    assert "every response 0" in line
    assert "held at its value at x 22" in line
    tail_x = X["x"].to_numpy()[-12:]
    assert f"x {tail_x.min():g} to {tail_x.max():g} (12 rows" in line

    # Past x = 22 level C's curve is its value at 22, exactly: the held rows
    # read the same basis row.  Each side is a sum of a few O(1) terms, so the
    # two agree to a few dozen roundings of numbers below 16.
    far = np.array([25.0, 30.0, 38.0, 45.0, 50.0])
    held = level_minus_base(model, "C", far)
    edge = level_minus_base(model, "C", np.array([22.0]))
    np.testing.assert_allclose(held, np.full_like(far, edge[0]), rtol=0, atol=64 * U * 16)


def test_ignore_leaves_the_curve_free():
    X, y, w = separated_tail_design()
    model, curve = fit(X, y, w, discrete=True, separation="ignore")
    assert curve == []
    held = level_minus_base(model, "C", np.array([45.0]))
    edge = level_minus_base(model, "C", np.array([22.0]))
    # The free curve keeps walking past its data, which is the behaviour the
    # hold replaces: it ends well below its value at 22.
    assert held[0] < edge[0] - 1.0


def test_error_mode_refuses_the_separated_curve():
    X, y, w = separated_tail_design()
    with pytest.raises(SeparationError, match=r"'x:f', level 'C'"):
        fit(X, y, w, discrete=True, separation="error")


def test_a_fit_without_separation_is_unchanged():
    """Every level has a positive response at every x it reaches: nothing is held."""
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.integers(0, 15, size=n).astype(float)
    f = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    w = rng.uniform(0.5, 1.0, size=n)
    counts = rng.poisson(w * np.exp(0.4 + 0.03 * x)).astype(float)
    X = pd.DataFrame({"x": x, "f": pd.Categorical(f)})
    occupied = pd.Series(counts).groupby([X["x"], X["f"]], observed=True).transform("max")
    assert occupied.min() > 0, "fixture must anchor every (level, x) cell"

    scanned, curve = fit(X, counts / w, w, discrete=True)
    plain, _ = fit(X, counts / w, w, discrete=True, separation="ignore")
    assert curve == []
    np.testing.assert_array_equal(scanned.result.beta, plain.result.beta)
    np.testing.assert_array_equal(scanned.predict(X), plain.predict(X))
