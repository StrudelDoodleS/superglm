"""A spline-by-factor level whose far rows all have response 0.

On the freMTPL2 screening-guide model, brand B12 has no claims past vehicle age
22.  On the book's uncorrected claim counts its VehAge curve took a smoothing
parameter near 1e-3, driven by a spike at age 0 that the corrupted block of
counts (``IDpol <= 24500``, see ``examples/fremtpl2_data.py``) puts there; on
the corrected counts there is no spike and the parameter is near 7.  Either
way, past 22 nothing in the data stops the curve: on the uncorrected counts it
fell until the predictor hit the log link's floor, pricing real policies at
``exp(-80)``, and the runtime warning named another term.  The build now finds the separated
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


def fit(X, y, w, base="most_exposed", unseen="error", **kwargs):
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=12), "f": Categorical(base=base, unseen=unseen)},
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


def test_level_log_deviations_hold_a_separated_level_as_score_does():
    """The tabulated curve of a separated level is the held one, not the free spline.

    ``level_log_deviations`` feeds the rating-table export, which would
    otherwise ship cells past x = 22 that follow the free curve while
    ``predict`` holds it.
    """
    X, y, w = separated_tail_design()
    model, _ = fit(X, y, w, discrete=True)
    ispec = model._interaction_specs["x:f"]
    assert "C" in ispec._level_clip
    beta = np.concatenate(
        [model.result.beta[g.sl] for g in model._groups if g.feature_name == "x:f"]
    )

    x = np.linspace(0.0, 50.0, 101)
    deviations = ispec.level_log_deviations(x, beta)
    for j, level in enumerate(ispec._non_base):
        np.testing.assert_allclose(
            deviations[:, j],
            ispec.score(x, np.array([level] * len(x), dtype=object), beta),
            rtol=0,
            atol=64 * U * 16,
        )
    # The design rows that carry the covariance to the curve hold the level the
    # same way: row @ coefficients is the tabulated column.
    offset = 0
    for j, (level, rows) in enumerate(ispec.level_design_rows(x)):
        np.testing.assert_allclose(
            rows @ beta[offset : offset + rows.shape[1]],
            deviations[:, j],
            rtol=0,
            atol=64 * U * 16,
        )
        offset += rows.shape[1]
    assert offset == len(beta)
    # The held range is where a free curve would differ.
    j = ispec._non_base.index("C")
    free = ispec._spline_spec._raw_basis_matrix(x[x > 22.0])
    raw = dict(ispec._level_raw_coefficients(beta))["C"]
    assert np.abs(deviations[x > 22.0, j] - free @ raw).max() > 1e-6


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


def interior_gap_design():
    """Level C has responses on [0, 10] and [30, 40] and only zeros between."""
    rng = np.random.default_rng(5)
    x = rng.integers(0, 41, size=9000).astype(float)
    f = rng.choice(["A", "B"], size=9000, p=[0.6, 0.4])
    x_c = np.concatenate([rng.integers(0, 11, size=1500), rng.integers(30, 41, size=1500)])
    x_gap = np.repeat(np.arange(14.0, 27.0, 2.0), 3)
    X = pd.DataFrame(
        {
            "x": np.concatenate([x, x_c.astype(float), x_gap]),
            "f": pd.Categorical(np.concatenate([f, ["C"] * (len(x_c) + len(x_gap))])),
        }
    )
    w = rng.uniform(0.5, 1.0, size=len(X))
    counts = rng.poisson(w * np.exp(0.3 + 0.01 * X["x"].to_numpy())).astype(float)
    counts[-len(x_gap) :] = 0.0
    return X, counts / w, w


def test_a_separated_range_between_nonzero_responses_is_interpolated():
    """Over the gap the fitted curve is the minimum-penalty join of its two sides.

    The directions only the gap's values move, ``free``, are held where the
    penalty alone puts them: ``free' Omega f = 0`` for level C's fitted raw
    curve ``f``.  The free fit (``separation="ignore"``) is not there.
    """
    from superglm.diagnostics.separation import separated_curve_support

    X, y, w = interior_gap_design()
    model, curve = fit(X, y, w, discrete=True)

    assert len(curve) == 1
    line = next(row for row in curve[0].splitlines() if "level 'C'" in row)
    assert "x 14 to 26 (21 rows" in line
    assert "held at the smooth interpolation of the curve on either side" in line

    level = (X["f"] == "C").to_numpy()
    values, idx = np.unique(X["x"].to_numpy()[level], return_inverse=True)
    lower = np.bincount(idx, weights=(y[level] <= 0.0), minlength=len(values)) > 0
    inside = np.bincount(idx, weights=(y[level] > 0.0), minlength=len(values)) > 0

    def penalty_gradient(fitted):
        spec = fitted._interaction_specs["x:f"]
        raw = np.asarray(spec._spline_spec._raw_basis_matrix(values), dtype=np.float64)
        separated, free = separated_curve_support(
            raw, np.eye(raw.shape[1]), lower & ~inside, np.zeros_like(inside), inside
        )
        np.testing.assert_array_equal(values[separated], np.arange(14.0, 27.0, 2.0))
        group = next(g for g in fitted._groups if g.name == "x:f[C]")
        r_inv = spec._R_inv_dict["C"]
        coefficients = fitted.result.beta[group.start : group.end]
        omega = np.asarray(spec._spline_spec._build_penalty(), dtype=np.float64)
        gradient = free.T @ omega @ (r_inv @ coefficients)
        # Rounding of the products that form it: a few dozen per entry, each
        # relative to the norms involved.
        bound = 64 * omega.shape[0] * U * np.linalg.norm(omega, 2)
        bound *= np.linalg.norm(r_inv, 2) * np.linalg.norm(coefficients)
        return np.max(np.abs(gradient)), bound

    held, bound = penalty_gradient(model)
    assert held <= bound
    free_fit, _ = fit(X, y, w, discrete=True, separation="ignore")
    loose, loose_bound = penalty_gradient(free_fit)
    assert loose > 1e6 * loose_bound, "the free fit is not at the minimum-penalty join"


def test_a_found_separation_is_reported_when_nothing_can_hold_it(monkeypatch):
    """When the held directions would fill the block, the range is still named."""
    import superglm.diagnostics.separation as separation

    monkeypatch.setattr(separation, "continuation_basis", lambda *args, **kwargs: None)
    X, y, w = interior_gap_design()
    model, curve = fit(X, y, w, discrete=True)
    assert len(curve) == 1
    line = next(row for row in curve[0].splitlines() if "level 'C'" in row)
    assert "x 14 to 26 (21 rows" in line
    assert "left to the smoothing penalty" in line


def test_a_separated_base_level_is_named():
    """The base level's curve is the main effect's, so it had no group and went unscanned.

    With level C as the factor's base its tail is now named, as the factor's
    base, and left to the smoothing penalty.
    """
    X, y, w = separated_tail_design()
    _, curve = fit(X, y, w, base="C", discrete=True)

    assert len(curve) == 1
    line = next(row for row in curve[0].splitlines() if "level 'C'" in row)
    assert "the factor's base" in line
    tail_x = X["x"].to_numpy()[-12:]
    assert f"x {tail_x.min():g} to {tail_x.max():g} (12 rows" in line
    assert "left to the smoothing penalty" in line


def main_and_curve_bases_disagree():
    """The base level A has responses off zero below x = 45 and none from 45 to 50.

    Most rows sit in [40, 50], so the interaction's quantile knots crowd
    there (one at 45.2) while the main effect's uniform knots leave a single
    segment from 42.9 to 50.  In the interaction's basis a curve can fall past
    45.2 and stay zero below it; the main effect's last segment is a cubic
    pinned by A's responses on [42.9, 45], so A's curve -- the main effect's
    -- cannot.
    """
    rng = np.random.default_rng(0)
    n = 20000
    dense = rng.uniform(size=n) < 0.6
    x = np.where(dense, rng.uniform(40.0, 50.0, size=n), rng.uniform(0.0, 40.0, size=n))
    f = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    w = rng.uniform(0.5, 1.0, size=n)
    counts = rng.poisson(w * 4.0).astype(float)
    counts[(f == "A") & (x >= 45.0)] = 0.0
    return pd.DataFrame({"x": x, "f": pd.Categorical(f)}), counts / w, w


def test_the_base_level_is_scanned_in_the_main_effects_basis():
    """A range the interaction's basis could separate but the main effect's cannot is not named.

    Scanned in the interaction's basis, A's range from 45.3 to 50 was named,
    and ``separation="error"`` refused the design though nothing could fall.
    """
    X, y, w = main_and_curve_bases_disagree()

    def model(separation):
        return SuperGLM(
            family="poisson",
            features={"x": Spline(n_knots=6), "f": Categorical(base="A")},
            interactions=[("x", "f")],
            discrete=True,
            separation=separation,
        )

    warned = model("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warned.fit_reml(X, y, sample_weight=w)
    main_knots, curve_knots = warned._specs["x"]._knots, warned._interaction_specs["x:f"]._knots
    assert main_knots[-5] < 43.0 and 45.0 < curve_knots[-3] < 45.5
    named = [
        line
        for c in caught
        if issubclass(c.category, SeparationWarning)
        for line in str(c.message).splitlines()
        if "level 'A'" in line
    ]
    assert named == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model("error").fit_reml(X, y, sample_weight=w)


def test_a_base_level_finding_does_not_refuse_the_design():
    """The base's range is found on the main effect alone, so it is reported, never refused."""
    X, y, w = separated_tail_design()
    _, curve = fit(X, y, w, base="C", discrete=True, separation="error")
    assert len(curve) == 1
    line = next(row for row in curve[0].splitlines() if "level 'C'" in row)
    assert "the factor's base" in line and "approximate" in line
    assert curve[0].splitlines()[-1].endswith("which it names and does not refuse.")


def test_continuation_basis_holds_the_free_directions_at_minimum_penalty():
    from superglm.diagnostics.separation import continuation_basis

    rng = np.random.default_rng(0)
    k_raw = 9
    difference = np.diff(np.eye(k_raw), 2, axis=0)
    omega = difference.T @ difference  # null space: constants and lines
    projection = np.linalg.svd(rng.uniform(0.5, 1.5, (1, k_raw)))[2][1:].T  # centring
    omega_null = np.linalg.eigh(omega)[1][:, :2]
    projected_null = np.linalg.eigh(projection.T @ omega @ projection)[1][:, :1]
    bound = 64 * k_raw * U * np.linalg.norm(omega, 2)

    free = np.linalg.qr(rng.normal(size=(k_raw, 3)))[0]
    basis = continuation_basis(free, omega, omega_null, projection, projected_null)
    assert basis.shape == (k_raw - 1, k_raw - 1 - 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=64 * k_raw * U)
    np.testing.assert_allclose(free.T @ omega @ projection @ basis, 0.0, atol=bound)

    # A free straight line has no continuation: the block's own line is held at zero.
    line = np.arange(k_raw, dtype=np.float64) - (k_raw - 1) / 2
    free = np.linalg.qr(np.column_stack([line, rng.normal(size=k_raw)]))[0]
    basis = continuation_basis(free, omega, omega_null, projection, projected_null)
    assert basis.shape == (k_raw - 1, k_raw - 1 - 2)
    np.testing.assert_allclose(projected_null.T @ basis, 0.0, atol=64 * k_raw * U)


def test_a_range_the_free_directions_see_faintly_is_found_as_one_program_per_value_finds_it():
    """The freMTPL2 guide's VehAge x Region level R73 at 256 bins, rebuilt from its knots.

    Its responses are 0 at 16, 21 to 29, 31 and 36 to 42, nonzero elsewhere up
    to 30.  The curve directions the nonzero values leave free see x = 23 at
    1.8e-7 of its row's length, and a linear program with unbounded
    coefficients failed there (HiGHS status 4) and found nothing.  The scan
    must agree with one program per value (Geyer 2009, section 3.12), each
    with bounded coefficients: 36 to 42 separate, and the values the nonzero
    responses pin do not.
    """
    from scipy.optimize import linprog

    from superglm.diagnostics.separation import _unseen_directions, separated_curve_support
    from superglm.features.interaction import _varying_coefficient_spline_spec

    interior = [5.692, 11.385, 17.077, 22.769, 28.462, 34.154, 39.846, 45.538, 51.231]
    interior += [61.846, 68.615, 80.308]
    spec = _varying_coefficient_spline_spec(
        Spline(knots=interior, boundary=(0.0, 100.0)), np.arange(0.0, 101.0)
    )
    values = np.array([*range(32), *range(36, 43)], dtype=np.float64)
    raw = np.asarray(spec._raw_basis_matrix(values), dtype=np.float64)
    inside = np.isin(values, [*range(16), 17, 18, 19, 20, 30])
    found = separated_curve_support(
        raw, np.eye(raw.shape[1]), ~inside, np.zeros_like(inside), inside
    )
    assert found is not None
    np.testing.assert_array_equal(values[found[0]], np.arange(36.0, 43.0))

    free = _unseen_directions(raw[inside])
    edge = -(raw[~inside] @ free)
    alone = []
    for j in range(len(edge)):
        others = np.delete(edge, j, axis=0)
        program = linprog(
            -edge[j], A_ub=-others, b_ub=np.zeros(len(others)), bounds=[(-1, 1)] * free.shape[1]
        )
        if program.status == 0 and -program.fun > 1e-9 * np.linalg.norm(raw[~inside][j]):
            alone.append(values[~inside][j])
    assert alone == list(np.arange(36.0, 43.0))


def test_unseen_directions_does_not_form_the_left_factor_of_a_tall_matrix():
    """On the exact path the rows are a level's distinct covariate values, tens of thousands.

    A full SVD builds the M x M left factor: 288 MB here, 20 GB at 50,000 rows,
    for directions only the right factor holds.
    """
    import tracemalloc

    from superglm.diagnostics.separation import _unseen_directions

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(6000, 12))
    rows[:, -2:] = rows[:, :2]  # two directions no row sees
    tracemalloc.start()
    try:
        free = _unseen_directions(rows)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert free.shape == (12, 2)
    np.testing.assert_allclose(rows @ free, 0.0, atol=64 * 12 * U * np.linalg.norm(rows, 2))
    # The left factor alone would be 500 times the input.
    assert peak < 64 * rows.nbytes


def _signed_rows(kind, rng):
    """``D = sign * X`` for a design that overlaps, separates completely or quasi-completely,
    or overlaps by a margin of ``1e-3`` (near separation)."""
    n, p = 200, 4
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    b = rng.normal(size=p)
    eta = X @ b
    if kind == "overlap":
        y = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))
    elif kind == "complete":
        y = eta > 0.0
    elif kind == "quasi":
        # 20 rows on the hyperplane x'b = 0 with both responses: they hold it, the rest move.
        X[:20, -1] = -(X[:20, :-1] @ b[:-1]) / b[-1]
        y = X @ b > 0.0
        y[:20] = np.arange(20) % 2 == 0
    else:  # near: separated at 0 except one row of each response crossing by 1e-3
        x = np.r_[rng.uniform(0.1, 1.0, n // 2), -rng.uniform(0.1, 1.0, n // 2), 1e-3, -1e-3]
        X = np.column_stack([np.ones(len(x)), x])
        y = np.r_[np.ones(n // 2), np.zeros(n // 2), 0.0, 1.0] > 0.0
    return np.where(y, 1.0, -1.0)[:, None] * X


@pytest.mark.parametrize("kind", ["overlap", "complete", "quasi", "near"])
def test_held_rows_decides_as_one_program_per_row(kind):
    """Each row is moved exactly when its own program, maximising ``D_j a`` under ``D a >= 0``
    in the box, clears the program tolerance (Geyer 2009, section 3.12; Konis 2007): overlap and
    the near-separated design move none, complete separation every row, and quasi-complete
    separation every row off the hyperplane.  The dual certificate (``_unmoved_rows``) may only
    skip programs, never change what they would decide."""
    from scipy.optimize import linprog

    from superglm.diagnostics.separation import _PROGRAM_TOL, _held_rows

    D = _signed_rows(kind, np.random.default_rng(11))
    unit = D / np.linalg.norm(D, axis=1, keepdims=True)
    box = [(-1.0, 1.0)] * D.shape[1]
    zero = np.zeros(len(D))
    reference = np.array(
        [-linprog(-row, A_ub=-unit, b_ub=zero, bounds=box).fun > _PROGRAM_TOL for row in unit]
    )
    expected = {
        "overlap": np.zeros(len(D), bool),
        "near": np.zeros(len(D), bool),
        "complete": np.ones(len(D), bool),
        "quasi": np.arange(len(D)) >= 20,
    }[kind]
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(_held_rows(D), expected)


@pytest.mark.parametrize("design", ["spline", "near"])
def test_held_rows_settles_data_that_do_not_separate_in_two_programs(design, monkeypatch):
    """The cost the scan used to have: a level whose rows overlap took one program per row,
    3,008 for the eight levels of a 3,000-row binomial spline-by-factor fit.  The dual
    certificate of ``_unmoved_rows`` (Konis 2007, chapter 4: overlap exactly when some
    ``y > 0`` has ``D' y = 0``) bounds every row's program within the tolerance, so it and the
    counting program decide every row, on a spline level's 400 rows and on the design that
    overlaps by only ``1e-3``.  Mutation check: with ``_unmoved_rows`` certifying no row the
    counts are 401 and 203, and with ``_CERTIFICATE_SPREAD`` at ``1e4`` the near design's is 124.
    """
    import scipy.optimize

    import superglm.diagnostics.separation as separation

    rng = np.random.default_rng(2)
    if design == "spline":
        x = np.sort(rng.uniform(size=400))
        spec = Spline(n_knots=8)
        spec.build(x)
        raw = np.asarray(spec._raw_basis_matrix(x), dtype=np.float64)
        y = rng.uniform(size=400) < 1.0 / (1.0 + np.exp(-2.0 * np.sin(6.0 * x)))
        D = np.where(y, 1.0, -1.0)[:, None] * raw
    else:
        D = _signed_rows("near", np.random.default_rng(11))

    calls = []
    real = scipy.optimize.linprog

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", counted)
    held = separation._held_rows(D)
    assert not held.any()
    assert len(calls) == 2


def test_a_counting_program_hit_the_certificate_rules_out_does_not_refuse_the_design():
    """A level whose signed rows have an exactly positive dual null vector cannot separate.

    Level B's eight responses alternate along ``x``, two of its values ``2e-7`` apart.  HiGHS's
    counting program marks rows moved by breaking its constraints by about ``5.6e-9``, under its
    feasibility tolerance; kept, those hits are refused as separation under
    ``separation="error"``.  The certificate (``_unmoved_rows``) covers all eight rows, so
    ``_held_rows`` drops the hits and the design fits.  Mutation check: keeping counting-program
    hits on certified rows raised ``SeparationError`` naming x 0.37 and 0.952 of level B.
    """
    import superglm.diagnostics.separation as separation

    xa = np.linspace(0.0, 1.0, 50)
    xb = np.array([0.347, 0.37, 0.403, 0.4030002, 0.47, 0.588, 0.808, 0.952])
    X = pd.DataFrame({"x": np.r_[xa, xa, xb], "f": ["A"] * 100 + ["B"] * 8})
    y = np.r_[np.zeros(50), np.ones(50), np.arange(8) % 2].astype(float)
    decided = []
    real = separation._held_rows

    def recorded(D):
        decided.append(real(D))
        return decided[-1]

    model = SuperGLM(
        family="binomial",
        features={
            "x": Spline(knots=[0.2, 0.4, 0.6, 0.8], boundary=(0.0, 1.0)),
            "f": Categorical(base="A"),
        },
        interactions=[("x", "f")],
        discrete=False,
        separation="error",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(separation, "_held_rows", recorded)
        model.fit(X, y)
    assert [held.tolist() for held in decided] == [[False] * 8]
    assert model.predict(X).shape == (len(X),)
