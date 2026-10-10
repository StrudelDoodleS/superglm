"""P-splines on uneven knots take Li and Cao's general difference penalty (arXiv:2201.06808)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.features._spline_penalties import (
    _polynomial_coefficients,
    build_difference_penalty,
    build_general_difference_penalty,
)
from superglm.solvers.rank import SHARED_RANK_POLICY

UNEVEN = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 5.0, 8.0]
u = np.finfo(np.float64).eps / 2


def _built(**kwargs):
    spline = Spline(kind="ps", **kwargs)
    spline.build(np.linspace(0.0, 10.0, 400))
    return spline


def _null_bound(spline, penalty, line) -> float:
    """Round-off bound on ``line' P line`` for a line Li and Cao's penalty annihilates.

    Building ``D`` costs ``gamma_{2m}`` per entry (m differences, m scalings),
    forming ``D'D`` ``gamma_n``, a final scale factor one rounding, and the
    quadratic form ``gamma_{2n}`` (Higham 2002, sections 3.1 and 3.5). The
    difference rows keep their sign pattern, so ``|D|'|D| = |P|`` up to those
    factors, and every error is bounded by ``|line|' |P| |line|``; with
    ``m = 2`` the sum of the counts is at most ``4n`` roundings, and twice that
    covers ``gamma_k <= 2 k u`` for ``k u <= 1/2``.
    """
    scale = np.abs(line) @ np.abs(penalty) @ np.abs(line)
    return 8 * spline._n_basis * u * scale


def _greville(spline) -> np.ndarray:
    """The coefficients of f(x) = x in the spline's B-spline basis."""
    t, d = spline._knots, spline.degree + 1
    return np.array([t[j + 1 : j + d].mean() for j in range(spline._n_basis)])


def _assert_the_projected_factor_keeps_the_line(spline, penalty, line):
    """The penalty is the Gram of ``F = Delta_2 (I - Q Q')`` and ``F`` sends the line to round-off.

    ``P = fl(F'F)`` is within ``gamma_{n-2} |F|'|F|`` of ``F'F`` (Higham 2002,
    section 3.5); two such Grams differ by twice that, which Li and Cao's rows or
    the unprojected ``Delta_2`` miss by orders of magnitude. So ``line' P line``
    is ``|F line|**2`` plus that Gram's rounding, and ``F line`` is bounded here.

    Exactly, ``line = C a`` with ``C`` Marsden's coefficients of ``1, s`` and
    ``a = ((lo + hi) / 2, (hi - lo) / 2)``. The computed ``line`` is off by
    ``p u mean|t|`` per entry (p - 1 sums and a division over ``p = degree``
    knots), and ``C``'s ``s`` column by ``(p + 4) u (mean|s| + |lo + hi| / (hi -
    lo))`` (four roundings map a knot to ``s``, p to average them). Instead of
    Householder's backward error with its unstated constant (Higham, theorem
    19.4), the computed ``Q``, ``R`` give theirs a posteriori: ``O = Q'Q - I``
    to ``(n + 1) u |Q|'|Q|`` and ``C - QR`` to ``(m + 1) u (|C| + |Q||R|)``. As
    ``(I - QQ') Q R a = -Q O R a``, and ``||Q|| <= sqrt(2)``, ``||I - QQ'|| <= 1``
    once ``||O|| <= 1``,
    ``||(I - QQ') line|| <= ||Q|| ||O|| ||R a|| + ||C - QR|| ||a|| + |a_1|
    ||C_1 error|| + ||line error||``. Forming ``F`` costs ``(n + m + 1) u (|Delta|
    + |Delta||Q||Q'|)`` and ``F @ line`` ``n u |F||line|``; ``||Delta_2|| <= 4``.
    Doubling covers ``gamma_k <= 2 k u``, the second-order terms and the
    bound's own rounding, all of relative size ``n u``.
    """
    t, p, m, n = spline._knots, spline.degree, 2, spline._n_basis
    coefficients = _polynomial_coefficients(t, p, m)
    Q, R = np.linalg.qr(coefficients)
    delta = np.diff(np.eye(n), n=m, axis=0)
    factor = delta - (delta @ Q) @ Q.T
    # Each Gram costs gamma_{n-2} of |F|'|F|; doubled as above.
    gram = np.abs(factor).T @ np.abs(factor)
    assert np.all(np.abs(penalty - factor.T @ factor) <= 4 * n * u * gram)

    lo, hi = t[p], t[n]
    a = np.array([(lo + hi) / 2, (hi - lo) / 2])
    s = (2.0 * t - (lo + hi)) / (hi - lo)
    windows = np.lib.stride_tricks.sliding_window_view
    mean_s = windows(np.abs(s[1 : n + p]), p).mean(axis=1)
    mean_t = windows(np.abs(t[1 : n + p]), p).mean(axis=1)
    orthogonality = np.linalg.norm(Q.T @ Q - np.eye(m)) + (n + 1) * u * np.linalg.norm(
        np.abs(Q).T @ np.abs(Q)
    )
    assert orthogonality <= 1
    residual = np.linalg.norm(coefficients - Q @ R) + (m + 1) * u * np.linalg.norm(
        np.abs(coefficients) + np.abs(Q) @ np.abs(R)
    )
    projection = (
        2 * orthogonality * np.linalg.norm(R) * np.linalg.norm(a)
        + residual * np.linalg.norm(a)
        + (p + 4) * u * a[1] * np.linalg.norm(mean_s + abs(lo + hi) / (hi - lo))
        + p * u * np.linalg.norm(mean_t)
    )
    rounding = np.abs(delta) + np.abs(delta) @ np.abs(Q) @ np.abs(Q).T
    bound = 2 * (
        4 * projection
        + (n + m + 1) * u * np.linalg.norm(rounding @ np.abs(line))
        + n * u * np.linalg.norm(np.abs(factor) @ np.abs(line))
    )
    assert np.linalg.norm(factor @ line) <= bound


@pytest.mark.parametrize(
    "kwargs",
    [{"knots": UNEVEN}, {"n_knots": 8, "knot_strategy": "quantile"}],
    ids=["stated", "quantile"],
)
def test_a_straight_line_is_unpenalised_on_uneven_knots(kwargs):
    """The general penalty's null space is the polynomials of degree below m, at any spacing.

    The standard difference penalty puts 5.7 on this line with the stated knots.
    """
    if "knot_strategy" in kwargs:
        spline = Spline(kind="ps", **kwargs)
        x = np.random.default_rng(0).gamma(2.0, 1.5, 3000)
        spline.build(x)
    else:
        spline = _built(**kwargs)
    line = _greville(spline)
    penalty = spline._build_penalty_for_order(2)
    assert abs(line @ penalty @ line) <= _null_bound(spline, penalty, line)


def test_the_uniform_rule_keeps_the_standard_penalty_exactly():
    spline = _built(n_knots=8)
    np.testing.assert_array_equal(
        spline._build_penalty_for_order(2), build_difference_penalty(spline._n_basis, 2)
    )


def test_on_evenly_spaced_knots_the_general_penalty_is_the_standard_one():
    knots = np.arange(-3.0, 15.0)
    for order in (1, 2, 3):
        general = build_general_difference_penalty(knots, 3, order)
        standard = build_difference_penalty(len(knots) - 4, order)
        np.testing.assert_allclose(general, standard, rtol=0, atol=64 * u * np.abs(standard).max())


def test_a_penalty_order_above_the_degree_keeps_the_standard_penalty():
    spline = _built(knots=UNEVEN, m=4, degree=3)
    np.testing.assert_array_equal(
        spline._build_penalty_for_order(4), build_difference_penalty(spline._n_basis, 4)
    )


def test_the_most_smoothed_fit_on_uneven_knots_is_a_straight_line():
    """Li and Cao's figure 4: the standard penalty's limit bends where the knots crowd.

    The fit minimises ``RSS / (2n) + lambda/2 beta' P beta`` (the Hessian
    ``compute_R_inv`` factors), so comparing it with the constant fit gives
    ``lambda beta' P beta <= TSS / n``, and the part of beta outside the
    penalty's null space has norm at most ``sqrt(TSS / (n lambda sigma))``,
    ``sigma`` the smallest nonzero eigenvalue of ``P``. The null-space part is
    a line, whose second differences on an even grid vanish, and B-splines
    are a nonnegative partition of unity, so a second difference of the fit
    is at most four times that norm. The standard penalty leaves 2.6% of the
    range as curvature at this lambda.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 10.0, 2000)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    lam = 1e11
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", knots=UNEVEN)},
    ).fit(pd.DataFrame({"x": x}), y)
    fitted = model.predict(pd.DataFrame({"x": np.linspace(0.5, 9.5, 50)}))
    spline = Spline(kind="ps", knots=UNEVEN)
    spline.build(x)
    sigma = np.linalg.eigvalsh(spline._build_penalty_for_order(2))[2]
    tss = np.sum((y - y.mean()) ** 2)
    assert np.abs(np.diff(fitted, 2)).max() <= 4 * np.sqrt(tss / (x.size * lam * sigma))


CLUSTER_X = np.r_[np.linspace(0.0, 1.0, 300), np.linspace(0.5, 0.5001, 300)]
CLUSTER_KNOTS = np.linspace(0.5, 0.5001, 8)


def test_knots_clustered_past_float64_keep_the_null_space_and_the_rank():
    """Li and Cao's rows reach (hbar/h)**4 = 2e16 here, and their Gram's null
    direction rounded to -5 (the correctly rounded exact Gram's to -3).

    The projected standard factor keeps the line unpenalised and REML's rank
    rule finds the penalty's true rank of n_basis - 2.
    """
    spline = Spline(kind="ps", knots=CLUSTER_KNOTS)
    spline.build(CLUSTER_X)
    penalty = spline._build_penalty_for_order(2)
    _assert_the_projected_factor_keeps_the_line(spline, penalty, _greville(spline))
    eigenvalues = np.linalg.eigvalsh(penalty)
    ranked = np.count_nonzero(eigenvalues > np.finfo(np.float64).eps ** (2 / 3) * eigenvalues[-1])
    assert ranked == spline._n_basis - 2


@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
def test_a_fit_on_clustered_stated_knots_is_the_line(fit):
    """Both fits raised LinAlgError: the null direction's negative penalty
    broke the reparametrisation's Cholesky factor.

    The line lies in the penalty's null space and the basis's span, so the
    penalised fit is the line; the standard penalty bends it by 0.27 of its
    range at lambda = 100. The fit solves ``H beta = X'y`` for the penalised
    Gram ``H = X'X + S`` by Cholesky; forming ``H`` costs ``gamma_N |X|'|X|``
    (N rows) and the solve ``gamma_{3k+1} |R'||R|`` (Higham 2002, section 3.5
    and theorem 10.4), each at most ``k`` times its bound in 2-norm for ``k``
    coefficients, and ``X'y`` costs ``gamma_N |X|'|y|``. As ``||X H^(-1/2)|| <=
    1`` and ``beta' H beta = ||y||**2`` (``S beta = 0``), the fitted values move
    by at most ``k (2N + 3k + 1) u cond(H) ||y||``, doubled for ``gamma``. The
    shared rank policy accepts a Gram without a factor certificate only below
    ``warning_condition = 1/sqrt(eps)``, and the fit's ``H`` is under it.
    """
    y = 2.0 * CLUSTER_X
    frame = pd.DataFrame({"x": CLUSTER_X})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", knots=CLUSTER_KNOTS)},
        **({"spline_penalty": 100.0} if fit == "fit" else {}),
    )
    getattr(model, fit)(frame, y)
    inverse = model._fit_active_info[3]  # (X'WX + S)^-1 with the intercept
    condition = np.linalg.cond(inverse)
    assert condition <= SHARED_RANK_POLICY.warning_condition
    k, rows = inverse.shape[0], y.size
    bound = 2 * k * (2 * rows + 3 * k + 1) * u * condition * np.linalg.norm(y)
    assert np.abs(model.predict(frame) - y).max() <= bound


@pytest.mark.parametrize(
    ("knots", "x", "kind"),
    [
        (np.linspace(0.0, 10.0, 10)[1:-1], np.linspace(0.0, 10.0, 400), "standard"),
        (UNEVEN, np.linspace(0.0, 10.0, 400), "general"),
        (CLUSTER_KNOTS, CLUSTER_X, "projected"),
    ],
    ids=["even", "uneven", "clustered"],
)
def test_the_spec_records_which_difference_penalty_it_took(knots, x, kind):
    """The clustered knots' general penalty has a condition of 1e19, past 1/sqrt(eps)."""
    spline = Spline(kind="ps", knots=knots)
    spline.build(x)
    assert spline._difference_penalty == {2: kind}


@pytest.mark.parametrize(
    ("knots", "x", "kind"),
    [(UNEVEN, np.linspace(0.0, 10.0, 400), "general"), (CLUSTER_KNOTS, CLUSTER_X, "projected")],
    ids=["uneven", "clustered"],
)
def test_the_reports_name_the_difference_penalty(knots, x, kind):
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=1.0,
        features={"x": Spline(kind="ps", knots=knots)},
    ).fit(frame, np.sin(3.0 * frame["x"]))
    rows = [row for row in model.summary()._coef_rows if row.is_spline]
    assert [row.difference_penalty for row in rows] == [kind]
    assert model.diagnostics()["x"]["difference_penalty"] == kind
    assert model.term_inference("x").spline.difference_penalty == kind
    assert model.knot_summary()["x"]["difference_penalty"] == kind


def _skewed(sigma: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, sigma, n)
    return x, rng


def _reml_ranks(model) -> dict[str, float]:
    from superglm.model.reml_setup import collect_reml_groups
    from superglm.reml.penalty_algebra import build_penalty_components

    matrices = model._dm.group_matrices
    components = build_penalty_components(matrices, collect_reml_groups(model._groups, matrices))
    return {component.name: component.rank for component in components}


@pytest.mark.parametrize("select", [False, True], ids=["plain", "select"])
def test_reml_ranks_skewed_quantile_knots_at_the_penalty_true_rank(select):
    """quantile_rows on lognormal(0, 1.5): Li and Cao's rows run from 1e-4 to 1e10.

    REML ranks a penalty at eps**(2/3) of its largest eigenvalue, which put the
    two tail directions under the cut (rank 10 of 12) while the fit still
    penalised them. The penalty REML sees has rank n_basis - 2, the
    polynomials of degree below 2 being its null space.
    """
    x, rng = _skewed(1.5, 10_000, 0)
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(np.log(x))))
    spline = Spline(kind="ps", n_knots=10, knot_strategy="quantile_rows", select=select)
    model = SuperGLM(family="poisson", features={"x": spline}).fit_reml(pd.DataFrame({"x": x}), y)
    ranks = _reml_ranks(model)
    wiggle = ranks["x:wiggle"] if select else ranks["x"]
    assert wiggle == model._specs["x"]._n_basis - 2


@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
@pytest.mark.parametrize("kind", [None, "cr"], ids=["kindless", "cr"])
def test_select_splits_a_skewed_integrated_penalty_at_its_structural_null_space(kind, fit):
    """quantile_rows on lognormal(0, 2): the cr penalty's tail direction sits at 4e-12 of
    its largest eigenvalue, under the eps**(2/3) cut, so the split counted three null
    eigenvalues and refused the kind. 0.39's kindless P-spline fitted it.

    The split now reads the null space from the structural penalty (unit-norm interval
    blocks), which the real penalty must annihilate. By Davis and Kahan's sin-theta
    theorem the computed null direction is within ``delta = n eps ||S|| / lambda_3(S)``
    of the exact one, 1e-12 here (``lambda_3`` is 2.8e-3 of the structural penalty's
    largest eigenvalue). So ``null' P null`` is under ``delta**2 ||P||`` plus the quadratic
    form's rounding, ``gamma_{2n} |null|' |P| |null|`` (Higham 2002, section 3.5), doubled
    for ``gamma_k <= 2 k u``.
    """
    x, rng = _skewed(2.0, 10_000, 0)
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(np.log(x))))
    kinds = {} if kind is None else {"kind": kind}
    spline = Spline(n_knots=10, knot_strategy="quantile_rows", select=True, **kinds)
    model = SuperGLM(family="poisson", features={"x": spline})
    getattr(model, fit)(pd.DataFrame({"x": x}), y)
    spec = model._specs["x"]
    penalty = spec._build_penalty()
    null = spec._U_null[:, 0]
    n = spec._n_basis
    structural = np.linalg.eigvalsh(spec._structural_penalty_for_order(2))
    delta = n * 2 * u * structural[-1] / structural[2]
    bound = delta**2 * np.linalg.norm(penalty, 2) + 4 * n * u * (
        np.abs(null) @ np.abs(penalty) @ np.abs(null)
    )
    assert np.all(np.isfinite(model.predict(pd.DataFrame({"x": x}))))
    assert null @ penalty @ null <= 2 * bound


@pytest.mark.parametrize("select", [False, True], ids=["plain", "select"])
@pytest.mark.parametrize(
    ("kind", "sigma", "n_knots", "nullity"),
    [(None, 2.0, 10, 4), ("bs", 1.5, 20, 2)],
    ids=["kindless_cr", "bs"],
)
def test_reml_ranks_a_skewed_integrated_penalty_at_its_structural_rank(
    kind, sigma, n_knots, nullity, select
):
    """REML ranked a cr or bs penalty at eps**(2/3) of its largest eigenvalue, and on
    these knots the tail direction (4e-12 for cr, 2e-11 for bs) fell under it: the fit
    penalised a direction that log|S|+ counted as unpenalised. The rank is the
    structure's: ``n_basis`` less the lines, and for cr less its two natural boundary
    conditions as well.
    """
    x, rng = _skewed(sigma, 10_000, 0)
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(np.log(x))))
    kinds = {} if kind is None else {"kind": kind}
    spline = Spline(n_knots=n_knots, knot_strategy="quantile_rows", select=select, **kinds)
    model = SuperGLM(family="poisson", features={"x": spline}).fit_reml(pd.DataFrame({"x": x}), y)
    ranks = _reml_ranks(model)
    wiggle = ranks["x:wiggle"] if select else ranks["x"]
    assert wiggle == model._specs["x"]._n_basis - nullity


def test_select_names_the_knot_spread_binary64_cannot_hold():
    """On lognormal(0, 3) quantile knots the cr penalty's tail curvature is below 1e-15 of
    its largest, under the eigensolver's resolution: no split can recover it. The refusal
    blamed the kind ("may not support select=True"); it names the spread and what to do.
    The kind it names, ps, fits these knots and REML ranks its penalty at n_basis - 2.
    """
    x, rng = _skewed(3.0, 10_000, 0)
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(np.log(x))))
    spline = Spline(n_knots=10, knot_strategy="quantile_rows", select=True)
    model = SuperGLM(family="poisson", features={"x": spline})
    with pytest.raises(ValueError, match="differ so much in width") as raised:
        model.fit(pd.DataFrame({"x": x}), y)
    assert "may not support" not in str(raised.value)
    assert 'kind="ps"' in str(raised.value)
    assert "whose penalty double precision can hold on any knots" in str(raised.value)
    ps = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="ps", n_knots=10, knot_strategy="quantile_rows", select=True)},
    ).fit_reml(pd.DataFrame({"x": x}), y)
    assert _reml_ranks(ps)["x:wiggle"] == ps._specs["x"]._n_basis - 2


@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
def test_a_third_order_penalty_fits_on_lognormal_quantile_knots(fit):
    """m = 3 on lognormal(0, 2) quantile knots spans (hbar/span)**6, past float64;
    the Cholesky factor of the reparametrisation raised LinAlgError."""
    x, rng = _skewed(2.0, 5_000, 7)
    y = rng.poisson(np.exp(-1.0 + 0.1 * np.log(x)))
    frame = pd.DataFrame({"x": x})
    spline = Spline(kind="ps", n_knots=20, knot_strategy="quantile", m=3)
    model = SuperGLM(family="poisson", features={"x": spline})
    getattr(model, fit)(frame, y)
    assert np.all(np.isfinite(model.predict(frame)))


@pytest.mark.parametrize("sigma", [1.0, 1.5])
def test_a_decomposed_tensor_with_a_skewed_quantile_margin_splits_off_the_bilinear(sigma):
    """The tensor split counted null eigenvalues under a fixed 1e-8 of the largest,
    which the margin's spread crossed (2 to 24 of them), and on lognormal(0, 1) the
    margin's 1e6 entries broke the component sum's absolute check."""
    from superglm.features.interaction import TensorInteraction

    x1, rng = _skewed(sigma, 4_000, 5)
    x2 = rng.uniform(0.0, 1.0, x1.size)
    margin_1 = Spline(kind="ps", n_knots=5, knot_strategy="quantile")
    margin_2 = Spline(kind="ps", n_knots=5)
    margin_1.build(x1)
    margin_2.build(x2)
    infos = TensorInteraction("a", "b", decompose=True).build(
        x1, x2, {"a": margin_1, "b": margin_2}
    )
    assert [info.subgroup_name for info in infos] == ["bilinear", "wiggly"]


@pytest.mark.parametrize("kind", ["cr", "bs"])
def test_a_decomposed_discrete_tensor_reads_its_null_space_from_the_structural_margins(kind):
    """The split counted the eigenvalues under eps**(2/3) of the largest: a skewed cr
    margin's spread put a range direction there (2 null eigenvalues, not 1), and the
    kindless default is cr. The structural margins give the bilinear direction; a bs
    margin's tail curvature (2.5e-15 of the largest) is refused by name instead."""
    from superglm.features._spline_select import _null_mask
    from superglm.features.interaction import TensorInteraction, _normalize_tensor_penalty

    x1, rng = _skewed(2.0, 6_000, 3)
    x2 = rng.uniform(0.0, 1.0, x1.size)
    margin_1 = Spline(kind=kind, n_knots=10, knot_strategy="quantile_rows")
    margin_2 = Spline(kind=kind, n_knots=5)
    margin_1.build(x1)
    margin_2.build(x2)
    tensor = TensorInteraction("a", "b", decompose=True)
    if kind == "bs":
        with pytest.raises(ValueError, match="decompose=True cannot split this penalty"):
            tensor.build_discrete(x1, x2, {"a": margin_1, "b": margin_2}, n_bins=(256, 256))
        return
    infos = tensor.build_discrete(x1, x2, {"a": margin_1, "b": margin_2}, n_bins=(256, 256)).infos
    assert [info.subgroup_name for info in infos] == ["bilinear", "wiggly"]
    m1, m2 = tensor._marginal1, tensor._marginal2
    omega = np.kron(_normalize_tensor_penalty(m1.penalty), np.eye(tensor._p2)) + np.kron(
        np.eye(tensor._p1), _normalize_tensor_penalty(m2.penalty)
    )
    # The real penalty alone puts a second direction under the cut.
    assert np.sum(_null_mask(np.linalg.eigvalsh(omega))) == 2
    # The bilinear direction is the product of the margins' centred lines: the
    # structural tensor penalty annihilates it to round-off of its own norm.
    bilinear = infos[0].projection
    structural = tensor._structural_tensor_penalty()
    unit = np.finfo(float).eps / 2
    residual = np.linalg.norm(structural @ bilinear)
    assert residual <= 8 * omega.shape[0] * unit * np.linalg.norm(structural, 2)


@pytest.mark.slow
def test_a_tensor_with_a_skewed_quantile_margin_fits_by_reml():
    """It raised PenaltyNumericalError: the reference root could not meet its accuracy contract."""
    x1, rng = _skewed(1.5, 4_000, 1)
    x2 = rng.uniform(0.0, 1.0, x1.size)
    y = rng.poisson(np.exp(-1.0 + 0.2 * np.log(x1) * x2))
    frame = pd.DataFrame({"a": x1, "b": x2})
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(kind="ps", n_knots=10, knot_strategy="quantile"),
            "b": Spline(kind="ps", n_knots=5),
        },
        interactions=[("a", "b")],
    ).fit_reml(frame, y)
    assert np.all(np.isfinite(model.predict(frame)))


def test_knots_too_close_for_float64_get_a_finite_penalty():
    """hbar / span reaches 1e159 here and its fourth power overflows: the penalty
    had inf and NaN entries. The projected standard factor keeps the line."""
    knots = np.r_[-0.2, np.arange(4) * 1e-160, 0.2]
    x = np.linspace(-0.5, 0.5, 500)
    spline = Spline(kind="ps", knots=knots)
    spline.build(x)
    penalty = spline._build_penalty_for_order(2)
    assert np.isfinite(penalty).all()
    _assert_the_projected_factor_keeps_the_line(spline, penalty, _greville(spline))
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"x": Spline(kind="ps", knots=knots)}
    ).fit_reml(frame, np.sin(3.0 * x))
    assert np.all(np.isfinite(model.predict(frame)))


def test_stated_knots_at_the_uniform_rule_positions_reproduce_the_uniform_fit():
    """``fitted_knots`` with ``fitted_boundary`` is the documented way to reproduce a
    fit; the open knot vector's widened ends made the general penalty differ by 2%."""
    x = np.random.default_rng(0).uniform(0.0, 10.0, 5_000)
    placed = Spline(kind="ps", n_knots=8)
    placed.build(x)
    stated = Spline(kind="ps", knots=placed.fitted_knots, boundary=placed.fitted_boundary)
    stated.build(x)
    np.testing.assert_array_equal(stated._build_penalty(), placed._build_penalty())


@pytest.mark.parametrize(
    ("lo", "hi", "n_knots"),
    [(0.0, 10.0, 8)] + [(1e3, 1e3 + 7.0, n_knots) for n_knots in range(1, 6)],
)
def test_refit_from_fitted_knots_reproduces_the_uniform_fit_on_any_domain(lo, hi, n_knots):
    """A refit from ``fitted_knots`` states the uniform rule's own ``linspace`` knots, which
    must reproduce the uniform penalty however few knots there are and however far the
    domain sits from 0."""
    x = np.random.default_rng(0).uniform(lo, hi, 2_000)
    placed = Spline(kind="ps", n_knots=n_knots, boundary=(lo, hi))
    placed.build(x)
    stated = Spline(kind="ps", knots=placed.fitted_knots, boundary=placed.fitted_boundary)
    stated.build(x)
    np.testing.assert_array_equal(stated._build_penalty(), placed._build_penalty())


@pytest.mark.parametrize("n_knots", range(1, 6))
def test_typed_knots_within_the_uniform_rounding_keep_the_standard_penalty(n_knots):
    """A knot 15 u max(|lo|, |hi|) off the uniform grid spreads the gaps by about 30 of
    that unit, which ``_evenly_spaced`` accepts (32). Its old tolerance, 4 (n_knots + 2)
    units, was below 30 for n_knots <= 5, so these knots took the general penalty."""
    lo, hi = 50.0, 60.0
    typed = np.linspace(lo, hi, n_knots + 2)[1:-1]
    typed[n_knots // 2] += 15 * u * max(abs(lo), abs(hi))
    x = np.random.default_rng(0).uniform(lo, hi, 2_000)
    spline = Spline(kind="ps", knots=typed, boundary=(lo, hi))
    spline.build(x)
    np.testing.assert_array_equal(
        spline._build_penalty_for_order(2), build_difference_penalty(spline._n_basis, 2)
    )
