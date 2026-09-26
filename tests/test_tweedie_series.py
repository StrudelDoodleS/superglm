"""The compiled Dunn-Smyth series against a 50-digit oracle and the p = 1.5 closed form."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import i1e

from superglm._tweedie_series import MAX_ROW_TERMS, series_moments

ORACLE = json.loads((Path(__file__).parent / "fixtures" / "tweedie_series_oracle.json").read_text())
EPS = np.finfo(np.float64).eps


def _peak_magnitude(log_t: float, a: float, mode: int) -> float:
    # |q(j_max)| components: the float64 error of each term's log is a few eps times these.
    return abs(mode * log_t) + abs(math.lgamma(mode + 1.0)) + abs(math.lgamma(a * mode))


def _series_bound(log_t: float, a: float, mode: int) -> float:
    # Each term q(j) = j log t - lgamma(j + 1) - lgamma(a j) is evaluated with about
    # 3 eps of its component magnitudes (the product, two lgammas within an ulp
    # and the rounded argument a j), and log W is a convex combination of the
    # q(j) so it inherits at most the largest per-term error; exp(q - peak) carries
    # both errors again, and the summation of at most MAX_ROW_TERMS positive terms
    # below 1 adds a relative error under that (Higham 2002, sec. 4.2). 16 eps
    # per unit of peak magnitude covers all three with a factor above 1.5.
    return 16.0 * EPS * max(1.0, _peak_magnitude(log_t, a, mode))


@pytest.mark.parametrize("row", ORACLE["rows"], ids=lambda r: f"p{r['p']}-lt{r['log_t']:.2f}")
def test_log_w_matches_50_digit_reference(row):
    ok, log_w, _, _ = series_moments(np.array([row["log_t"]]), row["a"])
    assert ok[0]
    reference = float(row["log_w"])
    assert abs(log_w[0] - reference) <= _series_bound(row["log_t"], row["a"], row["mode"])


def test_p15_matches_bessel_closed_form_at_large_modes():
    # At p = 1.5, a = 1 and W = sqrt(t) I_1(2 sqrt t) (DLMF 10.46.2); i1e is Cephes, ~1e-15.
    log_t = np.linspace(5.0, 30.0, 26)
    ok, log_w, _, _ = series_moments(log_t, 1.0)
    z = 2.0 * np.exp(0.5 * log_t)
    reference = 0.5 * log_t + np.log(i1e(z)) + z
    assert ok.all()
    # The series carries its own per-term cancellation bound at the peak index
    # e^(log t / 2); the closed form's error is the rounding of z (two eps relative,
    # and log W ~ z) plus i1e's few ulps.
    modes = np.maximum(1, np.floor(np.exp(0.5 * log_t))).astype(int)
    series_bound = np.array([_series_bound(lt, 1.0, m) for lt, m in zip(log_t, modes)])
    np.testing.assert_array_less(
        np.abs(log_w - reference), series_bound + 8.0 * EPS * np.abs(reference)
    )


def test_moments_are_row_local():
    rng = np.random.default_rng(3)
    log_t = rng.uniform(-3.0, 12.0, 200)
    alone = np.array([series_moments(log_t[i : i + 1], 0.7)[1][0] for i in range(log_t.size)])
    together = series_moments(log_t, 0.7)[1]
    np.testing.assert_array_equal(alone, together)


def test_mean_and_variance_match_finite_differences_of_log_w():
    # d log W / d log t = E[J]; d2 log W / d log t2 = Var[J].
    log_t, a, h = 4.0, 0.6, 1e-4
    _, lw, mean_j, var_j = series_moments(np.array([log_t - h, log_t, log_t + h]), a)
    assert mean_j[1] == pytest.approx((lw[2] - lw[0]) / (2 * h), rel=1e-7)
    assert var_j[1] == pytest.approx((lw[2] - 2 * lw[1] + lw[0]) / h**2, rel=1e-4)


def test_rows_past_the_work_bound_are_refused_not_nan():
    a = (2 - 1.95) / 0.95
    # Mode ~1e14: needs ~2 sqrt(74 mode/(a+1)) >> MAX_ROW_TERMS terms.
    log_t = (a + 1) * math.log(1e14) + a * math.log(a)
    ok, log_w, mean_j, var_j = series_moments(np.array([0.0, log_t]), a)
    assert ok.tolist() == [True, False]
    assert np.isnan(log_w[1]) and np.isnan(mean_j[1]) and np.isnan(var_j[1])
    assert MAX_ROW_TERMS == 1_000_000
