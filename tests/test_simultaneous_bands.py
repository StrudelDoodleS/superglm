"""Tests for simultaneous confidence bands."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._roundoff import EPS, gamma
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@pytest.fixture
def spline_model():
    """Fitted Poisson model with a spline feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.uniform(0, 10, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x))
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.001,
        features={"x": Spline(n_knots=10, penalty="ssp")},
    )
    model.fit(X, y)
    return model


class TestSimultaneousBands:
    def test_returns_dataframe(self, spline_model):
        result = spline_model.simultaneous_bands("x")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 200
        expected_cols = {
            "x",
            "log_relativity",
            "relativity",
            "se",
            "ci_lower_pointwise",
            "ci_upper_pointwise",
            "ci_lower_simultaneous",
            "ci_upper_simultaneous",
        }
        assert set(result.columns) == expected_cols

    def test_simultaneous_wider_than_pointwise(self, spline_model):
        result = spline_model.simultaneous_bands("x")
        # Simultaneous bands should always be wider
        sim_width = result["ci_upper_simultaneous"] - result["ci_lower_simultaneous"]
        pw_width = result["ci_upper_pointwise"] - result["ci_lower_pointwise"]
        assert np.all(sim_width >= pw_width - 1e-10)

    def test_smaller_alpha_gives_wider_bands(self, spline_model):
        r05 = spline_model.simultaneous_bands("x", alpha=0.05)
        r01 = spline_model.simultaneous_bands("x", alpha=0.01)

        w05 = (r05["ci_upper_simultaneous"] - r05["ci_lower_simultaneous"]).mean()
        w01 = (r01["ci_upper_simultaneous"] - r01["ci_lower_simultaneous"]).mean()
        assert w01 > w05

    def test_reproducible_with_seed(self, spline_model):
        r1 = spline_model.simultaneous_bands("x", seed=42)
        r2 = spline_model.simultaneous_bands("x", seed=42)
        np.testing.assert_array_equal(
            r1["ci_lower_simultaneous"].values,
            r2["ci_lower_simultaneous"].values,
        )

    def test_non_spline_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({"x": rng.standard_normal(n)})
        y = rng.poisson(1.0, n).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        with pytest.raises(TypeError, match="spline"):
            model.simultaneous_bands("x")

    def test_unfitted_raises(self):
        model = SuperGLM(features={"x": Spline(n_knots=5)})
        with pytest.raises(RuntimeError, match="fitted"):
            model.simultaneous_bands("x")

    def test_a_singular_face_covariance_still_gives_finite_bands(self, spline_model):
        """A constrained fit's active-face covariance is singular by construction.

        Rounding can leave its zero eigenvalue slightly negative. The term's
        block here has scale 1e6 and one eigenvalue at -1e-9 of it, so no
        Cholesky factor exists; the principal root takes it as a null
        direction. Mutation: a Cholesky root raises ``LinAlgError``.
        """
        from superglm.inference._term_covariance import simultaneous_bands

        model = spline_model
        cov, active = model._coef_covariance
        subs = [g for g in active if g.feature_name == "x"]
        idx = np.concatenate([np.arange(g.start, g.end) for g in subs])
        q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((len(idx), len(idx))))
        values = np.ones(len(idx))
        values[-1] = -1e-9
        crafted = np.zeros_like(cov)
        crafted[np.ix_(idx, idx)] = 1e6 * (q * values) @ q.T
        bands = simultaneous_bands(
            "x",
            result=model.result,
            groups=model._groups,
            specs=model._specs,
            interaction_specs=model._interaction_specs,
            covariance_fn=lambda: (crafted, active),
            n_sim=500,
        )
        assert np.isfinite(
            bands[["ci_lower_simultaneous", "ci_upper_simultaneous"]].to_numpy()
        ).all()


class TestCriticalValueUnits:
    @pytest.mark.parametrize("unit", [1e-5, 1e-19])
    def test_critical_value_ignores_the_units_of_the_covariance(
        self, spline_model, monkeypatch, unit
    ):
        """Rescaling the response rescales the covariance by its square, not the critical value."""
        from superglm.inference._term_covariance import simultaneous_bands

        covariance, active_groups = spline_model._coef_covariance
        real_quantile = np.quantile

        def critical_value(scale):
            seen = []

            def spy(values, q, *args, **kwargs):
                seen.append(real_quantile(values, q, *args, **kwargs))
                return seen[-1]

            with monkeypatch.context() as patch:
                patch.setattr(np, "quantile", spy)
                simultaneous_bands(
                    "x",
                    result=spline_model.result,
                    groups=spline_model._groups,
                    specs=spline_model._specs,
                    covariance_fn=lambda: (scale**2 * covariance, active_groups),
                    n_sim=2000,
                )
            return seen[-1]

        assert critical_value(unit) == pytest.approx(critical_value(1.0), rel=1e-9)


def test_singular_covariance_takes_one_sampler_however_its_cholesky_rounds(
    spline_model, monkeypatch
):
    """Whether Cholesky completes on a singular covariance is a rounding accident."""
    from superglm.inference._term_covariance import simultaneous_band_and_critical_value

    covariance, active_groups = spline_model._coef_covariance
    k = covariance.shape[0]
    # Project out one direction: a singular covariance, as on a constrained fit's face.
    direction = np.linalg.qr(np.random.default_rng(3).normal(size=(k, 1)))[0]
    projector = np.eye(k) - direction @ direction.T
    singular = projector @ covariance @ projector
    singular = 0.5 * (singular + singular.T)
    real_cholesky = np.linalg.cholesky

    def critical_value(cholesky):
        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, "cholesky", cholesky)
            _, value = simultaneous_band_and_critical_value(
                "x",
                result=spline_model.result,
                groups=spline_model._groups,
                specs=spline_model._specs,
                covariance_fn=lambda: (singular, active_groups),
                n_sim=2000,
            )
        return value

    def pivot_rounded_negative(matrix):
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    def pivot_rounded_positive(matrix):
        return real_cholesky(matrix + 1e-14 * np.max(np.diag(matrix)) * np.eye(len(matrix)))

    assert critical_value(pivot_rounded_negative) == critical_value(pivot_rounded_positive)


def _rotated_within_repeated_eigenvalues(eigh, mix):
    """``eigh`` with each repeated eigenvalue's eigenvectors mixed by ``mix``.

    Any orthonormal basis of an eigenspace is an eigensolver's valid output;
    which one LAPACK returns depends on its driver and on the matrix's rounding.
    """

    def changed(matrix, *args, **kwargs):
        values, vectors = eigh(matrix, *args, **kwargs)
        vectors = vectors.copy()
        repeated = np.isclose(values[:, None], values[None, :], rtol=1e-8, atol=0.0)
        for i in np.flatnonzero(repeated.sum(axis=1) > 1):
            block = np.flatnonzero(repeated[i])
            if block[0] == i and values[i] > 0.0:
                vectors[:, block] = vectors[:, block] @ mix(len(block))
        return values, vectors

    return changed


def _null_eigenvalues_under_the_cut(eigh, nullity):
    """``eigh`` with its ``nullity`` smallest eigenvalues placed at half the principal root's cut.

    The exact-zero eigenvalues of a constructed covariance come back as rounding
    of order ``u lambda_max``, whose sign and size depend on the BLAS kernel and
    on the units, so whether each lands under the cut is the platform's.  The
    test places them instead; ``test_the_principal_root_drops_...`` pins the cut.
    """

    def placed(matrix, *args, **kwargs):
        values, vectors = eigh(matrix, *args, **kwargs)
        values = values.copy()
        values[:nullity] = 0.5 * matrix.shape[0] * EPS * values[-1]
        return values, vectors

    return placed


def _reversal(size):
    return np.eye(size)[::-1]


def _rotation(size):
    rotation = np.eye(size)
    c, s = np.cos(0.7), np.sin(0.7)
    rotation[:2, :2] = [[c, s], [-s, c]]
    return rotation


@pytest.mark.parametrize("nullity", [0, 1, 4])
@pytest.mark.parametrize("mix", [_reversal, _rotation], ids=["permuted", "rotated"])
def test_a_repeated_eigenvalue_draws_one_band_in_any_eigenbasis_and_units(
    spline_model, monkeypatch, nullity, mix
):
    """The draws follow the covariance, not the eigenbasis LAPACK picks for it.

    The term's covariance has repeated positive eigenvalues and ``nullity``
    null directions, as on a constrained fit's active face.  Mixing each
    repeated eigenvalue's eigenvectors, as another driver or the matrix's
    rounding at other response units does, leaves the critical value alone at
    units 1, 10 and 1e-19.  The principal root moves with the eigensolver's
    backward error, ``k gamma_k`` of the covariance, amplified at most by the
    resolved condition number.  The null eigenvalues are placed under the
    root's cut (``_null_eigenvalues_under_the_cut``), so the comparison does not
    depend on where a platform's eigensolver rounds them.  Mutation: the
    sign-fixed root ``V Lambda^1/2`` moves it by 1e-4 to 3e-2 once
    ``nullity > 0``.
    """
    from superglm.inference._term_covariance import simultaneous_band_and_critical_value

    covariance, active_groups = spline_model._coef_covariance
    idx = np.concatenate(
        [np.arange(g.start, g.end) for g in active_groups if g.feature_name == "x"]
    )
    k = len(idx)
    values = np.concatenate([np.zeros(nullity), [1.0, 1, 1, 2, 2, 2, 3, 3], 4.0 + np.arange(k)])[:k]
    basis, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((k, k)))
    crafted = np.zeros_like(covariance)
    crafted[np.ix_(idx, idx)] = (basis * values) @ basis.T
    crafted = 0.5 * (crafted + crafted.T)
    real_eigh = np.linalg.eigh

    def critical_value(units, eigh):
        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, "eigh", eigh)
            _, value = simultaneous_band_and_critical_value(
                "x",
                result=spline_model.result,
                groups=spline_model._groups,
                specs=spline_model._specs,
                covariance_fn=lambda: (units**2 * crafted, active_groups),
                n_sim=2000,
            )
        return value

    placed = _null_eigenvalues_under_the_cut(real_eigh, nullity)
    reference = critical_value(1.0, placed)
    changed = _rotated_within_repeated_eigenvalues(placed, mix)
    tolerance = k * gamma(k) * values.max() / values[values > 0].min()
    for units in (1.0, 10.0, 1e-19):
        assert critical_value(units, changed) == pytest.approx(reference, rel=tolerance, abs=0.0)


def test_the_principal_root_drops_the_eigenvalues_within_its_cut(monkeypatch):
    """An eigenvalue within ``k eps lambda_max`` of zero is a null direction; one above it is kept.

    The decision is on the eigenvalues the eigensolver returns, so the test
    supplies them, at a factor of two either side of the cut: a negative one,
    one at half the cut and one at twice it, beside three resolved ones.  The
    root ``V diag(sqrt(lambda)) V'`` then has the square roots of the resolved
    and kept ones on its diagonal in ``V``'s basis, to ``k gamma_k`` of its norm.
    Mutation: dropping only the non-positive eigenvalues keeps the one at half
    the cut, 5e-8 of the root's diagonal against a tolerance of 1e-15.
    """
    from superglm.inference._term_covariance import _principal_root

    k = 6
    basis, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((k, k)))
    cut = k * EPS * 4.0
    values = np.array([-cut, 0.5 * cut, 2.0 * cut, 1.0, 2.0, 4.0])
    monkeypatch.setattr(np.linalg, "eigh", lambda matrix: (values, basis))

    root = _principal_root((basis * values) @ basis.T)

    kept = np.sqrt(np.where(values > cut, values, 0.0))
    tolerance = k * gamma(k) * kept.max()
    np.testing.assert_allclose(basis.T @ root @ basis, np.diag(kept), rtol=0.0, atol=tolerance)


def test_a_failing_cholesky_cannot_reach_term_inference(spline_model, monkeypatch):
    """The band sampler factorises nothing that can fail.

    Wilkinson's completion bound for Cholesky assumes no underflow, so a
    covariance at a subnormal scale can fail it however well conditioned it is.
    The bands draw through the principal root instead.  Mutation: the
    spectrum-chosen Cholesky root raises ``LinAlgError`` from ``term_inference``.
    """

    def fails(matrix, *args, **kwargs):
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", fails)
    inference = spline_model.term_inference("x", simultaneous=True, n_sim=500)
    assert np.isfinite(inference.critical_value_simultaneous)
    assert np.all(inference.ci_lower_simultaneous <= inference.ci_upper_simultaneous)
