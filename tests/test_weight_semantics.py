"""The declared weight contract, and what each reading is required to satisfy.

Each contract has a definition that pins it independently of the implementation:
``"frequency"`` says an integer weight is a repeated row, and ``"prior"`` says
the row's density carries ``phi / w``.  The tests below hold the code to those
two statements rather than to numbers it produced, so a regression in either
reading shows up as a disagreement with row replication or with scipy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import (
    Categorical,
    FractionalFrequencyWeightWarning,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Numeric,
    Poisson,
    PriorWeightLatticeWarning,
    Spline,
    SuperGLM,
)
from superglm.distributions import Binomial, Tweedie, weighted_log_likelihood
from superglm.reml.scale import (
    _gamma_saturated_normalizer,
    _gamma_saturated_normalizer_array,
    _scaled_trigamma_minus_inverse_array,
    _shape_times_log_minus_digamma,
    _shape_times_log_minus_digamma_array,
    _trigamma_minus_inverse,
    prepare_tweedie_reml_scale_data,
)
from superglm.solvers.dispersion import (
    dispersion_likelihood_size,
    pearson_residual_degrees_of_freedom,
)


def _frame(seed: int = 0, n: int = 180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(list("abc"), n)
    return rng, pd.DataFrame({"x": x, "g": g})


def _response(rng, frame, family):
    mu = np.exp(0.4 + 1.1 * frame["x"].to_numpy())
    if isinstance(family, Gaussian):
        return mu + rng.normal(0.0, 0.3, len(frame))
    if isinstance(family, Gamma):
        return rng.gamma(5.0, mu / 5.0)
    if isinstance(family, Poisson | NegativeBinomial):
        return rng.poisson(mu).astype(float)
    if isinstance(family, Binomial):
        return rng.integers(0, 2, len(frame)).astype(float)
    return np.where(rng.random(len(frame)) < 0.3, 0.0, rng.gamma(3.0, mu / 3.0))


FAMILIES = [
    pytest.param(Gaussian(), id="gaussian"),
    pytest.param(Gamma(), id="gamma"),
    pytest.param(Poisson(), id="poisson"),
    pytest.param(NegativeBinomial(theta=1.4), id="negative_binomial"),
    pytest.param(Binomial(), id="binomial"),
    pytest.param(Tweedie(p=1.5), id="tweedie"),
]


def _fit(family, frame, y, weights, semantics, *, features=None):
    model = SuperGLM(
        family=family,
        features=features if features is not None else {"x": Spline(n_knots=4), "g": Categorical()},
        weight_semantics=semantics,
    )
    model.fit(frame, y, sample_weight=weights)
    return model


class TestFrequencyIsReplication:
    """The frequency contract's definition: an integer weight is a repeated row."""

    @pytest.mark.parametrize("family", FAMILIES)
    def test_an_integer_weighted_fit_reproduces_the_expanded_one(self, family):
        """Compared on the fit, not on its coordinates.

        ``beta`` is deliberately not asserted.  A penalised spline's ``ssp``
        reparametrisation is derived from a design whose row count differs
        between the compressed and expanded frames, so the two fits express the
        same function in rotated bases: measured max ``|dbeta|`` reaches 1.9e-2
        while the predictions agree to 1.3e-14 and the deviance, effective
        degrees of freedom and dispersion all agree to round-off.  Asserting on
        the coordinates would pin the rotation rather than the identity.

        The bound is the worst relative gap measured across all six families
        and all four quantities (1.32e-14), with roughly 750x headroom.
        """
        rng, frame = _frame(seed=3, n=120)
        y = _response(rng, frame, family)
        counts = rng.integers(1, 4, len(frame))
        weights = counts.astype(float)

        compressed = _fit(family, frame, y, weights, "frequency")
        expanded_frame = frame.loc[frame.index.repeat(counts)].reset_index(drop=True)
        expanded_y = np.repeat(y, counts)
        expanded = _fit(family, expanded_frame, expanded_y, None, "frequency")

        np.testing.assert_allclose(
            np.asarray(compressed.predict(frame), dtype=np.float64),
            np.asarray(expanded.predict(frame), dtype=np.float64),
            rtol=1e-11,
        )
        assert compressed.result.deviance == pytest.approx(expanded.result.deviance, rel=1e-11)
        assert compressed.result.effective_df == pytest.approx(
            expanded.result.effective_df, rel=1e-11
        )
        assert compressed.result.phi == pytest.approx(expanded.result.phi, rel=1e-11)

    def test_the_replication_identity_is_what_fails_under_the_other_contract(self):
        """The control: prior weighting is a different likelihood, not a rescaling."""
        rng, frame = _frame(seed=4, n=120)
        y = _response(rng, frame, Gamma())
        counts = rng.integers(1, 4, len(frame))

        prior = _fit(Gamma(), frame, y, counts.astype(float), "prior")
        expanded_frame = frame.loc[frame.index.repeat(counts)].reset_index(drop=True)
        expanded = _fit(Gamma(), expanded_frame, np.repeat(y, counts), None, "frequency")

        assert prior.result.phi != pytest.approx(expanded.result.phi, rel=1e-3)


class TestPriorIsTheEDMLikelihood:
    """The prior contract's definition: row ``i``'s density carries ``phi / w_i``."""

    def test_a_weighted_gamma_reports_the_analytic_edm_log_likelihood(self):
        rng, frame = _frame(seed=5, n=150)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.3, 6.0, len(frame))

        model = _fit(Gamma(), frame, y, weights, "prior")
        metrics = model.metrics(frame, y, sample_weight=weights)
        mu = np.asarray(model.predict(frame), dtype=np.float64)
        phi = float(model.result.phi)

        analytic = float(stats.gamma.logpdf(y, a=weights / phi, scale=mu * phi / weights).sum())
        assert metrics.log_likelihood == pytest.approx(analytic, rel=1e-10)

    def test_a_weighted_gaussian_reports_the_analytic_edm_log_likelihood(self):
        rng, frame = _frame(seed=6, n=150)
        y = _response(rng, frame, Gaussian())
        weights = rng.uniform(0.3, 6.0, len(frame))

        model = _fit(Gaussian(), frame, y, weights, "prior")
        metrics = model.metrics(frame, y, sample_weight=weights)
        mu = np.asarray(model.predict(frame), dtype=np.float64)
        phi = float(model.result.phi)

        analytic = float(stats.norm.logpdf(y, loc=mu, scale=np.sqrt(phi / weights)).sum())
        assert metrics.log_likelihood == pytest.approx(analytic, rel=1e-10)

    @pytest.mark.parametrize(
        ("family", "reference"),
        [
            pytest.param(
                Poisson(),
                lambda y, mu, w, phi: stats.poisson.logpmf(w * y, w * mu).sum(),
                id="poisson",
            ),
            pytest.param(
                NegativeBinomial(theta=1.4),
                lambda y, mu, w, phi: stats.nbinom.logpmf(
                    w * y, n=w * 1.4, p=1.4 / (mu + 1.4)
                ).sum(),
                id="negative_binomial",
            ),
            pytest.param(
                Binomial(),
                lambda y, mu, w, phi: stats.binom.logpmf(w * y, w, mu).sum(),
                id="binomial",
            ),
        ],
    )
    def test_the_prior_form_matches_scipy_for_the_known_scale_families(self, family, reference):
        """The prior form is the scaled family's own density, checked against scipy.

        ``w Y`` is the same family at parameters scaled by ``w``, and scipy
        evaluates that independently.  Integer weights keep the reference's
        lattice arguments whole.

        What this does NOT establish is that the result is a normalised
        likelihood.  For Poisson and the negative binomial it is, on the
        lattice.  For **Binomial it is not**: ``validate_response`` pins ``y``
        to ``{0, 1}``, so only the all-failure and all-success outcomes of
        ``w`` trials are reachable and their masses sum to one only at
        ``w == 1`` -- see ``TestTheBinomialPriorFormIsNotNormalised``.
        Agreeing with scipy and being a distribution are different claims, and
        this test makes only the first.
        """
        rng, frame = _frame(seed=7, n=90)
        y = _response(rng, frame, family)
        weights = rng.integers(1, 5, len(frame)).astype(float)
        mu = (
            np.clip(0.2 + 0.5 * frame["x"].to_numpy(), 0.05, 0.95)
            if isinstance(family, Binomial)
            else np.exp(0.3 + 0.4 * frame["x"].to_numpy())
        )

        got = weighted_log_likelihood(family, y, mu, weights, 1.0, weight_semantics="prior")
        assert got == pytest.approx(reference(y, mu, weights, 1.0), rel=1e-10)


class TestTheContractsAgreeWhereTheyMust:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_unit_weights_leave_every_published_quantity_alone(self, family):
        rng, frame = _frame(seed=8, n=140)
        y = _response(rng, frame, family)
        ones = np.ones(len(frame))

        prior = _fit(family, frame, y, ones, "prior")
        frequency = _fit(family, frame, y, ones, "frequency")

        np.testing.assert_array_equal(prior.result.beta, frequency.result.beta)
        assert prior.result.phi == frequency.result.phi
        assert prior.result.deviance == frequency.result.deviance
        assert prior.result.effective_df == frequency.result.effective_df
        assert prior.metrics(frame, y, sample_weight=ones).log_likelihood == pytest.approx(
            frequency.metrics(frame, y, sample_weight=ones).log_likelihood, rel=1e-12
        )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_contracts_share_a_score_equation(self, family):
        """Both weightings scale the same sufficient statistic, so beta agrees."""
        rng, frame = _frame(seed=9, n=140)
        y = _response(rng, frame, family)
        weights = rng.uniform(0.4, 5.0, len(frame))
        if isinstance(family, Tweedie):
            weights = np.maximum(weights, 1e-3)

        prior = _fit(family, frame, y, weights, "prior", features={"g": Categorical()})
        frequency = _fit(family, frame, y, weights, "frequency", features={"g": Categorical()})

        np.testing.assert_allclose(prior.result.beta, frequency.result.beta, rtol=1e-7, atol=1e-9)
        assert prior.result.deviance == pytest.approx(frequency.result.deviance, rel=1e-9)


class TestTheContractMovesWhatItShould:
    def test_it_moves_dispersion_degrees_of_freedom_and_the_smoothing_parameter(self):
        rng, frame = _frame(seed=10, n=200)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 4.0, len(frame))

        prior = SuperGLM(
            family=Gamma(),
            features={"x": Spline(n_knots=6), "g": Categorical()},
            weight_semantics="prior",
        )
        prior.fit_reml(frame, y, sample_weight=weights)
        frequency = SuperGLM(
            family=Gamma(),
            features={"x": Spline(n_knots=6), "g": Categorical()},
            weight_semantics="frequency",
        )
        frequency.fit_reml(frame, y, sample_weight=weights)

        # The dispersion denominators are the two likelihood sizes, and their
        # ratio is what separates the fitted phi.
        assert prior.result.phi != pytest.approx(frequency.result.phi, rel=1e-3)
        assert prior._reml_lambdas["x"] != pytest.approx(frequency._reml_lambdas["x"], rel=1e-3)

    def test_learned_knots_follow_weight_mass_only_under_the_frequency_contract(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, 1.0, 300) ** 2
        frame = pd.DataFrame({"x": x})
        y = np.exp(0.5 + 1.2 * x) * rng.gamma(4.0, 0.25, 300)
        weights = np.where(x > 0.5, 8.0, 0.4)

        placed = {}
        for semantics in ("prior", "frequency"):
            model = SuperGLM(
                family=Gamma(),
                features={"x": Spline(n_knots=6, knot_strategy="quantile_rows")},
                weight_semantics=semantics,
            )
            model.fit(frame, y, sample_weight=weights)
            placed[semantics] = np.asarray(model._specs["x"]._knots, dtype=np.float64)

        assert not np.array_equal(placed["prior"], placed["frequency"])
        # Mass sits above 0.5, so only the frequency arm crowds its interior
        # knots there; the prior arm follows the physical rows, which do not.
        interior_prior = placed["prior"][4:10]
        interior_frequency = placed["frequency"][4:10]
        assert float(np.median(interior_frequency)) > 0.5
        assert float(np.median(interior_prior)) < 0.5


class TestZeroWeights:
    def test_a_zero_prior_weight_leaves_the_likelihood_size(self):
        weights = np.array([1.0, 2.5, 0.0, 0.0, 3.0])
        assert dispersion_likelihood_size(weights, weight_semantics="prior") == 3.0
        assert dispersion_likelihood_size(weights, weight_semantics="frequency") == 6.5
        assert pearson_residual_degrees_of_freedom(
            weights, 1.0, weight_semantics="prior"
        ) == pytest.approx(2.0)
        assert pearson_residual_degrees_of_freedom(
            weights, 1.0, weight_semantics="frequency"
        ) == pytest.approx(5.5)

    def test_a_zero_weight_row_drops_out_of_a_prior_fit(self):
        rng, frame = _frame(seed=12, n=150)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 3.0, len(frame))
        weights[:20] = 0.0

        kept = _fit(Gamma(), frame, y, weights, "prior", features={"g": Categorical()})
        filtered = _fit(
            Gamma(),
            frame.iloc[20:].reset_index(drop=True),
            y[20:],
            weights[20:],
            "prior",
            features={"g": Categorical()},
        )
        assert kept.result.phi == pytest.approx(filtered.result.phi, rel=1e-8)

    def test_tweedie_refuses_a_zero_weight_only_where_its_density_needs_to(self):
        rng, frame = _frame(seed=13, n=90)
        y = _response(rng, frame, Tweedie(p=1.5))
        weights = rng.uniform(0.5, 3.0, len(frame))
        weights[3] = 0.0

        with pytest.raises(ValueError, match="strictly positive"):
            _fit(Tweedie(p=1.5), frame, y, weights, "prior", features={"g": Categorical()})

        admitted = _fit(
            Tweedie(p=1.5), frame, y, weights, "frequency", features={"g": Categorical()}
        )
        assert np.isfinite(admitted.result.deviance)


class TestNegativeBinomialThetaProfile:
    """The theta profile is a likelihood too, so it follows the same contract.

    Under ``"prior"`` the construction is ``w Y ~ NB2(w mu, w theta)``, which
    changes the profile score's digamma pair to ``psi(w(y+theta)) -
    psi(w theta)`` and leaves every other term identical.  The two invariants
    below are what pin that: unit weights must not distinguish the contracts,
    and integer weights under ``"frequency"`` must reproduce replication.
    """

    def _theta(self, semantics, frame, response, weights):
        from superglm import Numeric

        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"x": Numeric()},
            weight_semantics=semantics,
        )
        return model.estimate_theta(frame, response, sample_weight=weights)

    def _counts(self, seed=4, n=250):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x": rng.normal(size=n)})
        mu = np.exp(0.3 + 0.4 * frame["x"].to_numpy())
        y = rng.negative_binomial(1.5, 1.5 / (1.5 + mu)).astype(float)
        return rng, frame, y

    def test_unit_weights_give_one_theta(self):
        _, frame, y = self._counts()
        ones = np.ones(len(frame))
        prior = self._theta("prior", frame, y, ones)
        frequency = self._theta("frequency", frame, y, ones)
        assert prior.theta_hat == frequency.theta_hat
        assert prior.nll == pytest.approx(frequency.nll, rel=1e-12)

    def test_integer_frequency_weights_reproduce_the_replicated_theta(self):
        rng, frame, y = self._counts()
        counts = rng.integers(1, 4, len(frame))
        compressed = self._theta("frequency", frame, y, counts.astype(float))
        expanded_frame = frame.loc[frame.index.repeat(counts)].reset_index(drop=True)
        expanded = self._theta("frequency", expanded_frame, np.repeat(y, counts), None)
        assert compressed.theta_hat == pytest.approx(expanded.theta_hat, rel=1e-9)

    def test_the_prior_score_differentiates_the_prior_likelihood(self):
        """The analytic score is checked against its own objective, not a number."""
        from superglm.profiling.nb import _nb2_nll, _theta_profile_score

        rng = np.random.default_rng(5)
        n = 60
        mu = np.exp(0.3 + rng.normal(0.0, 0.4, n))
        y = rng.poisson(mu).astype(float)
        w = rng.uniform(0.4, 5.0, n)
        size = float(np.count_nonzero(w > 0.0))

        for theta in (0.5, 3.0, 25.0):
            step = theta * 1e-6
            lower = _nb2_nll(y, mu, w, theta - step, weight_semantics="prior") * size
            upper = _nb2_nll(y, mu, w, theta + step, weight_semantics="prior") * size
            difference = -(upper - lower) / (2.0 * step)
            analytic = _theta_profile_score(y, mu, w, theta, weight_semantics="prior")
            assert analytic == pytest.approx(difference, rel=1e-6)

    def test_a_zero_prior_weight_deletes_its_row_from_the_score(self):
        """A row observed with infinite variance leaves; it does not contribute 0.

        Both digamma arguments sit on the psi pole at ``w == 0``, and
        ``0 * (-inf + inf)`` is ``nan`` rather than nothing, so the row has to
        go before the score is formed.  Zero non-Tweedie weights are admitted
        by validation, which makes this reachable from a plain fit.
        """
        from superglm.profiling.nb import _theta_profile_score

        rng = np.random.default_rng(21)
        n = 50
        mu = np.exp(0.3 + rng.normal(0.0, 0.4, n))
        y = rng.poisson(mu).astype(float)
        w = rng.uniform(0.4, 5.0, n)
        w[[3, 17, 40]] = 0.0
        carried = w > 0.0

        # Spans both the direct and the asymptotic branch: the switch is on
        # w * theta, so the dropped rows must not pin it to the direct one.
        #
        # The two arms are equal for different reasons, and the test says so.
        # The prior arm drops the rows before summing, so both sides reduce the
        # identical array and agree bitwise.  The frequency arm keeps its
        # summation over every row verbatim -- that is what stops shipped
        # numbers drifting -- so its zero-weight rows contribute an exact 0.0
        # at a different point in the pairwise reduction, and row deletion is
        # exact only up to that association.  Bound measured over 40 seeds and
        # four thetas: worst 1.08e-14.
        for theta in (0.5, 3.0, 25.0, 1e9):
            prior_full = _theta_profile_score(y, mu, w, theta, weight_semantics="prior")
            prior_deleted = _theta_profile_score(
                y[carried], mu[carried], w[carried], theta, weight_semantics="prior"
            )
            assert np.isfinite(prior_full)
            assert prior_full == prior_deleted

            frequency_full = _theta_profile_score(y, mu, w, theta, weight_semantics="frequency")
            frequency_deleted = _theta_profile_score(
                y[carried], mu[carried], w[carried], theta, weight_semantics="frequency"
            )
            assert frequency_full == pytest.approx(frequency_deleted, rel=1e-12)

    def test_a_zero_prior_weight_still_estimates_theta(self):
        rng, frame, y = self._counts()
        weights = rng.uniform(0.5, 3.0, len(frame))
        weights[:4] = 0.0
        carried = weights > 0.0

        estimated = self._theta("prior", frame, y, weights)
        deleted = self._theta(
            "prior",
            frame.loc[carried].reset_index(drop=True),
            y[carried],
            weights[carried],
        )
        assert np.isfinite(estimated.theta_hat)
        assert estimated.theta_hat == pytest.approx(deleted.theta_hat, rel=1e-9)

    def test_fit_reml_refines_theta_under_the_declared_contract(self):
        """The joint alternation must not re-estimate on the other likelihood.

        ``fit_reml`` re-solves theta at the REML fit; reading the contract from
        the parameter default there would overwrite a prior-contract estimate
        with a frequency-contract one, on the default path.

        The spline is load-bearing: with no penalised term ``fit_reml`` finds
        no REML-eligible groups, falls back to ``fit`` and leaves
        ``_reml_result`` unset, so the refinement returns at its first guard
        and never reaches the code this pins.
        """
        rng, frame, y = self._counts(seed=9, n=300)
        weights = rng.uniform(0.4, 5.0, len(frame))

        fitted = {}
        for semantics in ("prior", "frequency"):
            model = SuperGLM(
                family=NegativeBinomial(theta="auto"),
                features={"x": Spline(n_knots=6)},
                weight_semantics=semantics,
            )
            model.fit_reml(frame, y, sample_weight=weights)
            assert model._reml_result is not None
            fitted[semantics] = model

        for semantics, model in fitted.items():
            assert model._nb_profile_result._weight_semantics == semantics
        # Two different likelihoods reach two different optima; equality here
        # would mean one arm silently ran the other's score.
        assert fitted["prior"]._nb_profile_result.theta_hat != pytest.approx(
            fitted["frequency"]._nb_profile_result.theta_hat, rel=1e-6
        )

    def test_a_copied_profile_keeps_its_contract(self):
        """A deepcopy must not hand back a handle that evaluates another likelihood."""
        import copy

        rng, frame, y = self._counts(seed=12)
        weights = rng.uniform(0.4, 4.0, len(frame))
        prior = self._theta("prior", frame, y, weights)

        assert prior._weight_semantics == "prior"
        assert copy.deepcopy(prior)._weight_semantics == "prior"
        assert prior._detached_public_copy()._weight_semantics == "prior"
        # The interval reads the same likelihood the estimate came from.
        assert copy.deepcopy(prior).ci() == pytest.approx(prior.ci(), rel=1e-12)

    def test_the_two_score_arms_are_one_expression_at_unit_weight(self):
        from superglm.profiling.nb import _theta_profile_score

        rng = np.random.default_rng(6)
        n = 40
        mu = np.exp(rng.normal(0.0, 0.3, n))
        y = rng.poisson(mu).astype(float)
        ones = np.ones(n)
        for theta in (0.3, 2.0, 40.0):
            assert _theta_profile_score(
                y, mu, ones, theta, weight_semantics="prior"
            ) == _theta_profile_score(y, mu, ones, theta, weight_semantics="frequency")


class TestScaleProfilerInternals:
    def test_the_frequency_tweedie_saturated_arm_matches_row_replication(self):
        """A replicated row contributes the unit-weight density, ``w`` times.

        Agreement is exact at ``p = 1.5``, where the profiler evaluates a
        closed-form Bessel reduction.  At other powers the Dunn-Smyth series
        packs rows into shared term buffers, so a row's value depends on which
        rows share its batch: the same effect reproduces with the density
        evaluator alone, off any weight-contract code, and it bounds agreement
        here at the measured 1e-10 rather than at round-off.
        """
        rng = np.random.default_rng(14)
        y = np.where(rng.random(40) < 0.35, 0.0, rng.gamma(2.0, 1.5, 40))
        counts = rng.integers(1, 4, 40)
        weights = counts.astype(float)
        replicated = np.repeat(y, counts)

        for power, tolerance in ((1.5, 1e-14), (1.3, 1e-10), (1.7, 1e-10)):
            frequency = prepare_tweedie_reml_scale_data(
                y, weights, power, weight_semantics="frequency"
            )
            expanded = prepare_tweedie_reml_scale_data(
                replicated, np.ones_like(replicated), power, weight_semantics="prior"
            )
            assert frequency.positive_size == pytest.approx(expanded.positive_size)
            for phi in (0.5, 1.0, 3.0):
                assert frequency.saturated_log_likelihood(phi) == pytest.approx(
                    expanded.saturated_log_likelihood(phi), rel=tolerance
                )

    def test_the_vectorized_gamma_shape_helpers_match_their_scalar_forms(self):
        """The prior arm evaluates per row what the frequency arm evaluates once.

        The two must be the same function or the contracts would disagree at
        ``w == 1`` for arithmetic reasons rather than modelling ones.
        """
        probe = np.array([1e-12, 1e-6, 1e-5, 1e-4, 1e-3, 0.5, 3.0, 99.0, 100.0, 1e3, 1e6, 1e12])
        scored = _shape_times_log_minus_digamma_array(probe)
        normalized = _gamma_saturated_normalizer_array(probe)
        curved = _scaled_trigamma_minus_inverse_array(probe)
        for index, argument in enumerate(probe):
            assert scored[index] == pytest.approx(
                _shape_times_log_minus_digamma(float(argument)), rel=1e-13
            )
            assert normalized[index] == pytest.approx(
                _gamma_saturated_normalizer(float(argument)), rel=1e-13
            )
            if 1e-4 <= argument < 100.0:
                assert curved[index] == pytest.approx(
                    argument**2 * _trigamma_minus_inverse(float(argument)), rel=1e-13
                )


class TestTweediePowerProfileRefusesWhatItCannotHonour:
    """The power profile is prior-weight only, and says so rather than guessing.

    Its objective evaluates the compound-Poisson density with the weight inside
    the normalizer.  The replication contract would need the unit-weight density
    counted ``w`` times through the phi cache, its analytic score and the
    exact-Newton polish -- a different objective rather than a rescaled one --
    so the combination is refused.  It was unreachable before ``weight_semantics``
    existed, because Tweedie always read prior weights.
    """

    def _problem(self):
        from superglm import Numeric, families

        rng = np.random.default_rng(11)
        n = 200
        frame = pd.DataFrame({"x": rng.normal(size=n)})
        mu = np.exp(0.5 + 0.3 * frame["x"].to_numpy())
        y = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2.0, mu / 2.0))
        return frame, y, rng.uniform(0.5, 3.0, n), families.tweedie(p=1.5), Numeric()

    def test_non_unit_replication_weights_are_refused_by_name(self):
        frame, y, weights, family, spec = self._problem()
        model = SuperGLM(family=family, features={"x": spec}, weight_semantics="frequency")
        with pytest.raises(ValueError, match=r'weight_semantics="frequency"'):
            model.estimate_p(frame, y, sample_weight=weights)

    def test_unit_weights_are_admitted_under_either_contract(self):
        frame, y, _, family, spec = self._problem()
        ones = np.ones(len(frame))
        powers = []
        for semantics in ("prior", "frequency"):
            model = SuperGLM(family=family, features={"x": spec}, weight_semantics=semantics)
            powers.append(model.estimate_p(frame, y, sample_weight=ones).p_hat)
        assert powers[0] == powers[1]


class TestDeclaration:
    def test_an_unrecognised_contract_names_both_options(self):
        with pytest.raises(ValueError, match="'prior' or 'frequency'"):
            SuperGLM(family=Gamma(), weight_semantics="analytic")

    def test_the_resolved_contract_travels_with_the_configuration(self):
        rng, frame = _frame(seed=15, n=80)
        y = _response(rng, frame, Gamma())
        model = SuperGLM(
            family=Gamma(), features={"g": Categorical()}, weight_semantics="frequency"
        )
        model.fit(frame, y)

        from superglm.model.fit_state import ModelConfig

        config = ModelConfig.capture(model)
        assert config.weight_semantics == "frequency"
        assert config.constructor_kwargs()["weight_semantics"] == "frequency"
        assert config.materialize(SuperGLM)._weight_semantics == "frequency"

    @pytest.mark.parametrize(
        ("family", "restored"),
        [
            pytest.param("gamma", "frequency", id="gamma_was_frequency"),
            pytest.param(Tweedie(p=1.5), "prior", id="tweedie_was_prior"),
        ],
    )
    def test_a_configuration_pickled_before_the_field_keeps_its_family_rule(self, family, restored):
        """Restoring a recorded fit must reproduce the likelihood it was fitted
        under, not adopt the new default."""
        rng, frame = _frame(seed=16, n=80)
        y = _response(rng, frame, Gamma() if family == "gamma" else Tweedie(p=1.5))
        model = SuperGLM(family=family, features={"g": Categorical()})
        model.fit(frame, y)

        from superglm.model.fit_state import ModelConfig

        state = ModelConfig.capture(model).__dict__.copy()
        state.pop("weight_semantics")
        revived = ModelConfig.__new__(ModelConfig)
        revived.__setstate__(state)
        assert revived.weight_semantics == restored


class TestTheContractReachesTheDerivedReports:
    """Seams that read the contract second-hand, and got it wrong.

    Each of these produced a number under one contract while the fit ran under
    the other, in a place no test crossed.
    """

    def test_a_zero_prior_weight_leaves_the_vuong_moments(self):
        """The statistic must not move when an unobserved row is added.

        ``likelihood_size`` already counts only the carried rows; a zero-weight
        row whose density is exactly 0 under both models still fed an
        artificial zero to the mean and the variance, which can flip the
        selected model.
        """
        from superglm.stats.model_tests import vuong_test

        rng, frame = _frame(seed=31, n=200)
        y = _response(rng, frame, Poisson())
        weights = rng.uniform(0.5, 3.0, len(frame))

        def fitted(feature):
            model = SuperGLM(
                family=Poisson(),
                features=feature,
                weight_semantics="prior",
            )
            model.fit(frame, y, sample_weight=weights)
            return model

        a = fitted({"x": Spline(n_knots=5)})
        b = fitted({"g": Categorical()})
        without = vuong_test(a, b, frame, y, sample_weight=weights)

        padded_frame = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
        padded_y = np.concatenate([y, y[:5]])
        padded_weights = np.concatenate([weights, np.zeros(5)])
        with_ignored = vuong_test(a, b, padded_frame, padded_y, sample_weight=padded_weights)

        assert with_ignored.statistic == pytest.approx(without.statistic, rel=1e-12)

    def test_a_released_fit_keeps_its_likelihood_size(self):
        """``retain_fit_state=False`` drops the weights, not the contract.

        Standing in all-ones reported ``n`` for a prior fit carrying zero
        weights, which moves BIC and AICc and drops the summary's contract row.
        """
        rng, frame = _frame(seed=32, n=120)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 3.0, len(frame))
        weights[:8] = 0.0

        retained = SuperGLM(
            family=Gamma(), features={"x": Spline(n_knots=5)}, weight_semantics="prior"
        )
        retained.fit(frame, y, sample_weight=weights)
        released = SuperGLM(
            family=Gamma(),
            features={"x": Spline(n_knots=5)},
            weight_semantics="prior",
            retain_fit_state=False,
        )
        released.fit(frame, y, sample_weight=weights)

        assert released._fit_weights is None
        retained_ic = retained.summary()["information_criteria"]
        released_ic = released.summary()["information_criteria"]
        for field in ("bic", "aicc"):
            assert released_ic[field] == pytest.approx(retained_ic[field], rel=1e-12), field
        # The rendered row naming the contract survives the release too.
        assert "prior weights" in str(released.summary())

    def test_a_hosted_piecewise_boundary_ignores_a_zero_weight_row(self):
        """The physical-rows rule reaches hosted geometry or it is not a rule.

        The builder stamped a boolean and the host passed ``sample_weight=None``,
        which counts zero-weight rows -- so a row that shaped no top-level
        boundary still shaped a hosted one.
        """
        from superglm import OrderedCategorical, Piecewise

        rng = np.random.default_rng(33)
        n = 300
        levels = np.arange(1, 11)
        x = rng.choice(levels, n)
        frame = pd.DataFrame({"x": x})
        y = np.exp(0.4 + 0.1 * x) * rng.gamma(5.0, 0.2, n)
        weights = rng.uniform(0.5, 3.0, n)

        # Rows carrying no weight, parked at one extreme of the axis.
        extreme = x >= 9
        weights[extreme] = 0.0

        def knots(frame_in, y_in, sample_weight):
            model = SuperGLM(
                family=Gamma(),
                features={"x": OrderedCategorical(order=list(levels), basis=Piecewise(3))},
                weight_semantics="prior",
            )
            model.fit(frame_in, y_in, sample_weight=sample_weight)
            return np.asarray(model._specs["x"]._basis_spline._knots, dtype=np.float64)

        # A zero-weight row must shape the hosted boundary exactly as much as
        # deleting the row does -- which is not at all.
        with_ignored = knots(frame, y, weights)
        deleted = knots(
            frame.loc[~extreme].reset_index(drop=True),
            y[~extreme],
            weights[~extreme],
        )
        np.testing.assert_array_equal(with_ignored, deleted)


class TestTheCountingLatticeIsDeclared:
    """The prior construction for counting families has a support.

    ``w Y ~ Poisson(w mu)`` and ``w Y ~ NB2(w mu, w theta)`` live on the
    non-negative integers.  Off that lattice ``gammaln`` interpolates the
    counting density, so the reported likelihood is a quasi-likelihood -- said
    out loud rather than reported as an exact density.
    """

    def _counts(self, seed=40, n=200):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        counts = rng.poisson(np.exp(0.5 + 1.2 * frame["x"].to_numpy())).astype(float)
        exposure = rng.uniform(0.5, 4.0, n)
        return frame, counts, exposure

    def _fit_warnings(self, frame, y, weights, family, semantics):
        import warnings

        from superglm import Numeric
        from superglm.model.input_validation import PriorWeightLatticeWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SuperGLM(family=family, features={"x": Numeric()}, weight_semantics=semantics).fit(
                frame, y, sample_weight=weights
            )
        return [w for w in caught if issubclass(w.category, PriorWeightLatticeWarning)]

    def test_the_canonical_weighting_is_on_lattice_and_silent(self):
        """``y = count / exposure`` weighted by ``exposure`` recovers the count."""
        frame, counts, exposure = self._counts()
        assert self._fit_warnings(frame, counts / exposure, exposure, Poisson(), "prior") == []

    def test_unit_weights_are_silent(self):
        frame, counts, _ = self._counts()
        ones = np.ones(len(frame))
        assert self._fit_warnings(frame, counts, ones, Poisson(), "prior") == []

    def test_the_frequency_contract_never_scales_the_lattice(self):
        """A replication count multiplies the density; it does not move the support."""
        frame, counts, exposure = self._counts()
        assert self._fit_warnings(frame, counts, exposure, Poisson(), "frequency") == []

    @pytest.mark.parametrize(
        ("family", "mentions_theta"),
        [
            pytest.param(Poisson(), False, id="poisson"),
            pytest.param(NegativeBinomial(theta=1.5), True, id="negative_binomial"),
        ],
    )
    def test_an_off_lattice_prior_weight_says_so(self, family, mentions_theta):
        frame, counts, exposure = self._counts()
        caught = self._fit_warnings(frame, counts, exposure, family, "prior")
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "quasi-likelihood" in message
        # The NB interpolated factor is theta-dependent, so it reaches the
        # estimate itself and not only the reported likelihood.
        # "theta" rather than "theta_hat": at a fixed theta this call estimates
        # nothing, so naming an estimate would be false. The theta-DEPENDENCE
        # of the interpolated factor is what this asserts, and it still holds.
        assert ("theta" in message) is mentions_theta

    def test_a_released_fit_reports_exactly_what_a_retained_one_does(self):
        """Releasing the rows must not change a single published number.

        ``retain_fit_state=False`` drops the weights, and every quantity keyed
        on the likelihood size then had to fall back: BIC and AICc through
        ``dispersion_likelihood_size``, the smooth terms' Wald reference
        through ``_wood_residual_df``, and the contract row itself. Comparing
        the whole rendered summary covers all of them at once, including any
        added later.
        """
        rng, frame = _frame(seed=41, n=140)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 3.0, len(frame))
        weights[:9] = 0.0

        def summarised(retain):
            model = SuperGLM(
                family=Gamma(),
                features={"x": Spline(n_knots=5)},
                weight_semantics="prior",
                retain_fit_state=retain,
            )
            model.fit(frame, y, sample_weight=weights)
            return model.summary()

        retained, released = summarised(True), summarised(False)

        # The smooth's Wald p-value is the sharp one: it reads the residual
        # degrees of freedom directly, and the rendered table rounds to three
        # decimals where the whole difference hides. Measured without the fix,
        # 2.739e-06 against 2.408e-06.
        retained_p = [r.wald_p for r in retained._coef_rows if r.wald_p is not None]
        released_p = [r.wald_p for r in released._coef_rows if r.wald_p is not None]
        assert retained_p, "fixture must produce at least one Wald p-value"
        assert released_p == retained_p

        # And nothing else in the report moved either.
        assert str(released) == str(retained)


class TestTheContractReachesTheSmoothingParameters:
    """Seams where the contract changes lambda or the published dispersion.

    The Fellner-Schall dispersion denominators in ``reml/efs.py`` and
    ``reml/runner.py`` are corrected in the same change but are NOT pinned
    here: ``optimize_efs_reml`` runs only when ``use_direct`` is false, which
    needs ``lam1 > 0``, and ``fit_reml()`` refuses selection penalties
    outright -- so no public call reaches it. The correction stands on the
    arithmetic (a dispersion denominator must be the contract's likelihood
    size, exactly as everywhere else) rather than on a regression.
    """

    def test_reml_replication_identity_holds_where_lambda_is_identified(self):
        """A replication count is a repeated row, all the way to lambda.

        Measured across five seeds: where lambda is identified (7.7 to 48.6)
        the compressed and expanded fits agree to 1.7e-14 to 3.3e-14. Two
        seeds land at lambda ~3.5e6, effectively infinite smoothing on a flat
        objective, where the same identity holds only to 7.3e-6 -- so the
        fixture is chosen to sit in the identified regime, and ``phi``, which
        agrees to 4.5e-14 in every case including the boundary ones, is
        asserted alongside it.
        """
        rng = np.random.default_rng(50)
        n = 200
        x = rng.uniform(0.0, 1.0, n)
        frame = pd.DataFrame({"x": x})
        y = rng.gamma(5.0, np.exp(0.4 + 1.1 * x) / 5.0)
        counts = rng.integers(1, 4, n)

        compressed = SuperGLM(
            family=Gamma(),
            features={"x": Spline(n_knots=6)},
            weight_semantics="frequency",
        )
        compressed.fit_reml(frame, y, sample_weight=counts.astype(float))
        assert compressed._reml_lambdas["x"] < 1e4, "fixture drifted to the lambda boundary"

        expanded = SuperGLM(family=Gamma(), features={"x": Spline(n_knots=6)})
        expanded.fit_reml(
            frame.loc[frame.index.repeat(counts)].reset_index(drop=True),
            np.repeat(y, counts),
            sample_weight=None,
        )
        assert compressed._reml_lambdas["x"] == pytest.approx(
            expanded._reml_lambdas["x"], rel=1e-11
        )
        assert compressed.result.phi == pytest.approx(expanded.result.phi, rel=1e-11)

    def test_an_unfitted_legacy_tweedie_pickle_restores_prior_semantics(self):
        """``_distribution`` is only set once fitted, so the migration has to
        consult the configured family or it flips Tweedie to replication."""
        from superglm.solvers.dispersion import model_weight_semantics

        for family, expected in ((Tweedie(p=1.5), "prior"), (Gamma(), "frequency")):
            model = SuperGLM(family=family, features={"x": Spline(n_knots=4)})
            state = model.__dict__.copy()
            state.pop("_weight_semantics", None)
            revived = SuperGLM.__new__(SuperGLM)
            revived.__dict__.update(state)
            assert getattr(revived, "_distribution", None) is None
            assert model_weight_semantics(revived) == expected


class TestTheBinomialPriorFormIsNotNormalised:
    """``w Y ~ Binomial(w, mu)`` needs y on ``{0, 1/w, ..., 1}``.

    ``validate_response`` pins y to ``{0, 1}``, so only the all-failure and
    all-success outcomes are reachable and their masses sum to one only at
    ``w == 1``.  The binomial coefficient is exactly 1 at both endpoints, which
    makes each term look exact on its own -- that is what makes this easy to
    miss, and it is not the same thing as a distribution.
    """

    def test_the_two_reachable_masses_only_normalise_at_unit_weight(self):
        """The measurement the warning exists for, stated independently."""
        mu = 0.4
        assert stats.binom.pmf(1, 1, mu) + stats.binom.pmf(0, 1, mu) == pytest.approx(1.0)
        for w in (2, 3):
            total = stats.binom.pmf(w, w, mu) + stats.binom.pmf(0, w, mu)
            assert total < 0.99
        assert stats.binom.pmf(3, 3, mu) + stats.binom.pmf(0, 3, mu) == pytest.approx(0.28)

    @pytest.mark.parametrize(("weights_are_unit", "warns"), [(True, False), (False, True)])
    def test_a_non_unit_prior_weight_says_so(self, weights_are_unit, warns):
        import warnings as warnings_module

        from superglm.model.input_validation import PriorWeightLatticeWarning

        rng, frame = _frame(seed=51, n=120)
        y = _response(rng, frame, Binomial())
        weights = np.ones(len(frame)) if weights_are_unit else rng.uniform(1.5, 4.0, len(frame))

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            SuperGLM(
                family=Binomial(),
                features={"x": Spline(n_knots=4)},
                weight_semantics="prior",
            ).fit(frame, y, sample_weight=weights)
        hits = [c for c in caught if issubclass(c.category, PriorWeightLatticeWarning)]
        assert bool(hits) is warns
        if warns:
            assert "do not sum to one" in str(hits[0].message)

    def test_the_frequency_contract_is_silent(self):
        import warnings as warnings_module

        from superglm.model.input_validation import PriorWeightLatticeWarning

        rng, frame = _frame(seed=52, n=120)
        y = _response(rng, frame, Binomial())
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            SuperGLM(
                family=Binomial(),
                features={"x": Spline(n_knots=4)},
                weight_semantics="frequency",
            ).fit(frame, y, sample_weight=rng.integers(1, 4, len(frame)).astype(float))
        assert [c for c in caught if issubclass(c.category, PriorWeightLatticeWarning)] == []

    def test_the_residual_stays_contract_invariant_so_the_pit_stays_uniform(self):
        """The residual cannot invert a likelihood that is not normalised.

        A randomized quantile residual is a probability-integral transform, so
        it is only valid if the masses it splits sum to one. The prior form's
        two reachable masses do not, which leaves exactly one coherent choice:
        keep the transform uniform and say plainly that it inverts the
        unit-weight marginal rather than the reported likelihood.

        Splitting at ``1 - mu**w`` instead -- the PIT of the reported
        likelihood -- makes the transform non-uniform for a CORRECTLY
        specified fit, because the intermediate counts ``1..w-1`` are
        unobservable under ``y in {0, 1}``. Measured: kstest p goes from 0.82
        to 0.0 and the residual s.d. from 1.01 to 1.66.
        """

        rng = np.random.default_rng(53)
        n = 4000
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        linear = 0.5 + 0.8 * frame["x"].to_numpy()
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear))).astype(float)
        weights = np.full(n, 3.0)

        model = SuperGLM(
            family=Binomial(),
            features={"x": Spline(n_knots=4)},
            weight_semantics="prior",
        )
        with pytest.warns(UserWarning):
            model.fit(frame, y, sample_weight=weights)
        metrics = model.metrics(frame, y, sample_weight=weights)
        mu = np.asarray(model.predict(frame), dtype=np.float64)

        # The masses the reported likelihood assigns do not sum to one, which
        # is why no PIT of it exists to build.
        total = np.power(1.0 - mu, weights) + np.power(mu, weights)
        assert np.all(total < 0.99)

        # The residual is a valid transform, and is the same under both
        # contracts -- Binomial is deliberately contract-invariant here.
        residuals = metrics.residuals("quantile", seed=3)
        assert stats.kstest(residuals, "norm").pvalue > 0.01

        frequency = SuperGLM(
            family=Binomial(),
            features={"x": Spline(n_knots=4)},
            weight_semantics="frequency",
        )
        frequency.fit(frame, y, sample_weight=weights)
        np.testing.assert_array_equal(
            frequency.metrics(frame, y, sample_weight=weights).residuals("quantile", seed=3),
            residuals,
        )


class TestTheCustomFamilyScaleTermUsesTheContract:
    """The `0.5 * (n - M_p) * log(D)` fallback, and the Hessian that mirrors it.

    Estimated-scale families with no exact saturated-likelihood profiler fall
    back to a Gaussian-shaped scale term.  Its ``n`` is the declared contract's
    likelihood size like every other dispersion denominator, not the physical
    row count -- and the objective, its gradient and its Hessian must all agree
    on that ``n`` or the Newton step is inconsistent with its own surface.

    Built on the Wood (2011) oracle harness so the objective is exercised
    directly; a full custom-family REML fit is not needed to pin the term.
    """

    def _objective(self, weights, semantics):
        import warnings
        from types import SimpleNamespace

        from superglm.links import IdentityLink
        from superglm.reml.objective import reml_laml_objective
        from superglm.solvers.pirls import PIRLSResult

        from ._wood_reml_oracles import solve_gaussian_state

        x = np.linspace(-1.4, 1.2, 11)
        design = np.column_stack((x, x**2 - np.mean(x**2), np.sin(1.7 * x)))
        y = 0.8 + 1.1 * x - 0.6 * x**2 + 0.25 * np.cos(2.3 * x)
        slope_penalty = np.diag([2.5, 0.0, 0.0])
        state = solve_gaussian_state(design, y, slope_penalty)
        result = PIRLSResult(
            beta=state.beta.copy(),
            intercept=state.intercept,
            n_iter=1,
            deviance=state.deviance,
            converged=True,
            phi=1.0,
            effective_df=0.0,
        )

        class _NoMatvecDesign:
            group_matrices: list = []

            def matvec(self, beta):  # pragma: no cover - must not be reached
                raise AssertionError("cached objective path must not expand the design")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = reml_laml_objective(
                _NoMatvecDesign(),
                SimpleNamespace(scale_known=False),
                IdentityLink(),
                [],
                y,
                result,
                {},
                weights,
                np.zeros_like(y),
                XtWX=state.slope_xtwx,
                XtW1=state.full_hessian[1:, 0],
                sum_W=float(state.full_hessian[0, 0]),
                S_override=slope_penalty,
                weight_semantics=semantics,
            )
        return value, y, state, result

    def test_the_scale_term_counts_replication_not_rows(self):
        """Doubling every frequency weight doubles the term's ``n``.

        The two contracts see the same weights here, so the only thing that can
        move the objective is the size each reads out of them: ``sum(w) = 2n``
        under ``"frequency"`` against the ``n`` positive rows under
        ``"prior"``. The gap is therefore exactly one scale term's worth.
        """
        n = 11
        doubled = np.full(n, 2.0)
        frequency, y, state, result = self._objective(doubled, "frequency")
        prior, _, _, _ = self._objective(doubled, "prior")

        penalized_deviance = float(result.deviance + state.penalty_quad)
        expected_gap = 0.5 * (2.0 * n - float(n)) * np.log(penalized_deviance)
        assert frequency - prior == pytest.approx(expected_gap, rel=1e-10)

    def test_unit_weights_leave_the_term_alone(self):
        """The regression guard: at w == 1 both sizes are n, so nothing moves."""
        ones = np.ones(11)
        frequency, _, _, _ = self._objective(ones, "frequency")
        prior, _, _, _ = self._objective(ones, "prior")
        assert frequency == prior

    def test_the_statsmodels_comparison_picks_the_matching_weight_argument(self):
        """The oracle utility must compare against the contract, not the family.

        `var_weights` IS the prior reading and `freq_weights` IS the frequency
        one; choosing by Tweedie-ness was the pre-`weight_semantics` rule, and
        it compares a prior-contract Gamma fit against the wrong statsmodels
        model -- the exact mismatch the comparison exists to detect.
        """
        pytest.importorskip("statsmodels")
        import superglm.debug_weights as debug_weights

        captured = {}

        class _StubGLM:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def fit(self, *args, **kwargs):
                raise RuntimeError("stop after argument selection")

        rng, frame = _frame(seed=60, n=60)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 3.0, len(frame))

        for semantics, expected in (("prior", "var_weights"), ("frequency", "freq_weights")):
            captured.clear()
            model = SuperGLM(
                family=Gamma(),
                features={"x": Spline(n_knots=4)},
                weight_semantics=semantics,
            )
            model.fit(frame, y, sample_weight=weights)
            import statsmodels.api as sm

            original = sm.GLM
            sm.GLM = _StubGLM
            try:
                debug_weights.compare_irls_weights(model, frame[["x"]], y, sample_weight=weights)
            except Exception:
                pass
            finally:
                sm.GLM = original
            assert expected in captured, f"{semantics} should use {expected}, got {list(captured)}"
            assert ("freq_weights" in captured) is (expected == "freq_weights")


class TestTheContractSurvivesCloningAndRestoring:
    """Seams from the third review round: clones, legacy pickles, boundaries."""

    def test_a_feature_subset_clone_keeps_its_contract(self):
        """`drop1`, term importance and `refit_unpenalised` all clone.

        Falling through to the constructor default turned every
        frequency-contract model into a prior one the moment a term was
        dropped.
        """
        rng, frame = _frame(seed=70, n=200)
        y = _response(rng, frame, Gamma())
        weights = rng.uniform(0.5, 3.0, len(frame))

        for semantics in ("frequency", "prior"):
            model = SuperGLM(
                family=Gamma(),
                features={"x": Spline(n_knots=5), "g": Categorical()},
                weight_semantics=semantics,
            )
            model.fit(frame, y, sample_weight=weights)
            assert model._clone_without_features({"g"})._weight_semantics == semantics

    @pytest.mark.parametrize(
        ("family", "expected"),
        [("poisson", "frequency"), (Tweedie(p=1.5), "prior")],
        ids=["poisson", "tweedie"],
    )
    def test_a_legacy_sklearn_wrapper_restores_and_refits(self, family, expected):
        """Unpickling one from before the field must not raise.

        sklearn reads every constructor parameter off the instance, so a
        missing attribute breaks `get_params` and `clone` as well as refitting.
        """
        from superglm.sklearn import SuperGLMRegressor

        estimator = SuperGLMRegressor(family=family, numeric_features=["x"])
        state = estimator.__dict__.copy()
        state.pop("weight_semantics")
        revived = SuperGLMRegressor.__new__(SuperGLMRegressor)
        revived.__setstate__(state)

        assert revived.get_params()["weight_semantics"] == expected

        rng = np.random.default_rng(71)
        n = 120
        X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.5 + X["x"].to_numpy())).astype(float)
        if not isinstance(family, Tweedie):
            revived.fit(X, y)
            assert revived._model._weight_semantics == expected

    def test_evaluating_off_lattice_holdout_rows_warns(self):
        """The fit-time check does not cover evaluation's own rows.

        A clean fitted model scored on off-lattice holdout rows would publish
        a fabricated log-likelihood and residuals that round the impossible
        count onto a neighbour.
        """
        from superglm import Numeric, PriorWeightLatticeWarning

        rng = np.random.default_rng(72)
        n = 150
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        counts = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        exposure = rng.uniform(0.5, 4.0, n)

        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        model.fit(frame, counts / exposure, sample_weight=exposure)  # on-lattice

        with pytest.warns(PriorWeightLatticeWarning):
            model.metrics(frame, counts, sample_weight=exposure).log_likelihood

    def test_the_warning_says_when_coefficients_move(self):
        """At fixed theta only the criterion moves; at ``theta="auto"`` the
        estimate feeds the variance, so the fit itself moves."""
        import warnings as warnings_module

        from superglm import Numeric, PriorWeightLatticeWarning

        rng = np.random.default_rng(73)
        n = 150
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        counts = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        weights = rng.uniform(0.5, 4.0, n)

        def message(family):
            with warnings_module.catch_warnings(record=True) as caught:
                warnings_module.simplefilter("always")
                SuperGLM(family=family, features={"x": Numeric()}, weight_semantics="prior").fit(
                    frame, counts, sample_weight=weights
                )
            return next(
                str(c.message) for c in caught if issubclass(c.category, PriorWeightLatticeWarning)
            )

        moves = "coefficients, fitted means and deviance move"
        assert moves not in message(Poisson())
        assert moves not in message(NegativeBinomial(theta=1.5))
        assert moves in message(NegativeBinomial(theta="auto"))

    def test_the_lattice_warning_is_publicly_importable(self):
        """The migration note tells users to filter on it by name."""
        import superglm
        from superglm.model import input_validation

        assert superglm.PriorWeightLatticeWarning is input_validation.PriorWeightLatticeWarning
        assert "PriorWeightLatticeWarning" in superglm.__all__


class TestTheContractsAgreeAtUnitWeightNotAtIntegerWeight:
    """The rule the docstrings state, checked against the arithmetic.

    Four places -- including the module that defines the contract and the
    public ``SuperGLM.__init__`` docstring -- said the two readings "agree only
    at integer weights". They do not. This pins the true rule so the wording
    cannot drift back.
    """

    def test_integer_weights_do_not_make_the_contracts_agree(self):
        weights = np.full(100, 2.0)
        prior = dispersion_likelihood_size(weights, weight_semantics="prior")
        frequency = dispersion_likelihood_size(weights, weight_semantics="frequency")
        assert prior == 100.0
        assert frequency == 200.0

        assert pearson_residual_degrees_of_freedom(
            weights, 5.0, weight_semantics="prior"
        ) == pytest.approx(95.0)
        assert pearson_residual_degrees_of_freedom(
            weights, 5.0, weight_semantics="frequency"
        ) == pytest.approx(195.0)

        # And the reported normalizer differs at integer w too.
        y, mu, w = np.array([1.0]), np.array([1.0]), np.array([2.0])
        assert weighted_log_likelihood(
            Poisson(), y, mu, w, 1.0, weight_semantics="prior"
        ) == pytest.approx(-1.30685, abs=1e-5)
        assert weighted_log_likelihood(
            Poisson(), y, mu, w, 1.0, weight_semantics="frequency"
        ) == pytest.approx(-2.0, abs=1e-12)

    def test_the_stated_rule_is_unit_weight(self):
        """Guard on the prose: the old wording must not come back."""
        from pathlib import Path

        import superglm.solvers.dispersion as dispersion_module
        from superglm import SuperGLM as _SuperGLM

        sources = [
            dispersion_module.__doc__ or "",
            _SuperGLM.__init__.__doc__ or "",
        ]
        root = Path(dispersion_module.__file__).resolve().parents[3]
        for relative in (
            "docs/guide/families.md",
            "docs/development/migrations/weight-semantics-prior.md",
        ):
            path = root / relative
            if path.exists():
                sources.append(path.read_text())

        for text in sources:
            lowered = text.lower()
            assert "only at integer" not in lowered
            assert "coincide only at integer" not in lowered

    def test_a_zero_weight_alone_does_not_trigger_the_binomial_warning(self):
        """A binomial fit whose every CARRIED row is w == 1 is exact.

        `prior_weight_log_density` returns exactly 0.0 for a zero-weight
        binomial row, so the reported likelihood is an ordinary Bernoulli one
        and the warning's "not an exact density" would be false about it.
        """
        import warnings as warnings_module

        from superglm import PriorWeightLatticeWarning

        rng = np.random.default_rng(80)
        n = 200
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        linear = 0.5 + 0.8 * frame["x"].to_numpy()
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear))).astype(float)
        weights = np.ones(n)
        weights[:5] = 0.0
        carried = weights > 0.0

        def fitted(frame_in, y_in, w_in):
            model = SuperGLM(
                family=Binomial(),
                features={"x": Spline(n_knots=4)},
                weight_semantics="prior",
            )
            model.fit(frame_in, y_in, sample_weight=w_in)
            return model.metrics(frame_in, y_in, sample_weight=w_in).log_likelihood

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            with_zeros = fitted(frame, y, weights)
        assert [c for c in caught if issubclass(c.category, PriorWeightLatticeWarning)] == []

        deleted = fitted(frame.loc[carried].reset_index(drop=True), y[carried], weights[carried])
        assert with_zeros == pytest.approx(deleted, rel=1e-12)

    def test_the_warning_counts_only_the_carried_rows(self):
        """When it does fire, zero rows must not inflate the figure."""
        import warnings as warnings_module

        from superglm import PriorWeightLatticeWarning

        rng = np.random.default_rng(81)
        n = 200
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        linear = 0.5 + 0.8 * frame["x"].to_numpy()
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear))).astype(float)
        weights = np.full(n, 3.0)
        weights[:5] = 0.0

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            SuperGLM(
                family=Binomial(),
                features={"x": Spline(n_knots=4)},
                weight_semantics="prior",
            ).fit(frame, y, sample_weight=weights)
        message = next(
            str(c.message) for c in caught if issubclass(c.category, PriorWeightLatticeWarning)
        )
        assert message.startswith(f"{n - 5} of {n} rows")


class TestTheLatticeCheckReachesEveryEvaluationBoundary:
    """One fit-time check does not cover code that brings its own rows.

    `ModelMetrics`, the editor's dataset metrics, cross-validation's NLL and
    the Vuong test each take their own response and weights, so an off-lattice
    row can reach an interpolated pseudo-density without any fit ever warning.
    """

    def _counts(self, seed=90, n=150):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        counts = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        exposure = rng.uniform(0.5, 4.0, n)
        return frame, counts, exposure

    def _on_lattice_fit(self, frame, counts, exposure):
        from superglm import Numeric

        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        model.fit(frame, counts / exposure, sample_weight=exposure)
        return model

    def test_evaluation_without_explicit_weights_still_checks(self):
        """Synthesized all-ones weights define a lattice too.

        The earlier boundary check was nested under ``sample_weight is not
        None``, so an unweighted holdout with a fractional response reached the
        interpolated density with no warning at all.
        """
        from superglm import PriorWeightLatticeWarning

        frame, counts, exposure = self._counts()
        model = self._on_lattice_fit(frame, counts, exposure)

        with pytest.warns(PriorWeightLatticeWarning):
            model.metrics(frame, counts / exposure).log_likelihood

    def test_cross_validation_nll_checks_each_validation_slice(self):
        """A custom split can put the off-lattice row only in validation."""
        from superglm import Numeric, PriorWeightLatticeWarning
        from superglm.model_selection import cross_validate

        rng = np.random.default_rng(91)
        n = 160
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        y[-8:] += 0.5  # off-lattice at unit weight, in the last fold only

        class _Folds:
            """The off-lattice rows are VALIDATION-ONLY, never trained on.

            If any fold trains on them the fit itself warns and the test
            passes without the scorer's check ever running -- which is exactly
            how the first version of this test failed to pin anything.
            """

            n_splits = 1

            def split(self, X, y=None, groups=None):
                idx = np.arange(len(X))
                held_out = idx[-8:]
                yield np.setdiff1d(idx, held_out), held_out

        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with pytest.warns(PriorWeightLatticeWarning):
            cross_validate(model, frame, y, cv=_Folds(), scoring="nll")

    def test_the_vuong_test_checks_both_families(self):
        from superglm import Categorical, PriorWeightLatticeWarning
        from superglm.stats.model_tests import vuong_test

        rng, frame = _frame(seed=92, n=200)
        y = _response(rng, frame, Poisson())

        def fitted(features):
            model = SuperGLM(family=Poisson(), features=features, weight_semantics="prior")
            model.fit(frame, y)
            return model

        a = fitted({"x": Spline(n_knots=5)})
        b = fitted({"g": Categorical()})

        off_lattice = y.copy()
        off_lattice[:6] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            vuong_test(a, b, frame, off_lattice)

    def test_editor_dataset_metrics_check_their_own_split(self):
        from superglm import PriorWeightLatticeWarning
        from superglm.editor.metrics import compute_dataset_metrics

        frame, counts, exposure = self._counts(seed=93)
        model = self._on_lattice_fit(frame, counts, exposure)

        from superglm.editor.evaluation import EvaluationDataset

        dataset = EvaluationDataset(
            name="validation",
            label="Validation",
            X=frame,
            y=counts,  # off-lattice: the fit used counts / exposure
            sample_weight=exposure,
        )
        with pytest.warns(PriorWeightLatticeWarning):
            compute_dataset_metrics(model, dataset)


class TestFractionalReplicationCountsAreDeclared:
    """A row cannot appear 0.4 times, so `sum(w) - edf` stops being a count."""

    def _fit(self, semantics, weights):
        from superglm import Numeric

        rng = np.random.default_rng(94)
        n = 150
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics=semantics)
        model.fit(frame, y, sample_weight=weights(n, rng))
        return model

    def test_a_fractional_frequency_weight_says_so(self):
        from superglm import FractionalFrequencyWeightWarning

        with pytest.warns(FractionalFrequencyWeightWarning, match="quasi-likelihood"):
            self._fit("frequency", lambda n, rng: rng.uniform(0.5, 4.0, n))

    def test_integer_replication_counts_are_silent(self):
        import warnings as warnings_module

        from superglm import FractionalFrequencyWeightWarning

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            self._fit("frequency", lambda n, rng: rng.integers(1, 4, n).astype(float))
        assert [c for c in caught if issubclass(c.category, FractionalFrequencyWeightWarning)] == []

    def test_the_prior_contract_is_silent_at_fractional_weights(self):
        """Fractional weights are exactly what the prior reading is for."""
        import warnings as warnings_module

        from superglm import FractionalFrequencyWeightWarning

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            self._fit("prior", lambda n, rng: rng.uniform(0.5, 4.0, n))
        assert [c for c in caught if issubclass(c.category, FractionalFrequencyWeightWarning)] == []


class TestACustomFamilyDoesNotWarnAtUnitWeight:
    """The two contracts are the same likelihood at ``w == 1``.

    `prior_weight_log_density` returns None for any family it does not ship,
    so an ordinary unweighted custom-family fit reached the mismatch warning --
    twice, since the null likelihood is computed too -- and failed outright
    under ``-W error``.
    """

    def _custom_family(self):
        class _CustomFamily:
            """Genuinely unshipped: a subclass of a shipped family is matched
            by `isinstance` and never reaches the fallback at all."""

            scale_known = False

            def log_likelihood(self, y, mu, weights, phi):
                resid = np.asarray(y) - np.asarray(mu)
                return float(np.sum(np.asarray(weights) * -0.5 * resid**2 / phi))

            def deviance_unit(self, y, mu):
                return (np.asarray(y) - np.asarray(mu)) ** 2

        return _CustomFamily()

    def test_unit_weights_do_not_warn(self):
        import warnings as warnings_module

        from superglm.distributions import weighted_log_likelihood

        rng = np.random.default_rng(95)
        n = 80
        y = rng.normal(0.0, 1.0, n)
        mu = np.zeros(n)

        for weights in (np.ones(n), np.where(np.arange(n) < 5, 0.0, 1.0)):
            with warnings_module.catch_warnings(record=True) as caught:
                warnings_module.simplefilter("always")
                value = weighted_log_likelihood(
                    self._custom_family(), y, mu, weights, 1.0, weight_semantics="prior"
                )
            assert [c for c in caught if issubclass(c.category, UserWarning)] == []
            assert np.isfinite(value)

    def test_non_unit_weights_still_warn(self):
        from superglm.distributions import weighted_log_likelihood

        rng = np.random.default_rng(96)
        n = 80
        y = rng.normal(0.0, 1.0, n)
        with pytest.warns(UserWarning, match="not a SuperGLM-shipped family"):
            weighted_log_likelihood(
                self._custom_family(),
                y,
                np.zeros(n),
                np.full(n, 2.0),
                1.0,
                weight_semantics="prior",
            )


class TestTheFrequencyCheckReachesTheEvaluationBoundariesToo:
    """The lattice half of the contract check reached every boundary; the
    replication half did not.

    ``_check_counting_lattice`` returns immediately under ``"frequency"`` and
    ``_check_frequency_counts`` returns immediately under ``"prior"``, so a
    boundary wired to one of them looks right and silently covers half the
    models it sees. Every boundary now goes through ``check_weight_contract``,
    which is both.
    """

    @staticmethod
    def _fitted_on_whole_counts():
        """Fitted with integral weights, so the FIT never warns.

        The evaluation call is then the only thing that can produce a warning,
        which is what makes these tests pin the boundary rather than the fit.
        """
        rng = np.random.default_rng(11)
        n = 60
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = rng.poisson(3.0, n).astype(float)
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            model.fit(X, y, sample_weight=np.full(n, 2.0))
        return model, X, y, n

    def test_metrics_warns_on_fractional_replication_counts(self):
        model, X, y, n = self._fitted_on_whole_counts()
        with pytest.warns(FractionalFrequencyWeightWarning):
            model.metrics(X, y, sample_weight=np.full(n, 2.5))

    def test_cross_validation_warns_on_fractional_replication_counts(self):
        """The fractional counts are VALIDATION-ONLY, never trained on.

        ``cross_validate`` takes one weight array for both roles, so putting
        the fractional values everywhere would make the training fit warn and
        the test would pass with the scorer's check mutated out.
        """
        from superglm.model_selection import cross_validate

        model, X, y, n = self._fitted_on_whole_counts()
        weights = np.full(n, 2.0)
        weights[-8:] = 2.5  # fractional only in the held-out rows

        class _Folds:
            n_splits = 1

            def split(self, X, y=None, groups=None):
                idx = np.arange(len(X))
                held_out = idx[-8:]
                yield np.setdiff1d(idx, held_out), held_out

        with pytest.warns(FractionalFrequencyWeightWarning):
            cross_validate(model, X, y, sample_weight=weights, cv=_Folds(), scoring="nll")

    def test_vuong_warns_on_fractional_replication_counts(self):
        from superglm.stats.model_tests import vuong_test

        model, X, y, n = self._fitted_on_whole_counts()
        other = SuperGLM(family=Gaussian(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            other.fit(X, y, sample_weight=np.full(n, 2.0))
        with pytest.warns(FractionalFrequencyWeightWarning):
            vuong_test(model, other, X, y, sample_weight=np.full(n, 2.5))


class TestVuongChecksTheContractOnEveryArm:
    """The check sat inside the both-prior branch, so two of the three arms
    never ran it.

    A mixed-contract comparison is legal precisely when no weights are
    supplied -- unit weights make the two readings the same likelihood -- but
    unit weights still define a counting lattice, so a fractional response
    reaches the interpolated ``gammaln`` and returns a statistic and p-value
    with nothing marking them as a pseudo-density.
    """

    @staticmethod
    def _pair_fitted_on_lattice(contract_a, contract_b):
        rng = np.random.default_rng(12)
        n = 80
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y_fit = rng.poisson(3.0, n).astype(float)  # on-lattice: the fits stay silent
        a = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics=contract_a)
        b = SuperGLM(family=Gaussian(), features={"x": Numeric()}, weight_semantics=contract_b)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            a.fit(X, y_fit)
            b.fit(X, y_fit)
        y_eval = y_fit.copy()
        y_eval[:12] += 0.5  # off-lattice only in the EVALUATION rows
        return a, b, X, y_eval

    def test_the_mixed_contract_arm_is_checked(self):
        from superglm.stats.model_tests import vuong_test

        a, b, X, y_eval = self._pair_fitted_on_lattice("prior", "frequency")
        with pytest.warns(PriorWeightLatticeWarning):
            vuong_test(a, b, X, y_eval)

    def test_the_frequency_arm_is_checked(self):
        from superglm.stats.model_tests import vuong_test

        a, b, X, y_eval = self._pair_fitted_on_lattice("frequency", "frequency")
        with pytest.warns(FractionalFrequencyWeightWarning):
            vuong_test(a, b, X, y_eval, sample_weight=np.full(len(y_eval), 2.5))


class TestIntegralityToleranceSurvivesLargeMagnitudes:
    """A relative tolerance cannot express "close to an integer".

    Nothing is ever further than 0.5 from its nearest integer, so once the
    relative slack reaches 0.5 the test admits everything. At 1e-9 that
    happens at ``|v| = 5e8``, and ``1e9 + 0.5`` -- exactly representable in
    float64 -- was reported as a whole number of replications.
    """

    @pytest.mark.parametrize("magnitude", [1e6, 1e8, 5e8, 1e9, 1e12])
    def test_a_half_integer_is_never_called_whole(self, magnitude):
        from superglm.model.input_validation import _off_integer_lattice

        value = magnitude + 0.5
        assert value - magnitude == 0.5, "test needs an exactly representable half"
        assert bool(_off_integer_lattice(np.array([value]))[0])

    @pytest.mark.parametrize("magnitude", [1.0, 1e3, 1e6, 1e9])
    def test_round_off_is_still_absorbed(self, magnitude):
        from superglm.model.input_validation import _off_integer_lattice

        nudged = np.nextafter(magnitude, np.inf)
        assert not bool(_off_integer_lattice(np.array([nudged]))[0])

    def test_a_large_fractional_replication_count_warns_end_to_end(self):
        n = 40
        rng = np.random.default_rng(13)
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        # Noise deliberately: an exact linear fit drives phi to zero and the
        # weighted log-likelihood overflows, which would mask what is tested.
        y = np.linspace(1.0, 5.0, n) + rng.normal(0.0, 0.4, n)
        weights = np.full(n, 1e9)
        weights[0] = 1e9 + 0.5
        model = SuperGLM(family=Gaussian(), features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(FractionalFrequencyWeightWarning):
            model.fit(X, y, sample_weight=weights)


class TestTheContractWarningNeverPreemptsAValidationError:
    """The warning describes a likelihood that is about to be computed.

    On input that will never reach one it is noise, and under ``-W error`` it
    surfaces *instead of* the error the caller needs -- reporting a
    quasi-likelihood for a negative Poisson response rather than the negative
    response. Same defect as the custom-family warning this branch fixes.
    """

    def test_an_invalid_response_raises_rather_than_warning(self):
        n = 40
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote every warning
            with pytest.raises(ValueError, match="nonnegative"):
                model.fit(X, np.full(n, -1.0), sample_weight=np.full(n, 2.5))

    def test_a_valid_response_still_warns(self):
        """The control: the ordering fix must not have disabled the warning."""
        n = 40
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(FractionalFrequencyWeightWarning):
            model.fit(X, np.arange(n, dtype=float), sample_weight=np.full(n, 2.5))


class TestTheIntegralitySlackIsCorrectAtEveryMagnitude:
    """Swept over the exponent range, not sampled at chosen magnitudes.

    This check has been wrong three times, each time at a magnitude nobody
    had tried: a 1e-9 relative tolerance died at ``|v| >= 5e8``, a 1e-3
    absolute ceiling admitted ``1_000_000.0005``, and eight raw ulps died at
    ``|v| >= 2**48`` where one ulp is 1/16 and eight are exactly a half-count.

    Each was found by a reviewer naming the next magnitude, so testing more
    named magnitudes would only postpone the fourth. The sweep below asserts
    the property that actually has to hold -- a representable half-integer is
    always flagged, an exact integer never is, and a ulp of round-off is
    absorbed -- at every binade of the double range.
    """

    def test_every_binade_of_the_double_range(self):
        from superglm.model.input_validation import _off_integer_lattice

        violations = []
        for exponent in range(64):
            magnitude = float(2**exponent)
            spacing = np.spacing(magnitude)

            if _off_integer_lattice(np.array([magnitude]))[0]:
                violations.append(f"2^{exponent}: an exact integer was flagged")

            # Where a half-integer is representable it must always be caught:
            # this is the failure the three previous rules all shared.
            if spacing <= 0.5:
                half = magnitude + 0.5
                assert half - magnitude == 0.5
                if not _off_integer_lattice(np.array([half]))[0]:
                    violations.append(f"2^{exponent}: half-integer admitted (spacing={spacing})")

            # Below a half-count a ulp is round-off, and absorbing it is what
            # the slack is for. At or above it a ulp IS half a count, so the
            # two cannot be told apart and the property does not apply.
            if spacing < 0.5:
                nudged = np.nextafter(magnitude, np.inf)
                if _off_integer_lattice(np.array([nudged]))[0]:
                    violations.append(
                        f"2^{exponent}: one ulp of round-off was flagged (spacing={spacing})"
                    )

        assert not violations, "\n".join(violations)

    def test_the_slack_never_reaches_a_half_count(self):
        """The invariant underneath the sweep, stated directly.

        Any slack that reaches 0.5 admits every value, because nothing is
        further than 0.5 from its nearest integer. Every rule that scales with
        magnitude eventually crosses it, which is why the ceiling is the fix
        rather than a larger multiplier.
        """
        from superglm.model.input_validation import (
            _LATTICE_MAXIMUM_SLACK,
            _LATTICE_ULP_SLACK,
        )

        assert _LATTICE_MAXIMUM_SLACK < 0.5
        magnitudes = 2.0 ** np.arange(64, dtype=np.float64)
        slack = np.minimum(_LATTICE_ULP_SLACK * np.spacing(magnitudes), _LATTICE_MAXIMUM_SLACK)
        assert np.all(slack < 0.5)


class TestTheIntegralitySlackIsUlpScaled:
    """A relative slack cannot express "close to an integer" at any setting.

    Nothing is ever further than 0.5 from its nearest integer, so a relative
    slack that reaches 0.5 admits everything -- and an absolute ceiling only
    moves where it fails rather than removing the failure. Float64 spacing is
    the scale that actually applies.
    """

    @pytest.mark.parametrize(
        "value",
        [
            1_000_000.0005,  # inside a 1e-3 ceiling, still not a whole number
            1_000_000.5,
            500_000_000.5,
            1_000_000_000.5,  # a 1e-9 relative slack reaches 0.5 here
            1e12 + 0.5,
            281474976710656.5,  # 2**48: eight raw ulps are exactly 0.5 here
        ],
    )
    def test_an_exactly_representable_fraction_is_never_called_whole(self, value):
        from superglm.model.input_validation import _off_integer_lattice

        assert value != np.rint(value), "test needs a genuinely fractional value"
        assert bool(_off_integer_lattice(np.array([value]))[0])

    @pytest.mark.parametrize("magnitude", [1.0, 1e3, 1e6, 1e9, 1e15])
    def test_a_ulp_of_round_off_is_still_absorbed(self, magnitude):
        """The slack exists for this and must keep covering it.

        Measured: over 200,000 trials of ``count / exposure * exposure`` the
        worst deviation from the intended integer was 1.0 ulp.

        One ulp, not several: at 1e15 a ulp is already 0.125, so four of them
        are a half-count -- the thing the check must never absorb. An earlier
        version of this test asserted that four ulps are absorbed at every
        magnitude, which asserted the defect rather than the fix.
        """
        from superglm.model.input_validation import _off_integer_lattice

        nudged = np.nextafter(magnitude, np.inf)
        assert not bool(_off_integer_lattice(np.array([nudged]))[0])

    def test_the_canonical_exposure_round_trip_does_not_warn(self):
        """``count / exposure`` times ``exposure`` must read as integral."""
        from superglm.model.input_validation import _off_integer_lattice

        rng = np.random.default_rng(21)
        counts = rng.integers(0, 10_000, size=20_000).astype(float)
        exposure = rng.uniform(0.01, 50.0, size=20_000)
        assert not _off_integer_lattice((counts / exposure) * exposure).any()


class TestTheEvaluationWarningNeverPreemptsAValidationError:
    """Same ordering rule as the fit entry, at the evaluation boundary.

    ``predict`` is what validates the evaluation frame and the offset, so a
    contract warning raised ahead of it fires on input that will never reach a
    likelihood -- and under ``-W error`` it surfaces instead of the error the
    caller needs.
    """

    @staticmethod
    def _fitted():
        rng = np.random.default_rng(22)
        n = 60
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = rng.poisson(3.0, n).astype(float)
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            model.fit(X, y, sample_weight=np.full(n, 2.0))
        return model, X, y, n

    def test_a_malformed_offset_raises_rather_than_warning(self):
        model, X, y, n = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote every warning
            with pytest.raises(ValueError):
                model.metrics(
                    X,
                    y,
                    sample_weight=np.full(n, 2.5),  # fractional: would warn
                    offset=np.zeros(n - 3),  # but this is the real problem
                )

    def test_a_well_formed_call_still_warns(self):
        """The control: the reordering must not have disabled the check."""
        model, X, y, n = self._fitted()
        with pytest.warns(FractionalFrequencyWeightWarning):
            model.metrics(X, y, sample_weight=np.full(n, 2.5), offset=np.zeros(n))


class TestASuppliedCountIsTestedExactly:
    """A replication count is handed over, not computed, so nothing forms it.

    The product tolerance exists because ``w * y`` is built here from two user
    arrays and lands a ulp or so from an integral intent. A weight the caller
    supplies has no product in it, and lending it that allowance admitted
    representable fractional counts at large magnitudes -- at ``2**49`` a ulp
    is already an eighth of a count.
    """

    @pytest.mark.parametrize(
        "value",
        [
            2.0**49 + 0.125,  # exactly one ulp, and a real eighth of a count
            2.0**48 + 0.0625,
            281474976710656.5,
            1_000_000.0005,
            1_000_000_000.5,
        ],
    )
    def test_a_representable_fraction_is_flagged_however_small(self, value):
        from superglm.model.input_validation import _not_a_whole_number

        assert value != np.rint(value), "test needs a genuinely fractional value"
        assert bool(_not_a_whole_number(np.array([value]))[0])

    @pytest.mark.parametrize("value", [0.0, 1.0, 3.0, 1e6, 2.0**49, 1e15])
    def test_a_whole_count_is_never_flagged(self, value):
        from superglm.model.input_validation import _not_a_whole_number

        assert not bool(_not_a_whole_number(np.array([value]))[0])

    def test_counts_read_as_floats_are_exact(self):
        """The reason exactness costs nothing: real counts are exact."""
        from superglm.model.input_validation import _not_a_whole_number

        rng = np.random.default_rng(31)
        counts = rng.integers(0, 10**6, 100_000).astype(float)
        assert not _not_a_whole_number(counts).any()

    def test_the_product_rule_keeps_its_round_off_allowance(self):
        """The two rules must stay apart: this one still forgives a ulp."""
        from superglm.model.input_validation import _off_integer_lattice

        rng = np.random.default_rng(32)
        counts = rng.integers(0, 10_000, 50_000).astype(float)
        exposure = rng.uniform(0.01, 50.0, 50_000)
        assert not _off_integer_lattice((counts / exposure) * exposure).any()


class TestAReplicatedCountingResponseStaysOnItsLattice:
    """Replication does not move the support.

    Under ``"frequency"`` the likelihood is ``w log f(y; mu)`` -- the weight
    multiplies an ordinary per-row density rather than entering it, so it
    cannot rescue a response the family does not support. The lattice check
    returned early for every frequency model, so a fractional ``y`` reached the
    same interpolated ``gammaln`` the prior arm warns about, with nothing
    marking it.
    """

    @staticmethod
    def _frame(n=40):
        return pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})

    @pytest.mark.parametrize("family", [Poisson(), NegativeBinomial(theta=2.0)])
    def test_a_fractional_response_warns_with_an_integral_weight(self, family):
        n = 40
        y = np.arange(n, dtype=float)
        y[:6] += 0.5
        model = SuperGLM(family=family, features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(self._frame(n), y, sample_weight=np.full(n, 2.0))

    def test_a_fractional_response_warns_with_no_weight_at_all(self):
        """The weight never enters the question, so omitting it changes nothing."""
        n = 40
        y = np.arange(n, dtype=float)
        y[:6] += 0.5
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(self._frame(n), y)

    def test_a_whole_response_stays_silent(self):
        n = 40
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            model.fit(self._frame(n), np.arange(n, dtype=float), sample_weight=np.full(n, 2.0))

    def test_a_continuous_family_is_untouched(self):
        """Only the counting families have an integer support to leave."""
        n = 40
        model = SuperGLM(family=Gaussian(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.fit(self._frame(n), np.linspace(0.5, 4.5, n), sample_weight=np.full(n, 2.0))

    def test_the_evaluation_boundaries_check_it_too(self):
        """Fit on whole counts, evaluate on fractional ones."""
        n = 40
        frame = self._frame(n)
        y_fit = np.arange(n, dtype=float)
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.fit(frame, y_fit, sample_weight=np.full(n, 2.0))
        y_eval = y_fit.copy()
        y_eval[:6] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            model.metrics(frame, y_eval, sample_weight=np.full(n, 2.0))


class TestTheReplicatedResponseCheckObeysItsSiblings:
    """The invariants three sibling checks in the module already held.

    A fourth warning was added in the previous round without adopting them:
    it did not restrict to carried rows, and it copied an "unaffected" claim
    that is false when theta is being profiled. Both are now structural --
    carried-row filtering at the check, and the reach sentence derived once
    from the family by ``_interpolated_density_reach``.
    """

    @staticmethod
    def _frame(n=40):
        return pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})

    def test_a_zero_replication_row_is_ignored(self):
        """A row that appears no times contributes exactly zero."""
        n = 40
        y = np.arange(n, dtype=float)
        y[3] += 0.5
        weights = np.full(n, 2.0)
        weights[3] = 0.0
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.fit(self._frame(n), y, sample_weight=weights)

    def test_the_same_row_warns_once_it_is_carried(self):
        """The control: it is the zero weight that silences it, not the value."""
        n = 40
        y = np.arange(n, dtype=float)
        y[3] += 0.5
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(self._frame(n), y, sample_weight=np.full(n, 2.0))

    def test_the_count_reported_is_of_carried_rows(self):
        n = 40
        y = np.arange(n, dtype=float)
        y[:5] += 0.5  # five fractional rows ...
        weights = np.full(n, 2.0)
        weights[:2] = 0.0  # ... two of which are not carried
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with pytest.warns(PriorWeightLatticeWarning, match=r"3 of 38 carried rows"):
            model.fit(self._frame(n), y, sample_weight=weights)

    @pytest.mark.parametrize(
        ("theta", "estimates_move"),
        [("auto", True), (2.0, False)],
    )
    def test_auto_theta_is_described_as_estimate_affecting(self, theta, estimates_move):
        """theta profiled from an interpolated density reaches the estimates.

        It enters V(mu) and the IRLS working weights, so saying "coefficients
        are unaffected" is false there -- and that claim was copied into the
        frequency warning from a context where it held.
        """
        n = 40
        y = np.arange(n, dtype=float) + 1.0
        y[:6] += 0.5
        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            features={"x": Numeric()},
            weight_semantics="frequency",
        )
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            model.fit(self._frame(n), y, sample_weight=np.full(n, 2.0))
        message = str(caught[0].message)
        assert ("move as well" in message) is estimates_move
        # At a fixed theta "unaffected" is the true claim about the estimates,
        # so forbidding the phrase outright was over-broad. What must never
        # appear is a theta_hat this call did not produce.
        assert ("theta_hat" in message) is estimates_move

    def test_the_reach_sentence_is_derived_once(self):
        """Both arms answer the same question, so they answer it in one place."""
        from superglm.model.input_validation import (
            THETA_ESTIMATED,
            THETA_FIXED,
            _interpolated_density_reach,
        )

        assert "move as well" in _interpolated_density_reach(
            NegativeBinomial(theta="auto"), theta_role=THETA_ESTIMATED
        )
        assert "move as well" not in _interpolated_density_reach(
            NegativeBinomial(theta=2.0), theta_role=THETA_FIXED
        )
        assert "unaffected" in _interpolated_density_reach(Poisson(), theta_role=THETA_FIXED)


class TestTheEvaluationResponseLengthIsCheckedFirst:
    """``predict`` validates the frame and the offset, but never ``y``.

    So a length mismatch was still outstanding when the contract warning ran,
    and under ``-W error`` pre-empted it -- the same ordering defect as the
    offset, one argument over.
    """

    def test_a_mismatched_response_length_raises_rather_than_warning(self):
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        model.fit(frame, np.arange(n, dtype=float), sample_weight=np.full(n, 2.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="same observations"):
                model.metrics(
                    frame,
                    np.arange(n - 3, dtype=float) + 0.5,
                    sample_weight=np.full(n - 3, 2.5),
                )

    def test_a_matched_length_still_warns(self):
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        model.fit(frame, np.arange(n, dtype=float), sample_weight=np.full(n, 2.0))
        with pytest.warns(FractionalFrequencyWeightWarning):
            model.metrics(frame, np.arange(n, dtype=float), sample_weight=np.full(n, 2.5))


class TestTheStandaloneThetaProfileChecksTheContract:
    """``profile_ci_theta`` is a public likelihood boundary of its own.

    It takes arrays directly, so no fit-time or evaluation-time check has seen
    those rows, and every NLL it evaluates reads the same response and weights.
    An off-contract input returned an apparently exact interval with nothing
    marking it -- the sixth boundary, and the one this branch missed.
    """

    @staticmethod
    def _arrays(n=200, seed=41):
        rng = np.random.default_rng(seed)
        y = rng.poisson(4.0, n).astype(float)
        return y, np.full(n, 4.0), n

    def test_fractional_replication_counts_warn(self):
        from superglm.profiling.nb import profile_ci_theta

        y, mu, n = self._arrays()
        with pytest.warns(FractionalFrequencyWeightWarning):
            profile_ci_theta(y, mu, np.full(n, 2.5), 3.0, weight_semantics="frequency")

    def test_an_off_lattice_prior_response_warns(self):
        from superglm.profiling.nb import profile_ci_theta

        y, mu, n = self._arrays()
        y = y.copy()
        y[:20] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            profile_ci_theta(y, mu, np.full(n, 1.5), 3.0, weight_semantics="prior")

    def test_a_fractional_response_warns_under_frequency(self):
        from superglm.profiling.nb import profile_ci_theta

        y, mu, n = self._arrays()
        y = y.copy()
        y[:20] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            profile_ci_theta(y, mu, np.full(n, 2.0), 3.0, weight_semantics="frequency")

    @pytest.mark.parametrize("semantics", ["prior", "frequency"])
    def test_an_honoured_contract_stays_silent(self, semantics):
        from superglm.profiling.nb import profile_ci_theta

        y, mu, n = self._arrays()
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            profile_ci_theta(y, mu, np.full(n, 2.0), 3.0, weight_semantics=semantics)

    def test_it_reports_the_interval_moving_not_the_coefficients(self):
        """mu is held fixed here, so nothing can be refitted."""
        from superglm.profiling.nb import profile_ci_theta

        y, mu, n = self._arrays()
        y = y.copy()
        y[:20] += 0.5
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            profile_ci_theta(y, mu, np.full(n, 1.5), 3.0, weight_semantics="prior")
        message = str(caught[0].message)
        assert "profiled theta and its interval" in message
        assert "move as well" not in message  # the coefficients do not


class TestTheImpactClaimMatchesWhatTheCallIsDoing:
    """The reach depends on what the caller does with theta, not only on the family.

    An auto-theta fit stamps a numeric theta into the family when it finishes,
    so at evaluation the fitted model is indistinguishable from a fixed-theta
    one -- and at evaluation nothing is refitted. Deriving the claim from the
    family alone therefore asserted that ``theta_hat`` and its interval were
    affected in two places where no theta is estimated at all.
    """

    @staticmethod
    def _fractional_response(n=40):
        y = np.arange(n, dtype=float) + 1.0
        y[:6] += 0.5
        return pd.DataFrame({"x": np.linspace(0.0, 1.0, n)}), y, n

    def test_a_fixed_theta_fit_claims_nothing_about_theta_hat(self):
        frame, y, n = self._fractional_response()
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            features={"x": Numeric()},
            weight_semantics="frequency",
        )
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            model.fit(frame, y, sample_weight=np.full(n, 2.0))
        message = str(caught[0].message)
        assert "theta_hat" not in message
        assert "unaffected" in message

    def test_an_auto_theta_fit_still_says_the_estimates_move(self):
        """The control: the fix must not have silenced the case that is real."""
        frame, y, n = self._fractional_response()
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"x": Numeric()},
            weight_semantics="frequency",
        )
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            model.fit(frame, y, sample_weight=np.full(n, 2.0))
        assert "move as well" in str(caught[0].message)

    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("fixed", "unaffected"),
            ("profiled", "profiled theta and its interval"),
            ("estimated", "move as well"),
        ],
    )
    def test_each_role_states_only_what_that_call_can_move(self, role, expected):
        from superglm.model.input_validation import _interpolated_density_reach

        reach = _interpolated_density_reach(NegativeBinomial(theta=2.0), theta_role=role)
        assert expected in reach

    def test_a_finished_auto_fit_is_a_constant_from_then_on(self):
        """Evaluation refits nothing, so the claim must not survive the fit."""
        from superglm.model.input_validation import THETA_FIXED, _theta_role_for

        assert _theta_role_for(NegativeBinomial(theta=2.0)) == THETA_FIXED
        assert _theta_role_for(Poisson()) == THETA_FIXED
        assert _theta_role_for(NegativeBinomial(theta="auto")) == "estimated"


class TestTheProductToleranceIsOnlyEarnedByAProduct:
    """``w * y`` at a unit weight is the response, bit for bit.

    IEEE-754 multiplication by a power of two only shifts the exponent, so
    nothing is rounded and there is no round-off to forgive. Applying the
    product allowance there lent it to a directly supplied response -- in the
    default fit, since unit weights are what an unweighted call passes.
    """

    def test_multiplication_by_a_power_of_two_is_exact(self):
        """The property the branch rests on, asserted rather than assumed."""
        from superglm.model.input_validation import _is_exact_power_of_two

        rng = np.random.default_rng(51)
        y = rng.uniform(1.0, 1e6, 20_000)
        for w in (1.0, 2.0, 0.5, 4.0, 1024.0):
            assert bool(_is_exact_power_of_two(np.array([w]))[0])
            assert np.all((w * y) / w == y)
        for w in (3.0, 2.5, 0.1):
            assert not bool(_is_exact_power_of_two(np.array([w]))[0])

    def test_a_zero_weight_is_not_treated_as_exact(self):
        """0 * y is always integral, so the branch it takes cannot matter."""
        from superglm.model.input_validation import _is_exact_power_of_two

        assert not bool(_is_exact_power_of_two(np.array([0.0]))[0])

    @pytest.mark.parametrize("value", [2.0**49 + 0.125, 2.0**48 + 0.0625, 1_000_000.0005])
    def test_a_unit_weighted_response_is_tested_exactly(self, value):
        """These sit inside the product allowance but are genuinely fractional."""
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = np.full(n, 4.0)
        y[0] = value
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(frame, y, sample_weight=np.ones(n))

    def test_a_genuine_product_keeps_its_allowance(self):
        """The canonical rate weighting must not start warning."""
        from superglm.model.input_validation import _check_counting_lattice

        rng = np.random.default_rng(52)
        counts = rng.integers(0, 10_000, 20_000).astype(float)
        exposure = rng.uniform(0.01, 50.0, 20_000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            _check_counting_lattice(counts / exposure, exposure, Poisson(), "prior")


class TestEveryPublicLikelihoodEntryChecksTheContract:
    """Enumerated rather than listed from memory.

    "Every evaluation boundary" was five, then six. The entry points that take
    arrays or a model plus arrays -- rather than reading a fit's stored rows --
    are the ones no other check can cover, so they are the ones to enumerate.
    """

    @staticmethod
    def _fitted_on_whole_counts(n=200):
        rng = np.random.default_rng(53)
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(4.0, n).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            features={"x": Numeric()},
            weight_semantics="frequency",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            model.fit(frame, y, sample_weight=np.full(n, 2.0))
        return model, frame, y, n

    def test_estimate_nb_theta_warns_on_fractional_counts(self):
        from superglm import estimate_nb_theta

        model, frame, y, n = self._fitted_on_whole_counts()
        with pytest.warns(FractionalFrequencyWeightWarning):
            estimate_nb_theta(model, frame, y, sample_weight=np.full(n, 2.5))

    def test_estimate_nb_theta_warns_on_a_fractional_response(self):
        from superglm import estimate_nb_theta

        model, frame, y, n = self._fitted_on_whole_counts()
        y = y.copy()
        y[:30] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            estimate_nb_theta(model, frame, y, sample_weight=np.full(n, 2.0))

    def test_estimate_nb_theta_says_the_estimates_move(self):
        """It refits beta at each candidate theta, so the reach is the estimates."""
        from superglm import estimate_nb_theta

        model, frame, y, n = self._fitted_on_whole_counts()
        y = y.copy()
        y[:30] += 0.5
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            estimate_nb_theta(model, frame, y, sample_weight=np.full(n, 2.0))
        assert "move as well" in str(caught[0].message)

    def test_an_honoured_contract_stays_silent(self):
        from superglm import estimate_nb_theta

        model, frame, y, n = self._fitted_on_whole_counts()
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            estimate_nb_theta(model, frame, y, sample_weight=np.full(n, 2.0))


class TestAFoldIsCheckedOncePerCondition:
    """The per-fold score and the pooled aggregate are one evaluation.

    They divide the same numerator by the same denominator, so computing them
    separately predicted twice per fold -- and once the contract check landed
    in each, one off-contract fold warned twice from two call sites, scaling
    the noise with the fold count.
    """

    @staticmethod
    def _validation_only_off_lattice(n=160, seed=93):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        y[-8:] += 0.5  # off-lattice, held out, so the fit never warns

        class _Folds:
            n_splits = 1

            def split(self, X, y=None, groups=None):
                idx = np.arange(len(X))
                held_out = idx[-8:]
                yield np.setdiff1d(idx, held_out), held_out

        return frame, y, _Folds()

    def test_one_warning_per_affected_fold(self):
        from superglm.model_selection import cross_validate

        frame, y, folds = self._validation_only_off_lattice()
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            cross_validate(model, frame, y, cv=folds, scoring="nll")
        lattice = [w for w in caught if issubclass(w.category, PriorWeightLatticeWarning)]
        assert len(lattice) == 1

    def test_the_warning_count_does_not_scale_with_folds(self):
        """Three affected folds should say so three times, not six."""
        from superglm.model_selection import cross_validate

        rng = np.random.default_rng(94)
        n = 180
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.5 + frame["x"].to_numpy())).astype(float)
        y += 0.5  # every row off-lattice, so every validation slice warns

        class _Folds:
            n_splits = 3

            def split(self, X, y=None, groups=None):
                idx = np.arange(len(X))
                for fold in np.array_split(idx, 3):
                    yield np.setdiff1d(idx, fold), fold

        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cross_validate(model, frame, y, cv=_Folds(), scoring="nll")
        lattice = [w for w in caught if issubclass(w.category, PriorWeightLatticeWarning)]
        # Three fits (each trains on off-lattice rows too) plus three scored
        # slices -- what must not happen is the scoring half counting twice.
        assert len([w for w in lattice if "model_selection" in w.filename]) == 3


class TestEveryBoundaryValidatesBeforeItWarns:
    """One invariant, swept over the boundaries, instead of found one at a time.

    A contract warning describes a likelihood that is about to be computed, so
    on a request that cannot produce one it is noise -- and under ``-W error``
    it surfaces *instead of* the error the caller needs. That has now been
    found four times on this branch, at the fit entry, at the evaluation
    offset, at the evaluation response, and at the theta profile, each time by
    someone naming the next site.

    Sweeping it is what stops a fifth. A new boundary that warns before it
    validates fails here rather than in review.
    """

    @staticmethod
    def _fitted(n=80, seed=61):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(4.0, n).astype(float)
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            warnings.simplefilter("error", FractionalFrequencyWeightWarning)
            model.fit(frame, y, sample_weight=np.full(n, 2.0))
        return model, frame, y, n

    def test_the_fit_entry_raises_on_a_bad_response(self):
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                model.fit(frame, np.full(n, -1.0), sample_weight=np.full(n, 2.5))

    def test_metrics_raises_on_a_bad_offset(self):
        model, frame, y, n = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                model.metrics(frame, y, sample_weight=np.full(n, 2.5), offset=np.zeros(n - 3))

    def test_metrics_raises_on_a_bad_response_length(self):
        model, frame, y, n = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                model.metrics(
                    frame,
                    np.arange(n - 3, dtype=float) + 0.5,
                    sample_weight=np.full(n - 3, 2.5),
                )

    def test_the_theta_profile_raises_on_mismatched_arrays(self):
        from superglm.profiling.nb import profile_ci_theta

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="same rows"):
                profile_ci_theta(
                    np.array([1.5, 2.5, 3.5]),
                    np.ones(2),
                    np.full(3, 2.5),
                    3.0,
                    weight_semantics="frequency",
                )


class TestOneConditionWarnsOnce:
    """The other invariant nothing was enforcing.

    A single off-contract condition must be reported once per operation, not
    once per internal path that happens to touch it. Found three times -- the
    cross-validation double score, the auto-theta fit, and metrics reusing the
    fit's own statistics -- so it is swept here too.
    """

    def test_an_auto_theta_fit_warns_once(self):
        rng = np.random.default_rng(62)
        n = 120
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(4.0, n).astype(float)
        y[:10] += 0.5
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            features={"x": Numeric()},
            weight_semantics="prior",
        )
        with pytest.warns(PriorWeightLatticeWarning) as caught:
            model.fit(frame, y, sample_weight=np.full(n, 1.5))
        lattice = [w for w in caught if issubclass(w.category, PriorWeightLatticeWarning)]
        assert len(lattice) == 1

    def test_metrics_on_the_training_arrays_does_not_warn_again(self):
        """It reuses the fit's likelihood, which the fit already checked."""
        rng = np.random.default_rng(63)
        n = 100
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = np.arange(n, dtype=float)
        y[:5] += 0.5
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(frame, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.metrics(frame, y)

    def test_a_genuine_holdout_still_warns(self):
        """The control: suppression must key on reuse, not on having fitted."""
        rng = np.random.default_rng(64)
        n = 100
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.fit(frame, np.arange(n, dtype=float))
        holdout = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 40)})
        y_holdout = np.arange(40, dtype=float)
        y_holdout[:5] += 0.5
        with pytest.warns(PriorWeightLatticeWarning):
            model.metrics(holdout, y_holdout)


class TestTheProductShortcutRequiresARepresentableProduct:
    """Being a power of two is necessary but not sufficient.

    The exponent shift is lossless only while the result stays representable:
    ``w = 5e-324`` with ``y = 0.5`` underflows to exactly ``0.0``, which then
    reads as a whole number, so a rounded-away row was reported as on-lattice.
    """

    def test_an_underflowing_product_is_not_treated_as_exact(self):
        from superglm.model.input_validation import _product_was_exact

        w = np.array([np.nextafter(0.0, 1.0)])
        y = np.array([0.5])
        assert not bool(_product_was_exact(w, y, w * y)[0])

    @pytest.mark.parametrize("weight", [1.0, 2.0, 0.5, 1024.0, 2.0**-40])
    def test_a_representable_power_of_two_product_stays_exact(self, weight):
        from superglm.model.input_validation import _product_was_exact

        y = np.array([3.5, 100.25, 7.0])
        w = np.full(3, weight)
        scaled = w * y
        assert np.all(_product_was_exact(w, y, scaled))

    def test_scaling_back_agrees_with_exact_arithmetic(self):
        """The property the shortcut rests on, checked against Fraction."""
        from fractions import Fraction

        from superglm.model.input_validation import _product_was_exact

        rng = np.random.default_rng(65)
        exponents = rng.integers(-1060, 1000, 4000)
        weights = np.ldexp(1.0, exponents.astype(np.int32)).astype(np.float64)
        y = rng.uniform(-1e6, 1e6, 4000)
        # A zero weight is excluded by the helper's documented contract: 0 * y
        # is exactly 0, so the product IS exact, but the row is uncarried and
        # `scaled == 0` is integral under either rule, so which branch it takes
        # cannot change an outcome. Asserting past that would pin an
        # immaterial answer rather than the property.
        carried = weights > 0.0
        assert carried.sum() > 3000, "test needs mostly carried rows"
        scaled = weights * y
        claimed = _product_was_exact(weights, y, scaled)
        for i in range(0, 4000, 37):  # sampled: Fraction is slow but exact
            if not carried[i]:
                continue
            truth = Fraction(scaled[i]) == Fraction(weights[i]) * Fraction(y[i])
            assert bool(claimed[i]) == truth, (weights[i], y[i], scaled[i])

    def test_an_underflowed_product_is_not_judged_exactly(self):
        """End-to-end, because the helper's own tests cannot see the call site.

        The exact rule is justified by "nothing rounded this value". That
        justification fails when the product underflows, so such a row must
        fall back to the tolerance -- whichever way the verdict then falls.

        Here it falls towards silence, and that is right: the row carries a
        weight of 5e-324 and contributes nothing to the reported likelihood,
        so calling the fit a quasi-likelihood on its account would be noise.
        Judging a rounded value exactly is the defect; the direction of the
        resulting verdict is a consequence, not the goal.
        """
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = np.arange(n, dtype=float)
        y[0] = 1.5
        weights = np.ones(n)
        weights[0] = np.nextafter(0.0, 1.0)  # w * y underflows to a subnormal
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PriorWeightLatticeWarning)
            model.fit(frame, y, sample_weight=weights)

    def test_a_normally_weighted_fractional_row_still_warns(self):
        """The control: it is the underflow that silences it, not the value."""
        n = 40
        frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = np.arange(n, dtype=float)
        y[0] = 1.5
        model = SuperGLM(family=Poisson(), features={"x": Numeric()}, weight_semantics="prior")
        with pytest.warns(PriorWeightLatticeWarning):
            model.fit(frame, y, sample_weight=np.ones(n))
