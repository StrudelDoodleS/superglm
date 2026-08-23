"""The declared weight contract, and what each reading is required to satisfy.

Each contract has a definition that pins it independently of the implementation:
``"frequency"`` says an integer weight is a repeated row, and ``"prior"`` says
the row's density carries ``phi / w``.  The tests below hold the code to those
two statements rather than to numbers it produced, so a regression in either
reading shows up as a disagreement with row replication or with scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import Categorical, Gamma, Gaussian, NegativeBinomial, Poisson, Spline, SuperGLM
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
        assert ("theta_hat" in message) is mentions_theta

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

    def test_the_residual_inverts_the_all_success_count(self):
        """y == 1 is K == w, so its interval starts at P(K <= w-1) = 1 - mu**w.

        Starting at (1-mu)**w would hand the all-success row every
        intermediate count too, which is not the transform of the fitted
        likelihood.
        """

        rng, frame = _frame(seed=53, n=150)
        y = _response(rng, frame, Binomial())
        weights = np.full(len(frame), 3.0)

        model = SuperGLM(
            family=Binomial(),
            features={"x": Spline(n_knots=4)},
            weight_semantics="prior",
        )
        with pytest.warns(UserWarning):
            model.fit(frame, y, sample_weight=weights)
        metrics = model.metrics(frame, y, sample_weight=weights)
        mu = np.asarray(model.predict(frame), dtype=np.float64)
        residuals = metrics.residuals("quantile", seed=3)

        lower = stats.norm.ppf(1.0 - np.power(mu, weights))
        success = y == 1
        # Every all-success row sits above its own K = w lower bound, which is
        # strictly above the (1-mu)**w bound the previous code used.
        assert np.all(residuals[success] >= lower[success] - 1e-9)
        assert np.all(np.power(1.0 - mu, weights) < 1.0 - np.power(mu, weights))


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
