"""Pure recovered C3 fixtures, without experimental assembler/solver patches.

Source: binomial-core-audit/analysis/math_audit_20260905/{fit_ab,composite_ab}.py.
RNG draw order is intentional, including the Gaussian draw before Tweedie.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.features import Spline
from superglm.features.interaction import TensorInteraction


def fixture(n, knots=4, mix=0.0, seed=5915, signal="curved", family="gaussian"):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = mix * x + (1 - mix) * rng.uniform(0, 1, n)
    weight = np.exp(rng.uniform(-0.7, 0.7, n))
    mu = 0.7 * np.sin(5 * x) + 0.5 * np.cos(5 * z) + 0.4 * np.sin(3 * x) * np.cos(3 * z)
    sigma = np.exp(
        -0.5 + 0.35 * np.sin(4 * x) + 0.25 * np.cos(5 * z) + 0.25 * np.sin(4 * x) * np.cos(3 * z)
    )
    if signal == "curved":
        mu = 0.7 * np.sin(5 * x) + 0.5 * np.cos(5 * z) + 0.8 * np.sin(8 * x) * np.sin(7 * z)
        sigma = np.exp(
            -0.5 + 0.35 * np.sin(5 * x) + 0.25 * np.cos(5 * z) + 0.3 * np.cos(7 * x) * np.cos(8 * z)
        )
    y = mu + sigma / np.sqrt(weight) * rng.standard_normal(n)
    distribution = GaussianLS()
    if family == "gamma":
        distribution = GammaLS()
        mean = np.exp(mu / 2)
        shape = weight / sigma**2
        y = rng.gamma(shape, mean / shape)
    elif family == "tweedie":
        distribution = TweedieLSS()
        mean, phi = np.exp(mu / 2), sigma**2
        p = 1 + expit(
            0.4 * np.sin(5 * x) + 0.4 * np.cos(5 * z) + 0.6 * np.sin(8 * x) * np.sin(7 * z)
        )
        counts = rng.poisson(weight * mean ** (2 - p) / (phi * (2 - p)))
        y = np.zeros(n)
        positive = counts > 0
        y[positive] = (
            rng.gamma(
                counts[positive] * (2 - p[positive]) / (p[positive] - 1),
                phi[positive] * (p[positive] - 1) * mean[positive] ** (p[positive] - 1),
            )
            / weight[positive]
        )
    frame = pd.DataFrame({"x": x, "z": z})
    return lss_model(distribution, knots), frame, y, weight


def marked_book(n, tail=False):
    rng = np.random.default_rng(9506)
    total = n + max(1000, n // 4)
    x, z = rng.uniform(size=(2, total))
    interaction = np.sin(8 * x) * np.sin(7 * z)
    mean = np.exp(0.4 * np.sin(5 * x) + 0.3 * np.cos(5 * z) + 0.6 * interaction)
    phi = np.exp(0.8 + 0.3 * np.sin(5 * x) + 0.25 * np.cos(5 * z) + 0.3 * interaction)
    power = 1 + expit(0.3 * np.sin(5 * x) + 0.2 * np.cos(5 * z) + 0.4 * interaction)
    exposure = rng.uniform(0.15, 1.5, total)
    rate = mean ** (2 - power) / (phi * (2 - power))
    counts = rng.poisson(exposure * rate)
    policy = np.repeat(np.arange(total), counts)
    if not tail:
        shape = (2 - power) / (power - 1)
        scale = phi * (power - 1) * mean ** (power - 1)
        losses = rng.gamma(shape[policy], scale[policy])
        body_latent = None
    else:
        gate = expit(-1.7 + 0.45 * np.sin(5 * x) - 0.35 * np.cos(5 * z) + 0.6 * interaction)
        is_tail = rng.uniform(size=len(policy)) < gate[policy]
        mean_v = np.exp(-0.5 + 0.5 * np.sin(5 * x) + 0.3 * np.cos(5 * z) + 0.5 * interaction)
        cv = np.exp(-0.7 + 0.3 * np.sin(5 * x) - 0.2 * np.cos(5 * z) + 0.3 * interaction)
        body_latent = rng.gamma(1 / cv[policy] ** 2, mean_v[policy] * cv[policy] ** 2)
        losses = 1000 * -np.expm1(-body_latent)
        sigma = 500 * np.exp(0.5 * np.sin(5 * x) + 0.35 * np.cos(5 * z) + 0.5 * interaction)
        xi = 0.15 + 0.25 * expit(np.sin(5 * x) - 0.7 * np.cos(5 * z) + interaction)
        ids = policy[is_tail]
        excess = sigma[ids] * np.expm1(-xi[ids] * np.log1p(-rng.uniform(size=len(ids)))) / xi[ids]
        losses[is_tail] = 1000 + excess
    return {
        "frame": pd.DataFrame({"x": x, "z": z}),
        "exposure": exposure,
        "counts": counts,
        "policy": policy,
        "losses": losses,
        "aggregate": np.bincount(policy, weights=losses, minlength=total),
        "body_latent": body_latent,
    }


def lss_model(family, knots):
    from superglm.distributional.binding import _bind_predictor_template

    bound_family = family
    return SuperLSS(
        bound_family,
        *(
            _bind_predictor_template(bound_family, predictor)
            for predictor in [
                Predictor(
                    p.name,
                    {"x": Spline(n_knots=knots), "z": Spline(n_knots=knots)},
                    interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(knots, knots))},
                )
                for p in family.parameters
            ]
        ),
    )
