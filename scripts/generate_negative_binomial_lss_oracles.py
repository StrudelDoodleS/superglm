"""Regenerate the four frozen NB2 LSS row oracles without production code.

Run with ``uv run --with mpmath python scripts/generate_negative_binomial_lss_oracles.py``.
"""

from __future__ import annotations

import mpmath as mp

CASES = (
    ("zero-small-unit-prior", 0, "2.5", "0.35", "1", "prior"),
    ("nonunit-frequency", 2, "0.8", "0.65", "3", "frequency"),
    ("fractional-prior-exposure", 7, "9", "1.3", "0.5", "prior"),
    ("large-theta", 7, "2", "1e8", "1", "prior"),
)


def effective_law(
    mean: mp.mpf,
    theta: mp.mpf,
    weight: mp.mpf,
    semantics: str,
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    if semantics == "prior":
        return weight * mean, weight * theta, weight, mp.mpf(1)
    return mean, theta, mp.mpf(1), weight


def optimizing_value(count: int, mean: mp.mpf, theta: mp.mpf) -> mp.mpf:
    total = mean + theta
    return (
        mp.loggamma(count + theta)
        - mp.loggamma(theta)
        + theta * (mp.log(theta) - mp.log(total))
        + count * (mp.log(mean) - mp.log(total))
    )


def normalized_value(
    count: int,
    mean: mp.mpf,
    theta: mp.mpf,
    weight: mp.mpf,
    semantics: str,
) -> mp.mpf:
    effective_mean, effective_theta, _, multiplier = effective_law(mean, theta, weight, semantics)
    return multiplier * (
        optimizing_value(count, effective_mean, effective_theta) - mp.loggamma(count + 1)
    )


def recurrence_derivatives(
    count: int,
    mean: mp.mpf,
    theta: mp.mpf,
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    total = mean + theta
    reciprocal_sum = mp.fsum(1 / (theta + offset) for offset in range(count))
    reciprocal_square_sum = mp.fsum(-1 / (theta + offset) ** 2 for offset in range(count))
    return (
        theta * (count - mean) / (mean * total),
        reciprocal_sum - mp.log1p(mean / theta) + (mean - count) / total,
        -count / mean**2 + (count + theta) / total**2,
        (count - mean) / total**2,
        reciprocal_square_sum + 1 / theta - 1 / total + (count - mean) / total**2,
    )


def show(value: mp.mpf) -> str:
    return mp.nstr(value, 50)


def main() -> None:
    mp.mp.dps = 100
    for case_id, count, mean_text, theta_text, weight_text, semantics in CASES:
        mean = mp.mpf(mean_text)
        theta = mp.mpf(theta_text)
        weight = mp.mpf(weight_text)

        def varying(candidate_mean: mp.mpf, candidate_theta: mp.mpf) -> mp.mpf:
            return normalized_value(count, candidate_mean, candidate_theta, weight, semantics)

        full = varying(mean, theta)
        effective_mean, effective_theta, chain, multiplier = effective_law(
            mean, theta, weight, semantics
        )
        carrier = -multiplier * mp.loggamma(count + 1)
        differentiated = (
            mp.diff(lambda value: varying(value, theta), mean),
            mp.diff(lambda value: varying(mean, value), theta),
            mp.diff(lambda value: varying(value, theta), mean, 2),
            mp.diff(lambda value: mp.diff(lambda k: varying(value, k), theta), mean),
            mp.diff(lambda value: varying(mean, value), theta, 2),
        )
        recurrence = recurrence_derivatives(count, effective_mean, effective_theta)
        checked = (
            multiplier * chain * recurrence[0],
            multiplier * chain * recurrence[1],
            multiplier * chain**2 * recurrence[2],
            multiplier * chain**2 * recurrence[3],
            multiplier * chain**2 * recurrence[4],
        )
        assert all(
            mp.almosteq(left, right, rel_eps=mp.mpf("1e-70"), abs_eps=mp.mpf("1e-90"))
            for left, right in zip(differentiated, checked, strict=True)
        )

        print(f"CASE {case_id}")
        print(f"  full={show(full)}")
        print(f"  optimizing={show(full - carrier)}")
        print(f"  carrier={show(carrier)}")
        print(f"  score=({show(differentiated[0])}, {show(differentiated[1])})")
        print("  hessian=(" + ", ".join(show(value) for value in differentiated[2:]) + ")")


if __name__ == "__main__":
    main()
