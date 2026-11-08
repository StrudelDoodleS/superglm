"""Independent textbook laws: no production expression/generator imports."""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np

PARAMETERS = ("origin", "width", "drift")
COORDINATES = "natural"


def log_density(law, y, scale, shape):
    if law == "lomax":
        return mp.log(shape) - mp.log(scale) - (shape + 1) * mp.log1p(y / scale)
    z = (y / scale) ** shape
    return mp.log(shape / scale) + (shape - 1) * mp.log(y / scale) - 2 * mp.log1p(z)


def log_window(law, lower, upper, scale, shape):
    def logsf(x):
        if x == mp.inf:
            return -mp.inf
        if x <= 0:
            return mp.mpf(0)
        return -shape * mp.log1p(x / scale) if law == "lomax" else -mp.log1p((x / scale) ** shape)

    a, b = logsf(lower), logsf(upper)
    return a + mp.log(-mp.expm1(b - a))


def jets(function, theta, order):
    with mp.workdps(100):
        theta = tuple(mp.mpf(float(t)) for t in theta)
        result = [np.array([float(function(*theta))])]
        for d in range(1, order + 1):
            result.append(
                np.array(
                    [
                        float(
                            mp.diff(
                                function, theta, tuple(indices.count(j) for j in range(len(theta)))
                            )
                        )
                        for indices in combinations_with_replacement(range(len(theta)), d)
                    ]
                )
            )
        return result


def assert_scaled_jets(actual, reference, theta, *, factor=512):
    """Scale each natural partial by its parameter units; bound absolute error.

    Four differentiations and primitive composition cost O(100) operations.
    512*u times the order's dimensionless magnitude also covers cancellation
    at a root; no relative accuracy is demanded of a vanishing partial.
    This is an outcome tolerance, not a production certification bound.
    """
    for d, (got, want) in enumerate(zip(actual, reference, strict=True)):
        units = np.array(
            [
                np.prod([theta[j] for j in indices])
                for indices in combinations_with_replacement(range(len(theta)), d)
            ]
        )
        size = max(1.0, np.max(np.abs(want * units)))
        np.testing.assert_array_less(
            np.abs((np.ravel(got) - want) * units), factor * 2.0**-53 * size + 1e-300
        )


def carrier(y, multiplier):
    return np.asarray(multiplier) * (-0.5 * np.log(2 * np.pi))


def density_jets(y, theta, multiplier, *, order):
    """Test-only three-coordinate normal law, mean=origin+drift, sd=width.

    Its redundant mean coordinates are intentional: this is a plumbing test,
    not an identified fitting family or a production numerical oracle.
    """
    from superglm.distributional.kernels._declared_adapter import ScalarJetResult

    rows = [
        jets(
            lambda origin, width, drift: (
                -mp.log(width) - (mp.mpf(float(z)) - origin - drift) ** 2 / (2 * width**2)
            ),
            t,
            order,
        )
        for z, t in zip(y, theta, strict=True)
    ]
    levels = tuple(
        np.array([r[d] for r in rows]) * np.asarray(multiplier)[:, None] for d in range(order + 1)
    )
    return ScalarJetResult(
        (levels[0][:, 0], *levels[1:]), carrier(y, multiplier), np.ones(len(y), bool)
    )
