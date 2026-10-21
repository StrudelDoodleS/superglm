"""Exact rational NB2 size information, as a test oracle.

For an integer size ``theta`` and a rational mean ``mu`` every NB2 pmf value
is rational (``P(0) = q**theta``, ``q = theta/(mu + theta)``), so Lawless's
expected information (*Canadian J. Statist.* 15(3), 1987, eq. 2.8, written in
``k = 1/a``)::

    I = sum_{j>=0} P(Y > j)/(theta + j)**2 - mu/(theta (theta + mu))

is summed exactly over ``j < terms``. ``P(Y > j)`` decreases and
``sum_{j>=J} 1/(theta + j)**2 <= 1/(theta + J - 1)``, so the omitted tail is at
most ``P(Y > J - 1)/(theta + J - 1)``, also exact (Yu, Dousti Mousavi and
Yang, arXiv:2306.04798, Theorem 2). This module does not import production
code; it uses Lawless's form, not the summed-by-parts series the kernel uses.
"""

from __future__ import annotations

from fractions import Fraction


def size_information(mean: Fraction, size: int, terms: int) -> tuple[Fraction, Fraction]:
    """Return ``theta**2 I`` summed over ``j < terms`` and an exact bound on the rest."""
    if size < 1 or terms < 2 or mean <= 0:
        raise ValueError("the oracle needs a positive integer size, a positive mean and terms >= 2")
    theta = Fraction(size)
    p = mean / (mean + theta)
    pmf = (1 - p) ** size
    cdf = pmf
    total = Fraction(0)
    for j in range(terms):
        total += (1 - cdf) / (theta + j) ** 2
        pmf = pmf * p * (theta + j) / (j + 1)
        cdf += pmf
    survival_before_last = 1 - (cdf - pmf)
    tail = survival_before_last / (theta + terms - 1)
    value = theta**2 * (total - mean / (theta * (theta + mean)))
    return value, theta**2 * tail


def pmf_values(mean: Fraction, size: int, count: int) -> list[Fraction]:
    """Return the exact NB2 pmf at ``0..count-1`` for an integer size."""
    theta = Fraction(size)
    p = mean / (mean + theta)
    pmf = (1 - p) ** size
    values = []
    for y in range(count):
        values.append(pmf)
        pmf = pmf * p * (theta + y) / (y + 1)
    return values
