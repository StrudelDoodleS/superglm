"""Pure numerical predicates for non-negative count lattices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Slack when testing a value for integrality, in units of float64 spacing.
#:
#: The quantity tested is formed in floating point from user arrays -- an
#: exactly-integral intent (``count / exposure`` times ``exposure``) lands a few
#: ulps away -- so the allowance has to scale with magnitude.  Spacing is the
#: scale that does: it is what the nearest representable neighbour costs.
#:
#: A *relative* tolerance is not.  Nothing is ever further than 0.5 from its
#: nearest integer, so any relative slack that reaches 0.5 admits everything;
#: 1e-9 does that at ``|v| >= 5e8``, and a 1e-3 ceiling only moves the failure
#: rather than removing it -- ``1_000_000.0005`` is exactly representable, is
#: not a whole number of replications, and sits inside that ceiling.
#:
#: Sized from measurement, not taste.  Over 200,000 trials of
#: ``count / exposure * exposure`` the worst deviation from the intended
#: integer was **1.0 ulp**, and summed integer counts were exact.
_LATTICE_ULP_SLACK = 8.0

#: Hard ceiling on the slack, and the invariant that makes this check correct
#: at every magnitude rather than at the magnitudes anyone happened to try.
#:
#: Nothing is ever further than 0.5 from its nearest integer, so *any* slack
#: that reaches 0.5 admits every value and the check silently dies. This has
#: now been true of three successive rules -- a 1e-9 relative tolerance (dies
#: at ``|v| >= 5e8``), a 1e-3 absolute ceiling (admits ``1_000_000.0005``),
#: and eight raw ulps (dies at ``|v| >= 2**48``, where one ulp is 1/16 and
#: eight of them are exactly a half-count).  Every scale that grows with
#: magnitude eventually crosses a half, so the ceiling is not a patch for one
#: more reported magnitude; it is the thing that bounds the whole family.
#:
#: A quarter-count is deliberately loose.  It only binds above ``2**48``,
#: where a ulp is already 1/16 of a count, so it is still four ulps of
#: round-off there -- not the seven-orders-too-wide allowance that a fixed
#: absolute tolerance was at ordinary magnitudes.  Above ``2**52`` every
#: representable double is an integer and the question stops being askable.
_LATTICE_MAXIMUM_SLACK = 0.25


def _off_integer_lattice(values: NDArray) -> NDArray:
    """Which entries of a COMPUTED product are further from whole than round-off.

    For ``w * y`` only.  The product is formed here from two user arrays, so an
    exactly-integral intent lands a ulp or so away and a tolerance is required;
    measured worst case over 200,000 round-trips of ``count / exposure`` times
    ``exposure`` was 1.0 ulp.

    Do not use this on a value the caller supplied directly -- see
    :func:`_not_a_whole_number`, which is deliberately stricter.  Sharing one
    rule between the two was wrong: it lent the product's round-off allowance
    to quantities that have no product in them, and admitted representable
    fractional counts such as ``2**49 + 0.125`` as a result.

    Correct at every magnitude by construction: the slack tracks float64
    spacing where spacing is the binding scale, and is capped below a
    half-count everywhere else, so it can never admit a representable
    half-integer.  ``TestTheIntegralitySlackIsCorrectAtEveryMagnitude`` sweeps
    the exponent range rather than sampling magnitudes, because sampling is
    what let this be wrong three times.
    """

    spacing = np.spacing(np.maximum(1.0, np.abs(values)))
    slack = np.minimum(_LATTICE_ULP_SLACK * spacing, _LATTICE_MAXIMUM_SLACK)
    return np.abs(values - np.rint(values)) > slack


def _not_a_whole_number(values: NDArray) -> NDArray:
    """Which entries of a SUPPLIED array are not exactly whole numbers.

    For a declared replication count, and for a counting response under the
    frequency contract.  Both are values the caller hands over and declares to
    be counts; nothing here forms them, so there is no round-off to forgive and
    an exact test is the honest one.  ``counts.astype(float)`` is exactly
    integral, as is anything read from a count column.

    Exact rather than a ulp or two on purpose.  At ``2**49`` a ulp is already
    ``0.125`` of a count, so any non-zero allowance admits representable
    fractional counts at large magnitudes -- which is precisely the hole that
    borrowing the product tolerance opened.  A count that is not exactly an
    integer is not a count, and saying so is more useful than guessing which
    near-integers were meant.
    """

    return values != np.rint(values)


def _is_exact_power_of_two(values: NDArray) -> NDArray:
    """Which weights scale a response without introducing any rounding.

    IEEE-754 multiplication by a power of two only shifts the exponent, so
    ``w * y`` is exact there and carries no round-off to forgive.  ``w == 1``
    is the case that matters -- it is what an unweighted fit passes -- but the
    property is the same for 2, 0.5 and the rest, so the test is the property
    rather than the one value.

    Zero is excluded deliberately: ``0 * y`` is exactly zero and therefore
    always integral, so those rows never flag under either rule and the branch
    they take is immaterial.
    """

    finite = np.isfinite(values) & (values > 0.0)
    mantissa, _ = np.frexp(np.where(finite, values, 1.0))
    return finite & (mantissa == 0.5)


def _product_was_exact(weights: NDArray, y: NDArray, scaled: NDArray) -> NDArray:
    """Which rows had ``w * y`` computed without losing a single bit.

    Being a power of two is necessary but not sufficient: the exponent shift is
    lossless only while the result stays representable.  ``w = 5e-324`` with
    ``y = 0.5`` underflows to exactly ``0.0``, which then reads as a whole
    number, so a rounded-away row was reported as being on the lattice.

    Scaling back is the test that covers it.  Division by a power of two is
    itself exact, so ``scaled / w == y`` holds precisely when the
    multiplication lost nothing -- verified against exact rational arithmetic
    over 200,000 draws spanning the subnormal range, with no disagreement.
    The power-of-two gate is what makes the division trustworthy, so both
    conditions stay.
    """

    gate = _is_exact_power_of_two(weights) & np.isfinite(scaled)
    with np.errstate(divide="ignore", invalid="ignore"):
        recovered = np.where(gate, np.divide(scaled, np.where(gate, weights, 1.0)), np.nan)
    return gate & (recovered == y)


__all__ = [
    "_LATTICE_MAXIMUM_SLACK",
    "_LATTICE_ULP_SLACK",
    "_is_exact_power_of_two",
    "_not_a_whole_number",
    "_off_integer_lattice",
    "_product_was_exact",
]
