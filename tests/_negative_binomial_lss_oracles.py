"""Four frozen, independently generated normalized NB2 row oracles.

The companion script differentiates the complete normalized law with mpmath
and cross-checks the derivatives with finite-count recurrences. Neither this
module nor the generator imports production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WeightSemantics = Literal["prior", "frequency"]


@dataclass(frozen=True, slots=True)
class NegativeBinomialLSSOracleCase:
    """One normalized row with literal value, score, and Hessian evidence."""

    id: str
    count: int
    mean: float
    theta: float
    weight: float
    semantics: WeightSemantics
    full_log_likelihood: float
    optimizing_log_likelihood: float
    factorial_carrier: float
    natural_score: tuple[float, float]
    natural_hessian_packed: tuple[float, float, float]
    value_atol: float = 5.0e-14
    score_rtol: tuple[float, float] = (5.0e-13, 5.0e-13)
    score_atol: tuple[float, float] = (5.0e-14, 5.0e-14)
    hessian_rtol: tuple[float, float, float] = (5.0e-13, 5.0e-13, 5.0e-13)
    hessian_atol: tuple[float, float, float] = (5.0e-14, 5.0e-14, 5.0e-14)
    source: Literal["mpmath-differentiated-normalized-law"] = "mpmath-differentiated-normalized-law"
    source_precision_digits: int = 100


NEGATIVE_BINOMIAL_LSS_CASES = (
    NegativeBinomialLSSOracleCase(
        id="zero-small-unit-prior",
        count=0,
        mean=2.5,
        theta=0.35,
        weight=1.0,
        semantics="prior",
        full_log_likelihood=-0.733999391572732896204621973878519829286725,
        optimizing_log_likelihood=-0.733999391572732896204621973878519829286725,
        factorial_carrier=0.0,
        natural_score=(
            -0.122807017543859649122807017543859649122807,
            -1.21994813632309649542172694291105916137059,
        ),
        natural_hessian_packed=(
            0.043090181594336718990458602646968297937827,
            -0.30778701138811942136041859033548784241305,
            2.19847865277228158114584707382491316009322,
        ),
    ),
    NegativeBinomialLSSOracleCase(
        id="nonunit-frequency",
        count=2,
        mean=0.8,
        theta=0.65,
        weight=3.0,
        semantics="frequency",
        full_log_likelihood=-7.00228269412351542912015028309944188096182,
        optimizing_log_likelihood=-4.92284115244367950086845391872491217673532,
        factorial_carrier=-2.0794415416798359282516963643745297042265,
        natural_score=(
            2.01724137931034482758620689655172413793103,
            1.54376839530196652063041955772837870229687,
        ),
        natural_hessian_packed=(
            -5.59378715814506539833531510107015457788347,
            1.71224732461355529131985731272294887039239,
            -3.94385366787518737827644141306697537624755,
        ),
    ),
    NegativeBinomialLSSOracleCase(
        id="fractional-prior-exposure",
        count=7,
        mean=9.0,
        theta=1.3,
        weight=0.5,
        semantics="prior",
        full_log_likelihood=-3.31253454752233034872958795944785454236565,
        optimizing_log_likelihood=5.21262681354308395143594307689927050839401,
        factorial_carrier=-8.52516136106541430016553103634712505075967,
        natural_score=(
            0.0350593311758360302049622437971952535059331,
            0.391528783794443953420213575897185371370889,
        ),
        natural_hessian_packed=(
            -0.0143111660376875445842046526999554303415805,
            0.0235648977283438589876519935903478178904704,
            -0.403318580173894398774851136738010974134132,
        ),
    ),
    NegativeBinomialLSSOracleCase(
        id="large-theta",
        count=7,
        mean=2.0,
        theta=1.0e8,
        weight=1.0,
        semantics="prior",
        full_log_likelihood=-5.67313100714580055091144051947869390732389,
        optimizing_log_likelihood=2.85203035391961374925409051686843114343577,
        factorial_carrier=-8.52516136106541430016553103634712505075967,
        natural_score=(
            2.499999950000000999999980000000399999992,
            -8.99999931666670636666447806678696999331673e-16,
        ),
        natural_hessian_packed=(
            -1.7499999899999997000000159999994800000144,
            4.999999800000005999999840000003999999904e-16,
            1.79999979500001587999890570007218199532171e-23,
        ),
        score_rtol=(5.0e-14, 5.0e-14),
        score_atol=(5.0e-14, 1.0e-29),
        hessian_rtol=(5.0e-14, 5.0e-14, 5.0e-14),
        hessian_atol=(5.0e-14, 1.0e-29, 1.0e-37),
    ),
)
