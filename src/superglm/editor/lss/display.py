"""How the editor names and shows each SuperLSS parameter.

Every term curve in the editor is a relativity: the factor the term multiplies
its parameter by, ``exp(effect)`` on the link scale. A negative binomial
``theta`` term is the relativity of the overdispersion ``1 / theta``,
``exp(-effect)``, so it rises with overdispersion. A log-normal ``scale`` term
is the relativity of σ above its floor. No curve is read at a reference row, so
editing one curve of a parameter never moves another's display, and the
Cross-validation tab draws the same values (``cv.py``). A Tweedie ``power`` has
no curves, so it is named here and drawn nowhere else.

The names follow the parameter alone: ``mean`` is "Mean", ``scale`` is "CV" and
``theta`` is "Overdispersion" (a history chip reads "1/θ"), ``dispersion`` is
"Dispersion" and ``power`` is "Power". The ``role`` and the colour group the mean
apart from the other parameters and are not read as words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superglm.distributional.families import (
    GammaLS,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
)
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.view import LSSModelView, split_key

Role = Literal["average", "spread", "shape"]
Colour = Literal["blue", "orange"]
Kind = Literal["relativity", "inverse_theta"]

# The smallest positive value an edit is converted from: the log of a value at or
# below zero does not exist, so it is held here.
_SMALLEST = 1.0e-12

_ROLES: dict[str, Role] = {
    "mean": "average",
    "scale": "spread",
    "theta": "spread",
    "dispersion": "spread",
    "power": "shape",
}
_COLOURS: dict[Role, Colour] = {"average": "blue", "spread": "orange", "shape": "orange"}
_CHIPS: dict[str, str] = {
    "mean": "Mean",
    "scale": "CV",
    "theta": "1/θ",
    "dispersion": "Dispersion",
    "power": "Power",
}
_LABELS: dict[str, str] = {
    "mean": "Mean",
    "scale": "CV",
    "theta": "Overdispersion",
    "dispersion": "Dispersion",
    "power": "Power",
}
_FULL_NAMES: dict[str, str] = {
    "mean": "the mean: the expected value of the response",
    "scale": "coefficient of variation (CV): standard deviation ÷ mean",
    "theta": "overdispersion (1/θ): variance = mean + mean²/θ",
    "dispersion": "dispersion (φ): the variance is φ times the mean to the power p",
    "power": "power (p): the variance is the dispersion times the mean to the power p",
}
# The families the editor opens.
SPREAD_FAMILIES: tuple[type, ...] = (LogNormalLS, GammaLS, NegativeBinomialLS, TweedieLSS)
_Y_LABELS: dict[str, str] = {
    "mean": "relativity",
    "scale": "relativity of the CV",
    "theta": "relativity of the overdispersion",
    "dispersion": "relativity of the dispersion",
}
# The log-normal's scale term multiplies σ above its floor, not the CV the group is named for.
_LOGNORMAL_Y_LABEL = "relativity of σ"


def y_label_for(parameter: str, family: object = None) -> str:
    """What a term's relativity is a relativity of, as the chart's axis says it."""
    if parameter == "scale" and isinstance(family, LogNormalLS):
        return _LOGNORMAL_Y_LABEL
    return _Y_LABELS.get(parameter, "relativity")


@dataclass(frozen=True)
class ParameterDisplay:
    """A parameter's label, history chip, full name and identity colour."""

    name: str
    label: str
    chip: str
    role: Role
    colour: Colour
    full_name: str


def parameter_display(parameter: str) -> ParameterDisplay:
    """The label, chip and full name of ``parameter``: Mean, CV or Overdispersion."""
    role = _ROLES.get(parameter)
    if role is None:
        raise EditorValueError(
            f"The editor has no display for parameter {parameter!r}; it shows "
            f"{', '.join(map(repr, _ROLES))}."
        )
    return ParameterDisplay(
        parameter,
        _LABELS[parameter],
        _CHIPS[parameter],
        role,
        _COLOURS[role],
        _FULL_NAMES[parameter],
    )


@dataclass(frozen=True)
class DisplayTransform:
    """Between a term's link-scale effect and the relativity the editor shows for it.

    ``relativity`` shows ``exp(effect)``; ``inverse_theta`` shows ``exp(-effect)``,
    the relativity of the overdispersion ``1 / theta``, which falls as the
    recorded effect rises.
    """

    kind: Kind = "relativity"

    @property
    def _direction(self) -> float:
        return 1.0 if self.kind == "relativity" else -1.0

    def to_display(self, effect: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self._direction * np.asarray(effect, dtype=np.float64))

    def from_display(self, values: ArrayLike) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return self._direction * np.log(np.maximum(v, _SMALLEST))

    def to_payload(self) -> dict[str, str]:
        return {"kind": self.kind}

    @property
    def increasing(self) -> bool:
        """Whether the shown value rises with the effect; the overdispersion's falls."""
        return self.kind == "relativity"


RELATIVITY = DisplayTransform("relativity")
INVERSE_THETA = DisplayTransform("inverse_theta")


@dataclass(frozen=True)
class FoldTermDisplay:
    """How the Cross-validation tab shows a term.

    ``label`` names it anywhere and ``short`` within its ``group``'s list;
    ``transform`` turns its link-scale effect into the curve drawn, as the Editor
    tab draws it, and ``value`` says what the curve is a relativity of. A term of
    a SuperGLM has no group.
    """

    label: str
    short: str
    transform: DisplayTransform = RELATIVITY
    group: str | None = None
    role: Role | None = None
    value: str = "relativity"


def fold_chart_display(key: str, family: object = None) -> FoldTermDisplay:
    """How the Cross-validation tab shows the SuperLSS term ``key`` of a ``family`` model.

    The curve is the one the Editor tab draws for the term, so a theta curve is
    the relativity of the overdispersion and rises with it.
    """
    parameter, column = split_key(key)
    shown = parameter_display(parameter)
    return FoldTermDisplay(
        label=f"{column} · {shown.label}",
        short=column,
        transform=_transform_of(parameter),
        group=shown.label,
        role=shown.role,
        value=y_label_for(parameter, family),
    )


def _transform_of(parameter: str) -> DisplayTransform:
    return INVERSE_THETA if parameter == "theta" else RELATIVITY


def display_for(model: object, key: str) -> DisplayTransform:
    """How the term ``key`` of ``model`` is shown: a relativity, of 1/θ for a theta term."""
    if not isinstance(model, LSSModelView):
        return RELATIVITY
    return _transform_of(model.parameter_of(key))


_MIRRORED_OPERATIONS = {
    "isotonic_increasing": "isotonic_decreasing",
    "increasing": "decreasing",
    "isotonic_decreasing": "isotonic_increasing",
    "decreasing": "increasing",
    "snap_highest": "snap_lowest",
    "snap_lowest": "snap_highest",
}


def directed_operation(model: object, key: str | None, operation: str) -> tuple[str, float]:
    """The operation, and the sign of its effect-scale shift, that act on ``key``'s shown curve.

    A directional tool (Increase, Make increasing, Snap to highest) names the shown
    curve, so on a display that falls as its effect rises it runs the opposite way
    on the effect. With no term the operation is unchanged.
    """
    if key is None or display_for(model, key).increasing:
        return operation, 1.0
    return _MIRRORED_OPERATIONS.get(operation, operation), -1.0


def lss_history_fields(view: LSSModelView, key: str | None) -> dict[str, str | bool]:
    """The parameter, its chip, its role and the term a history entry names; empty for no term.

    ``predictor_role`` is the chip's colour group, so the browser need not read the chip text.

    ``display_decreasing`` says the shown curve falls as the recorded effect rises,
    so History names a recorded shift, monotone fit or snap by the shown direction.
    """
    if key is None:
        return {}
    parameter, label = split_key(key)
    shown = parameter_display(parameter)
    return {
        "predictor": parameter,
        "predictor_label": shown.chip,
        "predictor_role": shown.role,
        "term_label": label,
        "display_decreasing": not display_for(view, key).increasing,
    }


def lss_term_fields(view: LSSModelView, key: str) -> dict[str, object]:
    """The payload fields a SuperLSS term carries beyond a SuperGLM term's."""
    parameter, label = split_key(key)
    shown = parameter_display(parameter)
    return {
        "parameter": parameter,
        "parameter_label": shown.label,
        "parameter_role": shown.role,
        "term_label": label,
        "display": display_for(view, key).to_payload(),
        "y_label": y_label_for(parameter, view.family),
        "x_label": label,
        "title": label,
    }
