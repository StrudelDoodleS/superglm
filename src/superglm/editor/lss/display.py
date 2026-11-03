"""How the editor names and shows each SuperLSS parameter.

A mean with a log link is shown as relativities, as the editor shows a SuperGLM
term. A spread parameter is shown as the readable quantity it controls, at the
reference policy: the coefficient of variation of a claim for the log-normal and
gamma scales, and ``1 / theta`` for the negative binomial. An edit made in those
units converts back to the term's link-scale effect, clamped to the nearest
value the parameter can take.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superglm.distributional.families import GammaLS, LogNormalLS, NegativeBinomialLS
from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.view import LSSModelView, Response, split_key

Role = Literal["average", "spread"]
Colour = Literal["blue", "orange"]
Kind = Literal["relativity", "lognormal_cv", "gamma_cv", "inverse_theta"]

# The smallest positive value an edit is converted from: the link of a value at or
# below zero does not exist, so it is held here.
_SMALLEST = 1.0e-12

_ROLES: dict[str, Role] = {"mean": "average", "scale": "spread", "theta": "spread"}
_COLOURS: dict[Role, Colour] = {"average": "blue", "spread": "orange"}
_CHIPS: dict[Role, str] = {"average": "Average", "spread": "Spread"}
_CLAIM_LABELS = {
    "mean": "Average claim",
    "scale": "Spread of claim sizes",
}
_COUNT_LABELS = {
    "mean": "Claim frequency",
    "theta": "Extra variation in claim counts",
}
_GENERIC_LABELS: dict[Role, str] = {"average": "Average", "spread": "Spread"}
# The families the editor opens, with how each shows its spread parameter.
SPREAD_KINDS: dict[type, Kind] = {
    LogNormalLS: "lognormal_cv",
    GammaLS: "gamma_cv",
    NegativeBinomialLS: "inverse_theta",
}
_Y_LABELS: dict[Kind, str] = {
    "relativity": "relativity",
    "lognormal_cv": "coefficient of variation",
    "gamma_cv": "coefficient of variation",
    "inverse_theta": "1/theta",
}
_Y_FORMATS: dict[Kind, str] = {
    "relativity": "relativity",
    "lognormal_cv": "times",
    "gamma_cv": "times",
    "inverse_theta": "plain",
}


@dataclass(frozen=True)
class ParameterDisplay:
    """A parameter's plain name, history chip and identity colour."""

    name: str
    label: str
    chip: str
    role: Role
    colour: Colour


def parameter_display(family: object, parameter: str, *, response: Response) -> ParameterDisplay:
    """The plain-language name of ``family``'s ``parameter`` for a claims or generic response."""
    role = _ROLES.get(parameter)
    if role is None:
        raise EditorValueError(
            f"The editor has no display for parameter {parameter!r}; it shows 'mean', 'scale' "
            "and 'theta'."
        )
    if response == "generic":
        label = _GENERIC_LABELS[role]
    elif response == "claims":
        labels = _COUNT_LABELS if isinstance(family, NegativeBinomialLS) else _CLAIM_LABELS
        label = labels[parameter]
    else:
        raise EditorValueError(f"response must be 'claims' or 'generic'; got {response!r}.")
    return ParameterDisplay(parameter, label, _CHIPS[role], role, _COLOURS[role])


@dataclass(frozen=True)
class DisplayTransform:
    """Between a term's link-scale effect and the number the editor shows for it.

    ``reference_eta`` is the parameter's link predictor at the reference policy
    less the term's own part there, so ``reference_eta + effect`` is the link
    predictor with only this term's column moved. ``floor`` is the log-normal
    scale's lower bound.
    """

    kind: Kind
    reference_eta: float = 0.0
    floor: float = 0.0

    def to_display(self, effect: ArrayLike) -> NDArray[np.float64]:
        e = np.asarray(effect, dtype=np.float64)
        if self.kind == "relativity":
            return np.exp(e)
        eta = self.reference_eta + e
        if self.kind == "lognormal_cv":
            sigma = self.floor + np.exp(eta)
            return np.sqrt(np.expm1(sigma**2))
        if self.kind == "gamma_cv":
            return np.exp(eta)
        return np.exp(-eta)

    def from_display(self, values: ArrayLike) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        if self.kind == "relativity":
            return np.log(np.maximum(v, _SMALLEST))
        if self.kind == "lognormal_cv":
            sigma = np.sqrt(np.log1p(np.maximum(v, 0.0) ** 2))
            return np.log(np.maximum(sigma - self.floor, _SMALLEST)) - self.reference_eta
        if self.kind == "gamma_cv":
            return np.log(np.maximum(v, _SMALLEST)) - self.reference_eta
        return -np.log(np.maximum(v, _SMALLEST)) - self.reference_eta

    def to_payload(self) -> dict[str, float | str]:
        return {
            "kind": self.kind,
            "reference_eta": float(self.reference_eta),
            "floor": float(self.floor),
        }

    @property
    def increasing(self) -> bool:
        """Whether the shown value rises with the effect; ``1 / theta`` falls."""
        return self.kind != "inverse_theta"

    @property
    def y_label(self) -> str:
        return _Y_LABELS[self.kind]

    @property
    def y_format(self) -> str:
        return _Y_FORMATS[self.kind]


RELATIVITY = DisplayTransform("relativity")


def display_for(
    model: object, key: str, terms: Mapping[str, EditableTerm] | None = None
) -> DisplayTransform:
    """How the term ``key`` of ``model`` is shown; relativities for anything but a SuperLSS spread.

    A spread is shown at the reference policy under the model in force: with
    ``terms``, the editor's terms, the other edited terms of the parameter move
    the reference; without them it is the fit's.
    """
    if not isinstance(model, LSSModelView) or model.parameter_of(key) == "mean":
        return RELATIVITY
    family = model.family
    kind = next(kind for cls, kind in SPREAD_KINDS.items() if isinstance(family, cls))
    floor = family.scale_floor if isinstance(family, LogNormalLS) else 0.0
    reference = model.reference_eta(key)
    if terms is not None:
        from superglm.editor.lss.apply import reference_shift

        reference += reference_shift(model, terms, key)
    return DisplayTransform(kind, reference, floor)


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
    """The parameter, its chip and the term a history entry names; empty for no term.

    ``display_decreasing`` says the shown curve falls as the recorded effect rises,
    so History names a recorded shift, monotone fit or snap by the shown direction.
    """
    if key is None:
        return {}
    parameter, label = split_key(key)
    chip = parameter_display(view.family, parameter, response=view.response).chip
    return {
        "predictor": parameter,
        "predictor_label": chip,
        "term_label": label,
        "display_decreasing": not display_for(view, key).increasing,
    }


def lss_term_fields(
    view: LSSModelView, key: str, terms: Mapping[str, EditableTerm]
) -> dict[str, object]:
    """The payload fields a SuperLSS term carries beyond a SuperGLM term's.

    ``baseline`` is the display value when the term's own effect is zero, the
    line the chart measures the curve from; ``terms`` are the editor's terms.
    """
    parameter, label = split_key(key)
    shown = parameter_display(view.family, parameter, response=view.response)
    transform = display_for(view, key, terms)
    return {
        "parameter": parameter,
        "parameter_label": shown.label,
        "parameter_role": shown.role,
        "term_label": label,
        "display": transform.to_payload(),
        "baseline": float(transform.to_display(0.0)),
        "y_label": transform.y_label,
        "y_format": transform.y_format,
        "x_label": label,
        "title": label,
    }
