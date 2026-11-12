"""How the editor shows a term: decided by the link its predictor was fitted on.

A term's curve is its additive contribution ``f`` to one linear predictor. On a log
link, or a shifted log ``floor + exp(eta)``, adding ``f`` multiplies the parameter
(above its floor) by ``exp(f)``, so the curve is that relativity, 1 where the term
has no effect. On a logit link it multiplies the odds, so the curve is the odds
relativity. On any other link no factor exists: the curve is ``f`` itself, the
amount added on the link scale, 0 where the term has no effect. A parameter
declared with role ``size`` on a log link (the negative binomial's θ) shows the
relativity of its reciprocal, the overdispersion ``1 / theta``, ``exp(-f)``. No
curve is read at a reference row, so editing one curve of a parameter never moves
another's display, and the Cross-validation tab draws the same values (``cv.py``).

A SuperLSS parameter's name comes from its family's ``ParameterSpec``: its
``label``, or its name capitalised. One parameter of a family is the average
(blue): the one declared with role ``mean``, else ``location``, else the first
declared. Every other parameter is a spread (orange). A SuperGLM term reads as a
term of the mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superglm.distributional.api import SuperLSS
from superglm.distributional.families.gaussian import LowerBoundedLogLink
from superglm.distributional.family import ParameterSpec
from superglm.editor.errors import EditorKeyError
from superglm.links import IdentityLink, Link, LogitLink, LogLink, _is_shifted_log_link

Role = Literal["average", "spread"]
Colour = Literal["blue", "orange"]
Kind = Literal["relativity", "inverse_theta", "additive"]

# The smallest positive value a relativity is converted from: the log of a value at
# or below zero does not exist, so it is held here.
_SMALLEST = 1.0e-12

_COLOURS: dict[Role, Colour] = {"average": "blue", "spread": "orange"}


@dataclass(frozen=True)
class DisplayTransform:
    """Between a term's link-scale effect and the value the editor shows for it.

    ``relativity`` shows ``exp(effect)``; ``inverse_theta`` shows ``exp(-effect)``,
    the relativity of the overdispersion ``1 / theta``, which falls as the
    recorded effect rises; ``additive`` shows the effect itself. ``link`` names
    the fitted link for labels only, so two transforms of one kind compare equal.
    """

    kind: Kind = "relativity"
    link: str = field(default="log", compare=False)

    @property
    def _direction(self) -> float:
        return -1.0 if self.kind == "inverse_theta" else 1.0

    def to_display(self, effect: ArrayLike) -> NDArray[np.float64]:
        if self.kind == "additive":
            return np.array(effect, dtype=np.float64)
        return np.exp(self._direction * np.asarray(effect, dtype=np.float64))

    def from_display(self, values: ArrayLike) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        if self.kind == "additive":
            return v.copy()
        return self._direction * np.log(np.maximum(v, _SMALLEST))

    def to_payload(self) -> dict[str, str]:
        return {"kind": self.kind, "link": self.link}

    @property
    def increasing(self) -> bool:
        """Whether the shown value rises with the effect; the overdispersion's falls."""
        return self.kind != "inverse_theta"

    @property
    def no_effect(self) -> float:
        """The shown value of a zero effect: 1 for a relativity, 0 for an added amount."""
        return 0.0 if self.kind == "additive" else 1.0


RELATIVITY = DisplayTransform("relativity", "log")
INVERSE_THETA = DisplayTransform("inverse_theta", "log")
ADDITIVE = DisplayTransform("additive", "identity")


def link_name(link: Link) -> str:
    """The short name of ``link`` that labels read: ``log``, ``floored_log``, ``logit``,
    ``identity``, else the class name without ``Link`` in snake case."""
    if isinstance(link, LogLink):
        return "log"
    if isinstance(link, LowerBoundedLogLink):
        return "floored_log"
    if isinstance(link, LogitLink):
        return "logit"
    if isinstance(link, IdentityLink):
        return "identity"
    # The probe crosses eta = 0, where an inverse link divides by zero.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shifted_log = _is_shifted_log_link(link)
    if shifted_log:
        return "floored_log"
    stem = type(link).__name__.removesuffix("Link") or type(link).__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def transform_for_link(link: Link, *, reciprocal: bool = False) -> DisplayTransform:
    """How a term on ``link`` is shown: a relativity on a log, shifted log or logit link,
    the added amount on any other.

    ``reciprocal`` shows the relativity of the parameter's reciprocal, which a plain
    log link alone makes a factor: ``1 / theta`` is ``exp(-eta)``.
    """
    name = link_name(link)
    if name == "log":
        return INVERSE_THETA if reciprocal else RELATIVITY
    if name in ("floored_log", "logit"):
        return DisplayTransform("relativity", name)
    return ADDITIVE if name == "identity" else DisplayTransform("additive", name)


@dataclass(frozen=True)
class ParameterDisplay:
    """A parameter's label, history chip, full name and identity colour."""

    name: str
    label: str
    chip: str
    role: Role
    colour: Colour
    full_name: str


def parameter_display(spec: ParameterSpec, *, average: bool) -> ParameterDisplay:
    """The label, chip and full name ``spec`` declares; ``average`` makes it the blue one.

    With no declared label the parameter's name is shown, capitalised. A ``size``
    parameter's chip is "1/θ", the overdispersion its curves show.
    """
    label = spec.label or spec.name.replace("_", " ").capitalize()
    role: Role = "average" if average else "spread"
    return ParameterDisplay(
        name=spec.name,
        label=label,
        chip="1/θ" if spec.role == "size" else label,
        role=role,
        colour=_COLOURS[role],
        full_name=spec.description or label,
    )


def average_parameter(family: Any) -> str:
    """The parameter shown as the average: role ``mean``, else ``location``, else the first."""
    parameters = family.parameters
    for role in ("mean", "location"):
        for spec in parameters:
            if spec.role == role:
                return spec.name
    return parameters[0].name


def parameter_spec(family: Any, parameter: str) -> ParameterSpec:
    """The ``ParameterSpec`` ``family`` declares for ``parameter``."""
    for spec in family.parameters:
        if spec.name == parameter:
            return spec
    names = ", ".join(repr(spec.name) for spec in family.parameters)
    raise EditorKeyError(
        f"{type(family).__name__} has no parameter {parameter!r}; its parameters are {names}."
    )


def parameter_display_for(model: Any, parameter: str) -> ParameterDisplay:
    """How ``parameter`` of the SuperLSS ``model`` (or the editor's view of one) is named."""
    family = _superlss(model).family
    return parameter_display(
        parameter_spec(family, parameter), average=parameter == average_parameter(family)
    )


def _of(quantity: str) -> str:
    """``quantity`` after "of": "the CV", but "σ" bare."""
    return quantity if len(quantity) == 1 else f"the {quantity}"


def y_label_for(transform: DisplayTransform, quantity: str) -> str:
    """What a term's curve shows of ``quantity``, as the chart's axis says it.

    The mean, a SuperGLM's quantity, goes unnamed on a relativity: "relativity",
    "relativity of the odds", "effect on the inverse scale".
    """
    if transform.kind == "inverse_theta":
        return "relativity of the overdispersion"
    named = quantity != "mean"
    if transform.kind == "relativity":
        what = "relativity of the odds" if transform.link == "logit" else "relativity"
        return f"{what} of {_of(quantity)}" if named else what
    if transform.link == "identity":
        return f"effect on {_of(quantity)}"
    scale = f"effect on the {transform.link.replace('_', ' ')} scale"
    return f"{scale} of {_of(quantity)}" if named else scale


def _quantity(spec: ParameterSpec, transform: DisplayTransform) -> str:
    """What the link of ``spec`` acts on, as a label names it: its ``quantity``, else its label.

    A ``size`` parameter's declared quantity is its reciprocal's, which only a
    reciprocal display shows; on any other its curve is of the parameter itself.
    """
    if spec.role == "size" and transform.kind != "inverse_theta":
        return spec.name
    if spec.quantity:
        return spec.quantity
    label = spec.label or spec.name.replace("_", " ")
    # An acronym (CV) keeps its capitals; a word reads in lower case.
    return label if label[:2].isupper() else label[:1].lower() + label[1:]


def _superlss(model: Any) -> SuperLSS | None:
    """The SuperLSS ``model`` is or the editor's view reads; ``None`` for a SuperGLM."""
    # Imported here: superglm.editor.lss imports this module.
    from superglm.editor.lss.view import LSSModelView

    if isinstance(model, LSSModelView):
        return model.lss
    return model if isinstance(model, SuperLSS) else None


def _split(key: str) -> tuple[str, str]:
    """The parameter and the term a SuperLSS key names."""
    from superglm.editor.lss.view import split_key

    return split_key(key)


def _fitted_link(lss: SuperLSS, parameter: str) -> Link:
    for state in lss._require_fitted().layout.predictors:
        if state.name == parameter:
            return state.link
    raise EditorKeyError(f"The fit has no predictor for parameter {parameter!r}.")


def display_for(model: Any, key: str) -> DisplayTransform:
    """How the term ``key`` of ``model`` is shown, from the link its predictor was fitted on."""
    lss = _superlss(model)
    if lss is None:
        link = getattr(model, "_link", None)
        return RELATIVITY if link is None else transform_for_link(link)
    parameter, _ = _split(key)
    spec = parameter_spec(lss.family, parameter)
    return transform_for_link(_fitted_link(lss, parameter), reciprocal=spec.role == "size")


def term_y_label(model: Any, key: str) -> str:
    """The axis label of the term ``key`` of ``model``'s curve (:func:`y_label_for`)."""
    transform = display_for(model, key)
    lss = _superlss(model)
    if lss is None:
        return y_label_for(transform, "mean")
    spec = parameter_spec(lss.family, _split(key)[0])
    return y_label_for(transform, _quantity(spec, transform))


@dataclass(frozen=True)
class FoldTermDisplay:
    """How the Cross-validation tab shows a term.

    ``label`` names it anywhere and ``short`` within its ``group``'s list;
    ``transform`` turns its link-scale effect into the curve drawn, as the Editor
    tab draws it, and ``value`` says what the curve shows. A term of a SuperGLM
    has no group.
    """

    label: str
    short: str
    transform: DisplayTransform = RELATIVITY
    group: str | None = None
    role: Role | None = None
    value: str = "relativity"


def fold_chart_display(key: str, model: Any) -> FoldTermDisplay:
    """How the Cross-validation tab shows the term ``key`` of ``model``.

    The curve is the one the Editor tab draws for the term, so a theta curve is
    the relativity of the overdispersion and rises with it.
    """
    transform = display_for(model, key)
    lss = _superlss(model)
    if lss is None:
        return FoldTermDisplay(
            label=key, short=key, transform=transform, value=term_y_label(model, key)
        )
    parameter, column = _split(key)
    shown = parameter_display_for(lss, parameter)
    return FoldTermDisplay(
        label=f"{column} · {shown.label}",
        short=column,
        transform=transform,
        group=shown.label,
        role=shown.role,
        value=term_y_label(lss, key),
    )


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
