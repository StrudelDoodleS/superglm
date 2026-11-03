"""A fitted SuperLSS, read the way the editor reads a SuperGLM.

The editor keys every term by one string. A SuperLSS has the same term name in
several predictors, so its terms are keyed by the layout's own qualified name,
``"<parameter>:<term>"``. :class:`LSSModelView` answers the reads the editor
makes of a model (``_specs``, ``_feature_order``, ``_interaction_specs``,
``_result`` and ``term_inference``) under those keys, so the SuperGLM code paths
stay as they are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.prediction_design import _required_columns
from superglm.distributional.terms import (
    _coefficient_columns,
    _compiled_predictor,
    _reference_value,
    _resolve_term,
    _term_design,
)
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss.terms import LSSTermInference, lss_term_inference
from superglm.features.rebuild import interaction_users as glm_interaction_users

NATIVE_ONLY = "SuperLSS terms open on their native scale; pass centering='native'."
_SPECIAL_SUFFIX = ":special"


def lss_key(parameter: str, term: str) -> str:
    """The editor's key for ``term`` in the predictor of ``parameter``."""
    return f"{parameter}:{term}"


def split_key(key: str) -> tuple[str, str]:
    """The parameter and the term a key names; an interaction keeps its own colon."""
    parameter, separator, term = key.partition(":")
    if not separator or not parameter or not term:
        raise EditorKeyError(
            f"{key!r} is not a SuperLSS term key; expected '<parameter>:<term>', such as "
            "'mean:age'."
        )
    return parameter, term


def is_lss(model: object) -> bool:
    """Whether ``model`` is a SuperLSS or the view the editor holds of one."""
    return isinstance(model, SuperLSS | LSSModelView)


def predictor_of(model: object, term: str | None) -> str | None:
    """The parameter a SuperLSS key belongs to; ``None`` for a SuperGLM or no term."""
    if term is None or not is_lss(model):
        return None
    return split_key(term)[0]


def column_of(model: object, name: str) -> str:
    """The data column a term reads: a SuperLSS key's term, else ``name`` itself."""
    return split_key(name)[1] if is_lss(model) else name


def interaction_users(model: object, term: str) -> list[str]:
    """The interactions that use ``term`` as a parent; a SuperLSS key looks in its own predictor."""
    if not is_lss(model):
        return glm_interaction_users(model, term)
    parameter, column = split_key(term)
    return [
        name
        for name, spec in model._interaction_specs.items()
        if split_key(name)[0] == parameter and column in getattr(spec, "parent_names", ())
    ]


def parameter_offsets(
    name: str, offset: Any, parameters: Sequence[str], n_obs: int
) -> dict[str, NDArray[np.float64]] | None:
    """A split's per-parameter link-scale offsets, keyed by family parameter name."""
    if offset is None:
        return None
    names = ", ".join(parameters)
    if not isinstance(offset, Mapping):
        raise EditorValueError(
            f"Offsets for a SuperLSS model are a dict keyed by parameter name ({names}); "
            "got an array."
        )
    unknown = next((key for key in offset if key not in parameters), None)
    if unknown is not None:
        raise EditorValueError(
            f"Unknown offset parameter {unknown!r}; this model's parameters are {names}."
        )
    owned: dict[str, NDArray[np.float64]] = {}
    for key, values in offset.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (n_obs,):
            raise EditorValueError(
                f"{name}_data offset for parameter {key!r} must have one value per row of y."
            )
        owned[key] = array
    return owned


class LSSModelView:
    """The editor's view of a fitted SuperLSS and the rows it was fitted on.

    ``X_train`` is the editor's own copy of the training frame: a model restored
    from a file carries none, and every term sweep needs it. ``amount_unit`` only labels what the editor shows.
    """

    def __init__(
        self,
        lss: SuperLSS,
        X_train: Any,
        *,
        amount_unit: str | None = None,
    ) -> None:
        fitted = lss._require_fitted()
        self.lss = lss
        self.X_train = X_train
        self.amount_unit = amount_unit
        self.family = lss.family
        self._fitted = fitted
        specs: dict[str, Any] = {}
        interactions: dict[str, Any] = {}
        for parameter in lss.parameter_names_:
            compiled = _compiled_predictor(fitted, parameter).compiled
            for term in compiled.feature_order:
                specs[lss_key(parameter, term)] = compiled.specs[term]
            for name in compiled.interaction_order:
                interactions[lss_key(parameter, name)] = compiled.interaction_specs[name]
        self._specs: Mapping[str, Any] = MappingProxyType(specs)
        self._feature_order: list[str] = list(specs)
        self._interaction_specs: Mapping[str, Any] = MappingProxyType(interactions)
        # The terms as declared, keyed like ``_specs``: what the knot and basis tools rebuild from.
        self._config = SimpleNamespace(
            feature_templates=tuple(
                (lss_key(predictor.name, term), declared)
                for predictor in lss._predictors
                for term, declared in predictor.features.items()
            )
        )
        self._reference_row: pd.DataFrame | None = None
        self._reference_etas: dict[str, float] = {}
        # Each edited term's change to its predictor at the reference row, by its curve.
        self._reference_edits: dict[str, tuple[tuple[bytes, bytes], float]] = {}

    @property
    def _result(self):
        return self.lss.result_

    @property
    def _weight_semantics(self) -> str:
        return self.lss.weight_semantics

    @property
    def _discrete(self) -> bool:
        return self.lss.discrete

    @property
    def _n_bins(self) -> int | dict[str, int]:
        return self.lss._n_bins_config()

    def parameter_of(self, key: str) -> str:
        return split_key(key)[0]

    def column_of(self, key: str) -> str:
        return split_key(key)[1]

    def with_model(self, lss: SuperLSS) -> LSSModelView:
        """A view of ``lss``, another fit of the same model, over the same rows."""
        return LSSModelView(lss, self.X_train, amount_unit=self.amount_unit)

    def term_inference(
        self,
        name: str,
        *,
        with_se: bool = True,
        n_points: int = 200,
        centering: str = "native",
    ) -> LSSTermInference:
        """One term on its link scale, in the fields the editor builds an ``EditableTerm`` from."""
        if centering != "native":
            raise EditorValueError(NATIVE_ONLY)
        parameter, term = split_key(name)
        effect = self.lss.term_inference(
            parameter, term, X_train=self.X_train, n_points=n_points, simultaneous=False
        )
        return lss_term_inference(name, effect, self._specs[name], with_se=with_se)

    def data_columns(self) -> tuple[str, ...]:
        """The data columns the model reads: every predictor's features and interaction parents."""
        return _required_columns(self._fitted.compiled_predictors)

    def reference_row(self) -> pd.DataFrame:
        """One row with every column the model reads at its training centre.

        A numeric column sits at its median and any other at its most frequent
        level, as a term sweep holds the columns it is not sweeping. The row is
        computed once; each call returns a copy.
        """
        if self._reference_row is None:
            frame = as_eager_frame(self.X_train)
            self._reference_row = pd.DataFrame(
                {
                    name: [_reference_value(self._fitted, frame, name)]
                    for name in self.data_columns()
                }
            )
        return self._reference_row.copy()

    def reference_eta(self, key: str) -> float:
        """The fitted link predictor of ``key``'s parameter at the reference row, less the term's part.

        What the parameter would read at the reference row if this term's effect
        were zero there: adding the term's effect at any value of its column
        gives the parameter at the reference row with that column changed. The
        other edits of the parameter add ``lss.apply.reference_shift``.
        """
        if key not in self._reference_etas:
            parameter, term = split_key(key)
            qualified, state, term_slice = _resolve_term(self._fitted, parameter, term)
            special = self._fitted.layout.term_slices.get(f"{qualified}{_SPECIAL_SUFFIX}")
            blocks = (term_slice,) if special is None else (term_slice, special)
            row = self.reference_row()
            design = _term_design(self._fitted, row, state, *blocks)
            beta = np.asarray(self._fitted.coefficients, dtype=np.float64)
            own = float((design @ beta[_coefficient_columns(*blocks)])[0])
            eta = float(self.lss.predict_link(row)[parameter].iloc[0])
            self._reference_etas[key] = eta - own
        return self._reference_etas[key]
