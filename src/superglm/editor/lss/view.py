"""A fitted SuperLSS, read the way the editor reads a SuperGLM.

The editor keys every term by one string. A SuperLSS has the same term name in
several predictors, so its terms are keyed by the layout's own qualified name,
``"<parameter>:<term>"``. :class:`LSSModelView` answers the reads the editor
makes of a model (``_specs``, ``_feature_order``, ``_interaction_specs``,
``_result`` and ``term_inference``) under those keys, so the SuperGLM code paths
stay as they are.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.offsets import offset_columns, prediction_offsets
from superglm.distributional.prediction_design import _required_columns
from superglm.distributional.response import require_scalar_response
from superglm.distributional.revision import ShapeRepairedWarning
from superglm.distributional.terms import (
    _coefficient_columns,
    _compiled_predictor,
    _reference_design,
    _reference_value,
    _resolve_term,
    repaired_terms,
)
from superglm.distributional.weights import LikelihoodWeightError, resolve_likelihood_weights
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
    name: str, offset: Any, parameters: Sequence[str], frame: EagerFrame
) -> dict[str, NDArray] | None:
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
    owned: dict[str, NDArray] = {}
    for key, values in offset.items():
        # Verify labels before taking positional ownership. Keep raw values:
        # only the downstream retained-row resolver may validate numeric data.
        if isinstance(values, pd.Series):
            index = getattr(frame.native, "index", None)
            if index is not None and not values.index.equals(index):
                raise EditorValueError(
                    f"offset for {key!r} index must match X on the original rows"
                )
        array = np.array(
            values, dtype=object if isinstance(values, (list, tuple)) else None, copy=True
        )
        if array.shape != (len(frame),):
            raise EditorValueError(
                f"{name}_data offset for parameter {key!r} must have one value per row of y."
            )
        owned[key] = array
    return owned


def kept_rows(lss: SuperLSS, train: Any) -> Any:
    """The training rows the fit kept, which alone set the term grids and reference values.

    A Refit and the training scores still read every row of ``train``.
    """
    state = lss._require_fitted().fit_state
    if train.n_obs == state.weight_provenance.retained_count:
        return train.X
    if state.retained_rows is not None:
        positions = state.retained_rows.likelihood_weights.input_positions
    else:
        try:
            positions = resolve_likelihood_weights(
                train.sample_weight, n_observations=train.n_obs, contract=state.weight_contract
            ).input_positions
        except LikelihoodWeightError as exc:
            raise EditorValueError(
                "The editor finds the rows this model's fit kept from train_data's weights, "
                f"and this model cannot take them: {exc}. Open the editor with the rows and "
                "weights the model was fitted on: train_data=(X, y, sample_weight, offsets)."
            ) from exc
    return as_eager_frame(train.X).take_rows(np.asarray(positions, dtype=np.intp))


def fit_keeps_its_rows(lss: SuperLSS) -> bool:
    """Whether ``lss`` holds its fit's training rows: the frame and the arrays the fit kept."""
    return lss._require_fitted().fit_state.retained_rows is not None and (
        lss._training_frame is not None
    )


def rows_describe_fit(lss: SuperLSS, train: Any) -> bool:
    """Whether ``train`` holds the rows ``lss`` is fitted on now: as many as the fit was given
    or kept, with the same responses, weights, offsets and read columns on the rows it kept,
    each resolved as the fit resolves it, so that no weights match unit ones. A fit that kept
    no rows records nothing of its features or offsets, so no split can be shown to describe
    it. Read after a refit or repair in place (``follow_training_rows``), never at opening.
    """
    state = lss._require_fitted().fit_state
    provenance = state.weight_provenance
    if train.n_obs not in (provenance.original_count, provenance.retained_count):
        return False
    rows = state.retained_rows
    if rows is None:
        return False
    try:
        weights = resolve_likelihood_weights(
            train.sample_weight, n_observations=train.n_obs, contract=state.weight_contract
        )
        positions = np.asarray(weights.input_positions, dtype=np.intp)
        frame = as_eager_frame(train.X)
        offsets = prediction_offsets(
            state.predictor_templates, frame, train.offset, positions=positions
        )
        same_columns = _same_read_columns(lss, frame, positions)
    except (KeyError, LikelihoodWeightError, TypeError, ValueError):
        return False  # inputs the fit's own resolution refuses are not the fit's
    response = np.asarray(train.y, dtype=np.float64).reshape(-1)[positions]
    return (
        same_columns
        and np.array_equal(response, rows.response)
        and np.array_equal(weights.values, rows.likelihood_weights.values)
        and all(
            name in offsets and np.array_equal(offsets[name], values)
            for name, values in rows.offsets.items()
        )
    )


def _same_read_columns(lss: SuperLSS, frame: EagerFrame, positions: NDArray[np.intp]) -> bool:
    """Whether every column the model reads holds, at ``positions``, the fit's kept values;
    true where the fit holds no frame to compare (a model restored from a file)."""
    if lss._training_frame is None:
        return True
    kept = as_eager_frame(lss._training_frame)
    fitted = lss._require_fitted()
    names = dict.fromkeys(
        (
            *_required_columns(fitted.compiled_predictors),
            *offset_columns(fitted.fit_state.predictor_templates),
        )
    )
    return all(
        pd.Series(frame.column_array(name)[positions]).equals(pd.Series(kept.column_array(name)))
        for name in names
    )


@dataclass
class _FitReading:
    """What an ``LSSModelView`` read from one fit of its model: the compiled terms, keyed by
    ``lss_key``, and every answer kept from them. The view replaces it whole when the model's
    fit is replaced in place, so nothing read from an earlier fit outlives it."""

    fit: Any
    # The training rows the fit kept, which set the term grids and reference values.
    rows: Any
    specs: Mapping[str, Any]
    feature_order: list[str]
    interaction_specs: Mapping[str, Any]
    reference_row: pd.DataFrame | None = None
    column_scans: dict[str, tuple] = field(default_factory=dict)
    reference_designs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # Why Refit cannot run on the session's training rows, and why Final fit cannot fit its
    # splits, each with the rows it was read for.
    refit_reason: tuple[Any, str | None] | None = None
    final_split_reason: tuple[tuple[Any, ...], str | None] | None = None
    # Each term's sweep, by its arguments. A caller gets its own copy.
    term_inferences: dict[tuple[str, bool, int, str], Any] = field(default_factory=dict)
    # Each term's shape-rule rows (``lss.rules.term_rows``).
    term_rows: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, lss: SuperLSS, fitted: Any, rows: Any) -> _FitReading:
        specs: dict[str, Any] = {}
        interactions: dict[str, Any] = {}
        for parameter in lss.parameter_names_:
            compiled = _compiled_predictor(fitted, parameter).compiled
            for term in compiled.feature_order:
                specs[lss_key(parameter, term)] = compiled.specs[term]
            for name in compiled.interaction_order:
                interactions[lss_key(parameter, name)] = compiled.interaction_specs[name]
        return cls(
            fitted, rows, MappingProxyType(specs), list(specs), MappingProxyType(interactions)
        )


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
        splits: Mapping[str, Any] | None = None,
    ) -> None:
        fitted = lss._require_fitted()
        require_scalar_response(lss.family, "editor")
        self.lss = lss
        # The session's splits, shared, not copied (``share_splits``): their "train" split gives
        # each fit's kept rows; without one, ``X_train`` does.
        self._splits = splits
        self._given_rows = X_train
        self.amount_unit = amount_unit
        self.family = lss.family
        # The terms as declared, keyed like ``_specs``: what the knot and basis tools rebuild from.
        self._config = SimpleNamespace(
            feature_templates=tuple(
                (lss_key(predictor.name, term), declared)
                for predictor in lss._predictors
                for term, declared in predictor.features.items()
            )
        )
        self._read = self._reading_of(fitted, opening=True)

    @property
    def _train(self) -> Any:
        """The session's training split as it is now: one source for every view of the session."""
        return None if self._splits is None else self._splits.get("train")

    def share_splits(self, splits: Mapping[str, Any]) -> None:
        """Read the training split from ``splits``, the session's own, from now on."""
        self._splits = splits

    def _reading_of(self, fitted: Any, *, opening: bool = False) -> _FitReading:
        if self._train is None:
            rows = self._given_rows
        elif opening:
            rows = kept_rows(self.lss, self._train)
        elif fit_keeps_its_rows(self.lss):
            # After a refit or repair in place, the rows the fit kept (``follow_training_rows``).
            rows = self.lss._training_frame
        elif rows_describe_fit(self.lss, self._train):
            rows = kept_rows(self.lss, self._train)
        else:
            rows = self.lss._term_training_frame(None)
        return _FitReading.of(self.lss, fitted, rows)

    @property
    def _reading(self) -> _FitReading:
        """What the view read from the fit ``lss`` predicts with now. A refit or repair in
        place (``apply_shape_postfit``) replaces that fit, and with it everything read here."""
        fitted = self.lss._require_fitted()
        if fitted is not self._read.fit:
            self._read = self._reading_of(fitted)
        return self._read

    @property
    def _fitted(self):
        """The fit ``lss`` predicts with now (``_reading``)."""
        return self._reading.fit

    @property
    def X_train(self) -> Any:
        return self._reading.rows

    @property
    def _specs(self) -> Mapping[str, Any]:
        return self._reading.specs

    @property
    def _feature_order(self) -> list[str]:
        return self._reading.feature_order

    @property
    def _interaction_specs(self) -> Mapping[str, Any]:
        return self._reading.interaction_specs

    @property
    def _column_scans(self) -> dict[str, tuple]:
        return self._reading.column_scans

    @property
    def _term_rows(self) -> dict[str, Any]:
        return self._reading.term_rows

    @property
    def _reference_designs(self) -> dict[str, NDArray[np.float64]]:
        return self._reading.reference_designs

    @property
    def _refit_reason(self) -> tuple[Any, str | None] | None:
        return self._reading.refit_reason

    @_refit_reason.setter
    def _refit_reason(self, value: tuple[Any, str | None]) -> None:
        self._reading.refit_reason = value

    @property
    def _final_split_reason(self) -> tuple[tuple[Any, ...], str | None] | None:
        return self._reading.final_split_reason

    @_final_split_reason.setter
    def _final_split_reason(self, value: tuple[tuple[Any, ...], str | None]) -> None:
        self._reading.final_split_reason = value

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
        return LSSModelView(
            lss, self._given_rows, amount_unit=self.amount_unit, splits=self._splits
        )

    def term_coefficients(self, key: str) -> NDArray[np.float64]:
        """The fitted coefficients of the term ``key`` names, as its spec scores them."""
        parameter, term = split_key(key)
        qualified, _state, term_slice = _resolve_term(self._fitted, parameter, term)
        special = self._fitted.layout.term_slices.get(f"{qualified}{_SPECIAL_SUFFIX}")
        blocks = (term_slice,) if special is None else (term_slice, special)
        return np.asarray(self._fitted.coefficients, dtype=np.float64)[
            _coefficient_columns(*blocks)
        ]

    def term_inference(
        self,
        name: str,
        *,
        with_se: bool = True,
        n_points: int = 200,
        centering: str = "native",
    ) -> LSSTermInference:
        """One term on its link scale, in the fields the editor builds an ``EditableTerm`` from.

        A term repaired after the fit carries no band, as ``SuperGLM``'s editor shows a
        repaired term: its standard errors are withheld.
        """
        if centering != "native":
            raise EditorValueError(NATIVE_ONLY)
        key = (name, with_se, n_points, centering)
        reading = self._reading
        if key not in reading.term_inferences:
            parameter, term = split_key(name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ShapeRepairedWarning)
                effect = self.lss.term_inference(
                    parameter, term, X_train=self.X_train, n_points=n_points, simultaneous=False
                )
            repaired = name in repaired_terms(reading.fit)
            reading.term_inferences[key] = lss_term_inference(
                name, effect, reading.specs[name], with_se=with_se and not repaired
            )
        return copy.deepcopy(reading.term_inferences[key])

    def data_columns(self) -> tuple[str, ...]:
        """The model's feature, interaction-parent and offset columns."""
        return tuple(
            dict.fromkeys(
                (
                    *_required_columns(self._fitted.compiled_predictors),
                    *offset_columns(self._fitted.fit_state.predictor_templates),
                )
            )
        )

    def reference_row(self) -> pd.DataFrame:
        """One row with every column the model reads at its training centre.

        A numeric column sits at its median and any other at its most frequent
        level, as a term sweep holds the columns it is not sweeping. The row is
        computed once; each call returns a copy.
        """
        reading = self._reading
        if reading.reference_row is None:
            frame = as_eager_frame(self.X_train)
            # One row, even for a model that reads no column.
            reading.reference_row = pd.DataFrame(
                {
                    name: [_reference_value(self._fitted, frame, name)]
                    for name in self.data_columns()
                },
                index=[0],
            )
        return reading.reference_row.copy()

    def _term_blocks(self, key: str) -> tuple[Any, tuple[slice, ...]]:
        """The term's predictor state and its coefficient blocks, its free levels' last."""
        parameter, term = split_key(key)
        qualified, state, term_slice = _resolve_term(self._fitted, parameter, term)
        special = self._fitted.layout.term_slices.get(f"{qualified}{_SPECIAL_SUFFIX}")
        return state, (term_slice,) if special is None else (term_slice, special)

    def reference_design(self, key: str) -> NDArray[np.float64]:
        """The term's own columns at its reference, which its shown effect is measured from."""
        designs = self._reading.reference_designs
        if key not in designs:
            parameter, term = split_key(key)
            state, blocks = self._term_blocks(key)
            designs[key] = _reference_design(
                self._fitted,
                as_eager_frame(self.X_train),
                parameter,
                term,
                state,
                *blocks,
                reference=self.reference_row,
            )
        return designs[key]

    def reference_log_effect(self, key: str) -> float:
        """The term's scored contribution at its reference, which its shown effect leaves out."""
        _, blocks = self._term_blocks(key)
        beta = np.asarray(self._fitted.coefficients, dtype=np.float64)
        return float(self.reference_design(key) @ beta[_coefficient_columns(*blocks)])
