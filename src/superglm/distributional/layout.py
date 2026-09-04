"""Global coefficient and penalty layout for distributional predictors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.predictor import CompiledPredictor
from superglm.group_matrix import DesignMatrix
from superglm.links import Link
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.types import GroupSlice, PenaltyComponent


def _readonly_array(value: NDArray | None) -> NDArray | None:
    if value is None:
        return None
    owned = np.array(value, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


def _qualified_term(name: str, predictors: set[str]) -> bool:
    namespace, separator, local_name = name.partition(":")
    return bool(separator and namespace in predictors and local_name and "#" not in local_name)


def _qualified_penalty(name: str, predictors: set[str]) -> bool:
    term, marker, suffix = name.rpartition("#")
    return bool(marker and suffix and _qualified_term(term, predictors))


@dataclass(frozen=True)
class PredictorState:
    """Immutable global placement and local design state for one predictor."""

    name: str
    parameter_index: int
    link: Link
    design: DesignMatrix
    groups: tuple[GroupSlice, ...]
    coefficient_slice: slice
    intercept_index: int | None
    offset: NDArray[np.float64]
    penalties: tuple[PenaltyComponent, ...]


@dataclass(frozen=True)
class StackedLayout:
    """One authoritative coordinate system for all predictor coefficients."""

    predictors: tuple[PredictorState, ...]
    n_coefficients: int
    coefficient_names: tuple[str, ...]
    term_slices: Mapping[str, slice]
    penalties: tuple[PenaltyComponent, ...]

    def __post_init__(self) -> None:
        predictor_names = tuple(state.name for state in self.predictors)
        namespaces = set(predictor_names)
        if len(namespaces) != len(predictor_names):
            raise ValueError("stacked layout has duplicate predictor names")
        if len(self.coefficient_names) != self.n_coefficients:
            raise ValueError("coefficient_names must contain one name per coefficient")
        if any(not _qualified_term(name, namespaces) for name in self.coefficient_names):
            raise ValueError("coefficient names must be predictor-qualified")

        terms = dict(self.term_slices)
        if any(not _qualified_term(name, namespaces) for name in terms):
            raise ValueError("qualified predictor namespace required for every term name")
        if len(terms) != len(self.term_slices):
            raise ValueError("stacked layout has duplicate term names")

        penalty_names = tuple(component.name for component in self.penalties)
        if any(not _qualified_penalty(name, namespaces) for name in penalty_names):
            raise ValueError("qualified predictor namespace required for every penalty name")
        if len(set(penalty_names)) != len(penalty_names):
            raise ValueError("stacked layout has duplicate penalty names")

        expected_start = 0
        for parameter_index, state in enumerate(self.predictors):
            if state.parameter_index != parameter_index:
                raise ValueError("predictor states must remain in family parameter order")
            if state.coefficient_slice.start != expected_start:
                raise ValueError(
                    "predictor coefficient slices must be contiguous and non-overlapping"
                )
            expected_start = state.coefficient_slice.stop
        if expected_start != self.n_coefficients:
            raise ValueError("predictor coefficient slices do not cover the global layout")

        object.__setattr__(self, "term_slices", MappingProxyType(terms))
        object.__setattr__(self, "predictors", tuple(self.predictors))
        object.__setattr__(self, "coefficient_names", tuple(self.coefficient_names))
        object.__setattr__(self, "penalties", tuple(self.penalties))

        # ``_center_selected_smooths`` addresses ``design.group_matrices`` with an
        # index taken from ``groups``, so the two must stay index-parallel.  Nothing
        # in the layout consumes the pairing, but this is the one place the
        # assembled state is checked, so the count check stays.
        for state in self.predictors:
            if len(state.design.group_matrices) != len(state.groups):
                raise ValueError(
                    f"predictor {state.name!r} has {len(state.design.group_matrices)} "
                    f"group matrices for {len(state.groups)} groups"
                )

    @property
    def penalty_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.penalties)

    def predictor(self, name: str) -> PredictorState:
        for state in self.predictors:
            if state.name == name:
                return state
        raise KeyError(f"unknown predictor {name!r}")

    def penalty_matrix(self, lambdas: Mapping[str, float]) -> NDArray[np.float64]:
        """Embed weighted component penalties without cross-predictor terms."""
        expected = set(self.penalty_names)
        unknown = set(lambdas) - expected
        missing = expected - set(lambdas)
        if unknown:
            raise ValueError(f"unknown penalty lambda names: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing penalty lambda names: {sorted(missing)}")

        result = np.zeros((self.n_coefficients, self.n_coefficients), dtype=np.float64)
        for component in self.penalties:
            lam = float(lambdas[component.name])
            if not np.isfinite(lam) or lam < 0.0:
                raise ValueError(f"lambda for {component.name!r} must be finite and nonnegative")
            # No group matrix is passed: every component ``build_penalty_components``
            # produces for a predictor already carries its solver-space ``omega_ssp``
            # (the ``identity`` kind needs no matrix at all), so a component that
            # reaches here without one is malformed and must fail loudly rather than
            # be lifted from ``omega_raw`` against a matrix looked up by index.
            omega = penalty_component_dense_matrix(component)
            width = component.group_sl.stop - component.group_sl.start
            if omega.shape != (width, width):
                raise ValueError(f"penalty {component.name!r} does not match its global slice")
            result[component.group_sl, component.group_sl] += lam * omega
        return result


def _penalty_suffix(component: PenaltyComponent, group: GroupSlice) -> str:
    if component.name == group.name:
        return "wiggle"
    prefix = f"{group.name}:"
    if not component.name.startswith(prefix):
        raise ValueError(f"penalty {component.name!r} does not match local group {group.name!r}")
    suffix = component.name[len(prefix) :]
    if not suffix or "#" in suffix:
        raise ValueError(f"invalid local penalty suffix {suffix!r}")
    return suffix


def _validated_component_group(
    component: PenaltyComponent,
    *,
    predictor: CompiledPredictor,
) -> tuple[GroupSlice, str]:
    local_width = predictor.compiled.design.p
    local_slice = component.group_sl
    if (
        local_slice.start is None
        or local_slice.stop is None
        or local_slice.start < 0
        or local_slice.stop < local_slice.start
        or local_slice.stop > local_width
    ):
        raise ValueError(f"penalty {component.name!r} lies outside predictor {predictor.name!r}")
    if component.group_index < 0 or component.group_index >= len(predictor.compiled.groups):
        raise ValueError(f"penalty {component.name!r} has an invalid local group index")
    group = predictor.compiled.groups[component.group_index]
    if component.group_name != group.name or component.group_sl != group.sl:
        raise ValueError(
            f"penalty {component.name!r} does not match its declared local group block"
        )
    return group, _penalty_suffix(component, group)


def _copy_component(
    component: PenaltyComponent,
    *,
    name: str,
    group_name: str,
    group_index: int,
    group_sl: slice,
) -> PenaltyComponent:
    # ``dataclasses.replace`` carries every field the source component declares.
    # Enumerating them by hand here is what silently dropped ``penalty_kind``,
    # ``repeat_count`` and ``block_width`` when master widened the contract.
    return replace(
        component,
        name=name,
        group_name=group_name,
        group_index=group_index,
        group_sl=group_sl,
        omega_raw=_readonly_array(component.omega_raw),
        omega_ssp=_readonly_array(component.omega_ssp),
        eigvals_omega=_readonly_array(component.eigvals_omega),
    )


def _qualify_local_component(
    component: PenaltyComponent,
    *,
    predictor: CompiledPredictor,
) -> PenaltyComponent:
    group, suffix = _validated_component_group(component, predictor=predictor)
    qualified_group = f"{predictor.name}:{group.name}"
    return _copy_component(
        component,
        name=f"{qualified_group}#{suffix}",
        group_name=qualified_group,
        group_index=component.group_index,
        group_sl=component.group_sl,
    )


def _embed_component(
    component: PenaltyComponent,
    *,
    predictor: CompiledPredictor,
    slope_start: int,
    global_group_offset: int,
) -> PenaltyComponent:
    group, suffix = _validated_component_group(component, predictor=predictor)

    qualified_group = f"{predictor.name}:{group.name}"
    return _copy_component(
        component,
        name=f"{qualified_group}#{suffix}",
        group_name=qualified_group,
        group_index=global_group_offset + component.group_index,
        group_sl=slice(
            slope_start + component.group_sl.start,
            slope_start + component.group_sl.stop,
        ),
    )


def build_stacked_layout(builds: Sequence[CompiledPredictor]) -> StackedLayout:
    """Place locally compiled predictors in one deterministic global layout."""
    compiled = tuple(builds)
    if not compiled:
        raise ValueError("at least one compiled predictor is required")

    states: list[PredictorState] = []
    coefficient_names: list[str] = []
    term_slices: dict[str, slice] = {}
    penalties: list[PenaltyComponent] = []
    global_start = 0
    global_group_offset = 0

    for expected_index, predictor in enumerate(compiled):
        if predictor.parameter_index != expected_index:
            raise ValueError("compiled predictors must remain in family parameter order")
        local_width = predictor.compiled.design.p
        intercept_width = int(predictor.intercept)
        slope_start = global_start + intercept_width
        global_stop = slope_start + local_width
        intercept_index = global_start if predictor.intercept else None
        if predictor.intercept:
            coefficient_names.append(f"{predictor.name}:(intercept)")

        expected_local_start = 0
        seen_local_names: set[str] = set()
        for group in predictor.compiled.groups:
            if group.name in seen_local_names:
                raise ValueError(f"duplicate local term name {group.name!r}")
            seen_local_names.add(group.name)
            if group.start != expected_local_start or group.end < group.start:
                raise ValueError("local predictor groups must be contiguous and non-overlapping")
            expected_local_start = group.end
            qualified_name = f"{predictor.name}:{group.name}"
            if "#" in group.name:
                raise ValueError("local term names may not contain '#' in a stacked layout")
            term_slices[qualified_name] = slice(
                slope_start + group.start,
                slope_start + group.end,
            )
            if group.size == 1:
                coefficient_names.append(qualified_name)
            else:
                coefficient_names.extend(
                    f"{qualified_name}[{column}]" for column in range(group.size)
                )
        if expected_local_start != local_width:
            raise ValueError("local predictor groups do not cover the design columns")

        embedded = tuple(
            _embed_component(
                component,
                predictor=predictor,
                slope_start=slope_start,
                global_group_offset=global_group_offset,
            )
            for component in predictor.penalties
        )
        local_penalties = tuple(
            _qualify_local_component(component, predictor=predictor)
            for component in predictor.penalties
        )
        penalties.extend(embedded)
        states.append(
            PredictorState(
                name=predictor.name,
                parameter_index=predictor.parameter_index,
                link=predictor.link,
                design=predictor.compiled.design,
                groups=predictor.compiled.groups,
                coefficient_slice=slice(global_start, global_stop),
                intercept_index=intercept_index,
                offset=predictor.offset,
                penalties=local_penalties,
            )
        )
        global_start = global_stop
        global_group_offset += len(predictor.compiled.groups)

    return StackedLayout(
        predictors=tuple(states),
        n_coefficients=global_start,
        coefficient_names=tuple(coefficient_names),
        term_slices=MappingProxyType(term_slices),
        penalties=tuple(penalties),
    )
