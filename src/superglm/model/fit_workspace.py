"""Attempt-local model state for strongly exception-safe fitting."""

from __future__ import annotations

import copy
from dataclasses import dataclass

_ATTEMPT_RUNTIME_OPTIONS = (
    "_select_snap",
    "_suppress_reporting_support",
    "_discard_rank_info",
    "_random_effect_nesting_checked",
    # Profile-search design cache: shared BY REFERENCE so every candidate
    # workspace fills and reads the same dict (fit_ops._fetch_or_build_design).
    "_profile_design_cache",
)
_SUBCLASS_STATE_NAMES = "_fit_workspace_subclass_state_names"


def _copy_subclass_state(public_model, work_model) -> None:
    """Copy constructor-owned extension state into an isolated fit attempt.

    ``ModelConfig`` owns only the base ``SuperGLM`` contract, so on a
    subclass's first fit the attributes absent from the clean materialized
    dictionary are its constructor state. The names are remembered so later
    fitted buffers are not mistaken for subclass state.
    """
    tracked_names = getattr(public_model, _SUBCLASS_STATE_NAMES, None)
    if tracked_names is None:
        base_names = set(work_model.__dict__)
        tracked_names = tuple(
            sorted(
                name
                for name in public_model.__dict__
                if name not in base_names and name not in _ATTEMPT_RUNTIME_OPTIONS
            )
        )
    else:
        tracked_names = tuple(tracked_names)

    if not tracked_names:
        work_model.__dict__[_SUBCLASS_STATE_NAMES] = tracked_names
        return

    base_names = set(work_model.__dict__)

    # One memo keeps aliases between extension attributes; mapping the source
    # model to the workspace keeps self-references from cloning the public model.
    memo = {id(public_model): work_model}
    scalar_types = (type(None), bool, int, float, complex, str, bytes)
    for name in base_names:
        if name not in public_model.__dict__:
            continue
        source_value = public_model.__dict__[name]
        workspace_value = work_model.__dict__[name]
        if source_value is workspace_value or not isinstance(source_value, scalar_types):
            memo[id(source_value)] = workspace_value
    for name in tracked_names:
        if name not in public_model.__dict__:
            raise RuntimeError(f"tracked subclass fit state {name!r} is missing")
        try:
            work_model.__dict__[name] = copy.deepcopy(public_model.__dict__[name], memo)
        except Exception as exc:  # pragma: no cover - depends on extension object
            raise TypeError(
                f"subclass fit state {name!r} must support deepcopy for transactional fitting"
            ) from exc
    work_model.__dict__[_SUBCLASS_STATE_NAMES] = tracked_names


@dataclass
class FitWorkspace:
    """Fresh mutable state for one fit attempt."""

    model: object
    mode: str
    validated_inputs: object | None
    previous_revision: int

    @classmethod
    def start(
        cls,
        public_model,
        *,
        mode: str,
        validated_inputs,
        config_overrides: dict[str, object] | None = None,
    ):
        """Materialize only configuration intent, never prior fitted buffers."""
        config = public_model._config
        if config_overrides:
            config = config.with_value(**config_overrides)
        work_model = config.materialize(type(public_model))
        _copy_subclass_state(public_model, work_model)
        for name in _ATTEMPT_RUNTIME_OPTIONS:
            if hasattr(public_model, name):
                setattr(work_model, name, getattr(public_model, name))
        return cls(
            model=work_model,
            mode=mode,
            validated_inputs=validated_inputs,
            previous_revision=int(public_model._fit_revision),
        )
