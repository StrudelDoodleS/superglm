"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import copy
import copyreg
import dataclasses
import logging
import re
import threading
import types
from contextlib import contextmanager
from dataclasses import fields, replace

import numpy as np

from superglm.distributions import NegativeBinomial, Tweedie
from superglm.profiling._reporting import cached_tweedie_profile_ci

logger = logging.getLogger(__name__)

_TWEEDIE_PROFILE_REGISTRY_LOCK = threading.Lock()
_ACTIVE_TWEEDIE_PROFILE_CALLS: dict[int, object] = {}
_TWEEDIE_RLOCK_TYPE = type(threading.RLock())

_TWEEDIE_EXACT_REAL_SCALAR_TYPES = {
    int,
    float,
    *{
        scalar_type
        for scalar_type in np.sctypeDict.values()
        if isinstance(scalar_type, type)
        and issubclass(scalar_type, np.integer | np.floating)
        and not issubclass(scalar_type, np.bool_)
    },
}
_TWEEDIE_EXACT_INTEGER_SCALAR_TYPES = {
    int,
    *{
        scalar_type
        for scalar_type in np.sctypeDict.values()
        if isinstance(scalar_type, type)
        and issubclass(scalar_type, np.integer)
        and not issubclass(scalar_type, np.bool_)
    },
}
_TWEEDIE_EXACT_NUMPY_SCALAR_TYPES = {
    scalar_type
    for scalar_type in np.sctypeDict.values()
    if isinstance(scalar_type, type)
    and issubclass(scalar_type, np.generic)
    and not issubclass(scalar_type, np.void)
}


def _is_finite_tweedie_scalar(value) -> bool:
    """Accept one immutable built-in/NumPy real scalar without coercion hooks."""
    if type(value) not in _TWEEDIE_EXACT_REAL_SCALAR_TYPES:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_finite_tweedie_vector(value, *, length=None, positive=False) -> bool:
    """Validate exact owned-style numeric vector storage without mask coercion."""
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.kind not in "fiu":
        return False
    if length is not None and value.shape != (length,):
        return False
    if not np.all(np.isfinite(value)):
        return False
    return not positive or bool(np.all(value > 0.0))


def _is_finite_tweedie_matrix(value, *, shape=None) -> bool:
    """Validate exact numeric matrix storage without coercion or mask loss."""
    if type(value) is not np.ndarray or value.ndim != 2 or value.dtype.kind not in "fiu":
        return False
    if shape is not None and value.shape != shape:
        return False
    return bool(np.all(np.isfinite(value)))


def _is_tweedie_bool_vector(value, *, length=None) -> bool:
    """Validate an exact, unmasked NumPy boolean vector."""
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.kind != "b":
        return False
    return length is None or value.shape == (length,)


def _live_tweedie_profile_context(result):
    """Return the shared authoritative evaluator context, when still attached."""
    from superglm.profiling.tweedie import (
        _ProfileContext,
        _ProfileContextREML,
        _TweedieProfileEvaluator,
    )

    evaluator = getattr(result, "_evaluator", None)
    if type(evaluator) is _TweedieProfileEvaluator:
        owner = evaluator.context
        return owner if type(owner) in {_ProfileContext, _ProfileContextREML} else None

    callbacks = tuple(
        getattr(result, name, None)
        for name in ("_objective", "_evaluation_count", "_evaluation_record")
    )
    if not all(callable(callback) for callback in callbacks):
        return None
    owner = getattr(callbacks[0], "__self__", None)
    if type(owner) not in {_ProfileContext, _ProfileContextREML}:
        return None
    if any(getattr(callback, "__self__", None) is not owner for callback in callbacks[1:]):
        return None
    return owner


@contextmanager
def _exclusive_tweedie_profile_call(model):
    """Fail fast when the same model already has an active profile transaction."""
    key = id(model)
    with _TWEEDIE_PROFILE_REGISTRY_LOCK:
        if key in _ACTIVE_TWEEDIE_PROFILE_CALLS:
            raise RuntimeError(
                "Tweedie power estimation is already active for this model; "
                "wait for it to finish or use a separate model"
            )
        _ACTIVE_TWEEDIE_PROFILE_CALLS[key] = model
    try:
        yield
    finally:
        with _TWEEDIE_PROFILE_REGISTRY_LOCK:
            if _ACTIVE_TWEEDIE_PROFILE_CALLS.get(key) is model:
                del _ACTIVE_TWEEDIE_PROFILE_CALLS[key]


def _tweedie_profile_instance_dict(model) -> dict:
    """Read the real instance dictionary without subclass descriptor dispatch."""
    from superglm.model.api import SuperGLM

    model_mro = type.__getattribute__(type(model), "__mro__")
    if any(candidate_type is SuperGLM for candidate_type in model_mro):
        base_dict = type.__getattribute__(SuperGLM, "__dict__")
        descriptor = base_dict["__dict__"]
        mapping = descriptor.__get__(model, type(model))
    else:
        mapping = object.__getattribute__(model, "__dict__")
    if type(mapping) is not dict:
        raise TypeError("Tweedie profile refitting requires an exact model dictionary")
    return mapping


def estimate_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    phi_method="mle",
    method="brent",
    progress_callback=None,
    eager_ci_alpha=None,
    **kwargs,
):
    """Estimate Tweedie p via profile likelihood, refit, and return result."""
    with _exclusive_tweedie_profile_call(model):
        caller_mapping = _tweedie_profile_instance_dict(model)
        caller_state = caller_mapping.copy()
        try:
            return _estimate_p_transaction(
                model,
                X,
                y,
                sample_weight,
                offset,
                fit_mode=fit_mode,
                phi_method=phi_method,
                method=method,
                progress_callback=progress_callback,
                eager_ci_alpha=eager_ci_alpha,
                _caller_state=caller_state,
                **kwargs,
            )
        except BaseException:
            _restore_tweedie_profile_model_state(model, caller_mapping, caller_state)
            raise


def _estimate_p_transaction(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    phi_method="mle",
    method="brent",
    progress_callback=None,
    eager_ci_alpha=None,
    _caller_state=None,
    **kwargs,
):
    """Run one exclusively owned Tweedie profile/refit transaction."""
    from superglm.profiling.tweedie import (
        _clone_profile_model,
        _prepare_tweedie_profile_inputs,
        _use_prepared_tweedie_profile_inputs,
        estimate_tweedie_p,
    )

    _validate_tweedie_profile_model_storage(model)
    caller_state = (
        _tweedie_profile_instance_dict(model).copy() if _caller_state is None else _caller_state
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    eager_ci_alpha = _normalize_eager_tweedie_ci_alpha(eager_ci_alpha)
    if eager_ci_alpha is not None and phi_method == "pearson":
        raise ValueError(
            "eager_ci_alpha requires phi_method='mle'; likelihood-ratio intervals "
            "are unavailable for Pearson plug-in profiles"
        )
    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)
    prepared = _prepare_tweedie_profile_inputs(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        fit_mode=resolved_mode,
        phi_method=phi_method,
        method=method,
        **kwargs,
    )
    X_snapshot, y_snapshot, weight_snapshot, offset_snapshot = (
        _snapshot_tweedie_profile_refit_inputs(
            prepared.X,
            prepared.y,
            prepared.sample_weight,
            prepared.offset,
        )
    )
    prepared = replace(
        prepared,
        X=X_snapshot,
        y=y_snapshot,
        sample_weight=weight_snapshot,
        offset=offset_snapshot,
    )
    staged = _prepare_tweedie_profile_stage(
        model,
        X_snapshot,
        weight_snapshot,
        clone_profile_model=_clone_profile_model,
    )
    prepared = replace(prepared, _model_identity=id(staged))

    with _use_prepared_tweedie_profile_inputs(prepared):
        result = estimate_tweedie_p(
            staged,
            X_snapshot,
            y_snapshot,
            sample_weight=weight_snapshot,
            offset=offset_snapshot,
            p_bounds=prepared.p_bounds,
            xatol=prepared.xatol,
            maxiter=prepared.maxiter,
            verbose=prepared.verbose,
            fit_mode=prepared.fit_mode,
            phi_method=prepared.phi_method,
            method=prepared.method,
            n_grid=prepared.n_grid,
            grid=prepared.grid,
            n_grid_coarse=prepared.n_grid_coarse,
            optimizer=prepared.optimizer,
            trace_callback=prepared.trace_callback,
            trace_iterations=prepared.trace_iterations,
        )
    _validate_tweedie_profile_result_for_refit(result, prepared)
    if eager_ci_alpha is not None:
        result.ci(alpha=eager_ci_alpha)
    if progress_callback is not None:
        progress_callback(
            "best_found",
            {
                "profile_estimate": _tweedie_estimate_payload(
                    result,
                    ci_alpha=0.05 if eager_ci_alpha is None else eager_ci_alpha,
                )
            },
        )
    if progress_callback is not None:
        progress_callback(
            "final_refit",
            {
                "profile_estimate": _tweedie_estimate_payload(
                    result,
                    ci_alpha=0.05 if eager_ci_alpha is None else eager_ci_alpha,
                )
            },
        )

    # User callbacks run outside the staged model, so certify the returned
    # record again in case a callback retained and modified it.
    _validate_tweedie_profile_result_for_refit(result, prepared)

    _fit_tweedie_profile_stage(
        staged,
        X_snapshot,
        y_snapshot,
        weight_snapshot,
        offset_snapshot,
        result,
        resolved_mode,
    )
    if not staged._retain_fit_state:
        result.detach_evaluator()
        _validate_detached_tweedie_profile_result(result)
    _commit_tweedie_profile_state(model, staged, result, caller_state)

    return result


def _normalize_eager_tweedie_ci_alpha(value):
    """Normalize an optional eager CI level without executing coercion hooks."""
    if value is None:
        return None
    if not _is_finite_tweedie_scalar(value):
        raise ValueError("eager_ci_alpha must be finite and strictly between 0 and 1")
    alpha = float(value)
    if not 0.0 < alpha < 1.0:
        raise ValueError("eager_ci_alpha must be finite and strictly between 0 and 1")
    return alpha


def _is_superglm_profile_model(model) -> bool:
    """Identify the public model path without invoking instance lookup hooks."""
    from superglm.model.api import SuperGLM

    return any(
        candidate_type is SuperGLM
        for candidate_type in type.__getattribute__(type(model), "__mro__")
    )


def _validate_tweedie_profile_model_storage(model) -> None:
    """Reject subclass slot state that cannot participate in an atomic swap."""
    from superglm.model.api import SuperGLM

    model_mro = type.__getattribute__(type(model), "__mro__")
    model_dict = _tweedie_profile_instance_dict(model)
    is_superglm = _is_superglm_profile_model(model)
    if is_superglm:
        for candidate_type in model_mro:
            if candidate_type is SuperGLM or candidate_type is object:
                continue
            candidate_dict = type.__getattribute__(candidate_type, "__dict__")
            unsafe_members = sorted(
                name
                for name, value in candidate_dict.items()
                if name
                not in {
                    "__annotations__",
                    "__doc__",
                    "__module__",
                    "__slots__",
                }
                and (
                    callable(value)
                    or any(
                        "__get__" in type.__getattribute__(member_type, "__dict__")
                        for member_type in type.__getattribute__(type(value), "__mro__")
                    )
                )
            )
            if unsafe_members:
                raise TypeError(
                    "Tweedie profile refitting does not support model subclasses with "
                    "custom behavior: " + ", ".join(unsafe_members)
                )
        family = object.__getattribute__(model, "family")
        if isinstance(family, Tweedie) and type(family) is not Tweedie:
            raise TypeError(
                "Tweedie profile refitting requires an exact Tweedie family, not a subclass"
            )
        _validate_tweedie_profile_copy_protocols(model)
        shadowed_methods = sorted(
            name
            for name in model_dict
            if any(
                callable(type.__getattribute__(candidate_type, "__dict__").get(name))
                for candidate_type in model_mro
            )
        )
        if shadowed_methods:
            raise TypeError(
                "Tweedie profile refitting does not support instance-level method overrides: "
                + ", ".join(shadowed_methods)
            )
    else:
        has_static_clone = "_clone_without_features" in model_dict or any(
            "_clone_without_features" in type.__getattribute__(candidate_type, "__dict__")
            for candidate_type in model_mro
        )
        if has_static_clone:
            raise TypeError("Tweedie profile refitting requires a SuperGLM-compatible model")
    slot_names = []
    for candidate_type in model_mro:
        declared = type.__getattribute__(candidate_type, "__dict__").get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        slot_names.extend(name for name in declared if name not in {"__dict__", "__weakref__"})
    if slot_names:
        raise TypeError(
            "Tweedie profile refitting does not support model subclasses with slot state"
        )


def _tweedie_profile_configuration_roots(model):
    """Return mutable configuration roots copied into scratch profile models."""
    return (
        model.family,
        model.link,
        model.penalty,
        model.lambda2,
        model._specs,
        model._interaction_specs,
        model._pending_interactions,
        model._interaction_order,
        model._splines,
        model._n_knots,
        model._n_bins,
        model._feature_order,
        model._degree,
        model._categorical_base,
        model._active_set,
        model._direct_solve,
        model._discrete,
        model._tol,
        model._max_iter,
        model._convergence,
        model._retain_fit_state,
    )


def _validate_tweedie_profile_copy_protocols(model) -> None:
    """Reject user copy hooks that could mutate the caller during scratch cloning."""
    from fractions import Fraction
    from uuid import UUID, SafeUUID

    import pandas as pd

    from superglm.profiling.tweedie import (
        _PROFILE_INDEX_TYPES,
        _PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES,
        _canonical_tweedie_profile_timezone,
        _is_known_immutable_profile_value,
        _validate_tweedie_profile_dtype,
    )

    unsafe_protocols = {
        "__copy__",
        "__deepcopy__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
    }
    atomic_types = {
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        range,
        slice,
        type,
        types.FunctionType,
        np.ufunc,
        re.Pattern,
    }
    seen: set[int] = set()

    def visit(value) -> None:
        if (
            type(value) in atomic_types
            or type(value) in _TWEEDIE_EXACT_NUMPY_SCALAR_TYPES
            or _is_known_immutable_profile_value(value)
        ):
            return
        if type(value) is Fraction:
            if type(value.numerator) is not int or type(value.denominator) is not int:
                raise TypeError("Tweedie profile Fraction configuration is not immutable")
            return
        if type(value) is UUID:
            if (
                type(value.int) is not int
                or value.is_safe is not None
                and type(value.is_safe) is not SafeUUID
            ):
                raise TypeError("Tweedie profile UUID configuration is not immutable")
            return
        if type(value) is pd.Timestamp:
            try:
                _canonical_tweedie_profile_timezone(value.tzinfo)
            except TypeError as exc:
                raise TypeError("Tweedie profile Timestamp has an unsafe timezone") from exc
            return
        if type(value) is pd.Timedelta:
            return
        if type(value) is pd.Interval:
            if type(value.closed) is not str:
                raise TypeError("Tweedie profile Interval configuration is not immutable")
            visit(value.left)
            visit(value.right)
            return
        if type(value) in _PROFILE_INDEX_TYPES:
            for name in value.names:
                visit(name)
            for item in value.tolist():
                visit(item)
            return
        if isinstance(value, np.dtype) or type(value) in _PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES:
            _validate_tweedie_profile_dtype(value, name="model configuration dtype")
            if type(value) is pd.CategoricalDtype and value.categories is not None:
                visit(value.categories)
            return
        if type(value) is types.MethodType:
            visit(object.__getattribute__(value, "__self__"))
            return
        if type(value) is types.BuiltinFunctionType:
            owner = object.__getattribute__(value, "__self__")
            if owner is not None and type(owner) not in {types.ModuleType, type}:
                visit(owner)
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) is np.ndarray:
            if value.dtype.kind == "O":
                for item in value.flat:
                    visit(item)
            return
        if type(value) is dict:
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if type(value) in {list, tuple, set, frozenset}:
            for item in value:
                visit(item)
            return

        if type(value) in copyreg.dispatch_table:
            raise TypeError(
                "Tweedie profile refitting cannot safely clone configuration object "
                f"{type(value).__name__} with a registered copy reducer"
            )

        dataclass_params = type(value).__dict__.get("__dataclass_params__")
        frozen_dataclass = bool(
            "__dataclass_fields__" in type(value).__dict__
            and dataclass_params is not None
            and dataclass_params.frozen
        )
        defining_protocols = []
        for name in sorted(unsafe_protocols):
            for candidate_type in type(value).__mro__:
                if candidate_type is object or name not in candidate_type.__dict__:
                    continue
                definition = candidate_type.__dict__[name]
                generated_frozen_hook = frozen_dataclass and name in {
                    "__setattr__",
                    "__delattr__",
                }
                generated_slots_state = definition in {
                    getattr(dataclasses, "_dataclass_getstate", None),
                    getattr(dataclasses, "_dataclass_setstate", None),
                }
                if not generated_frozen_hook and not generated_slots_state:
                    defining_protocols.append(name)
                break
        if defining_protocols:
            raise TypeError(
                "Tweedie profile refitting cannot safely clone configuration object "
                f"{type(value).__name__} with custom copy hooks: " + ", ".join(defining_protocols)
            )
        slot_descriptors = []
        for candidate_type in type(value).__mro__:
            declared = candidate_type.__dict__.get("__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            for name in declared:
                if name in {"__dict__", "__weakref__"}:
                    continue
                descriptor = candidate_type.__dict__.get(name)
                if type(descriptor) is not types.MemberDescriptorType:
                    raise TypeError(
                        "Tweedie profile refitting cannot safely inspect slot state on "
                        f"{type(value).__name__}"
                    )
                resolved_descriptor = next(
                    (base.__dict__[name] for base in type(value).__mro__ if name in base.__dict__),
                    None,
                )
                if resolved_descriptor is not descriptor:
                    raise TypeError(
                        "Tweedie profile refitting cannot safely inspect overridden slot "
                        f"{name!r} on {type(value).__name__}"
                    )
                slot_descriptors.append((name, descriptor))
        for _name, descriptor in slot_descriptors:
            try:
                slot_value = descriptor.__get__(value, type(value))
            except AttributeError:
                continue
            visit(slot_value)
        try:
            attributes = object.__getattribute__(value, "__dict__")
        except AttributeError:
            return
        if type(attributes) is not dict:
            raise TypeError(
                "Tweedie profile refitting cannot safely inspect configuration object "
                f"{type(value).__name__}"
            )
        visit(attributes)

    for root in _tweedie_profile_configuration_roots(model):
        visit(root)


def _prepare_tweedie_profile_stage(model, X, sample_weight, *, clone_profile_model):
    """Construct a fresh configured model without copying unrelated user attributes."""
    if _is_superglm_profile_model(model):
        staged = clone_profile_model(model, X, sample_weight)
    else:
        # Internal duck-model integrations have no model configuration clone.
        staged = copy.deepcopy(model)
    if staged is model:
        raise RuntimeError("Tweedie final refit stage aliases the caller model")
    if type(staged) is not type(model):
        raise RuntimeError("Tweedie final refit stage changed the model type")
    if _is_superglm_profile_model(model):
        _validate_tweedie_profile_clone_isolation(model, staged)
    staged._retain_fit_state = model._retain_fit_state
    return staged


def _validate_tweedie_profile_clone_isolation(model, staged) -> None:
    """Require every mutable scratch configuration node to be caller-independent."""
    from fractions import Fraction
    from uuid import UUID, SafeUUID

    import pandas as pd

    from superglm.profiling.tweedie import (
        _PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES,
        _is_known_immutable_profile_value,
    )

    atomic_types = {
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        range,
        slice,
        type,
        types.FunctionType,
        np.ufunc,
        re.Pattern,
    }

    def mutable_ids(roots) -> set[int]:
        found: set[int] = set()

        def visit(value) -> None:
            if (
                type(value) in atomic_types
                or type(value) in _TWEEDIE_EXACT_NUMPY_SCALAR_TYPES
                or _is_known_immutable_profile_value(value)
                or type(value) is Fraction
                or (
                    type(value) is UUID
                    and type(value.int) is int
                    and (value.is_safe is None or type(value.is_safe) is SafeUUID)
                )
                or type(value) in {pd.Timestamp, pd.Timedelta}
                or isinstance(value, np.dtype)
                or type(value) in _PROFILE_SUPPORTED_EXTENSION_DTYPE_TYPES
            ):
                return
            if type(value) is pd.Interval:
                visit(value.left)
                visit(value.right)
                return
            if type(value) is types.MethodType:
                visit(object.__getattribute__(value, "__self__"))
                return
            if type(value) is types.BuiltinFunctionType:
                owner = object.__getattribute__(value, "__self__")
                if owner is not None and type(owner) not in {types.ModuleType, type}:
                    visit(owner)
                return
            identity = id(value)
            if identity in found:
                return
            if type(value) is tuple or type(value) is frozenset:
                for item in value:
                    visit(item)
                return
            found.add(identity)
            if type(value) is np.ndarray:
                if value.dtype.kind == "O":
                    for item in value.flat:
                        visit(item)
                return
            if type(value) is dict:
                for key, item in value.items():
                    visit(key)
                    visit(item)
                return
            if type(value) in {list, set}:
                for item in value:
                    visit(item)
                return
            for candidate_type in type(value).__mro__:
                declared = candidate_type.__dict__.get("__slots__", ())
                if isinstance(declared, str):
                    declared = (declared,)
                for name in declared:
                    if name in {"__dict__", "__weakref__"}:
                        continue
                    descriptor = candidate_type.__dict__.get(name)
                    if type(descriptor) is not types.MemberDescriptorType:
                        continue
                    try:
                        slot_value = descriptor.__get__(value, type(value))
                    except AttributeError:
                        continue
                    visit(slot_value)
            try:
                attributes = object.__getattribute__(value, "__dict__")
            except AttributeError:
                return
            if type(attributes) is dict:
                visit(attributes)

        for root in roots:
            visit(root)
        return found

    shared = mutable_ids(_tweedie_profile_configuration_roots(model)) & mutable_ids(
        _tweedie_profile_configuration_roots(staged)
    )
    if shared:
        raise RuntimeError(
            "Tweedie profile scratch model shares mutable configuration with the caller"
        )


def _fit_tweedie_profile_stage(
    staged,
    X,
    y,
    sample_weight,
    offset,
    result,
    resolved_mode,
):
    """Build the complete final fit without mutating the caller's model."""
    staged.family = Tweedie(p=result.p_hat)
    staged._tweedie_profile_result = None
    retain_fit_state = staged._retain_fit_state
    try:
        staged._retain_fit_state = True
        if resolved_mode == "fit_reml":
            staged.fit_reml(
                X,
                y,
                sample_weight=sample_weight,
                offset=offset,
            )
        else:
            staged.fit(
                X,
                y,
                sample_weight=sample_weight,
                offset=offset,
            )
        _synchronize_tweedie_profile_refit(staged, y, result)
        _validate_tweedie_profile_stage(
            staged,
            X,
            y,
            sample_weight,
            offset,
            result,
            resolved_mode,
        )
    finally:
        staged._retain_fit_state = retain_fit_state

    if not retain_fit_state:
        from superglm.model import fit_ops

        release_core = _snapshot_tweedie_profile_release_core(staged, X=X, offset=offset)
        fit_ops._maybe_release_fit_state(staged)
        _validate_released_tweedie_profile_stage(
            staged,
            X,
            offset,
            result,
            release_core=release_core,
        )
    return staged


def _commit_tweedie_profile_state(model, staged, result, caller_state) -> None:
    """Install one complete staged state while preserving caller identity."""
    # Begin from the pre-callback caller snapshot. This preserves unrelated
    # user attributes without installing attributes injected concurrently by
    # trace/progress callbacks (notably instance-level method overrides).
    installed_state = caller_state.copy()
    staged_state = staged.__dict__.copy()
    for cache_name in (
        "_coef_covariance",
        "_fit_active_info",
        "_fit_inference_info",
        "_group_edf",
    ):
        if cache_name not in staged_state:
            installed_state.pop(cache_name, None)
    installed_state.update(staged_state)
    installed_state["_tweedie_profile_result"] = result
    _replace_tweedie_profile_model_state(model, installed_state)


def _replace_tweedie_profile_model_state(model, replacement_state) -> None:
    """Replace an exact model dictionary, with rollback for mapping-only ducks."""
    try:
        object.__setattr__(model, "__dict__", replacement_state.copy())
    except AttributeError:
        # Simple Python containers can expose a read-only __dict__ attribute;
        # their exact built-in dict can still be replaced transactionally.
        target = model.__dict__
        original_state = target.copy()
        try:
            target.clear()
            target.update(replacement_state)
        except Exception:
            target.clear()
            target.update(original_state)
            raise


def _restore_tweedie_profile_model_state(model, original_mapping, original_state) -> None:
    """Restore both the caller's state and its original ``vars(model)`` identity."""

    def matches_snapshot(mapping) -> bool:
        if type(mapping) is not dict or len(mapping) != len(original_state):
            return False
        return all(
            current_key is snapshot_key and current_value is snapshot_value
            for (current_key, current_value), (snapshot_key, snapshot_value) in zip(
                mapping.items(), original_state.items(), strict=True
            )
        )

    current_mapping = _tweedie_profile_instance_dict(model)
    if current_mapping is original_mapping and matches_snapshot(current_mapping):
        return

    original_mapping.clear()
    original_mapping.update(original_state)
    if current_mapping is original_mapping:
        return
    try:
        object.__setattr__(model, "__dict__", original_mapping)
    except AttributeError:
        # Mapping-only duck models cannot replace the mapping object, so
        # restore the currently exposed exact dictionary in place.
        current_mapping.clear()
        current_mapping.update(original_state)


def _validate_detached_tweedie_profile_result(result) -> None:
    """Require release detachment to eliminate every live evaluator path."""
    failures = []
    if getattr(result, "_evaluator", None) is not None:
        failures.append("modern evaluator")
    if any(
        getattr(result, name, None) is not None
        for name in ("_objective", "_evaluation_count", "_evaluation_record")
    ):
        failures.append("legacy evaluator callbacks")
    if _live_tweedie_profile_context(result) is not None:
        failures.append("live profile context")
    frozen_count = getattr(result, "_frozen_evaluation_count", None)
    if type(frozen_count) is not int or frozen_count < int(result.n_evaluations):
        failures.append("frozen evaluation count")
    if failures:
        joined = ", ".join(failures)
        raise RuntimeError(f"Tweedie profile result did not detach safely: invalid {joined}")


def _validate_tweedie_profile_result_for_refit(result, prepared) -> None:
    """Reject an uncertified winner before callbacks or final fitting."""
    from superglm.profiling.tweedie import (
        _TWEEDIE_PROFILE_RESULT_TOKEN,
        TweedieProfileResult,
        _TweedieProfileEvaluator,
    )

    if (
        type(result) is not TweedieProfileResult
        or result._validation_token is not _TWEEDIE_PROFILE_RESULT_TOKEN
    ):
        raise RuntimeError("Tweedie profile result is not installable: invalid result provenance")
    failures = [
        name
        for name in (
            "converged",
            "outer_converged",
            "fit_converged",
            "solver_converged",
            "objective_finite",
            "phi_converged",
            "density_exact",
        )
        if getattr(result, name, None) is not True
    ]
    reml_converged = getattr(result, "reml_converged", None)
    if prepared.fit_mode == "fit_reml":
        if reml_converged is not True and reml_converged is not None:
            failures.append("reml_converged")
    elif reml_converged is not None:
        failures.append("reml_converged")

    numeric_failures = []
    normalized_values = {}
    for name in ("p_hat", "phi_hat", "nll"):
        raw_value = getattr(result, name, None)
        if not _is_finite_tweedie_scalar(raw_value):
            numeric_failures.append(name)
            continue
        value = float(raw_value)
        normalized_values[name] = value
        if name == "p_hat" and not 1.0 < value < 2.0:
            numeric_failures.append(name)
        elif name == "phi_hat" and value <= 0.0:
            numeric_failures.append(name)

    failures.extend(numeric_failures)
    if type(getattr(result, "method", None)) is not str or result.method != prepared.method:
        failures.append("method provenance")
    if (
        type(getattr(result, "phi_method", None)) is not str
        or result.phi_method != prepared.phi_method
    ):
        failures.append("dispersion-method provenance")

    p_hat = normalized_values.get("p_hat")
    if p_hat is not None:
        scale = max(abs(prepared.p_bounds[0]), abs(prepared.p_bounds[1]), 1.0)
        bound_atol = 32.0 * np.finfo(np.float64).eps * scale
        if prepared.method == "grid" and prepared.grid is not None:
            if not np.any(np.isclose(prepared.grid, p_hat, rtol=0.0, atol=bound_atol)):
                failures.append("explicit grid")
        elif not prepared.p_bounds[0] - bound_atol <= p_hat <= prepared.p_bounds[1] + bound_atol:
            failures.append("search bounds")

    n_evaluations = getattr(result, "n_evaluations", None)
    if type(n_evaluations) is not int or n_evaluations < 1:
        failures.append("evaluation count")

    evaluator_fields = tuple(
        getattr(result, name, None)
        for name in ("_objective", "_evaluation_count", "_evaluation_record")
    )
    evaluator = getattr(result, "_evaluator", None)
    has_modern_evaluator = type(evaluator) is _TweedieProfileEvaluator
    live_context = _live_tweedie_profile_context(result)
    if has_modern_evaluator and (
        live_context is None or type(getattr(evaluator, "_lock", None)) is not _TWEEDIE_RLOCK_TYPE
    ):
        failures.append("profile evaluator provenance")
    if not has_modern_evaluator and not all(callable(callback) for callback in evaluator_fields):
        failures.append("profile evaluator provenance")
    ll_scale = getattr(result, "_ll_scale", None)
    if not _is_finite_tweedie_scalar(ll_scale) or float(ll_scale) <= 0.0:
        failures.append("likelihood scale provenance")
    elif has_modern_evaluator and (
        not np.isclose(float(ll_scale), float(len(prepared.y)), rtol=0.0, atol=0.0)
        or not _is_finite_tweedie_scalar(getattr(live_context, "ll_scale", None))
        or not np.isclose(
            float(ll_scale),
            float(live_context.ll_scale),
            rtol=0.0,
            atol=0.0,
        )
    ):
        failures.append("likelihood scale provenance")
    if has_modern_evaluator:
        _validate_tweedie_profile_context_provenance(
            live_context,
            prepared,
            result,
            failures,
        )
    evaluation_count = evaluator.evaluation_count if has_modern_evaluator else evaluator_fields[1]
    if callable(evaluation_count):
        try:
            total_evaluations = evaluation_count()
        except Exception:
            failures.append("evaluation count provenance")
        else:
            if (
                type(total_evaluations) is not int
                or type(n_evaluations) is not int
                or total_evaluations < n_evaluations
            ):
                failures.append("evaluation count provenance")
    trace = getattr(result, "search_trace", None)
    _validate_tweedie_profile_trace(
        trace,
        result,
        n_evaluations,
        failures,
        strict_diagnostics=has_modern_evaluator,
    )

    _validate_tweedie_profile_result_diagnostics(
        result,
        prepared,
        trace,
        strict_diagnostics=has_modern_evaluator,
        failures=failures,
    )

    if (
        getattr(result, "density_method", None) != "exact"
        or getattr(result, "density_exact", None) is not True
        or type(getattr(result, "n_saddlepoint", None)) is not int
        or result.n_saddlepoint != 0
        or not _is_finite_tweedie_scalar(getattr(result, "saddlepoint_fraction", None))
        or float(result.saddlepoint_fraction) != 0.0
    ):
        failures.append("exact density provenance")
    if failures:
        joined = ", ".join(dict.fromkeys(failures))
        raise RuntimeError(f"Tweedie profile result is not installable: invalid {joined}")


def _validate_tweedie_profile_context_provenance(context, prepared, result, failures) -> None:
    """Bind a retained evaluator to this transaction's exact owned input graph."""
    from superglm.profiling.tweedie import _ProfileContext, _ProfileContextREML

    if type(context) is _ProfileContext:
        input_matches = (
            context.profile_x is prepared.X
            and context.profile_y is prepared.y
            and context.profile_sample_weight is prepared.sample_weight
            and context.profile_offset is prepared.offset
        )
        mode_matches = prepared.fit_mode == "fit"
    elif type(context) is _ProfileContextREML:
        input_matches = (
            context.X is prepared.X
            and context.y is prepared.y
            and context.sample_weight is prepared.sample_weight
            and context.offset is prepared.offset
        )
        mode_matches = prepared.fit_mode == "fit_reml"
    else:
        failures.append("profile evaluator input provenance")
        return
    if (
        not input_matches
        or not mode_matches
        or context.phi_method != prepared.phi_method
        or getattr(context, "trace_callback", None) is not None
    ):
        failures.append("profile evaluator input provenance")

    try:
        record = context.evaluation_record(float(result.p_hat))
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        failures.append("profile evaluator winner provenance")
        return
    if record is None:
        failures.append("profile evaluator winner provenance")
        return
    try:
        record_values = (record.p, record.phi, record.nll)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        failures.append("profile evaluator winner provenance")
        return
    expected_values = (result.p_hat, result.phi_hat, result.nll)
    if any(
        not _is_finite_tweedie_scalar(actual)
        or not np.isclose(float(actual), float(expected), rtol=1e-11, atol=1e-12)
        for actual, expected in zip(record_values, expected_values)
    ):
        failures.append("profile evaluator winner provenance")


def _validate_tweedie_profile_trace(
    trace,
    result,
    n_evaluations,
    failures,
    *,
    strict_diagnostics: bool,
) -> None:
    """Cross-check flat winner fields against the immutable search snapshot."""
    import pandas as pd

    required = {
        "p",
        "phi",
        "nll",
        "edf",
        "fit_converged",
        "solver_converged",
        "reml_converged",
        "phi_converged",
        "objective_finite",
        "n_saddlepoint",
        "density_method",
        "density_exact",
    }
    if strict_diagnostics:
        required.update(
            {
                "phi_n_evaluations",
                "phi_n_score_evaluations",
                "phi_n_value_only_evaluations",
                "phi_n_fallback_evaluations",
                "phi_boundary",
                "phi_optimizer",
                "phi_score",
                "n_positive",
                "phi_used_fallback",
                "phi_fallback_reason",
                "phi_branch_switch_detected",
                "phi_message",
            }
        )
    if type(trace) is not pd.DataFrame or trace.empty or not required.issubset(trace.columns):
        failures.append("search trace provenance")
        return
    if type(n_evaluations) is int and len(trace) != n_evaluations:
        failures.append("evaluation count")
    p_values = []
    for value in trace["p"]:
        if not _is_finite_tweedie_scalar(value):
            failures.append("search trace provenance")
            return
        p_values.append(float(value))
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.shape != (len(trace),):
        failures.append("search trace provenance")
        return
    winner_positions = np.flatnonzero(p_values == float(result.p_hat))
    if winner_positions.size != 1:
        failures.append("winning trace record")
        return
    winner = trace.iloc[int(winner_positions[0])]
    for name in ("phi", "nll"):
        value = winner[name]
        if not _is_finite_tweedie_scalar(value) or not np.isclose(
            float(value),
            float(getattr(result, f"{name}_hat", result.nll)),
            rtol=1e-11,
            atol=1e-12,
        ):
            failures.append("winning trace record")
    if not _is_finite_tweedie_scalar(winner["edf"]) or float(winner["edf"]) < 0.0:
        failures.append("winning trace record")
    for name in (
        "fit_converged",
        "solver_converged",
        "phi_converged",
        "objective_finite",
        "density_exact",
    ):
        if not isinstance(winner[name], bool | np.bool_) or not bool(winner[name]):
            failures.append("winning trace certification")
    if result.reml_converged is True:
        if not isinstance(winner["reml_converged"], bool | np.bool_) or not bool(
            winner["reml_converged"]
        ):
            failures.append("winning trace certification")
    elif not pd.isna(winner["reml_converged"]):
        failures.append("winning trace certification")
    if (
        winner["density_method"] != "exact"
        or type(winner["n_saddlepoint"]) not in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
        or int(winner["n_saddlepoint"]) != 0
    ):
        failures.append("winning trace density provenance")

    selectable_nll = []
    for _, row in trace.iterrows():
        p_value = row["p"]
        phi_value = row["phi"]
        nll_value = row["nll"]
        objective_finite = row["objective_finite"]
        if (
            _is_finite_tweedie_scalar(p_value)
            and _is_finite_tweedie_scalar(phi_value)
            and float(phi_value) > 0.0
            and _is_finite_tweedie_scalar(nll_value)
            and type(objective_finite) in {bool, np.bool_}
            and bool(objective_finite)
        ):
            selectable_nll.append(float(nll_value))
    if not selectable_nll:
        failures.append("search trace provenance")
    else:
        minimum_nll = min(selectable_nll)
        winner_nll = float(winner["nll"])
        tie_atol = 1e-10 * max(1.0, abs(minimum_nll))
        if winner_nll > minimum_nll + tie_atol:
            failures.append("non-minimum winning trace record")


def _validate_tweedie_profile_result_diagnostics(
    result,
    prepared,
    trace,
    *,
    strict_diagnostics: bool,
    failures,
) -> None:
    """Certify public diagnostic fields and their winning-row provenance."""
    import pandas as pd

    n_positive = getattr(result, "n_positive", None)
    count_names = (
        "phi_n_evaluations",
        "phi_n_score_evaluations",
        "phi_n_value_only_evaluations",
        "phi_n_fallback_evaluations",
    )
    counts = {name: getattr(result, name, None) for name in count_names}
    if (
        type(n_positive) not in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
        or int(n_positive) < 0
        or any(
            type(value) not in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES or int(value) < 0
            for value in counts.values()
        )
    ):
        failures.append("profile diagnostic counts")
    elif int(counts["phi_n_evaluations"]) != int(counts["phi_n_score_evaluations"]) + int(
        counts["phi_n_value_only_evaluations"]
    ) or int(counts["phi_n_fallback_evaluations"]) > int(counts["phi_n_evaluations"]):
        failures.append("profile diagnostic count coherence")

    if (
        getattr(result, "density_warning_severity", None) != "none"
        or getattr(result, "near_power_boundary", None) is not False
    ):
        failures.append("exact density diagnostics")
    warnings_list = getattr(result, "warnings", None)
    if type(warnings_list) is not list or any(
        type(message) is not str for message in warnings_list
    ):
        failures.append("profile warnings")
    if type(getattr(result, "outer_message", None)) is not str:
        failures.append("outer-search diagnostics")
    outer_boundary = getattr(result, "outer_boundary", None)
    if outer_boundary is not None and (
        type(outer_boundary) is not str or outer_boundary not in {"lower", "upper"}
    ):
        failures.append("outer-search boundary")

    phi_score = getattr(result, "phi_score", None)
    if phi_score is not None and not _is_finite_tweedie_scalar(phi_score):
        failures.append("dispersion-score diagnostics")
    if type(getattr(result, "phi_optimizer", None)) is not str:
        failures.append("dispersion-optimizer diagnostics")
    if type(getattr(result, "phi_used_fallback", None)) is not bool:
        failures.append("dispersion-fallback diagnostics")
    fallback_reason = getattr(result, "phi_fallback_reason", None)
    if fallback_reason is not None and type(fallback_reason) is not str:
        failures.append("dispersion-fallback diagnostics")
    if type(getattr(result, "phi_branch_switch_detected", None)) is not bool:
        failures.append("dispersion-branch diagnostics")
    phi_boundary = getattr(result, "phi_boundary", None)
    if type(phi_boundary) is not str or phi_boundary not in {"", "lower", "upper"}:
        failures.append("dispersion-boundary diagnostics")
    if type(getattr(result, "phi_message", None)) is not str:
        failures.append("dispersion-message diagnostics")

    if not strict_diagnostics:
        return
    if type(getattr(result, "phi_optimizer", None)) is not str or not result.phi_optimizer:
        failures.append("dispersion-optimizer diagnostics")
    expected_n_positive = int(np.count_nonzero(prepared.y > 0.0))
    if (
        type(n_positive) in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
        and int(n_positive) != expected_n_positive
    ):
        failures.append("positive-density count provenance")

    searched_bounds = prepared.p_bounds
    if prepared.method == "grid" and prepared.grid is not None and len(prepared.grid):
        searched_bounds = (float(np.min(prepared.grid)), float(np.max(prepared.grid)))
    expected_boundary = None
    lo, hi = searched_bounds
    scale = max(abs(lo), abs(hi), 1.0)
    atol = 16.0 * np.finfo(np.float64).eps * scale
    if not np.isclose(lo, hi, rtol=0.0, atol=atol):
        if np.isclose(float(result.p_hat), lo, rtol=0.0, atol=atol):
            expected_boundary = "lower"
        elif np.isclose(float(result.p_hat), hi, rtol=0.0, atol=atol):
            expected_boundary = "upper"
    if outer_boundary != expected_boundary:
        failures.append("outer-search boundary provenance")

    if type(trace) is not pd.DataFrame or "p" not in trace:
        return
    if not all(_is_finite_tweedie_scalar(value) for value in trace["p"]):
        return
    positions = np.flatnonzero(
        np.asarray([float(value) for value in trace["p"]], dtype=np.float64) == float(result.p_hat)
    )
    if positions.size != 1:
        return
    winner = trace.iloc[int(positions[0])]
    exact_fields = {
        "phi_n_evaluations": "phi_n_evaluations",
        "phi_n_score_evaluations": "phi_n_score_evaluations",
        "phi_n_value_only_evaluations": "phi_n_value_only_evaluations",
        "phi_n_fallback_evaluations": "phi_n_fallback_evaluations",
        "phi_boundary": "phi_boundary",
        "phi_optimizer": "phi_optimizer",
        "n_positive": "n_positive",
        "phi_used_fallback": "phi_used_fallback",
        "phi_fallback_reason": "phi_fallback_reason",
        "phi_branch_switch_detected": "phi_branch_switch_detected",
        "phi_message": "phi_message",
    }

    def trace_value_is_missing(value) -> bool:
        if value is None or value is pd.NA:
            return True
        return type(value) in _TWEEDIE_EXACT_REAL_SCALAR_TYPES and bool(np.isnan(value))

    for result_name, trace_name in exact_fields.items():
        result_value = getattr(result, result_name, None)
        trace_value = winner[trace_name]
        if result_value is None and trace_value_is_missing(trace_value):
            continue
        if trace_name in {
            "phi_used_fallback",
            "phi_branch_switch_detected",
        }:
            matches = (
                type(result_value) is bool
                and type(trace_value) in {bool, np.bool_}
                and bool(trace_value) is result_value
            )
        elif trace_name.startswith("phi_n_") or trace_name == "n_positive":
            matches = (
                type(result_value) in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
                and type(trace_value) in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
                and int(result_value) == int(trace_value)
            )
        else:
            matches = type(result_value) is type(trace_value) and result_value == trace_value
            if type(result_value) is str and isinstance(trace_value, str):
                matches = result_value == trace_value
        if not matches:
            failures.append("winning trace diagnostic provenance")
            break

    winner_score = winner["phi_score"]
    if phi_score is None:
        if not trace_value_is_missing(winner_score):
            failures.append("winning trace diagnostic provenance")
    elif not _is_finite_tweedie_scalar(winner_score) or not np.isclose(
        float(phi_score),
        float(winner_score),
        rtol=1e-12,
        atol=1e-13,
    ):
        failures.append("winning trace diagnostic provenance")


def _validate_tweedie_profile_stage(
    model,
    X,
    y,
    sample_weight,
    offset,
    result,
    resolved_mode: str,
) -> None:
    """Certify the synchronized staged fit before it can replace caller state."""
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import FitStats

    failures = []
    distribution = getattr(model, "_distribution", None)
    if not isinstance(model.family, Tweedie) or not isinstance(distribution, Tweedie):
        failures.append("Tweedie family/distribution")
    else:
        if model.family is not distribution:
            failures.append("family/distribution identity")
        if not _is_finite_tweedie_scalar(model.family.p) or not np.isclose(
            float(model.family.p), float(result.p_hat), rtol=0.0, atol=1e-14
        ):
            failures.append("family power")
        if not _is_finite_tweedie_scalar(distribution.p) or not np.isclose(
            float(distribution.p), float(result.p_hat), rtol=0.0, atol=1e-14
        ):
            failures.append("distribution power")

    public_result = getattr(model, "_result", None)
    solver_result = None
    try:
        solver_result = model._solver_pirls_result()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        failures.append("solver result")
    if public_result is None or getattr(public_result, "converged", None) is not True:
        failures.append("public solver convergence")
    if solver_result is None or getattr(solver_result, "converged", None) is not True:
        failures.append("internal solver convergence")

    for label, candidate_result in (
        ("public", public_result),
        ("solver", solver_result),
    ):
        if candidate_result is None:
            continue
        if type(candidate_result) is not PIRLSResult:
            failures.append(f"{label} fitted result")
            continue
        phi = candidate_result.phi
        intercept = candidate_result.intercept
        deviance = candidate_result.deviance
        effective_df = candidate_result.effective_df
        if not _is_finite_tweedie_scalar(phi) or not np.isclose(
            float(phi), float(result.phi_hat), rtol=1e-12, atol=1e-14
        ):
            failures.append(f"{label} dispersion")
        expected_beta_length = getattr(getattr(model, "_dm", None), "p", None)
        if not _is_finite_tweedie_vector(
            candidate_result.beta,
            length=expected_beta_length,
        ):
            failures.append(f"{label} coefficients")
        if (
            not _is_finite_tweedie_scalar(intercept)
            or not _is_finite_tweedie_scalar(deviance)
            or float(deviance) < 0.0
            or not _is_finite_tweedie_scalar(effective_df)
            or float(effective_df) < 0.0
            or type(candidate_result.n_iter) not in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
            or int(candidate_result.n_iter) < 0
        ):
            failures.append(f"{label} fitted result")

    if (
        type(public_result) is PIRLSResult
        and type(solver_result) is PIRLSResult
        and all(
            _is_finite_tweedie_scalar(value)
            for value in (
                public_result.deviance,
                solver_result.deviance,
                public_result.effective_df,
                solver_result.effective_df,
            )
        )
        and type(public_result.n_iter) in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
        and type(solver_result.n_iter) in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
    ):
        if (
            not np.isclose(
                float(public_result.deviance),
                float(solver_result.deviance),
                rtol=1e-12,
                atol=1e-13,
            )
            or not np.isclose(
                float(public_result.effective_df),
                float(solver_result.effective_df),
                rtol=1e-12,
                atol=1e-13,
            )
            or int(public_result.n_iter) != int(solver_result.n_iter)
        ):
            failures.append("public/internal solver scalar coherence")

    fit_meta = getattr(model, "_last_fit_meta", None)
    expected_fit_method = "fit_reml" if resolved_mode == "fit_reml" else "fit"
    if type(fit_meta) is not dict or fit_meta.get("method") != expected_fit_method:
        failures.append("final fit-mode provenance")
    if resolved_mode == "fit" and any(
        getattr(model, name, None) is not None
        for name in ("_reml_result", "_reml_lambdas", "_reml_penalties")
    ):
        failures.append("non-REML fit state")

    if resolved_mode == "fit_reml":
        reml_result = getattr(model, "_reml_result", None)
        if result.reml_converged is None:
            if reml_result is not None:
                failures.append("REML fallback coherence")
        elif reml_result is None or getattr(reml_result, "converged", None) is not True:
            failures.append("REML convergence")
        else:
            if getattr(reml_result, "pirls_result", None) is not solver_result:
                failures.append("REML solver coherence")
            objective = getattr(reml_result, "objective", None)
            if not _is_finite_tweedie_scalar(objective):
                failures.append("finite REML objective")

    fit_stats = getattr(model, "_fit_stats", None)
    statistic_names = (
        "log_likelihood",
        "null_log_likelihood",
        "null_deviance",
        "explained_deviance",
        "pearson_chi2",
    )
    if type(fit_stats) is not FitStats:
        failures.append("finite fit statistics")
    else:
        raw_statistics = {name: getattr(fit_stats, name) for name in statistic_names}
        if not all(_is_finite_tweedie_scalar(value) for value in raw_statistics.values()):
            failures.append("finite fit statistics")
        elif (
            float(raw_statistics["null_deviance"]) < 0.0
            or float(raw_statistics["pearson_chi2"]) < 0.0
        ):
            failures.append("valid fit statistics")
        if type(fit_stats.n_obs) is not int or fit_stats.n_obs != len(X):
            failures.append("fit observation count")

        if _live_tweedie_profile_context(result) is not None and _is_finite_tweedie_scalar(
            fit_stats.log_likelihood
        ):
            final_mean_nll = -float(fit_stats.log_likelihood) / len(X)
            if not np.isclose(
                final_mean_nll,
                float(result.nll),
                rtol=2e-6,
                atol=1e-7,
            ):
                failures.append("profile/final objective agreement")
    try:
        prediction = model.predict(X, offset=offset)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        failures.append("finite predictions")
    else:
        if not _is_finite_tweedie_vector(prediction, length=len(X), positive=True):
            failures.append("finite predictions")
        fit_mu = getattr(model, "_fit_mu", None)
        if not _is_finite_tweedie_vector(fit_mu, length=len(X), positive=True):
            failures.append("synchronized fitted means")
        comparison_prediction = prediction
        if type(fit_meta) is dict and fit_meta.get("discrete") is True:
            try:
                comparison_prediction = model._predict_fast_discrete(X, offset=offset)
            except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
                failures.append("discrete synchronized fitted means")
                comparison_prediction = None
            else:
                if not _is_finite_tweedie_vector(
                    comparison_prediction,
                    length=len(X),
                    positive=True,
                ):
                    failures.append("discrete synchronized fitted means")
        if _is_finite_tweedie_vector(
            fit_mu, length=len(X), positive=True
        ) and _is_finite_tweedie_vector(
            comparison_prediction,
            length=len(X),
            positive=True,
        ):
            if not np.allclose(
                comparison_prediction,
                fit_mu,
                rtol=1e-11,
                atol=1e-13,
            ):
                failures.append("synchronized fitted means")

    if solver_result is not None and type(solver_result) is PIRLSResult:
        try:
            from superglm.distributions import clip_mu
            from superglm.links import stabilize_eta

            solver_eta = model._dm.matvec(solver_result.beta) + float(solver_result.intercept)
            if model._fit_offset is not None:
                solver_eta = solver_eta + model._fit_offset
            solver_eta = stabilize_eta(solver_eta, model._link)
            solver_mu = clip_mu(model._link.inverse(solver_eta), distribution)
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            failures.append("solver fitted means")
        else:
            if (
                not _is_finite_tweedie_vector(solver_mu, length=len(X), positive=True)
                or not _is_finite_tweedie_vector(
                    getattr(model, "_fit_mu", None),
                    length=len(X),
                    positive=True,
                )
                or not np.allclose(
                    solver_mu,
                    model._fit_mu,
                    rtol=1e-11,
                    atol=1e-13,
                )
            ):
                failures.append("solver fitted means")

    retained_rows = {}
    for name in ("_fit_weights", "_fit_null_mu"):
        values = getattr(model, name, None)
        if not _is_finite_tweedie_vector(values, length=len(X), positive=True):
            failures.append("retained fit rows")
            continue
        retained_rows[name] = values
    expected_weights = (
        np.ones(len(X), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    retained_weights = retained_rows.get("_fit_weights")
    if retained_weights is None or not np.array_equal(retained_weights, expected_weights):
        failures.append("retained fit weights")

    fitted_mu = getattr(model, "_fit_mu", None)
    null_mu = retained_rows.get("_fit_null_mu")
    if (
        type(fit_stats) is FitStats
        and retained_weights is not None
        and null_mu is not None
        and _is_finite_tweedie_vector(fitted_mu, length=len(X), positive=True)
        and _is_finite_tweedie_vector(y, length=len(X))
        and isinstance(distribution, Tweedie)
    ):
        try:
            expected_null_log_likelihood = distribution.log_likelihood(
                y,
                null_mu,
                retained_weights,
                float(result.phi_hat),
            )
            expected_null_deviance = float(
                np.sum(retained_weights * distribution.deviance_unit(y, null_mu))
            )
            expected_deviance = float(
                np.sum(retained_weights * distribution.deviance_unit(y, fitted_mu))
            )
            expected_explained_deviance = (
                1.0 - expected_deviance / expected_null_deviance
                if expected_null_deviance > 0.0
                else 0.0
            )
            expected_pearson = float(
                np.sum(retained_weights * (y - fitted_mu) ** 2 / distribution.variance(fitted_mu))
            )
        except Exception:
            failures.append("fit statistics coherence")
        else:
            expected_statistics = {
                "null_log_likelihood": expected_null_log_likelihood,
                "null_deviance": expected_null_deviance,
                "explained_deviance": expected_explained_deviance,
                "pearson_chi2": expected_pearson,
            }
            if any(
                not np.isclose(
                    float(getattr(fit_stats, name)),
                    expected,
                    rtol=2e-10,
                    atol=1e-9,
                )
                for name, expected in expected_statistics.items()
            ):
                failures.append("fit statistics coherence")
            if public_result is None or not np.isclose(
                float(public_result.deviance),
                expected_deviance,
                rtol=2e-10,
                atol=1e-9,
            ):
                failures.append("public deviance coherence")
    if offset is not None:
        retained_offset = getattr(model, "_fit_offset", None)
        if not _is_finite_tweedie_vector(retained_offset, length=len(X)):
            failures.append("retained offset")
        else:
            if not np.array_equal(retained_offset, offset):
                failures.append("retained offset")
    elif getattr(model, "_fit_offset", None) is not None:
        failures.append("retained offset")

    for name, expected in (
        ("_fit_X_ref", X),
        ("_fit_y_ref", y),
        ("_fit_sample_weight_ref", sample_weight),
        ("_fit_offset_ref", offset),
    ):
        if getattr(model, name, None) is not expected:
            failures.append("final fit input ownership")

    if _live_tweedie_profile_context(result) is not None:
        trace = result.search_trace
        winner_rows = trace.loc[trace["p"] == float(result.p_hat)]
        if len(winner_rows) != 1:
            failures.append("profile/final EDF agreement")
        elif public_result is None or not _is_finite_tweedie_scalar(
            getattr(public_result, "effective_df", None)
        ):
            failures.append("profile/final EDF agreement")
        else:
            winner_edf = winner_rows.iloc[0]["edf"]
            if not _is_finite_tweedie_scalar(winner_edf) or not np.isclose(
                float(winner_edf),
                float(public_result.effective_df),
                rtol=2e-6,
                atol=1e-7,
            ):
                failures.append("profile/final EDF agreement")

    if failures:
        joined = ", ".join(dict.fromkeys(failures))
        raise RuntimeError(f"Tweedie final fit is not installable: invalid {joined}")


def _snapshot_tweedie_profile_release_core(model, *, X=None, offset=None):
    """Capture state that row-release is never permitted to alter."""
    mutable_keys = {
        "_coef_covariance",
        "_fit_active_info",
        "_fit_inference_info",
        "_group_edf",
        "_dm",
        "_fit_weights",
        "_fit_offset",
        "_fit_mu",
        "_fit_null_mu",
        "_fit_X_ref",
        "_fit_y_ref",
        "_fit_sample_weight_ref",
        "_fit_offset_ref",
        "_fit_metrics_cache",
        "_fit_metrics_cache_signature",
        "_summary_cache",
    }
    stable_state = {
        name: value for name, value in model.__dict__.items() if name not in mutable_keys
    }

    def fitted_result_state(candidate):
        if candidate is None:
            return None
        beta = getattr(candidate, "beta", None)
        return {
            "beta": None if type(beta) is not np.ndarray else beta.copy(),
            "intercept": getattr(candidate, "intercept", None),
            "n_iter": getattr(candidate, "n_iter", None),
            "deviance": getattr(candidate, "deviance", None),
            "converged": getattr(candidate, "converged", None),
            "phi": getattr(candidate, "phi", None),
            "effective_df": getattr(candidate, "effective_df", None),
        }

    fit_stats = getattr(model, "_fit_stats", None)
    fit_stats_state = (
        None
        if fit_stats is None
        else tuple(
            getattr(fit_stats, name, None)
            for name in (
                "log_likelihood",
                "null_log_likelihood",
                "null_deviance",
                "explained_deviance",
                "pearson_chi2",
                "n_obs",
            )
        )
    )
    reml_result = getattr(model, "_reml_result", None)
    reml_state = (
        None
        if reml_result is None
        else (
            getattr(reml_result, "pirls_result", None),
            getattr(reml_result, "objective", None),
            getattr(reml_result, "converged", None),
            getattr(reml_result, "n_reml_iter", None),
        )
    )
    meta = getattr(model, "_last_fit_meta", None)
    lambdas = getattr(model, "_reml_lambdas", None)
    inference = model._fit_inference_info
    inference_state = {
        name: value.copy()
        for name, value in inference.items()
        if name
        in {
            "XtWX_inv",
            "XtWX_inv_aug",
            "R_a",
            "edf",
            "edf1",
            "coefficient_estimable",
        }
    }
    inference_state["active_groups"] = inference["active_groups"]
    inference_state["group_edf_map"] = dict(inference["group_edf_map"])
    prediction = None
    if X is not None:
        prediction = model.predict(X, offset=offset).copy()
    return {
        "stable_state": stable_state,
        "public_result": fitted_result_state(getattr(model, "_result", None)),
        "solver_result": fitted_result_state(getattr(model, "_solver_result", None)),
        "fit_stats": fit_stats_state,
        "family_p": getattr(getattr(model, "family", None), "p", None),
        "distribution_p": getattr(getattr(model, "_distribution", None), "p", None),
        "reml": reml_state,
        "meta": None if type(meta) is not dict else meta.copy(),
        "lambdas": None if type(lambdas) is not dict else lambdas.copy(),
        "prediction": prediction,
        "inference": inference_state,
    }


def _validate_tweedie_profile_release_core_unchanged(
    model,
    snapshot,
    failures,
    *,
    X=None,
    offset=None,
) -> None:
    """Reject any semantic mutation performed by row-state release."""
    stable_state = snapshot["stable_state"]
    mutable_keys = {
        "_coef_covariance",
        "_fit_active_info",
        "_fit_inference_info",
        "_group_edf",
        "_dm",
        "_fit_weights",
        "_fit_offset",
        "_fit_mu",
        "_fit_null_mu",
        "_fit_X_ref",
        "_fit_y_ref",
        "_fit_sample_weight_ref",
        "_fit_offset_ref",
        "_fit_metrics_cache",
        "_fit_metrics_cache_signature",
        "_summary_cache",
    }
    released_stable_keys = set(model.__dict__) - mutable_keys
    if released_stable_keys != set(stable_state) or any(
        model.__dict__.get(name) is not value for name, value in stable_state.items()
    ):
        failures.append("released core identity state")

    def fitted_result_matches(candidate, expected) -> bool:
        if expected is None:
            return candidate is None
        beta = getattr(candidate, "beta", None)
        return bool(
            type(beta) is np.ndarray
            and type(expected["beta"]) is np.ndarray
            and np.array_equal(beta, expected["beta"])
            and all(
                getattr(candidate, name, None) == expected[name]
                for name in (
                    "intercept",
                    "n_iter",
                    "deviance",
                    "converged",
                    "phi",
                    "effective_df",
                )
            )
        )

    if not fitted_result_matches(getattr(model, "_result", None), snapshot["public_result"]):
        failures.append("released public core state")
    if not fitted_result_matches(getattr(model, "_solver_result", None), snapshot["solver_result"]):
        failures.append("released solver core state")

    fit_stats = getattr(model, "_fit_stats", None)
    current_fit_stats = (
        None
        if fit_stats is None
        else tuple(
            getattr(fit_stats, name, None)
            for name in (
                "log_likelihood",
                "null_log_likelihood",
                "null_deviance",
                "explained_deviance",
                "pearson_chi2",
                "n_obs",
            )
        )
    )
    if current_fit_stats != snapshot["fit_stats"]:
        failures.append("released fit-statistics state")
    if (
        getattr(getattr(model, "family", None), "p", None) != snapshot["family_p"]
        or getattr(getattr(model, "_distribution", None), "p", None) != snapshot["distribution_p"]
    ):
        failures.append("released family state")

    reml_result = getattr(model, "_reml_result", None)
    current_reml = (
        None
        if reml_result is None
        else (
            getattr(reml_result, "pirls_result", None),
            getattr(reml_result, "objective", None),
            getattr(reml_result, "converged", None),
            getattr(reml_result, "n_reml_iter", None),
        )
    )
    expected_reml = snapshot["reml"]
    reml_matches = (
        current_reml is None
        if expected_reml is None
        else current_reml is not None
        and current_reml[0] is expected_reml[0]
        and current_reml[1:] == expected_reml[1:]
    )
    if not reml_matches:
        failures.append("released REML state")
    meta = getattr(model, "_last_fit_meta", None)
    if (None if type(meta) is not dict else meta) != snapshot["meta"]:
        failures.append("released fit-mode state")
    lambdas = getattr(model, "_reml_lambdas", None)
    if (None if type(lambdas) is not dict else lambdas) != snapshot["lambdas"]:
        failures.append("released REML state")
    expected_prediction = snapshot["prediction"]
    if expected_prediction is not None:
        try:
            released_prediction = model.predict(X, offset=offset)
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            failures.append("released prediction state")
        else:
            if not _is_finite_tweedie_vector(
                released_prediction,
                length=len(expected_prediction),
                positive=True,
            ) or not np.array_equal(released_prediction, expected_prediction):
                failures.append("released prediction state")

    expected_inference = snapshot["inference"]
    released_inference = model.__dict__.get("_fit_inference_info")
    if type(released_inference) is not dict:
        failures.append("released inference identity state")
    else:
        for name in (
            "XtWX_inv",
            "XtWX_inv_aug",
            "R_a",
            "edf",
            "edf1",
            "coefficient_estimable",
        ):
            if not np.array_equal(released_inference.get(name), expected_inference[name]):
                failures.append("released inference identity state")
                break
        if (
            released_inference.get("active_groups") is not expected_inference["active_groups"]
            or released_inference.get("group_edf_map") != expected_inference["group_edf_map"]
        ):
            failures.append("released inference identity state")


def _validate_released_tweedie_profile_stage(
    model,
    X,
    offset,
    result,
    *,
    release_core,
) -> None:
    """Certify compact post-fit state after a non-retaining release."""
    from superglm.solvers.pirls import PIRLSResult

    failures = []
    _validate_tweedie_profile_release_core_unchanged(
        model,
        release_core,
        failures,
        X=X,
        offset=offset,
    )
    public_result = getattr(model, "_result", None)
    solver_result = getattr(model, "_solver_result", None)
    for label, candidate in (("public", public_result), ("solver", solver_result)):
        if type(candidate) is not PIRLSResult or getattr(candidate, "converged", None) is not True:
            failures.append(f"released {label} result")
            continue
        phi = getattr(candidate, "phi", None)
        if not _is_finite_tweedie_scalar(phi) or not np.isclose(
            float(phi), float(result.phi_hat), rtol=1e-12, atol=1e-14
        ):
            failures.append(f"released {label} dispersion")
        if (
            not _is_finite_tweedie_vector(getattr(candidate, "beta", None))
            or not _is_finite_tweedie_scalar(getattr(candidate, "intercept", None))
            or not _is_finite_tweedie_scalar(getattr(candidate, "deviance", None))
            or float(candidate.deviance) < 0.0
            or not _is_finite_tweedie_scalar(getattr(candidate, "effective_df", None))
            or float(candidate.effective_df) < 0.0
            or type(getattr(candidate, "n_iter", None)) not in _TWEEDIE_EXACT_INTEGER_SCALAR_TYPES
            or int(candidate.n_iter) < 0
        ):
            failures.append(f"released {label} result")
    for name in (
        "_dm",
        "_fit_weights",
        "_fit_offset",
        "_fit_mu",
        "_fit_null_mu",
        "_fit_X_ref",
        "_fit_y_ref",
        "_fit_sample_weight_ref",
        "_fit_offset_ref",
    ):
        if getattr(model, name, None) is not None:
            failures.append("released row state")
    _validate_released_tweedie_inference(model, result, failures)
    try:
        prediction = model.predict(X, offset=offset)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        failures.append("released predictions")
    else:
        if (
            type(prediction) is not np.ndarray
            or prediction.shape != (len(X),)
            or prediction.dtype.kind not in "fiu"
            or not np.all(np.isfinite(prediction))
            or np.any(prediction <= 0.0)
        ):
            failures.append("released predictions")
    if failures:
        joined = ", ".join(dict.fromkeys(failures))
        raise RuntimeError(f"Tweedie released fit is not installable: invalid {joined}")


def _validate_released_tweedie_inference(model, result, failures) -> None:
    """Validate compact coefficient-space caches distilled before row release."""
    from superglm.types import GroupSlice

    inference = model.__dict__.get("_fit_inference_info")
    required = {
        "W",
        "XtWX_inv",
        "XtWX_inv_aug",
        "active_groups",
        "R_a",
        "edf",
        "edf1",
        "group_edf_map",
        "coefficient_estimable",
    }
    if type(inference) is not dict or not required.issubset(inference):
        failures.append("released inference state")
        return

    edf = inference["edf"]
    if not _is_finite_tweedie_vector(edf):
        failures.append("released inference state")
        return
    if np.any(edf < -0.01) or np.any(edf > 1.01):
        failures.append("released coefficient EDF state")
    n_active = len(edf)
    edf1 = inference["edf1"]
    solver_result = getattr(model, "_solver_result", None)
    n_coefficients = (
        len(solver_result.beta)
        if _is_finite_tweedie_vector(getattr(solver_result, "beta", None))
        else n_active
    )
    if (
        not _is_finite_tweedie_vector(inference["W"], length=0)
        or not _is_finite_tweedie_matrix(inference["XtWX_inv"], shape=(n_active, n_active))
        or not _is_finite_tweedie_matrix(
            inference["XtWX_inv_aug"], shape=(n_active + 1, n_active + 1)
        )
        or not _is_finite_tweedie_matrix(inference["R_a"], shape=(n_active, n_active))
        or not _is_finite_tweedie_vector(edf1, length=n_active)
        or not _is_tweedie_bool_vector(inference["coefficient_estimable"], length=n_coefficients)
    ):
        failures.append("released inference state")
    elif np.any(edf1 < -0.01) or np.any(edf1 > 1.01):
        failures.append("released coefficient EDF1 state")

    active_groups = inference["active_groups"]
    active_groups_valid = type(active_groups) is list and all(
        type(group) is GroupSlice for group in active_groups
    )
    if not active_groups_valid:
        failures.append("released inference state")

    group_edf_map = inference["group_edf_map"]
    installed_group_edf = model.__dict__.get("_group_edf")
    if (
        type(group_edf_map) is not dict
        or type(installed_group_edf) is not dict
        or set(group_edf_map) != set(installed_group_edf)
        or any(type(name) is not str for name in group_edf_map)
        or any(
            not _is_finite_tweedie_scalar(value) or float(value) < -1e-8
            for value in group_edf_map.values()
        )
        or any(
            not _is_finite_tweedie_scalar(installed_group_edf[name])
            or not np.isclose(
                float(installed_group_edf[name]),
                float(group_edf_map[name]),
                rtol=1e-12,
                atol=1e-14,
            )
            for name in group_edf_map
        )
    ):
        failures.append("released group EDF state")
    elif active_groups_valid:
        expected_group_names = [group.name for group in active_groups]
        expected_start = 0
        contiguous_slices = True
        for group in active_groups:
            if group.start != expected_start or group.end <= group.start:
                contiguous_slices = False
                break
            expected_start = group.end
        contiguous_slices = contiguous_slices and expected_start == n_active
        slices_valid = (
            len(expected_group_names) == len(set(expected_group_names))
            and contiguous_slices
            and all(
                type(group.start) is int
                and type(group.end) is int
                and 0 <= group.start <= group.end <= n_active
                for group in active_groups
            )
        )
        active_name_set = set(expected_group_names)
        inactive_names = set(group_edf_map) - active_name_set
        if (
            not slices_valid
            or not active_name_set.issubset(group_edf_map)
            or any(abs(float(group_edf_map[name])) > 1e-8 for name in inactive_names)
        ):
            failures.append("released group EDF state")
        else:
            expected_group_edf = {
                group.name: float(np.sum(edf[group.sl])) for group in active_groups
            }
            if any(
                not np.isclose(
                    float(group_edf_map[name]),
                    expected_group_edf[name],
                    rtol=1e-10,
                    atol=1e-10,
                )
                for name in expected_group_edf
            ):
                failures.append("released group EDF state")

    public_result = getattr(model, "_result", None)
    if (
        public_result is None
        or not _is_finite_tweedie_scalar(getattr(public_result, "effective_df", None))
        or not np.isclose(
            float(public_result.effective_df),
            1.0 + float(np.sum(edf)),
            rtol=1e-9,
            atol=1e-9,
        )
    ):
        failures.append("released effective-DF state")

    covariance_state = model.__dict__.get("_coef_covariance")
    if type(covariance_state) is not tuple or len(covariance_state) != 2:
        failures.append("released covariance state")
        return
    covariance, covariance_groups = covariance_state
    if (
        not _is_finite_tweedie_matrix(covariance, shape=(n_active, n_active))
        or covariance_groups is not active_groups
    ):
        failures.append("released covariance state")
        return
    expected_covariance = float(result.phi_hat) * inference["XtWX_inv_aug"][1:, 1:]
    if not np.allclose(covariance, expected_covariance, rtol=1e-11, atol=1e-13):
        failures.append("released covariance state")


def _snapshot_tweedie_profile_refit_inputs(X, y, sample_weight, offset):
    """Own one coherent input set for both profiling and its final refit."""
    return X, y, sample_weight, offset


def _replace_dataclass_preserving_dynamic_attributes(instance, **changes):
    """Replace dataclass fields without dropping solver-added attributes."""
    replacement = replace(instance, **changes)
    declared_names = {field.name for field in fields(instance)}
    for name, value in vars(instance).items():
        if name not in declared_names:
            setattr(replacement, name, value)
    return replacement


def _replace_pirls_phi(result, phi):
    """Return a phi-adjusted PIRLS result with all runtime metadata intact."""
    return _replace_dataclass_preserving_dynamic_attributes(result, phi=float(phi))


def _synchronize_tweedie_profile_refit(model, y, profile_result) -> None:
    """Atomically synchronize a retained final refit to the profiled dispersion."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu

    distribution = model._distribution
    if not isinstance(distribution, Tweedie) or distribution.p != profile_result.p_hat:
        raise RuntimeError("Final Tweedie refit does not match the profiled power parameter")
    if model._dm is None or model._fit_weights is None:
        raise RuntimeError("Final Tweedie refit state was released before synchronization")

    public_result = model.result
    solver_result = model._solver_pirls_result()
    weights = model._fit_weights
    offset_arr = model._fit_offset
    y_arr = np.asarray(y, dtype=np.float64)

    eta = model._dm.matvec(solver_result.beta) + solver_result.intercept
    if offset_arr is not None:
        eta = eta + offset_arr
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), distribution)
    null_mu = _compute_null_mu(y_arr, weights, offset_arr, distribution, model._link)
    fit_stats = _compute_fit_stats(
        y_arr,
        mu,
        weights,
        offset_arr,
        distribution,
        model._link,
        profile_result.phi_hat,
        null_mu=null_mu,
    )

    replacement_public = _replace_pirls_phi(public_result, profile_result.phi_hat)
    replacement_solver = _replace_pirls_phi(solver_result, profile_result.phi_hat)
    reml_result = getattr(model, "_reml_result", None)
    replacement_reml = (
        None
        if reml_result is None
        else _replace_dataclass_preserving_dynamic_attributes(
            reml_result, pirls_result=replacement_solver
        )
    )

    model.family = distribution
    model._result = replacement_public
    model._solver_result = replacement_solver
    if reml_result is not None:
        model._reml_result = replacement_reml
    model._fit_mu = mu
    model._fit_null_mu = null_mu
    model._fit_stats = fit_stats

    for cache_name in (
        "_coef_covariance",
        "_fit_active_info",
        "_fit_inference_info",
        "_group_edf",
    ):
        model.__dict__.pop(cache_name, None)
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def estimate_theta(model, X, y, sample_weight=None, offset=None, *, fit_mode="fit", **kwargs):
    """Estimate NB theta via profile likelihood, refit, and return result."""
    from superglm.profiling.nb import estimate_nb_theta

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)

    progress_callback = kwargs.pop("progress_callback", None)
    result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset, **kwargs)
    if progress_callback is not None:
        progress_callback("best_found", {"profile_estimate": _theta_estimate_payload(result)})
    model.family = NegativeBinomial(theta=result.theta_hat)
    if progress_callback is not None:
        progress_callback("final_refit", {"profile_estimate": _theta_estimate_payload(result)})
    if resolved_mode == "fit_reml":
        model.fit_reml(X, y, sample_weight=sample_weight, offset=offset)
    else:
        model.fit(X, y, sample_weight=sample_weight, offset=offset)
    model._nb_profile_result = result  # after refit so fit()'s clear doesn't wipe it
    return result


def _resolve_profile_fit_mode(model, fit_mode: str) -> str:
    """Resolve public profile fit mode to an internal final-refit method."""
    valid_fit_modes = {"fit", "reml", "inherit"}
    if type(fit_mode) is not str or fit_mode not in valid_fit_modes:
        raise ValueError(
            f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(valid_fit_modes)}"
        )
    if fit_mode == "reml":
        return "fit_reml"
    if fit_mode == "inherit":
        meta = getattr(model, "_last_fit_meta", None)
        if meta is not None and meta.get("method") == "fit_reml":
            return "fit_reml"
    return "fit"


def _tweedie_estimate_payload(result, *, ci_alpha=0.05):
    ci, ci_status = cached_tweedie_profile_ci(result, ci_alpha)
    ci_low, ci_high = (None, None) if ci is None else ci
    return {
        "parameter": "p",
        "label": "p_hat",
        "value": getattr(result, "p_hat", None),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_alpha": ci_alpha,
        "ci_status": ci_status,
        "objective": getattr(result, "nll", None),
        "objective_label": "loss",
        "lower_is_better": True,
    }


def _theta_estimate_payload(result):
    return {
        "parameter": "theta",
        "label": "theta_hat",
        "value": getattr(result, "theta_hat", None),
        "ci_low": _cached_ci(result)[0],
        "ci_high": _cached_ci(result)[1],
        "objective": getattr(result, "nll", None),
        "objective_label": "loss",
        "lower_is_better": True,
    }


def _cached_ci(result):
    cache = getattr(result, "_ci_cache", None)
    if isinstance(cache, dict) and 0.05 in cache:
        return cache[0.05]
    return (None, None)
