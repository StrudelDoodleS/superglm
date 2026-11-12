"""Each built-in family parameter declares the name a user reads."""

from __future__ import annotations

import copyreg
import dataclasses
import io
import pickle

import pytest

import superglm.distributional.families as families
from superglm.distributional.families import (
    DirichletLSS,
    GammaLS,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
)
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.editor.payloads import session_payload
from superglm.links import LogLink
from tests._lss_editor_fixtures import gamma_case, open_case

_NEW_FIELDS = ("label", "description", "quantity")


def _exported_families():
    for name in families.__all__:
        cls = getattr(families, name)
        if not name.endswith(("LS", "LSS")):
            continue
        family = cls(components=("a", "b", "c")) if cls is DirichletLSS else cls()
        yield family
        if getattr(family, "parametrisation", None) == "mean":
            yield cls(parametrisation="location")


def _labels(family) -> dict[str, str | None]:
    return {parameter.name: parameter.label for parameter in family.parameters}


def test_every_built_in_parameter_has_a_label():
    built = list(_exported_families())
    assert len(built) == 15  # 12 exported families, 3 with a location form
    for family in built:
        for parameter in family.parameters:
            where = f"{type(family).__name__}.{parameter.name}"
            assert isinstance(parameter.label, str) and parameter.label.strip(), where
            assert isinstance(parameter.description, str), where
            assert parameter.description.endswith("."), where


def test_today_s_editor_labels_are_declared():
    assert _labels(GammaLS()) == {"mean": "Mean", "scale": "CV"}
    log_normal = {parameter.name: parameter for parameter in LogNormalLS().parameters}
    assert log_normal["scale"].label == "CV"
    assert log_normal["scale"].quantity == "σ"
    assert _labels(NegativeBinomialLS())["theta"] == "Overdispersion"
    assert _labels(TweedieLSS()) == {
        "mean": "Mean",
        "dispersion": "Dispersion",
        "power": "Power",
    }
    assert _labels(DirichletLSS(components=("a", "b", "c"))) == {
        "alpha_0": "a",
        "alpha_1": "b",
        "alpha_2": "c",
    }


class _MasterPickler(pickle.Pickler):
    """Pickles a ParameterSpec the way master did: its state has none of the new fields."""

    def reducer_override(self, obj):
        if type(obj) is not ParameterSpec:
            return NotImplemented
        state = {k: v for k, v in vars(obj).items() if k not in _NEW_FIELDS}
        return copyreg.__newobj__, (ParameterSpec,), state


def _master_pickle(obj) -> bytes:
    buffer = io.BytesIO()
    _MasterPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    return buffer.getvalue()


def test_parameter_spec_pickled_without_labels_loads():
    old = object.__new__(ParameterSpec)
    old.__dict__.update(
        name="scale",
        default_link=LogLink(),
        role="scale",
        support=ParameterSupport(lower=0.0),
        curvature="observed",
    )
    loaded = pickle.loads(pickle.dumps(old))
    assert loaded.label is None
    assert loaded.description is None
    assert loaded.quantity is None
    assert loaded.name == "scale"

    case = gamma_case()
    restored = pickle.loads(_master_pickle(case.model))
    payload = session_payload(open_case(dataclasses.replace(case, model=restored)))
    assert payload == session_payload(open_case(case))


def test_label_must_be_text():
    with pytest.raises(ValueError, match="label"):
        ParameterSpec("scale", LogLink(), "scale", ParameterSupport(lower=0.0), label="")
    with pytest.raises(ValueError, match="quantity"):
        ParameterSpec("scale", LogLink(), "scale", ParameterSupport(lower=0.0), quantity=" ")
