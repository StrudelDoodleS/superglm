"""Compact history has an explicit read barrier for older artifact readers."""

import json

import pytest

from superglm.distributional import serialization as serialization_module
from superglm.distributional.serialization import (
    DistributionalSerializationError,
    deserialize_distributional_model,
    distributional_manifest,
    serialize_distributional_model,
)
from tests.test_distributional_history_rows import _fit, _mutated_artifact


def test_full_history_manifest_adds_only_the_explicit_retention_flag():
    compact = distributional_manifest(_fit(retain_history_rows=False))["smoothing"]["config"]
    full = distributional_manifest(_fit(retain_history_rows=True))["smoothing"]["config"]
    assert "retain_history_rows" not in compact
    assert full == {**compact, "retain_history_rows": True}


@pytest.mark.parametrize("retain_history_rows", [False, True])
def test_history_artifact_refuses_schema_nine_reader_before_unpickling(
    monkeypatch, retain_history_rows
):
    artifact = serialize_distributional_model(_fit(retain_history_rows=retain_history_rows))
    assert json.loads(artifact)["schema_version"] == "10.0.0"

    # Model the previous reader's version gate. A new payload must be refused
    # there, before an old constructor or manifest comparison sees new state.
    monkeypatch.setattr(serialization_module, "SCHEMA_VERSION", "9.0.0")
    monkeypatch.setattr(serialization_module, "READABLE_PREVIOUS_MAJORS", frozenset({8}))

    def unexpected_unpickle(*args, **kwargs):
        raise AssertionError("the old reader must refuse before unpickling")

    monkeypatch.setattr(serialization_module.pickle, "loads", unexpected_unpickle)
    with pytest.raises(DistributionalSerializationError, match="major versions differ"):
        deserialize_distributional_model(artifact)


def test_schema_nine_full_history_without_new_fields_remains_readable():
    model = _fit(retain_history_rows=True)

    def old_fields(restored):
        vars(restored.smoothing.config).pop("retain_history_rows")
        for fit in restored.smoothing.coefficient_fits:
            vars(fit).pop("row_shape")

    artifact = json.loads(_mutated_artifact(model, old_fields, legacy=True))
    artifact["schema_version"] = "9.0.0"
    restored = deserialize_distributional_model(json.dumps(artifact))
    assert restored.smoothing.config.retain_history_rows is False
    assert restored.smoothing.history == model.smoothing.history
    assert all(fit.eta is not None for fit in restored.smoothing.coefficient_fits)
