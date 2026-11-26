"""The experimental label warns once per session, at the caller's line, and the class filter silences it."""

from __future__ import annotations

import warnings

import pytest

import superglm
from superglm import _experimental
from superglm._experimental import ExperimentalWarning, warn_experimental


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_experimental, "_announced", False)


def test_first_call_warns_at_the_callers_line_and_later_calls_do_not() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_experimental("predict_pdf")
        warn_experimental("mean_excess")
    assert [w.category for w in caught] == [ExperimentalWarning]
    assert caught[0].filename == __file__
    message = str(caught[0].message)
    assert message.startswith("predict_pdf is experimental")
    assert "superglm.ExperimentalWarning" in message


def test_the_class_filter_silences_it() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=superglm.ExperimentalWarning)
        warn_experimental("predict_pdf")
    assert caught == []


def test_an_error_filter_keeps_raising_until_the_class_is_filtered() -> None:
    """Mutation check: setting the flag before warning lets the second call pass silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        for name in ("predict_pdf", "mean_excess"):
            with pytest.raises(ExperimentalWarning, match=f"^{name} is experimental"):
                warn_experimental(name)
