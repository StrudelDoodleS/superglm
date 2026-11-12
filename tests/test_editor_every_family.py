"""Every exported family walks through every editor step.

The SuperLSS cases come from the example table in ``tests/_lss_editor_fixtures.py``
(``_SCALAR_LAWS``); the SuperGLM cases from ``GLM_EXAMPLES`` here, each family at its
default link plus one case for each kind of link the editor draws (spec section 8 of
``docs/superpowers/specs/2026-11-12-editor-every-family-design.md``). A family the
library exports without an entry fails ``test_every_exported_family_has_an_example``.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

import superglm.distributional.families as lss_families
import superglm.distributions as glm_distributions
from superglm import Categorical, Spline, SuperGLM, SuperLSS, cross_validate
from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
from superglm.editor import edit
from superglm.editor.cv import capture_cv_view, fold_log_curves, fold_term_items
from superglm.editor.display import display_for
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import (
    EXAMPLE_FAMILIES,
    EXAMPLE_FAMILY_CLASSES,
    open_case,
    scalar_family_case,
)

# Fitted on a vector response; the editor opens it in stage 3 of the design.
REFUSED = {"DirichletLSS": "editor is not supported for vector responses by DirichletLSS"}

LOG_RELATIVITY = {"kind": "relativity", "link": "log"}
N_GLM = 300
LEVELS = ("A", "B", "C", "D")


def _compound_poisson_gamma(rng, mu):
    """Tweedie responses at power 1.5 and dispersion 1 (Jorgensen 1987)."""
    power, dispersion = 1.5, 1.0
    rate = mu ** (2.0 - power) / (dispersion * (2.0 - power))
    shape = (2.0 - power) / (power - 1.0)
    scale = dispersion * (power - 1.0) * mu ** (power - 1.0)
    jumps = rng.poisson(rate)
    return np.where(jumps > 0, rng.gamma(shape * np.maximum(jumps, 1), scale), 0.0)


@dataclass(frozen=True)
class GLMExample:
    """A distribution, the mean of its intercept-only law and a sampler ``(rng, mu) -> y``."""

    make: Any
    mean: float
    sample: Any
    response: str


# The SuperGLM example table: one entry per distribution class in ``superglm.distributions``.
GLM_EXAMPLES = {
    "Poisson": GLMExample(Poisson, 2.0, lambda rng, mu: rng.poisson(mu), "count"),
    "Gaussian": GLMExample(
        Gaussian, 1000.0, lambda rng, mu: mu + 100.0 * rng.standard_normal(mu.size), "real"
    ),
    "Gamma": GLMExample(Gamma, 1000.0, lambda rng, mu: rng.gamma(2.0, mu / 2.0), "positive"),
    "NegativeBinomial": GLMExample(
        lambda: NegativeBinomial(theta=2.0),
        2.0,
        lambda rng, mu: rng.negative_binomial(2.0, 2.0 / (2.0 + mu)),
        "count",
    ),
    "Binomial": GLMExample(Binomial, 0.2, lambda rng, mu: rng.binomial(1, mu), "binary"),
    "Tweedie": GLMExample(lambda: Tweedie(p=1.5), 3.0, _compound_poisson_gamma, "zero mass"),
}
# Each kind of link once, on a family where it is conventional, beside the defaults.
GLM_LINKS = (
    ("Gaussian", "log"),
    ("Gamma", "inverse"),
    ("Binomial", "log"),
    ("Binomial", "probit"),
    ("Binomial", "cloglog"),
    ("Poisson", "sqrt"),
)
GLM_CASES = tuple((name, None) for name in GLM_EXAMPLES) + GLM_LINKS
# One case per response kind and model kind runs Cross-validation's fold curves.
CV_CASES = {
    ("glm", "Poisson", None),
    ("glm", "Gaussian", None),
    ("glm", "Gamma", None),
    ("glm", "Binomial", None),
    ("glm", "Tweedie", None),
    ("lss", "GaussianLS", None),
    ("lss", "GammaLS", None),
    ("lss", "NegativeBinomialLS", None),
    ("lss", "TweedieLSS", None),
}


def _exported_lss_families() -> set[str]:
    return {
        name
        for name in lss_families.__all__
        if inspect.isclass(cls := getattr(lss_families, name)) and hasattr(cls, "parameters")
    }


def _exported_glm_distributions() -> set[str]:
    return {
        name
        for name, cls in vars(glm_distributions).items()
        if inspect.isclass(cls)
        and cls.__module__ == glm_distributions.__name__
        and cls is not glm_distributions.Distribution
        and hasattr(cls, "default_link")
    }


def test_every_exported_family_has_an_example():
    for name in sorted(_exported_lss_families() - EXAMPLE_FAMILY_CLASSES - set(REFUSED)):
        pytest.fail(f"add {name} to _SCALAR_LAWS in tests/_lss_editor_fixtures.py")
    for name in sorted(_exported_glm_distributions() - set(GLM_EXAMPLES)):
        pytest.fail(f"add {name} to GLM_EXAMPLES in tests/test_editor_every_family.py")


def test_a_vector_response_family_still_refuses():
    from superglm.distributional.families import DirichletLSS

    rng = np.random.default_rng(20261112)
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, 120)})
    y = pd.DataFrame(rng.dirichlet([2.0, 3.0, 4.0], 120), columns=["a", "b", "c"])
    family = DirichletLSS(("a", "b", "c"))
    model = SuperLSS(family, *(family.alpha(c) for c in family.components)).fit(X, y)
    with pytest.raises(NotImplementedError, match=REFUSED["DirichletLSS"]):
        edit(model, train_data=(X, y))


@pytest.fixture(scope="module")
def glm_rows():
    rng = np.random.default_rng(20261112)
    X = pd.DataFrame(
        {"x": rng.uniform(0.0, 1.0, N_GLM), "g": rng.choice(LEVELS, N_GLM, p=[0.4, 0.3, 0.2, 0.1])}
    )
    effect = (
        0.3 * np.sin(np.pi * X["x"].to_numpy())
        + X["g"].map({"A": 0.0, "B": 0.2, "C": -0.15, "D": 0.3}).to_numpy()
    )
    return X, effect


def _glm_case(name, link, glm_rows):
    X, effect = glm_rows
    example = GLM_EXAMPLES[name]
    mu = example.mean * np.exp(effect)
    y = np.asarray(example.sample(np.random.default_rng(7), mu), dtype=np.float64)
    model = SuperGLM(
        family=example.make(),
        link=link,
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=3), "g": Categorical(base="first")},
    )
    model.fit(X, y)
    return model, X, y


def _job(widget, kind):
    started = widget._job_start(kind)
    status = widget._job_status(started["job_id"], wait=True)
    assert status["status"] == "done", status
    return status


def _walk(session, *, lss: bool, cv_rows=None, rating_tables: bool = True) -> None:
    """Drive every editor step on ``session``; ``cv_rows`` adds the fold curves.

    Each log-link term previews its rating table unless ``rating_tables`` is False.
    """
    payload = session_payload(session)
    assert payload
    for key in session.terms:
        assert np.all(np.isfinite(payload[key]["y"])), key
        assert payload[key]["y_label"]

    # An edit on each term moves its curve; undo puts the last one back.
    for key, term in session.terms.items():
        session.select_indices(key, list(range(term.size // 3, 2 * term.size // 3 + 1)))
        session.shift(key, 0.05)
        assert not np.allclose(term.edited_log_effect, term.original_log_effect), key
    last = list(session.terms)[-1]
    session.undo()
    np.testing.assert_allclose(
        session.terms[last].edited_log_effect, session.terms[last].original_log_effect
    )

    edited = session.to_model()
    X = session._evaluation_data["train"].X
    assert np.all(np.isfinite(np.asarray(edited.predict(X), dtype=np.float64)))

    widget = session.widget(mode="server")
    try:
        state = widget._state()
        assert set(state["terms"]) == set(session.terms)
        validation = widget._report("validation")
        (train,) = validation["splits"]
        assert train.get("available", True), train
        # A family with neither a closed-form CRPS nor a distribution function scores none.
        scores = train["metrics"]["edited"]
        assert np.isfinite(scores.get("log_score", scores.get("deviance"))), scores
        if lss:
            column = "age"
            assert widget._distribution(column)["parameters"]
            row = widget._risk_row(None)
            assert column in row["values"]
        for key in session.terms if rating_tables else ():
            if display_for(session.model, key).to_payload() == LOG_RELATIVITY:
                table = widget._rating_table(key)
                assert table["available"], table
        _job(widget, "final_fit")
        final, _source = widget._final_fit_for_export()
        assert np.all(np.isfinite(np.asarray(final.predict(X), dtype=np.float64)))
        exported = joblib.load(io.BytesIO(widget._export_bytes("joblib").data))
        assert np.all(np.isfinite(np.asarray(exported.predict(X), dtype=np.float64)))
    finally:
        widget.close()

    if cv_rows is not None:
        model, X, y = cv_rows
        cv = cross_validate(
            model,
            X,
            y,
            cv=KFold(2, shuffle=True, random_state=0),
            return_estimators=True,
            error_score="raise",
            **({"fit_mode": "fit_reml"} if lss else {}),
        )
        curves = {i: fold_log_curves(est, session.terms) for i, est in enumerate(cv.estimators)}
        view = capture_cv_view(session, run=None, final_fit=None)
        items = fold_term_items(session.terms, curves, display=view.display)
        assert {item["name"] for item in items} == set(session.terms)


@pytest.mark.parametrize(
    ("name", "link"), GLM_CASES, ids=[f"{n}-{link or 'default'}" for n, link in GLM_CASES]
)
def test_superglm_family_works_in_the_editor(name, link, glm_rows):
    model, X, y = _glm_case(name, link, glm_rows)
    session = edit(model, train_data=(X, y))
    assert list(session.terms) == ["x", "g"]
    cv_rows = (model, X, y) if ("glm", name, link) in CV_CASES else None
    # The Excel export refuses a binomial: its predictions are clamped, and a clamp is no factor.
    _walk(session, lss=False, cv_rows=cv_rows, rating_tables=name != "Binomial")


@pytest.mark.parametrize("name", EXAMPLE_FAMILIES)
def test_superlss_family_works_in_the_editor(name):
    case = scalar_family_case(name, levels=True)
    session = open_case(case)
    first, second, *_ = case.model.parameter_names_
    assert list(session.terms) == [f"{first}:age", f"{first}:region", f"{second}:age"]
    cv_rows = (case.model, case.X, case.y) if ("lss", name, None) in CV_CASES else None
    _walk(session, lss=True, cv_rows=cv_rows)
