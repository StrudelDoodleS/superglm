"""SuperLSS hand-edited coefficients: predictions follow the edit, the fit's records do not."""

from __future__ import annotations

import dataclasses
import itertools
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import LogNormalLS, SuperLSS, s
from superglm.distributional import DistributionFunctionFamily, EditedCoefficientsWarning
from superglm.distributional.fit_state import DistributionalFitState
from superglm.distributional.revision import HandEdit
from tests._lss_editor_fixtures import EPS, gamma_case, lognormal_case, nb_case

# (case builder, the parameter whose terms the tests edit)
CASES = [
    pytest.param(lognormal_case, "scale", id="lognormal"),
    pytest.param(gamma_case, "scale", id="gamma"),
    pytest.param(nb_case, "theta", id="negative-binomial"),
]

EDITED_WARNING = (
    "Coefficients of {keys} were edited by hand; "
    "standard errors, bands and tests are from the fit before the edit."
)
CORRECTED_REFUSAL = (
    "The corrected covariance needs the fitted coefficients; use covariance='fixed', or refit."
)
NO_FRAME = (
    "Pass X_train= to revise the coefficients of a model whose training frame is not attached."
)


def _coefficients(model: SuperLSS) -> np.ndarray:
    return np.fromiter(model.coef_.values(), dtype=np.float64)


def _layout(model: SuperLSS):
    return model._model.layout


def _intercept(model: SuperLSS, parameter: str) -> int:
    return _layout(model).predictors[model.parameter_names_.index(parameter)].intercept_index


def _link(model: SuperLSS, case, X: pd.DataFrame | None = None) -> pd.DataFrame:
    return model.predict_link(case.X if X is None else X, offsets=case.offsets)


def _edit_term(model: SuperLSS, key: str, scale: float) -> np.ndarray:
    coefficients = _coefficients(model)
    coefficients[_layout(model).term_slices[key]] *= scale
    return coefficients


def _inference_calls(model: SuperLSS, case, parameter: str):
    return {
        "term_inference": lambda **kw: model.term_inference(
            parameter, "age", X_train=case.X, simultaneous=False, **kw
        ),
        "term_test": lambda **kw: model.term_test(parameter, "age", X_train=case.X, **kw),
        "summary": lambda **kw: model.summary(X_train=case.X, **kw),
    }


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_intercept_shift_moves_only_its_parameter(builder, parameter):
    case = builder()
    m = case.model
    other = next(name for name in m.parameter_names_ if name != parameter)
    coef = _coefficients(m)
    coef[_intercept(m, parameter)] += 0.3
    edited = m._with_revised_coefficients(coef, edited_terms=[f"{parameter}:age"], X_train=case.X)
    before, after = _link(m, case), _link(edited, case)
    np.testing.assert_array_equal(after[other], before[other])
    np.testing.assert_allclose(
        after[parameter] - before[parameter],
        0.3,
        rtol=0,
        atol=8 * EPS * np.abs(before[parameter]).max(),
    )


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_zeroed_term_has_no_effect(builder, parameter):
    case = builder()
    edited = case.model._with_revised_coefficients(
        _edit_term(case.model, f"{parameter}:age", 0.0),
        edited_terms=[f"{parameter}:age"],
        X_train=case.X,
    )
    with pytest.warns(EditedCoefficientsWarning):
        effect = edited.term_inference(parameter, "age", X_train=case.X, simultaneous=False)
    np.testing.assert_array_equal(effect.effect, 0.0)
    moved = case.X.assign(age=108.0 - case.X["age"])
    before, after = _link(edited, case), _link(edited, case, moved)
    np.testing.assert_array_equal(after[parameter], before[parameter])
    # The mean still moves with age, so the equality above is not vacuous.
    assert not np.array_equal(after["mean"], before["mean"])


def test_original_model_is_untouched():
    case = lognormal_case()
    m = case.model
    before = _link(m, case)
    coefficients = _coefficients(m)
    edited = m._with_revised_coefficients(
        _edit_term(m, "scale:age", 0.5), edited_terms=["scale:age"], X_train=case.X
    )
    assert edited is not m
    pd.testing.assert_frame_equal(_link(m, case), before)
    np.testing.assert_array_equal(_coefficients(m), coefficients)
    assert m._hand_edited_terms == ()
    assert m._model.fit_state.hand_edit is None
    assert edited._hand_edited_terms == ("scale:age",)
    assert not np.array_equal(_coefficients(edited), coefficients)


def test_fit_records_keep_describing_the_fit():
    case = lognormal_case()
    m = case.model
    edited = m._with_revised_coefficients(
        _edit_term(m, "scale:age", 0.5), edited_terms=["scale:age"], X_train=case.X
    )
    fitted, revised = m._model, edited._model
    assert revised.result is fitted.result
    assert revised.inference is fitted.inference
    assert revised.smoothing is fitted.smoothing
    assert revised.fitted_result is fitted.fitted_result
    np.testing.assert_array_equal(edited.covariance_, m.covariance_)
    np.testing.assert_array_equal(edited.result_.coefficients, m.result_.coefficients)
    # The accessors that describe predictions read the edit.
    layout = _layout(m)
    scale_slice = layout.predictors[1].coefficient_slice
    np.testing.assert_array_equal(
        edited.coef_by_predictor_["scale"], _coefficients(edited)[scale_slice]
    )
    assert not np.array_equal(edited.coef_by_predictor_["scale"], m.coef_by_predictor_["scale"])
    np.testing.assert_array_equal(edited.coef_by_predictor_["mean"], m.coef_by_predictor_["mean"])


def test_second_revision_unions_the_edited_terms_and_compares_with_the_fit():
    case = lognormal_case()
    m = case.model
    first = m._with_revised_coefficients(
        _edit_term(m, "scale:age", 0.5), edited_terms=["scale:age"], X_train=case.X
    )
    coefficients = _coefficients(first)
    coefficients[_layout(m).term_slices["mean:age"]] += 0.1
    second = first._with_revised_coefficients(
        coefficients, edited_terms=["mean:age"], X_train=case.X
    )
    assert second._hand_edited_terms == ("scale:age", "mean:age")
    assert second._model.fit_state.hand_edit.edited_terms == ("scale:age", "mean:age")
    # The earlier edit differs from the fit, and the union still allows it.
    np.testing.assert_array_equal(_coefficients(second), coefficients)
    assert first._hand_edited_terms == ("scale:age",)
    # Terms outside the union are refused against the fit, not against the last edit.
    stray = coefficients.copy()
    stray[_layout(m).term_slices["mean:power"].start] += 1.0
    with pytest.raises(ValueError, match=re.escape("mean:power[0]")):
        second._with_revised_coefficients(stray, edited_terms=["mean:age"], X_train=case.X)


def _stray(model: SuperLSS, entry: str) -> np.ndarray:
    coefficients = _coefficients(model)
    coefficients[_layout(model).coefficient_names.index(entry)] += 0.25
    return coefficients


@pytest.mark.parametrize(
    ("entry", "edited_terms"),
    [
        ("mean:power[0]", ["mean:age"]),
        ("mean:age[3]", ["scale:age"]),
        ("scale:(intercept)", ["mean:age"]),
        ("scale:region[2]", ["scale:age"]),
    ],
)
def test_coefficients_outside_the_edited_terms_are_refused(entry, edited_terms):
    case = lognormal_case()
    m = case.model
    with pytest.raises(ValueError, match=re.escape(entry)):
        m._with_revised_coefficients(_stray(m, entry), edited_terms=edited_terms, X_train=case.X)


def test_the_first_offending_entry_is_named():
    case = lognormal_case()
    m = case.model
    names = _layout(m).coefficient_names
    coefficients = _stray(m, "scale:region[2]")
    coefficients[names.index("mean:power[1]")] += 0.25
    with pytest.raises(ValueError, match=re.escape("mean:power[1]")) as excinfo:
        m._with_revised_coefficients(coefficients, edited_terms=["scale:age"], X_train=case.X)
    assert "scale:region[2]" not in str(excinfo.value)


def test_the_fit_state_itself_checks_the_edit():
    case = lognormal_case()
    state = case.model._model.fit_state
    bad = HandEdit(_stray(case.model, "mean:power[0]"), ("mean:age",))
    with pytest.raises(ValueError, match=re.escape("mean:power[0]")):
        dataclasses.replace(state, hand_edit=bad)
    good = HandEdit(_edit_term(case.model, "mean:age", 0.5), ("mean:age",))
    assert dataclasses.replace(state, hand_edit=good).reported_coefficients is good.coefficients


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda c, n: c[:-1], "31"),
        (lambda c, n: np.where(np.arange(len(c)) == 1, np.nan, c), "finite"),
    ],
    ids=["wrong-length", "non-finite"],
)
def test_malformed_vectors_are_refused(make, message):
    case = lognormal_case()
    m = case.model
    coefficients = make(_coefficients(m), _layout(m).coefficient_names)
    with pytest.raises(ValueError, match=message):
        m._with_revised_coefficients(coefficients, edited_terms=["mean:age"], X_train=case.X)


def test_unknown_edited_term_is_refused():
    case = lognormal_case()
    m = case.model
    with pytest.raises(ValueError, match=re.escape("mean:nope")):
        m._with_revised_coefficients(_coefficients(m), edited_terms=["mean:nope"], X_train=case.X)


def test_hand_edit_array_is_read_only_and_frozen():
    case = lognormal_case()
    m = case.model
    edit = HandEdit(_coefficients(m), ["mean:age", "mean:age"])
    assert edit.edited_terms == ("mean:age",)
    assert not edit.coefficients.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        edit.edited_terms = ()


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_inference_methods_warn_after_an_edit(builder, parameter):
    case = builder()
    m = case.model
    keys = [f"{parameter}:age", f"{parameter}:region"]
    coefficients = _edit_term(m, keys[0], 0.5)
    edited = m._with_revised_coefficients(coefficients, edited_terms=keys, X_train=case.X)
    expected = EDITED_WARNING.format(keys=", ".join(keys))
    for name, call in _inference_calls(edited, case, parameter).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        edits = [w for w in caught if issubclass(w.category, EditedCoefficientsWarning)]
        assert [str(w.message) for w in edits] == [expected], name
        assert edits[0].filename == __file__, name
        with pytest.raises(ValueError, match=re.escape(CORRECTED_REFUSAL)):
            call(covariance="corrected")


def test_an_unedited_model_does_not_warn():
    case = lognormal_case()
    with warnings.catch_warnings():
        warnings.simplefilter("error", EditedCoefficientsWarning)
        for call in _inference_calls(case.model, case, "scale").values():
            call()
        for call in _posterior_calls(case.model, case, "scale").values():
            call()


def _edited_by_intercept_shift(case, parameter: str, shift: float = 1.5):
    """The case's model with ``parameter``'s intercept moved by ``shift`` on the link scale."""
    m = case.model
    coefficients = _coefficients(m)
    coefficients[_intercept(m, parameter)] += shift
    return m._with_revised_coefficients(
        coefficients, edited_terms=[f"{parameter}:age"], X_train=case.X
    )


def _posterior_calls(model: SuperLSS, case, parameter: str):
    rows = case.X.head(20)
    offsets = None if case.offsets is None else {k: v[:20] for k, v in case.offsets.items()}
    calls = {
        "posterior_draws": lambda: model.posterior_draws(64, seed=3),
        "posterior_bounds": lambda: model.posterior_bounds(
            case.X, ("parameter", parameter), n_draws=200, seed=3, offsets=case.offsets
        ),
    }
    if isinstance(model.family_, DistributionFunctionFamily):
        calls["posterior_predictive"] = lambda: model.posterior_predictive(
            rows, 16, seed=3, offsets=offsets
        )
        calls["risk_curves"] = lambda: model.risk_curves(
            case.X.iloc[0], "age", X_train=case.X, quantiles=(0.5,), n_points=5, n_draws=32
        )
        calls["portfolio"] = lambda: model.portfolio(rows, n_draws=16, seed=3, offsets=offsets)
    return calls


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_posterior_methods_warn_after_an_edit(builder, parameter):
    case = builder()
    edited = _edited_by_intercept_shift(case, parameter)
    expected = EDITED_WARNING.format(keys=f"{parameter}:age")
    for name, call in _posterior_calls(edited, case, parameter).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        edits = [w for w in caught if issubclass(w.category, EditedCoefficientsWarning)]
        assert [str(w.message) for w in edits] == [expected], name
        assert edits[0].filename == __file__, name


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_posterior_bounds_contain_the_estimate_of_an_edited_model(builder, parameter):
    case = builder()
    edited = _edited_by_intercept_shift(case, parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        bounds = edited.posterior_bounds(
            case.X, ("parameter", parameter), n_draws=400, seed=3, offsets=case.offsets
        )
    assert (bounds["lower"] <= bounds["estimate"]).all()
    assert (bounds["estimate"] <= bounds["upper"]).all()
    # The estimate is the edited curve, well outside the fit's own band.
    fit_bounds = case.model.posterior_bounds(
        case.X, ("parameter", parameter), n_draws=400, seed=3, offsets=case.offsets
    )
    assert (bounds["estimate"] > fit_bounds["upper"]).all()


def test_draws_from_one_model_are_refused_by_another_edit_of_it():
    case = lognormal_case()
    edited = _edited_by_intercept_shift(case, "scale")
    other_edit = _edited_by_intercept_shift(case, "mean")
    models = (case.model, edited, other_edit)
    rows, quantity = case.X.iloc[:5], ("parameter", "scale")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        draws = [model.posterior_draws(16, seed=3) for model in models]
        for model, own in zip(models, draws, strict=True):
            model.posterior_bounds(rows, quantity, draws=own)
        for (model, _), (_, other) in itertools.permutations(zip(models, draws, strict=True), 2):
            with pytest.raises(ValueError, match="belong to a different fit or revision"):
                model.posterior_bounds(rows, quantity, draws=other)


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_posterior_draws_move_with_the_edit_and_keep_the_fit_covariance(builder, parameter):
    case = builder()
    edited = _edited_by_intercept_shift(case, parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EditedCoefficientsWarning)
        moved = edited.posterior_draws(64, seed=3).coefficients
    original = case.model.posterior_draws(64, seed=3).coefficients
    shift = _coefficients(edited) - _coefficients(case.model)
    np.testing.assert_allclose(
        moved - original,
        np.broadcast_to(shift, moved.shape),
        rtol=0,
        atol=8 * EPS * np.abs(moved).max(),
    )


def test_simultaneous_band_of_an_edited_term_is_centred_on_the_edit():
    case = lognormal_case()
    m = case.model
    edited = m._with_revised_coefficients(
        _edit_term(m, "scale:age", 2.0), edited_terms=["scale:age"], X_train=case.X
    )
    plain = m.term_inference("scale", "age", X_train=case.X)
    with pytest.warns(EditedCoefficientsWarning):
        band = edited.term_inference("scale", "age", X_train=case.X)
    # The same draws and covariance give the same critical value; a band centred on
    # the fit's coefficients would carry the whole edit into it.
    assert band.critical_value == pytest.approx(plain.critical_value, rel=np.sqrt(EPS))
    np.testing.assert_array_equal(band.se, plain.se)


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_fitted_rows_follow_the_edit(builder, parameter):
    case = builder()
    m = case.model
    coefficients = _edit_term(m, f"{parameter}:age", 0.5)
    coefficients[_layout(m).term_slices["mean:power"]] *= 0.5
    edited = m._with_revised_coefficients(
        coefficients, edited_terms=[f"{parameter}:age", "mean:power"], X_train=case.X
    )
    original = m._model.fit_state.retained_rows
    rows = edited._model.fit_state.retained_rows
    np.testing.assert_array_equal(
        rows.fitted_parameters, edited.predict_parameters(case.X, offsets=case.offsets)
    )
    np.testing.assert_array_equal(rows.fitted_eta, _link(edited, case))
    assert not np.allclose(rows.fitted_parameters, original.fitted_parameters)
    for name in ("response", "null_eta", "null_parameters"):
        np.testing.assert_array_equal(getattr(rows, name), getattr(original, name))
    assert rows.likelihood_weights is original.likelihood_weights
    # The model the edit started from keeps its own rows.
    np.testing.assert_allclose(
        original.fitted_eta,
        _link(m, case),
        rtol=0,
        atol=8 * EPS * np.abs(original.fitted_eta).max(),
    )


def test_a_revision_builds_the_fit_state_once(monkeypatch):
    case = lognormal_case()
    m = case.model
    built = []
    validate = DistributionalFitState.__post_init__

    def counted(self):
        built.append(self)
        validate(self)

    monkeypatch.setattr(DistributionalFitState, "__post_init__", counted)
    edited = m._with_revised_coefficients(
        _edit_term(m, "scale:age", 0.5), edited_terms=["scale:age"], X_train=case.X
    )
    assert built == [edited._model.fit_state]


def _small_case(*, retain_rows: bool = True, zero_weight_rows: int = 0):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, 300)})
    y = np.exp(5.0 + 0.4 * np.sin(X["age"] / 20.0) + 0.3 * rng.standard_normal(300))
    weight = np.ones(300)
    weight[:zero_weight_rows] = 0.0
    family = LogNormalLS()
    model = SuperLSS(family, family.mean(s("age", k=6)), family.scale())
    model.fit_reml(X, y, sample_weight=weight, retain_rows=retain_rows)
    return model, X


def test_rows_dropped_by_zero_weight_are_skipped_when_the_full_frame_is_given():
    model, X = _small_case(zero_weight_rows=20)
    coefficients = _coefficients(model)
    coefficients[_layout(model).term_slices["mean:age"]] *= 0.5
    edited = model._with_revised_coefficients(coefficients, edited_terms=["mean:age"], X_train=X)
    rows = edited._model.fit_state.retained_rows
    assert rows.fitted_eta.shape[0] == len(X) - 20
    np.testing.assert_array_equal(rows.fitted_parameters, edited.predict_parameters(X.iloc[20:]))
    with pytest.raises(ValueError, match="rows"):
        model._with_revised_coefficients(
            coefficients, edited_terms=["mean:age"], X_train=X.iloc[:100]
        )


def test_a_model_without_retained_rows_needs_no_frame(tmp_path):
    model, X = _small_case(retain_rows=False)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    coefficients = _coefficients(restored)
    coefficients[_layout(restored).term_slices["mean:age"]] *= 0.5
    edited = restored._with_revised_coefficients(coefficients, edited_terms=["mean:age"])
    assert edited._hand_edited_terms == ("mean:age",)
    assert edited._model.fit_state.retained_rows is None
    assert not np.array_equal(edited.predict_link(X), restored.predict_link(X))


def test_hand_edit_survives_joblib(tmp_path):
    case = nb_case()
    m = case.model
    edited = m._with_revised_coefficients(
        _edit_term(m, "theta:age", 0.5), edited_terms=["theta:age"], X_train=case.X
    )
    path = tmp_path / "edited.joblib"
    joblib.dump(edited, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(_link(restored, case), _link(edited, case))
    assert restored._hand_edited_terms == edited._hand_edited_terms == ("theta:age",)
    np.testing.assert_array_equal(_coefficients(restored), _coefficients(edited))
    # A restored model has no training frame; the next revision names it.
    with pytest.raises(ValueError, match=re.escape(NO_FRAME)):
        restored._with_revised_coefficients(_coefficients(restored), edited_terms=["theta:age"])
    again = restored._with_revised_coefficients(
        _coefficients(restored), edited_terms=["theta:region"], X_train=case.X
    )
    assert again._hand_edited_terms == ("theta:age", "theta:region")


def test_fit_state_from_before_hand_edits_reads_as_unedited():
    case = lognormal_case()
    state = dataclasses.replace(case.model._model.fit_state)
    # An older pickle has no ``hand_edit`` entry; the class default is read.
    del state.__dict__["hand_edit"]
    assert state.hand_edit is None
    assert state.reported_coefficients is state.result.coefficients


@pytest.mark.parametrize(("builder", "parameter"), CASES)
def test_tests_and_summary_describe_the_fit_after_an_edit(builder, parameter):
    case = builder()
    m = case.model
    key = f"{parameter}:age"
    coefficients = _edit_term(m, key, 0.0)
    coefficients[_intercept(m, parameter)] += 0.3
    edited = m._with_revised_coefficients(coefficients, edited_terms=[key], X_train=case.X)
    fitted = m.term_test(parameter, "age", X_train=case.X)
    with pytest.warns(EditedCoefficientsWarning):
        tested = edited.term_test(parameter, "age", X_train=case.X)
    # A test of the zeroed curve has statistic 0, so equality with the fit's is not vacuous.
    assert fitted.statistic > 1e3 * EPS
    assert (tested.statistic, tested.rank, tested.p_value) == (
        fitted.statistic,
        fitted.rank,
        fitted.p_value,
    )
    before = m.summary(X_train=case.X)
    with pytest.warns(EditedCoefficientsWarning):
        after = edited.summary(X_train=case.X)
    tests = ["edf", "statistic", "rank", "p_value", "se", "note"]
    pd.testing.assert_frame_equal(after[tests], before[tests])
    intercept = (after["parameter"] == parameter) & (after["term"] == "(intercept)")
    np.testing.assert_array_equal(
        after.loc[intercept, "estimate"], before.loc[intercept, "estimate"] + 0.3
    )
