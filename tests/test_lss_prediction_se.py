"""Exact link-scale prediction standard errors, and the prediction design streamed in row blocks.

The reference is dense and goes through the prediction route itself: coefficient ``j``'s design
column is ``predict_eta_at(e_j) - predict_eta_at(0)`` and a row's variance is ``x' V x`` on the
dense matrix, so nothing here shares code with the design builder under test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

import superglm as sg
from superglm import SuperLSS
from superglm._frame import as_eager_frame
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional import prediction_design as design_module
from superglm.distributional.posterior import (
    link_standard_errors,
    posterior_covariance,
    posterior_draws,
    posterior_parameters,
)
from superglm.distributional.prediction_design import (
    build_joint_prediction_design,
    iter_joint_prediction_design,
)
from superglm.features import (
    Categorical,
    FactorSmooth,
    Numeric,
    OrderedCategorical,
    Spline,
    SplineCategorical,
    _basis_kernels,
)
from superglm.features._spline_cardinal import eval_cardinal_dense
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    TensorInteraction,
)
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.random_effect import RandomEffect
from tests.bound_predictor_fixtures import model_from_templates

U = np.finfo(np.float64).eps / 2
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


def _frame(n: int, seed: int, *, wide: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Covariates on [-1, 1]; ``wide`` reaches twice as far, past every fitted knot."""
    rng = np.random.default_rng(seed)
    reach = 2.0 if wide else 1.0
    frame = pd.DataFrame(
        {
            "x": rng.uniform(-reach, reach, n),
            "z": rng.normal(size=n) * reach,
            "p": rng.uniform(-reach, reach, n),
            "q": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], n),
            "h": rng.choice(["u", "v"], n),
            "band": rng.choice(LEVELS, n),
            "r": rng.choice([f"r{i}" for i in range(9)], n),
        }
    )
    mean = 0.5 * np.sin(2 * frame.x) + 0.3 * frame.z + 0.2 * (frame.g == "a") + 0.1 * frame.p**2
    return frame, (mean + np.exp(-1 + 0.3 * frame.x) * rng.standard_normal(n)).to_numpy()


def _terms() -> dict[str, tuple[dict, dict]]:
    """One location predictor per term type that streams from ``transform``, then the ones that do not."""
    return {
        "spline cat re tensor": (
            {
                "x": Spline("cr", k=6),
                "g": Categorical(base="a"),
                "r": RandomEffect(),
                "z": Spline("cr", k=5),
            },
            {"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        "polynomial": (
            {"p": Polynomial(powers=[1, 2, 3]), "g": Categorical(base="a")},
            {"p:g": PolynomialCategorical("p", "g")},
        ),
        "polynomial product": (
            {"p": Polynomial(powers=[1, 2, 3]), "q": Polynomial(powers=[1, 2])},
            {"p:q": PolynomialInteraction("p", "q")},
        ),
        "spline by level": (
            {"x": Spline("cr", k=6), "g": Categorical(base="a")},
            {"x:g": SplineCategorical("x", "g")},
        ),
        "numeric products": (
            {"z": Numeric(), "q": Numeric(), "g": Categorical(base="a")},
            {"z:g": NumericCategorical("z", "g"), "z:q": NumericInteraction("z", "q")},
        ),
        "level pairs": (
            {"g": Categorical(base="a"), "h": Categorical(base="u")},
            {"g:h": CategoricalInteraction("g", "h")},
        ),
        "ordered polynomial": (
            {"band": OrderedCategorical(order=LEVELS, basis=Polynomial(powers=[1, 2]))},
            {},
        ),
        "ordered spline": (
            {"band": OrderedCategorical(order=LEVELS, basis=Spline("cr", k=4))},
            {},
        ),
        "piecewise": (
            {
                "x": Piecewise([-0.5, 0.0, 0.5], base=0.0, lower=-1.0, upper=1.0),
                "g": Categorical(base="a"),
            },
            {},
        ),
        "sz": (
            {"x": Spline("cr", k=6)},
            {"x:g:sz": FactorSmooth("x", group="g", basis="sz", k=5)},
        ),
        "fs": (
            {"x": Spline("cr", k=6)},
            {"x:g:fs": FactorSmooth("x", group="g", basis="fs", k=5)},
        ),
    }


def _fit(case: str, *, discrete: bool) -> SuperLSS:
    features, interactions = _terms()[case]
    frame, y = _frame(500, 1)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(
            Predictor("location", features, interaction_specs=interactions),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ),
        discrete=discrete,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_reml(frame, y)


def _dense(model: SuperLSS, frame: pd.DataFrame, matrix=None):
    """Dense designs, ``x' V x`` and ``|x|' |V| |x|`` per predictor, from unit coefficient vectors."""
    inner = model._require_fitted()
    coefficients = np.asarray(inner.prediction_coefficients)
    covariance = np.asarray(model.covariance_ if matrix is None else matrix)
    base = inner.predict_eta_at(np.zeros_like(coefficients), frame)
    designs, variance, scale = [], [], []
    for index, state in enumerate(inner.layout.predictors):
        block = state.coefficient_slice
        columns = [
            inner.predict_eta_at(np.eye(len(coefficients))[j], frame)[:, index] - base[:, index]
            for j in range(block.start, block.stop)
        ]
        x, v = np.stack(columns, axis=1), covariance[block, block]
        designs.append(x)
        variance.append(np.einsum("ij,jk,ik->i", x, v, x))
        scale.append(np.einsum("ij,jk,ik->i", np.abs(x), np.abs(v), np.abs(x)))
    return designs, np.stack(variance, axis=1), np.stack(scale, axis=1)


def _assert_se_equals_dense(se: pd.DataFrame, variance, scale, width: int) -> None:
    """Two orders of one quadratic form, each within gamma_(p+2) of ``|x|'|V||x|`` (Higham 2002, 3.1)."""
    assert np.all(np.abs(se.to_numpy() ** 2 - variance) <= 4 * (width + 4) * U * scale)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("case", list(_terms()))
def test_every_term_type_streams_the_dense_design_and_exact_se(case, discrete):
    model = _fit(case, discrete=discrete)
    frame, _ = _frame(120, 2, wide=True)
    designs, variance, scale = _dense(model, frame)
    fitted = model._require_fitted()
    built = build_joint_prediction_design(frame, fitted.compiled_predictors, fitted.layout)
    for state, dense in zip(fitted.layout.predictors, designs, strict=True):
        tolerance = 64 * U * max(1.0, float(np.abs(dense).max()))
        np.testing.assert_allclose(built.local[state.name], dense, rtol=0.0, atol=tolerance)
    link, se = model.predict_link(frame, se_fit=True)
    pd.testing.assert_frame_equal(link, model.predict_link(frame))
    assert list(se.columns) == list(link.columns) and se.index.equals(frame.index)
    _assert_se_equals_dense(se, variance, scale, max(d.shape[1] for d in designs))


def _response(name: str, rng, x):
    n = len(x)
    if name == "GaussianLS":
        return 1.0 + 0.3 * x + rng.normal(0.0, 0.5, n)
    if name == "TwoPieceNormalLSS":
        sign = np.where(rng.uniform(size=n) < 0.6, -1.0, 2.0)
        return 1.0 + 0.3 * x + sign * np.abs(rng.normal(0.0, 0.5, n))
    if name == "NegativeBinomialLS":
        return rng.negative_binomial(1.5, 1.5 / (1.5 + np.exp(0.3 * x))).astype(float)
    if name == "TweedieLSS":
        counts = rng.poisson(np.exp(0.3 * x + 0.5) * 0.5)
        return np.array([rng.gamma(2.0, 1.0, count).sum() for count in counts])
    if name == "GeneralizedParetoLSS":
        return rng.pareto(4.0, n) * np.exp(0.09 * x) + 1e-3
    return rng.gamma(3.0, np.exp(0.15 * x + 1.0) / 3.0)


@pytest.mark.parametrize(
    "name",
    [
        "GaussianLS", "GammaLS", "LogNormalLS", "NegativeBinomialLS", "TweedieLSS",
        "GeneralizedGammaLSS", "GeneralizedParetoLSS", "TwoPieceNormalLSS", "TwoPieceLogNormalLSS",
    ],
)  # fmt: skip
def test_link_se_matches_the_dense_form_for_every_family_with_offsets(name):
    rng = np.random.default_rng(11)
    frame, _ = _frame(400, 3)
    y = _response(name, rng, frame["x"].to_numpy())
    family = getattr(sg, name)()
    helpers = ["mu", "phi", "p"] if name == "TweedieLSS" else [p.name for p in family.parameters]
    terms = (sg.s("x", kind="cr", k=6), sg.s("z", kind="cr", k=5), sg.cat("g"), sg.ti("x", "z"))
    declarations = [getattr(family, helpers[0])(*terms)]
    declarations += [getattr(family, helper)(sg.s("x", kind="cr", k=4)) for helper in helpers[1:2]]
    declarations += [getattr(family, helper)() for helper in helpers[2:]]
    model = SuperLSS(family, *declarations, discrete=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    test, _ = _frame(60, 4, wide=True)
    offsets = {model.parameter_names_[0]: np.linspace(-0.1, 0.1, len(test))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        link, se = model.predict_link(test, offsets=offsets, se_fit=True)
        plain, plain_se = model.predict_link(test, se_fit=True)
    designs, variance, scale = _dense(model, test)
    _assert_se_equals_dense(se, variance, scale, max(d.shape[1] for d in designs))
    first = model.parameter_names_[0]
    np.testing.assert_array_equal(se.to_numpy(), plain_se.to_numpy())
    np.testing.assert_allclose(link[first] - plain[first], offsets[first], atol=1e-12)


def test_blocks_stitch_to_the_whole_frame_design():
    model = _fit("spline cat re tensor", discrete=True)
    frame, _ = _frame(50, 5, wide=True)
    fitted = model._require_fitted()
    whole = build_joint_prediction_design(frame, fitted.compiled_predictors, fitted.layout)
    seen = []
    for rows, local in iter_joint_prediction_design(
        frame, fitted.compiled_predictors, fitted.layout, block_rows=7
    ):
        seen.append(rows)
        for name, matrix in local.items():
            tolerance = 64 * U * max(1.0, float(np.abs(whole.local[name]).max()))
            np.testing.assert_allclose(matrix, whole.local[name][rows], rtol=0.0, atol=tolerance)
    assert [(r.start, r.stop) for r in seen] == [(lo, min(lo + 7, 50)) for lo in range(0, 50, 7)]
    (rows, _), *_ = iter_joint_prediction_design(
        frame, fitted.compiled_predictors, fitted.layout, start=20, stop=31, block_rows=64
    )
    assert (rows.start, rows.stop) == (20, 31)


def test_a_frame_is_priced_in_blocks_and_each_term_is_read_once_per_block(monkeypatch):
    """Work counts: no matrix with a row per observation, no column rebuilt per coefficient.

    Mutation check: with ``_TRANSFORM_TERMS`` and ``_TRANSFORM_PAIRS`` emptied every term is read
    one column at a time through ``score`` (the route before this change): ``column`` becomes the
    coefficient count, ``transform`` zero, and the assertion fails.
    """
    model = _fit("spline cat re tensor", discrete=True)
    fitted = model._require_fitted()
    frame, _ = _frame(1000, 6)
    calls = {"transform": 0, "column": 0, "rows": []}
    monkeypatch.setattr(design_module, "DESIGN_BLOCK_ROWS", 256)
    for attribute, key in (("_term_rows", "transform"), ("_feature_design", "column")):
        original = getattr(design_module, attribute)

        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(design_module, attribute, counted)
    original_rows = design_module._predictor_rows

    def sized(predictor, state, columns, rows, named):
        calls["rows"].append(rows.stop - rows.start)
        return original_rows(predictor, state, columns, rows, named)

    monkeypatch.setattr(design_module, "_predictor_rows", sized)
    link_standard_errors(fitted, frame)
    n_terms = sum(
        len(p.compiled.feature_order) + len(p.compiled.interaction_order)
        for p in fitted.compiled_predictors
    )
    assert max(calls["rows"]) == 256 and sum(calls["rows"]) == 2 * len(frame)
    coded = len(design_module.label_columns(as_eager_frame(frame), fitted.compiled_predictors))
    assert coded == 2  # the categorical `g` and the random effect `r` are read from codes
    assert (calls["transform"], calls["column"]) == (4 * (n_terms - coded), 0)


def test_terms_read_only_by_score_still_give_the_dense_se():
    """A level declared with no weight sits at the population curve, which is not ``transform @ beta``."""
    from tests.test_factor_smooth_sz_thin_and_influence import (
        _superlss_with_a_declared_empty_sz_level,
    )

    model, frame, y, weight, lambdas = _superlss_with_a_declared_empty_sz_level()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y, sample_weight=weight, lambdas=lambdas)
    fitted = model._require_fitted()
    assert fitted.compiled_predictors[0].compiled.interaction_specs["x:g:sz"]._has_population_offset
    probe = pd.DataFrame({"x": np.linspace(-1.5, 1.5, 40), "g": np.tile(["a", "b", "c", "d"], 10)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, se = model.predict_link(probe, se_fit=True)
        _, variance, scale = _dense(model, probe)
    _assert_se_equals_dense(se, variance, scale, len(model.coef_))
    assert np.all(se.to_numpy() > 0.0)


def _unseen_model() -> tuple[SuperLSS, pd.DataFrame]:
    frame, y = _frame(400, 7)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(
            Predictor(
                "location", {"x": Spline("cr", k=5), "g": Categorical(base="a", unseen="base")}
            ),
            Predictor("scale", {}),
        ),
    )
    model.fit_reml(frame, y)
    test, _ = _frame(300, 8)
    test.loc[test.index[:40], "g"] = "zz"
    return model, test


@pytest.mark.parametrize(
    "price",
    [
        lambda m, f: m.predict_link(f, se_fit=True),
        lambda m, f: m.posterior_bounds(f, ("parameter", "location"), n_draws=4),
        lambda m, f: m.posterior_predictive(f, n_draws=3),
    ],
    ids=["se", "bounds", "predictive"],
)
def test_unseen_levels_are_reported_once_with_the_frames_count(price, monkeypatch):
    """Each block would count only its own rows; the whole-frame prediction says it once."""
    model, test = _unseen_model()
    monkeypatch.setattr(design_module, "DESIGN_BLOCK_ROWS", 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price(model, test)
    routed = [w for w in caught if "unseen at fit" in str(w.message)]
    assert len(routed) == 1 and "over 40 row(s)" in str(routed[0].message)


def test_an_unseen_level_has_the_base_levels_se():
    model, test = _unseen_model()
    base = test.copy()
    base.loc[base["g"] == "zz", "g"] = "a"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, se = model.predict_link(test, se_fit=True)
        _, base_se = model.predict_link(base, se_fit=True)
    np.testing.assert_array_equal(se.to_numpy(), base_se.to_numpy())


def test_the_streamed_draw_pushforward_equals_the_dense_one_and_its_sd_the_exact_se():
    model = _fit("spline cat re tensor", discrete=True)
    fitted = model._require_fitted()
    frame, _ = _frame(90, 9, wide=True)
    draws = posterior_draws(fitted, 4000, seed=3)
    designs, _, _ = _dense(model, frame)
    parts = [
        (rows, block) for rows, block in posterior_parameters(fitted, frame, draws, chunk_rows=35)
    ]
    streamed = np.concatenate([block for _, block in parts], axis=1)
    for index, (state, dense) in enumerate(zip(fitted.layout.predictors, designs, strict=True)):
        eta = draws.coefficients[:, state.coefficient_slice] @ dense.T
        theta = np.asarray(state.link.inverse(eta))
        np.testing.assert_allclose(streamed[:, :, index], theta, rtol=1e-11, atol=1e-12)
    se = link_standard_errors(fitted, frame)
    sd = np.std(
        draws.coefficients[:, fitted.layout.predictors[0].coefficient_slice] @ designs[0].T, axis=0
    )
    # 4,000 draws estimate a standard deviation to about 1.1%; five times that is the bound.
    np.testing.assert_allclose(sd, se[:, 0], rtol=0.06)


def test_the_corrected_covariance_gives_the_dense_form_and_never_less_variance():
    model = _fit("spline cat re tensor", discrete=True)
    fitted = model._require_fitted()
    frame, _ = _frame(60, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        corrected = posterior_covariance(fitted, kind="corrected")
        _, se_corrected = model.predict_link(frame, se_fit=True, covariance="corrected")
    _, variance, scale = _dense(model, frame, matrix=corrected)
    _assert_se_equals_dense(se_corrected, variance, scale, corrected.shape[0])
    _, se = model.predict_link(frame, se_fit=True)
    assert np.all(se_corrected.to_numpy() >= se.to_numpy() * (1 - 64 * U))


def test_an_empty_frame_and_a_missing_column_behave_as_predict_link_does():
    model = _fit("spline cat re tensor", discrete=True)
    frame, _ = _frame(10, 12)
    link, se = model.predict_link(frame.iloc[:0], se_fit=True)
    assert link.shape == se.shape == (0, 2)
    with pytest.raises(ValueError, match="missing required columns"):
        model.predict_link(frame.drop(columns="z"), se_fit=True)
    polars = pytest.importorskip("polars")
    _, from_polars = model.predict_link(polars.from_pandas(frame), se_fit=True)
    np.testing.assert_array_equal(from_polars.to_numpy(), model.predict_link(frame, se_fit=True)[1])


class _FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    ("labels", "missing"),
    [
        (["a", "b", "a"], False),
        (["a", None, "a"], True),
        (["a", float("nan"), "b"], True),
        (["a", np.float64("nan")], True),
        (["a", _FloatSubclass("nan")], True),
        (["a", np.float32("nan"), pd.NA], False),  # not the scan's definition of missing
        ([1.0, 2.0, 1.0], False),
        ([1.0, float("nan")], True),
    ],
)
def test_the_missing_label_scan_over_distinct_values_is_the_per_element_scan(labels, missing):
    from superglm.features.categorical import _validate_missing_only

    column = np.array(labels * 50, dtype=object)
    assert missing == any(v is None or (isinstance(v, float) and np.isnan(v)) for v in column)
    if missing:
        with pytest.raises(ValueError, match="missing values"):
            _validate_missing_only(column)
    else:
        _validate_missing_only(column)


def _on_the_pool(monkeypatch, rows_per_block: int = 40):
    """Make a 300-row frame a book: pooled scoring from one row, blocks of ``rows_per_block``."""
    from superglm.distributional import model as model_module

    monkeypatch.setattr(model_module, "POOLED_PREDICTION_ROWS", 1)
    monkeypatch.setattr(model_module, "DESIGN_BLOCK_ROWS", rows_per_block)
    monkeypatch.setattr(design_module, "DESIGN_BLOCK_ROWS", rows_per_block)


def test_pooled_scoring_gives_the_serial_numbers_at_any_worker_count(monkeypatch):
    from superglm._parallel import parallel_config

    model = _fit("spline cat re tensor", discrete=True)
    frame, _ = _frame(300, 13, wide=True)
    serial_link, serial_se = model.predict_link(frame, se_fit=True)
    _on_the_pool(monkeypatch)
    results = {}
    for workers in (1, 4):
        with parallel_config(n_jobs=workers):
            results[workers] = model.predict_link(frame, se_fit=True)
    one, four = results[1], results[4]
    np.testing.assert_array_equal(one[0].to_numpy(), four[0].to_numpy())
    np.testing.assert_array_equal(one[1].to_numpy(), four[1].to_numpy())
    np.testing.assert_allclose(one[0].to_numpy(), serial_link.to_numpy(), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(one[1].to_numpy(), serial_se.to_numpy(), rtol=1e-12)


def test_a_single_block_call_takes_its_products_on_one_blas_thread_from_a_few_dozen_rows(
    monkeypatch,
):
    """Work counts: from ``BLAS_ONE_THREAD_ROWS`` rows a one-block call enters the one-thread scope.

    A block's ``x V`` product is large enough from there for BLAS to start its threads and too
    small to gain from them: at 1,000 rows and 16 BLAS threads the median call took 12.9 ms against
    13.1 at one thread, and the 99th percentile 69 ms against 20. A quote's products are too small
    to thread, so a quote does not pay for the scope. Mutation checks: with the scope taken out of
    ``map_design_blocks``'s single-block branch the 1,000-row call never enters it and the first
    assertion fails; with the size test taken out the quote enters it and the second fails.
    """
    import contextlib

    import superglm._blas_threads as blas

    entered = []
    original = blas.pooled_blas_threads

    @contextlib.contextmanager
    def counted():
        entered.append(1)
        with original():
            yield

    monkeypatch.setattr(blas, "pooled_blas_threads", counted)
    model = _fit("spline cat re tensor", discrete=True)
    rows = design_module.BLAS_ONE_THREAD_ROWS
    for n in (rows, 1000):
        entered.clear()
        link, se = model.predict_link(_frame(n, 30)[0], se_fit=True)
        assert len(entered) == 1 and np.all(np.isfinite(se.to_numpy()))
    entered.clear()
    model.predict_link(_frame(rows - 1, 31)[0], se_fit=True)
    model.predict_link(_frame(1, 32)[0], se_fit=True)
    assert entered == []


def test_a_quote_starts_no_pool_and_a_book_does(monkeypatch):
    import superglm._parallel as parallel

    model = _fit("spline cat re tensor", discrete=True)
    started = []
    original = parallel.pooled_for
    monkeypatch.setattr(
        parallel, "pooled_for", lambda *a, **k: (started.append(len(a[1])), original(*a, **k))[1]
    )
    quote, _ = _frame(1, 14)
    thousand, _ = _frame(1000, 15)
    model.predict_link(quote, se_fit=True)
    model.predict_link(thousand, se_fit=True)
    model.posterior_bounds(thousand, ("parameter", "location"), n_draws=4)
    assert started == []
    _on_the_pool(monkeypatch)
    model.predict_link(thousand, se_fit=True)
    assert len(started) == 2  # the scoring tasks, then the design blocks


def test_a_pooled_draw_pushforward_and_unseen_warning_match_the_inline_ones(monkeypatch):
    model, test = _unseen_model()
    inline = model.posterior_bounds(test, ("parameter", "location"), n_draws=6, seed=1)
    _on_the_pool(monkeypatch, 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pooled = model.posterior_bounds(test, ("parameter", "location"), n_draws=6, seed=1)
    np.testing.assert_allclose(pooled.to_numpy(), inline.to_numpy(), rtol=1e-12, atol=1e-13)
    assert len([w for w in caught if "unseen at fit" in str(w.message)]) == 1


@pytest.mark.parametrize("n", [3, 64, 65, 400])
@pytest.mark.parametrize("levels", [["a", "b", "c", "d"], [3, 1, 2], ["x", 1, 2.5]])
def test_level_codes_are_the_same_whether_a_dict_or_an_index_reads_them(n, levels):
    """A quote's rows are looked up in a dict, a book's in a pandas Index: one answer."""
    from superglm.features.categorical import _codes_against

    rng = np.random.default_rng(n)
    pool = [*levels, "unseen", 99, None, pd.NA]
    values = np.array([pool[i] for i in rng.integers(0, len(pool), n)], dtype=object)
    expected = np.asarray(pd.Index(levels).get_indexer(values), dtype=np.intp)
    np.testing.assert_array_equal(_codes_against(values, levels), expected)


def test_level_columns_are_coded_once_per_frame_and_a_quiet_entry_never_silences_a_warning():
    model, test = _unseen_model()
    frame = as_eager_frame(test)
    predictors = model._require_fitted().compiled_predictors
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = design_module.label_columns(frame, predictors, quiet=True)
        again = design_module.label_columns(frame, predictors, quiet=True)
        assert not caught and all(first[key] is again[key] for key in first)
        design_module.label_columns(
            frame, predictors
        )  # a caller that is to warn is not served quiet
        design_module.label_columns(frame, predictors)  # and what it coded is read back
    assert len([w for w in caught if "unseen at fit" in str(w.message)]) == 1


def _assert_rows_agree_to_rounding(rows, reference, degree):
    """Each entry within a few ``u`` of its row's largest entry, which grows past the knots.

    SciPy's own build may fuse a multiply and an add where the platform's compiler does, so
    against SciPy the rows agree to rounding and their bits differ by platform.
    """
    scale = np.max(np.abs(reference), axis=1, keepdims=True)
    assert np.all(np.abs(rows - reference) <= 8 * (degree + 1) * U * scale)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("repeated_interior", [False, True])
def test_the_bspline_kernel_matches_scipys_design_inside_and_outside_the_knots(
    degree, repeated_interior
):
    """A row agrees with ``BSpline.design_matrix``'s to rounding, so a saved model predicts as it did.

    Mutation check: clamping ``x`` to the knots' range, which stops a row extrapolating, fails this.
    """
    rng = np.random.default_rng(degree)
    interior = np.sort(rng.uniform(0.0, 1.0, 7))
    if repeated_interior:
        interior[3] = interior[2]
    knots = np.r_[[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    x = np.r_[rng.uniform(-0.5, 1.5, 300), knots, 0.0, 1.0]
    reference = BSpline.design_matrix(x, knots, degree, extrapolate=True).toarray()
    _assert_rows_agree_to_rounding(
        _basis_kernels.bspline_dense(x, knots, degree), reference, degree
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("repeated_interior", [False, True])
def test_the_python_bspline_rows_are_the_kernels_rows_and_scipys_to_rounding(
    degree, repeated_interior
):
    """A quote's rows are the kernel's bits (both run the same unfused operations in one order).

    Mutation check: dividing through a reciprocal, ``step = p * (1 / (xb - xa))``, changes a
    share of the rows by an ulp from the kernel's and fails the first assertion.
    """
    rng = np.random.default_rng(degree + 10)
    interior = np.sort(rng.uniform(0.0, 1.0, 7))
    if repeated_interior:
        interior[3] = interior[2]
    knots = np.r_[[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    x = np.r_[rng.uniform(-0.5, 1.5, 100), knots, 0.0, 1.0]
    rows = _basis_kernels.bspline_dense_python(x, knots, degree)
    np.testing.assert_array_equal(rows, _basis_kernels.bspline_dense(x, knots, degree))
    reference = BSpline.design_matrix(x, knots, degree, extrapolate=True).toarray()
    _assert_rows_agree_to_rounding(rows, reference, degree)


def test_the_python_cardinal_rows_are_the_kernels_rows_and_the_numpy_evaluation_to_rounding():
    """Mutation check: writing ``t`` for ``a`` in the first ``+=`` fails both assertions."""
    rng = np.random.default_rng(15)
    knots = np.sort(rng.uniform(0.0, 1.0, 9))
    second_derivative_map = rng.standard_normal((9, 9))
    x = np.r_[rng.uniform(-0.2, 1.2, 100), knots]
    rows = _basis_kernels.cardinal_dense_python(x, knots, second_derivative_map)
    np.testing.assert_array_equal(
        rows, _basis_kernels.cardinal_dense(x, knots, second_derivative_map)
    )
    reference = eval_cardinal_dense(x, knots, second_derivative_map)
    np.testing.assert_allclose(rows, reference, rtol=0.0, atol=16 * U * np.abs(reference).max())


def test_a_quote_compiles_no_basis_kernel_and_a_full_block_reads_them(monkeypatch):
    """Work counts: a block reaches a kernel only from ``KERNEL_MIN_ROWS`` rows, so no compile or cache load.

    Up to ``SCALAR_MAX_ROWS`` rows the Python rows are read, between the two the library's
    evaluation. The rows are the same from any of them, to rounding (``basis @ R`` differs in its
    last bit with the block's row count). Mutation checks: with ``KERNEL_MIN_ROWS`` zero, or the
    size test taken out of ``_evaluations``, a quote reaches the kernels and the first assertion
    fails; with ``KERNEL_MIN_ROWS`` past every frame the full block never does and the last
    fails; with ``SCALAR_MAX_ROWS`` zero a quote never reads the Python rows.
    """
    import sys

    kernels = _basis_kernels
    assert kernels.SCALAR_MAX_ROWS < kernels.KERNEL_MIN_ROWS <= design_module.DESIGN_BLOCK_ROWS
    model = _fit("spline cat re tensor", discrete=True)
    reached: dict[str, list[str]] = {"kernel": [], "python": []}
    for name, route in (
        ("bspline_dense", "kernel"),
        ("cardinal_dense", "kernel"),
        ("bspline_dense_python", "python"),
        ("cardinal_dense_python", "python"),
    ):
        original = getattr(kernels, name)

        def spy(*args, _original=original, _route=route, **kwargs):
            reached[_route].append(sys._getframe(1).f_code.co_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(kernels, name, spy)
    block, _ = _frame(kernels.KERNEL_MIN_ROWS, 21, wide=True)

    def routes(rows: int) -> tuple[bool, bool]:
        for read in reached.values():
            read.clear()
        model.predict_link(block.iloc[:rows], se_fit=True)
        return bool(reached["python"]), bool(reached["kernel"])

    assert routes(1) == (True, False) and routes(kernels.SCALAR_MAX_ROWS) == (True, False)
    for rows in (kernels.SCALAR_MAX_ROWS + 1, 1000, kernels.KERNEL_MIN_ROWS - 1):
        assert routes(rows) == (False, False)
    assert routes(kernels.KERNEL_MIN_ROWS) == (False, True)
    assert set(reached["kernel"]) == {"dense_basis", "dense_raw_basis"}
    whole = model.predict_link(block).to_numpy()
    tolerance = 64 * U * float(np.abs(whole).max())
    for rows in (1, kernels.SCALAR_MAX_ROWS, 1000):
        part = model.predict_link(block.iloc[:rows]).to_numpy()
        np.testing.assert_allclose(part, whole[:rows], rtol=0.0, atol=tolerance)


def test_the_cardinal_kernel_is_the_numpy_evaluation_to_rounding():
    rng = np.random.default_rng(5)
    knots = np.sort(rng.uniform(0.0, 1.0, 9))
    second_derivative_map = rng.standard_normal((9, 9))
    x = np.r_[rng.uniform(-0.2, 1.2, 300), knots]
    reference = eval_cardinal_dense(x, knots, second_derivative_map)
    scale = np.abs(reference).max()
    np.testing.assert_allclose(
        _basis_kernels.cardinal_dense(x, knots, second_derivative_map),
        reference,
        rtol=0.0,
        atol=16 * U * scale,
    )
