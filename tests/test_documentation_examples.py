"""Executable contracts for published Python examples and documentation links."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold

import superglm
import superglm.model_selection
import superglm.validation
from superglm.editor import EditorSession
from superglm.export.rating_tables import RatingTableBlock
from superglm.features.spline import CardinalCRSpline
from superglm.inference._term_types import SplineMetadata, TermInference
from superglm.inference.factor_smooths import FactorSmoothResult
from superglm.inference.metrics import ModelMetrics
from superglm.inference.random_effects import RandomEffectResult
from superglm.model.fit_ops import PathResult
from superglm.model_selection import CrossValidationResult
from superglm.profiling.nb import NBProfileResult
from superglm.profiling.tweedie import TweedieProfileResult
from superglm.validation import DoubleLiftChartResult, LorenzCurveResult

_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_FENCE = re.compile(r"^```python\s*$\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
_NATIVE_FIT_OR_PROFILE_METHODS = {
    "estimate_p",
    "estimate_theta",
    "fit",
    "fit_path",
    "fit_reml",
}


@dataclass(frozen=True)
class _PythonBlock:
    path: Path
    index: int
    line: int
    source: str

    @property
    def filename(self) -> str:
        return f"{self.path.relative_to(_ROOT)}#python-{self.index}-line-{self.line}"


def _python_blocks(path: Path) -> list[str]:
    return _PYTHON_FENCE.findall(path.read_text(encoding="utf-8"))


def _published_python_blocks() -> list[_PythonBlock]:
    paths = [_ROOT / "README.md", *sorted((_ROOT / "docs/guide").glob("*.md"))]
    published: list[_PythonBlock] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(_PYTHON_FENCE.finditer(text), start=1):
            published.append(
                _PythonBlock(
                    path=path,
                    index=index,
                    line=text.count("\n", 0, match.start()) + 1,
                    source=match.group(1),
                )
            )
    return published


def _python_block_after_heading(path: Path, heading: str) -> str:
    source = path.read_text(encoding="utf-8")
    section = source.split(heading, maxsplit=1)[1]
    blocks = _PYTHON_FENCE.findall(section)
    assert blocks, f"{path} has no Python block after {heading!r}"
    return blocks[0]


def _method_double(monkeypatch, owner, name: str, *, side_effect=None, return_value=None):
    """Replace one callable while retaining its current public signature."""
    original = getattr(owner, name)
    replacement = create_autospec(original, side_effect=side_effect)
    if side_effect is None:
        replacement.return_value = return_value
    monkeypatch.setattr(owner, name, replacement)
    return replacement


def _execution_results() -> dict[str, object]:
    x = np.linspace(0.0, 1.0, 4)
    table = pd.DataFrame(
        {
            "level": ["A"],
            "exposure": [1.0],
            "fit_weight": [1.0],
            "unpooled_effect": [0.0],
            "effect": [0.0],
            "posterior_se": [0.1],
            "effective_df": [1.0],
            "credibility": [0.5],
            "shrinkage": [0.5],
            "sufficient_support": [True],
        }
    )
    curves = pd.DataFrame(
        {
            "level": ["A"] * len(x),
            "DrivAge": x,
            "effect": x,
            "posterior_se": np.full(len(x), 0.1),
            "lower": x - 0.2,
            "upper": x + 0.2,
        }
    )
    spline = SplineMetadata(
        kind="PSpline",
        knot_strategy="quantile",
        interior_knots=x[1:-1],
        boundary=(0.0, 1.0),
        n_basis=4,
        degree=3,
        extrapolation="clip",
    )
    term = TermInference(
        name="DrivAge",
        kind="spline",
        active=True,
        x=x,
        log_relativity=x,
        relativity=np.exp(x),
        ci_lower=np.exp(x - 0.1),
        ci_upper=np.exp(x + 0.1),
        edf=2.0,
        spline=spline,
    )
    random_effect = RandomEffectResult(
        name="VehBrand",
        lambda_value=1.0,
        phi=1.0,
        tau_squared=1.0,
        standard_deviation=1.0,
        effective_df=1.0,
        collapsed=False,
        at_lower_boundary=False,
        at_upper_boundary=False,
        table=table,
        diagnostics={},
    )
    factor_smooth = FactorSmoothResult(
        name="DrivAge:Region:fs",
        variable="DrivAge",
        grouping_variable="Region",
        basis="fs",
        lambdas={"wiggle": 1.0},
        phi=1.0,
        variance_components={"wiggle": 1.0},
        effective_df=1.0,
        collapsed=False,
        at_lower_boundary={"wiggle": False},
        at_upper_boundary={"wiggle": False},
        table=table,
        curves=curves,
        diagnostics={},
    )
    path = PathResult(
        lambda_seq=np.array([1.0]),
        coef_path=np.zeros((1, 1)),
        intercept_path=np.zeros(1),
        deviance_path=np.zeros(1),
        n_iter_path=np.ones(1, dtype=np.int64),
        converged_path=np.ones(1, dtype=np.bool_),
    )
    nb_profile = NBProfileResult(
        theta_hat=1.0,
        nll=1.0,
        n_evaluations=1,
        converged=True,
    )
    tweedie_profile = TweedieProfileResult(
        p_hat=1.5,
        phi_hat=1.0,
        nll=1.0,
        n_evaluations=1,
        converged=True,
        method="grid",
        phi_method="mle",
        search_trace=pd.DataFrame(),
    )
    cross_validation = CrossValidationResult(
        fold_scores=pd.DataFrame(),
        mean_scores={},
        pooled_scores={},
        std_scores={},
    )
    rating_block = RatingTableBlock(name="Term", kind="offset", table=pd.DataFrame())
    rating_payload = type(
        "_RatingPayload",
        (),
        {"main_effects": [rating_block]},
    )()
    plot_payload = {
        "terms": [{"effect": pd.DataFrame(), "density": pd.DataFrame(), "knots": pd.DataFrame()}],
        "effect": pd.DataFrame(),
        "density": pd.DataFrame(),
    }
    lorenz = LorenzCurveResult(
        curve=pd.DataFrame(),
        gini_model=0.25,
        gini_perfect=0.5,
        gini_ratio=0.5,
        figure=None,
    )
    double_lift = DoubleLiftChartResult(bins=pd.DataFrame(), figure=None)
    return {
        "cross_validation": cross_validation,
        "double_lift": double_lift,
        "factor_smooth": factor_smooth,
        "lorenz": lorenz,
        "nb_profile": nb_profile,
        "path": path,
        "plot_payload": plot_payload,
        "random_effect": random_effect,
        "rating_payload": rating_payload,
        "term": term,
        "tweedie_profile": tweedie_profile,
    }


def _install_execution_doubles(monkeypatch) -> dict[str, object]:
    """Keep documentation execution fast and confined while checking call signatures."""
    results = _execution_results()

    def return_model(model, *_args, **_kwargs):
        return model

    def predict(_model, X, *_args, **_kwargs):
        return np.ones(len(X), dtype=np.float64)

    def return_fitted_estimator(estimator, *_args, **_kwargs):
        estimator.documentation_fitted_ = True
        return estimator

    metrics = create_autospec(ModelMetrics, instance=True, spec_set=True)
    metrics.summary.return_value = "model metrics"
    editor_session = create_autospec(EditorSession, instance=True, spec_set=True)
    editor_session.widget.return_value = object()

    for name in ("fit", "fit_reml", "apply_shape_postfit"):
        _method_double(monkeypatch, superglm.SuperGLM, name, side_effect=return_model)
    _method_double(monkeypatch, superglm.SuperGLM, "predict", side_effect=predict)
    _method_double(monkeypatch, superglm.SuperGLM, "summary", return_value="model summary")
    _method_double(monkeypatch, superglm.SuperGLM, "metrics", return_value=metrics)
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "term_inference",
        return_value=results["term"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "random_effects",
        return_value=results["random_effect"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "factor_smooth",
        return_value=results["factor_smooth"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "fit_path",
        return_value=results["path"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "estimate_theta",
        return_value=results["nb_profile"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "estimate_p",
        return_value=results["tweedie_profile"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "rating_table_payload",
        return_value=results["rating_payload"],
    )
    for name in ("export_rating_tables", "plot", "relativities", "plot_diagnostics"):
        _method_double(monkeypatch, superglm.SuperGLM, name, return_value={})
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "plot_data",
        return_value=results["plot_payload"],
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLM,
        "screen_interactions",
        return_value=pd.DataFrame(),
    )

    _method_double(
        monkeypatch,
        superglm.SuperGLMRegressor,
        "fit",
        side_effect=return_fitted_estimator,
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLMRegressor,
        "predict",
        side_effect=predict,
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLMClassifier,
        "fit",
        side_effect=return_fitted_estimator,
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLMClassifier,
        "predict",
        side_effect=predict,
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLMClassifier,
        "predict_proba",
        side_effect=lambda _model, X: np.full((len(X), 2), 0.5),
    )
    _method_double(
        monkeypatch,
        superglm.SuperGLMClassifier,
        "decision_function",
        side_effect=predict,
    )
    _method_double(monkeypatch, NBProfileResult, "ci", return_value=(0.5, 2.0))
    _method_double(monkeypatch, NBProfileResult, "profile_plot", return_value=object())
    _method_double(monkeypatch, TweedieProfileResult, "ci", return_value=(1.2, 1.8))
    _method_double(monkeypatch, TweedieProfileResult, "trace_plot", return_value=object())
    _method_double(monkeypatch, TweedieProfileResult, "profile_plot", return_value=object())
    _method_double(
        monkeypatch,
        EditorSession,
        "from_model",
        return_value=editor_session,
    )

    cross_validate = create_autospec(
        superglm.model_selection.cross_validate,
        return_value=results["cross_validation"],
    )
    monkeypatch.setattr(superglm.model_selection, "cross_validate", cross_validate)
    monkeypatch.setattr(superglm, "cross_validate", cross_validate)
    lorenz_curve = create_autospec(
        superglm.validation.lorenz_curve,
        return_value=results["lorenz"],
    )
    double_lift_chart = create_autospec(
        superglm.validation.double_lift_chart,
        return_value=results["double_lift"],
    )
    monkeypatch.setattr(superglm.validation, "lorenz_curve", lorenz_curve)
    monkeypatch.setattr(superglm.validation, "double_lift_chart", double_lift_chart)
    return results


def _documentation_namespace(results: dict[str, object]) -> dict[str, object]:
    """Supply the data and prior-result context assumed by narrative snippets."""
    n_rows = 20
    index = np.arange(n_rows)
    exposure = np.linspace(0.5, 2.0, n_rows)
    frame = pd.DataFrame(
        {
            "age": np.linspace(18.0, 80.0, n_rows),
            "density": np.linspace(1.0, 10.0, n_rows),
            "region": np.where(index % 2, "B", "A"),
            "log_exposure": np.log(exposure),
            "DrivAge": np.linspace(18.0, 80.0, n_rows),
            "VehAge": index % 12,
            "BonusMalus": 50 + index,
            "Area": np.take(["A", "E", "F", "B"], index % 4),
            "BonusClass": np.take(["A", "B", "C", "D"], index % 4),
            "LogDensity": np.log1p(index),
            "Density": index + 1.0,
            "VehBrand": np.take(["V1", "V2"], index % 2),
            "Region": np.take(["R11", "R24"], index % 2),
            "Exposure": exposure,
            "ClaimNb": index % 3,
            "x": np.linspace(0.0, 1.0, n_rows),
            "x1": np.linspace(0.0, 1.0, n_rows),
            "x2": np.linspace(1.0, 2.0, n_rows),
        }
    )
    y = 0.2 + index / n_rows
    features = {"age": superglm.Numeric()}
    model = superglm.SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features=features,
    )
    namespace = {
        name: getattr(superglm, name) for name in dir(superglm) if not name.startswith("_")
    }
    namespace.update(
        {
            "X": frame,
            "X_train": frame,
            "X_validation": frame,
            "claim_count": y,
            "claim_counts": y,
            "claim_rate": y / exposure,
            "column_transformer": ColumnTransformer(
                [("age", "passthrough", ["age"])],
            ),
            "df": frame.copy(),
            "exposure": exposure,
            "exposure_holdout": exposure,
            "exposure_train": exposure,
            "features": features,
            "feats": features,
            "holdout_df": frame,
            "log_exposure": np.log(exposure),
            "model": model,
            "mu_baseline": np.ones(n_rows),
            "mu_holdout": np.ones(n_rows),
            "mu_new": np.ones(n_rows),
            "np": np,
            "offset": np.log(exposure),
            "pd": pd,
            "result": results["tweedie_profile"],
            "score_df": frame,
            "score_exposure": exposure,
            "test": frame,
            "train": frame,
            "train_df": frame,
            "validation_weight": exposure,
            "y": y,
            "y_holdout": y,
            "y_obs": y,
            "y_pred": np.ones(n_rows),
            "y_train": y,
            "y_validation": y,
        }
    )
    return namespace


def test_all_published_guide_python_blocks_execute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    published = _published_python_blocks()
    assert len(published) >= 80
    results = _install_execution_doubles(monkeypatch)
    exercised: list[str] = []

    for block in published:
        block_directory = tmp_path / f"block-{len(exercised) + 1}"
        block_directory.mkdir()
        monkeypatch.chdir(block_directory)
        namespace = _documentation_namespace(results)
        exec(compile(block.source, block.filename, "exec"), namespace)
        exercised.append(block.filename)

    assert exercised == [block.filename for block in published]


def test_native_fit_and_profile_examples_configure_features_explicitly() -> None:
    checked: list[str] = []
    family_methods: set[str] = set()
    offenders: list[str] = []

    for block in _published_python_blocks():
        tree = ast.parse(block.source, filename=block.filename)
        method_names = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _NATIVE_FIT_OR_PROFILE_METHODS
        }
        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SuperGLM"
        ]
        is_family_example = block.path.name == "families.md"
        if not constructors or (not method_names and not is_family_example):
            continue

        checked.append(block.filename)
        if block.path.name == "families.md":
            family_methods.update(method_names)
        for constructor in constructors:
            feature_keywords = [
                keyword
                for keyword in constructor.keywords
                if keyword.arg == "features"
                and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
            ]
            if not feature_keywords:
                offenders.append(block.filename)

    assert len(checked) >= 20
    assert {"estimate_p", "estimate_theta", "fit"} <= family_methods
    assert not offenders, f"native fit/profile examples omit features=: {offenders}"


def test_families_binomial_example_executes_real_fit_with_features() -> None:
    block = _python_block_after_heading(
        _ROOT / "docs/guide/families.md",
        "## Binomial (binary classification)",
    )
    frame = pd.DataFrame({"age": [-1.0, -1.0, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0]})
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    namespace = {
        "df": frame,
        "features": {"age": superglm.Numeric()},
        "y": y,
    }

    exec(compile(block, "docs/guide/families.md#binomial", "exec"), namespace)

    model = namespace["model"]
    probabilities = namespace["probabilities"]
    assert isinstance(model, superglm.SuperGLM)
    assert model.result is not None
    assert model._feature_order == ["age"]
    np.testing.assert_allclose(probabilities, np.full(len(frame), 0.5), atol=1e-8)


def test_readme_lorenz_example_keeps_result_and_gini_ratio(
    monkeypatch,
) -> None:
    from matplotlib import pyplot as plt

    cross_validate = create_autospec(superglm.cross_validate, return_value=object())
    double_lift_chart = create_autospec(
        superglm.validation.double_lift_chart,
        return_value=DoubleLiftChartResult(bins=pd.DataFrame(), figure=None),
    )
    monkeypatch.setattr(superglm, "cross_validate", cross_validate)
    monkeypatch.setattr(superglm.validation, "double_lift_chart", double_lift_chart)
    block = _python_block_after_heading(
        _ROOT / "README.md",
        "## Validation And Model Comparison",
    )
    y_holdout = np.array([0.0, 1.0, 3.0, 0.0, 2.0])
    namespace = {
        "exposure_holdout": np.ones(len(y_holdout)),
        "exposure_train": np.ones(10),
        "model": object(),
        "mu_baseline": np.ones(len(y_holdout)),
        "mu_holdout": np.array([0.2, 0.8, 2.5, 0.3, 1.7]),
        "train_df": object(),
        "y_holdout": y_holdout,
        "y_train": np.arange(10, dtype=np.float64),
    }

    exec(compile(block, "README.md#validation", "exec"), namespace)

    lorenz = namespace["lorenz"]
    assert "gini" not in namespace
    assert isinstance(lorenz, LorenzCurveResult)
    assert np.isfinite(lorenz.gini_ratio)
    plt.close(lorenz.figure)


def test_corrected_shape_api_docstrings_remain_current() -> None:
    cardinal_doc = inspect.getdoc(CardinalCRSpline)
    monotonize_doc = inspect.getdoc(superglm.SuperGLM.monotonize)

    assert cardinal_doc is not None
    assert "Constraint.postfit.increasing" in cardinal_doc
    assert "Constraint.postfit.convex" in cardinal_doc
    assert "Fit-time shape constraints are not implemented" in cardinal_doc
    assert "monotone_mode" not in cardinal_doc

    assert monotonize_doc is not None
    assert "Constraint.postfit.*" in monotonize_doc
    assert "weighted shape projection" in monotonize_doc
    assert "isotonic regression" not in monotonize_doc


def test_readme_shape_constraint_example_executes_current_api() -> None:
    block = _python_block_after_heading(_ROOT / "README.md", "## Monotone Splines")
    namespace: dict[str, object] = {}

    exec(compile(block, "README.md#monotone-splines", "exec"), namespace)

    assert isinstance(namespace["qp_model"], superglm.SuperGLM)
    assert isinstance(namespace["scop_model"], superglm.SuperGLM)


def test_workflow_cross_validation_example_supplies_splitter(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def recording_cross_validate(
        model,
        X,
        y,
        *,
        cv,
        sample_weight=None,
        fit_mode="fit",
        scoring=("deviance",),
        return_oof=False,
    ):
        calls.append(
            {
                "model": model,
                "X": X,
                "y": y,
                "cv": cv,
                "sample_weight": sample_weight,
                "fit_mode": fit_mode,
                "scoring": scoring,
                "return_oof": return_oof,
            }
        )
        return object()

    monkeypatch.setattr(superglm, "cross_validate", recording_cross_validate)
    block = _python_block_after_heading(
        _ROOT / "docs/guide/workflows.md",
        "## 6. Validation And Challenger Comparison",
    )
    namespace = {
        "model": object(),
        "train_df": object(),
        "y_train": np.arange(10, dtype=np.float64),
        "exposure_train": np.ones(10, dtype=np.float64),
    }

    exec(compile(block, "docs/guide/workflows.md#validation", "exec"), namespace)

    assert len(calls) == 1
    assert isinstance(calls[0]["cv"], KFold)
    assert calls[0]["fit_mode"] == "fit_reml"
    assert calls[0]["return_oof"] is True


def test_published_docs_do_not_reference_removed_examples_or_modules() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    deployment = (_ROOT / "docs/guide/deployment.md").read_text(encoding="utf-8")
    optimization = (_ROOT / "docs/guide/optimization.md").read_text(encoding="utf-8")

    assert "monotone_mode=" not in readme
    assert "monotone=" not in readme
    assert "scratch/examples/" not in deployment
    for stale_path in (
        "src/superglm/reml_optimizer.py",
        "src/superglm/reml.py",
        "src/superglm/metrics.py",
        "src/superglm/wood_pvalue.py",
    ):
        assert stale_path not in optimization
