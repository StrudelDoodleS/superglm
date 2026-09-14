"""Training-only pair proposals from a fixed pairwise histogram booster.

For each internal node, take the distinct predictors on its root-to-node
path, including its own split. If that support has exactly two predictors,
charge max(0, node gain) once to that unordered pair. Repeated splits on the
same two predictors contribute their own gains. One-feature nodes and leaves
contribute nothing. A path with more than two predictors is refused.

These summed training split gains are a proposal heuristic. They are not a
pure-interaction variance, an ANOVA decomposition, a held-out improvement,
or a certificate. Main effects, correlated proxies and greedy tree choices
can all affect this ranking. A later bounded SuperGLM comparison must decide
whether any proposed pair helps, including the option of selecting none.
Greedy boosting can miss pure interactions with no profitable first split,
as in a balanced XOR. An empty proposal list is not evidence of additivity.

The caller supplies training rows, an already fitted training-only adapter,
and optional training weights. This module has no validation/test argument,
does not split or subsample, and does not launch subprocesses. The caller
owns process isolation, timeouts and numerical thread limits.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from collections import defaultdict

import numpy as np
from benchmark_gbm_interactions import build_model, native_features, peak_rss, tree_diagnostics
from benchmark_housing_tensor import retained_model_storage


def _original_feature_positions(model, columns):
    """Recover fitted ColumnTransformer output positions, not input positions.

    In sklearn 1.9 the encoder output precedes the numeric columns. Reading
    the fitted selectors and output slices also detects incompatible layouts.
    """
    n_features = model.n_features_in_
    if len(columns) != n_features or len(set(columns)) != n_features:
        raise ValueError("Original column count or uniqueness differs from the fitted model")
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != columns:
        raise ValueError("Original column order differs from the fitted model")
    preprocessor = model._preprocessor
    if preprocessor is None:
        if model.is_categorical_ is not None and np.any(model.is_categorical_):
            raise ValueError("Categorical model is missing its fitted column mapping")
        return np.arange(n_features)

    original = np.full(n_features, -1, dtype=int)
    selectors = {name: selector for name, _, selector in preprocessor.transformers_}
    for name in ("encoder", "numerical"):
        selector = np.asarray(selectors[name])
        if selector.dtype != bool or selector.shape != (n_features,):
            raise ValueError("Unsupported fitted column selector layout")
        positions = np.arange(n_features)[preprocessor.output_indices_[name]]
        inputs = np.flatnonzero(selector)
        if len(positions) != len(inputs) or np.any(original[positions] != -1):
            raise ValueError("Fitted column transform changes width or overlaps outputs")
        original[positions] = inputs
    if not np.array_equal(np.sort(original), np.arange(n_features)):
        raise ValueError("Fitted column transform does not preserve every predictor")
    if not np.array_equal(
        np.asarray(model.is_categorical_)[original], model._is_categorical_remapped
    ):
        raise ValueError("Fitted native categorical mapping is inconsistent")
    return original


def score_fitted_pairs(model, original_columns):
    """Return ranked original-label pairs from an already fitted tree ensemble.

    Pairs are tuples in Python and serialize as two-element JSON arrays.
    Within each pair, original column order supplies the canonical order.
    Score ties use those original positions, independent of tree traversal.
    """
    columns = list(original_columns)
    original = _original_feature_positions(model, columns)
    contributions = defaultdict(list)
    for iteration in model._predictors:
        for tree in iteration:
            nodes = tree.nodes
            stack = [(0, frozenset())]
            visited = set()
            while stack:
                index, ancestors = stack.pop()
                if index in visited or not 0 <= index < len(nodes):
                    raise ValueError("Fitted nodes do not form a rooted tree")
                visited.add(index)
                node = nodes[index]
                if node["is_leaf"]:
                    continue
                feature = int(node["feature_idx"])
                if not 0 <= feature < len(columns):
                    raise ValueError("Fitted split feature is outside the column mapping")
                support = ancestors | {int(original[feature])}
                if len(support) > 2:
                    raise ValueError("A fitted branch contains more than two distinct predictors")
                gain = float(node["gain"])
                if not math.isfinite(gain):
                    raise ValueError("Fitted split gain must be finite")
                if len(support) == 2:
                    contributions[tuple(sorted(support))].append(max(0.0, gain))
                stack.extend((int(node[side]), support) for side in ("left", "right"))

    ranked = []
    for positions, gains in contributions.items():
        try:
            score = math.fsum(gains)
        except OverflowError as error:
            raise ValueError("Accumulated pair gain must be finite") from error
        if not math.isfinite(score):
            raise ValueError("Accumulated pair gain must be finite")
        ranked.append((score, positions, len(gains)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    scores = [
        {
            "pair": tuple(columns[position] for position in positions),
            "score": score,
            "split_nodes": count,
        }
        for score, positions, count in ranked
    ]
    return {
        "pairs": [record["pair"] for record in scores],
        "scores": scores,
        "internal_feature_order": [columns[position] for position in original],
        "policy": "Sum max(0, split gain) once per internal node with exactly two distinct root-to-node predictors, including the current split; ties follow original column positions",
        "scope": "Training tree heuristic for candidate proposals; not pure interaction strength or held-out benefit",
    }


def discover_pairs(raw_train_frame, state, response, family, sample_weight=None):
    """Fit the fixed pairwise/leaves15/200-round proposal model on supplied rows."""
    started = time.perf_counter()
    y = np.asarray(response, dtype=float)
    if y.shape != (len(raw_train_frame),) or not len(y) or not np.isfinite(y).all():
        raise ValueError("Training response must be finite, one-dimensional and row-aligned")
    if family == "binomial" and not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Binomial training response must contain only zero and one")
    total_weight = float(len(y))
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Training weights must be finite, nonnegative and row-aligned")
        try:
            total_weight = math.fsum(weights)
        except OverflowError as error:
            raise ValueError("Training weights must have a finite positive total") from error
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Training weights must have a finite positive total")
    train = native_features(raw_train_frame, state)
    prep_seconds = time.perf_counter() - started
    model = build_model(family, "pairwise", "leaves15")
    fit_started = time.perf_counter()
    model.fit(train, y, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - fit_started
    fit_end_rss = peak_rss()
    storage = retained_model_storage(model)
    extraction_started = time.perf_counter()
    receipt = score_fitted_pairs(model, train.columns)
    diagnostics = tree_diagnostics(model)
    extraction_seconds = time.perf_counter() - extraction_started
    if model.n_iter_ != 200 or model.do_early_stopping_:
        raise ValueError("Proposal model did not complete the fixed 200-round budget")
    receipt.update(
        training_rows=len(y),
        training_total_weight=total_weight,
        sample_weight_provided=sample_weight is not None,
        input_columns=list(train.columns),
        native_categorical_levels={
            name: train[name].cat.categories.tolist()
            for name, spec in state["features"].items()
            if spec["kind"] == "categorical"
        },
        family=family,
        model_parameters=model.get_params(),
        sklearn_version=importlib.metadata.version("scikit-learn"),
        tree_diagnostics=diagnostics,
        retained_model_storage=storage,
        fit_end_peak_process_rss_mib=fit_end_rss,
        memory_scope="Process high-water at fit end includes earlier caller work; retained storage counts model NumPy/byte owners, not the full Python heap",
        timing={
            "native_preparation_seconds": prep_seconds,
            "fit_seconds": fit_seconds,
            "extraction_seconds": extraction_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    )
    return receipt
