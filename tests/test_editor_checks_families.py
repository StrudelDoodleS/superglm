"""Every built-in SuperLSS family through the editor's four checks on a held-out split.

A check the family cannot serve is unavailable with the page's sentence, and is not offered;
none raises. A family added to ``superglm.distributional.families`` joins the sweep by itself.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import superglm.distributional.families as families
from superglm import SuperLSS, bind_predictor
from superglm.distributional.family import DistributionalFamily
from superglm.editor.checks import CHECK_REPORTS, check_body, offered_checks
from superglm.editor.evaluation import EvaluationDataset

SEED = 20261128
N_FIT, N_HELD = 200, 100
#: Arguments of the families that need some to be built.
_ARGUMENTS = {"DirichletLSS": (("a", "b", "c"),)}


def _family_names() -> list[str]:
    names = []
    for name in families.__all__:
        kind = getattr(families, name)
        # A family binds a likelihood; links and errors are exported beside the families.
        if not (isinstance(kind, type) and hasattr(kind, "bind_likelihood")):
            continue
        if isinstance(kind(*_ARGUMENTS.get(name, ())), DistributionalFamily):
            names.append(name)
    return names


def _responses(name: str, rng: np.random.Generator) -> np.ndarray:
    """Responses each family takes: counts, compositions, else positive amounts."""
    n = N_FIT + N_HELD
    if name == "NegativeBinomialLS":
        return stats.nbinom.rvs(2, 0.3, size=n, random_state=rng).astype(np.float64)
    if name == "DirichletLSS":
        return stats.dirichlet.rvs([2.0, 3.0, 4.0], size=n, random_state=rng)
    return stats.gamma.rvs(4.0, scale=0.5, size=n, random_state=rng)


@pytest.mark.parametrize("name", _family_names())
def test_every_check_answers_or_says_why_for_each_family(name):
    """Each family intercept-only, its four checks on a held-out split: each body answers, or
    is unavailable with a sentence, and a check not offered is never answered. Mutation check:
    without the variance capability in ``_unsupported`` NegativeBinomialLS offers actual over
    expected and its body raises ``NotImplementedError``."""
    family = getattr(families, name)(*_ARGUMENTS.get(name, ()))
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, N_FIT + N_HELD)})
    y = _responses(name, rng)
    predictors = [bind_predictor(family, parameter.name) for parameter in family.parameters]
    with warnings.catch_warnings():
        # A start or a stop on these few rows may warn; the checks are what is read here.
        warnings.simplefilter("ignore")
        lss = SuperLSS(family, *predictors).fit(X.iloc[:N_FIT], y[:N_FIT])
    held = EvaluationDataset(
        name="validation",
        label="Validation",
        X=X.iloc[N_FIT:].reset_index(drop=True),
        y=y[N_FIT:],
    )
    offered = offered_checks(lss)
    for report in CHECK_REPORTS:
        body = check_body(report, lss, held, inputs=["x"])
        json.dumps(body, allow_nan=False)
        if not body["available"]:
            assert isinstance(body["reason"], str) and body["reason"], (name, report)
        else:
            assert report in offered, (name, report)
