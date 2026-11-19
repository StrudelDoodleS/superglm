"""Saturated fits whose unseen cell has no mean under a link bounded at ``eta = 0``.

Three of the four cells of two two-level factors fix the fit, so the fourth
cell's linear predictor is ``2 g(m) - g(1) = -0.5`` in closed form, where the
inverse, inverse-squared and negative-power links have no mean.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit

from superglm import Categorical, SuperGLM
from superglm.links import IdentityLink, PowerLink

CELLS = pd.DataFrame({"a": ["0", "1", "0", "1"], "b": ["0", "0", "1", "1"]})
CELLS_Y = np.array([1.0, 4.0, 4.0, 3.0])
# The 30 training rows of the editor's reproducers: cells (0, 0), (1, 0), (0, 1) at means 1, 4, 4.
TRAIN = pd.DataFrame(
    {"a": ["0", "0", "1", "1", "0", "0"] * 5, "b": ["0", "0", "0", "0", "1", "1"] * 5}
)
TRAIN_Y = np.array([0.5, 1.5, 2.0, 6.0, 2.0, 6.0] * 5)

_LINKS = {"power-1": lambda: PowerLink(-1.0), "power-2": lambda: PowerLink(-2.0)}

# Each link with the response levels whose cell mean ``m`` puts ``g(m) = 1/4``.
THREE_CELL_LINKS = pytest.mark.parametrize(
    ("link", "y_level"),
    [
        ("inverse", [2.0, 6.0]),
        ("inverse_squared", [1.0, 3.0]),
        ("power-1", [2.0, 6.0]),
        ("power-2", [1.0, 3.0]),
    ],
)


def three_cell_fit(link, y_level, *, unseen_weight=None, family="gamma"):
    """The saturated fit on the three seen cells; its means are the cell means ``1, m, m``.

    ``unseen_weight`` adds the fourth cell to the training rows at this weight,
    which must be zero for the fit to stay the same.
    """
    y_base = [0.5, 1.5]
    y = [*y_base, *y_level, *y_level]
    a = ["0"] * len(y_base) + ["1"] * len(y_level) + ["0"] * len(y_level)
    b = ["0"] * len(y_base) + ["0"] * len(y_level) + ["1"] * len(y_level)
    weights = None
    if unseen_weight is not None:
        y, a, b = [*y, 3.0], [*a, "1"], [*b, "1"]
        weights = np.r_[np.ones(len(y) - 1), unseen_weight]
    X = pd.DataFrame({"a": a, "b": b})
    model = SuperGLM(
        family=family,
        link=_LINKS[link]() if link in _LINKS else link,
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(X, np.asarray(y), sample_weight=weights)
    assert model.result.converged
    return model, X, np.asarray(y), weights


def editor_fit(family="gamma", y=TRAIN_Y):
    """The editor reproducers' Gamma/inverse fit on ``TRAIN``: the unseen cell (1, 1) has eta -0.5."""
    return SuperGLM(
        family=family,
        link="inverse",
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, y)


def cv_frame():
    """Training rows, the four cells as fold 0's held-out rows, and three priced cells as fold 1's.

    Fold 0's fit is the editor reproducers' (``editor_fit``): row 33, the unseen cell, has no mean.
    """
    X = pd.concat([TRAIN, CELLS, CELLS.iloc[:3]], ignore_index=True)
    y = np.r_[TRAIN_Y, CELLS_Y, CELLS_Y[:3]]
    folds = np.r_[np.full(len(TRAIN_Y), -1), np.zeros(4), np.ones(3)]
    return X, y, PredefinedSplit(folds)


class UpperBoundedIdentity(IdentityLink):
    """An identity link whose inverse has no value above 10: a NaN the rule does not declare."""

    def inverse(self, eta):
        return np.where(eta <= 10.0, eta, np.nan)
