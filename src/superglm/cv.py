"""K-fold cross-validation for lambda selection."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, clip_mu, initial_mean
from superglm.group_matrix import DesignMatrix
from superglm.links import Link, stabilize_eta
from superglm.penalties.base import Penalty
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Result of cross-validated lambda selection.

    Attributes
    ----------
    lambda_seq : NDArray
        Decreasing lambda values evaluated, shape (n_lambda,).
    mean_cv_deviance : NDArray
        Mean test deviance across folds, shape (n_lambda,).
    se_cv_deviance : NDArray
        Standard error of test deviance across folds, shape (n_lambda,).
    best_lambda : float
        Lambda at minimum mean CV deviance.
    best_lambda_1se : float
        Largest lambda within 1 SE of the minimum (most regularised
        model whose CV error is within 1 SE of the best).
    best_index : int
        Index of ``best_lambda`` in ``lambda_seq``.
    best_index_1se : int
        Index of ``best_lambda_1se`` in ``lambda_seq``.
    fold_deviance : NDArray
        Per-fold test deviance, shape (n_folds, n_lambda).
    path_result : object | None
        ``PathResult`` from the refit on full data (if ``refit=True``),
        otherwise ``None``.
    """

    lambda_seq: NDArray
    mean_cv_deviance: NDArray
    se_cv_deviance: NDArray
    best_lambda: float
    best_lambda_1se: float
    best_index: int
    best_index_1se: int
    fold_deviance: NDArray
    path_result: object | None = None


def _fit_cv_folds(
    dm: DesignMatrix,
    y: NDArray,
    sample_weight: NDArray,
    groups: list[GroupSlice],
    family: Distribution,
    link: Link,
    penalty: Penalty,
    lambda_seq: NDArray,
    fold_indices: list[NDArray],
    offset: NDArray | None,
    active_set: bool,
) -> NDArray:
    """Run the K-fold CV inner loop.

    Returns
    -------
    fold_deviance : NDArray, shape (n_folds, n_lambda)
        Normalised test deviance per fold per lambda.
    """
    n_folds = len(fold_indices)
    n_lambda = len(lambda_seq)
    n = dm.n
    all_idx = np.arange(n)
    fold_deviance = np.zeros((n_folds, n_lambda))

    for k in range(n_folds):
        test_idx = fold_indices[k]
        train_idx = np.setdiff1d(all_idx, test_idx)

        dm_train = dm.row_subset(train_idx)
        dm_test = dm.row_subset(test_idx)

        y_train, y_test = y[train_idx], y[test_idx]
        exp_train, exp_test = sample_weight[train_idx], sample_weight[test_idx]
        off_train = offset[train_idx] if offset is not None else None
        off_test = offset[test_idx] if offset is not None else None

        # Null deviance for this fold (fallback for non-convergence)
        mu_null = initial_mean(y_test, exp_test, family)
        dev_null = family.deviance_unit(y_test, np.full_like(y_test, mu_null))
        null_dev = np.sum(exp_test * dev_null) / np.sum(exp_test)

        # Deep-copy penalty to avoid lambda1 state leaking between folds
        fold_penalty = copy.deepcopy(penalty)

        beta_warm = None
        intercept_warm = None

        for i, lam in enumerate(lambda_seq):
            fold_penalty.lambda1 = lam
            result: PIRLSResult = fit_pirls(
                X=dm_train,
                y=y_train,
                weights=exp_train,
                family=family,
                link=link,
                groups=groups,
                penalty=fold_penalty,
                offset=off_train,
                beta_init=beta_warm,
                intercept_init=intercept_warm,
                active_set=active_set,
            )

            if result.converged:
                beta_warm = result.beta
                intercept_warm = result.intercept
            else:
                # Reset warm start to prevent cascade from diverged solution
                beta_warm = None
                intercept_warm = None
                logger.warning(
                    f"CV fold {k + 1}: PIRLS did not converge at "
                    f"lambda={lam:.6g}, using null deviance"
                )
                fold_deviance[k, i] = null_dev
                continue

            # Score on test fold
            eta_test = dm_test.matvec(result.beta) + result.intercept
            if off_test is not None:
                eta_test = eta_test + off_test
            eta_test = stabilize_eta(eta_test, link)
            mu_test = clip_mu(link.inverse(eta_test), family)
            dev_unit = family.deviance_unit(y_test, mu_test)
            test_dev = np.sum(exp_test * dev_unit) / np.sum(exp_test)

            # Cap at 2x null deviance to prevent one wild fold from
            # dominating the mean (overflow from extreme extrapolation)
            fold_deviance[k, i] = min(test_dev, 2.0 * null_dev)

        logger.info(f"CV fold {k + 1}/{n_folds} complete")

    return fold_deviance


def _select_lambda(
    lambda_seq: NDArray,
    mean_cv: NDArray,
    se_cv: NDArray,
    rule: str,
) -> tuple[float, float, int, int]:
    """Select best lambda and 1-SE lambda.

    Returns (best_lambda, best_lambda_1se, best_index, best_index_1se).
    """
    best_idx = int(np.argmin(mean_cv))
    best_lambda = float(lambda_seq[best_idx])

    # 1-SE rule: largest lambda (most regularised) within 1 SE of min
    threshold = mean_cv[best_idx] + se_cv[best_idx]
    # lambda_seq is decreasing, so scan from index 0 (largest lambda)
    candidates = np.where(mean_cv <= threshold)[0]
    idx_1se = int(candidates[0]) if len(candidates) > 0 else best_idx
    lambda_1se = float(lambda_seq[idx_1se])

    if rule == "min":
        return best_lambda, lambda_1se, best_idx, idx_1se
    elif rule == "1se":
        return lambda_1se, lambda_1se, idx_1se, idx_1se
    else:
        raise ValueError(f"Unknown rule: {rule!r}. Use 'min' or '1se'.")
