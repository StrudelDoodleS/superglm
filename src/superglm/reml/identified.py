"""The identified part of the Laplace approximation (one-engine design §3.9).

**The class.**  A slope is weakly identified by its specification when no
penalty touches it and its column, centred on its shifted prior-weighted
mean (design §3.2), carries curvature only through rows whose prior weight
is within the rounding of an accumulation at the largest prior weight:

    sum_r w_r x~_rj^2 <= gamma_{n+} max_r(w_r) sum_{w_r > 0} x~_rj^2,

``gamma_k = k u / (1 - k u)`` (Higham 2002, Lemma 3.1), ``n+`` the rows with
positive prior weight: the design's §3.9 test with the working weights taken
at unit variance factor, so that it is a function of the design and the prior
weights alone -- rows weighted 1e-15 beside rows weighted 1.  A penalty
identifies a slope however small its data curvature, so only slopes outside
every penalty component are candidates (a penalty at the rounding of the data
is the border's step-5 truncation instead, design §3.6).

**Why the Laplace approximation leaves it out.**  Along such a slope the
likelihood is flat to within float64: PIRLS reaches no reproducible value of
it (a deviance-stopped iteration never converges it, a certificate-stopped
one excludes it by design §3.8), yet its log-curvature enters ``log|H|`` at
full weight, because ``log`` is scale-free.  ``log|H|`` then moves with
wherever PIRLS stopped, and REML's smoothing parameters for every other term
move with it: path-dependent, and silently wrong (the stage-0 verifier's
``offset_rare`` Poisson, lambda_u 19.0 against 10.47).  The approximation is
therefore taken over the identified slopes ``I`` with the weakly identified
slopes ``W`` held fixed at the fitted values (decision 7: flag, keep, and
certify the identified part).  By the Schur determinant identity and the
block inverse,

    log|H_II| = log|H| + log det((H^-1)_WW),
    [H_II^-1, 0; 0, 0] = H^-1 - H^-1 E_W ((H^-1)_WW)^-1 E_W' H^-1,

the second with exactly zero ``W`` rows and columns.  The REML objective, its
gradient, the ``W(rho)`` correction and the outer Hessian all read these, so
they are one function of the smoothing parameters whatever value of
``beta_W`` PIRLS leaves: the identified block depends on ``beta_W`` only
through rows weighted at the rounding.  The coefficient rank the scale
profile counts loses ``|W|`` with them.  The fit keeps ``beta_W``, and its
standard error comes from the full ``H``, never from this part.

The set is decided once per fit, before any iterate, from the design and the
prior weights: no classification changes between REML candidates, and none
selects a route (design §6).  With ``W`` empty every method returns its
argument, so every other fit is unchanged bit for bit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.group_matrix import CategoricalGroupMatrix, DenseGroupMatrix, DesignMatrix

_UNIT_ROUNDOFF = 2.0**-53
_CHUNK = 8192


class WeakIdentificationWarning(UserWarning):
    """Some coefficients carry information only at the noise level of the data.

    Fires once per ``fit_reml`` and names the coefficients.  Each belongs to a
    factor level or column with little or no weight, few observations or
    little information: the data pin it down no better than the rounding of
    the computation does.  The fit keeps these coefficients in the model and
    says so plainly: those whose information is at noise level are left out
    of smoothing-parameter (REML) selection, and every named coefficient's
    estimate and standard error carry little information.  They are listed
    in ``model.diagnostics()`` (per group under ``"weakly_identified"`` and,
    with a note, under ``"_model"``).  To remove the warning, drop the column
    or merge the level, or give its rows more weight or data.
    """


WEAK_IDENTIFICATION_NOTE = (
    "These coefficients carry information only at the noise level of the data (a factor "
    "level or column with little or no weight, observations or information). They stay in "
    "the model; those whose information is at noise level are left out of "
    "smoothing-parameter selection, and their estimates and standard errors carry little "
    "information."
)


def _gamma(count: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for ``k`` roundings, ``u = 2^-53``."""
    product = count * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else float("inf")


def penalized_columns(width: int, penalties: Sequence | None) -> NDArray:
    """``(width,)`` bool: the slopes some penalty component's block covers."""
    mask = np.zeros(width, dtype=bool)
    for component in penalties or ():
        mask[component.group_sl] = True
    return mask


def laplace_excluded_coefficients(
    dm: DesignMatrix,
    sample_weight: NDArray,
    penalties: Sequence | None,
) -> NDArray:
    """The slopes the Laplace approximation leaves out (module docstring), ascending.

    One-hot columns are evaluated in closed form from two transpose products
    (entries 0 and 1 about a centre in [0, 1] cannot cancel); a
    ``DenseGroupMatrix`` column is centred row by row in fixed chunks about its
    shifted prior-weighted mean, so its rounding scales with ``|x - c|`` and
    not with a column offset; any other unpenalized column is formed once by a
    design product.  A column with no centred mass on the weighted rows is an
    exact alias of the intercept, a rank decision, and is never in the set.
    """
    weights = np.asarray(sample_weight, dtype=np.float64)
    positive = weights > 0.0
    row_count = int(np.count_nonzero(positive))
    candidates = ~penalized_columns(dm.p, penalties)
    if row_count == 0 or not np.any(candidates):
        return np.zeros(0, dtype=np.intp)
    total = float(np.sum(weights))
    bar = _gamma(row_count) * float(np.max(weights))
    first = int(np.argmax(positive))
    curvature = np.zeros(dm.p)
    mass = np.zeros(dm.p)
    evaluated = np.zeros(dm.p, dtype=bool)
    on_weight = on_rows = None
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = np.arange(offset, offset + width)
        offset += width
        take = candidates[columns]
        if not np.any(take):
            continue
        evaluated[columns] = True
        if isinstance(matrix, CategoricalGroupMatrix):
            if on_weight is None:
                on_weight = dm.rmatvec(weights)
                on_rows = dm.rmatvec(positive.astype(np.float64))
            assert on_rows is not None
            level_weight, level_rows = on_weight[columns], on_rows[columns]
            share = level_weight / total
            curvature[columns] = (1.0 - share) ** 2 * level_weight + share**2 * (
                total - level_weight
            )
            mass[columns] = (1.0 - share) ** 2 * level_rows + share**2 * (row_count - level_rows)
        elif isinstance(matrix, DenseGroupMatrix):
            values = matrix.M
            reference = np.array(values[first], dtype=np.float64)
            shift = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                shift += weights[lo:hi] @ (values[lo:hi] - reference)
            centre = reference + shift / total
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                squares = (values[lo:hi] - centre) ** 2
                curvature[columns] += weights[lo:hi] @ squares
                mass[columns] += positive[lo:hi].astype(np.float64) @ squares
        else:
            for column in columns[take]:
                unit = np.zeros(dm.p)
                unit[column] = 1.0
                values = dm.matvec(unit)
                reference = float(values[first])
                centre = reference + float(weights @ (values - reference)) / total
                squares = (values - centre) ** 2
                curvature[column] = float(weights @ squares)
                mass[column] = float(np.sum(squares[positive]))
    weak = evaluated & candidates & (mass > 0.0) & (curvature <= bar * mass)
    return np.flatnonzero(weak).astype(np.intp)


class IdentifiedLaplace:
    """The identified part of one fit's Laplace approximation (module docstring).

    ``excluded`` holds the slope indices ``W``.  Every method maps the full
    ``H`` object a REML evaluation holds to its identified part, and returns
    its argument unchanged when ``W`` is empty.  ``unsupported`` counts the
    factors of a family this cannot restrict (the ``sz`` block factor,
    replaced by the engine in stage 3): those evaluations keep the full
    ``H``, and the count is published.  The ``fs`` leaf factor and the nested
    factor are rebuilt with the excluded border columns left out.
    """

    def __init__(self, excluded: NDArray | Sequence[int] = ()):
        self.excluded = np.asarray(excluded, dtype=np.intp)
        self.unsupported = 0

    def __bool__(self) -> bool:
        return bool(self.excluded.size)

    def _block(self, inverse) -> NDArray:
        """``(H^-1)_WW`` from a dense inverse or any Hessian factor."""
        W = self.excluded
        if isinstance(inverse, np.ndarray):
            return np.asarray(inverse[np.ix_(W, W)], dtype=np.float64)
        from superglm.solvers.hessian_factor import as_hessian_factor

        return np.asarray(as_hessian_factor(inverse).selected_inverse_block(W), dtype=np.float64)

    def _lower(self, block: NDArray) -> NDArray | None:
        """The Cholesky factor of ``(H^-1)_WW``, ``None`` when the full factor already truncated ``W``.

        A generalized inverse whose ``W`` block is not positive definite
        belongs to a factor that took a weakly identified direction as a null
        already (gram's rank decision): its pseudo-determinant has left it out.
        """
        block = 0.5 * (block + block.T)
        if not np.all(np.isfinite(block)):
            return None
        try:
            return scipy.linalg.cholesky(block, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None

    def log_det(self, inverse, log_det: float | None) -> float | None:
        """``log|H_II| = log|H| + log det((H^-1)_WW)``."""
        if not self or inverse is None or log_det is None:
            return log_det
        lower = self._lower(self._block(inverse))
        if lower is None:
            return log_det
        return float(log_det + 2.0 * np.sum(np.log(np.diag(lower))))

    def log_det_of_hessian(self, hessian: NDArray, log_det: float) -> tuple[float, bool]:
        """``log|H_II|`` from a dense ``H`` itself (a cached solve with no inverse).

        ``(H^-1)_WW`` by one Cholesky solve of the Jacobi-equilibrated ``H``;
        the flag says it applied (``False`` leaves ``log_det`` for a system
        whose factor did not certify).
        """
        if not self:
            return log_det, False
        matrix = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian).T)
        diagonal = np.diag(matrix)
        if not np.all(np.isfinite(matrix)) or not np.all(diagonal > 0.0):
            return log_det, False
        scale = 1.0 / np.sqrt(diagonal)
        try:
            lower = scipy.linalg.cholesky(
                scale[:, None] * matrix * scale[None, :], lower=True, check_finite=False
            )
        except np.linalg.LinAlgError:
            return log_det, False
        units = np.zeros((len(diagonal), self.excluded.size))
        units[self.excluded, np.arange(self.excluded.size)] = scale[self.excluded]
        solved = scipy.linalg.cho_solve((lower, True), units, check_finite=False)
        block = scale[self.excluded, None] * solved[self.excluded]
        chol = self._lower(block)
        if chol is None:
            return log_det, False
        return float(log_det + 2.0 * np.sum(np.log(np.diag(chol)))), True

    def rank(self, rank: int | None, inverse) -> int | None:
        """The coefficient rank of the identified part."""
        if not self or rank is None or inverse is None:
            return rank
        if self._lower(self._block(inverse)) is None:
            return rank
        return int(rank) - int(self.excluded.size)

    def _dense(self, inverse: NDArray) -> NDArray:
        W = self.excluded
        full = np.asarray(inverse, dtype=np.float64)
        lower = self._lower(full[np.ix_(W, W)])
        part = full.copy()
        if lower is not None:
            columns = full[:, W]
            part -= columns @ scipy.linalg.cho_solve((lower, True), columns.T, check_finite=False)
            part = 0.5 * (part + part.T)
        part[W, :] = 0.0
        part[:, W] = 0.0
        return part

    def inverse(self, inverse):
        """The identified inverse ``[H_II^-1, 0; 0, 0]`` in the same representation."""
        if not self or inverse is None:
            return inverse
        from superglm.solvers._structured.nested import (
            NestedSchurFactor,
            ProfiledNestedSchurFactor,
        )
        from superglm.solvers.hessian_factor import DenseHessianFactor

        if isinstance(inverse, np.ndarray):
            return self._dense(inverse)
        if isinstance(inverse, DenseHessianFactor):
            log_det = self.log_det(inverse, inverse.logdet())
            return DenseHessianFactor(
                inverse=self._dense(inverse.inverse),
                log_det=float("nan") if log_det is None else log_det,
            )
        if isinstance(inverse, ProfiledNestedSchurFactor):
            augmented = inverse.augmented_factor
            border = self.excluded[np.isin(self.excluded + 1, augmented.small_indices)]
            if border.size != self.excluded.size:
                self.unsupported += 1
                return inverse
            rebuilt = NestedSchurFactor(
                augmented.operator,
                chain_group_names=augmented.chain_group_names,
                chain_group_indices=augmented.chain_group_indices,
                intercept=True,
                max_structured_inverse_block=augmented.max_structured_inverse_block,
                excluded=tuple(int(index) + 1 for index in border),
            )
            return ProfiledNestedSchurFactor(
                augmented_factor=rebuilt,
                sum_w=inverse.sum_w,
                xtw=inverse.xtw,
                data_operator=inverse.data_operator,
            )
        from superglm.solvers._structured.block_leaves import (
            FactorSmoothLeafFactor,
            ProfiledFactorSmoothLeafFactor,
        )

        if isinstance(inverse, ProfiledFactorSmoothLeafFactor):
            augmented = inverse.augmented_factor
            border = self.excluded[np.isin(self.excluded + 1, augmented.small_indices)]
            if border.size != self.excluded.size:
                self.unsupported += 1
                return inverse
            rebuilt = FactorSmoothLeafFactor(
                augmented.system,
                augmented.penalized,
                max_structured_inverse_block=augmented.max_structured_inverse_block,
                excluded=tuple(int(index) + 1 for index in border),
            )
            return ProfiledFactorSmoothLeafFactor(
                augmented_factor=rebuilt, sum_w=inverse.sum_w, xtw=inverse.xtw
            )
        from superglm.solvers._structured.balance_tree import (
            ProfiledSumToZeroTreeFactor,
            SumToZeroTreeFactor,
        )

        if isinstance(inverse, ProfiledSumToZeroTreeFactor):
            augmented = inverse.augmented_factor
            border = self.excluded[np.isin(self.excluded + 1, augmented.small_indices)]
            if border.size != self.excluded.size:
                self.unsupported += 1
                return inverse
            rebuilt = SumToZeroTreeFactor(
                augmented.system,
                augmented.penalized,
                max_structured_inverse_block=augmented.max_structured_inverse_block,
                excluded=tuple(int(index) + 1 for index in border),
            )
            return ProfiledSumToZeroTreeFactor(
                augmented_factor=rebuilt, sum_w=inverse.sum_w, xtw=inverse.xtw
            )
        self.unsupported += 1
        return inverse

    def geometry(self, geometry):
        """An observed REML geometry with its inverse, determinant and rank restricted."""
        if not self or geometry is None:
            return geometry
        inverse = geometry.hessian_inverse
        return dataclasses.replace(
            geometry,
            hessian_inverse=self.inverse(inverse),
            log_det_H=self.log_det(inverse, geometry.log_det_H),
            hessian_rank=self.rank(geometry.hessian_rank, inverse),
        )


def penalty_diagonal(width: int, lambdas: dict[str, float], penalties: Sequence | None) -> NDArray:
    """``diag(S)`` ``(width,)`` from the penalty components, never forming ``S``.

    Identity blocks add ``lambda``; dense blocks ``lambda diag(Omega)``;
    repeated blocks tile it; a sum-to-zero block is ``(C'C) kron Omega`` with
    ``C = [I; -1]``, whose diagonal is ``2 diag(Omega)`` per level.
    """
    diagonal = np.zeros(width)
    for component in penalties or ():
        lam = float(lambdas.get(component.name, 0.0))
        if lam == 0.0:
            continue
        block = component.group_sl
        size = len(range(*block.indices(width)))
        if component.penalty_kind == "identity":
            diagonal[block] += lam
            continue
        local = np.diag(np.asarray(component.omega_ssp, dtype=np.float64))
        if component.penalty_kind == "repeated":
            local = np.tile(local, int(component.repeat_count))
        elif component.penalty_kind == "sum_to_zero":
            local = 2.0 * np.tile(local, int(component.repeat_count) - 1)
        if local.shape != (size,):
            raise ValueError(f"Penalty component {component.name!r} does not match its block.")
        diagonal[block] += lam * local
    return diagonal


def final_mode_weak_slopes(
    *,
    dm: DesignMatrix,
    distribution,
    link,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result,
    lambdas: dict[str, float],
    penalties: Sequence | None,
) -> NDArray:
    """The §3.9 weak test at a fit's final mode, from its Fisher rows (``mode_score.weakly_identified_mask``)."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor, weakly_identified_mask
    from superglm.solvers.working_rows import fisher_working_weights

    weights = np.asarray(sample_weight, dtype=np.float64)
    eta = stabilize_eta(linear_predictor(dm, result, offset_arr), link)
    mu = clip_mu(link.inverse(eta), distribution)
    fisher = fisher_working_weights(
        distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=weights
    )
    fisher = np.where(np.isfinite(fisher), np.maximum(fisher, 0.0), 0.0)
    total = float(np.sum(fisher))
    if not total > 0.0:
        return np.zeros(0, dtype=np.intp)
    mask = weakly_identified_mask(
        dm=dm,
        fisher_weights=fisher,
        positive_prior=weights > 0.0,
        mean_x=dm.rmatvec(fisher) / total,
        penalty_diagonal=penalty_diagonal(dm.p, lambdas, penalties),
        unpenalized=~penalized_columns(dm.p, penalties),
    )
    return np.flatnonzero(mask).astype(np.intp)


def coefficient_labels(groups: Sequence, indices: Sequence[int]) -> tuple[str, ...]:
    """``group[column]`` names of slope indices (the column within its group)."""
    labels = []
    for index in indices:
        group = next((g for g in groups if g.start <= int(index) < g.end), None)
        labels.append(
            f"coef[{int(index)}]" if group is None else f"{group.name}[{int(index) - group.start}]"
        )
    return tuple(labels)
