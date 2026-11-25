"""An affine restriction of a REML design to its freely fitted level and cell effects."""

from dataclasses import dataclass, replace

import numpy as np

from superglm._family_link import pair_of
from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
from superglm.model.separated_levels import separated_labels


@dataclass(frozen=True)
class _HeldIndicators:
    full: CategoricalGroupMatrix
    free: CategoricalGroupMatrix
    levels: np.ndarray
    columns: np.ndarray


def _free_indicators(matrix, levels):
    codes = np.arange(matrix.n_levels + 1)
    codes[levels] = matrix.n_levels
    return CategoricalGroupMatrix(codes[matrix.codes], matrix.n_levels)


def inference_design(model):
    """Restore the fit's free columns when forming covariance on the fitted design."""
    dm = model._dm
    held = getattr(getattr(model, "_result", None), "held_level_columns", ())
    if not held:
        return dm
    matrices = list(dm.group_matrices)
    for i, group in enumerate(model._groups):
        local = [col - group.start for col in held if group.start <= col < group.end]
        if local:
            matrices[i] = _free_indicators(matrices[i], local)
    return DesignMatrix(matrices, dm.n, dm.p)


class REMLLevelHold:
    """Keep native column indices for the penalty, factor and inference consumers.

    Map held indicator columns to the base (zero) bin. Their exactly zero
    columns have no free coordinate in the solver's rank decomposition.
    Only their original, finite contribution enters the offset.
    """

    @classmethod
    def for_model(cls, model, weights, offset, anchors):
        blocks = {}
        separated_columns = []
        for i, (group, matrix) in enumerate(
            zip(model._groups, model._dm.group_matrices, strict=True)
        ):
            if type(matrix) is not CategoricalGroupMatrix:
                continue
            spec = model._specs.get(group.feature_name)
            cross = model._interaction_specs.get(group.feature_name)
            labels = set(separated_labels(spec if cross is None else cross))
            if not labels:
                continue
            # A crossed cell has its own indicator only on the emitted non-base
            # grid. Margins and pruned cells have no column in this block.
            non_base = spec._non_base if cross is None else cross._pairs
            if matrix.n_levels != len(non_base):
                continue
            held = [j for j, level in enumerate(non_base) if level in labels]
            n_separated = len(held)
            if cross is None and spec._base_level in labels:
                if pair_of(model._distribution, model._link).mean_space_violation is not None:
                    # Its reference direction spans the intercept and every
                    # level. The existing mean-space solve preserves invariance
                    # to offset shifts, including starts lowered from above 1.
                    continue
                # The separated base is held against the first occupied healthy
                # level: in native coding this fixes that level's contrast to
                # the base. The intercept and other healthy contrasts stay free.
                support = np.bincount(matrix.codes[weights > 0], minlength=len(non_base) + 1)
                ref = next(
                    (j for j, level in enumerate(non_base) if level not in labels and support[j]),
                    None,
                )
                if ref is not None:
                    held.append(ref)
            if held:
                local = np.asarray(held)
                free = _free_indicators(matrix, local)
                blocks[i] = _HeldIndicators(matrix, free, local, group.start + local)
                separated_columns.extend(blocks[i].columns[:n_separated])
        if not blocks:
            return None
        return cls(
            blocks,
            np.zeros_like(weights) if offset is None else offset.copy(),
            anchors,
            separated_columns,
        )

    def __init__(self, blocks, offset, anchors, separated_columns):
        self.blocks = blocks
        self.offset = offset
        self.anchors = anchors
        self.columns = np.concatenate([block.columns for block in blocks.values()])
        self.separated_columns = separated_columns
        if anchors:
            self._add_offset()

    def design(self, dm, *, original=False):
        matrices = list(dm.group_matrices)
        for i, block in self.blocks.items():
            matrices[i] = block.full if original else block.free
        return DesignMatrix(matrices, dm.n, dm.p)

    def _add_offset(self):
        for block in self.blocks.values():
            shift = np.zeros(block.full.n_levels + 1)
            shift[block.levels] = [self.anchors[j] for j in block.columns]
            self.offset += shift[block.full.codes]

    def _capture(self, result):
        self.anchors.update(zip(self.columns, result.beta[self.columns], strict=True))
        self._add_offset()

    def bootstrap(self, solve, **kwargs):
        if self.anchors:
            return solve(**kwargs)
        # No geometry formed on the unrestricted columns can seed a later
        # coefficient solve. Its inverse still seeds the initial lambda step.
        kwargs["X"] = self.design(kwargs["X"], original=True)
        for key in ("_initial_data_reuse", "_fisher_data_reuse", "_raw_moment_policy"):
            kwargs.pop(key, None)
        result, inverse, gram = solve(**kwargs)
        beta = result.beta.copy()
        self._capture(result)
        beta[self.columns] = 0.0
        # The next solve re-centres the raw intercept in the free design.
        return replace(result, beta=beta, centred_intercept=None, state_center=None), inverse, gram

    def lift(self, model, result):
        model._dm = self.design(model._dm, original=True)
        beta = result.beta.copy()
        beta[self.columns] = [self.anchors[j] for j in self.columns]
        # The free design's centre is zero on the held columns, so its
        # centred intercept is also valid with their contributions restored.
        return replace(result, beta=beta, held_level_columns=tuple(map(int, self.columns)))
