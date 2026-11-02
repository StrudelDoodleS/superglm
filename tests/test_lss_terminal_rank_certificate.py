"""SuperLSS's terminal rank, certified against the error of forming its curvature.

Forming ``H = sum_r X_r' C_r X_r + S`` from the rows moves its
Jacobi-equilibrated eigenvalues by more than the rank cut, so a direction
within that distance of the cut was decided by rounding: an exact alias was
kept at rank 3 with variances near 1e10, and a near-null estimable direction
kept the rank but published a variance off by percents.  The terminal
decision now reads the rows on those directions once
(``solver/certificate.py``).  Every fixture forces its event: an alias the
formed matrix keeps (the Cholesky shortcut on constant weights, the
eigensolver on uniform ones), a near pair and an offset column whose
eigenvalues sit inside the formation bound, so the decision is the rows'
and not the rounding's.  Variances are checked against an exact rational
inverse of the fit's own rows.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver._formation as _formation
import superglm.distributional.solver.certificate as certificate
import superglm.distributional.solver.chunks as chunking
import superglm.distributional.solver.curvature as curvature_module
from superglm import NegativeBinomialLS, SuperLSS, cat, interaction, re, s, term, ti
from superglm.distributional import GaussianLS
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.distributional.solver.packing import packed_pairs
from superglm.features import Numeric
from superglm.features.interaction import SplineCategorical


@pytest.fixture
def terminal(monkeypatch):
    """Record each terminal decision and count ``project`` calls."""
    decisions, calls = [], []
    certify = certificate.certify_terminal_rank
    deflated = certificate.decompose_symmetric_deflated

    def recorded(decision, formation, **kwargs):
        out = certify(decision, formation, **kwargs)
        decisions.append((decision, formation, kwargs, out))
        return out

    def counted(matrix, project, **kwargs):
        def wrapped(basis):
            calls.append(basis.shape[1])
            return project(basis)

        return deflated(matrix, wrapped, **kwargs)

    monkeypatch.setattr(certificate, "certify_terminal_rank", recorded)
    monkeypatch.setattr(certificate, "decompose_symmetric_deflated", counted)
    return decisions, calls


def _main(decisions, width):
    """The last terminal decision of the model's own layout (the null model has its own)."""
    return next(r for r in reversed(decisions) if r[2]["layout"].n_coefficients == width)


def _uncertified(monkeypatch):
    """The unfixed terminal decision: the policy's formed rank, as before the certificate."""
    monkeypatch.setattr(
        certificate, "certify_terminal_rank", lambda decision, formation, **kwargs: decision
    )


def _fit(model, frame, y, weights=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y, sample_weight=weights)
    return model._require_fitted().fit_state.solver_result


def _alias(constant: bool, discrete: bool = True):
    """``x = 1[b]`` and the categorical's level ``b``: an exact alias, rank 2 of 3."""
    n = 32768 if constant else 30001
    b = np.arange(n) % 2 == 1
    frame = pd.DataFrame(
        {"x": b.astype(float), "g": np.where(b, "b", "a"), "z": (~b).astype(float)}
    )
    weights = np.ones(n)
    weights[b] = (1.0 + 2.0**-40) if constant else np.random.default_rng(1).uniform(0.1, 2, b.sum())
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(
        family,
        family.location(term("x", Numeric()), cat("g", base="a"), intercept=False),
        family.scale(term("z", Numeric()), intercept=False),
        discrete=discrete,
    )
    return model, frame, (~b).astype(float), weights


def _formed_by_the_eigensolver(monkeypatch):
    """Make the curvature policy's formed decision keep the alias, through the eigensolver.

    Whether rounding leaves the formed alias above the cut, and to the
    Cholesky shortcut or the eigensolver, depends on the BLAS build.  So the
    formed decision sees the matrix plus ``1e-13`` of its diagonal, far inside
    the formation bound (about ``1e-11`` here), as formation error would put
    it; ``allow_indefinite`` skips only the shortcut and the refusal of a
    materially indefinite matrix, which this fixture never meets.  The
    certificate decides the matrix as formed.
    """
    formed = curvature_module._decompose_gram

    def eigensolver(matrix, **kwargs):
        kwargs.pop("spectrum", None)
        lifted = matrix + 1e-13 * np.diag(np.diag(matrix))
        return formed(lifted, allow_indefinite=True, **kwargs)

    monkeypatch.setattr(curvature_module, "_decompose_gram", eigensolver)


@pytest.mark.parametrize("constant", [True, False], ids=["constant_shortcut", "uniform_eigh"])
def test_exact_alias_is_certified_rank_deficient(terminal, monkeypatch, constant):
    decisions, calls = terminal
    if not constant:
        _formed_by_the_eigensolver(monkeypatch)
    model, frame, y, weights = _alias(constant)
    result = _fit(model, frame, y, weights)
    assert result.terminal_rank.rank == 2
    assert result.terminal_curvature.certified is True
    assert result.terminal_curvature.grey_k == 1
    assert calls == [1]
    standard_errors = np.sqrt(np.diag(np.asarray(model.covariance_)))
    assert standard_errors[0] == pytest.approx(0.008, abs=4e-4)
    # The formed decision keeps the alias, through the shortcut on constant weights.
    formed = _main(decisions, 3)[0].decomposition
    assert formed.rank == 3
    assert formed.method == ("cholesky" if constant else "gram_eigh")

    _uncertified(monkeypatch)
    model, frame, y, weights = _alias(constant)
    assert _fit(model, frame, y, weights).terminal_rank.rank == 3


def test_well_conditioned_fit_reads_no_rows(terminal):
    decisions, calls = terminal
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abc"), n)})
    y = 1.0 + 0.5 * frame.x.to_numpy() + rng.normal(size=n)
    for discrete in (True, False):
        family = GaussianLS()
        model = SuperLSS(
            family,
            family.location(term("x", Numeric()), cat("g")),
            family.scale(),
            discrete=discrete,
        )
        result = _fit(model, frame, y)
        assert result.terminal_curvature.certified is True
        assert result.terminal_curvature.grey_k == 0
    assert calls == []
    assert {formation.route for _, formation, _, _ in decisions} == {"dense", "chunked"}


def _rows_and_design(record):
    """The certificate's own channels and the exact design values of a fit with dense columns."""
    _, _, kwargs, _ = record
    layout = kwargs["layout"]
    channels = np.vstack([np.asarray(block.curvature) for block in kwargs["rows"]()])
    matrices = []
    for state in layout.predictors:
        columns = [np.ones(len(channels))] if state.intercept_index is not None else []
        columns += [
            group.M[:, j] for group in state.design.group_matrices for j in range(group.shape[1])
        ]
        matrices.append(np.column_stack(columns))
    return layout, channels, matrices


def _exact_inverse_diagonal(layout, channels, matrices, penalty):
    """``diag(H^-1)`` of the exact sum of the float rows' products (Gauss-Jordan in Fractions)."""
    width = layout.n_coefficients
    exact = [[Fraction(float(penalty[i, j])) for j in range(width)] for i in range(width)]
    slices = [state.coefficient_slice for state in layout.predictors]
    for channel, (k, j) in enumerate(packed_pairs(len(matrices))):
        weights = [Fraction(float(value)) for value in channels[:, channel]]
        left = [[Fraction(float(v)) for v in column] for column in matrices[k].T]
        right = (
            left if k == j else [[Fraction(float(v)) for v in column] for column in matrices[j].T]
        )
        for a, xa in enumerate(left):
            for b, xb in enumerate(right):
                total = sum(
                    (p * w * q for p, w, q in zip(xa, weights, xb, strict=True)), Fraction(0)
                )
                exact[slices[k].start + a][slices[j].start + b] += total
                if k != j:
                    exact[slices[j].start + b][slices[k].start + a] += total
    augmented = [row + [Fraction(int(i == c)) for c in range(width)] for i, row in enumerate(exact)]
    for column in range(width):
        pivot = next(r for r in range(column, width) if augmented[r][column] != 0)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = 1 / augmented[column][column]
        augmented[column] = [value * scale for value in augmented[column]]
        for r in range(width):
            if r != column and augmented[r][column] != 0:
                factor = augmented[r][column]
                augmented[r] = [
                    a - factor * b for a, b in zip(augmented[r], augmented[column], strict=True)
                ]
    return np.array([float(augmented[i][width + i]) for i in range(width)])


def _near_pair(discrete):
    rng = np.random.default_rng(12345)
    n = 3000
    x1 = rng.standard_normal(n)
    x2 = x1 + 1e-7 * rng.standard_normal(n)
    y = 1.0 + 0.5 * x1 + rng.standard_normal(n)
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(
        family,
        family.location(term("x1", Numeric()), term("x2", Numeric())),
        family.scale(),
        discrete=discrete,
    )
    return model, pd.DataFrame({"x1": x1, "x2": x2}), y, rng.uniform(0.1, 2, n)


@pytest.mark.parametrize("discrete", [True, False], ids=["chunked", "dense"])
def test_near_pair_keeps_rank_with_exact_variance(terminal, monkeypatch, discrete):
    decisions, calls = terminal
    # The dense route's row pass streams blocks of rows; force several.
    monkeypatch.setattr(certificate, "_DENSE_BLOCK_ROWS", 700)
    model, frame, y, weights = _near_pair(discrete)
    result = _fit(model, frame, y, weights)
    assert result.terminal_rank.rank == 4
    assert result.terminal_curvature.certified is True
    assert result.terminal_curvature.grey_k == 1 and calls == [1]
    record = _main(decisions, 4)
    if not discrete:
        assert [block.stop - block.start for block in record[2]["rows"]()] == [700] * 4 + [200]
    layout, channels, matrices = _rows_and_design(record)
    exact = _exact_inverse_diagonal(layout, channels, matrices, np.asarray(result.penalty))
    published = np.diag(np.asarray(result.terminal_pseudo_inverse()))
    np.testing.assert_allclose(published, exact, rtol=1e-8, atol=0.0)
    # The formed matrix's own inverse misses the pair's variance by far more.
    formed = np.diag(np.asarray(record[0].decomposition.pseudo_inverse()))
    assert np.max(np.abs(formed / exact - 1.0)) > 1e-6

    _uncertified(monkeypatch)
    model, frame, y, weights = _near_pair(discrete)
    unfixed = np.diag(np.asarray(_fit(model, frame, y, weights).terminal_pseudo_inverse()))
    assert np.max(np.abs(unfixed / exact - 1.0)) > 1e-6


def test_near_pair_edf_follows_the_certified_curvature(terminal):
    """Four unpenalized coefficients have EDF 4 when the rows decide the near-null direction.

    Orthogonal +-1 Walsh columns: ``x``, ``x + 2**-23 e`` and the intercept,
    with ``y`` orthogonal to all three, so the residual scale is 1 and each
    slope's standard error is ``sqrt(1 + d**2) / (16 d)``, ``d = 2**-23``,
    which is ``2**19`` to float64.  Master publishes 741,455 there and EDF 4.
    The certificate publishes the exact error, and the influence of its own
    decomposition keeps EDF at 4 (``inference._influence``).
    """
    _, calls = terminal
    index = np.arange(256)
    x = 1.0 - 2.0 * (index % 2)
    e = 1.0 - 2.0 * ((index // 2) % 2)
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(
        family,
        family.location(term("x1", Numeric()), term("x2", Numeric())),
        family.scale(),
        discrete=False,
    )
    result = _fit(
        model, pd.DataFrame({"x1": x, "x2": x + 2.0**-23 * e}), 1.0 - 2.0 * ((index // 4) % 2)
    )
    assert result.terminal_curvature.grey_k == 1 and calls == [1]
    assert result.terminal_rank.rank == 4
    assert result.terminal_rank.grey_subspace["retained"] == 1.0
    assert model.result_.total_effective_df == pytest.approx(4.0, abs=1e-9)
    standard_errors = np.sqrt(np.diag(np.asarray(model.covariance_)))
    np.testing.assert_allclose(standard_errors[1:3], 2.0**19, rtol=1e-9)


def test_offset_column_keeps_its_rank(terminal):
    _, calls = terminal
    rng = np.random.default_rng(12345)
    n = 3000
    z = rng.standard_normal(n)
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(family, family.location(term("x", Numeric())), family.scale(), discrete=True)
    result = _fit(
        model,
        pd.DataFrame({"x": 1e6 + z}),
        1.0 + 0.5 * z + rng.standard_normal(n),
        rng.uniform(0.1, 2, n),
    )
    # The slope's eigenvalue (about 3e-13) sits inside the formation bound.
    assert result.terminal_curvature.grey_k == 1 and calls == [1]
    assert result.terminal_rank.rank == 3
    assert result.terminal_curvature.certified is True
    assert result.terminal_rank.grey_subspace["retained"] == 1.0


def test_formation_statistics_on_negative_diagonal_channels(terminal):
    decisions, _ = terminal
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + x)
    y = rng.negative_binomial(1.5, 1.5 / (1.5 + mu)).astype(float)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(term("x", Numeric())), family.theta(), discrete=False)
    result = _fit(model, pd.DataFrame({"x": x}), y)
    record = _main(decisions, 3)
    _, formation, kwargs, _ = record
    layout, channels, matrices = _rows_and_design(record)
    diagonal = [packed_pairs(2).index((k, k)) for k in range(2)]
    # Forced: the theta channel of the observed curvature is negative on some rows.
    assert np.any(channels[:, diagonal[1]] < 0.0) and not np.any(channels[:, diagonal[0]] < 0.0)
    # The row weights dominate each row's absolute block on nonnegative vectors.
    cross = [(index, k, j) for index, (k, j) in enumerate(packed_pairs(2)) if k != j]
    weights = _formation.channel_weights(channels, diagonal, cross)
    probe = np.random.default_rng(5).uniform(0.0, 1.0, (len(channels), 2))
    absolute = np.abs(channels)
    quadratic = (
        absolute[:, diagonal[0]] * probe[:, 0] ** 2
        + absolute[:, diagonal[1]] * probe[:, 1] ** 2
        + 2.0 * absolute[:, cross[0][0]] * probe[:, 0] * probe[:, 1]
    )
    assert np.any(
        absolute[:, cross[0][0]] ** 2 > absolute[:, diagonal[0]] * absolute[:, diagonal[1]]
    )
    assert np.all(
        quadratic <= (weights[0] * probe[:, 0] ** 2 + weights[1] * probe[:, 1] ** 2) * (1 + 1e-12)
    )
    # Rescaling a predictor by a power of two rescales its own weight exactly
    # and leaves the other's unchanged: each weight is |c_kk| (1 + kappa).
    scaled = channels.copy()
    scaled[:, diagonal[0]] *= 2.0**40
    scaled[:, cross[0][0]] *= 2.0**20
    rescaled = _formation.channel_weights(scaled, diagonal, cross)
    np.testing.assert_array_equal(rescaled[0], weights[0] * 2.0**40)
    np.testing.assert_array_equal(rescaled[1], weights[1])
    for state, matrix, index in zip(layout.predictors, matrices, diagonal, strict=True):
        floor = (matrix * matrix).T @ np.abs(channels[:, index])
        assert np.all(formation.majorant[state.coefficient_slice] >= floor * (1.0 - 1e-12))
    assert result.terminal_curvature.certified is True
    bound = certificate.formation_bound(
        formation,
        kwargs["penalized"],
        kwargs["penalty"],
        [state.coefficient_slice for state in layout.predictors],
    )
    assert 0.0 < bound < 1e-9


def _penalty_names(model, frame, y):
    """The layout's penalty names, from the error a fit without smoothing parameters raises."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.iloc[:400], y[:400])
    except ValueError as exc:
        names = str(exc).split("missing penalty lambda names: ")[1]
        return [name.strip(" '[]") for name in names.split(",")]
    raise AssertionError("a fit without smoothing parameters must name them")


def test_row_pass_agrees_with_the_rows_on_every_group_type(terminal, monkeypatch):
    """``project`` on support, exact-spline, tensor, by-level, categorical, random-effect and dense groups."""
    decisions, _ = terminal
    rng = np.random.default_rng(3)
    n = 2000
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.gamma(2, 1, n),
            "x3": rng.normal(0, 1, n),
            "x4": rng.normal(0, 1, n),
            "g": rng.choice(list("abc"), n),
            "h": rng.choice([f"l{i}" for i in range(12)], n),
        }
    )
    y = np.sin(3 * frame.x1) + 0.2 * frame.x2 + rng.normal(0, 0.5, n)
    family = GaussianLS()
    location = family.location(
        s("x1"),
        s("x2", discrete=False),
        s("x3"),
        cat("g"),
        interaction(SplineCategorical("x3", "g")),
        ti("x1", "x3"),
        term("x4", Numeric()),
        re("h"),
    )
    model = SuperLSS(family, location, family.scale(s("x1")), discrete=True)
    lambdas = {name: 10.0 for name in _penalty_names(model, frame, y.to_numpy())}
    decisions.clear()
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 64 * 1024)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y.to_numpy(), lambdas=lambdas)
    width = max(r[2]["layout"].n_coefficients for r in decisions)
    _, formation, kwargs, _ = _main(decisions, width)
    layout = kwargs["layout"]
    assert formation.route == "chunked"
    blocks = list(kwargs["rows"]())
    # The row pass replays the assembly's many small likelihood chunks.
    assert len(blocks) > 10
    channels = np.vstack([np.asarray(block.curvature) for block in blocks])
    matrices = [
        np.column_stack(
            [np.ones(n)] * (state.intercept_index is not None) + [state.design.toarray()]
        )
        for state in layout.predictors
    ]
    pairs = packed_pairs(len(matrices))
    data, magnitude = np.zeros((width, width)), np.zeros((width, width))
    for channel, (k, j) in enumerate(pairs):
        left, right = layout.predictors[k].coefficient_slice, layout.predictors[j].coefficient_slice
        weights = channels[:, channel][:, None]
        block = matrices[k].T @ (weights * matrices[j])
        absolute = np.abs(matrices[k]).T @ (np.abs(weights) * np.abs(matrices[j]))
        data[left, right] += block
        magnitude[left, right] += absolute
        if k != j:
            data[right, left] += block.T
            magnitude[right, left] += absolute.T
    for k, state in enumerate(layout.predictors):
        floor = (matrices[k] ** 2).T @ np.abs(channels[:, pairs.index((k, k))])
        assert np.all(formation.majorant[state.coefficient_slice] >= floor * (1.0 - 1e-12))
    # The statistics each group closes per assembly (Cauchy-Schwarz on factored
    # supports) dominate the squares of the absolute design formed row by row.
    diagonal = [pairs.index((k, k)) for k in range(len(matrices))]
    cross = [(index, k, j) for index, (k, j) in enumerate(pairs) if k != j]
    weights = _formation.channel_weights(channels, diagonal, cross)
    design = _formation.AbsDesign(layout)
    for k, state in enumerate(layout.predictors):
        columns = state.coefficient_slice
        unit = np.zeros((width, columns.stop - columns.start))
        unit[columns] = np.eye(columns.stop - columns.start)
        squares = (design.matvec(k, 0, n, unit) ** 2).T @ weights[k]
        assert np.all(formation.majorant[columns] >= squares * (1.0 - 1e-12))
    penalty = np.asarray(kwargs["penalty"])
    exact = data + penalty
    spline = next(name for name in layout.penalty_names if name.endswith("x1#wiggle"))
    for face in (None, build_penalty_face(layout, (spline,))):
        null = np.eye(width) if face is None else face.null_basis
        basis = rng.standard_normal((null.shape[1], 3))
        projection = certificate.project_terminal_curvature(
            basis,
            kwargs["rows"](),
            layout=layout,
            penalty=penalty,
            formation=formation,
            face=face,
            product_roundings=certificate.product_roundings(layout, formation),
        )
        lifted = null @ basis
        reference_block = lifted.T @ exact @ lifted
        reference_cross = null.T @ (exact @ lifted)
        # The float reference's own error: materialised rows, a product of
        # length n, the lift and the reduction (Higham 2002, eq. 3.13).
        reference_error = 2.0 * (2 * n + 4 * width) * np.finfo(float).eps
        absolute = magnitude + np.abs(penalty)
        lifted_magnitude = np.abs(null) @ np.abs(basis)
        block_error = reference_error * (lifted_magnitude.T @ absolute @ lifted_magnitude)
        cross_error = reference_error * (np.abs(null).T @ (absolute @ lifted_magnitude))
        assert np.all(
            np.abs(projection.block - reference_block) <= projection.block_bound + block_error
        )
        assert np.all(
            np.abs(projection.cross - reference_cross) <= projection.cross_bound + cross_error
        )
        assert np.max(projection.block_bound) < 1e-7 * np.max(np.abs(reference_block))
        # And at least the first-order term of Higham's bound on the row sums of T' C T.
        rows_product = [
            np.abs(matrix @ lifted[state.coefficient_slice])
            for matrix, state in zip(matrices, layout.predictors, strict=True)
        ]
        first_order = sum(
            rows_product[k].T @ (np.abs(channels[:, c])[:, None] * rows_product[j])
            for c, (k, j) in enumerate(pairs)
        )
        assert np.all(projection.block_bound >= 0.25 * n * np.finfo(float).eps * first_order)
        reduced = exact if face is None else face.reduce_matrix(np.asarray(kwargs["penalized"]))
        bound = certificate.formation_bound(
            formation,
            kwargs["penalized"],
            penalty,
            [state.coefficient_slice for state in layout.predictors],
            face=face,
            reduced=reduced,
        )
        assert 0.0 < bound < 1e-6


def _placed(offsets, values, width):
    rows = np.zeros((len(offsets), width))
    columns = np.asarray(offsets)[:, None] + np.arange(values.shape[1])
    rows[np.arange(len(offsets))[:, None], columns] = np.abs(values)
    return rows


@pytest.mark.parametrize("kind", ["cr", "ps"])
def test_factored_statistics_match_cauchy_schwarz_row_by_row(kind):
    """Each factored support closes its bin sums to ``sum_r w_r |a_r|_1 (a_r (A o A))``."""
    from superglm._frame import as_eager_frame
    from superglm.distributional.layout import build_stacked_layout
    from superglm.distributional.predictor import Predictor, compile_predictors
    from superglm.features import Spline

    from ._distributional_weights import resolved_prior

    rng = np.random.default_rng(6)
    n = 500
    frame = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(0, 1, n)})
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"a": Spline(kind=kind, k=12), "b": Spline(kind=kind, k=12)},
                interactions=(("a", "b"),),
            ),
            Predictor("scale", {}),
        ),
        model_discrete=True,
        n_bins_config=40,
    )
    groups = build_stacked_layout(compiled).predictors[0].design.group_matrices
    tensor = next(g for g in groups if type(g).__name__ == "DiscretizedTensorGroupMatrix")
    # Forced: P-spline margins carry a raw band, cubic regression ones do not.
    assert (tensor._current_raw_channels() is None) == (kind == "cr")
    weights = rng.uniform(0.1, 2.0, n)
    for group in groups:
        absolute = _formation.abs_group(group)
        transform = np.abs(np.asarray(group.R_inv))
        scale = 1.0
        if group is tensor:
            band = group._current_raw_channels()
            if band is None:
                first = np.abs(np.asarray(group.B1_unique_t))[group.idx1]
                second = np.abs(np.asarray(group.B2_unique_t))[group.idx2]
            else:
                projection = np.abs(np.asarray(band.projection))
                width2 = int(band.k2_raw)
                width1 = projection.shape[0] // width2
                first = _placed(band.offsets1, band.values1, width1)[group.idx1]
                second = _placed(band.offsets2, band.values2, width2)[group.idx2]
                transform = projection @ transform
            rows = np.einsum("ra,rb->rab", first, second).reshape(n, -1)
            scale = 1.0 + 16.0 * 2.0**-53
        else:
            rows = np.abs(np.asarray(group.B_unique))[group.bin_idx]
        expected = ((weights * rows.sum(axis=1)) @ rows) @ (transform * transform) * scale
        got = absolute.square_close(absolute.square_partial(0, n, weights))
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_tensor_row_pass_forms_no_bin_grid():
    """A 512-by-512-bin tensor's row-pass products stay at its rows' and margins' size.

    The whole grid's absolute rows would be ``512**2`` rows of 16 columns
    (32 MiB); the margins and a 4,000-row block are well under 4 MiB.
    """
    import tracemalloc

    from superglm._frame import as_eager_frame
    from superglm.distributional.layout import build_stacked_layout
    from superglm.distributional.predictor import Predictor, compile_predictors
    from superglm.features import Spline

    from ._distributional_weights import resolved_prior

    rng = np.random.default_rng(7)
    n = 20000
    frame = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"a": Spline(kind="cr", k=5), "b": Spline(kind="cr", k=5)},
                interactions=(("a", "b"),),
            ),
            Predictor("scale", {}),
        ),
        model_discrete=True,
        n_bins_config=512,
    )
    groups = build_stacked_layout(compiled).predictors[0].design.group_matrices
    tensor = next(g for g in groups if type(g).__name__ == "DiscretizedTensorGroupMatrix")
    assert tensor.n_bins1 * tensor.n_bins2 * tensor.shape[1] * 8 >= 32 * 2**20
    absolute = _formation.abs_group(tensor)
    values = rng.uniform(0, 1, (tensor.shape[1], 2))
    rows = rng.uniform(0, 1, (4000, 2))
    tracemalloc.start()
    try:
        products = absolute.matvec(1000, 5000, values)
        transposed = absolute.rmatvec(1000, 5000, rows)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 2**20
    # Against the block's own absolute rows, formed from the margins.
    first = np.abs(np.asarray(tensor.B1_unique_t))[tensor.idx1[1000:5000]]
    second = np.abs(np.asarray(tensor.B2_unique_t))[tensor.idx2[1000:5000]]
    block = np.einsum("ra,rb->rab", first, second).reshape(4000, -1) @ np.abs(tensor.R_inv)
    np.testing.assert_allclose(products, block @ values, rtol=1e-12)
    np.testing.assert_allclose(transposed, block.T @ rows, rtol=1e-12)


def _exact_and_direct_groups():
    """Exact spline and spline-by-category groups from a compiled layout, and direct sparse and SCOP ones."""
    import scipy.sparse as sp

    from superglm._frame import as_eager_frame
    from superglm._group_matrix._group_matrix_core import SparseGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import DiscretizedSCOPGroupMatrix
    from superglm.distributional.layout import build_stacked_layout
    from superglm.distributional.predictor import Predictor, compile_predictors
    from superglm.features import Categorical, Spline

    from ._distributional_weights import resolved_prior

    rng = np.random.default_rng(8)
    n = 400
    frame = pd.DataFrame({"a": rng.uniform(0, 1, n), "g": rng.choice(list("abc"), n)})
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"a": Spline(k=8, discrete=False), "g": Categorical()},
                interactions=(("a", "g"),),
            ),
            Predictor("scale", {}),
        ),
        model_discrete=False,
    )
    groups = list(build_stacked_layout(compiled).predictors[0].design.group_matrices)
    sparse = sp.random(n, 6, density=0.3, random_state=9, format="csr") - sp.random(
        n, 6, density=0.2, random_state=10, format="csr"
    )
    groups.append(SparseGroupMatrix(sp.csr_matrix(sparse)))
    groups.append(DiscretizedSCOPGroupMatrix(rng.normal(size=(30, 5)), rng.integers(0, 30, n)))
    return n, groups


def test_absolute_designs_dominate_exact_sparse_and_scop_groups():
    """Each group's absolute design and squared statistic dominate its rows, on row ranges too."""
    n, groups = _exact_and_direct_groups()
    kinds = [type(group).__name__ for group in groups]
    for kind in ("SparseSSPGroupMatrix", "SplineCategoricalGroupMatrix", "SparseGroupMatrix"):
        assert kind in kinds
    assert "DiscretizedSCOPGroupMatrix" in kinds
    rng = np.random.default_rng(11)
    weights = rng.uniform(0.1, 2.0, n)
    for group in groups:
        absolute = _formation.abs_group(group)
        assert absolute is not None, type(group).__name__
        rows = np.abs(np.asarray(group.toarray()))
        start, stop = 50, 330
        values = rng.uniform(0.0, 1.0, (group.shape[1], 2))
        products = rng.uniform(0.0, 1.0, (stop - start, 2))
        lower = 1.0 - 1e-12
        # The squared statistic over two chunks, closed once.
        partial = absolute.square_partial(0, start, weights[:start])
        partial = partial + absolute.square_partial(start, n, weights[start:])
        squares = (rows * rows).T @ weights
        assert np.all(absolute.square_close(partial) >= squares * lower), type(group).__name__
        assert np.all(absolute.matvec(start, stop, values) >= rows[start:stop] @ values * lower)
        assert np.all(
            absolute.rmatvec(start, stop, products) >= rows[start:stop].T @ products * lower
        )
        if type(group).__name__ not in ("SparseSSPGroupMatrix", "SplineCategoricalGroupMatrix"):
            # No factor chain: the absolute design is |X| itself.
            np.testing.assert_allclose(absolute.square_close(partial), squares, rtol=1e-12)
            np.testing.assert_allclose(
                absolute.matvec(start, stop, values), rows[start:stop] @ values, rtol=1e-12
            )
        else:
            # |B| |R_inv| on the group's own rows, through Cauchy-Schwarz for the statistic.
            basis = np.abs(group.B.toarray())
            if hasattr(group, "row_idx"):
                mask = np.zeros(n, dtype=bool)
                mask[np.asarray(group.row_idx)] = True
                basis = basis * mask[:, None]
            transform = np.abs(np.asarray(group.R_inv))
            expected = ((weights * basis.sum(axis=1)) @ basis) @ (transform * transform)
            np.testing.assert_allclose(absolute.square_close(partial), expected, rtol=1e-12)
            np.testing.assert_allclose(
                absolute.matvec(start, stop, values),
                basis[start:stop] @ transform @ values,
                rtol=1e-12,
            )
