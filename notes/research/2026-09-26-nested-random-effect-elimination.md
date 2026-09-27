# Nested random-effect elimination (fix D): mathematics, contracts and spec

Date: 2026-09-27. Branch `claude/superglm-exotic-types-17abdd` (rebased on
`origin/master` 9e0fe6f9; fix A committed as 36b330a0). No production source
was changed for this note. Companion artifacts: the Lean contracts in
[lean-gaussian-certificate/NestedElimination.lean](lean-gaussian-certificate/NestedElimination.lean),
and the scratch prototype and verification (`nested.py`, `exact.py`,
`verify.py`, `scale.py`) in this session's scratchpad under
`wf2/nested_elimination_math_max/`; the `verify_run*.log` and `scale_*.log`
files there are the builder's runs. Revised on 2026-09-27 after review: the
revision's checks are in `wf2/nested_elimination_revise_max/` (`verify2.py`,
`verify2_run2.log`, `critic_*.log` re-runs, `lake_build.log`) and the review's
own scratch checks (`crit*.py`, `floors.py`, `fallback.py`, `lam0.py`,
`wsign.py`, `decision1.py`) in `wf2/nested_spec_critic_xhigh/`; every number
quoted below names its run.

## 1. The problem in plain language

A model with several random effects that nest inside each other (a vehicle
`variant` sits inside one `make_model`, which sits inside one `make`) is fitted
today by eliminating only the largest of them as a diagonal block. Every other
random effect joins the dense border, so the border is as wide as the parent
levels are numerous. Each factorization then costs a dense border cube, and
every REML Hessian entry between two border penalties costs a border cube
again.

Measured (probes of 2026-09-26, threads pinned, exact path unless stated):

- pg17, step C (splines + `RandomEffect(vh_make)` + `RandomEffect(make_model)`,
  Poisson, 77k training rows): 743 s, of which 716 s in the REML Newton
  Hessian. The Hessian calls the factor's cross-trace methods for every pair of
  smoothing parameters, and each call rebuilds full `r x r` low-rank products
  just to take a trace. This is row-count independent: at 15k rows the same
  step still takes over 300 s. `discrete=True` finishes in 4.7 s
  (`structured`, 16 REML iterations, holdout deviance 0.5722) because it avoids
  most of the exact-path Hessian work, but it calls the same Hessian.
- DVSA MOT, step D (`make` / `make_model` / `variant`, binomial): at 5,000
  rows `auto` declines the structured backend (p=1946, K=1192, q=754, cost
  ratio 0.150) and the dense fit takes 56.7 s; forcing `structured` takes
  24.8 s; at 20,000 rows forced `structured` takes 117.8 s; at 200,000 rows
  (train 157,593 rows, levels 223 / 3,749 / 6,038) the forced structured fit
  did not finish in 300 s, before or after fix A. The border there is about 4k
  columns; at full data (7,294 / 38,366 / 64,761 levels) it would be about 45k.

The fix is the standard one: eliminate the whole nested chain, leaves first,
beside a dense border that holds only the ordinary columns (and any crossed
random effect). The chain creates no fill, every pivot has a closed form, and
the border Schur complement can be assembled as a sum of positive
semi-definite pieces so that nothing large is subtracted from anything large.
At full DVSA size the prototype builds the whole factor in 0.1 s (section 8).

## 2. The standard method and what it assumes

- lme4 (Bates, "lme4: Mixed-effects modeling with R", ch. 2, §2.2; Bates,
  Mächler, Bolker and Walker, JSS 67(1) 2015, arXiv 1406.5823): nested
  grouping factors give a sparse Cholesky factor with no fill-in after
  reordering, terms are ordered by descending number of levels, and the fixed
  effects form a dense trailing block (their `R_ZX`, `R_X`), which is our
  border. lme4 warns that implicitly nested labels (a level name reused under
  several parents) must be turned into explicit interaction codes first.
- Papaspiliopoulos, Stumpf-Fétizon and Zanella (arXiv 2103.10875, Prop. 3):
  for nested multilevel models under a depth-last ordering the Cholesky factor
  has the sparsity of the precision matrix and costs O(p); this is belief
  propagation on the tree. The proposition is stated for a Gaussian likelihood
  in the hierarchically centred parameterisation (block-tridiagonal in the
  tree) with no dense fixed-effect border; the bordered no-fill argument of
  section 3.2 is this note's own derivation.
- Selected inversion (Takahashi–Fagan–Chin; Erisman–Tinney 1975; Lin et al.
  2011; survey in Zhu and Wathen, arXiv 1911.00685): the entries of the inverse
  on the sparsity pattern of the factor follow from the LDLᵀ factors at about
  twice the factorization cost. Andersen, Dahl and Vandenberghe (arXiv
  1203.2742) give the log-det gradient and Hessian projections on a chordal
  pattern at the same order; an arrow pattern costs O(n·w²).
- Wood and Fasiolo (arXiv 1606.04802): the generalized Fellner–Schall update
  needs only `tr(H⁻¹S_j)`, no cross traces.
- SAS `PROC GLM ABSORB` documentation: absorbed effects must each be nested in
  the preceding one and do not enlarge the matrix; no inverse elements.

Assumptions and whether we meet them: no-fill is purely structural and holds
for any working weights; the cancellation-free pivot recursion and the
guarantee `pivot >= λ` need non-negative working weights; Gaussianity is not
needed (our H is X'WX + S with diagonal W). Crossed factors have no zero-fill
ordering (Gao and Owen; Ghosh, Hastie and Owen, as reviewed in arXiv
2103.10875) and stay in the border. Provenance: the literature report of
2026-09-26 read lme4 chapter 2, arXiv 1406.5823, 2103.10875 (Prop. 3 and Def.
4), 1911.00685, 1203.2742 (§4.3.1 and Algorithm 5.1), 1606.04802, 1509.00660
and the SAS ABSORB documentation directly; Erisman and Tinney (1975), Lin et
al. (2011), Takahashi–Fagan–Chin, Gao and Owen, Ghosh, Hastie and Owen and
Longford (1987) are known only through those papers' citations. Nothing in the
derivation below is taken from GPL source.

## 3. Derived algebra

### 3.1 Objects

Rows `r` carry a leaf code `ℓ(r)` at the finest chain level. Node `u` of the
tree has parent `p(u)` at the next coarser level; roots have none. Write
`anc*(u)` for the ancestors of `u` including `u`, `M` for the leaf-to-node
incidence (`M_{ℓu} = 1` iff `u ∈ anc*(ℓ)`), `Z = Z_leaf M` for the tree design,
`X_b` for the border design, `w ≥ 0` the working weights, `Λ = diag(λ_u)` the
per-level ridge and `S_b` the border penalty. After the intercept augmentation
that `assembly.build_augmented_scalar_factor` performs today (the intercept is
border column 0 with `A_00 = Σw`), the penalized Hessian is

    H = [[T, C], [C', A]],  T = M' diag(w_leaf) M + Λ,  C = M' C_leaf,  A = X_b'WX_b + S_b,

with the leaf statistics `w_ℓ = Σ_{r∈ℓ} w_r` and `C_leaf,ℓ = Σ_{r∈ℓ} w_r x_r`
that the existing leaf kernels already produce. `T_{uv}` is the total weight of
the subtree below the deeper of `u, v` when they are comparable, else zero.

### 3.2 No fill (theorem)

The support of `T` is the ancestor closure of the tree, a chordal graph
(comparability graph of a forest). In any ordering that eliminates every node
after all its descendants, the not-yet-eliminated neighbours of a node are its
strict ancestors, which are pairwise comparable, plus the border. Outer-product
elimination of a pivot changes only entries `(i, j)` with both `i` and `j`
neighbours of the pivot, so no entry appears outside the pattern
(ancestor pairs, tree x border, border x border). Lean: `eliminate_support`,
`eliminate_no_fill`, `comparable_of_support`, `ancestors_pairwise_comparable`,
`nested_leaf_pivot_no_fill` (section 7). Measured: zero fill entries in the
leaves-first order on every fixture; a crossed factor eliminated first fills
the tree (1,585 entries on the literature report's fixture) and in the border
costs only its own pairs.

### 3.3 Pivot recursion

Eliminate leaves first (level `L-1` down to level 0). Start with
`ω_ℓ = w_ℓ`, `c̃_ℓ = C_leaf,ℓ`, and at every node

    D_u = ω_u + λ_u,   ρ_u = λ_u / D_u,   σ_u = ω_u / D_u (= 1 − ρ_u),   s_u = ω_u ρ_u,
    ω_{p} = Σ_{c ∈ children(p)} s_c,     c̃_{p} = Σ_{c ∈ children(p)} ρ_c c̃_c.

`D_u` is the pivot of `u`, the multiplier from `u` to every strict ancestor is
`σ_u`, and the multiplier to the border is `c̃_u / D_u`. The factorization
`T = L D L'` with `L_{a,u} = σ_u` for `a` a strict ancestor of `u` is proved
from these local equations and the forest structure (Lean `chain_ldl`; its
load-bearing lemma `subtree_telescope` is the invariant that, once a node's
descendants are eliminated, its column restricted to its ancestors is the
constant `ω_u`), with `det L = 1` (`chainL_det`) and `det T = Π_u D_u`
(`chain_det`). Then `c̃ = L⁻¹C`, `F := T⁻¹C = L⁻ᵀ D⁻¹ c̃`, and with
`det_bordered_schur`

    log|H| = Σ_u log D_u + log|Q|,   Q = A − Σ_u c̃_u c̃_u' / D_u.

Every `D_u ≥ λ_u` because `ω_u` is a sum of non-negative terms (the scalar
`star_pivot_ge`; `star_pivot` is the scalar identity between the two forms of
one pivot, not an elimination statement). The partial-minimization view: with
`S_u = Σ_{a∈anc*(u)} β_a` the cumulative effect, minimizing `w_c (β_c + S_p)² +
λ_c β_c²` over `β_c` leaves `s_c S_p²` plus border cross terms, which is the
parent's leaf-like form with `ω_p = Σ s_c`.

Compute `σ_u` as `ω_u / D_u`, never as `1 − ρ_u` (exact when `ρ ≈ 1`).

`F` and the leaf path sums `MF` by the top-down `g/e` recursion on the node
means `m_u = c̃_u / ω_u` of section 3.4, with `d_u = m_u − m_{p(u)}` (roots:
`d_u = m_u`):

    g_u = d_u + e_{p(u)} (e = 0 above the roots),   F_u = σ_u g_u,   e_u = ρ_u g_u,
    (MF)_ℓ = Σ_{a ∈ anc*(ℓ)} F_a = m_ℓ − e_ℓ.

Proof by induction down the tree with `S_u := Σ_{a∈anc*(u)} F_a = m_u − e_u`:
`F_u = c̃_u/D_u − σ_u S_{p(u)} = σ_u m_u − σ_u (m_p − e_p) = σ_u g_u`, and
`S_u = S_p + F_u = m_p − e_p + σ_u g_u = m_u − ρ_u g_u`. The back-substitution
form `F_u = c̃_u/D_u − σ_u A_u` (`A_u = Σ_{a∈anc(u)} F_a`) is the same number in
exact arithmetic but subtracts `σ_u A_u ≈ m_u` from `c̃_u/D_u ≈ m_u` when `ρ`
is small: its maximum entry error against the exact `F` is 5.0e-8 relative on
the mixed fixture against 3.0e-14 for the `g/e` form (critic run `crit7`), and
it fails the `diag(H⁻¹O)` bound of section 8 at ratio 4.0e+2 (mutation
table). Cost O(k q); memory k x q; `MF` needs no path sum.

### 3.4 The Schur complement as a sum of PSD terms

With `m_u = c̃_u / ω_u` (zero when `ω_u = 0`; for leaves `m_ℓ = x̄_ℓ`, the
weighted leaf mean of the border rows), computed as shifted means: at a leaf
`m_ℓ = x_ref(ℓ) + Σ_{r∈ℓ} w_r (x_r − x_ref(ℓ)) / w_ℓ` with `x_ref(ℓ)` one row
of the leaf, and at a parent `m_p = m_ref(p) + Σ_c s_c (m_c − m_ref(p)) / ω_p`
with `m_ref(p)` the mean of the child with the largest `s_c`, so that a column
constant within a node gives `d_c = m_c − m_p = 0` exactly instead of
`ε`-noise (see the end of this subsection),

    Q = S_b + Σ_r w_r (x_r − x̄_{ℓ(r)})(x_r − x̄_{ℓ(r)})'          (within-leaf scatter)
          + Σ_{u non-root} s_u (m_u − m_{p(u)})(m_u − m_{p(u)})'      (between-child scatter)
          + Σ_{u root} s_u m_u m_u'.                                  (roots)

Proof: the weighted-scatter identity `Σ w (x − m)(x − m)' = Σ w x x' − W m m'`
at every leaf and at every internal node (with weights `s_c`, means `m_c`),
plus `ω_u² / D_u = ω_u − s_u`, and the telescoping `Σ_u ω_u = Σ w + Σ_{non-root}
s_u`. Lean: `weighted_scatter`, `pivot_mass_split`, `star_psd_eq_subtraction`
(one star; the telescoping over an arbitrary tree is not formalized).

Why it matters: the subtraction form subtracts `Σ_u c̃_u c̃_u' / D_u ≈ Σ ω m m'`
(scaled by the raw weights) from `A` and keeps a residue scaled by the shrunk
weights `s ≈ λ`. Its rounding error is `ε` times the raw mass. For the
intercept column `m_u = 1` exactly at every node, the within and between terms
vanish exactly, and `Q_00 = Σ_{roots} s_u` exactly. Measured on the stress
fixture (λ = 1e-7, weights x1e4): subtraction form `Q_00` off by 2.4e-1
relative and `log|H|` off by 8.3e-2; PSD-sum `Q` entries within 3.7e-9 absolute
(bound 5.3e-6) and `log|H|` within 8.5e-14. The between-child and root terms
are products of shrunk quantities, but `m_c − m_p` on a column that is constant
within the parent (a root or leaf attribute) is `ε |m|` noise unless the means
are shifted as above, and `s_c` times that noise lands in a small `Q`
direction: the mixed fixture's `Q(3,4)` had a scaled error of 5.7e-13 with
plain means and 1.7e-16 with shifted means (critic runs `crit2`, `crit4`), and
plain means fail the `Q` bound of section 8 at ratio 30 (mixed fixture) and
42 (depth 4). The within-leaf term must be the centred row pass `Σ_r w_r (x_r
− m_ℓ)(x_r − m_ℓ)'`: the subtraction `X_b'WX_b − Σ_ℓ c_ℓ c_ℓ' / w_ℓ` puts
`ε Σw` back into every column that is constant within leaves, including the
intercept, so `Q_00 ≈ Σ_roots s_u` is lost exactly as in the full subtraction
form (λ = 1e-7, weights x1e4: `Q_00` relative error 3.3e-2 and `log|H|` error
1.8e-1 against 7.9e-16 and 1.6e-13 for the row pass; critic run `decision1`,
re-run here). Decision 1 is therefore settled by measurement, not open.

### 3.5 Inverse structure and the Takahashi scalars

    H⁻¹ = blockdiag(Z, 0) + U Q⁻¹ U',   Z = T⁻¹,   U = [−F; I].

The entries of `Z` on the pattern follow from three per-node scalars
(top-down, O(k)):

    t_u = Var(S_{p(u)}) = 1/D_p + ρ_p² t_p   (t = 0 at roots),
    Z_uu = 1/D_u + σ_u² t_u,     v_u = Var(S_u) = 1/D_u + ρ_u² t_u,     κ_u = Cov(β_u, S_u) = 1/D_u − ρ_u σ_u t_u,

and the path products `π_ℓ(x) = Π_{y ∈ (x, ℓ]} ρ_y`. Then for `x = lca(ℓ, ℓ')`,
`Cov(S_ℓ, S_ℓ') = v_x π_ℓ(x) π_ℓ'(x)`; `Cov(β_u, S_ℓ) = κ_u π_ℓ(u)` for `ℓ`
under `u` and `−σ_u π_{p(u)}(x) π_ℓ(x) v_x` otherwise. The general pattern
entries `Z_{u,a}` for `a` an ancestor of `u` come from the congruence
`Z = L⁻ᵀ D⁻¹ L⁻¹` with the chain vectors `(L⁻¹e_u)_{a_j} = −σ_u Π_{i<j} ρ_{a_i}`
(O(k d²) for depth d). All verified against the exact reference (section 8).

### 3.6 Protocol methods

Notation: `k` tree nodes, `K` leaves, `q` border width (intercept included),
`d` depth, `m` smoothing parameters. `MF = m_leaf − e_leaf` is the path sum of
`F` at the leaves (K x q), `E_I` the selector of chain level `I`, and a
"leaf-form" operator is `O = [[M' diag(a) M, M' C_O], [C_O' M, A_O]]` given by
row weights `a_r` (signed for the W-derivative operators). Three kinds of
operator reach the factor: leaf-form operators from a row pass; low-rank pieces
(`CenteredBlockOperator`, `LowRankSymmetricOperator`, sums), handled by solves;
and node-diagonal level selectors `λ_I E_I` on any chain level, which the
batched REML directions of `derivative_cross_traces` combine with the first two
and which are neither leaf-form nor low-rank. The selectors are handled by the
level-pattern (`h`, `e`) closed forms below; the earlier claim that every
operator is leaf-form plus low-rank was wrong for them.

Operator quantities are formed from the row pass about the factor's own leaf
means `m_ℓ` and from `e` (section 3.3), never as `A_O` minus path-sum products:

    a_ℓ = Σ_{r∈ℓ} a_r,   dev_ℓ = Σ_{r∈ℓ} a_r (x_r − m_ℓ),   W_a = Σ_r a_r (x_r − m_ℓ)(x_r − m_ℓ)',
    V = dev + a ⊙ e   (K x q),
    U'OU = W_a + dev'e + e'dev + Σ_ℓ a_ℓ e_ℓ e_ℓ'   (q x q, symmetric by construction).

These equal `V = C_O − a ⊙ MF` and `U'OU = A_O − (MF)'C_O − C_O'(MF) +
(MF)'diag(a)(MF)` (`U = [−F; I]`) in exact arithmetic: expand `x_r = m_ℓ + (x_r
− m_ℓ)` and `MF = m − e`, and every `m m'`, `m dev'` and `m e'` term cancels
exactly. The subtraction forms carry `ε` times the raw mass, which `Q⁻¹ ~ 1/λ`
amplifies: measured relative errors of `tr(H⁻¹O)` (subtraction / row pass) are
1.3e-2 / 0.0 at λ = 1e-7 with weights x1e4, 4.2e-8 / 2.7e-16 at λ = 1e-3 with
weights x1e2 and 4.6e-3 / 0 on the mixed fixture, worse than dense float64
Cholesky (3.1e-3, 4.4e-9, 7.3e-4); for `tr(H⁻¹O_lH⁻¹O_r)` 4.2e-3 / 3.4e-16 and
1.5e-4 / 2.1e-15 (critic run `crit3`; the mutation table of section 8 has the
same forms at ratios up to 8.8e+8). The earlier shorthand `A_O − 2 (MF)'C_O +
…` is not symmetric; inside a trace it is harmless, but reused in the fourth
cross-trace term it gives the wrong answer (relative error 5.0e+2 on the base
fixture, ratio 6.1e+13 against the bound). For the data operator (`a = w`)
`dev = 0` and `U'OU = W_in + Σ_ℓ w_ℓ e_ℓ e_ℓ'`.

| Method | Formula | Cost |
|---|---|---|
| `solve(r)` | `u = T⁻¹ r_t` (tree forward/diagonal/backward passes), `x_b = Q⁻¹(r_b − C_leaf' (M u)_leaf)`, `x_t = u − F x_b` | O(k q + q²) per column |
| `logdet()` | `Σ_u log D_u + log|Q|` | O(1) after build |
| `selected_inverse_diagonal` | tree `u`: `Z_uu + F_u Q⁻¹ F_u'`; border: `diag(Q⁻¹)` | O(k q²) once, cached |
| `selected_inverse_block(idx)` | columns `H⁻¹ e_i` by `solve` (cap 256 over every chain level; the caller consequence is in section 6) | O(|idx| (k q + q²)) |
| `trace_inverse_penalty(E_I)` | `Σ_{u∈I} (Z_uu + F_u Q⁻¹ F_u')`; border penalties through `Q⁻¹` blocks as today | O(K_I) after the diagonal |
| `penalty_cross_trace(E_I, E_J)` | `‖(H⁻¹)_{IJ}‖_F² = t1 + t2 + t3` with `t1 = ‖Z_{IJ}‖_F² = Σ_a (h^I_a + 1_I(a))(h^J_a + 1_J(a))/D_a² + 2 Σ_a e^I_a e^J_a t_a / D_a`, `h^I_a = Σ_{c} (1_I(c) σ_c² + ρ_c² h^I_c)` bottom-up, `e^I_a = ρ_a h^I_a − 1_I(a) σ_a`; `t2 = 2 tr(Q⁻¹ F_J' (T⁻¹E_I F)_J)`; `t3 = tr(Q⁻¹ G_J Q⁻¹ G_I)`, `G_I = F_I'F_I` | `t1` O(k); `t2` O(k q); `t3` O(q³) with `G_I` cached per level |
| level x border, border x border | `tr((Q⁻¹ G_I Q⁻¹)_{JJ} Ω_J)`, `Q⁻¹` blocks | O(q³) |
| `trace_inverse_operator(O)` | `Σ_ℓ a_ℓ v_ℓ + ⟨Q⁻¹, U'OU⟩` with the row-pass `U'OU` | O(K q²) after the O(n q²) row pass |
| `inverse_operator_diagonal(O)` | the factor's own data operator (`O = H − S`, the `edf` caller in `state_ops`, recognised by identity with the operator the factor was built from): identity route `1 − λ_u (H⁻¹)_uu` on the tree and `1 − diag(Q⁻¹ S_b)` on the border; other operators: tree `u`: `κ_u δ_u − F_u Q⁻¹ (M'V)_u'` with `δ_ℓ = a_ℓ`, `δ_u = Σ_c ρ_c δ_c`; border: `diag(Q⁻¹ (U'OU + V'(MF)))`. The earlier closed form `κ_u(α_u − γ_u) + F_uQ⁻¹(α_uA_u + Φ_u − Γ_u)'`, `diag(Q⁻¹A_O − Q⁻¹F'Γ)` is the same in exact arithmetic (`α_u − γ_u = δ_u`; `α_uA_u + Φ_u = Σ_{ℓ ⪯ u} a_ℓ (MF)_ℓ`, so `α_uA_u + Φ_u − Γ_u = −(M'V)_u`) but built by subtraction: for the data operator it is off by 1.25e-1 absolute in the edf at λ = 1e-7 (section 8) | O(k) after `diag(H⁻¹)`; generic O(k q²) |
| `operator_cross_trace(O_l, O_r)` | `tr(Ẑ diag(a_l) Ẑ diag(a_r)) + ⟨Q⁻¹, V_r' Ẑ V_l⟩ + ⟨Q⁻¹, V_l' Ẑ V_r⟩ + ⟨Q⁻¹ (U'O_lU) Q⁻¹, U'O_rU⟩`, `Ẑ = M Z M'`, `V` and `U'OU` in the row-pass form, with `tr(Ẑ diag(a) Ẑ diag(b)) = Σ_x v_x² [A_x B_x − Σ_{c∈ch(x)} ρ_c⁴ A_c B_c]`, `A_x = Σ_c ρ_c² A_c` (leaves `a_ℓ`), and `Ẑ V = M T⁻¹ M' V` one tree solve with `q` right-hand sides. Per operator, form once `ẐV` (K x q), `V Q⁻¹` (K x q), `P = Q⁻¹(U'OU)Q⁻¹` (q x q) and the `A_x` accumulations; a pair is then `⟨V_rQ⁻¹, ẐV_l⟩ + ⟨V_lQ⁻¹, ẐV_r⟩ + ⟨P_l, U'O_rU⟩` plus the O(k) `Ẑ` term | O(K q² + q³) per operator, O(K q + q² + k) per pair |
| `penalty_operator_cross_trace(E_I, O)` | `tr(H⁻¹E_IH⁻¹O) = Σ_{i∈I} [ (Z M'diag(a) M Z)_ii − 2 W_i Q⁻¹ F_i' + F_i P F_i' ]` with `W = T⁻¹ M'V` (one tree solve with `q` right-hand sides), `P = Q⁻¹(U'OU)Q⁻¹`, and the first term from the tree covariances: per leaf `ℓ`, `Σ_{i∈I} Cov(S_ℓ, β_i)² = κ_{i_ℓ}² π_ℓ(i_ℓ)² + Σ_{x ∈ anc(i_ℓ)} v_x² π_ℓ(x)² [h^I_x − 1_I(c_x) σ_{c_x}² − ρ_{c_x}² h^I_{c_x}]` (`i_ℓ` the level-`I` ancestor of `ℓ`, `c_x` the child of `x` toward `ℓ`, `h^I` as in `penalty_cross_trace`, `π` the path products of section 3.5), summed with weights `a_ℓ`. Derivation: `H⁻¹OH⁻¹ = BOB + BOUQ⁻¹U' + UQ⁻¹U'OB + UQ⁻¹(U'OU)Q⁻¹U'` with `B = blockdiag(Z, 0)`, and row `i` of `BOU Q⁻¹ U'` on the diagonal is `−(ZM'V)_i Q⁻¹ F_i'`. This replaces the pattern-congruence sandwich of the earlier draft (equal in exact arithmetic, O(k d²) more expensive); the explicit-column route `Σ_{i∈I} h_i'Oh_i` with `h_i = H⁻¹e_i` by solves is forward-error limited (9.4e-3 at λ = 1e-7, section 8) and is not used | O(K d + k q + q³) |
| `penalty_operator_cross_trace(Ω_b, O)` for a border penalty | `λ ⟨Q⁻¹ Ω_b Q⁻¹, U'OU⟩`, since `H⁻¹Ω_bH⁻¹ = UQ⁻¹Ω_bQ⁻¹U'` | O(q³) |
| `derivative_cross_traces(directions)` (batched; conditional on the working-tree `DerivativeCrossTraceFactor` protocol landing, which `reml_direct_hessian` now dispatches on by `isinstance`) | direction `O_i = λ_i Ω_i + dH_i` with `Ω_i` a level selector `E_I` or a border penalty and `dH_i` leaf-form plus low-rank; by bilinearity `tr(H⁻¹O_iH⁻¹O_j) = λ_iλ_j tr(H⁻¹Ω_iH⁻¹Ω_j) + λ_i tr(H⁻¹Ω_iH⁻¹dH_j) + λ_j tr(H⁻¹dH_iH⁻¹Ω_j) + tr(H⁻¹dH_iH⁻¹dH_j)`, each piece a row above. Per direction, form once `ẐV_i`, `V_iQ⁻¹`, `P_i`, `W_i = T⁻¹M'V_i` and the `A_x` accumulations; per level `G_I`, `T⁻¹E_IF` and `h^I`. Without this method the nested factor takes the pairwise path, which calls the four methods above per pair and is the phase fix D targets (716 of 743 s). Verified on all pairs of `L + 1` directions in section 8 | O(m (K q² + q³) + m² (K q + q² + k)) |
| `inverse_operator_square_diagonal(O)` | data operator (`O = H − S`, the only caller): `1 − 2 λ_u Z^H_uu + λ_u (H⁻¹SH⁻¹)_uu` on the tree and `1 − 2 (Q⁻¹S_b)_jj + ((H⁻¹SH⁻¹)_bb S_b)_jj` on the border, with `H⁻¹SH⁻¹` the penalty sandwich (weighted level pattern `Σ_I λ_I N_I` plus `S_b`); generic operators: `Σ_v (H⁻¹OH⁻¹)_uv O_vu` from the operator sandwich | O(k d² + k q²) |
| `coefficient_estimable()` | null basis of `Q_s` (section 3.7) mapped back by `D_s` and re-orthonormalised, lifted as `[−F z; z]`, as in the scalar factor | O(k q) |

`Z^H` above is `diag(H⁻¹)`. The profiled (intercept-free) view is the same
index shift as `ProfiledScalarSchurFactor`: the leaf-level low-rank
representation with the intercept column dropped from the basis and kept in
the core.

### 3.7 Refusals and where rank decisions live

- Working weights: correctness of the tree elimination needs only positive
  pivots (`T = L D L'` holds as algebra for any `w`, Lean `chain_ldl`), and the
  accuracy guarantees of section 8 need `w ≥ 0`. The factor therefore refuses
  on the pivots, not on row signs: `D_u ≤ γ_u` with the per-pivot cancellation
  floor `γ_u = 10 ε (fan_u + 2) (|w_u| + Σ_{c ∈ ch(u)} |s_c| + λ_u)`, the
  analogue of today's `d ≤ 0` refusal in `ScalarSchurFactor` with the
  accumulated absolute mass as its scale; it never fires for `w ≥ 0` (there
  `D_u ≥ λ_u > 0` exactly). A row-level test (`moments.py`'s
  `signed = any(w < 0)`) would trip on round-off-negative rows of families whose
  observed weights are non-negative in exact arithmetic, and would turn
  today's mid-line-search negative-row iterates on the observed-geometry path,
  which work with parents in the border, into
  `ObservedGeometryInfeasibleError`. Audit of
  `compute_observed_information_weights` on 20,000 simulated rows (critic run
  `wsign`, re-run here as `critic_wsign.log`): minimum weights are
  non-negative for Poisson/log, Binomial/logit, probit and cloglog, Gamma/log
  and inverse, Tweedie(1.5)/log, NB2/log and Gaussian/identity; negative for
  Gaussian/log, sqrt and inverse, Gamma/identity and sqrt, Binomial/cauchit,
  Poisson/inverse, Tweedie/identity and inverse, and NB2/identity, sqrt and
  inverse; Poisson/identity (−8.9e-16) and Tweedie/sqrt (−3.6e-15) are
  negative only by rounding. `classify_reml_curvature` returns `observed` for
  Tweedie/log and Gamma/log, the pricing families, whose rows are
  non-negative. Selection declines the chain (today's single-level backend,
  parents in the border) only for the (family, link) pairs with negative
  weights in exact arithmetic, and for pairs not in the table until audited.
- Non-strict or implicit nesting: for every consecutive pair of chain levels
  the number of distinct `(child, parent)` code pairs over the rows must equal
  the number of distinct observed child codes; otherwise refuse with the
  instruction to build the interaction code (`make:model`), as lme4 documents.
- Zero penalties. A chain level whose penalty is identically zero makes `H`
  exactly singular: that level's indicators sum to the intercept (and to every
  coarser level), all its `s_u` vanish, so `Q_00 = Σ_roots s_u = 0` exactly
  while every tree pivot stays positive (exact rational check, critic run
  `lam0`, re-run here: `λ = (1,1,0)`, `(0,1,1)` and `(1,0,1)` all give
  `det H = 0`, minimum pivots 1.0, 0.494 and 0.973, `Q_00 = 0`). The earlier
  claim that `λ_u = 0` with `ω_u > 0` is fine was wrong. Rule:
  `resolve_structured_backend` treats a zero penalty on any chain level
  (`lambda2` scalar 0, a dict entry that is 0 or absent, or an `S_override`
  diagonal that is zero on the whole level) by moving that level out of the
  chain into the border, which is today's placement of every parent (today
  such a level is accepted there as an uncoupled rank deficiency, `F z = 0`),
  and rebuilding the chain over the remaining levels with parent pointers
  composed through the removed level (nesting is transitive; the count test
  above is re-run on the composed pairs); a zero penalty on the leaf level
  itself keeps today's refusal (`RandomEffect group … has zero penalty and is
  aliased with the fitted intercept`). Per-node zero entries of an
  `S_override` diagonal are accepted when `ω_u > 0` at every such node (the
  pivot `D_u = ω_u` is positive); a node with `λ_u = 0` and `ω_u = 0` is a
  singular pivot and refuses as today's `d ≤ 0` check does. The factor
  additionally refuses when `Q_00 = Σ_roots s_u` is exactly zero, the exact
  singularity of `H` (no leaf-to-root path with all `λ_u > 0` and `w_ℓ > 0`),
  with the intercept-aliasing message rather than the generic coupled-null one.
- `S_override`: the authoritative penalty is validated with the chain's union
  of indices as the structured block (`_structured_override_incompatibility`):
  it must be diagonal on every chain level with no cross-level and no
  chain-to-border mass; per-node `λ_u` are taken from the diagonal. Otherwise
  the chain is declined and today's single-level validation applies. REML
  leaves `S_override` as `None` on the structured paths (`discrete.py`,
  `reml_finalize.py`); fixed-penalty callers (`scop_efs`, a user
  `S_override`) reach this rule.
- Zero-weight and unobserved levels: `σ_u = 0` exactly, so the node is severed
  from whatever parent the layout assigns it (`parent := 0` for unobserved
  levels, which is exact and verified). Crossed factors are border columns
  only.
- Rank: every tree pivot is a sum of non-negative terms and needs no floor. All
  rank decisions stay in `Q`, but they are taken on the Jacobi-scaled
  `Q_s = D_s Q D_s`, `D_s = diag(Q)^{-1/2}`, not on `Q` itself, because the
  PSD-sum form knows `Q` componentwise: each entry carries error at most `γ_Q
  sqrt(Q_ii Q_jj)` (`γ_Q` in section 8), so `Q_s` is known to `γ_Q` per entry
  and its small directions (the intercept, `Q_00 = Σ_roots s_u ≈ λ`) are
  well-determined. Today's floors charge `ε` times the raw subtracted mass and
  the SVD fallback cuts at `1e-10 σ_max(Q)` of the unscaled `Q`; both
  misclassify those directions, so they do not apply unchanged: at λ = 1e-7
  with weights x1e4 the PSD-sum `Q_00` is 1.669e-7 (exact 1.669e-7) but
  `_cancellation_pivot_floor` gives pivot² 1.174e-7 against a floor of
  5.917e-7, so the fit falls back, and the fallback would truncate 3 of 5
  directions (threshold 6.54e-3) although `κ_s(Q) = 12.8`; on a border with one
  duplicated unpenalised column (true nullity 1) the unscaled rule truncates
  2 directions at λ = 1e-4 and 3 at λ = 1e-5 (weights x1e2) and
  `_reject_coupled_schur_null_space` then refuses a valid fit (critic runs
  `floors`, `fallback`, re-run here as `critic_floors.log`,
  `critic_fallback.log`). The scaled rules: the Cholesky probe residual on
  `Q_s`; the cancellation floor `pivot_s² ≤ 20 q ε` (on the fixtures of
  section 8 the smallest scaled pivot² is 7.2e-2 and the largest scaled probe
  residual 9.4e-16, so no fallback fires); the SVD fallback on `Q_s` with
  cutoff `1e-10 σ_max(Q_s)`, four orders above the componentwise floor
  `q γ_Q`; the negative-curvature test on `Q_s`; and the coupled-null-space
  test on the null vectors of `Q` obtained as `D_s z_s` and re-orthonormalised,
  with `F = T⁻¹C` in place of `D⁻¹C`. On the duplicated-column fixture the
  scaled rule truncates exactly one direction at every λ from 0.5 to 1e-7
  (scaled singular values ≤ 7.5e-18 for the null, ≥ 3.7e-2 for the rest) and
  the coupled-null test accepts (section 8); section 9 makes this a regression
  test.

## 4. Data layout: `NestedStructuredLayout`

Fields (immutable, cached on the `DesignMatrix` like today's layouts, keyed by
the chain's group indices and the group table):

- `chain_group_indices`, `chain_group_names`: coarsest to finest, each a
  `RandomEffectGroupMatrix`.
- `sizes`: `K_j` per level; `parent`: for `j ≥ 1` an `intp` array of length
  `K_j` giving the local index at level `j−1`; unobserved child codes get 0.
- `structured_indices`: per level the global coefficient range of that group
  (the coefficient order is the design's group order; the factor works on
  per-level arrays and maps through these ranges).
- `small_indices`, `small_matrices`, `local_groups`, `dense_small_matrix`,
  `small_execution_plan`: the border exactly as `ScalarStructuredLayout` builds
  it, with the chain's parent groups removed from the border and every other
  group (including crossed random effects) kept.

Validation at build: chain groups are random effects with consistent
coefficient geometry, no constraints or SCOP on any group (today's
`select_structured_group` checks), the strict-nesting count test above for
every consecutive pair (re-run on composed pairs when a zero-penalty level has
left the chain, section 3.7), `n_levels` consistency between codes and slices,
and, when an `S_override` is authoritative, the diagonal-per-level rule of
section 3.7.

Row pass per PIRLS iteration (only the leaf group touches rows): `w_leaf`,
`C_leaf`, `xtwz_leaf` from the existing leaf kernels; the shifted leaf means
and the within-leaf centred border Gram `W_in` (one O(n q²) pass, the only
route that keeps `Q_00`; section 3.4); the border moments as today. Each
signed W-derivative operator is one more row pass giving `a_ℓ`, `dev_ℓ` and
`W_a` about the same leaf means (section 3.6). Parent-level sufficient
statistics (`xtw`, `xtwz`) are subtree sums of the leaf ones, not row passes.
Per λ-trial: the up pass with shifted parent means, `Q`, its scaled
factorization, `F` and `e`, the Takahashi scalars.

## 5. Where it plugs in

- `selection.py`: `select_structured_group` returns one dominant index today,
  and a FactorSmooth term takes precedence over every RandomEffect. Rule B
  keeps that precedence: with a FactorSmooth present there is no chain and the
  RandomEffect terms stay in the border as today. Otherwise take the largest
  random-effect term as the leaf and grow the chain upward by testing every
  remaining random-effect term, not only the next-larger one, for being a
  function of the current coarsest code (the count test of section 3.7),
  choosing the finest such term at each step; a crossed term whose size falls
  between two chain levels is therefore skipped, not a chain terminator.
  Levels with a zero penalty leave the chain (section 3.7). A chain of length
  1 is today's scalar Schur backend; a chain of length ≥ 2 selects the nested
  factor. Terms not in the chain stay in the border. `StructuredGroupSelection`
  and `StructuredBackendDecision` carry `chain_group_indices`.
- Cost model for `auto` (flops of one factorization; `p` coefficients,
  `n_b = q + 1` border width with intercept, `k = Σ K_j`):

      dense gram:     (p+1)³ / 3                     memory (p+1)²
      single level:   K_max n_b1² + n_b1³ / 3        memory K_max n_b1 + n_b1²,  n_b1 = p + 1 − K_max
      nested chain:   k n_b² + n_b³ / 3 + 2 k n_b    memory k n_b + n_b²,        n_b = p + 1 − k

  With `P = k − K_max` parent nodes and `n_b1 = n_b + P`, single-level minus
  nested is `2 K_max n_b (P − 1) + K_max P² + P n_b (P − 2) + P³/3`, which is
  non-negative for every `P ≥ 2`; for a single parent node (`P = 1`) it is
  `K_max − n_b + 1/3`, so nested is costlier by up to `n_b − K_max` flop units
  when the border is wider than the leaf level. The earlier claim "never
  costlier whenever `n_b < 2 K_max`" was wrong in that corner; the `auto` rule
  decides. The existing crossover applies to the nested ratio `(k n_b² +
  n_b³) / (p+1)³ ≤ 0.05` with `p ≥ 32`, but its constant 0.05 was calibrated
  end to end on the scalar backend's derivative machinery (`selection.py`),
  and the two proposed nested anchors have ratios near 1e-3 or below, which
  cannot recalibrate the threshold region. REML derivative work per λ-trial
  changes from O(m_b² n_b1³) (all parent penalties in the border) to
  O(m (K q² + q³) + m² (K q + q² + k)) with the small `n_b` (section 3.6).
- Anchors that must still hold: the five #343 single-random-effect anchors
  (ratios 0.596 to 0.033) were measured on a real pricing workload whose term
  list is not in the repository (`selection.py` records only `K` and `q`), so
  whether they contain a nested random-effect pair that Rule B would reroute
  cannot be checked from the code; they must be re-measured under Rule B and
  their term lists recorded with the benchmark. The FactorSmooth and SZ
  crossovers are untouched. New anchors to record: pg17 step C (chain 87 +
  942 at 77k rows) and DVSA step D at 200k rows (chain 223 + 3,749 + 6,038,
  border about 46), plus at least one mid-ratio nested anchor (ratio between
  0.03 and 0.15, for example a chain of a few hundred nodes beside a border of
  one to two hundred spline columns), recorded through the existing
  `structured_auto_cost_ratio` profile field.
- `assembly.py`: `build_structured_system` gains the nested system (leaf
  statistics + border moments), `build_penalized_structured_operator` adds the
  per-level λ to the tree penalty vector and `S_b` to `A`,
  `build_augmented_*` augments the border with the intercept exactly as now,
  `solve_cached_*` gains the nested case.
- `hessian_factor.py`: the new factor implements the existing protocol; no
  protocol change.

## 6. Coupling that breaks

- `ProfiledScalarSchurFactor` reads the private `_F`, `_d_inv`, `_Q_inverse()`;
  the nested factor needs its own profiled adapter (index shift plus the
  leaf-level low-rank view with the intercept column dropped), or a small
  shared interface for "inverse low-rank parts". Named contracts the adapter
  must carry: `mean_x`, `sum_w` and `augmented_factor` (`w_derivatives.py`
  reads `mean_x` and `sum_w` through `getattr` and falls back silently to
  `mean_x = 0`, uncentred W-derivative operators, when they are missing),
  `dominant_group_name`, `small_indices`, `structured_indices`, `rank`,
  `rank_truncated`, `used_dense_fallback`, `fallback_reason`,
  `schur_condition_estimate` (now of `Q_s`) and `minimum_local_diagonal`.
- Implicit centring: `ProfiledScalarSchurFactor.trace_inverse_operator` and
  `inverse_operator_diagonal` wrap a raw `SymmetricBlockOperator` in
  `CenteredBlockOperator` themselves, and `irls_direct.py` relies on that for
  `p_eff` (it wraps only `BlockSymmetricOperator | SumToZeroBlockOperator`
  explicitly). A nested raw operator type bypasses both unless the nested
  adapter copies the contract; it must, and the `isinstance` in `irls_direct`
  must name the nested type.
- `w_derivatives.py` finds the structured layout through
  `factor.dominant_group_name` and `get_structured_layout(dominant_group_index=...)`;
  the nested layout is keyed by the chain, so the factor must expose
  `chain_group_indices` and the layout getter must accept them. Its
  `centered_signed_gram` must build the nested (leaf-form) operator with the
  row-pass quantities of section 3.6 (`a_ℓ`, `dev_ℓ`, `W_a`).
- `observed_geometry.py` builds its own structured system and factor from a
  single `dominant_group_index` (fed from `direct.py`'s candidate and trial
  evaluations), reads the private `augmented_factor._Q` for its eigenvalue
  check and dispatches by `isinstance` with a `TypeError` else-branch. It must
  take the chain, use a public curvature check of the factor (the scaled
  negative-curvature test of section 3.7) instead of `_Q`, and name the nested
  factor in the dispatch.
- `layout.structured_design_matvec` and `structured_design_rmatvec` assume one
  dominant group. The nested versions are one leaf matvec of the path-summed
  coefficients (`Z β = Z_leaf (M β_tree) + X_b β_b`) and the subtree sums of
  the leaf rmatvec, O(n + k) with the existing leaf kernels.
- `irls_direct.py` records `structured_dominant_group`; record the chain.
- `reml_finalize.py` rebuilds coefficient factors by `isinstance`; `state.py`
  unions; `state_ops.py` `edf` and `edf1` take the identity route
  (`inverse_operator_diagonal` and `inverse_operator_square_diagonal` on the
  factor's own data operator, section 3.6); `_structured_small_data_factor`
  reads the border's `(A, cross, center, total)` and is unchanged for the
  border.
- `covariance.py` caps block requests at `max_structured_inverse_block = 256`
  over `structured_indices`. Parent random effects are uncapped in today's
  border and become capped chain levels, and `metrics._feature_se_impl`
  requests a term's full block before it branches on the spec type, so for a
  parent random effect with more than 256 levels it changes from returning a
  result to raising `RuntimeError`. Applying the cap only to the leaf level is
  not viable (DVSA parents have 38k levels); the caller must request the block
  after branching (the `RandomEffect` branch of `feature_se_from_cov` already
  uses `covariance_selected_diagonal`, which is uncapped). This is a required
  change in `metrics.py`, and until it lands a documented behaviour change.
- Estimability. `centered_operator_coefficient_estimable` dispatches
  `_independent_block_centered_estimability`, which eliminates the unpenalised
  data Gram of the structured block level by level; an unknown raw type raises
  `TypeError`, which the caller does not catch (it catches only `LinAlgError`
  and Arpack errors), so `summary()` would crash. In the nested geometry that
  block, `M' diag(w) M`, is singular: each parent's indicator is the sum of its
  children's, so `z = e_p − Σ_{c ∈ ch(p)} e_c` has `M z = 0`. Exact reduction:
  the column span of `Z = Z_leaf M` equals that of `Z_leaf` (`M = [M_parents |
  I_K]`), so border estimability equals today's single-level computation with
  the leaf level as the dominant level, and every chain coordinate is
  non-estimable in the data sense (every node is a parent or a child in one
  such `z`). Implement nested estimability as that reduction and test it
  against the dense `decompose_gram` on a small chain (section 9).
- `operators.py`: the tree block of a leaf-form operator is not diagonal in
  node space, so `_operator_dlr` cannot represent it; the nested factor owns
  its operator conversion (leaf projection) and never calls the DLR helpers
  for the tree part.

## 7. Lean contracts

`NestedElimination.lean` (Lean 4 v4.33.1, the pinned Mathlib
0df444a3; registered as a `lean_lib` and default target) proves over finite
real matrices: `det_bordered_schur` (Mathlib `det_fromBlocks₁₁`),
`det_ldl_product`, `eliminate_support`, `eliminate_no_fill`,
`comparable_of_support`, `ancestors_pairwise_comparable`,
`nested_leaf_pivot_no_fill`, `star_pivot`, `star_pivot_ge`,
`weighted_scatter`, `pivot_mass_split`, `star_psd_eq_subtraction`,
`trace_selector_frobenius`, and, added in this revision, the factorization
itself: `subtree_telescope` (the constant-ancestor-column invariant
`Σ_{v ⪯ u} ω_v = Σ_{v ⪯ u} w_v + Σ_{v ≺ u} s_v`), `chain_ldl` (`L diag(D) L' = T`
for the recursion's `L_{a,u} = σ_u`, `D_u = ω_u + λ_u`, from the local
equations `σ_u = ω_u/D_u`, `s_u = ω_u − ω_u σ_u`, `ω_u = w_u + Σ_{parent c = u}
s_c` and a forest `anc` that is reflexive, antisymmetric, transitive, strictly
nested and generated by `parent`), `chainL_det` (`det L = 1` given a depth
function increasing from ancestors to descendants; `L` is block triangular
with identity diagonal blocks) and `chain_det` (`det T = Π D_u`). All
seventeen depend only on `propext`, `Classical.choice` and `Quot.sound`; no
`sorry`. `star_pivot` is a scalar identity between the two forms of one pivot,
not an elimination statement; the elimination content is in `chain_ldl`. Not
proved: floating point, complexity, convergence, that the float64 code
computes the exact recursion's `ω, s, σ, D`, the tree-wide telescoping of the
PSD identity (one star only), and the multi-step iteration of the one-step
no-fill kernel. Build record: section 8.

## 8. Verification and measurements

Two references. The builder's `verify.py` (60-digit `Decimal` Gauss-Jordan on
5/15/40 fixtures with 3,000 rows, `verify_run4.log`) checked the closed forms
of section 3.5 and the pattern-congruence routes at 340 checks, but its border
tolerance `tol_border = 8ε p cond(Q)` used the unscaled `cond(Q)` of 1e13 to
1e14 and was vacuous on the stress fixtures (fixture F1: `tol_border` 1.0e+1,
`log|H|` bound 1.76e+1, level-pair bounds 41.7 to 82.6, sandwich bounds 208 to
413), and several operator checks were normalised by `Σ|parts|`, which hides
cancellation; the planned mutation "replace the PSD-sum `Q` by the
subtraction form" (`log|H|` error 8.28e-2) would have passed those bounds.
That run is superseded for every quantity that passes through `Q⁻¹`.

The revision's reference is the review's exact rational assembly of `H` from
the float64 rows (`fractions.Fraction`, exact inverse and determinant), re-used
by `verify2.py` (`verify2_run2.log`): a 3/7/16 chain, 600 rows, border 5
(intercept, a normal, a column with mean 10, a leaf attribute, a root
attribute), two zero-weight leaves and unobserved nodes at every level, with
λ = (0.7, 0.05, 3); λ = 1e-7 with weights x1e4; λ = 1e8; mixed λ = (1e8, 1e-7,
1) with weights x1e4; mixed λ = (1e-7, 1e8, 1e-3) with weights x1e-3; and a
2/4/8/14 chain of depth 4, 500 rows, λ = (1e-4, 2, 1e-6, 5), in which the root
level is exactly aliased by the intercept and the root attribute (its `H⁻¹`
cross blocks are exactly zero, a maximal-cancellation case for `t1 + t2 +
t3`). Bounds are derived per quantity from dims, `ε` and the Jacobi-scaled
condition number `κ_s(Q)` (1.5 to 74 on these fixtures, against `cond(Q)` up
to 1.9e15): `γ_tree = n_tree ε` with `n_tree = rows per leaf + L (fan + 4)`
for the recursions that sum non-negative terms, `γ_Q = (n_tree + q + 10) ε`
for the componentwise error of the PSD-sum `Q` (measured against
`sqrt(Q_ii Q_jj)`), and `γ_border = q κ_s(Q) γ_Q` for one pass through the
Cholesky-formed `Q⁻¹`, with small integer multiples counting the passes.
Positive quantities are measured relative to their value; signed quantities
against a Cauchy–Schwarz scale (`sqrt(tr(H⁻¹O_lH⁻¹O_l) tr(H⁻¹O_rH⁻¹O_r))` for
`tr(H⁻¹O_lH⁻¹O_r)`, `‖H⁻¹_II‖_F ‖H⁻¹_JJ‖_F` for `‖H⁻¹_IJ‖_F²`,
`sqrt((H⁻¹)_uu (OH⁻¹O)_uu)` for `diag(H⁻¹O)`, `sqrt(p tr(H⁻¹OH⁻¹O))` for
`tr(H⁻¹O)`), never against `Σ|parts|`. Checked on every fixture: pivots,
`log|H|`, `Q` entries, `diag(H⁻¹)`, `F`, every level pair (`t1 + t2 + t3`,
with the value-relative error and the cancellation factor `Σ|t_i| / |Σ t_i|`
reported alongside), level x border, border x border, `tr(H⁻¹O)`,
`diag(H⁻¹O)`, `tr(H⁻¹O_lH⁻¹O_r)`, `tr(H⁻¹E_IH⁻¹O)` by the closed form of
section 3.6 (and, for information, by explicit inverse columns, which is
forward-error limited: 9.4e-3 at λ = 1e-7 against 6.0e-7 of the bound for the
closed form), `edf` by the identity route, and the batched
`derivative_cross_traces` matrix (one direction `λ_I E_I + dH` per level plus a
border penalty, all pairs).

Result (`verify2_run2.log`): 144 checks, none exceeding its bound, worst ratio
0.107 (`F` entries on the λ = 1e8 fixture). Headline numbers:

| Fixture | `κ_s(Q)` | `log|H|` error (bound) | `Q` scaled entry error (bound) | `tr(H⁻¹O_lH⁻¹O_r)` CS-relative (bound) | worst level-pair CS-relative | notes |
|---|---|---|---|---|---|---|
| base 3/7/16, λ=(0.7, 0.05, 3) | 3.4 | 8.5e-14 (2.3e-12) | 3.9e-16 (2.1e-14) | 1.1e-16 (1.6e-12) | 2.6e-15 | |
| λ=1e-7, weights x1e4 | 12.8 | 1.6e-13 (1.9e-12) | 4.6e-16 (2.0e-14) | 1.6e-16 (5.4e-12) | 7.7e-16 | `cond(Q)` 1.9e15 |
| λ=1e8 | 53.6 | 1.1e-13 (6.0e-12) | 4.3e-16 (2.0e-14) | 2.2e-16 (2.2e-11) | 2.3e-15 | |
| mixed λ=(1e8, 1e-7, 1), weights x1e4 | 21.3 | 5.7e-14 (1.1e-11) | 7.0e-16 (1.9e-14) | 1.6e-16 (8.4e-12) | 4.2e-16 | pair (0,2): value 1.7e-30, value-relative 2.2e-9 with cancellation factor 8 |
| mixed λ=(1e-7, 1e8, 1e-3), weights x1e-3 | 74.2 | 0 (4.3e-11) | 1.1e-15 (2.3e-14) | 4.8e-17 (3.4e-11) | 4.2e-16 | |
| depth 4, λ=(1e-4, 2, 1e-6, 5) | 1.5 | 7.5e-14 (1.3e-12) | 9.3e-16 (2.0e-14) | 5.3e-17 (7.4e-13) | 3.7e-16 | root level exactly aliased; pair (1,3) value-relative 3.4e-10 |

`edf` by the identity route is within 2.8e-15 absolute on every fixture (Σedf
equal to the exact value to six decimals); the generic subtraction closed
form for the same quantity is off by 1.25e-1 (tree 6.6e-3, border 1.2e-1;
Σedf 15.873 against 16.000) at λ = 1e-7 and by 1.15e-1 on the mixed fixture
(critic run `crit4`; mutation table below). The level pairs across a 1e15
ratio of penalties (mixed fixture, pair (0,2)) have value-relative errors of
2e-9 on a value of 1.7e-30, twenty orders below the Cauchy–Schwarz scale of
the Hessian entry, from the float `F` and `Q⁻¹` inputs (the `t3` formula with
exact inputs is at 8e-16, critic run `crit6`; the Cholesky-whitened form
`⟨L⁻¹G_IL⁻ᵀ, L⁻¹G_JL⁻ᵀ⟩` does not change it); they are harmless for the
Newton step and are reported as a conditioning fact, not hidden by the scale.

Mutation checks (each must fail at least one check; ratios are error over
bound):

| Mutation | checks failing (of 144) | worst ratio, where |
|---|---|---|
| `Q` by the subtraction form | 37 | 1.4e+12, `Q` entries at λ=1e-7 (`log|H|` 2.3e+10) |
| drop the between-child term | 115 | 1.8e+55 |
| within-leaf scatter by subtraction (the decision-1 alternative) | 37 | 1.6e+12, `Q` entries at λ=1e-7 |
| plain (unshifted) leaf and parent means | 5 | 8.3e+2, `diag(H⁻¹O)` at λ=1e-7; `Q` entries 30 (mixed), 42 (depth 4) |
| `σ = 1 − ρ` | 2 | 7.0e+5, `F` entries on the second mixed fixture; 1.3e+3 at λ=1e8 |
| `F` by `G_u − σ_u A_u` | 2 | 4.0e+2, `diag(H⁻¹O)` at λ=1e-7; 32 on the mixed fixture |
| `diag(1/D)` in place of the Takahashi scalars | 51 | 5.5e+12, `diag(H⁻¹)` |
| one parent pointer moved | 111 | 3.9e+19 |
| chain treated as a single level (parents severed) | 114 | 5.8e+25 |
| `U'OU` and `V` by subtraction | 21 | 8.8e+8, `tr(H⁻¹E_0H⁻¹O)` at λ=1e-7; `tr(H⁻¹O)` 7.2e+8, `tr(H⁻¹O_lH⁻¹O_r)` 7.4e+7 |
| non-symmetrised `U'OU` shorthand | 25 | 1.4e+23; 6.1e+13 already on the base fixture |
| `edf` by the generic subtraction closed form | 5 | 4.7e+10 at λ=1e-7; 1.8 already on the base fixture |
| `edf` by the generic row-pass closed form (positive control, not a mutation) | 0 | 0.11: the row-pass generic route is accurate on the data operator too; the identity route is kept because it is exact and O(k) |

Rank decisions (`verify2_run2.log`, duplicated unpenalised border column,
true nullity 1): today's unscaled rule truncates 1 direction for λ from 0.5 to
1e-3, 2 at λ = 1e-4, 3 at λ = 1e-5 (weights x1e2) and 4 at λ = 1e-7 (weights
x1e4); the scaled rule truncates exactly 1 at every λ (scaled null singular
value ≤ 7.5e-18, next ≥ 3.7e-2) and the coupled-null test on the mapped-back
null vector accepts. Cancellation floors: on the base, λ = 1e-5, λ = 1e-7 and
mixed fixtures the smallest scaled pivot² is 0.71, 7.2e-2, 0.27 and 0.17
against the floor `20 q ε = 2.2e-14`, and the scaled probe residual at most
9.4e-16; today's unscaled floor falls back on the λ = 1e-7 (pivot² 1.17e-7
against 5.92e-7) and mixed (1.33e-7 against 2.28e-7) fixtures.

The generic `diag((H⁻¹O)²)` decomposition on the builder's λ = 1e-7 fixture
(`verify_run4.log`): 7.9e-16 relative to its own terms, 1.9e-3 relative to
`Σ_v |(H⁻¹O)_uv (H⁻¹O)_vu|`; the data-operator identity route: 1.1e-16 on
every fixture. This is the decision-3 caveat. Lean build record
(`lake_build.log`, this revision): `lake build` ended with `Build completed
successfully (8713 jobs)`, exit 0, no warnings, `NestedElimination` built in
5.9 s; `lake env lean NestedElimination.lean` re-elaborates the file with no
errors or warnings and prints seventeen axiom lines, all `[propext,
Classical.choice, Quot.sound]` except `ancestors_pairwise_comparable`, which
uses none.

Prototype timing at the full DVSA tree (7,294 / 38,366 / 64,761 nodes,
k = 110,421, border 40, numpy, one thread; `scale_dvsa_full.log`): factor build
0.099 s, one solve 0.004 s, inverse diagonal 0.014 s, all six level-pair cross
traces 0.413 s, `trace_inverse_operator` 0.017 s, `inverse_operator_diagonal`
0.088 s, one operator sandwich 0.244 s, one level sandwich 0.207 s; `F` is
33.7 MiB. These are prototype numbers with Python-level level loops and
`np.add.at`; they bound the linear algebra, not a production fit.

## 9. Test plan and performance-evidence plan

Tests (focused, mutation-checked):

- Parity against the exact reference on the fixtures of section 8, ported from
  `verify2.py` with its bounds (`γ_tree`, `γ_Q`, `γ_border` from `κ_s(Q)`, and
  the Cauchy–Schwarz scales for signed quantities), through the public
  protocol methods of the augmented and profiled factors.
- End-to-end parity: `fit_reml` on a small nested dataset with
  `direct_solve="gram"`, today's `"structured"` (single level) and the nested
  factor: coefficients, `log|H|`, REML gradient and Hessian, `edf`, `edf1`,
  covariance blocks, within tolerances derived from `κ_s(Q)` and dims, never
  from the unscaled `cond(Q)`.
- Refusals and routing: implicit nesting; a node with `λ_u = 0` and `ω_u = 0`;
  a whole level with zero penalty routed to the border (from `lambda2`, a dict
  and an `S_override` diagonal) with the chain rebuilt over composed parents;
  an `S_override` with cross-level or chain-to-border mass declining the
  chain; pivots below their cancellation floor on signed iterates; the
  family/link table of section 3.7 at selection; constrained or SCOP groups; a
  crossed factor requested in the chain.
- Rank: a rank-deficient border (one duplicated unpenalised column) with small
  penalties (λ = 1e-4 and 1e-5 with weights x1e2, λ = 1e-7 with weights x1e4)
  must be accepted with nullity exactly 1 and the null basis lifted as
  `[−F z; z]`; the unscaled rules refuse or over-truncate these (section 8),
  so the test fails against them.
- Estimability: the leaf-level reduction against the dense `decompose_gram`
  on a small chain, with every chain coordinate non-estimable.
- `derivative_cross_traces` against the pairwise methods on a nested fit with
  W-derivative directions (`dH_extra` set) and against the dense reference.
- Mutation checks that must fail against the mutated code, with the bounds of
  section 8 (all twelve measured there): the subtraction-form `Q`; the
  dropped between-child term; the subtraction within-leaf scatter; plain
  means; `σ = 1 − ρ`; `F` by back-substitution; `diag(1/D)` for the Takahashi
  scalars; a moved parent pointer; the chain treated as a single level;
  subtraction-form `U'OU` and `V`; the non-symmetric shorthand; the generic
  subtraction `edf`.

Performance evidence (complete fits, `benchmark_hierarchical_credibility.py`,
threads pinned, `timeout -k 60 3600`, JSON per fit): pg17 step C exact and
discrete against the 743 s and 4.7 s baselines; DVSA step D at 200k, 1M and
full rows against the current backend where it finishes (200k did not finish
in 300 s), reporting wall time, peak RSS, the dispatched backend and its cost
ratio, REML iterations, and holdout deviance parity. Attribute time with the
phase profile (`reml_hessian`, `reml_structured_cache_solve`, row passes).
Both the exact and discrete paths, since both call `reml_direct_hessian`.

## 10. Numerical risks

- `cond(Q)` is driven by the intercept when the root penalty is tiny
  (`Q_00 = Σ_roots s_u ≈ λ`), but the PSD-sum form knows `Q` componentwise and
  the Jacobi-scaled condition number stays small (1.2 to 74 on the fixtures
  against unscaled 1.9e15), so the Cholesky, its solves and the rank
  decisions on `Q_s` are accurate; quantities formed through the explicit
  `Q⁻¹` inherit `q κ_s(Q) γ_Q`, the `γ_border` of section 8 (worst measured
  ratio against it 0.107). The unscaled floors, cutoffs and `cond(Q)`-based
  tolerances are the wrong instruments for this factor (section 3.7).
- The generic decomposition `Σ_v (H⁻¹OH⁻¹)_uv O_vu` for `diag((H⁻¹O)²)` is
  ill-conditioned when `λ ≪ ω`: its terms are of size `1/λ` and cancel to a
  result of size `1/w`. Measured 1.9e-3 relative to the natural scale of the
  result, 9e-14 relative to its own terms. The only caller passes the factor's
  own data operator, for which the identity route is exact (verified). Keep the
  generic route for other operators with this caveat, or refuse them
  (decision 3).
- `κ_u = 1/D_u − ρ_u σ_u t_u` is a difference; its absolute error is bounded by
  `ε (1/D_u + ρ_u σ_u t_u)` and it enters only `inverse_operator_diagonal`.
- Any Schur or operator quantity formed as a raw mass minus path-sum products
  (the subtraction `Q`, the subtraction within-leaf scatter, `A_O − (MF)'C_O`,
  `G_u − σ_u A_u`, plain means) loses `ε` times the raw mass, which `Q⁻¹ ~ 1/λ`
  amplifies to the errors of the mutation table; the row-pass and `g/e` forms
  of sections 3.3 and 3.6 are the only permitted forms.
- Level pairs across extreme penalty ratios are value-relatively inaccurate
  (2e-9 on a value of 1.7e-30) but twenty orders below the Cauchy–Schwarz
  scale of the Hessian entry (section 8).
- Unobserved nodes attached to parent 0 rely on `ω_u = 0` exactly, which
  `bincount` guarantees; `λ_u > 0` keeps `D_u > 0`.

## 11. Non-goals

Crossed factors beyond the border (no zero-fill ordering exists; they are the
genuinely crossed problem of the literature); families whose observed weights
are negative in exact arithmetic (they keep today's single-level backend,
section 3.7); FactorSmooth or sum-to-zero chains; changing the REML drivers,
the border machinery, the intercept profiling or the protocol; any change to
the discrete row passes.

## 12. Decisions for approval

1. Within-leaf scatter: settled by measurement, not open. The exact centred
   row pass (a new O(n q²) kernel about the shifted leaf means; no `X_b'WX_b`
   is needed for the chain path) is the only viable form: the subtraction
   alternative restores the whole subtraction-form failure through the
   intercept and every column constant within leaves (section 3.4). What
   remains to decide is the kernel's placement, not its form.
2. Rule B scope: nested factor only for chains of length ≥ 2, single-level
   chains keep `ScalarSchurFactor` — recommended — or replace the scalar factor
   by the nested one with L = 1 (one code path, larger diff and re-anchoring).
3. Generic operators in `inverse_operator_diagonal` and
   `inverse_operator_square_diagonal`: the factor's own data operator always
   takes the identity route (exact, O(k) after `diag(H⁻¹)`); other operators
   take the row-pass closed form for `diag(H⁻¹O)` (accurate on every fixture,
   including the data operator as a control) and the generic sandwich for
   `diag((H⁻¹O)²)` with the documented conditioning caveat — recommended — or
   refuse the latter (`NotImplementedError`) so no caller can depend on it
   silently. Today's only callers pass the data operator.
