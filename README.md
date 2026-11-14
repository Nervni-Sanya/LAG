# LAG — Lie Algebra Geometry

PyTorch library for matrix Lie groups: closed-form exponentials and their differentials, Jacobian
log-determinants of Lie-group steps, equivariant connection heads, Lie-group integrators (Lie–Euler, RKMK4),
and a generic `MatrixLieGroup` core into which any matrix Lie group can be plugged. Every formula is
specified in [MATH_SPEC.md](MATH_SPEC.md) and checked numerically by the test suite. Items the spec marks as
not verified, such as the completeness of the heads and the SE(3) invertibility bound, remain open.
[MATH_SPEC_LATTICE.md](MATH_SPEC_LATTICE.md) specifies multi-link lattice gauge layers: masks, the total
log-determinant of a masked sub-layer, its inverse and an exact 2D reference. They are built as the
`lag.lattice` package (see Status).

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0 (developed with 2.11). Computations are float64 / complex128 on
the CPU. float32 and GPU use are not supported yet (MATH_SPEC.md §9, item 6).

## Layout

| Module | MATH_SPEC | Contents |
|---|---|---|
| `lag/algebra.py` | §1 | `so3_hat`, `so3_vee`, `T2`, `T3`, `su_hat`, `su_vee`, `proj_su`, `se3_hat`, `se3_vee_first_order`, `se3_ad` |
| `lag/exp.py` | §2 | `so3_exp`, `su2_exp`, `se3_exp`, `su3_exp`, `dexpL_so3`, `dexpL_se3`, `dexpL_generic`, `exp_vjp_normal` |
| `lag/jacobian.py` | §3 | `logabsdet_step` |
| `lag/heads.py` | §4 | `eps_dual_su2`, `cross_dual_su3`, `head_fundamental`, `head_adjoint`, `head_loops`; plus `wedge_connection` (from the prototype, not in the spec) |
| `lag/flows.py` | §3.2–3.3 | `lie_euler`, `rkmk4` |
| `lag/groups/base.py` | §10, step 6 | `MatrixLieGroup` (generic core), `LieGroupLike`, `ExpFn` |
| `lag/groups/builtin.py` | §2 | `SO3`, `SU2`, `SU3`, `SE3`, with the closed forms wired in as backends |
| `tests/test_math_spec.py` | §8 | the 47 acceptance checks |
| `tests/test_groups.py`, `test_heads.py`, `test_exp.py`, `test_example_smoke.py` | | generic core, custom groups, flows, heads, exp; smoke test of the example |
| `examples/su3_single_variable_flow.py` | §7 | the end-to-end SU(3) flow (acceptance run) |

The lattice stage, in the build order of MATH_SPEC_LATTICE.md §11:

| Module | MATH_SPEC_LATTICE | Contents |
|---|---|---|
| `lag/lattice/field.py` | §1, §3.1 | field layout, `GaugeGroup`, `haar`, `shift`, `staples`, `head`, `gauge_transform`, `plaquettes`, `wilson_action`, `CoeffFn` |
| `lag/lattice/masks.py` | §3.2–3.3 | `mask` (even L only), `validate_mask` |
| `lag/lattice/layers.py` | §2, §6 | `masked_layer` (batched Dv, `create_graph` flag), `invert_masked_layer` |
| `lag/lattice/spectral.py` | §5 | `logdet_spectral` |
| `lag/lattice/exact.py` | §7 | `exact_logZ_2d`; the Bessel forms `bessel_i`, `su2_coefficient_bessel`, `exact_logZ_2d_su2_bessel`, `su3_trivial_coefficient_bessel` |
| `examples/lattice_flow_2d.py` | §8 | the end-to-end 2D SU(2) and SU(3) flows with their acceptance checks |
| `tests/test_lattice.py` | §9 | the 64 acceptance checks |
| `tests/test_lattice_api.py`, `test_lattice_flow_smoke.py` | | even-L check, inverse, Bessel forms, exports; smoke test of the lattice example |

MATH_SPEC.md refers to two files that are not in this repository:
- `lagmath.py` is now the `lag` package, with the same function names.
- `verify_math_spec.py` is now `tests/test_math_spec.py`.

MATH_SPEC_LATTICE.md also refers to files that are not in this repository:
- `laglattice.py` is now `lag.lattice`, with the same function names.
- `verify_lattice.py` is now `tests/test_lattice.py`.
- `lattice_flow_2d.py` and `eval_saved.py` are now `examples/lattice_flow_2d.py`.
- The records that `fill_spec_lattice.py` generates the spec from, and the trained §8 models, stay outside the
  repository.

All public names of the modules in the first table are also re-exported from `lag`. The lattice names stay in
`lag.lattice`, which `import lag` loads.

## Plugging in a group

A group is defined by one of three things: generators in a faithful matrix representation, the invariant
tensors it stabilises, or its structure constants. Everything else is derived from the generators: hat/vee,
projection, bracket, `ad`, `exp`, `dexpL`, the action, and the closure check.

```python
import itertools
import torch
from lag.groups import MatrixLieGroup, SU3
from lag.flows import rkmk4
from lag.heads import wedge_connection

# G2: the stabiliser of the associative 3-form on R^7
phi = torch.zeros(7, 7, 7, dtype=torch.float64)
terms = {(1, 2, 3): 1, (1, 4, 5): 1, (1, 6, 7): 1, (2, 4, 6): 1, (2, 5, 7): -1, (3, 4, 7): -1, (3, 5, 6): -1}
for (i, j, k), s in terms.items():
    idx = (i - 1, j - 1, k - 1)
    for p in itertools.permutations(range(3)):
        sign = 1 if p in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
        phi[idx[p[0]], idx[p[1]], idx[p[2]]] = s * sign
G2 = MatrixLieGroup.from_invariants(7, tensors=(phi,), name="G2")
assert G2.dim == 14 and G2.closure_residual() < 1e-12

# Any centreless algebra from structure constants f[a, b, c] ([e_a, e_b] = f_abc e_c), in the adjoint
# representation. E8: MatrixLieGroup.from_structure_constants(f_e8, name="E8")  -> 248 x 248 matrices
adj = MatrixLieGroup.from_structure_constants(SU3().f, name="su(3) adjoint")

# Closed forms plug in as backends with signature (group, w) -> tensor:
#   MatrixLieGroup(generators, "MyGroup", closed_form_exp=my_exp, closed_form_dexpL=my_dexpL)

# Flows only need group.act and group.bracket
Z = torch.randn(3, 7, dtype=torch.float64)
field = wedge_connection(G2, lambda gram: 0.3 * torch.tanh(gram))
Z1 = rkmk4(G2, field, Z, 0.1)
assert torch.allclose(Z1.norm(dim=-1), Z.norm(dim=-1))      # G2 acts by rotations of R^7
```

Notes:
- `from_invariants` treats tensors as covariant (forms): it keeps A when T(g u, g v, ...) = T(u, v, ...). Its
  memory grows like N⁴, so build large algebras such as E8 with `from_structure_constants`.
  `closure_residual()` and the lazily computed `f` run in O(d N²) memory.
- Without closed forms, `exp` uses `torch.linalg.matrix_exp` (`exp_backend="pade"`) or an eigendecomposition
  (`"spectral_skew"`, which needs real skew-symmetric or anti-Hermitian generators). `dexpL` uses the
  `2d × 2d` block exponential of §2.3.
- The constructor rejects wrongly shaped or linearly dependent generators. Closure is checked on demand
  with `closure_residual()`.
- `wedge_connection` is equivariant only for real groups with skew-symmetric generators (G ⊂ O(N)).
- The alternative constructors return `cls(...)`, so they are meant for `MatrixLieGroup` and for subclasses
  that keep its `__init__` signature. The built-in `SO3`/`SU2`/`SU3`/`SE3` do not.

## Tests

```bash
python -m pytest tests/ -v
```

`tests/test_math_spec.py` is the acceptance suite ported from `verify_math_spec.py`. It has one pytest item
per recorded result (47), with the same seeds and tolerances, and the tolerances are pinned. Run as a script,
it writes `verify_results.json` to the current directory, like the original:

```bash
python tests/test_math_spec.py
```

`tests/test_lattice.py` is the acceptance suite ported from `verify_lattice.py`. It has one pytest item per
recorded result (64), with the same seeds. The tolerances are pinned together with their direction: negative
controls are lower bounds. Run as a script, it writes `verify_lattice_results.json` to the current directory and
exits with status 1 if a check fails, like the original:

```bash
python tests/test_lattice.py
```

## Differences from the reference implementation

### `su3_exp`: `matrix_exp` fallback near degeneracy and zero

The reference's derivatives depend on the torch version at exactly degenerate eigenvalues. There `c0/(2s³)`
can round to exactly ±1, and `arccos' = ∞` then meets a zero upstream gradient. On torch 2.11 the unmodified
reference fails one check (`grad through su3 closed form: non-finite count` is 7 of 100), and 14 of 300
exactly degenerate inputs give NaN first derivatives. On torch 2.14.1 both counts are 0. Second derivatives
at those inputs are NaN in about 57 % of cases on both versions.

`lag.exp.su3_exp` keeps the closed form for generic elements and sends the rest to `torch.linalg.matrix_exp`,
which is smooth to every order. "The rest" means near-zero elements (`c1 < 1e-12`) and near-degenerate ones
(`1 − |cos 3φ| ≤ 1e-6`, a relative eigenvalue gap below about 1.6e-3). The routing mask is computed without
gradient. The reference's Taylor branch and clamp are no longer used. Measured on torch 2.11 and 2.14.1:
- forward values and gradients on generic inputs are bit-for-bit those of the reference (batched input), and
  none of 4,000 random elements is routed to `matrix_exp`;
- first and second derivatives are finite everywhere. Over 300 exactly degenerate inputs, Hessian-vector
  products equal those of `matrix_exp`, and across eigenvalue gaps from 1e-1 to 0 the worst relative
  Hessian error is 3.8e-12;
- it is faster than the previous version. On torch 2.11 with 6 CPU threads, 16,384 matrices take 24 ms
  forward plus backward (previously 35 ms; `matrix_exp` 67 ms), and 61 ms with a double backward (previously
  84 ms; `matrix_exp` 268 ms). torch 2.14.1 gives 25 ms and 64 ms.

Consequences:
- Three §8 checks (`exp su3 closed: norm x1e-7`, `exp su3 closed: degenerate eigenvalues`,
  `grad through su3 closed form at X = 0`) now exercise the `matrix_exp` fallback rather than the closed form,
  so they record 0.
- The routing is data-dependent control flow, so `torch.func.vmap` cannot wrap `su3_exp`, and
  `torch.compile` traces it with a graph break.

### `MatrixLieGroup` (from `lag_proto.py`)

- Complex generators are accepted: `vee` and `project` use the real Frobenius inner product Re tr(AᴴB). For
  real generators this is bit-for-bit the prototype.
- `hat` casts real coordinates.
- The `spectral_skew` backend accepts anti-Hermitian matrices.
- `f` and `closure_residual` are evaluated one generator at a time. The math is the same; memory is
  O(d N²) instead of O(d² N²).
- The constructor validates its input.
- New members: `ad`, `dexpL`, `from_generators`, and the `closed_form_dexpL` backend.

### Test port

The four results that the original script only recorded now assert the qualitative claims of MATH_SPEC.md:
- §3.5: no folding for the small field, folding for the large one;
- §4.2: the d-type covariant vanishes for SU(2) but not for SU(3).

### `lag.lattice` (from `laglattice.py`)

With docstrings and annotations stripped, the AST of every function and class of `laglattice.py` is identical
to its counterpart in `lag.lattice`, with one exception, `mask`. The differences:
- `mask` raises `ValueError` for odd L, the even-L assertion that MATH_SPEC_LATTICE.md §11 asks for. On an odd
  torus the transverse parity wraps around, the active set is not independent, and the log-determinant is
  silently wrong (the odd-L negative controls of §9). Without the check, `mask` is the reference function.
  Callers that need the parity set anyway pass it through `active`.
- `invert_masked_layer` is the fixed-point iteration of §6. It is moved out of `sec_inverse` of
  `verify_lattice.py` with the same arithmetic. Its loop is AST-identical to the original's once `cf`, `1e-14`
  and `200` are renamed `coeff`, `tol` and `max_iter`. It returns the input field, the number of iterations and
  the last change, which exceeds `tol` only if `max_iter` was reached.
- The Bessel forms of §7 are new code. The context files record their checks (`exact_bessel_checks.json`) but
  not their implementation, so they are written from the §7 formulas, with I_n from its power series. At the 12
  recorded points they agree with `exact_logZ_2d` to 3.7e-14 (recorded: 3.8e-14), and `bessel_i` agrees with
  `scipy.special.iv` to 1.8e-15 (relative, n < 40, x ≤ 30).
- `head`, the loop head summed over the loops of a link, lives in `field.py` rather than `layers.py`, because
  `validate_mask` needs it too. It differs from `lag.heads.head_loops`, which does not sum.
- As in the reference, `GaugeGroup(N)` builds SU(2) for N = 2 and SU(3) for any other N, so only N = 2 and 3
  are meaningful.

### Lattice test port

`tests/test_lattice.py` keeps the section functions of `verify_lattice.py`. Their AST differs only where the
library changed:
- the two odd-L negative controls pass the parity set through `active`;
- `sec_inverse` calls `invert_masked_layer`;
- `check()` also records the tolerance.

The 11 results that the original only records assert the qualitative claims of MATH_SPEC_LATTICE.md:
- §3.2: a full sweep changes every link.
- §6: the inverse converges before its cap of 200 iterations, with max ‖Dω‖₂ < 1.
- §7: the Monte Carlo triples are finite with positive errors, and the trivial-representation-only log Z of the
  2×2 tori lies below the exact value.
- §4: the volume table has the expected shape.

Fidelity, on torch 2.11: `python tests/test_lattice.py` and a fresh run of the original `verify_lattice.py`
give the same 62 non-timing values bit for bit. Against the recorded file (torch 2.14.1), 34 of them differ at
round-off, by at most 1.1e-13. The fresh run of the original differs in the same 34 values.

### Lattice example (from `lattice_flow_2d.py` and `eval_saved.py`)

The AST of `Coeff`, `Flow`, `evaluate` and the training loop is identical to `lattice_flow_2d.py`.
`evaluate_snis` is the per-L loop body of `eval_saved.py`. The seeds are the same: training draws from
`--seed`, and the two evaluation sets from 1000 + L and 5000 + L. A 30-step run of both reference scripts and
the same run of the example give identical training histories and evaluations, bit for bit, for SU(2) and
SU(3). What is new:
- the acceptance checks of §8. "Reverse KL ≥ 0 within noise" means at least −3 standard errors of mean log w.
  The reweighted plaquette is checked on set 2, which has the self-normalised importance-sampling error.
- `--load`, which evaluates a saved model and replaces `eval_saved.py`;
- the per-group defaults (β and the number of steps of the §8 runs);
- `--out`: files are written only on request;
- `run()` restores the default dtype and the thread count when it returns.

## Status

Build order (MATH_SPEC.md §10):
- **Done:** steps 1–3, 5 and 6. Step 4 is done except for the optional custom backward of §5, whose reference
  sweep lives in `tests/test_math_spec.py::sec_adjoint`.
- Step 5 is `examples/su3_single_variable_flow.py`, the §7 run (`python examples/su3_single_variable_flow.py`;
  options `--steps`, `--seed`, `--batch`). With its defaults (seed 0, batch 1024, 350 Adam steps, about 5 to
  6 minutes of training on 6 CPU threads) it reaches ESS 99.96 % on both sets of 32,768 samples,
  log Z = 1.23602 ± 0.00011 and 1.23588 ± 0.00011 (exact 1.23589), all det J > 0 and max ‖Dω‖₂ = 0.31. These
  numbers are identical on torch 2.11 and 2.14.1, before and after the `su3_exp` fallback. The default suite
  runs only `tests/test_example_smoke.py`; set `LAG_FULL_FLOW=1` to include the full run.

The open items of MATH_SPEC.md §9 are unchanged except item 2. Masking patterns and the total log-determinant of
multi-link lattice layers are now specified and verified in MATH_SPEC_LATTICE.md.

Build order of MATH_SPEC_LATTICE.md §11: steps 1–7 are done.
- Steps 1–5 are the `lag.lattice` package.
- Step 7 is `tests/test_lattice.py`, which passes all 64 checks. `tests/test_lattice_api.py` covers what the
  port does not reach:
  - the even-L check;
  - `invert_masked_layer` against the loop it replaces;
  - the Bessel forms against `exact_logZ_2d`.
- Step 6 is `examples/lattice_flow_2d.py`, the §8 run (`python examples/lattice_flow_2d.py --N 3`, or `--N 2`).
  Training was run only briefly here; its first step reproduces the recorded §8 KL exactly. The trained models
  of the §8 runs were evaluated with `--load`. Both pass every acceptance check. Both evaluation sets reproduce
  the §8 tables, within 1e-12 relative of the recorded values (torch 2.11 against 2.14.1). The ESS thresholds
  come from single-seed runs; §8 asks for a second seed before pinning them, and that has not been run.
- Cost of a full run on 6 CPU threads. §8 reports 1705 s of training for SU(3) and 833 s for SU(2); the two
  evaluation sets add about 21 and 7 minutes. On the development machine (torch 2.11), a training step at
  batch 256 takes 3.3 s for SU(3) and 1.0 s for SU(2): about 17 and 7 minutes for the §8 step counts. Evaluation
  took about 25 and 4 minutes there.
- The default suite runs only `tests/test_lattice_flow_smoke.py`. Set `LAG_FULL_LATTICE_FLOW=1` to include both
  full runs.

The open items of MATH_SPEC_LATTICE.md §10 are unchanged.
