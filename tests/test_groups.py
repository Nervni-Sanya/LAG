"""Tests of lag.groups, lag.flows and pluggability of custom groups (MATH_SPEC.md sections 1-3, 10).

All tensors are float64 / complex128 with explicit dtypes; every test is seeded.
Round-off tolerance in float64 is about 1e-12 unless stated otherwise.
"""
from __future__ import annotations

import math
import pickle
from itertools import permutations

import pytest
import torch
from torch import Tensor

from lag.algebra import T2, T3, proj_su, se3_ad, se3_hat, se3_vee_first_order, so3_hat, so3_vee, su_hat, su_vee
from lag.exp import dexpL_generic, dexpL_se3, dexpL_so3, se3_exp, so3_exp, su2_exp, su3_exp
from lag.flows import lie_euler, rkmk4
from lag.groups import SE3, SO3, SU2, SU3, LieGroupLike, MatrixLieGroup
from lag.heads import wedge_connection

F64 = torch.float64
C128 = torch.complex128
TOL = 1e-12


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _randn(*shape: int, seed: int = 0, scale: float = 1.0, dtype: torch.dtype = F64) -> Tensor:
    return scale * torch.randn(*shape, generator=_gen(seed), dtype=dtype)


def _g2_phi() -> Tensor:
    phi = torch.zeros(7, 7, 7, dtype=F64)
    terms = {(1, 2, 3): 1, (1, 4, 5): 1, (1, 6, 7): 1, (2, 4, 6): 1, (2, 5, 7): -1, (3, 4, 7): -1, (3, 5, 6): -1}
    for idx, s in terms.items():
        for perm in permutations(range(3)):
            sign = torch.linalg.det(torch.eye(3, dtype=F64)[list(perm)]).round().item()
            phi[tuple(idx[p] - 1 for p in perm)] = s * sign
    return phi


@pytest.fixture(scope="module")
def G2() -> MatrixLieGroup:
    return MatrixLieGroup.from_invariants(7, tensors=(_g2_phi(),), name="G2")


@pytest.fixture(scope="module")
def so3_inv() -> MatrixLieGroup:
    """so(3) as the stabiliser of the metric and of the volume form (rotations)."""
    eps = torch.zeros(3, 3, 3, dtype=F64)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    return MatrixLieGroup.from_invariants(3, tensors=(torch.eye(3, dtype=F64), eps), name="so3")


@pytest.fixture(scope="module")
def su3() -> SU3:
    return SU3()


def _builtin_cases():
    return [("SO3", SO3, 3), ("SU2", SU2, 3), ("SU3", SU3, 8), ("SE3", SE3, 6)]


def _batches(d: int, seed: int) -> list[Tensor]:
    """A generic batch and a tiny batch with |w| ~ 1e-4."""
    big = _randn(5, d, seed=seed)
    tiny = 1e-4 * _randn(4, d, seed=seed + 1) / math.sqrt(d)
    return [big, tiny]


def _hat_ref(name: str, w: Tensor) -> Tensor:
    return {"SO3": lambda: so3_hat(w), "SU2": lambda: su_hat(w, T2),
            "SU3": lambda: su_hat(w, T3), "SE3": lambda: se3_hat(w)}[name]()


def _exp_ref(name: str, w: Tensor) -> Tensor:
    return {"SO3": lambda: so3_exp(w), "SU2": lambda: su2_exp(w),
            "SU3": lambda: su3_exp(su_hat(w, T3)), "SE3": lambda: se3_exp(w)}[name]()


# ---------------------------------------------------------------------------------------------
# 1. built-in groups vs the reference formulas
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls,d", _builtin_cases())
def test_builtin_hat_vee(name, cls, d):
    G = cls()
    assert G.dim == d
    for w in _batches(d, 10):
        X = G.hat(w)
        assert torch.allclose(X, _hat_ref(name, w), atol=TOL, rtol=0)
        assert torch.allclose(G.vee(X), w, atol=TOL, rtol=0)
    # vee against the reference vee maps on random algebra elements
    w = _randn(6, d, seed=3)
    X = _hat_ref(name, w)
    ref = {"SO3": lambda: so3_vee(X), "SU2": lambda: su_vee(X, T2), "SU3": lambda: su_vee(X, T3),
           "SE3": lambda: se3_vee_first_order(X)}[name]()
    assert torch.allclose(G.vee(X), ref, atol=TOL, rtol=0)
    assert torch.allclose(ref, w, atol=TOL, rtol=0)


@pytest.mark.parametrize("name,cls,d", _builtin_cases())
def test_builtin_exp(name, cls, d):
    G = cls()
    for w in _batches(d, 20):
        E = G.exp(w)
        assert torch.allclose(E, _exp_ref(name, w), atol=TOL, rtol=0)
        # float64 matrix_exp of a norm ~ 1 matrix: a few ulp
        assert torch.allclose(E, torch.linalg.matrix_exp(G.hat(w)), atol=1e-12, rtol=0)


@pytest.mark.parametrize("name,cls,d", _builtin_cases())
def test_builtin_ad_dexpL_f(name, cls, d):
    G = cls()
    for w in _batches(d, 30):
        ad = G.ad(w)
        if name in ("SO3", "SU2"):
            assert torch.allclose(ad, so3_hat(w), atol=TOL, rtol=0)
        if name == "SE3":
            assert torch.allclose(ad, se3_ad(w), atol=TOL, rtol=0)
        D = G.dexpL(w)
        if name in ("SO3", "SU2"):
            assert torch.allclose(D, dexpL_so3(w), atol=TOL, rtol=0)
        if name == "SE3":
            assert torch.allclose(D, dexpL_se3(w), atol=TOL, rtol=0)
        # generic formula phi_1(-ad); one extra matrix_exp of size 2d, a few ulp
        assert torch.allclose(D, dexpL_generic(ad), atol=1e-11, rtol=0)
    f = G.f
    assert f.shape == (d, d, d)
    assert torch.allclose(f, -f.transpose(0, 1), atol=TOL, rtol=0)
    jac = (torch.einsum("abe,ecd->abcd", f, f) + torch.einsum("bce,ead->abcd", f, f)
           + torch.einsum("cae,ebd->abcd", f, f))
    assert jac.abs().max() < TOL


def test_su3_structure_constants(su3):
    s3 = math.sqrt(3.0)
    vals = {(0, 1, 2): 1.0, (0, 3, 6): 0.5, (0, 4, 5): -0.5, (1, 3, 5): 0.5, (1, 4, 6): 0.5,
            (2, 3, 4): 0.5, (2, 5, 6): -0.5, (3, 4, 7): s3 / 2, (5, 6, 7): s3 / 2}
    expected = torch.zeros(8, 8, 8, dtype=F64)
    for idx, v in vals.items():
        for perm in permutations(range(3)):
            sign = torch.linalg.det(torch.eye(3, dtype=F64)[list(perm)]).round().item()
            expected[tuple(idx[p] for p in perm)] = sign * v
    assert torch.allclose(su3.f, expected, atol=TOL, rtol=0)


# ---------------------------------------------------------------------------------------------
# 2. projection
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls,N,complex_", [(SO3, 3, False), (SU3, 3, True)])
def test_projection_idempotent_orthogonal(cls, N, complex_):
    G = cls()
    X = _randn(4, N, N, seed=40, dtype=C128 if complex_ else F64)
    P = G.project(X)
    assert torch.allclose(G.project(P), P, atol=TOL, rtol=0)
    R = X - P
    # real Frobenius inner product Re tr(A^H B) against every generator
    ip = torch.einsum("aij,...ij->...a", G.G.conj(), R).real
    assert ip.abs().max() < TOL


@pytest.mark.parametrize("N", [2, 3])
def test_su_projection_matches_proj_su(N):
    G = SU2() if N == 2 else SU3()
    X = _randn(5, N, N, seed=41, dtype=C128)
    assert torch.allclose(G.project(X), proj_su(X), atol=TOL, rtol=0)


# ---------------------------------------------------------------------------------------------
# 3. custom groups from invariants
# ---------------------------------------------------------------------------------------------
def test_g2_from_invariants(G2):
    phi = _g2_phi()
    assert G2.dim == 14 and G2.N == 7
    assert G2.closure_residual() < 1e-12
    assert torch.allclose(G2.G, -G2.G.transpose(-1, -2), atol=TOL, rtol=0)      # g2 inside so(7)
    w = _randn(4, 14, seed=50, scale=0.7)
    E = G2.exp(w)
    I7 = torch.eye(7, dtype=F64)
    assert torch.allclose(E @ E.transpose(-1, -2), I7.expand_as(E), atol=TOL, rtol=0)
    assert torch.allclose(torch.linalg.det(E), torch.ones(4, dtype=F64), atol=TOL, rtol=0)
    for g in E:
        assert torch.allclose(torch.einsum("ijk,ia,jb,kc->abc", phi, g, g, g), phi, atol=1e-12, rtol=0)
    # every generator annihilates phi infinitesimally: sum over slots of phi(.., A v_s, ..) = 0
    for A in G2.G:
        dphi = (torch.einsum("ijk,ia->ajk", phi, A) + torch.einsum("ijk,jb->ibk", phi, A)
                + torch.einsum("ijk,kc->ijc", phi, A))
        assert dphi.abs().max() < TOL
    wh = _randn(14, seed=51)
    assert torch.allclose(G2.vee(G2.hat(wh)), wh, atol=TOL, rtol=0)
    a, b = _randn(3, 14, seed=52), _randn(3, 14, seed=53)
    assert torch.allclose(G2.bracket(a, b), -G2.bracket(b, a), atol=TOL, rtol=0)
    # bracket agrees with the matrix commutator
    Xa, Xb = G2.hat(a), G2.hat(b)
    assert torch.allclose(G2.hat(G2.bracket(a, b)), Xa @ Xb - Xb @ Xa, atol=1e-11, rtol=0)


def test_so4_from_metric():
    G = MatrixLieGroup.from_invariants(4, tensors=(torch.eye(4, dtype=F64),), name="so4")
    assert G.dim == 6
    assert G.closure_residual() < 1e-12
    E = G.exp(_randn(3, 6, seed=60))
    assert torch.allclose(E @ E.transpose(-1, -2), torch.eye(4, dtype=F64).expand_as(E), atol=TOL, rtol=0)


def test_sp4R_from_symplectic_form():
    J = torch.zeros(4, 4, dtype=F64)
    J[0, 2] = J[1, 3] = 1
    J[2, 0] = J[3, 1] = -1
    G = MatrixLieGroup.from_invariants(4, tensors=(J,), name="sp4R")
    assert G.dim == 10
    assert G.closure_residual() < 1e-12
    w = _randn(3, 10, seed=61, scale=0.5)
    E = G.exp(w)
    # g^T J g = J  (non-compact: E is not orthogonal)
    lhs = E.transpose(-1, -2) @ J @ E
    # entries of E are O(1..10): relative round-off still < 1e-11
    assert torch.allclose(lhs, J.expand_as(lhs), atol=1e-11, rtol=0)
    assert (E @ E.transpose(-1, -2) - torch.eye(4, dtype=F64)).abs().max() > 1e-3


# ---------------------------------------------------------------------------------------------
# 4. custom groups from structure constants (adjoint representation)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["G2", "SU3"])
def test_from_structure_constants(which, G2, su3):
    G = G2 if which == "G2" else su3
    A = MatrixLieGroup.from_structure_constants(G.f, name="adj")
    assert A.dim == G.dim
    assert A.closure_residual() < 1e-11
    a, b = _randn(3, G.dim, seed=70), _randn(3, G.dim, seed=71)
    assert torch.allclose(A.bracket(a, b), G.bracket(a, b), atol=TOL, rtol=0)
    w = _randn(3, G.dim, seed=72)
    assert torch.allclose(A.hat(w), A.ad(w), atol=TOL, rtol=0)
    assert torch.allclose(A.ad(w), G.ad(w), atol=TOL, rtol=0)
    v = _randn(3, G.dim, seed=73)
    lhs = (A.exp(0.5 * w) @ v.unsqueeze(-1)).squeeze(-1)
    Eg = G.exp(0.5 * w)
    rhs = G.vee(Eg @ G.hat(v) @ torch.linalg.inv(Eg))
    # exp(ad_w) = Ad_exp(w): both sides involve a matrix_exp / inverse, a few ulp
    assert torch.allclose(lhs, rhs, atol=1e-11, rtol=0)


# ---------------------------------------------------------------------------------------------
# 5. dexpL convention: E(w)^-1 dE/dw_b == hat(dexpL(w)[:, b])
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["G2", "SU3", "SE3"])
def test_dexpL_convention_finite_differences(which, G2):
    G = {"G2": lambda: G2, "SU3": SU3, "SE3": SE3}[which]()
    d = G.dim
    w = _randn(d, seed=80, scale=0.6)
    D = G.dexpL(w)
    Einv = torch.linalg.inv(G.exp(w))
    eps = 1e-6
    for b in range(d):
        e = torch.zeros(d, dtype=F64)
        e[b] = eps
        dE = (G.exp(w + e) - G.exp(w - e)) / (2 * eps)
        # central difference truncation error ~ eps^2 * E''' ~ 1e-12, round-off ~ 1e-16 / eps ~ 1e-10
        assert torch.allclose(Einv @ dE, G.hat(D[:, b]), atol=1e-8, rtol=0), f"column {b}"


# ---------------------------------------------------------------------------------------------
# 6. exp backends
# ---------------------------------------------------------------------------------------------
def test_backends_real_skew(so3_inv):
    assert so3_inv.dim == 3
    w = _randn(5, 3, seed=90)
    X = so3_inv.hat(w)
    spec = MatrixLieGroup(so3_inv.G, "s", exp_backend="spectral_skew").exp(w)
    pade = MatrixLieGroup(so3_inv.G, "p", exp_backend="pade").exp(w)
    assert spec.dtype == F64
    # the invariant basis is a rotation of so3_hat's basis; compare via the generators themselves
    assert torch.allclose(spec, pade, atol=TOL, rtol=0)
    assert torch.allclose(spec, torch.linalg.matrix_exp(X), atol=TOL, rtol=0)
    closed = SO3()
    w2 = _randn(5, 3, seed=91)
    spec2 = MatrixLieGroup(closed.G, "s", exp_backend="spectral_skew").exp(w2)
    assert torch.allclose(spec2, closed.exp(w2), atol=TOL, rtol=0)


@pytest.mark.parametrize("T,ref,d", [(T3, lambda w: su3_exp(su_hat(w, T3)), 8), (T2, su2_exp, 3)])
def test_backends_complex_antihermitian(T, ref, d):
    w = _randn(5, d, seed=92)
    spec = MatrixLieGroup(T, "su", exp_backend="spectral_skew").exp(w)
    pade = MatrixLieGroup(T, "su", exp_backend="pade").exp(w)
    assert spec.is_complex()
    assert torch.allclose(spec, ref(w), atol=TOL, rtol=0)
    assert torch.allclose(spec, pade, atol=TOL, rtol=0)


def test_closure_failure_and_bad_backend():
    two = so3_hat(torch.eye(3, dtype=F64))[:2]
    assert MatrixLieGroup(two, "not closed").closure_residual() > 0.1
    with pytest.raises(ValueError):
        MatrixLieGroup(so3_hat(torch.eye(3, dtype=F64)), "x", exp_backend="nope")


# ---------------------------------------------------------------------------------------------
# 7. flows on G2 acting on R^7
# ---------------------------------------------------------------------------------------------
def _coeff_fn(K: int, seed: int):
    M = _randn(K * K, K * K, seed=seed, scale=1.0 / K)

    def coeff(gram: Tensor) -> Tensor:
        return 0.3 * torch.tanh(M @ gram.reshape(-1)).reshape(K, K)
    return coeff


@pytest.mark.parametrize("stepper", [lie_euler, rkmk4])
def test_flow_norm_preservation_and_equivariance(stepper, G2):
    K = 3
    field = wedge_connection(G2, _coeff_fn(K, 100))
    Z = _randn(K, 7, seed=101)
    h = 0.1
    z = Z
    for _ in range(5):
        z = stepper(G2, field, z, h)
        assert torch.allclose(z.norm(dim=-1), Z.norm(dim=-1), atol=1e-13, rtol=0)
    g = G2.exp(_randn(14, seed=102, scale=0.5))
    lhs = stepper(G2, field, Z @ g.T, h)
    rhs = stepper(G2, field, Z, h) @ g.T
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------------------------
# 8. convergence order (MATH_SPEC 10, step 4)
# ---------------------------------------------------------------------------------------------
def _integrate(stepper, G, field, Z0: Tensor, n: int) -> Tensor:
    z, h = Z0, 1.0 / n
    for _ in range(n):
        z = stepper(G, field, z, h)
    return z


_REF_CACHE: dict[tuple[float, ...], Tensor] = {}


def _so3_field():
    coeff = _coeff_fn(3, 110)
    # strongly non-constant coefficients so the field is genuinely nonlinear
    return wedge_connection(SO3(), lambda gram: 3.0 * coeff(gram) / 0.3)


def _reference(Z0: Tensor) -> Tensor:
    key = tuple(Z0.flatten().tolist())
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _integrate(rkmk4, SO3(), _so3_field(), Z0, 512)
    return _REF_CACHE[key]


@pytest.mark.parametrize("stepper,n,lo,hi", [(rkmk4, 16, 3.5, 4.5), (lie_euler, 64, 0.8, 1.2)])
def test_convergence_order_so3(stepper, n, lo, hi):
    G = SO3()
    field = _so3_field()
    Z0 = _randn(3, 3, seed=111)
    # reference: rkmk4 with 512 steps (error ~ 16^-4 of the 32-step rkmk4 error: negligible); cached
    ref = _reference(Z0)
    e1 = (_integrate(stepper, G, field, Z0, n) - ref).abs().max()
    e2 = (_integrate(stepper, G, field, Z0, 2 * n) - ref).abs().max()
    order = math.log2(float(e1 / e2))
    # errors are far above round-off (1e-15); reference error is negligible against e2
    assert e1 > 1e-9 and e2 > 1e-10
    assert lo <= order <= hi, (order, float(e1), float(e2))


# ---------------------------------------------------------------------------------------------
# 9. protocol, repr, pickle
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls,d", _builtin_cases())
def test_protocol_repr_pickle(name, cls, d):
    G = cls()
    assert isinstance(G, LieGroupLike)
    r = repr(G)
    assert G.name in r and f"dim={d}" in r
    G2_ = pickle.loads(pickle.dumps(G))
    assert type(G2_) is cls and G2_.dim == d and G2_.name == G.name
    w = _randn(2, d, seed=120)
    assert torch.allclose(G2_.exp(w), G.exp(w), atol=0, rtol=0)


def test_custom_group_protocol_and_repr(G2):
    assert isinstance(G2, LieGroupLike)
    assert "G2" in repr(G2) and "dim=14" in repr(G2)


# ---------------------------------------------------------------------------------------------
# 10. constructor validation
# ---------------------------------------------------------------------------------------------
def _sp4_generators() -> Tensor:
    J = torch.zeros(4, 4, dtype=F64)
    J[0, 2] = J[1, 3] = 1
    J[2, 0] = J[3, 1] = -1
    return MatrixLieGroup.from_invariants(4, tensors=(J,), name="sp4R").G


def test_ctor_rejects_linearly_dependent_generators():
    base = so3_hat(torch.eye(3, dtype=F64))
    dep = torch.cat([base, (base[0] + base[1]).unsqueeze(0)])          # 4 generators spanning 3 dimensions
    with pytest.raises(ValueError, match="linearly dependent"):
        MatrixLieGroup(dep, "dep")
    assert MatrixLieGroup(base, "ok").dim == 3                        # the independent basis is accepted


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 4), (2, 3, 3, 3)])
def test_ctor_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match=r"\(d, N, N\)"):
        MatrixLieGroup(_randn(*shape, seed=200), "bad")


def test_ctor_rejects_abelian_structure_constants():
    # adjoint representation of an abelian algebra is zero: not faithful, Gram matrix singular
    with pytest.raises(ValueError, match="linearly dependent"):
        MatrixLieGroup.from_structure_constants(torch.zeros(2, 2, 2, dtype=F64), "abelian")


def test_ctor_spectral_skew_validation():
    sp = _sp4_generators()
    with pytest.raises(ValueError, match="spectral_skew"):
        MatrixLieGroup(sp, "sp4", exp_backend="spectral_skew")
    assert MatrixLieGroup(sp, "sp4", exp_backend="pade").dim == 10   # the same generators are fine with pade
    # real skew-symmetric and anti-Hermitian generators are accepted
    assert MatrixLieGroup(so3_hat(torch.eye(3, dtype=F64)), "so3", exp_backend="spectral_skew").dim == 3
    assert MatrixLieGroup(T3, "su3", exp_backend="spectral_skew").dim == 8


# ---------------------------------------------------------------------------------------------
# 11. act
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["SO3", "G2", "SU3"])
def test_act_equals_exp_matvec(which, G2):
    G = {"SO3": SO3, "G2": lambda: G2, "SU3": SU3}[which]()
    B = 4
    w = _randn(B, G.dim, seed=210)
    z = _randn(B, G.N, seed=211, dtype=C128 if which == "SU3" else F64)
    E = G.exp(w)
    ref = torch.einsum("bij,bj->bi", E, z)               # explicit contraction, no act involved
    out = G.act(w, z)
    assert out.shape == (B, G.N) and out.dtype == z.dtype
    # exp of norm ~ 1..3 matrices, a few ulp
    assert torch.allclose(out, ref, atol=TOL, rtol=0), which
    # a single vector is acted on by every group element of the batch
    z0 = z[0]
    assert torch.allclose(G.act(w, z0), torch.einsum("bij,j->bi", E, z0), atol=TOL, rtol=0), which
    # unbatched
    assert torch.allclose(G.act(w[1], z[1]), E[1] @ z[1], atol=TOL, rtol=0), which


def test_so3_act_convention_pin():
    e1 = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    e3 = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    for theta in (0.3, 1.1, 2.5, -0.7):
        out = SO3().act(theta * e3, e1)
        ref = torch.tensor([math.cos(theta), math.sin(theta), 0.0], dtype=F64)   # counter-clockwise rotation
        assert torch.allclose(out, ref, atol=TOL, rtol=0), theta


# ---------------------------------------------------------------------------------------------
# 12. from_invariants with "commuting"
# ---------------------------------------------------------------------------------------------
def _complex_structure() -> Tensor:
    J = torch.zeros(4, 4, dtype=F64)
    J[:2, 2:] = -torch.eye(2, dtype=F64)
    J[2:, :2] = torch.eye(2, dtype=F64)
    return J


def test_from_invariants_commuting_u2():
    J = _complex_structure()
    G = MatrixLieGroup.from_invariants(4, tensors=(torch.eye(4, dtype=F64),), commuting=(J,), name="u2")
    assert G.dim == 4
    assert G.closure_residual() < 1e-12
    for A in G.G:
        assert (A @ J - J @ A).abs().max() < 1e-12
        assert (A + A.T).abs().max() < 1e-12                   # inside so(4)
    E = G.exp(_randn(3, 4, seed=220))
    assert torch.allclose(E @ J, J @ E, atol=1e-12, rtol=0)    # group elements are complex-linear unitary


def test_from_invariants_commuting_only_gl2c():
    J = _complex_structure()
    G = MatrixLieGroup.from_invariants(4, commuting=(J,), name="gl2c")
    assert G.dim == 8                                          # gl(2, C) realified: 2 * 2^2 real dimensions
    assert G.closure_residual() < 1e-12
    for A in G.G:
        assert (A @ J - J @ A).abs().max() < 1e-12
    # not all skew: the centraliser contains the identity (a symmetric matrix)
    assert G.vee(torch.eye(4, dtype=F64)).abs().max() > 0.1
    assert (G.project(torch.eye(4, dtype=F64)) - torch.eye(4, dtype=F64)).abs().max() < 1e-12


# ---------------------------------------------------------------------------------------------
# 13. from_generators and the dtype branches of vee
# ---------------------------------------------------------------------------------------------
def test_from_generators_matches_su3(su3):
    G = MatrixLieGroup.from_generators(T3, "su3")
    assert G.dim == 8 and G.name == "su3" and G.G.is_complex()
    w = _randn(4, 8, seed=230)
    assert torch.allclose(G.hat(w), su3.hat(w), atol=TOL, rtol=0)
    X = su3.hat(w)
    assert torch.allclose(G.vee(X), su3.vee(X), atol=TOL, rtol=0)
    assert torch.allclose(G.vee(X), w, atol=TOL, rtol=0)
    assert torch.allclose(G.f, su3.f, atol=TOL, rtol=0)


def test_vee_real_skew_matrix_with_complex_generators():
    G = MatrixLieGroup.from_generators(T3, "su3")
    w = _randn(5, 3, seed=231)
    X = so3_hat(w)                                              # real skew-symmetric 3x3 lies in su(3)
    assert not X.is_complex()
    c = G.vee(X)
    assert c.dtype == F64
    assert torch.allclose(G.hat(c), X.to(C128), atol=TOL, rtol=0)
    # T_1, T_4, T_6 = -(i/2) lambda_2, lambda_5, lambda_7 are the real antisymmetric ones;
    # e.g. T_1 has entry (0, 1) = -1/2 and so3_hat(w)[0, 1] = -w_3  =>  c_1 = 2 w_3; similarly c_4 = -2 w_2, c_6 = 2 w_1
    expected = torch.zeros(5, 8, dtype=F64)
    expected[:, 1] = 2 * w[:, 2]
    expected[:, 4] = -2 * w[:, 1]
    expected[:, 6] = 2 * w[:, 0]
    assert torch.allclose(c, expected, atol=TOL, rtol=0)


def test_vee_complex_matrix_with_real_generators_uses_real_part():
    G = SO3()
    Z = _randn(5, 3, 3, seed=232, dtype=C128)
    ref = so3_vee((Z.real - Z.real.transpose(-1, -2)) / 2)     # skew part of the real part
    assert torch.allclose(G.vee(Z), ref, atol=TOL, rtol=0)
    assert torch.allclose(G.vee(Z), G.vee(Z.real), atol=TOL, rtol=0)
    # the imaginary part alone is invisible
    assert G.vee(1j * Z.imag).abs().max() < TOL


# ---------------------------------------------------------------------------------------------
# 14. chunked structure constants and closure residual against reference formulas
# ---------------------------------------------------------------------------------------------
def _fresh_groups(G2):
    return {"G2": MatrixLieGroup(G2.G, "g2"),
            "so5": MatrixLieGroup.from_invariants(5, tensors=(torch.eye(5, dtype=F64),), name="so5"),
            "SU3": MatrixLieGroup(T3, "su3")}


@pytest.mark.parametrize("which", ["G2", "so5", "SU3"])
def test_chunked_f_and_closure_match_reference(which, G2):
    G = _fresh_groups(G2)[which]
    Gm = G.G
    C = torch.einsum("aij,bjk->abik", Gm, Gm)
    C = C - C.transpose(0, 1)                                   # C[a, b] = [G_a, G_b]
    f_ref = G.vee(C)
    assert f_ref.shape == (G.dim, G.dim, G.dim)
    # same linear algebra, different contraction order: round-off only
    assert torch.allclose(G.f, f_ref, atol=TOL, rtol=0), which
    res_ref = float((C - G.hat(G.vee(C))).abs().max())
    assert res_ref < 1e-12                                      # these are genuine Lie algebras
    assert abs(G.closure_residual() - res_ref) < 1e-13, which


def test_closure_residual_matches_reference_when_not_closed():
    two = so3_hat(torch.eye(3, dtype=F64))[:2]                  # span{L1, L2} is not closed: [L1, L2] = L3
    G = MatrixLieGroup(two, "open")
    C = torch.einsum("aij,bjk->abik", two, two)
    C = C - C.transpose(0, 1)
    res_ref = float((C - G.hat(G.vee(C))).abs().max())
    assert res_ref == pytest.approx(1.0, abs=1e-12)             # L3 has an entry of modulus 1
    assert G.closure_residual() == pytest.approx(res_ref, abs=1e-14)


# ---------------------------------------------------------------------------------------------
# 15. flows on a generic (non-equivariant) field: commutator corrections matter
# ---------------------------------------------------------------------------------------------
def _generic_field():
    W = _randn(3, 3, seed=240)
    b = _randn(3, seed=241)
    return lambda z: 1.5 * torch.tanh(W @ z + b)


def _rkmk4_spec(field, z: Tensor, h: float) -> Tensor:
    """MATH_SPEC 3.3 for SO(3): the bracket is the cross product, the action is so3_exp(w) @ z."""
    br = torch.linalg.cross

    def act(w, x):
        return so3_exp(w) @ x
    k1 = h * field(z)
    k2 = h * field(act(k1 / 2, z))
    k3 = h * field(act(k2 / 2 - br(k1, k2) / 8, z))
    k4 = h * field(act(k3, z))
    v = (k1 + 2 * k2 + 2 * k3 + k4) / 6 - br(k1, k4) / 12
    return act(v, z)


def test_flow_single_steps_match_spec():
    G, field = SO3(), _generic_field()
    z = _randn(3, seed=242)
    h = 0.4
    # the commutator terms are O(h^3) ~ 1e-2 here, so a wrong sign is far above the 1e-14 round-off level
    assert torch.allclose(rkmk4(G, field, z, h), _rkmk4_spec(field, z, h), atol=1e-14, rtol=0)
    assert torch.allclose(lie_euler(G, field, z, h), so3_exp(h * field(z)) @ z, atol=1e-14, rtol=0)
    # the corrections are visible: dropping them changes the step by far more than round-off
    k1 = h * field(z)
    k2 = h * field(so3_exp(k1 / 2) @ z)
    k3 = h * field(so3_exp(k2 / 2) @ z)
    k4 = h * field(so3_exp(k3) @ z)
    no_corr = so3_exp((k1 + 2 * k2 + 2 * k3 + k4) / 6) @ z
    assert (no_corr - rkmk4(G, field, z, h)).abs().max() > 1e-5


_SO3 = SO3()


def _integrate_vec(stepper, field, z0: Tensor, n: int) -> Tensor:
    z, h = z0, 1.0 / n
    for _ in range(n):
        z = stepper(_SO3, field, z, h)
    return z


_GEN_REF: dict[tuple[float, ...], Tensor] = {}


def _generic_reference(field, z0: Tensor) -> Tensor:
    # rkmk4 with 256 steps: error ~ 16^-4 * 8e-6 ~ 1e-10, negligible against the 8/16-step errors (1e-4, 1e-5);
    # cached per initial value (the field itself is fixed by its seeds)
    key = tuple(z0.flatten().tolist())
    if key not in _GEN_REF:
        _GEN_REF[key] = _integrate_vec(rkmk4, field, z0, 256)
    return _GEN_REF[key]


@pytest.mark.parametrize("stepper,ns,lo,hi", [(rkmk4, (8, 16), 3.5, 4.5), (lie_euler, (64, 128), 0.8, 1.2)])
def test_flow_observed_order_generic_field(stepper, ns, lo, hi):
    field = _generic_field()
    z0 = _randn(3, seed=243)
    ref = _generic_reference(field, z0)
    e1 = float((_integrate_vec(stepper, field, z0, ns[0]) - ref).abs().max())
    e2 = float((_integrate_vec(stepper, field, z0, ns[1]) - ref).abs().max())
    # errors far above round-off (1e-15) and above the reference error
    assert e1 > 1e-7 and e2 > 1e-9, (e1, e2)
    order = math.log2(e1 / e2)
    assert lo <= order <= hi, (order, e1, e2)
