"""Value tests of lag.heads: wedge_connection, head_adjoint, head_loops, head_fundamental (MATH_SPEC.md section 4).

Every expected value is written out from an independent formula (explicit loops over indices, explicit
commutators, an explicit su(N) projection); nothing is compared with the function against itself.
All tensors are float64 / complex128 with explicit dtypes; every test is seeded with a local generator.
Round-off tolerance in float64 for O(1) entries is about 1e-12 unless stated otherwise.
"""
from __future__ import annotations

from itertools import permutations

import pytest
import torch
from torch import Tensor

from lag.algebra import T2, T3, su_hat
from lag.groups import SO3, MatrixLieGroup
from lag.heads import head_adjoint, head_fundamental, head_loops, wedge_connection

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


def _group(name: str, G2: MatrixLieGroup):
    return SO3() if name == "SO3" else G2


def _proj_su_ref(M: Tensor) -> Tensor:
    """Traceless anti-Hermitian part, written out: (M - M^H)/2 minus (tr/N) * 1."""
    N = M.shape[-1]
    A = (M - M.conj().transpose(-1, -2)) / 2
    tr = torch.einsum("...ii->...", A)
    return A - tr[..., None, None] * torch.eye(N, dtype=M.dtype) / N


def _su_elements(N: int, K: int, seed: int) -> Tensor:
    """K random elements of su(N), (K, N, N) complex128, as su_hat of real coordinates."""
    T = T2 if N == 2 else T3
    return su_hat(_randn(K, N * N - 1, seed=seed), T)


def _smooth_coeff(K: int, seed: int):
    """A smooth function of the (K, K) Gram matrix with values in (K, K)."""
    M = _randn(K * K, K * K, seed=seed, scale=1.0 / K)

    def coeff(gram: Tensor) -> Tensor:
        return 0.3 * torch.tanh(M @ gram.reshape(-1)).reshape(K, K)
    return coeff


# ---------------------------------------------------------------------------------------------
# 1. wedge_connection
# ---------------------------------------------------------------------------------------------
def test_wedge_so3_value_cross_product():
    """so(3): z_a z_b^T - z_b z_a^T = hat(z_b x z_a), so field[a] = sum_b c[a,b] z_b x z_a."""
    K = 3
    Z = _randn(K, 3, seed=1)
    c = _randn(K, K, seed=2)
    out = wedge_connection(SO3(), lambda gram: c)(Z)
    assert out.shape == (K, 3) and out.dtype == F64
    expected = torch.zeros(K, 3, dtype=F64)
    for a in range(K):
        for b in range(K):
            expected[a] = expected[a] + c[a, b] * torch.linalg.cross(Z[b], Z[a])
    # a handful of multiply-adds on O(1) numbers: a few ulp
    assert torch.allclose(out, expected, atol=1e-14, rtol=0), (out - expected).abs().max()
    assert out.abs().max() > 0.1          # the expected field is not trivially zero


def test_wedge_coeff_receives_gram_matrix():
    K = 3
    Z = _randn(K, 3, seed=3)
    seen = []

    def coeff(gram):
        seen.append(gram)
        return torch.zeros(K, K, dtype=F64)
    wedge_connection(SO3(), coeff)(Z)
    expected = torch.tensor([[float((Z[a] * Z[b]).sum()) for b in range(K)] for a in range(K)], dtype=F64)
    assert len(seen) == 1 and torch.allclose(seen[0], expected, atol=1e-14, rtol=0)


def test_wedge_g2_value_explicit_vee(G2):
    K = 3
    Z = _randn(K, 7, seed=4)
    c = _randn(K, K, seed=5)
    out = wedge_connection(G2, lambda gram: c)(Z)
    assert out.shape == (K, 14)
    for a in range(K):
        expected = torch.zeros(14, dtype=F64)
        M = torch.zeros(7, 7, dtype=F64)
        for b in range(K):
            skew = torch.outer(Z[a], Z[b]) - torch.outer(Z[b], Z[a])
            expected = expected + c[a, b] * G2.vee(skew)
            M = M + c[a, b] * skew
        assert torch.allclose(out[a], expected, atol=1e-13, rtol=0), f"row {a}"
        # independent of vee: hat(out[a]) is the Frobenius-orthogonal projection of M onto g2, i.e. the residual
        # is orthogonal to every generator (this fails if the antisymmetrisation or its factor is wrong)
        resid = M - G2.hat(out[a])
        assert torch.einsum("kij,ij->k", G2.G, resid).abs().max() < TOL, f"row {a}"


@pytest.mark.parametrize("name", ["SO3", "G2"])
def test_wedge_field_equivariance(name, G2):
    """field(Z g^T) = Ad_g field(Z) row by row; the Gram matrix Z Z^T is invariant for g in O(N)."""
    G = _group(name, G2)
    N, d, K = G.N, G.dim, 3
    field = wedge_connection(G, _smooth_coeff(K, 7))
    Z = _randn(K, N, seed=8)
    g = G.exp(_randn(d, seed=9, scale=0.5))
    assert torch.allclose(g @ g.T, torch.eye(N, dtype=F64), atol=TOL, rtol=0)
    lhs = field(Z @ g.T)
    base = field(Z)
    if name == "SO3":
        rhs = torch.stack([g @ w for w in base])                       # Ad_g w = g w for so(3)
    else:
        rhs = torch.stack([G.vee(g @ G.hat(w) @ g.T) for w in base])
    assert base.abs().max() > 1e-2
    assert torch.allclose(lhs, rhs, atol=TOL, rtol=0), (lhs - rhs).abs().max()


# ---------------------------------------------------------------------------------------------
# 2. head_adjoint
# ---------------------------------------------------------------------------------------------
def _anti_ref(Xb: Tensor, Xc: Tensor) -> Tensor:
    return _proj_su_ref(1j * (Xb @ Xc + Xc @ Xb))


def _onehot_coeff(K: int, which: int, idx: tuple[int, ...]):
    c = [torch.zeros(K, K, dtype=F64), torch.zeros(K, K, K, dtype=F64), torch.zeros(K, K, K, dtype=F64)]
    c[which][idx] = 1.0
    return lambda I2: tuple(c)


@pytest.mark.parametrize("N", [2, 3])
def test_head_adjoint_I2_is_trace_form(N):
    """The invariant passed to coeff is I2[a,b] = Re tr(X_a X_b) = -(1/2) w_a . w_b in the basis tr(T_a T_b) = -delta/2."""
    K = 3
    w = _randn(K, N * N - 1, seed=10)
    Xs = su_hat(w, T2 if N == 2 else T3)
    seen = []

    def coeff(I2):
        seen.append(I2)
        return torch.zeros(K, K), torch.zeros(K, K, K), torch.zeros(K, K, K)
    out = head_adjoint(Xs, coeff)
    assert out.shape == (K, N, N) and out.dtype == C128 and out.abs().max() == 0
    assert seen[0].shape == (K, K) and not seen[0].is_complex()
    assert torch.allclose(seen[0], -0.5 * (w @ w.T), atol=1e-14, rtol=0)


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a,b", [(0, 1), (2, 0), (1, 1)])
def test_head_adjoint_c1_returns_X(N, a, b):
    K = 3
    Xs = _su_elements(N, K, seed=11)
    out = head_adjoint(Xs, _onehot_coeff(K, 0, (a, b)))
    expected = torch.zeros(K, N, N, dtype=C128)
    expected[a] = Xs[b]
    assert torch.allclose(out, expected, atol=1e-14, rtol=0), (out - expected).abs().max()


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a,b,c", [(0, 1, 2), (2, 0, 1), (1, 2, 0), (1, 0, 2), (0, 1, 1)])
def test_head_adjoint_c2_returns_commutator(N, a, b, c):
    K = 3
    Xs = _su_elements(N, K, seed=12)
    out = head_adjoint(Xs, _onehot_coeff(K, 1, (a, b, c)))
    expected = torch.zeros(K, N, N, dtype=C128)
    expected[a] = Xs[b] @ Xs[c] - Xs[c] @ Xs[b]
    assert torch.allclose(out, expected, atol=1e-13, rtol=0), (out - expected).abs().max()
    if b != c:
        assert expected[a].abs().max() > 0.1    # a genuinely non-trivial commutator, so sign/order errors show


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a,b,c", [(0, 1, 2), (2, 0, 1), (1, 0, 1), (1, 1, 1), (0, 2, 2)])
def test_head_adjoint_c3_returns_d_type_covariant(N, a, b, c):
    """i({X_b, X_c} - trace part): nonzero for SU(3), identically zero for SU(2) ({X_b, X_c} ~ 1 there)."""
    K = 3
    Xs = _su_elements(N, K, seed=13)
    out = head_adjoint(Xs, _onehot_coeff(K, 2, (a, b, c)))
    expected = torch.zeros(K, N, N, dtype=C128)
    expected[a] = _anti_ref(Xs[b], Xs[c])
    assert torch.allclose(out, expected, atol=1e-13, rtol=0), (out - expected).abs().max()
    if N == 2:
        assert out.abs().max() < 1e-13
    else:
        assert out.abs().max() > 1e-2, "the d-type covariant must not vanish for SU(3)"


@pytest.mark.parametrize("N", [2, 3])
def test_head_adjoint_dense_coefficients(N):
    """Linear combination with dense random coefficients, against explicit loops over all indices."""
    K = 3
    Xs = _su_elements(N, K, seed=14)
    c1 = _randn(K, K, seed=15)
    c2 = _randn(K, K, K, seed=16)
    c3 = _randn(K, K, K, seed=17)
    out = head_adjoint(Xs, lambda I2: (c1, c2, c3))
    expected = torch.zeros(K, N, N, dtype=C128)
    for a in range(K):
        for b in range(K):
            expected[a] = expected[a] + c1[a, b] * Xs[b]
            for c in range(K):
                expected[a] = expected[a] + c2[a, b, c] * (Xs[b] @ Xs[c] - Xs[c] @ Xs[b])
                expected[a] = expected[a] + c3[a, b, c] * _anti_ref(Xs[b], Xs[c])
    assert torch.allclose(out, expected, atol=TOL, rtol=0), (out - expected).abs().max()
    # the result lies in su(N)
    assert torch.allclose(out, _proj_su_ref(out), atol=TOL, rtol=0)


# ---------------------------------------------------------------------------------------------
# 3. head_loops
# ---------------------------------------------------------------------------------------------
def _loops_W(seed: int) -> Tensor:
    return _randn(2, 4, 3, 3, seed=seed, dtype=C128)


def test_head_loops_constant_complex_coefficients():
    W = _loops_W(20)
    c1, c2 = 0.7 - 0.4j, -0.3 + 1.1j
    out = head_loops(W, lambda t: (c1 * torch.ones_like(t), c2 * torch.ones_like(t)))
    assert out.shape == W.shape and out.dtype == C128
    expected = _proj_su_ref(c1 * W + c2 * (W @ W))
    assert torch.allclose(out, expected, atol=TOL, rtol=0), (out - expected).abs().max()
    # both terms matter: dropping either one changes the answer by O(1)
    assert (expected - _proj_su_ref(c1 * W)).abs().max() > 0.1
    assert (expected - _proj_su_ref(c2 * (W @ W))).abs().max() > 0.1


def test_head_loops_coefficient_depends_on_trace():
    W = _loops_W(21)
    seen = []

    def coeff(t):
        seen.append(t)
        return 0.5 * t + 0.2j * t.conj(), 0.1 * t * t - 0.3 + 0.25j
    out = head_loops(W, coeff)
    t = torch.einsum("...ii->...", W)
    assert seen[0].shape == W.shape[:-2] and torch.allclose(seen[0], t, atol=1e-14, rtol=0)
    c1, c2 = 0.5 * t + 0.2j * t.conj(), 0.1 * t * t - 0.3 + 0.25j
    expected = _proj_su_ref(c1[..., None, None] * W + c2[..., None, None] * (W @ W))
    assert torch.allclose(out, expected, atol=TOL, rtol=0), (out - expected).abs().max()


def test_head_loops_output_in_su3():
    W = _loops_W(22)
    out = head_loops(W, lambda t: (1.0 + 0.5j + 0 * t, 0.3 - 0.2j + 0 * t))
    assert torch.einsum("...ii->...", out).abs().max() < TOL
    assert (out + out.conj().transpose(-1, -2)).abs().max() < TOL


# ---------------------------------------------------------------------------------------------
# 4. head_fundamental
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [2, 3])
def test_head_fundamental_value(N):
    """omega_a = proj_su(sum_bc C[a,b,c] v_b v_c^H) with an arbitrary fixed complex C."""
    K, M = 3, 5
    V = _randn(M, N, seed=30, dtype=C128)
    Cc = _randn(K, M, M, seed=31, dtype=C128)
    seen = []

    def coeff(G):
        seen.append(G)
        return Cc
    out = head_fundamental(V, coeff)
    assert out.shape == (K, N, N) and out.dtype == C128
    G_ref = torch.stack([torch.stack([(V[b] * V[c].conj()).sum() for c in range(M)]) for b in range(M)])
    assert torch.allclose(seen[0], G_ref, atol=1e-14, rtol=0)
    expected = torch.zeros(K, N, N, dtype=C128)
    for a in range(K):
        S = torch.zeros(N, N, dtype=C128)
        for b in range(M):
            for c in range(M):
                S = S + Cc[a, b, c] * torch.outer(V[b], V[c].conj())
        expected[a] = _proj_su_ref(S)
    # sums of 25 O(1) products: a few ulp
    assert torch.allclose(out, expected, atol=1e-13, rtol=0), (out - expected).abs().max()
    assert expected.abs().max() > 0.1


def test_head_fundamental_one_hot_is_projected_outer_product():
    N, K, M = 3, 2, 4
    V = _randn(M, N, seed=32, dtype=C128)
    Cc = torch.zeros(K, M, M, dtype=C128)
    Cc[1, 2, 0] = 1.0
    out = head_fundamental(V, lambda G: Cc)
    assert out[0].abs().max() == 0
    assert torch.allclose(out[1], _proj_su_ref(torch.outer(V[2], V[0].conj())), atol=1e-14, rtol=0)
