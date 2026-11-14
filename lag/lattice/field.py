"""The link field of a lattice gauge theory (MATH_SPEC_LATTICE.md section 1).

Field layout: U has shape (B, D, L, ..., L, N, N): batch, direction mu, the D lattice axes (periodic), colour.
U[b, mu, x] is the link U_mu(x) from site x to x + mu^. Gauge transformation:
U_mu(x) -> g(x) U_mu(x) g(x + mu^)^dagger. Reference measure: the product of Haar measures over the links.
float64 / complex128 throughout; the gauge group is SU(2) or SU(3) with the bases of MATH_SPEC.md section 1.

Moved verbatim from the reference laglattice.py (only type hints and docstrings added): GaugeGroup, haar, shift,
staples, head, gauge_transform, plaquettes, wilson_action. `head` lives here rather than in lag.lattice.layers
because the mask validator (lag.lattice.masks) needs it too.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lag.algebra import C, T2, T3, proj_su, su_hat, su_vee
from lag.exp import dexpL_generic, dexpL_so3, su2_exp, su3_exp

__all__ = [
    "C", "CoeffFn", "GaugeGroup",
    "haar", "shift", "staples", "head", "gauge_transform", "plaquettes", "wilson_action",
]

CoeffFn = Callable[[Tensor], tuple[Tensor, Tensor]]
"""Coefficient function of the loop head: complex traces t (..., n_loops) -> complex (c1, c2), each (..., n_loops)."""


class GaugeGroup:
    """SU(2) or SU(3) with the bases of MATH_SPEC section 1.

    N = 2 uses T2 = -(i/2) sigma_a, the closed-form su2_exp and dexpL_so3 (ad_w = [w]_x in this basis); any other N
    uses T3 = -(i/2) lambda_a, su3_exp and the generic block-exponential dexpL_generic, so only N = 2 and N = 3 are
    meaningful. Attributes: N, d = N^2 - 1, the basis T (d, N, N) and the structure constants f (d, d, d) with
    [T_a, T_b] = f_abc T_c.
    """

    def __init__(self, N: int) -> None:
        self.N, self.d, self.T = N, N * N - 1, (T2 if N == 2 else T3)
        comm = torch.einsum("aij,bjk->abik", self.T, self.T)
        self.f = su_vee(comm - comm.transpose(0, 1), self.T)             # [T_a, T_b] = f_abc T_c

    def hat(self, w: Tensor) -> Tensor:
        """Coordinates (..., d) -> su(N) matrices (..., N, N) complex128."""
        return su_hat(w, self.T)

    def vee(self, X: Tensor) -> Tensor:
        """su(N) matrices (..., N, N) -> coordinates (..., d) real."""
        return su_vee(X, self.T)

    def exp(self, w: Tensor) -> Tensor:
        """exp(hat w) from coordinates (..., d): closed form for SU(2), su3_exp for SU(3). Returns (..., N, N)."""
        return su2_exp(w) if self.N == 2 else su3_exp(self.hat(w))

    def dexpL(self, w: Tensor) -> Tensor:
        """Left-trivialised dexp_w = (1 - exp(-ad_w)) / ad_w in coordinates: (..., d) -> (..., d, d)."""
        if self.N == 2:
            return dexpL_so3(w)
        return dexpL_generic(torch.einsum("...a,abc->...cb", w, self.f))  # ad_w[c, b] = w_a f_abc


def haar(n: int, N: int) -> Tensor:
    """n Haar-distributed SU(N) matrices (QR of a complex Gaussian, phase fix, cube/square root of det).

    Draws from the global torch random stream. Returns (n, N, N) complex128.
    """
    Q, R = torch.linalg.qr(torch.randn(n, N, N, dtype=C))
    dg = R.diagonal(dim1=-2, dim2=-1)
    Q = Q * (dg / dg.abs()).unsqueeze(-2)
    return Q * torch.exp(-1j * torch.angle(torch.linalg.det(Q)) / N)[:, None, None]


def shift(A: Tensor, nu: int, k: int = 1) -> Tensor:
    """A(x) -> A(x + k nu) for A of shape (B, L, ..., L, N, N) (periodic)."""
    return torch.roll(A, -k, dims=1 + nu)


def staples(U: Tensor, mu: int) -> Tensor:
    """Staples of every link (x, mu): (B, L^D..., 2(D-1), N, N); U_mu(x) @ staple is a closed loop based at x.
    A staple of (x, mu) contains the other mu-links (x +- nu, mu) and nu-links, never (x, mu) itself.

    For each nu != mu in increasing order: the upper staple U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger, then the
    lower staple U_nu(x+mu-nu)^dagger U_mu(x-nu)^dagger U_nu(x-nu) (MATH_SPEC_LATTICE section 3.1).
    """
    D = U.shape[1]
    Um = U[:, mu]
    out = []
    for nu in range(D):
        if nu == mu:
            continue
        Un = U[:, nu]
        out.append(shift(Un, mu) @ shift(Um, nu).mH @ Un.mH)
        out.append(shift(shift(Un, mu), nu, -1).mH @ shift(Um, nu, -1).mH @ shift(Un, nu, -1))
    return torch.stack(out, -3)


def head(W: Tensor, coeff: CoeffFn) -> Tensor:
    """Gauge-covariant connection from loops W (..., n_loops, N, N) based at the link's start site:
    omega = proj_su( sum_s c1_s W_s + c2_s W_s^2 ), (c1, c2) = coeff(tr W) complex, each (..., n_loops).

    The loop head of MATH_SPEC.md section 4.3 summed over the loops of one link. Returns (..., N, N) in su(N).
    """
    c1, c2 = coeff(W.diagonal(dim1=-2, dim2=-1).sum(-1))
    return proj_su((c1[..., None, None] * W + c2[..., None, None] * W @ W).sum(-3))


def gauge_transform(U: Tensor, g: Tensor) -> Tensor:
    """U_mu(x) -> g(x) U_mu(x) g(x + mu)^dagger ; g: (B, L^D..., N, N)."""
    return torch.stack([g @ U[:, mu] @ shift(g, mu).mH for mu in range(U.shape[1])], 1)


def plaquettes(U: Tensor) -> Tensor:
    """All plaquettes P_{mu<nu}(x): (B, D(D-1)/2, L^D..., N, N).

    P_mu,nu(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger, planes ordered (0,1), (0,2), ..., (D-2,D-1).
    """
    D = U.shape[1]
    return torch.stack([U[:, mu] @ shift(U[:, nu], mu) @ shift(U[:, mu], nu).mH @ U[:, nu].mH
                        for mu in range(D) for nu in range(mu + 1, D)], 1)


def wilson_action(U: Tensor, beta: float) -> Tensor:
    """S = -(beta/N) sum_plaquettes Re tr P ; returns (B,)."""
    P = plaquettes(U)
    N = U.shape[-1]
    return -(beta / N) * P.diagonal(dim1=-2, dim2=-1).sum(-1).real.flatten(1).sum(-1)
