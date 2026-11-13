"""Left-trivialised dexp and closed-form exponentials (MATH_SPEC.md section 2).

dexpL_w = (1 - exp(-ad_w))/ad_w :  d exp(hat w) = exp(hat w) hat(dexpL_w dw)   (left-trivialised)
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from lag.algebra import C, T2, so3_hat, su_hat

__all__ = [
    "dexpL_so3", "dexpL_se3", "dexpL_generic",
    "so3_exp", "su2_exp", "se3_exp", "su3_exp", "exp_vjp_normal",
]

# ------------------------------------------------------------------ dexp
def _coeffs(th2: Tensor, K: int = 12, cut: float = 0.25) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """a=(1-cos)/th^2, b=(th-sin)/th^3, alpha=a'/th, beta=b'/th ; power series below the cut.

    th2: (...) squared angle -> four tensors of shape (...). MATH_SPEC section 2.
    """
    small = th2 < cut
    t2 = torch.where(small, torch.ones_like(th2), th2)
    th = t2.sqrt()
    s, c = torch.sin(th), torch.cos(th)
    x = torch.where(small, th2, torch.zeros_like(th2))
    sa = sb = sal = sbe = torch.zeros_like(th2)
    for k in range(K):
        sgn = (-1) ** k
        sa = sa + sgn * x ** k / float(math.factorial(2 * k + 2))
        sb = sb + sgn * x ** k / float(math.factorial(2 * k + 3))
        if k >= 1:
            sal = sal + sgn * 2 * k * x ** (k - 1) / float(math.factorial(2 * k + 2))
            sbe = sbe + sgn * 2 * k * x ** (k - 1) / float(math.factorial(2 * k + 3))
    a = torch.where(small, sa, (1 - c) / t2)
    b = torch.where(small, sb, (th - s) / (t2 * th))
    al = torch.where(small, sal, (th * s - 2 * (1 - c)) / t2 ** 2)
    be = torch.where(small, sbe, (1 - c) / t2 ** 2 - 3 * (th - s) / (t2 ** 2 * th))
    return a, b, al, be
def dexpL_so3(w: Tensor) -> Tensor:
    """Also valid for su(2) in the basis T_a = -(i/2) sigma_a (ad_w = [w]_x).

    w: (..., 3) -> (..., 3, 3). MATH_SPEC section 2.
    """
    Phi = so3_hat(w)
    a, b, _, _ = _coeffs((w * w).sum(-1))
    I = torch.eye(3, dtype=w.dtype)
    return I - a[..., None, None] * Phi + b[..., None, None] * Phi @ Phi
def dexpL_se3(xi: Tensor) -> Tensor:
    """Left-trivialised dexp of se(3). xi: (..., 6) -> (..., 6, 6). MATH_SPEC section 2."""
    phi, rho = xi[..., :3], xi[..., 3:]
    Phi, P = so3_hat(phi), so3_hat(rho)
    a, b, al, be = (t[..., None, None] for t in _coeffs((phi * phi).sum(-1)))
    J = torch.eye(3, dtype=xi.dtype) - a * Phi + b * Phi @ Phi
    Q = -a * P + b * (Phi @ P + P @ Phi) + (phi * rho).sum(-1)[..., None, None] * (-al * Phi + be * Phi @ Phi)
    Z = torch.zeros_like(J)
    return torch.cat([torch.cat([J, Z], -1), torch.cat([Q, J], -1)], -2)
def dexpL_generic(ad: Tensor) -> Tensor:
    """phi_1(-ad) from one matrix exponential of twice the size; any Lie algebra.

    ad: (..., d, d) -> (..., d, d). MATH_SPEC section 2.
    """
    d = ad.shape[-1]
    I = torch.eye(d, dtype=ad.dtype).expand_as(ad)
    Z = torch.zeros_like(ad)
    M = torch.cat([torch.cat([-ad, I], -1), torch.cat([Z, Z], -1)], -2)
    return torch.linalg.matrix_exp(M)[..., :d, d:]

# ------------------------------------------------------------------ closed-form exponentials
def so3_exp(w: Tensor) -> Tensor:
    """Rodrigues exponential of so(3). w: (..., 3) -> (..., 3, 3). MATH_SPEC section 2."""
    Phi = so3_hat(w)
    th2 = (w * w).sum(-1)
    a, b, _, _ = _coeffs(th2)
    sinc = 1 - th2 * b                                             # sin(th)/th
    return torch.eye(3, dtype=w.dtype) + sinc[..., None, None] * Phi + a[..., None, None] * Phi @ Phi
def su2_exp(w: Tensor) -> Tensor:
    """Closed-form exponential of su(2) in the basis T2. w: (..., 3) -> (..., 2, 2) complex128. MATH_SPEC section 2."""
    th2 = (w * w).sum(-1)
    a, b, _, _ = _coeffs(th2 / 4)                                  # half angle
    cos_h = 1 - (th2 / 4) * a
    sinc_h = 1 - (th2 / 4) * b                                     # cos(th/2), sin(th/2)/(th/2)
    return cos_h[..., None, None] * torch.eye(2, dtype=C) + sinc_h[..., None, None] * su_hat(w, T2)
def se3_exp(xi: Tensor) -> Tensor:
    """Closed-form exponential of se(3). xi: (..., 6) -> (..., 4, 4). MATH_SPEC section 2."""
    phi, rho = xi[..., :3], xi[..., 3:]
    Phi = so3_hat(phi)
    a, b, _, _ = _coeffs((phi * phi).sum(-1))
    V = torch.eye(3, dtype=xi.dtype) + a[..., None, None] * Phi + b[..., None, None] * Phi @ Phi
    top = torch.cat([so3_exp(phi), (V @ rho[..., None])], -1)
    bot = torch.zeros_like(top[..., :1, :])
    bot[..., 0, 3] = 1
    return torch.cat([top, bot], -2)


class _ArccosClamped(torch.autograd.Function):
    """arccos(clamp(x, -1, 1)) with a finite first derivative at |x| >= 1 (helper of su3_exp).

    First derivative: arccos'(x) for |x| < 1 and 0 for |x| >= 1; the reference (torch.arccos of torch.clamp)
    gives 0 outside [-1, 1] and NaN at exactly +-1. Second derivative: that of arccos for |x| < 1 and NaN for
    |x| >= 1, as in the reference (unsupported there).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.arccos(torch.clamp(x, -1.0, 1.0))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, g: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return g * _DArccosClamped.apply(x)


class _DArccosClamped(torch.autograd.Function):
    """Derivative of _ArccosClamped: -1/sqrt(1 - x^2) for |x| < 1, 0 for |x| >= 1; its derivative is NaN there."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        edge = x.abs() >= 1.0
        safe = torch.where(edge, torch.zeros_like(x), x)
        return torch.where(edge, torch.zeros_like(x), -((-safe * safe + 1).rsqrt()))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, g: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        edge = x.abs() >= 1.0
        safe = torch.where(edge, torch.zeros_like(x), x)
        return g * torch.where(edge, torch.full_like(x, float("nan")), -safe * (-safe * safe + 1) ** -1.5)


_arccos_clamped = _ArccosClamped.apply


def su3_exp(X: Tensor, taylor_cut: float = 1e-12, K: int = 4) -> Tensor:
    """exp of X in su(3) by Cayley-Hamilton in Newton form; Taylor series for small norm.

    X: (..., 3, 3) complex128 anti-Hermitian -> (..., 3, 3). MATH_SPEC section 2.4.

    Gradient guard (the only deviation from the reference lagmath.py): arccos(clamp(c0 / (2 s^3), -1, 1))
    is evaluated by _arccos_clamped. Forward values, and all derivatives where |c0 / (2 s^3)| < 1, are the
    reference's. Where |c0 / (2 s^3)| >= 1 (exactly degenerate eigenvalues after rounding) the first
    derivative of the phi path is 0, its exact limit, so FIRST derivatives are exact and finite; the
    reference gives NaN at exactly +-1 (arccos'(+-1) = inf times a zero upstream gradient). Second
    derivatives there are NaN, as in the reference: Hessians at degenerate X are not supported (use
    torch.linalg.matrix_exp for them).
    """
    I = torch.eye(3, dtype=C)
    Q = -1j * X
    Q2 = Q @ Q
    c1 = Q2.diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
    small = c1 < taylor_cut
    Qs = torch.where(small[..., None, None], torch.diag(torch.tensor([1.0, 0.0, -1.0], dtype=C)).expand_as(Q), Q)
    Q2s = Qs @ Qs
    c1s = Q2s.diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
    c0s = (Q2s @ Qs).diagonal(dim1=-2, dim2=-1).sum(-1).real / 3
    s = torch.sqrt(c1s / 3)
    ph = _arccos_clamped(c0s / (2 * s ** 3)) / 3                   # arccos(clamp(., -1, 1)) / 3; see _ArccosClamped
    lam = [2 * s * torch.cos(ph - 2 * math.pi * k / 3) for k in range(3)]       # lam0 >= lam1 >= lam2, lam0-lam2 >= 3s
    dd = lambda p, q: 1j * torch.exp(0.5j * (p + q)) * torch.sinc((q - p) / (2 * math.pi))   # f[p,q], f = exp(i.)
    h0 = torch.exp(1j * lam[0])
    h1 = dd(lam[0], lam[1])
    h2 = (dd(lam[1], lam[2]) - h1) / (lam[2] - lam[0])
    A0 = Qs - lam[0][..., None, None] * I
    A1 = Qs - lam[1][..., None, None] * I
    closed = h0[..., None, None] * I + h1[..., None, None] * A0 + h2[..., None, None] * A0 @ A1
    Xs = torch.where(small[..., None, None], X, torch.zeros_like(X))
    tay = I.expand_as(X).clone()
    term = I.expand_as(X)
    for k in range(1, K + 1):
        term = term @ Xs / k
        tay = tay + term
    return torch.where(small[..., None, None], tay, closed)
def exp_vjp_normal(X: Tensor, Ebar: Tensor) -> Tensor:
    """PyTorch-convention gradient of E = exp(X) for anti-Hermitian X (Daleckii-Krein); finite at degeneracy.

    X, Ebar: (..., N, N) complex -> (..., N, N). MATH_SPEC section 2.
    """
    lam, V = torch.linalg.eigh(-1j * X)
    li, lj = lam[..., :, None], lam[..., None, :]
    Gam = torch.exp(0.5j * (li + lj)) * torch.sinc((li - lj) / (2 * math.pi))
    return V @ ((V.mH @ Ebar @ V) * Gam.conj()) @ V.mH
