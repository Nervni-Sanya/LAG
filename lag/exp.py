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


def _su3_closed(Q: Tensor) -> Tensor:
    """exp(iQ) for Hermitian traceless Q (..., 3, 3) with a well-separated spectrum; MATH_SPEC section 2.4.

    The Cayley-Hamilton closed form in Newton form, without the clamp and the Taylor branch of the reference:
    su3_exp only calls it where 1 - |c0 / (2 s^3)| > deg_tol and c1 >= taylor_cut.
    """
    I = torch.eye(3, dtype=C)
    Q2 = Q @ Q
    c1 = Q2.diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
    c0 = (Q2 @ Q).diagonal(dim1=-2, dim2=-1).sum(-1).real / 3
    s = torch.sqrt(c1 / 3)
    ph = torch.arccos(c0 / (2 * s ** 3)) / 3
    lam = [2 * s * torch.cos(ph - 2 * math.pi * k / 3) for k in range(3)]       # lam0 >= lam1 >= lam2, lam0-lam2 >= 3s
    dd = lambda p, q: 1j * torch.exp(0.5j * (p + q)) * torch.sinc((q - p) / (2 * math.pi))   # f[p,q], f = exp(i.)
    h0 = torch.exp(1j * lam[0])
    h1 = dd(lam[0], lam[1])
    h2 = (dd(lam[1], lam[2]) - h1) / (lam[2] - lam[0])
    A0 = Q - lam[0][..., None, None] * I
    A1 = Q - lam[1][..., None, None] * I
    return h0[..., None, None] * I + h1[..., None, None] * A0 + h2[..., None, None] * A0 @ A1


def su3_exp(X: Tensor, deg_tol: float = 1e-6, taylor_cut: float = 1e-12) -> Tensor:
    """exp of X in su(3): closed form on generic elements, torch.linalg.matrix_exp on the rare others.

    X: (..., 3, 3) complex128 anti-Hermitian -> (..., 3, 3). MATH_SPEC section 2.4. Finite derivatives of
    every order.

    Deviation from the reference lagmath.py (the reviewed "fallback patch", applied with the user's approval):
    elements with c1 < taylor_cut (near zero) or 1 - |c0 / (2 s^3)| <= deg_tol (relative eigenvalue gap below
    about 1.6e-3 for deg_tol = 1e-6) go to torch.linalg.matrix_exp, which is smooth to every order; the routing
    mask is computed without gradient. Generic elements use the closed form with the reference's arithmetic.
    The reference's Taylor branch and clamp are no longer used, so first and second derivatives are finite
    and accurate at exactly degenerate X (e.g. any multiple of T_8) and at X = 0.
    Caveat: the routing is data-dependent control flow, so torch.func.vmap cannot wrap this function and
    torch.compile traces it with a graph break.
    """
    shape = X.shape
    Xf = X.reshape(-1, 3, 3)
    with torch.no_grad():
        Q = -1j * Xf
        Q2 = Q @ Q
        c1 = Q2.diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
        c0 = (Q2 @ Q).diagonal(dim1=-2, dim2=-1).sum(-1).real / 3
        ok = c1 >= taylor_cut
        x = c0 / (2 * (torch.where(ok, c1, torch.ones_like(c1)) / 3) ** 1.5)
        generic = ok & ((1 - x.abs()) > deg_tol)
    if bool(generic.all()):
        return _su3_closed(-1j * Xf).reshape(shape)
    out = torch.empty_like(Xf)
    if bool(generic.any()):
        out[generic] = _su3_closed(-1j * Xf[generic])
    out[~generic] = torch.linalg.matrix_exp(Xf[~generic])
    return out.reshape(shape)


def exp_vjp_normal(X: Tensor, Ebar: Tensor) -> Tensor:
    """PyTorch-convention gradient of E = exp(X) for anti-Hermitian X (Daleckii-Krein); finite at degeneracy.

    X, Ebar: (..., N, N) complex -> (..., N, N). MATH_SPEC section 2.
    """
    lam, V = torch.linalg.eigh(-1j * X)
    li, lj = lam[..., :, None], lam[..., None, :]
    Gam = torch.exp(0.5j * (li + lj)) * torch.sinc((li - lj) / (2 * math.pi))
    return V @ ((V.mH @ Ebar @ V) * Gam.conj()) @ V.mH
