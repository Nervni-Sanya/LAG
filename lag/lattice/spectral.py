"""Spectral shortcut for single-loop kernels (MATH_SPEC_LATTICE.md section 5, Proposition 2).

For omega(W) = proj_su(c1(t) W + c2(t) W^2), t = tr W, the map W -> exp(omega(W)) W keeps the eigenvectors of W and
moves its eigen-angles theta_j to theta'_j = theta_j + a_j - mean(a), a_j = Im(c1 z_j + c2 z_j^2), z_j = exp(i theta_j).
With respect to Haar measure

    log|det J| = log|det d theta'/d theta| + sum_{j<k} log( sin^2((theta'_j - theta'_k)/2) / sin^2((theta_j - theta_k)/2) ).

On the lattice this is log|det J_l| of an active link whose connection reads exactly one loop W = U_l S through l
(S frozen); coefficients may also read loops that contain no active link, but never another loop through l (the
negative control of section 9 is wrong by 0.18 to 0.38). Gradients at degenerate spectra are not treated.

`logdet_spectral` is moved verbatim from the reference laglattice.py (type hints and docstring added).
"""
from __future__ import annotations

import torch
from torch import Tensor

from lag.lattice.field import CoeffFn

__all__ = ["logdet_spectral"]


def logdet_spectral(W: Tensor, coeff: CoeffFn) -> Tensor:
    """log|det J| of W -> exp(omega(W)) W for a single unitary loop W (..., N, N), omega = head(W[None], coeff),
    from eigen-angles only: log|det d theta'/d theta| + sum_{j<k} log sin^2(dtheta'_jk/2)/sin^2(dtheta_jk/2).

    coeff receives the trace as a (..., 1) complex tensor and returns (c1, c2), each (..., 1). The N x N matrix
    d theta'/d theta comes from N reverse passes (all N angles treated as independent). Returns (...).
    """
    N = W.shape[-1]
    th = torch.angle(torch.linalg.eigvals(W)).detach().requires_grad_(True)       # (..., N)
    z = torch.exp(1j * th)
    c1, c2 = coeff(z.sum(-1, keepdim=True))
    a = (c1 * z + c2 * z * z).imag
    thp = th + a - a.mean(-1, keepdim=True)
    Jt = torch.stack([torch.autograd.grad(thp[..., k].sum(), th, create_graph=True)[0] for k in range(N)], -2)
    out = torch.linalg.slogdet(Jt)[1]
    for j in range(N):
        for k in range(j + 1, N):
            out = out + torch.log(torch.sin((thp[..., j] - thp[..., k]) / 2) ** 2 / torch.sin((th[..., j] - th[..., k]) / 2) ** 2)
    return out
