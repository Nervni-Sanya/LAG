"""Lie-algebra bases and hat/vee maps (MATH_SPEC.md section 1).

Conventions
  so(3): hat(w) = [w]_x, [hat a, hat b] = hat(a x b), ad_w = [w]_x
  su(2): T_a = -(i/2) sigma_a ; su(3): T_a = -(i/2) lambda_a ; tr(T_a T_b) = -delta_ab/2 ; vee(X)_a = -2 Re tr(T_a X)
  se(3): xi = (phi, rho), hat = [[phi_x, rho],[0,0]], ad_xi = [[Phi,0],[P,Phi]]
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "C", "T2", "T3",
    "so3_hat", "so3_vee", "su_hat", "su_vee", "proj_su",
    "se3_hat", "se3_vee_first_order", "se3_ad",
]

C = torch.complex128

# ------------------------------------------------------------------ bases
def so3_hat(w: Tensor) -> Tensor:
    """Hat map so(3): [w]_x. w: (..., 3) -> (..., 3, 3). MATH_SPEC section 1."""
    O = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([O, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], O, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], O], -1)], -2)
def so3_vee(X: Tensor) -> Tensor:
    """Inverse of so3_hat on the antisymmetric part of X. X: (..., 3, 3) -> (..., 3). MATH_SPEC section 1."""
    return torch.stack([X[..., 2, 1] - X[..., 1, 2], X[..., 0, 2] - X[..., 2, 0], X[..., 1, 0] - X[..., 0, 1]], -1) / 2
def _pauli() -> Tensor:
    """Pauli matrices sigma_a. Returns (3, 2, 2) complex128."""
    return torch.tensor([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=C)
def _gellmann() -> Tensor:
    """Gell-Mann matrices lambda_a. Returns (8, 3, 3) complex128."""
    l = torch.zeros(8, 3, 3, dtype=C)
    l[0, 0, 1] = l[0, 1, 0] = 1
    l[1, 0, 1] = -1j
    l[1, 1, 0] = 1j
    l[2, 0, 0] = 1
    l[2, 1, 1] = -1
    l[3, 0, 2] = l[3, 2, 0] = 1
    l[4, 0, 2] = -1j
    l[4, 2, 0] = 1j
    l[5, 1, 2] = l[5, 2, 1] = 1
    l[6, 1, 2] = -1j
    l[6, 2, 1] = 1j
    l[7] = torch.diag(torch.tensor([1, 1, -2], dtype=C)) / math.sqrt(3)
    return l
T2 = -0.5j * _pauli()
T3 = -0.5j * _gellmann()
def su_hat(w: Tensor, T: Tensor) -> Tensor:
    """Hat map su(N) in basis T. w: (..., d), T: (d, N, N) -> (..., N, N) complex128. MATH_SPEC section 1."""
    return torch.einsum("...a,aij->...ij", w.to(C), T)
def su_vee(X: Tensor, T: Tensor) -> Tensor:
    """Vee map su(N) in basis T. X: (..., N, N), T: (d, N, N) -> (..., d) real. MATH_SPEC section 1."""
    return -2 * torch.einsum("aij,...ji->...a", T, X).real
def proj_su(X: Tensor) -> Tensor:
    """Traceless anti-Hermitian part: the orthogonal projection gl(N,C) -> su(N).

    X: (..., N, N) -> (..., N, N). MATH_SPEC section 1.
    """
    A = (X - X.mH) / 2
    N = X.shape[-1]
    return A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).expand(*A.shape[:-1])) / N
def se3_hat(xi: Tensor) -> Tensor:
    """Hat map se(3). xi: (..., 6) = (phi, rho) -> (..., 4, 4). MATH_SPEC section 1."""
    top = torch.cat([so3_hat(xi[..., :3]), xi[..., 3:, None]], -1)
    return torch.cat([top, torch.zeros_like(top[..., :1, :])], -2)
def se3_vee_first_order(M: Tensor) -> Tensor:               # coordinates of M - 1 to first order
    """Coordinates of M - 1 to first order. M: (..., 4, 4) -> (..., 6). MATH_SPEC section 1."""
    return torch.cat([so3_vee(M[..., :3, :3]), M[..., :3, 3]], -1)
def se3_ad(xi: Tensor) -> Tensor:
    """Adjoint representation ad_xi of se(3). xi: (..., 6) -> (..., 6, 6). MATH_SPEC section 1."""
    Phi, P = so3_hat(xi[..., :3]), so3_hat(xi[..., 3:])
    Z = torch.zeros_like(Phi)
    return torch.cat([torch.cat([Phi, Z], -1), torch.cat([P, Phi], -1)], -2)
