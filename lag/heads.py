"""Equivariant connection heads (MATH_SPEC.md section 4): SU(N) heads and the generic wedge connection."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lag.algebra import C, proj_su

if TYPE_CHECKING:
    from lag.groups.base import MatrixLieGroup

__all__ = [
    "eps_dual_su2", "cross_dual_su3",
    "head_fundamental", "head_adjoint", "head_loops",
    "wedge_connection",
]

# ------------------------------------------------------------------ equivariant connection heads for SU(N)
def eps_dual_su2(z: Tensor) -> Tensor:
    """z~ = eps conj(z): transforms like z under SU(2).

    z: (..., 2) complex -> (..., 2). MATH_SPEC section 4.
    """
    return torch.stack([z[..., 1].conj(), -z[..., 0].conj()], -1)
def cross_dual_su3(x: Tensor, y: Tensor) -> Tensor:
    """conj(x cross y): transforms like a fundamental vector under SU(3).

    x, y: (..., 3) complex -> (..., 3). MATH_SPEC section 4.
    """
    return torch.linalg.cross(x, y).conj()
def head_fundamental(V: Tensor, coeff: Callable[[Tensor], Tensor]) -> Tensor:
    """V: (M, N) covariant vectors (inputs and their duals). coeff(G) -> (K, M, M) complex, G = V V^H invariant.
    Returns (K, N, N) elements of su(N):  omega_a = proj_su( sum_bc C[a,b,c] v_b v_c^H ).

    MATH_SPEC section 4.
    """
    G = V @ V.mH
    return proj_su(torch.einsum("abc,bi,cj->aij", coeff(G), V, V.conj()))
def head_adjoint(Xs: Tensor, coeff: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]) -> Tensor:
    """Xs: (K, N, N) in su(N). Covariants: X_a, [X_a,X_b], i({X_a,X_b} - trace part) (the last is zero for SU(2)).

    coeff(I2) with I2: (K, K) -> (c1, c2, c3); returns (K, N, N). MATH_SPEC section 4.
    """
    I2 = torch.einsum("aij,bji->ab", Xs, Xs).real
    comm = torch.einsum("aij,bjk->abik", Xs, Xs)
    comm = comm - comm.transpose(0, 1)
    anti = torch.einsum("aij,bjk->abik", Xs, Xs)
    anti = proj_su(1j * (anti + anti.transpose(0, 1)))
    c1, c2, c3 = (c.to(C) for c in coeff(I2))
    return torch.einsum("ab,bij->aij", c1, Xs) + torch.einsum("abc,bcij->aij", c2, comm) + torch.einsum("abc,bcij->aij", c3, anti)
def head_loops(W: Tensor, coeff: Callable[[Tensor], tuple[Tensor, Tensor]]) -> Tensor:
    """W: (..., N, N) untraced closed loops based at the link's start site (W -> g W g^H).
    omega = proj_su( c1(t) W + c2(t) W^2 ), t = tr W (complex invariant); c1, c2 complex.

    coeff(t) with t: (...) -> (c1, c2); returns (..., N, N). MATH_SPEC section 4.
    """
    t = W.diagonal(dim1=-2, dim2=-1).sum(-1)
    c1, c2 = coeff(t)
    return proj_su(c1[..., None, None] * W + c2[..., None, None] * W @ W)
def wedge_connection(group: "MatrixLieGroup", coeff: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Generic equivariant connection for any G in O(N): w_a = sum_b coeff(Gram)_ab * proj_g(z_a ^ z_b).

    Generic extension from lag_proto.py, not part of MATH_SPEC.md. Equivariant only for real groups whose
    generators are skew-symmetric (G inside O(N)); there the Gram matrix Z Z^T is invariant and proj_g
    commutes with the action. coeff maps the (K, K) Gram matrix to (K, K) coefficients; the returned
    field maps real Z: (K, N) -> (K, d) coordinates.
    """
    def field(Z: Tensor) -> Tensor:                             # Z: (K, N)
        c = coeff(Z @ Z.T)
        W = Z.unsqueeze(1).unsqueeze(-1) * Z.unsqueeze(0).unsqueeze(-2)      # z_a z_b^T
        return torch.einsum("ab,abc->ac", c, group.vee(W - W.transpose(-1, -2)))
    return field
