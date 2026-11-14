"""Masked lattice sub-layers, their total log-determinant and their inverse (MATH_SPEC_LATTICE.md sections 2 and 6).

A sub-layer updates the active links of a mask, U_l -> exp(omega_l) U_l, and leaves every other link unchanged. When
each omega_l depends only on U_l and on frozen links (Proposition 1), the Jacobian of the whole lattice map is block
triangular and log|det J| = sum over active links of log|det(1 + dexpL_{v_l} Dv_l)|, v_l = vee(omega_l). All the Dv_l
come from d reverse passes at any volume (the batched-Dv corollary).

`masked_layer` is moved verbatim from the reference laglattice.py (type hints and docstring added).
`invert_masked_layer` is the fixed-point iteration of section 6, moved from `sec_inverse` of the reference
verify_lattice.py into a function with the same arithmetic.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from lag.lattice.field import C, CoeffFn, GaugeGroup, head, staples
from lag.lattice.masks import mask

__all__ = ["masked_layer", "invert_masked_layer"]


def masked_layer(U: Tensor, mu: int, p: int, coeff: CoeffFn, G: GaugeGroup, logdet: bool = True,
                 active: Tensor | None = None,
                 create_graph: bool = True) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
    """One sub-layer: U_l -> exp(omega_l) U_l on the active links, all other links unchanged.
    Returns U' and, if logdet, (log|det J| per configuration, min sign of the per-link determinants, max ||D omega||_2).
    `active` overrides the mask (used only for negative controls).

    U: (B, D, L, ..., L, N, N) on a hypercubic lattice; the active links are mask(L, D, mu, p) unless `active` (a
    boolean (L,) * D tensor over the sites of the mu-links) is given. omega_l = head(U_l @ staples of l, coeff).
    With logdet=True the return value is (U', logdet (B,), sign (B,), max ||Dv||_2 (B,)): Dv from d reverse passes
    of the xi-perturbation (I + hat xi) U_l (first order in xi suffices), then slogdet(1 + dexpL_v Dv) per link.
    create_graph=True keeps the log-determinant differentiable for training; use False for evaluation. The sign is
    the smallest sign of det(1 + dexpL_v Dv) over the links of a configuration, and the log-determinant is exact only
    if the masking condition holds (MATH_SPEC_LATTICE section 2, failure mode; check new kernels with validate_mask).
    """
    B, D = U.shape[:2]
    lat = U.shape[2:2 + D]
    V = math.prod(lat)
    N = G.N
    m = (mask(lat[0], D, mu, p) if active is None else active).reshape(-1)
    idx = m.nonzero().squeeze(-1)
    Umu = U[:, mu].reshape(B, V, N, N)
    Ua = Umu[:, idx]                                                   # active links (B, nA, N, N)
    S = staples(U, mu).reshape(B, V, -1, N, N)[:, idx]                 # their staples (B, nA, n_loops, N, N)
    omega = lambda X: G.vee(head(X.unsqueeze(-3) @ S, coeff))          # (B, nA, d)
    w = omega(Ua)
    new = Umu.index_copy(1, idx, G.exp(w) @ Ua).reshape(U[:, mu].shape)
    Unew = torch.cat([U[:, :mu], new.unsqueeze(1), U[:, mu + 1:]], 1)
    if not logdet:
        return Unew
    xi = torch.zeros(B, len(idx), G.d, dtype=torch.float64, requires_grad=True)
    w_xi = omega((torch.eye(N, dtype=C) + G.hat(xi)) @ Ua)             # first order in xi suffices
    Dw = torch.stack([torch.autograd.grad(w_xi[..., a].sum(), xi, create_graph=create_graph, retain_graph=True)[0] for a in range(G.d)], -2)
    sign, ld = torch.linalg.slogdet(torch.eye(G.d, dtype=torch.float64) + G.dexpL(w) @ Dw)
    return Unew, ld.sum(-1), sign.amin(-1), torch.linalg.matrix_norm(Dw.detach(), 2).amax(-1)


def invert_masked_layer(U: Tensor, mu: int, p: int, coeff: CoeffFn, G: GaugeGroup, active: Tensor | None = None,
                        tol: float = 1e-14, max_iter: int = 200) -> tuple[Tensor, int, float]:
    """Inverse of masked_layer(., mu, p, coeff, G) by fixed-point iteration (MATH_SPEC_LATTICE section 6).

    U is the OUTPUT field of the sub-layer. Its frozen links equal those of the input, so the staples of every active
    link are known, and each active input link X solves X = exp(-v(X)) Y with Y the output link and
    v(X) = vee(head(X @ staples, coeff)). Starting from X = Y, the iteration stops when successive iterates differ by
    at most `tol` in every entry or after `max_iter` iterations.

    Returns (input field, iterations, last change). By Proposition 3 the iteration converges from any start when
    q = sup ||D omega||_2 < 1 (masked_layer returns max ||D omega||_2), with error decreasing like q^k; a last change
    above `tol` means `max_iter` was reached without convergence. Sub-layers of a stack are inverted in reverse order.
    """
    B, D = U.shape[:2]
    lat = U.shape[2:2 + D]
    V = math.prod(lat)
    N = G.N
    m = (mask(lat[0], D, mu, p) if active is None else active).reshape(-1)
    idx = m.nonzero().squeeze(-1)
    Umu = U[:, mu].reshape(B, V, N, N)
    Ya = Umu[:, idx]                                                   # output active links (B, nA, N, N)
    S = staples(U, mu).reshape(B, V, -1, N, N)[:, idx]                 # staples are frozen: same as in the input
    X = Ya.clone()
    it = 0
    res = 1.0
    while res > tol and it < max_iter:
        Xn = G.exp(-G.vee(head(X.unsqueeze(-3) @ S, coeff))) @ Ya
        res = float((Xn - X).abs().max())
        X = Xn
        it += 1
    new = Umu.index_copy(1, idx, X).reshape(U[:, mu].shape)
    return torch.cat([U[:, :mu], new.unsqueeze(1), U[:, mu + 1:]], 1), it, res
