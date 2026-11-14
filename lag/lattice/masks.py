"""Masks for plaquette kernels and their numerical validation (MATH_SPEC_LATTICE.md sections 3.2 and 3.3).

The mask M(mu, p) holds the links (x, mu) whose transverse coordinates sum to p mod 2. Two links of M(mu, p) never
share a plaquette, so a sub-layer that updates them with plaquette-staple kernels satisfies the masking condition of
Proposition 1. A sweep is the 2D sub-layers (mu, p), mu = 0..D-1, p = 0, 1; every link is updated exactly once.

`validate_mask` is moved verbatim from the reference laglattice.py (type hints and docstring added). `mask` is the
reference function plus the even-L check that MATH_SPEC_LATTICE.md section 11 asks for: on an odd torus the
transverse parity wraps around and the masking condition fails silently (negative controls of section 9).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from lag.lattice.field import CoeffFn, GaugeGroup, head, staples

__all__ = ["mask", "validate_mask"]


def mask(L: int, D: int, mu: int, p: int) -> Tensor:
    """Active links of sub-layer (mu, p): direction mu, sum of the transverse coordinates = p mod 2. L must be even.

    Returns a boolean tensor of shape (L,) * D over the sites x of the links (x, mu). Raises ValueError for odd L,
    where M(mu, p) is not an independent set (MATH_SPEC_LATTICE section 3.2). Negative controls that need the
    parity set on an odd torus pass it to masked_layer / validate_mask through their `active` argument.
    """
    if L % 2:
        raise ValueError(f"mask: L = {L} is odd; the masks M(mu, p) need an even L in every direction "
                         f"(MATH_SPEC_LATTICE section 3.2)")
    grids = torch.meshgrid(*[torch.arange(L)] * D, indexing="ij")
    return (sum(g for nu, g in enumerate(grids) if nu != mu) % 2) == p


def validate_mask(U: Tensor, mu: int, p: int, coeff: CoeffFn, G: GaugeGroup, active: Tensor | None = None,
                  eps: float = 1e-3, seed: int = 0) -> float:
    """Numerical check of the masking condition (MATH_SPEC_LATTICE Proposition 1) for a given kernel: perturb each
    active link in turn and return the largest change of the connection of any OTHER active link. A valid mask gives
    exactly 0 (the other links' inputs are bit-identical); a violation gives O(eps); a NaN anywhere is returned as NaN.
    One forward pass per active link, so use a small lattice.

    The kernel is the one of masked_layer: omega_l = head(U_l @ staples of l, coeff). `active` overrides mask(L, D,
    mu, p) (negative controls); the perturbations exp(eps xi) use their own generator seeded with `seed`.
    """
    B, D = U.shape[:2]
    lat = U.shape[2:2 + D]
    V = math.prod(lat)
    N = G.N
    m = (mask(lat[0], D, mu, p) if active is None else active).reshape(-1)
    idx = m.nonzero().squeeze(-1)

    def omegas(Uf):
        S = staples(Uf, mu).reshape(B, V, -1, N, N)[:, idx]
        return G.vee(head(Uf[:, mu].reshape(B, V, N, N)[:, idx].unsqueeze(-3) @ S, coeff))
    gen = torch.Generator().manual_seed(seed)
    w0 = omegas(U)
    worst = 0.0
    for k in range(len(idx)):
        Umu = U[:, mu].reshape(B, V, N, N).clone()
        Umu[:, idx[k]] = G.exp(eps * torch.randn(B, G.d, generator=gen, dtype=torch.float64)) @ Umu[:, idx[k]]
        Up = torch.cat([U[:, :mu], Umu.reshape(U[:, mu].shape).unsqueeze(1), U[:, mu + 1:]], 1)
        dw = (omegas(Up) - w0).abs()
        dw[:, k] = 0                       # its own connection may change
        m = float(dw.max())
        worst = m if math.isnan(m) else max(worst, m)    # keep a NaN: Python's max would drop it
    return worst
