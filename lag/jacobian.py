"""Log-determinants of one flow step (MATH_SPEC.md section 3)."""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["logabsdet_step"]

# ------------------------------------------------------------------ log-determinants
def logabsdet_step(dexpL_v: Tensor, Dv: Tensor, A: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """log|det J| of x -> exp(v(x)).x.  Group state: A=None, Dv is (d,d) in right-trivialised coordinates.
    Vector state in R^n: Dv is (d,n), A is (n,d) with A[:,a] = hat(e_a) z.

    dexpL_v: (..., d, d). Returns (sign, log|det|), each of shape (...). MATH_SPEC section 3.
    """
    M = dexpL_v @ Dv if A is None else dexpL_v @ Dv @ A
    return torch.linalg.slogdet(torch.eye(M.shape[-1], dtype=M.dtype) + M)
