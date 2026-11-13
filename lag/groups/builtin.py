"""Built-in groups with closed-form exponentials (MATH_SPEC.md sections 1, 2.1-2.4).

SO3, SU2, SU3, SE3 are MatrixLieGroup subclasses; the closed forms of ``lag.exp`` are wired in
as module-level (picklable) backend functions with signature (group, w) -> Tensor.
"""
from __future__ import annotations

import torch
from torch import Tensor

from lag.algebra import T2, T3, se3_hat, so3_hat
from lag.exp import dexpL_se3, dexpL_so3, se3_exp, so3_exp, su2_exp, su3_exp
from lag.groups.base import MatrixLieGroup

__all__ = ["SO3", "SU2", "SU3", "SE3"]


def _so3_exp(group: MatrixLieGroup, w: Tensor) -> Tensor:
    """SO(3) closed-form exp: w (..., 3) -> (..., 3, 3); MATH_SPEC section 2.1."""
    return so3_exp(w)


def _so3_dexpL(group: MatrixLieGroup, w: Tensor) -> Tensor:
    """SO(3) closed-form dexp^L: w (..., 3) -> (..., 3, 3); MATH_SPEC section 2.1."""
    return dexpL_so3(w)


def _su2_exp(group: MatrixLieGroup, w: Tensor) -> Tensor:
    """SU(2) closed-form exp: w (..., 3) -> (..., 2, 2); MATH_SPEC section 2.1."""
    return su2_exp(w)


def _su2_dexpL(group: MatrixLieGroup, w: Tensor) -> Tensor:
    """SU(2) dexp^L: same 3x3 matrix as SO(3) in the basis T_a; MATH_SPEC section 2.1."""
    return dexpL_so3(w)


def _su3_exp(group: MatrixLieGroup, w: Tensor) -> Tensor:
    """SU(3) closed-form exp: w (..., 8) -> (..., 3, 3); MATH_SPEC section 2.4."""
    return su3_exp(group.hat(w))


def _se3_exp(group: MatrixLieGroup, xi: Tensor) -> Tensor:
    """SE(3) closed-form exp: xi (..., 6) -> (..., 4, 4); MATH_SPEC section 2.2."""
    return se3_exp(xi)


def _se3_dexpL(group: MatrixLieGroup, xi: Tensor) -> Tensor:
    """SE(3) closed-form dexp^L: xi (..., 6) -> (..., 6, 6); MATH_SPEC section 2.2."""
    return dexpL_se3(xi)


class SO3(MatrixLieGroup):
    """SO(3) in the 3x3 rotation representation."""

    def __init__(self) -> None:
        super().__init__(so3_hat(torch.eye(3, dtype=torch.float64)), "SO(3)",
                         closed_form_exp=_so3_exp, closed_form_dexpL=_so3_dexpL)


class SU2(MatrixLieGroup):
    """SU(2) in the basis T_a = -(i/2) sigma_a."""

    def __init__(self) -> None:
        super().__init__(T2, "SU(2)", closed_form_exp=_su2_exp, closed_form_dexpL=_su2_dexpL)


class SU3(MatrixLieGroup):
    """SU(3) in the basis T_a = -(i/2) lambda_a (generic dexp^L; open item 9.1)."""

    def __init__(self) -> None:
        super().__init__(T3, "SU(3)", closed_form_exp=_su3_exp)


class SE3(MatrixLieGroup):
    """SE(3) in the 4x4 homogeneous representation, xi = (phi, rho): rotation first, then translation (MATH_SPEC 1)."""

    def __init__(self) -> None:
        super().__init__(se3_hat(torch.eye(6, dtype=torch.float64)), "SE(3)",
                         closed_form_exp=_se3_exp, closed_form_dexpL=_se3_dexpL)
